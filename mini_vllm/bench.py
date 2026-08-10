"""Step 2.3 — the measuring instrument for the rest of the project.

Phase 3 exists to make things faster, so it lives or dies on whether this file
tells the truth. Three ways a GPU benchmark lies, all handled here:

* **Asynchronous launches.** CUDA calls return before the work finishes, so a
  naive timer measures how fast Python can enqueue kernels. Every timed region is
  bracketed by `torch.cuda.synchronize()`.
* **A cold first iteration.** Allocator growth, cuBLAS autotuning and JIT loading
  land on whichever iteration runs first, so there is a warmup.
* **A throttled GPU.** This is the one that quietly ruins everything: a card
  parked in a low power state runs 30x slow, and a benchmark that does not look
  will happily report the number anyway. :func:`gpu_state` reads the clocks and
  the harness prints a loud warning when they are far below maximum.

The report also prints which ops ran as CUDA kernels, because "my kernel made no
difference" is usually "my kernel never ran".

Prefill and decode are reported separately because they are separate problems.
**TTFT** is dominated by a compute-bound pass over the whole prompt; **decode
tokens/sec** is dominated by memory bandwidth, since each step reads all 1.2 GB of
weights to produce one token. An optimization almost always helps one and not the
other, so a single average would hide exactly what you need to see.

`--mode kernels` measures the Phase 3 kernels instead of the model, and reports
achieved memory bandwidth rather than wall-clock. That is the right score for
RMSNorm, RoPE and SwiGLU: they do a few flops per element, so their ceiling is how
fast the card can move the bytes, and "80% of peak bandwidth" says something a
millisecond figure does not.

Run it::

    python -m mini_vllm.bench --mode single --input-len 128 --output-len 128 --warmup 2
    python -m mini_vllm.bench --mode single --compare hf
    python -m mini_vllm.bench --mode kernels
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from mini_vllm.generate import generate_ids_cached, load
from mini_vllm.kernels import ops
from mini_vllm.model.loader import DEFAULT_MODEL_ID

__all__ = [
    "BandwidthResult",
    "ClockSampler",
    "GpuState",
    "KernelCase",
    "SingleResult",
    "build_input_ids",
    "copy_ceiling",
    "gpu_state",
    "kernel_cases",
    "measure_bandwidth",
    "measure_hf",
    "measure_ours",
    "theoretical_bandwidth",
]

# Repeated to fill any requested prompt length. Real text rather than random ids:
# speed does not care, but it keeps the generated continuation readable when
# something looks wrong.
FILLER = (
    "The city of Rome was founded in 753 BC by the twin brothers Romulus and Remus, "
    "and grew over the following centuries into the capital of an empire. "
)


# --------------------------------------------------------------------- gpu state


@dataclass(frozen=True)
class GpuState:
    """What the GPU was actually doing, as opposed to what it can do."""

    name: str
    sm_clock: int
    sm_clock_max: int
    memory_clock: int
    memory_clock_max: int
    power_state: str
    power_draw: str

    @property
    def sm_fraction(self) -> float:
        return self.sm_clock / self.sm_clock_max if self.sm_clock_max else 1.0

    @property
    def memory_fraction(self) -> float:
        return self.memory_clock / self.memory_clock_max if self.memory_clock_max else 1.0

    @property
    def is_throttled(self) -> bool:
        """True when the card is running far enough below its clocks to void results.

        Half speed is the threshold. Boost behaviour means a healthy GPU never
        sits at exactly its maximum, but a card at less than half of it is not
        being benchmarked, it is being sampled while asleep.
        """
        return self.sm_fraction < 0.5 or self.memory_fraction < 0.5

    def describe(self) -> str:
        return (
            f"{self.name}\n"
            f"  sm clock     {self.sm_clock:>5} / {self.sm_clock_max} MHz "
            f"({self.sm_fraction:.0%})\n"
            f"  memory clock {self.memory_clock:>5} / {self.memory_clock_max} MHz "
            f"({self.memory_fraction:.0%})\n"
            f"  power state  {self.power_state}, drawing {self.power_draw}"
        )


def gpu_state() -> GpuState | None:
    """Read the current clocks from `nvidia-smi`, or None if that is not possible.

    Only meaningful while the GPU is busy: clocks fall back to idle within
    milliseconds of the work finishing, so a reading taken after a benchmark
    always looks throttled. Use :class:`ClockSampler` to observe a run.
    """
    if not shutil.which("nvidia-smi"):
        return None

    fields = [
        "name",
        "clocks.sm",
        "clocks.max.sm",
        "clocks.mem",
        "clocks.max.mem",
        "pstate",
        "power.draw",
    ]
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None

    values = [value.strip() for value in output.strip().splitlines()[0].split(",")]
    if len(values) != len(fields):
        return None

    def megahertz(text: str) -> int:
        digits = "".join(character for character in text if character.isdigit())
        return int(digits) if digits else 0

    return GpuState(
        name=values[0],
        sm_clock=megahertz(values[1]),
        sm_clock_max=megahertz(values[2]),
        memory_clock=megahertz(values[3]),
        memory_clock_max=megahertz(values[4]),
        power_state=values[5],
        power_draw=values[6],
    )


class ClockSampler:
    """Watches the GPU clocks in the background and keeps the highest seen.

    A single reading proves nothing: clocks ramp, and one taken at the wrong
    moment is either idle or a boost spike. The *peak over a run* answers the
    question that matters — was the card ever allowed to go fast while we were
    measuring it?
    """

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.peak: GpuState | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            state = gpu_state()
            if state is not None and (
                self.peak is None or state.sm_clock > self.peak.sm_clock
            ):
                self.peak = state
            self._stop.wait(self.interval)

    def __enter__(self) -> ClockSampler:
        if shutil.which("nvidia-smi") and torch.cuda.is_available():
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exception) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


# ----------------------------------------------------------------------- results


@dataclass(frozen=True)
class SingleResult:
    """One measurement of one implementation."""

    label: str
    input_len: int
    output_len: int
    ttft_ms: float
    decode_tokens_per_second: float

    @property
    def decode_ms_per_token(self) -> float:
        return 1000.0 / self.decode_tokens_per_second if self.decode_tokens_per_second else 0.0

    def describe(self) -> str:
        return (
            f"{self.label:<16} "
            f"TTFT {self.ttft_ms:8.1f} ms   "
            f"decode {self.decode_tokens_per_second:7.1f} tok/s "
            f"({self.decode_ms_per_token:6.2f} ms/token)"
        )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------- measuring


def measure_ours(
    model,
    input_ids: torch.Tensor,
    output_len: int,
    warmup: int = 1,
    label: str = "mini-vllm",
) -> SingleResult:
    """Time the cached model: prefill once, then ``output_len - 1`` decode steps.

    Prefill is timed on its own rather than inferred, since the model exposes it
    directly. Stop tokens are deliberately not passed: a run that ends early would
    be timing a different amount of work than it claims.
    """
    device = input_ids.device

    for _ in range(warmup):
        generate_ids_cached(model, input_ids, max_tokens=min(output_len, 8))
    _synchronize(device)

    with torch.no_grad():
        caches = model.create_kv_cache()

        started = time.perf_counter()
        logits = model(input_ids, caches, last_only=True)
        token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        _synchronize(device)
        ttft = time.perf_counter() - started

        decode_steps = max(output_len - 1, 1)
        started = time.perf_counter()
        for _ in range(decode_steps):
            logits = model(token, caches, last_only=True)
            token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        _synchronize(device)
        decode_elapsed = time.perf_counter() - started

    batch = input_ids.shape[0]
    return SingleResult(
        label=label,
        input_len=input_ids.shape[1],
        output_len=output_len,
        ttft_ms=ttft * 1000.0,
        decode_tokens_per_second=decode_steps * batch / decode_elapsed,
    )


def measure_hf(
    hf_model,
    input_ids: torch.Tensor,
    output_len: int,
    warmup: int = 1,
    label: str = "transformers",
) -> SingleResult:
    """Time `transformers.generate` on identical inputs.

    TTFT comes from a one-token generation and the decode rate from the remainder
    of a full run, because `generate` does not expose its prefill separately.
    `min_new_tokens` pins the token count so an early stop cannot flatter the
    result.
    """
    device = input_ids.device

    def run(new_tokens: int):
        return hf_model.generate(
            input_ids,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            pad_token_id=hf_model.generation_config.pad_token_id,
        )

    with torch.no_grad():
        for _ in range(warmup):
            run(min(output_len, 8))
        _synchronize(device)

        started = time.perf_counter()
        run(1)
        _synchronize(device)
        ttft = time.perf_counter() - started

        started = time.perf_counter()
        run(output_len)
        _synchronize(device)
        total = time.perf_counter() - started

    decode_steps = max(output_len - 1, 1)
    decode_elapsed = max(total - ttft, 1e-9)
    batch = input_ids.shape[0]

    return SingleResult(
        label=label,
        input_len=input_ids.shape[1],
        output_len=output_len,
        ttft_ms=ttft * 1000.0,
        decode_tokens_per_second=decode_steps * batch / decode_elapsed,
    )


# ------------------------------------------------------------------- bandwidth


@dataclass(frozen=True)
class BandwidthResult:
    """One kernel, timed and converted into bytes per second.

    `bytes_moved` is the traffic the op *cannot avoid* — read the input, write the
    output — not the traffic it happened to generate. Counting a redundant reread
    would flatter a kernel for being wasteful, since the number would go up.
    """

    label: str
    bytes_moved: int
    seconds: float

    @property
    def gigabytes_per_second(self) -> float:
        return self.bytes_moved / self.seconds / 1e9 if self.seconds else 0.0

    @property
    def microseconds(self) -> float:
        return self.seconds * 1e6

    def fraction_of(self, peak: float | None) -> float | None:
        return self.gigabytes_per_second / peak if peak else None

    def describe(self, peak: float | None = None) -> str:
        share = self.fraction_of(peak)
        suffix = f" ({share:.0%} of peak)" if share is not None else ""
        return (
            f"{self.label:<40} {self.microseconds:9.1f} us   "
            f"{self.gigabytes_per_second:7.1f} GB/s{suffix}"
        )


def theoretical_bandwidth(device: int = 0) -> float | None:
    """The card's peak memory bandwidth in GB/s, from its clock and bus width.

    Double data rate, so the transfer rate is twice the reported memory clock. On
    the hardware of record this comes to 384 GB/s (12.001 GHz over 128 bits),
    which matches the published figure — worth checking on a new card, because a
    peak that is wrong by 2x turns every percentage below it into fiction.
    """
    if not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(device)
    clock_hertz = properties.memory_clock_rate * 1e3  # the attribute is in kHz
    return 2.0 * clock_hertz * (properties.memory_bus_width / 8) / 1e9


def measure_bandwidth(
    call: Callable[[], object],
    bytes_moved: int,
    label: str,
    warmup: int = 5,
    iterations: int = 50,
) -> BandwidthResult:
    """Time ``call`` back to back and report the traffic rate it sustained.

    The whole run is bracketed by one pair of syncs rather than syncing each
    iteration: a sync per call would time the synchronization for the short cases,
    which is most of them.
    """
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(iterations):
        call()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    return BandwidthResult(label=label, bytes_moved=bytes_moved, seconds=elapsed / iterations)


def copy_ceiling(megabytes: int = 256, dtype: torch.dtype = torch.bfloat16) -> BandwidthResult:
    """What a bare `copy_` of a large buffer achieves — the practical ceiling.

    The theoretical peak is an upper bound nothing reaches; this is the number a
    kernel that does nothing but move bytes actually gets on this machine today,
    which makes it the fairer thing to be measured against. It doubles as the
    throttle check from Step 2.3: if this is an order of magnitude below spec, the
    GPU is asleep and every figure below it is meaningless.
    """
    elements = megabytes * 1024 * 1024 // torch.tensor([], dtype=dtype).element_size()
    source = torch.randn(elements, device="cuda", dtype=dtype)
    destination = torch.empty_like(source)
    moved = 2 * source.numel() * source.element_size()  # one read, one write

    return measure_bandwidth(
        lambda: destination.copy_(source), moved, f"copy_ ({megabytes} MB)", iterations=20
    )


@dataclass(frozen=True)
class KernelCase:
    """A kernel, the PyTorch expression it replaces, and the traffic both must move."""

    label: str
    kernel: Callable[[], object]
    reference: Callable[[], object]
    bytes_moved: int


def rows_exceeding_l2(width: int, dtype: torch.dtype, multiple: int = 8) -> int:
    """How many rows it takes for the working set to be `multiple` times the L2.

    Needed because this card has a **32 MB** L2, which is larger than most of the
    tensors a 0.6B model touches. A benchmark whose input and output both fit in
    L2 is not measuring memory bandwidth at all, and it does not fail quietly: it
    reports a number *above* the card's DRAM peak, which is the only reason the
    problem is noticeable. See the Step 3.1 note in PLAN.md.
    """
    element_size = torch.tensor([], dtype=dtype).element_size()
    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    bytes_per_row = 2 * width * element_size  # the row is read once and written once
    return max(1, multiple * l2_bytes // bytes_per_row)


def kernel_cases(dtype: torch.dtype = torch.bfloat16) -> list[KernelCase]:
    """One case per implemented kernel per interesting shape.

    Widths are the model's real ones; the row counts are chosen to separate three
    regimes that a single number would blur together:

    * **1 row** — one block on one SM. This measures launch latency, and a decode
      step is exactly this shape, so it is the case the engine actually lives in.
    * **128 rows** — a prefill chunk. Enough blocks to fill the card, small enough
      to sit in cache.
    * **past L2** — the only regime where "fraction of peak bandwidth" means what
      it says.
    """
    from mini_vllm.kernels.extension import load_extension

    if not torch.cuda.is_available():
        return []

    module = load_extension()
    implemented = ops.cuda_kernel_names()
    cases: list[KernelCase] = []

    def token_counts(width: int) -> tuple[tuple[int, str], ...]:
        return (
            (1, "decode, 1 token"),
            (128, "prefill, 128 tokens"),
            (rows_exceeding_l2(width, dtype), "past L2, DRAM-bound"),
        )

    if "rmsnorm" in implemented:
        hidden_size = 1024  # Qwen3-0.6B's E
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
        for rows, note in token_counts(hidden_size):
            x = torch.randn(rows, hidden_size, device="cuda", dtype=dtype)
            moved = (2 * x.numel() + weight.numel()) * x.element_size()
            cases.append(
                KernelCase(
                    label=f"rmsnorm E={hidden_size} ({note})",
                    kernel=lambda x=x, weight=weight: module.rmsnorm(x, weight, 1e-6),
                    reference=lambda x=x, weight=weight: ops.rmsnorm(x, weight, 1e-6),
                    bytes_moved=moved,
                )
            )

    if "swiglu" in implemented:
        intermediate = 3072  # Qwen3-0.6B's MLP width
        for rows, note in token_counts(2 * intermediate):
            gate = torch.randn(rows, intermediate, device="cuda", dtype=dtype)
            up = torch.randn_like(gate)
            # Two reads and one write, so the traffic is 3x the tensor, not 2x.
            moved = 3 * gate.numel() * gate.element_size()
            cases.append(
                KernelCase(
                    label=f"swiglu I={intermediate} ({note})",
                    kernel=lambda gate=gate, up=up: module.swiglu(gate, up),
                    reference=lambda gate=gate, up=up: ops.swiglu(gate, up),
                    bytes_moved=moved,
                )
            )

    if "rope" in implemented:
        from mini_vllm.positional_encoding import RoPE

        heads, head_dim = 16, 128  # Qwen3-0.6B's H_q and D
        per_token = heads * head_dim
        longest = max(rows for rows, _ in token_counts(per_token))
        tables = RoPE(head_dim, longest, 1_000_000.0, device="cuda")

        for rows, note in token_counts(per_token):
            x = torch.randn(1, rows, heads, head_dim, device="cuda", dtype=dtype)
            positions = torch.arange(rows, device="cuda")
            # The table rows are shared by every head of a token, so they add
            # `D` fp32 pairs per token on top of the activation traffic.
            moved = 2 * x.numel() * x.element_size() + 2 * rows * head_dim * 4
            cases.append(
                KernelCase(
                    label=f"rope H={heads} D={head_dim} ({note})",
                    kernel=lambda x=x, p=positions: module.rope(x, p, tables.cos, tables.sin),
                    reference=lambda x=x, p=positions: ops.rope(x, p, tables.cos, tables.sin),
                    bytes_moved=moved,
                )
            )

    if "decode_attention" in implemented:
        # Attention is the one kernel whose work grows with the context rather
        # than with the batch, so the axis swept here is the cache length. At 8192
        # the K/V of a single layer is 32 MB — exactly this card's L2 — which is
        # why only the longest context here is honestly DRAM-bound.
        query_heads, kv_heads, head_dim = 16, 8, 128
        scale = 1.0 / math.sqrt(head_dim)
        for source_len in (128, 1024, 8192):
            q = torch.randn(1, query_heads, 1, head_dim, device="cuda", dtype=dtype)
            k = torch.randn(1, kv_heads, source_len, head_dim, device="cuda", dtype=dtype)
            v = torch.randn_like(k)
            # The kernel reads K and V once each and writes one row of output.
            moved = (k.numel() + v.numel()) * k.element_size()
            cases.append(
                KernelCase(
                    label=f"decode_attention S={source_len}",
                    kernel=lambda q=q, k=k, v=v: module.decode_attention(q, k, v, scale),
                    reference=lambda q=q, k=k, v=v: ops.attention(q, k, v),
                    bytes_moved=moved,
                )
            )

    if "flash_prefill" in implemented:
        # Prefill is the one compute-bound case in this table, so its GB/s column
        # is close to meaningless and the speedup column is the whole point: the
        # oracle materializes an L x S score matrix that this kernel never writes.
        query_heads, kv_heads, head_dim = 16, 8, 128
        scale = 1.0 / math.sqrt(head_dim)
        for query_len in (128, 512, 2048):
            q = torch.randn(1, query_heads, query_len, head_dim, device="cuda", dtype=dtype)
            k = torch.randn(1, kv_heads, query_len, head_dim, device="cuda", dtype=dtype)
            v = torch.randn_like(k)
            moved = (q.numel() + k.numel() + v.numel()) * q.element_size()
            cases.append(
                KernelCase(
                    label=f"flash_prefill L=S={query_len}",
                    kernel=lambda q=q, k=k, v=v: module.flash_prefill(q, k, v, scale),
                    reference=lambda q=q, k=k, v=v: ops.attention(q, k, v, mask="causal"),
                    bytes_moved=moved,
                )
            )

    return cases


@dataclass(frozen=True)
class KernelComparison:
    """A kernel measured beside the PyTorch expression it replaced."""

    kernel: BandwidthResult
    reference: BandwidthResult

    @property
    def speedup(self) -> float:
        return self.reference.seconds / self.kernel.seconds if self.kernel.seconds else 0.0


def run_kernels(
    arguments,
) -> tuple[BandwidthResult | None, list[KernelComparison], GpuState | None]:
    """`--mode kernels`: achieved bandwidth for each kernel and its PyTorch twin."""
    cases = kernel_cases(torch.bfloat16)
    if not cases:
        return None, [], None

    with ClockSampler() as sampler:
        ceiling = copy_ceiling()
        comparisons = [
            KernelComparison(
                kernel=measure_bandwidth(
                    case.kernel, case.bytes_moved, case.label, arguments.warmup, 100
                ),
                reference=measure_bandwidth(
                    case.reference,
                    case.bytes_moved,
                    "  ...the torch it replaces",
                    arguments.warmup,
                    100,
                ),
            )
            for case in cases
        ]

    return ceiling, comparisons, sampler.peak


def build_input_ids(tokenizer, input_len: int, batch: int, device) -> torch.Tensor:
    """A prompt of exactly ``input_len`` tokens, repeated across the batch."""
    repeats = -(-input_len // max(len(tokenizer(FILLER).input_ids), 1))
    ids = tokenizer(FILLER * repeats, return_tensors="pt").input_ids[:, :input_len]
    if ids.shape[1] < input_len:
        raise ValueError(f"could not build a prompt of {input_len} tokens")
    return ids.expand(batch, -1).contiguous().to(device)


# ------------------------------------------------------------------------- modes


def run_single(arguments) -> tuple[list[SingleResult], GpuState | None]:
    """`--mode single`: one sequence, latency-oriented."""
    print(f"loading {arguments.model} ...")
    loaded = load(
        arguments.model,
        arguments.device,
        cached=not arguments.no_cache,
        use_cuda_kernels=arguments.use_cuda_kernels,
    )
    device = loaded.model.embedding.weight.device
    input_ids = build_input_ids(loaded.tokenizer, arguments.input_len, arguments.batch, device)

    label = "mini-vllm (no cache)" if arguments.no_cache else "mini-vllm"

    print("measuring ...")
    with ClockSampler() as sampler:
        if arguments.no_cache:
            from mini_vllm.generate import generate_ids

            # The uncached model has no cache to prefill, so TTFT and the decode
            # rate are both timed through the naive loop.
            results = [
                _measure_uncached(loaded.model, input_ids, arguments, label, generate_ids)
            ]
        else:
            results = [
                measure_ours(
                    loaded.model, input_ids, arguments.output_len, arguments.warmup, label
                )
            ]

        if arguments.compare == "hf":
            from transformers import AutoModelForCausalLM

            from mini_vllm.model.loader import resolve_model_path

            print("loading the transformers baseline ...")
            hf = (
                AutoModelForCausalLM.from_pretrained(
                    resolve_model_path(arguments.model), dtype=torch.bfloat16
                )
                .to(device)
                .eval()
            )
            results.append(measure_hf(hf, input_ids, arguments.output_len, arguments.warmup))

    return results, sampler.peak


def _measure_uncached(model, input_ids, arguments, label, generate_ids) -> SingleResult:
    """Time the Step 1.9 loop, for the comparison that motivates the cache."""
    device = input_ids.device

    for _ in range(arguments.warmup):
        generate_ids(model, input_ids, max_tokens=min(arguments.output_len, 4))
    _synchronize(device)

    with torch.no_grad():
        started = time.perf_counter()
        model(input_ids)
        _synchronize(device)
        ttft = time.perf_counter() - started

        started = time.perf_counter()
        generate_ids(model, input_ids, max_tokens=arguments.output_len)
        _synchronize(device)
        total = time.perf_counter() - started

    decode_steps = max(arguments.output_len - 1, 1)
    return SingleResult(
        label=label,
        input_len=input_ids.shape[1],
        output_len=arguments.output_len,
        ttft_ms=ttft * 1000.0,
        decode_tokens_per_second=decode_steps * input_ids.shape[0] / max(total - ttft, 1e-9),
    )


def _report_gpu_state(state: GpuState | None) -> None:
    if state is None:
        return
    print(f"\ngpu (peak clocks observed during the run): {state.describe()}")
    if state.is_throttled:
        print(
            "\n  WARNING: the GPU was running far below its clocks, so these numbers\n"
            "  measure the power state rather than the code. Every result above is\n"
            "  invalid as an absolute figure. Fix the throttling before drawing any\n"
            "  conclusion from a kernel change."
        )


def report_kernels(arguments) -> None:
    """Print the bandwidth table for `--mode kernels`."""
    ceiling, comparisons, state = run_kernels(arguments)
    if ceiling is None:
        print("no CUDA kernels are implemented yet — nothing to measure")
        return

    peak = theoretical_bandwidth()
    l2_megabytes = torch.cuda.get_device_properties(0).L2_cache_size // (1024 * 1024)

    print(f"\ntheoretical peak: {peak:.0f} GB/s" if peak else "\ntheoretical peak: unknown")
    print(f"{ceiling.describe(peak)}   <- the practical ceiling")
    print(
        f"\nL2 is {l2_megabytes} MB on this card, so any case whose input and output fit\n"
        f"inside it reports cache bandwidth, not memory bandwidth — which is why one\n"
        f"case is sized past L2 and is the only one whose share of peak means anything.\n"
    )

    for comparison in comparisons:
        print(comparison.kernel.describe(peak))
        print(f"{comparison.reference.describe(peak)}   ({comparison.speedup:.2f}x)")

    _report_gpu_state(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Mini-vLLM.")
    parser.add_argument("--mode", default="single", choices=["single", "kernels"])
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compare", default=None, choices=["hf"])
    parser.add_argument("--no-cache", action="store_true", help="measure the Step 1.9 loop")
    parser.add_argument(
        "--use-cuda-kernels",
        action="store_true",
        help="route ops through hand-written kernels where they exist (Phase 3)",
    )
    arguments = parser.parse_args()

    if arguments.mode == "kernels":
        report_kernels(arguments)
        return

    results, state = run_single(arguments)

    print(f"\ninput {arguments.input_len} tokens, output {arguments.output_len}, "
          f"batch {arguments.batch}, warmup {arguments.warmup}")
    print("\nops:")
    print(ops.dispatch_report(arguments.use_cuda_kernels))

    print()
    for result in results:
        print(result.describe())

    if len(results) == 2:
        ours, theirs = results
        print(
            f"\nrelative to {theirs.label}: "
            f"TTFT {theirs.ttft_ms / ours.ttft_ms:.2f}x, "
            f"decode {ours.decode_tokens_per_second / theirs.decode_tokens_per_second:.2f}x"
        )

    _report_gpu_state(state)


if __name__ == "__main__":
    main()
