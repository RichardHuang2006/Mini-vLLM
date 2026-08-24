"""Measurement harness for the kernels, the model and the engine.

Three ways a GPU benchmark reports the wrong number, all handled here:

* Asynchronous launches. CUDA calls return before the work finishes, so a naive timer
  measures how fast Python enqueues kernels. Every timed region is bracketed by
  `torch.cuda.synchronize()`.
* A cold first iteration. Allocator growth, cuBLAS autotuning and JIT loading land on
  whichever iteration runs first, so there is a warmup.
* A throttled GPU. A card parked in a low power state runs 30x slow. :func:`gpu_state`
  reads the clocks and the harness warns when they are far below maximum.

The report also prints which ops ran as CUDA kernels, since a kernel that made no
difference is usually a kernel that never ran.

Prefill and decode are reported separately. TTFT is dominated by a compute-bound pass over
the whole prompt; decode tokens/sec is dominated by memory bandwidth, since each step reads
all 1.2 GB of weights to produce one token. An optimization typically helps one and not the
other, so a single average hides the effect.

`--mode kernels` measures the CUDA kernels rather than the model and reports achieved
memory bandwidth. That is the right metric for RMSNorm, RoPE and SwiGLU: they do a few
flops per element, so their ceiling is how fast the card moves bytes.

`--mode throughput` measures the engine rather than a single request, and is the mode the
README quotes. The unit is output tokens per second over a whole request set submitted at
once, against `transformers.generate` on the same set. The comparison is asymmetric by
construction, which is the finding: `generate` takes one padded rectangle, so sixteen
prompts of 32 to 512 tokens all run for 512, while the engine gives each sequence its own
length and admits a replacement the iteration a request finishes. That gap is what
continuous batching and paging buy.

`--mode scheduler` measures the scheduler on an adversarial mix: many 32-token requests
with a 2048-token prompt dropped in every twentieth, arriving on a Poisson schedule. It
replays that schedule twice through one engine — once with chunked prefill, once with the
prefill-prioritized policy vLLM shipped before it — and reports the tail of the inter-token
latency. Throughput barely moves between the two: the same work is done either way, and
what changes is which requests wait for it.

`--mode prefix-cache` runs the workload the radix tree exists for, many requests behind one
long shared preamble, with caching on and off, and reports TTFT: a cache hit removes prompt
tokens from the prefill and leaves decode unchanged.

`--mode spec` sweeps the draft depth for speculative decoding and reports the acceptance
rate beside the wall clock. Both columns are needed, since a draft deep enough to be
accepted costs nearly what the target costs while a shallow one is cheap and rejected.

Run it::

    python -m mini_vllm.bench --mode single --input-len 128 --output-len 128 --warmup 2
    python -m mini_vllm.bench --mode single --compare hf
    python -m mini_vllm.bench --mode kernels
    python -m mini_vllm.bench --mode throughput --batch-sizes 1,4,16 --compare hf
    python -m mini_vllm.bench --mode scheduler --num-requests 2000
    python -m mini_vllm.bench --mode prefix-cache
    python -m mini_vllm.bench --mode spec --draft-layers 4,14,28
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence as SequenceABC
from dataclasses import dataclass, field

import torch

from mini_vllm.generate import generate_ids_cached, load
from mini_vllm.kernels import ops
from mini_vllm.model.loader import DEFAULT_MODEL_ID

__all__ = [
    "BandwidthResult",
    "ClockSampler",
    "GpuState",
    "KernelCase",
    "LatencyStats",
    "SingleResult",
    "StressRequest",
    "ThroughputResult",
    "build_input_ids",
    "copy_ceiling",
    "gpu_state",
    "kernel_cases",
    "measure_bandwidth",
    "measure_engine",
    "measure_hf",
    "measure_hf_throughput",
    "measure_ours",
    "percentile",
    "poisson_arrivals",
    "run_stress",
    "stress_requests",
    "theoretical_bandwidth",
    "throughput_prompts",
]

# Repeated to fill any requested prompt length. Real text rather than random ids: speed is
# unaffected, but it keeps the generated continuation readable when something looks wrong.
FILLER = (
    "The city of Rome was founded in 753 BC by the twin brothers Romulus and Remus, "
    "and grew over the following centuries into the capital of an empire. "
)


# --------------------------------------------------------------------- gpu state


@dataclass(frozen=True)
class GpuState:
    """What the GPU was doing, as opposed to what it is capable of."""

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

        Half speed is the threshold: boost behaviour means a healthy GPU never sits at
        exactly its maximum, but under half of it the card is idle rather than working.
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

    A single reading is not informative: clocks ramp, so one taken at the wrong moment is
    either idle or a boost spike. The peak over a run establishes whether the card was
    allowed to run fast while the measurement was in flight.
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

    Prefill is timed on its own rather than inferred, since the model exposes it directly.
    Stop tokens are not passed: a run that ended early would time less work than it claims.
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

    TTFT comes from a one-token generation and the decode rate from the remainder of a full
    run, since `generate` does not expose its prefill separately. `min_new_tokens` pins the
    token count so an early stop cannot inflate the result.
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

    `bytes_moved` is the traffic the op cannot avoid — read the input, write the output —
    rather than the traffic it happened to generate, so a redundant reread lowers the
    reported rate instead of raising it.
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

    Double data rate, so the transfer rate is twice the reported memory clock. On the
    hardware of record this comes to 384 GB/s (12.001 GHz over 128 bits), matching the
    published figure. Worth re-checking on a new card: a peak that is wrong by 2x
    invalidates every percentage derived from it.
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

    The whole run is bracketed by one pair of syncs rather than syncing per iteration,
    which for the short cases would time the synchronization instead.
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

    The theoretical peak is an upper bound nothing reaches; this is what a kernel that only
    moves bytes achieves on this machine, making it the fairer baseline. It doubles as the
    throttle check described at the top of this module: an order of magnitude below spec
    means the GPU is idling and every figure below it is invalid.
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

    This card has a 32 MB L2, larger than most tensors a 0.6B model touches. A benchmark
    whose input and output both fit in L2 is not measuring memory bandwidth; the symptom is
    a reported rate above the card's DRAM peak.
    """
    element_size = torch.tensor([], dtype=dtype).element_size()
    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    bytes_per_row = 2 * width * element_size  # the row is read once and written once
    return max(1, multiple * l2_bytes // bytes_per_row)


def _quantize_into(key, value, pools, use_cuda: bool):
    """Scatter one step's KV into ``pools``, and hand back the key pool to compare.

    `ops.quantize_scatter` writes in place and returns nothing, but a benchmark case must
    return what it computed so `test_bench` can check that the kernel and its reference
    agree rather than assuming a speedup is not one side doing less work. Returning the
    pool is free: it is the tensor just written, not a copy.
    """
    ops.quantize_scatter(key, value, *pools, 1.0, 1.0, use_cuda=use_cuda)
    return pools[0]


def kernel_cases(dtype: torch.dtype = torch.bfloat16) -> list[KernelCase]:
    """One case per implemented kernel per interesting shape.

    Widths are the model's real ones; the row counts separate three regimes a single number
    would blur together:

    * 1 row: one block on one SM, measuring launch latency. A decode step has exactly this
      shape, so it is the regime the engine spends its time in.
    * 128 rows: a prefill chunk. Enough blocks to fill the card, small enough to sit in
      cache.
    * past L2: the only regime where a fraction of peak bandwidth is meaningful.
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
        # Attention is the one kernel whose work grows with the context rather than the
        # batch, so the swept axis is the cache length. At 8192 the K/V of a single layer
        # is 32 MB, exactly this card's L2, so only the longest context is DRAM-bound.
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
        # Prefill is the one compute-bound case in this table, so its GB/s column carries
        # little information and the speedup column is what matters: the oracle
        # materializes an L x S score matrix that this kernel never writes.
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

    if "paged_attention" in implemented:
        # The engine's decode shape: a batch of sequences, one token each, attending over
        # pages scattered through a pool. The reference is the dense-gather oracle, so the
        # speedup column measures something different from the rows above — not a better
        # loop over the same data, but the removal of a full copy of every sequence's
        # cache per iteration.
        query_heads, kv_heads, head_dim, block_size = 16, 8, 128, 16
        scale = 1.0 / math.sqrt(head_dim)

        for batch, context_len in ((1, 8192), (16, 1024), (64, 1024)):
            blocks_each = -(-context_len // block_size)
            pool_blocks = batch * blocks_each
            keys = torch.randn(
                pool_blocks, block_size, kv_heads, head_dim, device="cuda", dtype=dtype
            )
            values = torch.randn_like(keys)
            # Shuffled deliberately: a pool in logical order would give the kernel
            # sequential reads it will not get after a few thousand allocations.
            shuffled = torch.randperm(pool_blocks, device="cuda", dtype=torch.int32)
            block_tables = shuffled.reshape(batch, blocks_each).contiguous()

            q = torch.randn(batch, query_heads, head_dim, device="cuda", dtype=dtype)
            cu_seqlens = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
            contexts = torch.full((batch,), context_len, device="cuda", dtype=torch.int32)
            lengths = torch.ones(batch, device="cuda", dtype=torch.int32)
            moved = 2 * batch * context_len * kv_heads * head_dim * keys.element_size()

            paged = (keys, values, block_tables, cu_seqlens, contexts, lengths)
            cases.append(
                KernelCase(
                    label=f"paged_attention B={batch} S={context_len} (decode)",
                    kernel=lambda q=q, p=paged, s=context_len: module.paged_attention(
                        q, *p, 1, s, scale
                    ),
                    reference=lambda q=q, p=paged, s=context_len: ops.paged_attention(
                        q, *p, 1, s, scale
                    ),
                    bytes_moved=moved,
                )
            )

        # A prefill chunk, the case the scalar kernel loses: the oracle materializes an
        # L x S score matrix but computes it with cuBLAS, and tensor cores win at this
        # arithmetic intensity regardless of shared-memory layout. Kept in the table so
        # the shapes are not selected to favour the kernel; a tensor-core prefill kernel
        # is deferred.
        for query_len, context_len in ((512, 2048), (2048, 2048)):
            blocks_each = -(-context_len // block_size)
            keys = torch.randn(
                blocks_each, block_size, kv_heads, head_dim, device="cuda", dtype=dtype
            )
            values = torch.randn_like(keys)
            block_tables = (
                torch.randperm(blocks_each, device="cuda", dtype=torch.int32)
                .reshape(1, blocks_each)
                .contiguous()
            )
            q = torch.randn(query_len, query_heads, head_dim, device="cuda", dtype=dtype)
            cu_seqlens = torch.tensor([0, query_len], device="cuda", dtype=torch.int32)
            contexts = torch.tensor([context_len], device="cuda", dtype=torch.int32)
            lengths = torch.tensor([query_len], device="cuda", dtype=torch.int32)
            moved = (
                q.numel() + keys.numel() + values.numel()
            ) * q.element_size()

            paged = (keys, values, block_tables, cu_seqlens, contexts, lengths)
            cases.append(
                KernelCase(
                    label=f"paged_attention L={query_len} S={context_len} (prefill chunk)",
                    kernel=lambda q=q, p=paged, ql=query_len, s=context_len: module.paged_attention(
                        q, *p, ql, s, scale
                    ),
                    reference=lambda q=q, p=paged, ql=query_len, s=context_len: ops.paged_attention(
                        q, *p, ql, s, scale
                    ),
                    bytes_moved=moved,
                )
            )

    if "kv_quantize_scatter" in implemented:
        # Writing a step's KV into the FP8 cache. Pure memory traffic — a division, a cast
        # and a store per element — making it the clearest fusion case here: the reference
        # reads the activations, writes a whole quantized temporary, reads it back and
        # scatters it, where the kernel reads once and writes once. The swept axis is the
        # token count, which is what separates a decode step from a prefill chunk.
        kv_heads, head_dim = 8, 128
        for tokens, note in ((1, "decode, 1 token"), (512, "prefill chunk"), (4096, "past L2")):
            key = torch.randn(tokens, kv_heads, head_dim, device="cuda", dtype=dtype)
            value = torch.randn_like(key)
            # Room for the tokens several times over, so the slots are scattered rather
            # than the contiguous run a fresh pool would hand out.
            pool_slots = max(4096, 4 * tokens)
            slots = torch.randperm(pool_slots, device="cuda")[:tokens].to(torch.int64)

            def pools() -> tuple:
                shape = (pool_slots, kv_heads, head_dim)
                empty = torch.zeros(shape, device="cuda", dtype=torch.float8_e4m3fn)
                return (empty, torch.zeros_like(empty), slots)

            # One read of the activations and one write of the quantized byte per element
            # of each of K and V. The reference moves a whole quantized temporary on top
            # of that, written and read back, which is what the case measures.
            moved = 2 * key.numel() * (key.element_size() + 1)

            kernel_pools, reference_pools = pools(), pools()
            cases.append(
                KernelCase(
                    label=f"kv_quantize_scatter T={tokens} ({note})",
                    kernel=lambda k=key, v=value, p=kernel_pools: _quantize_into(k, v, p, True),
                    reference=lambda k=key, v=value, p=reference_pools: _quantize_into(
                        k, v, p, False
                    ),
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


# ------------------------------------------------------------------ throughput

# The prompt lengths a throughput run cycles through. Varied deliberately: equal-length
# prompts are the one case where padding costs nothing, so using them would hide most of
# what continuous batching and paging buy. A 16x spread between shortest and longest is
# ordinary for a chat workload.
THROUGHPUT_LENGTHS = (32, 64, 128, 256, 512)


@dataclass(frozen=True)
class ThroughputResult:
    """One implementation's rate over a whole request set."""

    label: str
    num_requests: int
    prompt_tokens: int
    generated_tokens: int
    seconds: float

    @property
    def tokens_per_second(self) -> float:
        """Output tokens per second, the unit a serving SLO is written against."""
        return self.generated_tokens / self.seconds if self.seconds else 0.0

    @property
    def requests_per_second(self) -> float:
        return self.num_requests / self.seconds if self.seconds else 0.0

    def describe(self) -> str:
        return (
            f"{self.label:<24} {self.seconds:6.2f} s   "
            f"{self.tokens_per_second:8.1f} out tok/s   "
            f"{self.requests_per_second:6.2f} req/s"
        )


def throughput_prompts(tokenizer, num_requests: int) -> list[str]:
    """`num_requests` prompts whose lengths cycle through `THROUGHPUT_LENGTHS`."""
    lengths = [THROUGHPUT_LENGTHS[i % len(THROUGHPUT_LENGTHS)] for i in range(num_requests)]
    prompts = []
    for length in lengths:
        repeats = -(-length // max(len(tokenizer(FILLER).input_ids), 1))
        ids = tokenizer(FILLER * repeats).input_ids[:length]
        prompts.append(tokenizer.decode(ids))
    return prompts


def measure_engine(llm, prompts: list[str], output_len: int, warmup: int = 1) -> ThroughputResult:
    """Time the engine over the whole request set, submitted all at once.

    All at once, since that is the workload continuous batching is for: the engine decides
    what shares each iteration, admits a replacement the moment a request finishes, and
    never waits for the longest member of a fixed batch.

    `ignore_eos` fixes the token count; without it a run where three requests stop at
    token 9 would be credited with less work than the baseline performs.
    """
    for _ in range(warmup):
        llm.generate(prompts[: min(2, len(prompts))], max_tokens=4, ignore_eos=True)
    torch.cuda.synchronize()

    started = time.perf_counter()
    completions = llm.generate(prompts, max_tokens=output_len, ignore_eos=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    return ThroughputResult(
        label="mini-vllm",
        num_requests=len(prompts),
        prompt_tokens=sum(len(llm.tokenizer(prompt).input_ids) for prompt in prompts),
        generated_tokens=sum(completion.num_tokens for completion in completions),
        seconds=elapsed,
    )


def measure_hf_throughput(
    hf_model, tokenizer, prompts: list[str], output_len: int, warmup: int = 1
) -> ThroughputResult:
    """Time `transformers.generate` on the same set, as one padded batch.

    One padded batch is what `generate` offers, and the padding is the baseline's real cost
    rather than a handicap imposed here: sixteen prompts of 32 to 512 tokens become a
    `16 x 512` rectangle, every row runs for the longest row's length, and the KV cache is
    allocated for all of it.
    """
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    encoded = {key: value.to(hf_model.device) for key, value in encoded.items()}

    def run(new_tokens: int):
        return hf_model.generate(
            **encoded,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    with torch.no_grad():
        for _ in range(warmup):
            run(4)
        torch.cuda.synchronize()

        started = time.perf_counter()
        run(output_len)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

    return ThroughputResult(
        label="transformers",
        num_requests=len(prompts),
        prompt_tokens=sum(len(tokenizer(prompt).input_ids) for prompt in prompts),
        generated_tokens=len(prompts) * output_len,
        seconds=elapsed,
    )


def run_throughput(arguments) -> tuple[list[tuple[int, ThroughputResult, ThroughputResult | None]], GpuState | None]:
    """`--mode throughput`: output tokens/sec against `transformers`, per batch size."""
    from transformers import AutoTokenizer

    from mini_vllm import LLM
    from mini_vllm.model.loader import resolve_model_path

    path = resolve_model_path(arguments.model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    batch_sizes = [int(size) for size in arguments.batch_sizes.split(",")]

    # The baseline is loaded first when there is one, because the engine sizes its KV pool
    # from the memory free at startup. Loading it second would hand the engine a budget it
    # then has to give back, which on an 8 GB card is an out-of-memory error.
    hf_model = None
    if arguments.compare == "hf":
        from transformers import AutoModelForCausalLM

        print("loading the transformers baseline ...")
        hf_model = (
            AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
            .to(arguments.device)
            .eval()
        )

    print(f"loading {arguments.model} into the engine ...")
    llm = LLM(
        arguments.model,
        device=arguments.device,
        max_sequences=max(batch_sizes),
        max_batched_tokens=arguments.max_batched_tokens,
        use_cuda_kernels=arguments.use_cuda_kernels,
        kv_fraction=arguments.kv_fraction,
    )
    print(f"  {llm}")

    rows = []
    with ClockSampler() as sampler:
        for batch in batch_sizes:
            prompts = throughput_prompts(tokenizer, batch)
            print(f"measuring {batch} concurrent requests ...")
            ours = measure_engine(llm, prompts, arguments.output_len, arguments.warmup)
            theirs = (
                measure_hf_throughput(
                    hf_model, tokenizer, prompts, arguments.output_len, arguments.warmup
                )
                if hf_model is not None
                else None
            )
            rows.append((batch, ours, theirs))

    return rows, sampler.peak


def report_throughput(arguments) -> None:
    """Print the batch-size scaling table for `--mode throughput`."""
    rows, state = run_throughput(arguments)

    print(
        f"\n{len(THROUGHPUT_LENGTHS)} prompt lengths cycling through {THROUGHPUT_LENGTHS}, "
        f"{arguments.output_len} output tokens each, eos ignored so the work is fixed."
    )
    print("\nops:")
    print(ops.dispatch_report(arguments.use_cuda_kernels))
    print()

    for batch, ours, theirs in rows:
        print(f"batch {batch:>3} ({ours.prompt_tokens} prompt tokens in)")
        print(f"  {ours.describe()}")
        if theirs is not None:
            print(f"  {theirs.describe()}")
            print(f"  {'':24} {ours.tokens_per_second / theirs.tokens_per_second:.2f}x")

    if len(rows) > 1:
        first, last = rows[0][1], rows[-1][1]
        scaling = last.tokens_per_second / first.tokens_per_second
        print(
            f"\nfrom batch {rows[0][0]} to {rows[-1][0]}: {scaling:.1f}x the output rate.\n"
            "Decode is memory-bound on the weights, and every sequence in an iteration "
            "reads them once between them, so the batch is nearly free until the "
            "arithmetic runs out."
        )

    _report_gpu_state(state)


# -------------------------------------------------------------- scheduler stress

# The adversarial mix. Every default here is chosen to make the head-of-line stall visible
# rather than to favour the scheduler: many short requests are the chat workload, and the
# occasional 2048-token prompt is a document summary landing in the middle of it, the one
# request that can delay every other.
STRESS_SHORT_LEN = 32
STRESS_LONG_LEN = 2048
STRESS_LONG_EVERY = 20
STRESS_OUTPUT_LEN = 32


@dataclass(frozen=True)
class StressRequest:
    """One arrival: when it shows up, what it asks for, and how much it wants back."""

    arrival: float
    prompt_token_ids: tuple[int, ...]
    output_len: int

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)


def poisson_arrivals(num_requests: int, rate: float, seed: int = 0) -> list[float]:
    """Arrival times of a Poisson process of `rate` requests per second.

    Poisson rather than evenly spaced: bursts are what stress a scheduler, and exponential
    gaps produce them, with a third of the gaps shorter than a third of the mean. Rate 0
    means all at once, the throughput benchmark's workload rather than a serving one.
    """
    if num_requests < 0:
        raise ValueError(f"num_requests must be >= 0, got {num_requests}")
    if rate < 0:
        raise ValueError(f"rate must be >= 0, got {rate}")
    if rate == 0:
        return [0.0] * num_requests

    generator = random.Random(seed)
    times, clock = [], 0.0
    for _ in range(num_requests):
        clock += generator.expovariate(rate)
        times.append(clock)
    return times


def stress_requests(
    num_requests: int,
    rate: float,
    vocab_size: int,
    short_len: int = STRESS_SHORT_LEN,
    long_len: int = STRESS_LONG_LEN,
    long_every: int = STRESS_LONG_EVERY,
    output_len: int = STRESS_OUTPUT_LEN,
    seed: int = 0,
) -> list[StressRequest]:
    """`num_requests` arrivals, one long prompt every `long_every` short ones.

    Prompts are random token ids rather than text: nothing here reads the output, and a
    tokenizer call per request would put minutes of Python between the engine and the
    measurement.
    """
    generator = random.Random(seed + 1)
    arrivals = poisson_arrivals(num_requests, rate, seed)

    requests = []
    for index, arrival in enumerate(arrivals):
        is_long = long_every > 0 and index % long_every == long_every - 1
        length = long_len if is_long else short_len
        prompt = tuple(generator.randrange(1, vocab_size) for _ in range(length))
        requests.append(
            StressRequest(arrival=arrival, prompt_token_ids=prompt, output_len=output_len)
        )
    return requests


def percentile(values: SequenceABC[float], fraction: float) -> float:
    """The nearest-rank percentile: the smallest sample at or above `fraction` of them.

    No interpolation: a P99 that averages two neighbours reports a latency no request
    experienced, where a tail figure should name one a caller actually waited.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


@dataclass
class LatencyStats:
    """One policy's run over the stress mix, in the units an SLO is written in.

    `inter_token` is the gap between consecutive tokens of the same sequence, the latency a
    caller reading a stream sees. Its tail is the metric the scheduler stress benchmark
    exists to move: a mean hides a stall, since one iteration spent on a 2048-token prompt
    is amortized away by the hundreds of fast iterations around it.
    """

    label: str
    num_requests: int
    completed: int
    generated_tokens: int
    seconds: float
    ttft: list[float] = field(default_factory=list)
    inter_token: list[float] = field(default_factory=list)
    iteration_seconds: list[float] = field(default_factory=list)
    preemptions: int = 0
    iterations: int = 0

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.seconds if self.seconds else 0.0

    @property
    def p50_inter_token_ms(self) -> float:
        return percentile(self.inter_token, 0.50) * 1000.0

    @property
    def p99_inter_token_ms(self) -> float:
        return percentile(self.inter_token, 0.99) * 1000.0

    @property
    def max_inter_token_ms(self) -> float:
        return max(self.inter_token, default=0.0) * 1000.0

    @property
    def max_iteration_ms(self) -> float:
        """The slowest single iteration, which is the tail's floor.

        Reported beside the inter-token tail because it separates the two causes of a long
        gap: a decode that was not scheduled for several iterations (a policy problem, what
        this mode measures) and one iteration that took a long time (allocator growth, a
        driver stall, other work on the GPU). A P99 gap of roughly one iteration means the
        scheduler is behaving.
        """
        return max(self.iteration_seconds, default=0.0) * 1000.0

    @property
    def p99_iteration_ms(self) -> float:
        return percentile(self.iteration_seconds, 0.99) * 1000.0

    @property
    def stalled_seconds(self) -> float:
        """Total time callers spent waiting beyond a normal iteration, summed.

        The load-robust view of what the tail describes. A percentile reports how bad the
        bad case was, which depends on how many callers were unlucky and therefore on the
        arrival rate; this reports how much waiting the policy caused in total, which does
        not. One median iteration per gap is the floor, since no caller gets a token faster
        than the engine produces one, so only the excess counts.
        """
        floor = percentile(self.iteration_seconds, 0.50)
        return sum(max(gap - floor, 0.0) for gap in self.inter_token)

    @property
    def p50_ttft_ms(self) -> float:
        return percentile(self.ttft, 0.50) * 1000.0

    @property
    def p99_ttft_ms(self) -> float:
        return percentile(self.ttft, 0.99) * 1000.0

    def describe(self) -> str:
        return (
            f"{self.label:<20} "
            f"decode p50 {self.p50_inter_token_ms:6.1f} ms  "
            f"p99 {self.p99_inter_token_ms:7.1f} ms  "
            f"max {self.max_inter_token_ms:7.1f} ms   "
            f"TTFT p99 {self.p99_ttft_ms:8.1f} ms   "
            f"{self.tokens_per_second:7.1f} out tok/s"
        )


def run_stress(
    llm, requests: SequenceABC[StressRequest], label: str, progress_every: int = 0
) -> LatencyStats:
    """Replay an arrival schedule through the engine, timing every token.

    The driver matches the shape of a synchronous engine: admit whatever is due, run one
    iteration, record what came out. Arrivals are therefore quantized to iteration
    boundaries and the resulting wait is counted in TTFT, which is correct since a real
    server admits at the same boundaries.

    Timing is taken after `step` returns. `step` samples, which reads a device tensor and
    so waits for the iteration's work, making an extra synchronize unnecessary.
    """
    pending = deque(sorted(requests, key=lambda request: request.arrival))
    arrival_of: dict[int, float] = {}
    last_token_at: dict[int, float] = {}
    stats = LatencyStats(
        label=label, num_requests=len(requests), completed=0, generated_tokens=0, seconds=0.0
    )
    preemptions_before = llm.stats.preemptions

    started = time.perf_counter()
    while pending or llm.scheduler.num_unfinished:
        now = time.perf_counter() - started
        while pending and pending[0].arrival <= now:
            request = pending.popleft()
            sequence = llm.add_request(
                request.prompt_token_ids, max_tokens=request.output_len, ignore_eos=True
            )
            arrival_of[sequence.seq_id] = now

        if not llm.scheduler.num_unfinished:
            time.sleep(max(pending[0].arrival - now, 0.0))
            continue

        before = time.perf_counter()
        emitted = llm.step()
        stats.iteration_seconds.append(time.perf_counter() - before)

        for sequence, _token in emitted:
            at = time.perf_counter() - started
            previous = last_token_at.get(sequence.seq_id)
            if previous is None:
                stats.ttft.append(at - arrival_of[sequence.seq_id])
            else:
                stats.inter_token.append(at - previous)
            last_token_at[sequence.seq_id] = at
            stats.generated_tokens += 1
            stats.completed += int(sequence.is_done())

        stats.iterations += 1
        # The finished list exists for the tests; over thousands of requests it would hold
        # every prompt of the run in memory.
        llm.scheduler.finished.clear()

        if progress_every and stats.iterations % progress_every == 0:
            print(
                f"  {label}: {stats.completed}/{len(requests)} done, "
                f"{len(pending)} unarrived, {llm.scheduler.num_unfinished} in flight",
                flush=True,
            )

    stats.seconds = time.perf_counter() - started
    stats.preemptions = llm.stats.preemptions - preemptions_before
    llm.manager.check_no_leaks()
    return stats


def run_scheduler(arguments) -> tuple[list[LatencyStats], GpuState | None]:
    """`--mode scheduler`: the same arrival schedule under both policies."""
    from mini_vllm import LLM

    print(f"loading {arguments.model} into the engine ...")
    llm = LLM(
        arguments.model,
        device=arguments.device,
        max_sequences=arguments.max_sequences,
        max_batched_tokens=arguments.max_batched_tokens,
        chunk_size=arguments.chunk_size,
        use_cuda_kernels=arguments.use_cuda_kernels,
        kv_fraction=arguments.kv_fraction,
    )
    print(f"  {llm}")

    requests = stress_requests(
        num_requests=arguments.num_requests,
        rate=arguments.rate,
        vocab_size=llm.config.vocab_size,
        short_len=arguments.short_len,
        long_len=arguments.long_len,
        long_every=arguments.long_every,
        output_len=arguments.output_len,
        seed=arguments.seed,
    )
    long_prompts = sum(1 for request in requests if request.prompt_len == arguments.long_len)
    print(
        f"  {len(requests)} requests at {arguments.rate:.1f}/s: "
        f"{len(requests) - long_prompts} of {arguments.short_len} tokens, "
        f"{long_prompts} of {arguments.long_len}"
    )

    # Same requests, same engine, same pool, one boolean pair apart. The chunked run goes
    # first so a crash in the baseline still leaves the interesting number.
    policies = (
        ("chunked prefill", {"enable_chunked_prefill": True, "prefill_priority": False}),
        ("prefill-priority", {"enable_chunked_prefill": False, "prefill_priority": True}),
    )

    results = []
    with ClockSampler() as sampler:
        for label, changes in policies:
            print(f"running: {label} ...", flush=True)
            llm.reconfigure(**changes)
            results.append(run_stress(llm, requests, label, arguments.progress_every))

    return results, sampler.peak


def report_scheduler(arguments) -> None:
    """Print the policy comparison for `--mode scheduler`."""
    results, state = run_scheduler(arguments)

    print(
        f"\n{arguments.num_requests} requests, Poisson at {arguments.rate:.1f}/s, "
        f"one {arguments.long_len}-token prompt every {arguments.long_every}, "
        f"{arguments.output_len} output tokens each.\n"
        f"budget {arguments.max_batched_tokens} tokens per iteration, "
        f"chunk {arguments.chunk_size}.\n"
    )
    for result in results:
        print(result.describe())
        print(
            f"{'':20} {result.completed} completed, {result.iterations} iterations "
            f"(p99 {result.p99_iteration_ms:.0f} ms, slowest {result.max_iteration_ms:.0f} ms), "
            f"{result.stalled_seconds:.1f} s of decode stalled, "
            f"{result.preemptions} preemptions, no leaked blocks"
        )

    if len(results) == 2:
        chunked, baseline = results
        rows = (
            ("worst decode gap", chunked.max_inter_token_ms, baseline.max_inter_token_ms, "ms"),
            ("P99 decode gap", chunked.p99_inter_token_ms, baseline.p99_inter_token_ms, "ms"),
            ("slowest iteration", chunked.max_iteration_ms, baseline.max_iteration_ms, "ms"),
            ("decode stalled", chunked.stalled_seconds, baseline.stalled_seconds, "s"),
            ("output rate", chunked.tokens_per_second, baseline.tokens_per_second, "tok/s"),
        )
        print("\nchunked prefill against the baseline, on identical arrivals:")
        for label, ours, theirs, unit in rows:
            print(
                f"  {label:<20} {ours:8.1f} {unit:<6} vs {theirs:8.1f} {unit:<6} "
                f"{theirs / max(ours, 1e-9):5.2f}x"
            )

        print(
            "\nThe worst-case gap is the bound that holds regardless of load. Chunking caps an\n"
            f"iteration at the {arguments.max_batched_tokens}-token budget, so the longest a "
            "decode can wait is set\nby configuration rather than by the longest prompt to "
            f"arrive. The baseline gives\na {arguments.long_len}-token prompt an iteration to "
            "itself and excludes decodes from it, so their\nwait is that prompt's length plus "
            "the next iteration.\n"
            "\n`decode stalled` is the same total under both policies: chunked prefill\n"
            "redistributes the stall rather than removing it, spreading one chunk of waiting\n"
            "across many decodes instead of a whole prompt across a few. Whether that reaches\n"
            "P99 depends on how many decodes are in flight when a long prompt lands, so read\n"
            "the P99 row alongside the TTFT above it: a run whose queue never grows has few\n"
            "affected decodes per prompt and a P99 that misses them entirely."
        )

    _report_gpu_state(state)


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

            # The uncached model has no cache to prefill, so TTFT and the decode rate are
            # both timed through the naive loop.
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
    """Time the uncached generation loop, for the comparison that motivates the cache."""
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


def run_prefix_cache(arguments) -> tuple[list[tuple[str, float, int, int]], GpuState | None]:
    """`--mode prefix-cache`: the same shared-preamble workload with the cache on and off.

    The workload is the one prefix caching exists for and the one a chat server has: many
    requests opening with the same long system prompt and differing only in a short
    question. The reported figure is TTFT, where a cache hit lands: it removes prompt
    tokens from the prefill, so the first token arrives sooner while decode is untouched.
    """
    from mini_vllm import LLM
    from mini_vllm.sampler import SamplingParams

    preamble = (
        "You are a careful, concise assistant. Answer accurately, admit uncertainty, "
        "and prefer short replies to long ones. "
    ) * 12
    questions = [
        "What is the capital of France?",
        "Name a prime number above fifty.",
        "What colour is a ripe banana?",
        "How many days are in a leap year?",
        "What is the chemical symbol for gold?",
        "Who wrote the Odyssey?",
    ]
    prompts = [preamble + question for question in questions]
    greedy = SamplingParams(temperature=0.0)

    rows: list[tuple[str, float, float, int]] = []
    with ClockSampler() as sampler:
        for label, enabled in (("caching off", False), ("caching on", True)):
            llm = LLM(
                arguments.model,
                device=arguments.device,
                kv_fraction=arguments.kv_fraction,
                enable_prefix_caching=enabled,
                use_cuda_kernels=arguments.use_cuda_kernels,
            )
            # Warm the cache the way a server would: by the time the second request
            # arrives the preamble is resident, and a cold first run would understate
            # what the cache does for every request after it.
            llm.generate(prompts[0], sampling_params=greedy, max_tokens=1)

            # One request at a time, so TTFT is per request. Submitted together, six
            # prefills share an iteration and the figure would measure the batch instead.
            ttfts = []
            for prompt in prompts:
                started = time.perf_counter()
                for _update in llm.generate_stream(
                    prompt, sampling_params=greedy, max_tokens=arguments.output_len
                ):
                    ttfts.append(time.perf_counter() - started)
                    break

            total_started = time.perf_counter()
            llm.generate(prompts, sampling_params=greedy, max_tokens=arguments.output_len)
            total = time.perf_counter() - total_started

            rows.append(
                (label, 1000 * sum(ttfts) / len(ttfts), total, llm.stats.cached_tokens)
            )
            del llm
            _free_device_memory()

    return rows, sampler.peak


def report_prefix_cache(arguments) -> None:
    """Print the cache-on/cache-off comparison for `--mode prefix-cache`."""
    rows, state = run_prefix_cache(arguments)

    print(
        f"\n6 requests sharing one long preamble, {arguments.output_len} output tokens "
        f"each, greedy.\n"
    )
    print(f"{'policy':<14} {'mean TTFT':>11} {'batch secs':>11} {'cached tokens':>15}")
    for label, ttft, total, cached in rows:
        print(f"{label:<14} {ttft:>9.1f}ms {total:>11.3f} {cached:>15}")

    if len(rows) == 2:
        off, on = rows[0], rows[1]
        print(
            f"\nwith the cache: {off[1] / on[1]:.2f}x on TTFT. Decode is untouched, which is\n"
            "why the whole-batch column barely moves — a cache hit removes prompt tokens\n"
            "from the prefill and nothing else, so a run dominated by decode cannot show it."
        )

    _report_gpu_state(state)


def run_spec(arguments) -> tuple[list[tuple[str, float, int, float, float]], GpuState | None]:
    """`--mode spec`: speculation off, then on at a sweep of draft depths.

    Reports the acceptance rate beside the wall clock, since either alone is misleading: a
    deep draft is accepted often and costs nearly what the target costs, while a shallow one
    is cheap and rejected.
    """
    from mini_vllm import LLM
    from mini_vllm.sampler import SamplingParams

    prompt = "Explain, step by step, why the sky appears blue during the day."
    greedy = SamplingParams(temperature=0.0)
    depths = [int(depth) for depth in arguments.draft_layers.split(",") if depth]

    rows: list[tuple[str, float, int, float, float]] = []
    with ClockSampler() as sampler:
        for label, spec_tokens, layers in [("no speculation", 0, 0)] + [
            (f"k={arguments.num_speculative_tokens}, {depth} draft layers",
             arguments.num_speculative_tokens, depth)
            for depth in depths
        ]:
            llm = LLM(
                arguments.model,
                device=arguments.device,
                kv_fraction=arguments.kv_fraction,
                num_speculative_tokens=spec_tokens,
                num_draft_layers=layers or None,
                use_cuda_kernels=arguments.use_cuda_kernels,
            )
            llm.generate(prompt, sampling_params=greedy, max_tokens=4)  # warm up

            started = time.perf_counter()
            completion = llm.generate(
                prompt, sampling_params=greedy, max_tokens=arguments.output_len
            )[0]
            elapsed = time.perf_counter() - started

            if llm.spec is None:
                rows.append((label, elapsed, len(completion.token_ids), 0.0, 1.0))
            else:
                stats = llm.spec.stats
                rows.append((
                    label, elapsed, len(completion.token_ids),
                    stats.acceptance_rate, stats.tokens_per_step,
                ))
            del llm
            _free_device_memory()

    return rows, sampler.peak


def report_spec(arguments) -> None:
    """Print the speculation table for `--mode spec`."""
    rows, state = run_spec(arguments)

    print(f"\none greedy request, {arguments.output_len} output tokens.\n")
    print(f"{'configuration':<32} {'seconds':>9} {'tokens':>7} {'accept':>8} {'tok/pass':>9}")
    for label, elapsed, tokens, acceptance, per_step in rows:
        print(f"{label:<32} {elapsed:>9.3f} {tokens:>7} {acceptance:>8.3f} {per_step:>9.2f}")

    print(
        "\nAcceptance and wall clock have to be read together: a draft deep enough to be\n"
        "accepted costs nearly what the target costs, and a shallow one is rejected."
    )
    _report_gpu_state(state)


def _free_device_memory() -> None:
    """Release the allocator's cache so the next engine sizes itself against real free
    memory."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Mini-vLLM.")
    parser.add_argument(
        "--mode",
        default="single",
        choices=["single", "kernels", "throughput", "scheduler", "prefix-cache", "spec"],
    )
    parser.add_argument(
        "--num-speculative-tokens", type=int, default=4, help="--mode spec: proposals per step"
    )
    parser.add_argument(
        "--draft-layers", default="4,14,28", help="--mode spec: draft depths to sweep"
    )
    parser.add_argument("--input-len", type=int, default=128)
    # Resolved below, since the right default depends on the mode: a latency run wants a
    # long generation from few requests, and a stress run wants the opposite.
    parser.add_argument("--output-len", type=int, default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--batch-sizes", default="1,4,16", help="--mode throughput: concurrency to sweep")
    parser.add_argument("--max-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-sequences", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--kv-fraction", type=float, default=0.4)
    parser.add_argument("--num-requests", type=int, default=2000, help="--mode scheduler")
    # Near what this card sustains on the mix below, which is where scheduling decisions
    # show up. Far under capacity a long prompt catches only one or two decodes and the
    # tail percentiles never see it; far over, every policy accumulates an unbounded queue
    # and the tail measures the backlog rather than the decision. The run reports TTFT:
    # around a second means the rate suits this machine, tens of seconds means lower it.
    parser.add_argument("--rate", type=float, default=16.0, help="--mode scheduler: arrivals/sec")
    parser.add_argument("--short-len", type=int, default=STRESS_SHORT_LEN)
    parser.add_argument("--long-len", type=int, default=STRESS_LONG_LEN)
    parser.add_argument("--long-every", type=int, default=STRESS_LONG_EVERY)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=0, help="iterations between updates")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compare", default=None, choices=["hf"])
    parser.add_argument("--no-cache", action="store_true", help="measure the uncached loop")
    parser.add_argument(
        "--use-cuda-kernels",
        action="store_true",
        help="route ops through hand-written kernels where they exist",
    )
    arguments = parser.parse_args()
    if arguments.output_len is None:
        arguments.output_len = STRESS_OUTPUT_LEN if arguments.mode == "scheduler" else 128

    if arguments.mode == "kernels":
        report_kernels(arguments)
        return
    if arguments.mode == "throughput":
        report_throughput(arguments)
        return
    if arguments.mode == "scheduler":
        report_scheduler(arguments)
        return
    if arguments.mode == "prefix-cache":
        report_prefix_cache(arguments)
        return
    if arguments.mode == "spec":
        report_spec(arguments)
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
