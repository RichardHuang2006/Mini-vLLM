"""The benchmark harness: one mode per performance claim in the README.

Every timed region is bracketed by torch.cuda.synchronize(), warmed up first, and
watched by ClockSampler, which voids a run whose clocks were throttled.
"""

from __future__ import annotations

import argparse
import gc
import math
import platform
import random
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence as SequenceABC
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_vllm import kernels
from mini_vllm.cache import PagedKvPool
from mini_vllm.config import DEFAULT_MODEL_ID, SamplingParams
from mini_vllm.engine import LLM, generate_ids, generate_ids_cached, load
from mini_vllm.model import resolve_model_path
from mini_vllm.ops import RoPE

__all__ = [
    "GpuState",
    "gpu_state",
    "ClockSampler",
    "BandwidthResult",
    "KernelCase",
    "LatencyStats",
    "StressRequest",
    "build_input_ids",
    "copy_ceiling",
    "kernel_cases",
    "measure_bandwidth",
    "percentile",
    "poisson_arrivals",
    "run_stress",
    "stress_requests",
    "theoretical_bandwidth",
    "throughput_prompts",
]

# Repeated to fill any prompt length; real text keeps the continuation readable.
FILLER = (
    "The city of Rome was founded in 753 BC by the twin brothers Romulus and Remus, "
    "and grew over the following centuries into the capital of an empire. "
)

# Prompt lengths a throughput run cycles through; equal lengths would hide the padding.
THROUGHPUT_LENGTHS = (32, 64, 128, 256, 512)

# The scheduler stress mix: short chat-like requests, one long prompt in every twenty.
STRESS_SHORT_LEN = 32
STRESS_LONG_LEN = 2048
STRESS_LONG_EVERY = 20
STRESS_OUTPUT_LEN = 32


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
        """True below half the rated clocks, where the card is idling rather than working."""
        return self.sm_fraction < 0.5 or self.memory_fraction < 0.5

    def describe(self) -> str:
        return (
            f"{self.name}: sm {self.sm_clock}/{self.sm_clock_max} MHz "
            f"({self.sm_fraction:.0%}), mem {self.memory_clock}/{self.memory_clock_max} MHz "
            f"({self.memory_fraction:.0%}), {self.power_state} at {self.power_draw}"
        )


def gpu_state() -> GpuState | None:
    """Read the current clocks from `nvidia-smi`, or None if that is not possible.

    Only meaningful while the GPU is busy, since clocks fall back to idle within
    milliseconds; use :class:`ClockSampler` to observe a run.
    """
    if not shutil.which("nvidia-smi"):
        return None

    fields = ["name", "clocks.sm", "clocks.max.sm", "clocks.mem", "clocks.max.mem",
              "pstate", "power.draw"]
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
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
    either idle or a boost spike.
    """

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.peak: GpuState | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            state = gpu_state()
            if state is not None and (self.peak is None or state.sm_clock > self.peak.sm_clock):
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


def report_gpu_verdict(state: GpuState | None) -> None:
    """The mandatory last line of every mode: was this measurement valid?"""
    if state is None:
        print("\ngpu: clocks not observed (no nvidia-smi or no CUDA) — treat results as unverified")
        return
    print(f"\ngpu peak during run: {state.describe()}")
    if state.is_throttled:
        print(
            "  WARNING: the GPU ran far below its clocks, so these numbers measure the\n"
            "  power state rather than the code. Every result above is INVALID as an\n"
            "  absolute figure."
        )
    else:
        print("  clocks healthy; no throttling detected")


def evidence_header(
    mode: str,
    metric: str,
    model: str,
    dtype: str,
    batch: object,
    prompt_lens: object,
    output_tokens: object,
    warmup: int,
) -> None:
    """The uniform preamble: everything needed to reproduce or disqualify the run."""
    state = gpu_state()
    gpu = state.name if state else ("cpu-only" if not torch.cuda.is_available() else "unknown GPU")
    print(f"== mode {mode} ==")
    print(f"  metric        {metric}")
    print(f"  hardware      {gpu} | {platform.platform(terse=True)}")
    print(f"  torch         {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"  model         {model} | dtype {dtype}")
    print(f"  batch         {batch}")
    print(f"  prompt lens   {prompt_lens}")
    print(f"  output tokens {output_tokens}")
    print(f"  warmup        {warmup}")
    print()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _free_device_memory() -> None:
    """Release the allocator's cache so the next engine sizes itself against real free
    memory; two engines do not fit on an 8 GB card at once."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_input_ids(tokenizer, input_len: int, batch: int, device) -> torch.Tensor:
    """A prompt of exactly ``input_len`` tokens, repeated across the batch."""
    repeats = -(-input_len // max(len(tokenizer(FILLER).input_ids), 1))
    ids = tokenizer(FILLER * repeats, return_tensors="pt").input_ids[:, :input_len]
    if ids.shape[1] < input_len:
        raise ValueError(f"could not build a prompt of {input_len} tokens")
    return ids.expand(batch, -1).contiguous().to(device)


def percentile(values: SequenceABC[float], fraction: float) -> float:
    """The nearest-rank percentile, so a tail figure names a latency someone waited."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def mode_single(args) -> None:
    """One request: TTFT and decode tok/s, optionally against transformers.

    TTFT is dominated by a compute-bound pass over the whole prompt; decode tokens/sec by
    memory bandwidth. `--no-cache` times the quadratic uncached loop instead.
    """
    evidence_header(
        "single", "TTFT (ms) and decode tok/s", args.model, "bfloat16",
        args.batch, args.input_len, args.output_len, args.warmup,
    )

    print(f"loading {args.model} ...")
    loaded = load(args.model, args.device, cached=not args.no_cache,
                  use_cuda_kernels=args.use_cuda_kernels)
    device = loaded.model.embedding.weight.device
    input_ids = build_input_ids(loaded.tokenizer, args.input_len, args.batch, device)

    rows: list[tuple[str, float, float]] = []  # label, ttft_ms, decode tok/s
    with ClockSampler() as sampler:
        with torch.no_grad():
            if args.no_cache:
                # No cache to prefill: TTFT is one full forward, decode is the naive loop.
                for _ in range(args.warmup):
                    generate_ids(loaded.model, input_ids, max_tokens=4)
                _synchronize(device)
                started = time.perf_counter()
                loaded.model(input_ids)
                _synchronize(device)
                ttft = time.perf_counter() - started
                started = time.perf_counter()
                generate_ids(loaded.model, input_ids, max_tokens=args.output_len)
                _synchronize(device)
                total = time.perf_counter() - started
                steps = max(args.output_len - 1, 1)
                rows.append(("mini-vllm (no cache)", ttft * 1e3,
                             steps * args.batch / max(total - ttft, 1e-9)))
            else:
                for _ in range(args.warmup):
                    generate_ids_cached(loaded.model, input_ids, max_tokens=8)
                _synchronize(device)
                caches = loaded.model.create_kv_cache()
                started = time.perf_counter()
                logits = loaded.model(input_ids, caches, last_only=True)
                token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                _synchronize(device)
                ttft = time.perf_counter() - started
                steps = max(args.output_len - 1, 1)
                started = time.perf_counter()
                for _ in range(steps):
                    logits = loaded.model(token, caches, last_only=True)
                    token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                _synchronize(device)
                decode = time.perf_counter() - started
                rows.append(("mini-vllm", ttft * 1e3, steps * args.batch / decode))

            if args.compare == "hf":
                print("loading the transformers baseline ...")
                hf = (AutoModelForCausalLM.from_pretrained(
                    resolve_model_path(args.model), dtype=torch.bfloat16).to(device).eval())

                def run(new_tokens: int):
                    return hf.generate(
                        input_ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens,
                        do_sample=False, pad_token_id=hf.generation_config.pad_token_id)

                for _ in range(args.warmup):
                    run(8)
                _synchronize(device)
                started = time.perf_counter()
                run(1)
                _synchronize(device)
                hf_ttft = time.perf_counter() - started
                started = time.perf_counter()
                run(args.output_len)
                _synchronize(device)
                hf_total = time.perf_counter() - started
                steps = max(args.output_len - 1, 1)
                rows.append(("transformers", hf_ttft * 1e3,
                             steps * args.batch / max(hf_total - hf_ttft, 1e-9)))

    print("\nops dispatch:")
    print(kernels.dispatch_report(args.use_cuda_kernels))
    print()
    for label, ttft_ms, decode_rate in rows:
        print(f"{label:<22} TTFT {ttft_ms:8.1f} ms   decode {decode_rate:7.1f} tok/s")
    if len(rows) == 2:
        (l0, t0, d0), (l1, t1, d1) = rows
        print(f"\nrelative to {l1}: TTFT {t1 / t0:.2f}x, decode {d0 / d1:.2f}x")
    report_gpu_verdict(sampler.peak)


@dataclass(frozen=True)
class BandwidthResult:
    """One kernel, timed and converted into bytes per second.

    `bytes_moved` is the traffic the op cannot avoid, so a redundant reread lowers the
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

    def describe(self, peak: float | None = None) -> str:
        share = self.gigabytes_per_second / peak if peak else None
        suffix = f" ({share:.0%} of peak)" if share is not None else ""
        return (f"{self.label:<44} {self.microseconds:9.1f} us   "
                f"{self.gigabytes_per_second:7.1f} GB/s{suffix}")


def theoretical_bandwidth(device: int = 0) -> float | None:
    """The card's peak memory bandwidth in GB/s, from its clock and bus width."""
    if not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(device)
    clock_hertz = properties.memory_clock_rate * 1e3  # the attribute is in kHz
    return 2.0 * clock_hertz * (properties.memory_bus_width / 8) / 1e9  # double data rate


def measure_bandwidth(
    call: Callable[[], object],
    bytes_moved: int,
    label: str,
    warmup: int = 5,
    iterations: int = 100,
) -> BandwidthResult:
    """Time ``call`` back to back and report the traffic rate it sustained."""
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    # One pair of syncs for the whole run: per iteration it would time the sync.
    started = time.perf_counter()
    for _ in range(iterations):
        call()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    return BandwidthResult(label=label, bytes_moved=bytes_moved, seconds=elapsed / iterations)


def copy_ceiling(megabytes: int = 256, dtype: torch.dtype = torch.bfloat16) -> BandwidthResult:
    """What a bare `copy_` of a large buffer achieves: the practical bandwidth ceiling."""
    elements = megabytes * 1024 * 1024 // torch.tensor([], dtype=dtype).element_size()
    source = torch.randn(elements, device="cuda", dtype=dtype)
    destination = torch.empty_like(source)
    moved = 2 * source.numel() * source.element_size()  # one read, one write
    return measure_bandwidth(lambda: destination.copy_(source), moved,
                             f"copy_ ({megabytes} MB)", iterations=20)


@dataclass(frozen=True)
class KernelCase:
    """A kernel, the PyTorch expression it replaces, and the traffic both must move."""

    label: str
    kernel: Callable[[], object]
    reference: Callable[[], object]
    bytes_moved: int


def rows_exceeding_l2(width: int, dtype: torch.dtype, multiple: int = 8) -> int:
    """How many rows it takes for the working set to be `multiple` times the L2.

    A case whose input and output both fit in L2 reports cache bandwidth, not memory
    bandwidth; the symptom is a rate above the card's DRAM peak.
    """
    element_size = torch.tensor([], dtype=dtype).element_size()
    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    bytes_per_row = 2 * width * element_size  # the row is read once and written once
    return max(1, multiple * l2_bytes // bytes_per_row)


def _quantize_into(key, value, pools, use_cuda: bool):
    """Scatter one step's KV into ``pools`` and return the key pool for comparison.

    `kernels.quantize_scatter` writes in place and returns nothing, but a case must
    return what it computed so a speedup cannot be one side doing less work.
    """
    kernels.quantize_scatter(key, value, *pools, 1.0, 1.0, use_cuda=use_cuda)
    return pools[0]


def kernel_cases(dtype: torch.dtype = torch.bfloat16) -> list[KernelCase]:
    """One case per implemented kernel per regime that matters.

    Widths are the model's real ones. Elementwise kernels get one row (launch latency,
    the decode shape) and a past-L2 size; attention sweeps the context length instead.
    """
    if not torch.cuda.is_available():
        return []

    module = kernels.load_extension()
    implemented = kernels.cuda_kernel_names()
    cases: list[KernelCase] = []

    def regimes(width: int) -> tuple[tuple[int, str], ...]:
        return ((1, "decode, 1 token"), (rows_exceeding_l2(width, dtype), "past L2, DRAM-bound"))

    if "rmsnorm" in implemented:
        hidden_size = 1024  # Qwen3-0.6B's E
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
        for rows, note in regimes(hidden_size):
            x = torch.randn(rows, hidden_size, device="cuda", dtype=dtype)
            moved = (2 * x.numel() + weight.numel()) * x.element_size()
            cases.append(KernelCase(
                label=f"rmsnorm E={hidden_size} ({note})",
                kernel=lambda x=x, w=weight: module.rmsnorm(x, w, 1e-6),
                reference=lambda x=x, w=weight: kernels.rmsnorm(x, w, 1e-6),
                bytes_moved=moved))

    if "swiglu" in implemented:
        intermediate = 3072  # Qwen3-0.6B's MLP width
        for rows, note in regimes(2 * intermediate):
            gate = torch.randn(rows, intermediate, device="cuda", dtype=dtype)
            up = torch.randn_like(gate)
            # Two reads and one write, so the traffic is 3x the tensor, not 2x.
            moved = 3 * gate.numel() * gate.element_size()
            cases.append(KernelCase(
                label=f"swiglu I={intermediate} ({note})",
                kernel=lambda gate=gate, up=up: module.swiglu(gate, up),
                reference=lambda gate=gate, up=up: kernels.swiglu(gate, up),
                bytes_moved=moved))

    if "rope" in implemented:
        heads, head_dim = 16, 128  # Qwen3-0.6B's H_q and D
        per_token = heads * head_dim
        longest = max(rows for rows, _ in regimes(per_token))
        tables = RoPE(head_dim, longest, 1_000_000.0, device="cuda")
        for rows, note in regimes(per_token):
            x = torch.randn(1, rows, heads, head_dim, device="cuda", dtype=dtype)
            positions = torch.arange(rows, device="cuda")
            # Table rows are shared by a token's heads: D fp32 pairs per token extra.
            moved = 2 * x.numel() * x.element_size() + 2 * rows * head_dim * 4
            cases.append(KernelCase(
                label=f"rope H={heads} D={head_dim} ({note})",
                kernel=lambda x=x, p=positions: module.rope(x, p, tables.cos, tables.sin),
                reference=lambda x=x, p=positions: kernels.rope(x, p, tables.cos, tables.sin),
                bytes_moved=moved))

    query_heads, kv_heads, head_dim, block_size = 16, 8, 128, 16
    scale = 1.0 / math.sqrt(head_dim)

    if "decode_attention" in implemented:
        # At 8192 one layer's K/V is 32 MB, this card's L2: only that case is DRAM-bound.
        for source_len in (1024, 8192):
            q = torch.randn(1, query_heads, 1, head_dim, device="cuda", dtype=dtype)
            k = torch.randn(1, kv_heads, source_len, head_dim, device="cuda", dtype=dtype)
            v = torch.randn_like(k)
            moved = (k.numel() + v.numel()) * k.element_size()
            cases.append(KernelCase(
                label=f"decode_attention S={source_len}",
                kernel=lambda q=q, k=k, v=v: module.decode_attention(q, k, v, scale),
                reference=lambda q=q, k=k, v=v: kernels.attention(q, k, v),
                bytes_moved=moved))

    if "flash_prefill" in implemented:
        # The one compute-bound case, so the speedup column matters and GB/s does not.
        for query_len in (512,):
            q = torch.randn(1, query_heads, query_len, head_dim, device="cuda", dtype=dtype)
            k = torch.randn(1, kv_heads, query_len, head_dim, device="cuda", dtype=dtype)
            v = torch.randn_like(k)
            moved = (q.numel() + k.numel() + v.numel()) * q.element_size()
            cases.append(KernelCase(
                label=f"flash_prefill L=S={query_len}",
                kernel=lambda q=q, k=k, v=v: module.flash_prefill(q, k, v, scale),
                reference=lambda q=q, k=k, v=v: kernels.attention(q, k, v, mask="causal"),
                bytes_moved=moved))

    if "paged_attention" in implemented:
        # The engine's decode shape: the speedup is a removed cache copy per iteration.
        for batch, context_len in ((16, 1024), (64, 1024)):
            blocks_each = -(-context_len // block_size)
            pool_blocks = batch * blocks_each
            keys = torch.randn(pool_blocks, block_size, kv_heads, head_dim,
                               device="cuda", dtype=dtype)
            values = torch.randn_like(keys)
            # Shuffled: a pool in logical order would give reads an aged one never does.
            shuffled = torch.randperm(pool_blocks, device="cuda", dtype=torch.int32)
            block_tables = shuffled.reshape(batch, blocks_each).contiguous()
            q = torch.randn(batch, query_heads, head_dim, device="cuda", dtype=dtype)
            cu_seqlens = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
            contexts = torch.full((batch,), context_len, device="cuda", dtype=torch.int32)
            lengths = torch.ones(batch, device="cuda", dtype=torch.int32)
            moved = 2 * batch * context_len * kv_heads * head_dim * keys.element_size()
            paged = (keys, values, block_tables, cu_seqlens, contexts, lengths)
            cases.append(KernelCase(
                label=f"paged_attention B={batch} S={context_len} (decode)",
                kernel=lambda q=q, p=paged, s=context_len: module.paged_attention(
                    q, *p, 1, s, scale),
                reference=lambda q=q, p=paged, s=context_len: kernels.paged_attention(
                    q, *p, 1, s, scale),
                bytes_moved=moved))

        # A prefill chunk: the case the kernel loses, kept so the shapes stay honest.
        for query_len, context_len in ((512, 2048),):
            blocks_each = -(-context_len // block_size)
            keys = torch.randn(blocks_each, block_size, kv_heads, head_dim,
                               device="cuda", dtype=dtype)
            values = torch.randn_like(keys)
            block_tables = (torch.randperm(blocks_each, device="cuda", dtype=torch.int32)
                            .reshape(1, blocks_each).contiguous())
            q = torch.randn(query_len, query_heads, head_dim, device="cuda", dtype=dtype)
            cu_seqlens = torch.tensor([0, query_len], device="cuda", dtype=torch.int32)
            contexts = torch.tensor([context_len], device="cuda", dtype=torch.int32)
            lengths = torch.tensor([query_len], device="cuda", dtype=torch.int32)
            moved = (q.numel() + keys.numel() + values.numel()) * q.element_size()
            paged = (keys, values, block_tables, cu_seqlens, contexts, lengths)
            cases.append(KernelCase(
                label=f"paged_attention L={query_len} S={context_len} (prefill chunk)",
                kernel=lambda q=q, p=paged, ql=query_len, s=context_len:
                    module.paged_attention(q, *p, ql, s, scale),
                reference=lambda q=q, p=paged, ql=query_len, s=context_len:
                    kernels.paged_attention(q, *p, ql, s, scale),
                bytes_moved=moved))

    if "kv_quantize_scatter" in implemented:
        # The clearest fusion case: the reference's temporary against one pass.
        for tokens, note in ((1, "decode, 1 token"), (4096, "past L2")):
            key = torch.randn(tokens, kv_heads, head_dim, device="cuda", dtype=dtype)
            value = torch.randn_like(key)
            pool_slots = max(4096, 4 * tokens)
            slots = torch.randperm(pool_slots, device="cuda")[:tokens].to(torch.int64)

            def pools() -> tuple:
                shape = (pool_slots, kv_heads, head_dim)
                empty = torch.zeros(shape, device="cuda", dtype=torch.float8_e4m3fn)
                return (empty, torch.zeros_like(empty), slots)

            moved = 2 * key.numel() * (key.element_size() + 1)
            kernel_pools, reference_pools = pools(), pools()
            cases.append(KernelCase(
                label=f"kv_quantize_scatter T={tokens} ({note})",
                kernel=lambda k=key, v=value, p=kernel_pools: _quantize_into(k, v, p, True),
                reference=lambda k=key, v=value, p=reference_pools:
                    _quantize_into(k, v, p, False),
                bytes_moved=moved))

    return cases


def mode_kernels(args) -> None:
    """Achieved memory bandwidth for each kernel beside the PyTorch it replaced."""
    evidence_header(
        "kernels", "achieved GB/s and kernel-vs-torch speedup", "synthetic tensors",
        "bfloat16", "per-case", "per-case (see labels)", "n/a", args.warmup,
    )
    cases = kernel_cases(torch.bfloat16)
    if not cases:
        print("no CUDA device — nothing to measure")
        return

    with ClockSampler() as sampler:
        ceiling = copy_ceiling()
        rows = [
            (measure_bandwidth(case.kernel, case.bytes_moved, case.label, args.warmup),
             measure_bandwidth(case.reference, case.bytes_moved,
                               "  ...the torch it replaces", args.warmup))
            for case in cases
        ]

    peak = theoretical_bandwidth()
    l2_megabytes = torch.cuda.get_device_properties(0).L2_cache_size // (1024 * 1024)
    print(f"theoretical peak: {peak:.0f} GB/s" if peak else "theoretical peak: unknown")
    print(f"{ceiling.describe(peak)}   <- the practical ceiling")
    print(
        f"\nL2 is {l2_megabytes} MB on this card, so a case whose input and output fit\n"
        f"inside it reports cache bandwidth, not memory bandwidth; only the past-L2\n"
        f"rows' share of peak means anything.\n"
    )
    for kernel_result, reference_result in rows:
        speedup = reference_result.seconds / kernel_result.seconds if kernel_result.seconds else 0
        print(kernel_result.describe(peak))
        print(f"{reference_result.describe(peak)}   ({speedup:.2f}x)")
    report_gpu_verdict(sampler.peak)


def throughput_prompts(tokenizer, num_requests: int) -> list[str]:
    """`num_requests` prompts whose lengths cycle through `THROUGHPUT_LENGTHS`."""
    lengths = [THROUGHPUT_LENGTHS[i % len(THROUGHPUT_LENGTHS)] for i in range(num_requests)]
    prompts = []
    for length in lengths:
        repeats = -(-length // max(len(tokenizer(FILLER).input_ids), 1))
        ids = tokenizer(FILLER * repeats).input_ids[:length]
        prompts.append(tokenizer.decode(ids))
    return prompts


def mode_throughput(args) -> None:
    """Output tokens/sec over a whole request set, by concurrency, vs transformers.

    The comparison is asymmetric by construction, which is the finding: `generate` takes
    one padded rectangle, while the engine gives each sequence its own length.
    """
    batch_sizes = [int(size) for size in args.batch_sizes.split(",")]
    evidence_header(
        "throughput", "output tok/s over a whole request set (eos ignored)", args.model,
        "bfloat16", batch_sizes, f"cycling {THROUGHPUT_LENGTHS}", args.output_len, args.warmup,
    )

    path = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path)

    # Loaded first: the engine sizes its KV pool from the memory free at startup.
    hf_model = None
    if args.compare == "hf":
        print("loading the transformers baseline ...")
        hf_model = (AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
                    .to(args.device).eval())

    print(f"loading {args.model} into the engine ...")
    llm = LLM(args.model, device=args.device, max_sequences=max(batch_sizes),
              max_batched_tokens=args.max_batched_tokens,
              use_cuda_kernels=args.use_cuda_kernels, kv_fraction=args.kv_fraction)
    print(f"  {llm}\nops dispatch:\n{kernels.dispatch_report(args.use_cuda_kernels)}\n")

    rows = []
    with ClockSampler() as sampler:
        for batch in batch_sizes:
            prompts = throughput_prompts(tokenizer, batch)
            print(f"measuring {batch} concurrent requests ...")

            llm.generate(prompts[:2], max_tokens=4, ignore_eos=True)  # warmup
            torch.cuda.synchronize()
            started = time.perf_counter()
            completions = llm.generate(prompts, max_tokens=args.output_len, ignore_eos=True)
            torch.cuda.synchronize()
            ours_seconds = time.perf_counter() - started
            ours_tokens = sum(completion.num_tokens for completion in completions)

            theirs_rate = None
            if hf_model is not None:
                encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                                    padding_side="left")
                encoded = {key: value.to(hf_model.device) for key, value in encoded.items()}
                with torch.no_grad():
                    hf_model.generate(**encoded, max_new_tokens=4, min_new_tokens=4,
                                      do_sample=False,
                                      pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    hf_model.generate(**encoded, max_new_tokens=args.output_len,
                                      min_new_tokens=args.output_len, do_sample=False,
                                      pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                    torch.cuda.synchronize()
                    theirs_seconds = time.perf_counter() - started
                theirs_rate = len(prompts) * args.output_len / theirs_seconds

            rows.append((batch, ours_tokens / ours_seconds, theirs_rate))

    print()
    for batch, ours, theirs in rows:
        line = f"batch {batch:>3}   mini-vllm {ours:8.1f} out tok/s"
        if theirs is not None:
            line += f"   transformers {theirs:8.1f} out tok/s   {ours / theirs:5.2f}x"
        print(line)
    if len(rows) > 1:
        scaling = rows[-1][1] / rows[0][1]
        print(
            f"\nfrom batch {rows[0][0]} to {rows[-1][0]}: {scaling:.1f}x the output rate.\n"
            "Decode is memory-bound on the weights, and every sequence in an iteration\n"
            "reads them once between them, so the batch is nearly free until the\n"
            "arithmetic runs out."
        )
    report_gpu_verdict(sampler.peak)


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
    """Arrival times of a Poisson process of `rate` requests per second; 0 means at once.

    Poisson rather than evenly spaced, because bursts are what stress a scheduler.
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

    Prompts are random token ids: nothing here reads the output, and a tokenizer call per
    request would put minutes of Python between the engine and the measurement.
    """
    generator = random.Random(seed + 1)
    arrivals = poisson_arrivals(num_requests, rate, seed)

    requests = []
    for index, arrival in enumerate(arrivals):
        is_long = long_every > 0 and index % long_every == long_every - 1
        length = long_len if is_long else short_len
        prompt = tuple(generator.randrange(1, vocab_size) for _ in range(length))
        requests.append(StressRequest(arrival=arrival, prompt_token_ids=prompt,
                                      output_len=output_len))
    return requests


@dataclass
class LatencyStats:
    """One policy's run over the stress mix, in the units an SLO is written in.

    `inter_token` is the gap between consecutive tokens of one sequence — the latency a
    streaming caller sees, and the tail this mode exists to move.
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

        It separates a decode that waited several iterations (a policy problem) from one
        iteration that simply took a long time.
        """
        return max(self.iteration_seconds, default=0.0) * 1000.0

    @property
    def p99_iteration_ms(self) -> float:
        return percentile(self.iteration_seconds, 0.99) * 1000.0

    @property
    def stalled_seconds(self) -> float:
        """Total time callers spent waiting beyond a normal iteration, summed.

        The load-robust view of the tail: a percentile depends on how many callers were
        unlucky, and so on the arrival rate; this does not.
        """
        floor = percentile(self.iteration_seconds, 0.50)
        return sum(max(gap - floor, 0.0) for gap in self.inter_token)

    @property
    def p99_ttft_ms(self) -> float:
        return percentile(self.ttft, 0.99) * 1000.0

    def describe(self) -> str:
        return (f"{self.label:<20} decode p50 {self.p50_inter_token_ms:6.1f} ms  "
                f"p99 {self.p99_inter_token_ms:7.1f} ms  max {self.max_inter_token_ms:7.1f} ms   "
                f"TTFT p99 {self.p99_ttft_ms:8.1f} ms   {self.tokens_per_second:7.1f} out tok/s")


def run_stress(llm, requests: SequenceABC[StressRequest], label: str,
               progress_every: int = 0) -> LatencyStats:
    """Replay an arrival schedule through the engine, timing every token.

    Arrivals are quantized to iteration boundaries and the resulting wait counts toward
    TTFT, which is correct: a real server admits at the same boundaries.
    """
    pending = deque(sorted(requests, key=lambda request: request.arrival))
    arrival_of: dict[int, float] = {}
    last_token_at: dict[int, float] = {}
    stats = LatencyStats(label=label, num_requests=len(requests), completed=0,
                         generated_tokens=0, seconds=0.0)
    preemptions_before = llm.stats.preemptions

    started = time.perf_counter()
    while pending or llm.scheduler.num_unfinished:
        now = time.perf_counter() - started
        while pending and pending[0].arrival <= now:
            request = pending.popleft()
            sequence = llm.add_request(request.prompt_token_ids,
                                       max_tokens=request.output_len, ignore_eos=True)
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
        # The finished list is for the tests; it would hold every prompt of this run.
        llm.scheduler.finished.clear()

        if progress_every and stats.iterations % progress_every == 0:
            print(f"  {label}: {stats.completed}/{len(requests)} done, "
                  f"{len(pending)} unarrived, {llm.scheduler.num_unfinished} in flight",
                  flush=True)

    stats.seconds = time.perf_counter() - started
    stats.preemptions = llm.stats.preemptions - preemptions_before
    llm.manager.check_no_leaks()
    return stats


def mode_scheduler(args) -> None:
    """The same Poisson arrival schedule under chunked prefill and prefill-priority.

    Throughput barely moves between the two: what changes is which requests wait. The
    chunked run goes first so a crash in the baseline still leaves the interesting number.
    """
    evidence_header(
        "scheduler", "inter-token latency tails under both policies", args.model, "bfloat16",
        f"{args.num_requests} requests, Poisson {args.rate:.1f}/s, max_sequences {args.max_sequences}",
        f"{args.short_len} with {args.long_len} every {args.long_every}",
        args.output_len, args.warmup,
    )

    print(f"loading {args.model} into the engine ...")
    llm = LLM(args.model, device=args.device, max_sequences=args.max_sequences,
              max_batched_tokens=args.max_batched_tokens, chunk_size=args.chunk_size,
              use_cuda_kernels=args.use_cuda_kernels, kv_fraction=args.kv_fraction)
    print(f"  {llm}")

    requests = stress_requests(
        num_requests=args.num_requests, rate=args.rate, vocab_size=llm.config.vocab_size,
        short_len=args.short_len, long_len=args.long_len, long_every=args.long_every,
        output_len=args.output_len, seed=args.seed)

    policies = (
        ("chunked prefill", {"enable_chunked_prefill": True, "prefill_priority": False}),
        ("prefill-priority", {"enable_chunked_prefill": False, "prefill_priority": True}),
    )
    results = []
    with ClockSampler() as sampler:
        for label, changes in policies:
            print(f"running: {label} ...", flush=True)
            llm.reconfigure(**changes)
            results.append(run_stress(llm, requests, label, args.progress_every))

    print()
    for result in results:
        print(result.describe())
        print(f"{'':20} {result.completed} completed, {result.iterations} iterations "
              f"(p99 {result.p99_iteration_ms:.0f} ms, slowest {result.max_iteration_ms:.0f} ms), "
              f"{result.stalled_seconds:.1f} s of decode stalled, "
              f"{result.preemptions} preemptions, no leaked blocks")

    if len(results) == 2:
        chunked, baseline = results
        print("\nchunked prefill against the baseline, on identical arrivals:")
        for label, ours, theirs, unit in (
            ("worst decode gap", chunked.max_inter_token_ms, baseline.max_inter_token_ms, "ms"),
            ("P99 decode gap", chunked.p99_inter_token_ms, baseline.p99_inter_token_ms, "ms"),
            ("decode stalled", chunked.stalled_seconds, baseline.stalled_seconds, "s"),
            ("output rate", chunked.tokens_per_second, baseline.tokens_per_second, "tok/s"),
        ):
            print(f"  {label:<20} {ours:8.1f} {unit:<6} vs {theirs:8.1f} {unit:<6} "
                  f"{theirs / max(ours, 1e-9):5.2f}x")
        print(
            "\nThe worst-case gap is the bound that holds regardless of load: chunking caps\n"
            f"an iteration at the {args.max_batched_tokens}-token budget, so the longest a decode "
            "can wait is set by\nconfiguration rather than by the longest prompt to arrive. "
            "`decode stalled` is\nsimilar under both policies: chunking redistributes the stall "
            "(many decodes\nwaiting one chunk each) rather than removing it, and what it removes "
            "is the\nunbounded case. Whether the improvement reaches P99 depends on how many\n"
            "decodes are in flight when a long prompt lands."
        )
    report_gpu_verdict(sampler.peak)


def mode_prefix_cache(args) -> None:
    """The same shared-preamble workload with the radix tree on and off.

    The figure is TTFT, where a hit lands: it removes prompt tokens from the prefill and
    leaves decode exactly as it was.
    """
    preamble = ("You are a careful, concise assistant. Answer accurately, admit "
                "uncertainty, and prefer short replies to long ones. ") * 12
    questions = [
        "What is the capital of France?", "Name a prime number above fifty.",
        "What colour is a ripe banana?", "How many days are in a leap year?",
        "What is the chemical symbol for gold?", "Who wrote the Odyssey?",
    ]
    prompts = [preamble + question for question in questions]
    greedy = SamplingParams(temperature=0.0)

    evidence_header(
        "prefix-cache", "mean TTFT with radix-tree caching on vs off", args.model, "bfloat16",
        f"{len(prompts)} requests, one at a time", "one shared preamble + short question",
        args.output_len, 1,
    )

    rows: list[tuple[str, float, float, int]] = []
    with ClockSampler() as sampler:
        for label, enabled in (("caching off", False), ("caching on", True)):
            llm = LLM(args.model, device=args.device, kv_fraction=args.kv_fraction,
                      enable_prefix_caching=enabled, use_cuda_kernels=args.use_cuda_kernels)
            # Warm the cache the way a server would, before the timed requests.
            llm.generate(prompts[0], sampling_params=greedy, max_tokens=1)

            # One request at a time, so TTFT is per request rather than per batch.
            ttfts = []
            for prompt in prompts:
                started = time.perf_counter()
                for _update in llm.generate_stream(prompt, sampling_params=greedy,
                                                   max_tokens=args.output_len):
                    ttfts.append(time.perf_counter() - started)
                    break

            total_started = time.perf_counter()
            llm.generate(prompts, sampling_params=greedy, max_tokens=args.output_len)
            total = time.perf_counter() - total_started

            rows.append((label, 1000 * sum(ttfts) / len(ttfts), total, llm.stats.cached_tokens))
            del llm
            _free_device_memory()

    print(f"{'policy':<14} {'mean TTFT':>11} {'batch secs':>11} {'cached tokens':>15}")
    for label, ttft, total, cached in rows:
        print(f"{label:<14} {ttft:>9.1f}ms {total:>11.3f} {cached:>15}")
    if len(rows) == 2:
        off, on = rows
        print(
            f"\nwith the cache: {off[1] / on[1]:.2f}x on TTFT. Decode is untouched, which is\n"
            "why the whole-batch column barely moves — a cache hit removes prompt tokens\n"
            "from the prefill and nothing else, so a run dominated by decode cannot show it."
        )
    report_gpu_verdict(sampler.peak)


def mode_fp8(args) -> None:
    """FP8 vs BF16 KV cache: capacity for the same budget, and greedy agreement.

    The model computes in bf16 either way and only the resident cache is quantized, so
    divergence appears where a rounding in a cached key or value flips a near-tie.
    """
    prompts = [
        "The capital of France is",
        "Water boils at a temperature of",
        "The theory of relativity was developed by",
        "In computer science, a hash table is",
    ]
    greedy = SamplingParams(temperature=0.0)

    evidence_header(
        "fp8", "KV pages per budget and greedy token agreement, fp8 vs bf16", args.model,
        "bfloat16 compute, e4m3 vs bf16 KV storage", len(prompts), "short natural prompts",
        args.output_len, 1,
    )

    outputs: dict[str, list[tuple[int, ...]]] = {}
    pool_rows: list[tuple[str, int, float]] = []
    with ClockSampler() as sampler:
        for label, kv_dtype in (("bf16", "auto"), ("fp8", "fp8")):
            llm = LLM(args.model, device=args.device, kv_fraction=args.kv_fraction,
                      kv_cache_dtype=kv_dtype, use_cuda_kernels=args.use_cuda_kernels)
            pool_rows.append((label, llm.manager.num_blocks, llm.kv_cache_bytes / 2**30))
            completions = llm.generate(prompts, sampling_params=greedy,
                                       max_tokens=args.output_len, ignore_eos=True)
            outputs[label] = [completion.token_ids for completion in completions]
            del llm
            _free_device_memory()

    print(f"{'cache':<8} {'pages':>8} {'pool GiB':>10}   (same kv_fraction of free memory)")
    for label, pages, gib in pool_rows:
        print(f"{label:<8} {pages:>8} {gib:>10.2f}")
    if len(pool_rows) == 2:
        ratio = pool_rows[1][1] / max(pool_rows[0][1], 1)
        print(f"\nfp8 holds {ratio:.2f}x the pages of bf16 for the same budget "
              f"(2.0x is the arithmetic; the measured ratio drifts with free-memory noise).")

    per_block = PagedKvPool.bytes_for(1, 1, 16, 8, 128, torch.float8_e4m3fn)
    per_block_bf16 = PagedKvPool.bytes_for(1, 1, 16, 8, 128, torch.bfloat16)
    print(f"bytes per 16-token page per layer: bf16 {per_block_bf16}, fp8 {per_block} "
          f"({per_block_bf16 / per_block:.1f}x)")

    prefixes = []
    for bf16_ids, fp8_ids in zip(outputs["bf16"], outputs["fp8"], strict=True):
        span = min(len(bf16_ids), len(fp8_ids))
        prefixes.append(next((i for i in range(span) if bf16_ids[i] != fp8_ids[i]), span))
    print(f"\ngreedy matching prefix per prompt (of {args.output_len} tokens): {prefixes}")
    print(
        "The matching prefix is the right measure, not whole-trajectory agreement:\n"
        "greedy divergence compounds — the step after the first disagreement runs on a\n"
        "different context, so the tail is noise. Quantizing the cache rounds every\n"
        "stored key and value, so a greedy near-tie eventually flips; the claim is that\n"
        "fp8 tracks bf16 for a useful prefix while halving cache bytes, with the\n"
        "per-element tolerance pinned by the kernel tests."
    )
    report_gpu_verdict(sampler.peak)


def mode_spec(args) -> None:
    """Speculation off, then on at a sweep of self-draft depths.

    Acceptance sits beside the wall clock because either alone misleads: a deep draft is
    accepted often and costs nearly what the target costs, a shallow one is rejected.
    """
    prompt = "Explain, step by step, why the sky appears blue during the day."
    greedy = SamplingParams(temperature=0.0)
    depths = [int(depth) for depth in args.draft_layers.split(",") if depth]

    evidence_header(
        "spec", "acceptance rate, tokens per target pass, and wall clock", args.model,
        "bfloat16", f"1 request; k={args.num_speculative_tokens}; draft layers {depths}",
        "one ~14-token prompt", args.output_len, 1,
    )

    rows: list[tuple[str, float, int, float, float]] = []
    with ClockSampler() as sampler:
        for label, spec_tokens, layers in [("no speculation", 0, 0)] + [
            (f"k={args.num_speculative_tokens}, {depth} draft layers",
             args.num_speculative_tokens, depth)
            for depth in depths
        ]:
            llm = LLM(args.model, device=args.device, kv_fraction=args.kv_fraction,
                      num_speculative_tokens=spec_tokens, num_draft_layers=layers or None,
                      use_cuda_kernels=args.use_cuda_kernels)
            llm.generate(prompt, sampling_params=greedy, max_tokens=4)  # warm up

            started = time.perf_counter()
            completion = llm.generate(prompt, sampling_params=greedy,
                                      max_tokens=args.output_len)[0]
            elapsed = time.perf_counter() - started

            if llm.spec is None:
                rows.append((label, elapsed, len(completion.token_ids), 0.0, 1.0))
            else:
                stats = llm.spec.stats
                rows.append((label, elapsed, len(completion.token_ids),
                             stats.acceptance_rate, stats.tokens_per_step))
            del llm
            _free_device_memory()

    print(f"{'configuration':<32} {'seconds':>9} {'tokens':>7} {'accept':>8} {'tok/pass':>9}")
    for label, elapsed, tokens, acceptance, per_step in rows:
        print(f"{label:<32} {elapsed:>9.3f} {tokens:>7} {acceptance:>8.3f} {per_step:>9.2f}")
    print(
        "\nAcceptance and wall clock have to be read together: a draft deep enough to be\n"
        "accepted costs nearly what the target costs, and a shallow one is rejected.\n"
        "Full-depth self-draft (acceptance near 1.0 with identical output) is the\n"
        "strongest available correctness statement for the verification path."
    )
    report_gpu_verdict(sampler.peak)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Mini-vLLM.")
    parser.add_argument("--mode", default="single", choices=[
        "single", "kernels", "throughput", "scheduler", "prefix-cache", "fp8", "spec"])
    parser.add_argument("--num-speculative-tokens", type=int, default=4,
                        help="--mode spec: proposals per step")
    parser.add_argument("--draft-layers", default="4,14,28",
                        help="--mode spec: draft depths to sweep")
    parser.add_argument("--input-len", type=int, default=128)
    # Resolved below, since the right default depends on the mode.
    parser.add_argument("--output-len", type=int, default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--batch-sizes", default="1,4,16",
                        help="--mode throughput: concurrency to sweep")
    parser.add_argument("--max-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-sequences", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--kv-fraction", type=float, default=0.4)
    parser.add_argument("--num-requests", type=int, default=2000, help="--mode scheduler")
    # Near what this card sustains on the mix, which is where scheduling decisions show up.
    parser.add_argument("--rate", type=float, default=16.0,
                        help="--mode scheduler: arrivals/sec")
    parser.add_argument("--short-len", type=int, default=STRESS_SHORT_LEN)
    parser.add_argument("--long-len", type=int, default=STRESS_LONG_LEN)
    parser.add_argument("--long-every", type=int, default=STRESS_LONG_EVERY)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=0,
                        help="iterations between progress lines")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compare", default=None, choices=["hf"])
    parser.add_argument("--no-cache", action="store_true",
                        help="--mode single: measure the uncached quadratic loop")
    parser.add_argument("--use-cuda-kernels", action="store_true",
                        help="route ops through hand-written kernels where they exist")
    args = parser.parse_args()
    if args.output_len is None:
        args.output_len = STRESS_OUTPUT_LEN if args.mode == "scheduler" else (
            64 if args.mode == "fp8" else 128)

    modes = {
        "single": mode_single,
        "kernels": mode_kernels,
        "throughput": mode_throughput,
        "scheduler": mode_scheduler,
        "prefix-cache": mode_prefix_cache,
        "fp8": mode_fp8,
        "spec": mode_spec,
    }
    modes[args.mode](args)


if __name__ == "__main__":
    main()
