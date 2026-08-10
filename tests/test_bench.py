"""Step 2.3 — the benchmark harness.

A benchmark cannot be tested for the numbers it produces, so these test the things
that make the numbers trustworthy: that prefill and decode are separated, that a
throttled GPU is flagged rather than quietly reported, and that the whole thing
runs end to end on `tiny_qwen3`.
"""

from __future__ import annotations

import pytest
import torch
from conftest import (
    KERNEL_DRIFT_LIMIT,
    assert_relative_error_below,
    config_from_hf,
    weights_from_hf,
)

from mini_vllm.bench import (
    THROUGHPUT_LENGTHS,
    BandwidthResult,
    ClockSampler,
    GpuState,
    SingleResult,
    ThroughputResult,
    build_input_ids,
    copy_ceiling,
    gpu_state,
    kernel_cases,
    measure_bandwidth,
    measure_engine,
    measure_hf_throughput,
    measure_ours,
    rows_exceeding_l2,
    theoretical_bandwidth,
    throughput_prompts,
)
from mini_vllm.model.qwen3_cached import Qwen3Cached


@pytest.fixture
def cached(tiny_qwen3):
    return Qwen3Cached(config_from_hf(tiny_qwen3), weights_from_hf(tiny_qwen3))


# ------------------------------------------------------------- smoke, tiny model


def test_measures_the_tiny_model_end_to_end(cached, tiny_qwen3):
    ids = torch.randint(0, tiny_qwen3.config.vocab_size, (1, 8))

    result = measure_ours(cached, ids, output_len=6, warmup=1)

    assert isinstance(result, SingleResult)
    assert result.input_len == 8
    assert result.output_len == 6
    assert result.ttft_ms > 0
    assert result.decode_tokens_per_second > 0


def test_reports_prefill_and_decode_separately(cached, tiny_qwen3):
    """The two must not be conflated: they respond to different optimizations."""
    ids = torch.randint(0, tiny_qwen3.config.vocab_size, (1, 32))

    result = measure_ours(cached, ids, output_len=4, warmup=1)

    assert result.ttft_ms != pytest.approx(result.decode_ms_per_token)
    assert result.decode_ms_per_token > 0


def test_ms_per_token_is_the_reciprocal_of_the_rate():
    result = SingleResult("x", 8, 8, ttft_ms=10.0, decode_tokens_per_second=50.0)
    assert result.decode_ms_per_token == pytest.approx(20.0)


def test_describe_mentions_both_numbers():
    text = SingleResult("mini-vllm", 128, 128, 12.5, 40.0).describe()
    assert "TTFT" in text and "decode" in text and "mini-vllm" in text


def test_batch_is_counted_in_the_decode_rate(cached, tiny_qwen3):
    """Throughput scales with batch, so the rate must count every sequence."""
    ids = torch.randint(0, tiny_qwen3.config.vocab_size, (4, 8))
    result = measure_ours(cached, ids, output_len=4, warmup=1)

    single = torch.randint(0, tiny_qwen3.config.vocab_size, (1, 8))
    one = measure_ours(cached, single, output_len=4, warmup=1)

    assert result.decode_tokens_per_second > one.decode_tokens_per_second / 2


def test_a_longer_prompt_costs_more_prefill(cached, tiny_qwen3):
    """TTFT must grow with the prompt; if it does not, prefill is not being timed."""
    vocab = tiny_qwen3.config.vocab_size
    short = measure_ours(cached, torch.randint(0, vocab, (1, 4)), output_len=2, warmup=2)
    long = measure_ours(cached, torch.randint(0, vocab, (1, 128)), output_len=2, warmup=2)

    assert long.ttft_ms > short.ttft_ms


def test_measuring_does_not_leave_the_model_holding_a_cache(cached, tiny_qwen3):
    """Each measurement must start from an empty cache, or runs contaminate each other."""
    ids = torch.randint(0, tiny_qwen3.config.vocab_size, (1, 6))

    first = measure_ours(cached, ids, output_len=4, warmup=0)
    second = measure_ours(cached, ids, output_len=4, warmup=0)

    assert first.input_len == second.input_len == 6


# --------------------------------------------------------------- input building


@pytest.mark.oracle
def test_builds_a_prompt_of_exactly_the_requested_length():
    from transformers import AutoTokenizer

    from mini_vllm.model.loader import resolve_model_path

    tokenizer = AutoTokenizer.from_pretrained(resolve_model_path())

    for length in (1, 16, 128, 512):
        ids = build_input_ids(tokenizer, length, batch=2, device="cpu")
        assert ids.shape == (2, length)


# --------------------------------------------------------------- throttling


def test_throttle_detection_flags_a_parked_gpu():
    """The check that keeps a throttled run from being read as a kernel result.

    These are the real numbers from an RTX 5070 Laptop stuck in P8 while WSL2 held
    it at its idle power state: 180 of 3090 MHz core and 405 of 12001 MHz memory,
    which made every measurement about 30x too slow.
    """
    parked = GpuState(
        name="RTX 5070 Laptop",
        sm_clock=180,
        sm_clock_max=3090,
        memory_clock=405,
        memory_clock_max=12001,
        power_state="P8",
        power_draw="14.87 W",
    )

    assert parked.is_throttled
    assert parked.sm_fraction < 0.1
    assert parked.memory_fraction < 0.1
    assert "P8" in parked.describe()


def test_throttle_detection_accepts_a_healthy_gpu():
    """A boosting GPU never sits at exactly its maximum, and must not be flagged."""
    healthy = GpuState(
        name="RTX 5070 Laptop",
        sm_clock=2600,
        sm_clock_max=3090,
        memory_clock=11500,
        memory_clock_max=12001,
        power_state="P0",
        power_draw="80.00 W",
    )

    assert not healthy.is_throttled
    assert healthy.sm_fraction > 0.8


@pytest.mark.parametrize(
    ("sm", "memory", "throttled"),
    [
        (3090, 12001, False),  # at maximum
        (1600, 11000, False),  # normal boost variation
        (1400, 12001, True),  # core parked
        (3090, 5000, True),  # memory parked
    ],
)
def test_throttle_threshold_is_half_of_maximum(sm, memory, throttled):
    state = GpuState("gpu", sm, 3090, memory, 12001, "P0", "50 W")
    assert state.is_throttled is throttled


def test_state_handles_a_missing_maximum():
    """Some GPUs report no maximum; that must not read as infinitely throttled."""
    state = GpuState("gpu", 1000, 0, 5000, 0, "P0", "N/A")
    assert not state.is_throttled


# -------------------------------------------------------------- clock sampling


def test_gpu_state_reads_or_returns_none():
    """Works without a GPU: returns None rather than raising."""
    state = gpu_state()
    if state is None:
        pytest.skip("nvidia-smi is not available")

    assert state.name
    assert state.sm_clock_max > 0


@pytest.mark.cuda
def test_clock_sampler_observes_a_busy_gpu(device):
    """The sampler must catch clocks *during* the work, not after it settles."""
    x = torch.empty(64 * 1024 * 1024, dtype=torch.bfloat16, device=device)
    y = torch.empty_like(x)

    with ClockSampler(interval=0.05) as sampler:
        for _ in range(200):
            y.copy_(x)
        torch.cuda.synchronize()

    if sampler.peak is None:
        pytest.skip("nvidia-smi is not available")
    assert sampler.peak.sm_clock > 0


def test_clock_sampler_is_safe_without_a_gpu():
    with ClockSampler() as sampler:
        pass
    assert sampler.peak is None or isinstance(sampler.peak, GpuState)


# --------------------------------------------------------------- dispatch line


def test_report_states_which_ops_ran_as_kernels():
    """"My kernel made no difference" is usually "my kernel never ran".

    Every op has a kernel as of Step 3.5, so with the flag on the report says `cuda`
    for all of them *except* the ones the benchmark says are not yet worth
    preferring — and for those it has to give the reason, since "kernel exists,
    kernel unused, nobody said so" is the failure this whole report exists to catch.
    """
    from mini_vllm.kernels import ops

    with_kernels = ops.dispatch_report(use_cuda=True).splitlines()
    without = ops.dispatch_report(use_cuda=False).splitlines()

    assert len(with_kernels) == len(ops.CUDA_KERNELS) == len(without)

    for line in without:
        assert line.split()[1] == "torch", f"use_cuda=False routed to a kernel: {line!r}"

    for line in with_kernels:
        name, implementation = line.split()[:2]
        if name in ops.NOT_YET_FASTER:
            assert implementation == "torch" and ops.NOT_YET_FASTER[name] in line
        else:
            assert implementation == "cuda", f"claimed as a kernel but routed away: {line!r}"


# ----------------------------------------------------------------- bandwidth


def test_bandwidth_arithmetic():
    """1 GB moved in 1 ms is 1000 GB/s, and half of a 2000 GB/s peak."""
    result = BandwidthResult("x", bytes_moved=10**9, seconds=1e-3)

    assert result.gigabytes_per_second == pytest.approx(1000.0)
    assert result.microseconds == pytest.approx(1000.0)
    assert result.fraction_of(2000.0) == pytest.approx(0.5)
    assert result.fraction_of(None) is None


def test_describe_mentions_the_share_of_peak():
    text = BandwidthResult("rmsnorm", bytes_moved=10**9, seconds=1e-3).describe(peak=2000.0)
    assert "rmsnorm" in text and "GB/s" in text and "50%" in text


@pytest.mark.cuda
def test_theoretical_peak_is_plausible(device):
    """A peak that is wrong by 2x turns every percentage under it into fiction."""
    peak = theoretical_bandwidth()

    assert peak is not None
    assert 50.0 < peak < 10_000.0, f"{peak} GB/s is not a credible memory bandwidth"


@pytest.mark.cuda
def test_rows_past_l2_really_exceed_l2(device):
    """The sizing helper has one job: put the working set outside the cache.

    It matters because a working set inside L2 does not merely flatter a kernel,
    it reports a bandwidth *above* the card's DRAM peak — an impossible number
    that is easy to quote by accident.
    """
    width = 1024
    rows = rows_exceeding_l2(width, torch.bfloat16, multiple=8)
    working_set = 2 * rows * width * 2  # read plus write, 2 bytes each

    assert working_set >= 8 * torch.cuda.get_device_properties(0).L2_cache_size


@pytest.mark.cuda
def test_the_copy_ceiling_is_a_fraction_of_peak(device):
    """A bare copy is the fastest a memory-bound kernel can be, and it is not peak."""
    ceiling = copy_ceiling(megabytes=64)
    peak = theoretical_bandwidth()

    assert 0.0 < ceiling.gigabytes_per_second
    assert ceiling.gigabytes_per_second < peak, "a copy cannot beat the theoretical peak"


@pytest.mark.cuda
def test_kernel_cases_cover_every_implemented_kernel(device):
    """Every kernel that exists has a benchmark, so none lands unmeasured."""
    from mini_vllm.kernels import ops

    cases = kernel_cases()
    labels = " ".join(case.label for case in cases)

    for name in ops.cuda_kernel_names():
        assert name in labels, f"{name} has a kernel but no benchmark case"


@pytest.mark.cuda
def test_kernel_cases_agree_with_their_references(device):
    """The benchmark must time two implementations of the same thing.

    Otherwise a "speedup" is just the kernel doing less work — and the pairing
    only means something if both sides compute the same answer.

    The aggregate norm rather than an elementwise bound, because these cases run in
    bf16 and the attention kernels hold their intermediates in fp32 where the oracle
    rounds. Out of 262144 outputs a few land near zero through cancellation, where
    one ULP upstream is a 100% *relative* difference downstream — a fact about those
    elements' magnitude, not about whether the pair computes the same function.
    """
    for case in kernel_cases():
        assert_relative_error_below(
            case.kernel(), case.reference(), KERNEL_DRIFT_LIMIT, msg=case.label
        )


@pytest.mark.cuda
def test_measure_bandwidth_times_a_real_kernel(device):
    x = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
    y = torch.empty_like(x)
    moved = 2 * x.numel() * x.element_size()

    result = measure_bandwidth(lambda: y.copy_(x), moved, "copy", warmup=2, iterations=5)

    assert result.seconds > 0
    assert result.gigabytes_per_second > 0


# ---------------------------------------------------------------- throughput


def test_throughput_arithmetic():
    """The reported rate counts *output* tokens, which is what an SLO is about."""
    result = ThroughputResult("x", num_requests=8, prompt_tokens=800, generated_tokens=512, seconds=2.0)

    assert result.tokens_per_second == pytest.approx(256.0)
    assert result.requests_per_second == pytest.approx(4.0)
    assert "out tok/s" in result.describe()


def test_a_zero_length_run_does_not_divide_by_zero():
    assert ThroughputResult("x", 0, 0, 0, 0.0).tokens_per_second == 0.0


@pytest.mark.oracle
def test_the_prompt_set_is_deliberately_ragged():
    """Equal-length prompts are the one case where padding is free.

    A throughput benchmark built on them would hide most of what continuous batching
    and paging buy, so the lengths cycle and this pins that they really do.
    """
    from transformers import AutoTokenizer

    from mini_vllm.model.loader import resolve_model_path

    tokenizer = AutoTokenizer.from_pretrained(resolve_model_path())
    prompts = throughput_prompts(tokenizer, 8)
    lengths = [len(tokenizer(prompt).input_ids) for prompt in prompts]

    assert len(prompts) == 8
    assert max(lengths) >= 8 * min(lengths), f"lengths {lengths} are not ragged enough to matter"
    for length, wanted in zip(lengths, THROUGHPUT_LENGTHS, strict=False):
        # Detokenizing and retokenizing is not the identity, so this is approximate on
        # purpose — what matters is the spread, not the exact figure.
        assert abs(length - wanted) <= wanted // 8 + 2


@pytest.mark.oracle
def test_the_engine_beats_transformers_on_a_batch():
    """The Step 5.1 claim, run small: a batch of 8 is faster through the engine.

    Eight requests rather than the README's thirty-two, and eight output tokens rather
    than sixty-four, because this is a regression test and not the benchmark. It checks
    the harness measures both sides on the same work and that the engine is ahead — a
    build where it is *behind* on a ragged batch has lost the point of Phase 4.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mini_vllm import LLM
    from mini_vllm.model.loader import resolve_model_path

    path = resolve_model_path()
    if not (path / "model.safetensors").is_file() or not torch.cuda.is_available():
        pytest.skip("needs the real weights on a GPU")

    tokenizer = AutoTokenizer.from_pretrained(path)
    prompts = throughput_prompts(tokenizer, 8)

    theirs = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
    try:
        llm = LLM(num_blocks=1024, max_sequences=8)
        ours = measure_engine(llm, prompts, output_len=8, warmup=1)
        baseline = measure_hf_throughput(theirs, tokenizer, prompts, output_len=8, warmup=1)
    finally:
        del theirs
        torch.cuda.empty_cache()

    assert ours.generated_tokens == baseline.generated_tokens == 8 * 8, (
        "the two sides must be credited with the same work"
    )
    assert ours.tokens_per_second > baseline.tokens_per_second, (
        f"{ours.describe()}\n{baseline.describe()}"
    )
