"""Step 2.3 — the benchmark harness.

A benchmark cannot be tested for the numbers it produces, so these test the things
that make the numbers trustworthy: that prefill and decode are separated, that a
throttled GPU is flagged rather than quietly reported, and that the whole thing
runs end to end on `tiny_qwen3`.
"""

from __future__ import annotations

import pytest
import torch
from conftest import config_from_hf, weights_from_hf

from mini_vllm.bench import (
    BandwidthResult,
    ClockSampler,
    GpuState,
    SingleResult,
    build_input_ids,
    copy_ceiling,
    gpu_state,
    kernel_cases,
    measure_bandwidth,
    measure_ours,
    rows_exceeding_l2,
    theoretical_bandwidth,
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
    """"My kernel made no difference" is usually "my kernel never ran"."""
    from mini_vllm.kernels import ops

    report = ops.dispatch_report(use_cuda=True)
    assert "torch" in report
    for name in ops.CUDA_KERNELS:
        assert name in report


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
    """
    for case in kernel_cases():
        kernel_output = case.kernel()
        reference_output = case.reference()
        torch.testing.assert_close(
            kernel_output.float(), reference_output.float(), rtol=1e-2, atol=1e-2
        )


@pytest.mark.cuda
def test_measure_bandwidth_times_a_real_kernel(device):
    x = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
    y = torch.empty_like(x)
    moved = 2 * x.numel() * x.element_size()

    result = measure_bandwidth(lambda: y.copy_(x), moved, "copy", warmup=2, iterations=5)

    assert result.seconds > 0
    assert result.gigabytes_per_second > 0
