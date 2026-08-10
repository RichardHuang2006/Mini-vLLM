"""Step 5.2 — the scheduler under an adversarial arrival schedule.

Two claims are on trial here, and they need different instruments.

**"Chunked prefill lowers tail decode latency"** is a claim about time, and a test that
asserts on milliseconds is a test that fails on a busy laptop. So the tests measure the
thing the milliseconds are made of: how many *iterations* a decoding sequence goes
without being scheduled. That number is deterministic, it is what the wall-clock tail is
a monotone function of, and it is the head-of-line stall stated exactly —
[§7.2](../DESIGN.md#72-chunked-prefill) says a decode should never wait for a whole
prompt, and "stall of 1 iteration" is that sentence in a form pytest can check.
`bench --mode scheduler` is where the same claim is measured in milliseconds.

**"A multi-thousand-request run leaks nothing"** is a claim about bookkeeping, and it
needs volume rather than realism: thousands of requests through a pool small enough that
preemption is routine, ending with every block back in the pool. The tiny model makes
that a few seconds instead of an hour, and nothing about block accounting depends on
how good the weights are.

The invariant underneath both: **a scheduling policy may change timing, never output.**
Every test that runs two policies compares their tokens as well as their latency.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from conftest import config_from_hf, weights_from_hf

from mini_vllm.bench import (
    LatencyStats,
    StressRequest,
    percentile,
    poisson_arrivals,
    run_stress,
    stress_requests,
)
from mini_vllm.block.block_manager import BlockManager
from mini_vllm.model.qwen3_paged import Qwen3Paged
from mini_vllm.sampler import SamplingParams
from mini_vllm.serve.runner import PagedModelRunner
from mini_vllm.serve.scheduler import Scheduler, SchedulerConfig
from mini_vllm.serve.sequence import Sequence

GREEDY = SamplingParams(temperature=0.0)
BLOCK_SIZE = 8

# The two policies, as the flags that distinguish them. `prefill_priority` without
# chunking is what vLLM served with before chunked prefill landed, and it is the
# baseline every latency comparison below is against.
CHUNKED = {"enable_chunked_prefill": True, "prefill_priority": False}
PREFILL_FIRST = {"enable_chunked_prefill": False, "prefill_priority": True}


# ------------------------------------------------------------- the arrival schedule


def test_arrivals_are_monotone_and_average_out():
    rate = 25.0
    times = poisson_arrivals(4000, rate, seed=3)

    assert len(times) == 4000
    assert times == sorted(times)
    assert times[0] > 0.0

    # The mean gap of a Poisson process is 1/rate. Four thousand samples put the
    # sample mean within a couple of percent, which is loose enough not to flake and
    # tight enough to catch a rate that means something other than requests per second.
    mean_gap = times[-1] / len(times)
    assert mean_gap == pytest.approx(1.0 / rate, rel=0.1)


def test_the_gaps_are_bursty_rather_than_even():
    """The reason for Poisson: a third of the gaps are much shorter than the mean.

    An evenly spaced schedule is the easy case for a scheduler, because it never has
    to hold more than one arrival at a time. If this ever became a uniform generator
    the tests above would still pass and the benchmark would stop being adversarial.
    """
    times = poisson_arrivals(2000, 10.0, seed=1)
    gaps = [later - earlier for earlier, later in zip(times, times[1:], strict=False)]
    mean = sum(gaps) / len(gaps)

    short = sum(1 for gap in gaps if gap < mean / 2)
    assert short / len(gaps) > 0.3
    assert max(gaps) > 4 * mean  # and a long quiet stretch somewhere


def test_a_zero_rate_means_everything_at_once():
    assert poisson_arrivals(5, 0.0) == [0.0] * 5


def test_a_negative_rate_is_refused():
    with pytest.raises(ValueError, match="rate must be >= 0"):
        poisson_arrivals(5, -1.0)


def test_the_mix_is_mostly_short_with_a_long_tail():
    requests = stress_requests(100, rate=10.0, vocab_size=1000, long_every=20)

    lengths = [request.prompt_len for request in requests]
    assert lengths.count(2048) == 5
    assert lengths.count(32) == 95
    assert [request.arrival for request in requests] == sorted(r.arrival for r in requests)


def test_prompt_ids_are_inside_the_vocabulary():
    """Not a style point: an id past the embedding table is an illegal memory read."""
    for request in stress_requests(20, rate=0.0, vocab_size=64, long_every=0):
        assert all(0 < token < 64 for token in request.prompt_token_ids)


def test_the_same_seed_is_the_same_workload():
    first = stress_requests(30, rate=10.0, vocab_size=100, seed=7)
    again = stress_requests(30, rate=10.0, vocab_size=100, seed=7)
    other = stress_requests(30, rate=10.0, vocab_size=100, seed=8)

    assert first == again
    assert first != other


# ------------------------------------------------------------------- the statistics


def test_the_percentile_is_a_sample_somebody_waited():
    values = [float(index) for index in range(1, 101)]

    assert percentile(values, 0.50) == 50.0
    assert percentile(values, 0.99) == 99.0
    assert percentile(values, 1.0) == 100.0
    # Nearest-rank, so every answer is one of the inputs — no interpolated latency
    # that no request experienced.
    assert percentile([1.0, 2.0], 0.75) in (1.0, 2.0)


def test_an_empty_run_does_not_divide_by_zero():
    empty = LatencyStats(label="nothing", num_requests=0, completed=0, generated_tokens=0, seconds=0.0)

    assert empty.tokens_per_second == 0.0
    assert empty.p99_inter_token_ms == 0.0
    assert empty.p99_ttft_ms == 0.0
    assert empty.max_inter_token_ms == 0.0
    assert "nothing" in empty.describe()


def test_the_rates_are_what_they_say():
    stats = LatencyStats(
        label="run",
        num_requests=4,
        completed=4,
        generated_tokens=200,
        seconds=2.0,
        ttft=[0.1, 0.2, 0.3, 0.4],
        inter_token=[0.01] * 99 + [1.0],
    )

    assert stats.tokens_per_second == 100.0
    assert stats.p50_inter_token_ms == pytest.approx(10.0)
    assert stats.p99_inter_token_ms == pytest.approx(10.0)  # 99 of 100 are the fast ones
    assert stats.max_inter_token_ms == pytest.approx(1000.0)
    assert stats.p99_ttft_ms == pytest.approx(400.0)


# ------------------------------------------------------------------- a tiny engine


@dataclass(frozen=True)
class Iteration:
    """What one iteration did, which is all the latency claims below are built from."""

    emitted: list[int]
    tokens: int
    scheduled: list[int]


class TinyEngine:
    """Enough of `LLM` for the stress driver, over a two-layer random model.

    `run_stress` only needs five things — `add_request`, `step`, `scheduler`, `manager`
    and `stats.preemptions` — and providing them here means the driver that produces the
    published numbers is the driver under test, rather than a paraphrase of it. What is
    left out is everything to do with text: no tokenizer, no detokenization, no
    checkpoint.
    """

    def __init__(self, model, manager: BlockManager, **config) -> None:
        from mini_vllm.serve.engine import EngineStats

        self.manager = manager
        self.scheduler = Scheduler(SchedulerConfig(**config), manager=manager)
        self.runner = PagedModelRunner(model, manager)
        self.stats = EngineStats()
        # Per iteration: who got a token, how many tokens the iteration computed, and
        # who was in it. That is enough to reconstruct every latency claim below without
        # a clock.
        self.history: list[Iteration] = []

    def add_request(self, prompt, max_tokens: int = 4, ignore_eos: bool = True) -> Sequence:
        sequence = Sequence(
            prompt_token_ids=list(prompt), max_tokens=max_tokens, sampling_params=GREEDY
        )
        self.scheduler.add(sequence)
        return sequence

    def step(self) -> list[tuple[Sequence, int]]:
        output = self.scheduler.schedule()
        logits = self.runner.execute(output)
        tokens = self.runner.sample_tokens(output, logits)
        finished = self.scheduler.commit(output, tokens)

        emitted = [
            (sequence, sequence.output_token_ids[-1])
            for sequence, _ in output.scheduled
            if not sequence.is_prefill() and sequence.output_token_ids
        ]
        for sequence in finished:
            self.runner.free(sequence)

        self.stats.iterations += 1
        self.stats.preemptions += len(output.preempted)
        self.history.append(
            Iteration(
                emitted=[sequence.seq_id for sequence, _ in emitted],
                tokens=output.total_tokens,
                scheduled=[sequence.seq_id for sequence, _ in output.scheduled],
            )
        )
        return emitted

    def reconfigure(self, **changes) -> None:
        from dataclasses import replace

        self.scheduler = Scheduler(replace(self.scheduler.config, **changes), manager=self.manager)
        self.history = []


@pytest.fixture
def engine_parts(tiny_qwen3, device):
    """Weights and config for the tiny model, in fp32 on the GPU."""
    theirs = tiny_qwen3.to(device=device, dtype=torch.float32)
    return weights_from_hf(theirs), config_from_hf(theirs)


def build(engine_parts, num_blocks: int = 512, **config) -> TinyEngine:
    weights, config_object = engine_parts
    manager = BlockManager(
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        num_layers=config_object.num_hidden_layers,
        num_kv_heads=config_object.num_key_value_heads,
        head_dim=config_object.head_dim,
        dtype=config_object.dtype,
        device=weights["embedding"].device,
    )
    model = Qwen3Paged(config_object, weights, manager, use_cuda=True)
    return TinyEngine(model, manager, **config)


# ------------------------------------------------------- the head-of-line stall

# The mix, scaled to the tiny model but the same shape as the benchmark's: short
# requests being served while long prompts queue up behind them. `LONG_LEN` is four
# times the budget, so with chunking off it cannot share an iteration with anything, and
# it stays inside the tiny model's 256-position RoPE table with its output on top.
SHORT_LEN, LONG_LEN, BUDGET, CHUNK = 4, 128, 32, 8

# How many iterations the short requests run alone before the long prompts arrive. The
# interesting case is a prompt landing in the middle of a run, not at the start of one.
SETTLE = 3


@dataclass
class Mix:
    """One policy's run over the mix, reduced to the numbers a latency claim needs."""

    iterations: int
    stall: int
    wait: int
    longest_iteration: int
    prompt_start: list[int]
    outputs: list[tuple[int, ...]]


def run_mix(engine_parts, num_short: int = 6, num_long: int = 4, **policy) -> Mix:
    """Serve `num_short` short requests, drop `num_long` long prompts in, run it out.

    The measurements, and why each is the one that matters:

    * `stall` — the most iterations a short request went without a token once it had its
      first. 1 means it got one in every iteration.
    * `wait` — the most *tokens the engine computed* between two of its tokens. This is
      the honest proxy for the wall-clock tail, because an iteration's cost is roughly
      linear in its token count, and it is the quantity `bench --mode scheduler` sees
      through a clock. Counting iterations instead would call sixteen small stalls worse
      than five huge ones.
    * `longest_iteration` — the largest iteration in the run. Chunked prefill's promise
      is that this stays inside the configured budget.
    * `prompt_start` — iterations from arrival to a long prompt's *first* scheduled
      chunk, which is the head-of-line stall seen from the prompt's side.
    """
    engine = build(engine_parts, max_batched_tokens=BUDGET, chunk_size=CHUNK, **policy)
    shorts = [engine.add_request([7] * SHORT_LEN, max_tokens=8) for _ in range(num_short)]
    for _ in range(SETTLE):
        engine.step()
    longs = [engine.add_request([3] * LONG_LEN, max_tokens=2) for _ in range(num_long)]

    while engine.scheduler.num_unfinished:
        assert engine.stats.iterations < 2000, "the engine stopped making progress"
        engine.step()
    engine.manager.check_no_leaks()

    short_ids = {sequence.seq_id for sequence in shorts}
    long_ids = [sequence.seq_id for sequence in longs]

    seen: dict[int, tuple[int, int]] = {}
    started: dict[int, int] = {}
    stall = wait = computed = 0
    for index, iteration in enumerate(engine.history):
        computed += iteration.tokens
        for seq_id in iteration.scheduled:
            started.setdefault(seq_id, index)
        for seq_id in iteration.emitted:
            if seq_id in short_ids:
                if seq_id in seen:
                    stall = max(stall, index - seen[seq_id][0])
                    wait = max(wait, computed - seen[seq_id][1])
                seen[seq_id] = (index, computed)

    return Mix(
        iterations=len(engine.history),
        stall=stall,
        wait=wait,
        longest_iteration=max(iteration.tokens for iteration in engine.history),
        prompt_start=[started[seq_id] - SETTLE for seq_id in long_ids],
        outputs=[tuple(sequence.output_token_ids) for sequence in shorts + longs],
    )


@pytest.mark.cuda
def test_the_default_policy_gives_every_decode_a_token_every_iteration(engine_parts):
    """The piggyback claim in its strong form, with four long prompts arriving mid-run.

    A decode is one token, so six of them are six of a 32-token budget and no prefill
    chunk can crowd them out. The work a decode waits through is then one iteration's
    worth, and one iteration cannot exceed the budget — which is the whole of the
    latency guarantee: it is set by configuration, not by whatever prompt showed up.
    """
    mix = run_mix(engine_parts, **CHUNKED)

    assert mix.stall == 1
    assert mix.wait == BUDGET
    assert mix.longest_iteration <= BUDGET
    # And the prompts do not pay for it either: all four start immediately, sharing
    # iterations with the decodes and with each other.
    assert mix.prompt_start == [0, 0, 0, 0]


@pytest.mark.cuda
def test_prefill_priority_makes_a_decode_wait_for_the_prompts(engine_parts):
    """The baseline, on the identical arrival pattern. This is the number that moves.

    Both prefill-first variants are measured, because the difference between them is
    instructive: chunking splits the stall into many small iterations instead of a few
    huge ones, and the *total work* a decode waits through is the same either way. What
    removes the stall is running the decodes first, and what makes running them first
    affordable is that a chunk fits in whatever they leave.
    """
    chunked = run_mix(engine_parts, **CHUNKED)
    whole_prompts = run_mix(engine_parts, **PREFILL_FIRST)
    chunked_prompts = run_mix(engine_parts, enable_chunked_prefill=True, prefill_priority=True)

    assert whole_prompts.wait > 8 * chunked.wait
    assert chunked_prompts.wait > 8 * chunked.wait
    assert whole_prompts.stall > 1 and chunked_prompts.stall > 1

    # Few enormous stalls against many small ones, adding up to the same wait.
    assert chunked_prompts.stall > whole_prompts.stall
    assert chunked_prompts.wait == pytest.approx(whole_prompts.wait, rel=0.25)


@pytest.mark.cuda
def test_chunking_is_what_keeps_an_iteration_inside_its_budget(engine_parts):
    """[§7.2](../DESIGN.md#72-chunked-prefill), stated as the property it actually is.

    With chunking off, a prompt longer than the budget still has to run, so the engine
    runs it alone and overruns — a 128-token iteration under a 32-token budget. The
    iteration is then sized by the request rather than by the configuration, and no
    latency figure survives that.

    Turning chunking off costs the prompts too, and the second half of this test is the
    less obvious half: without it a long prompt cannot share an iteration with anything,
    so four of them serialize, each waiting for the one before. Chunked, all four start
    at once.
    """
    chunked = run_mix(engine_parts, **CHUNKED)
    whole = run_mix(engine_parts, enable_chunked_prefill=False, prefill_priority=False)

    assert chunked.longest_iteration <= BUDGET
    assert whole.longest_iteration == LONG_LEN > BUDGET

    assert chunked.prompt_start == [0, 0, 0, 0]
    assert whole.prompt_start == sorted(whole.prompt_start)
    assert whole.prompt_start[-1] > whole.prompt_start[0] > 0


@pytest.mark.cuda
def test_the_policy_does_not_change_the_answer(engine_parts):
    """Four policies, one mix, identical tokens. The invariant Phase 4 rests on.

    Compared positionally rather than by sequence id: ids come from a global counter, so
    the second run of the same workload gets different ones.
    """
    reference = None
    for policy in (
        CHUNKED,
        PREFILL_FIRST,
        {"enable_chunked_prefill": True, "prefill_priority": True},
        {"enable_chunked_prefill": False, "prefill_priority": False},
    ):
        outputs = run_mix(engine_parts, num_short=4, num_long=2, **policy).outputs
        if reference is None:
            reference = outputs
        else:
            assert outputs == reference, f"{policy} changed the output"


# ---------------------------------------------------------------- volume and leaks


@pytest.mark.cuda
def test_the_driver_times_every_token(engine_parts):
    """`run_stress` itself: one TTFT per request, one gap per token after the first."""
    engine = build(engine_parts, max_batched_tokens=BUDGET, chunk_size=CHUNK, **CHUNKED)
    output_len = 4
    requests = [
        StressRequest(arrival=index * 0.001, prompt_token_ids=(5,) * SHORT_LEN, output_len=output_len)
        for index in range(24)
    ]

    stats = run_stress(engine, requests, label="tiny")

    assert stats.completed == len(requests)
    assert stats.generated_tokens == len(requests) * output_len
    assert len(stats.ttft) == len(requests)
    assert len(stats.inter_token) == len(requests) * (output_len - 1)
    assert stats.seconds > 0
    assert all(value >= 0 for value in stats.ttft + stats.inter_token)
    assert len(stats.iteration_seconds) == stats.iterations
    assert stats.max_iteration_ms > 0
    assert "tiny" in stats.describe()


@pytest.mark.cuda
def test_a_request_that_arrives_late_is_waited_for(engine_parts):
    """An idle engine sleeps until the next arrival instead of spinning on empty queues.

    The gap is long enough that a driver which returned early — or one which called
    `schedule()` with nothing to schedule — would show up as a missing request rather
    than as a slow test.
    """
    engine = build(engine_parts, max_batched_tokens=BUDGET, chunk_size=CHUNK, **CHUNKED)
    requests = [
        StressRequest(arrival=0.0, prompt_token_ids=(5,) * SHORT_LEN, output_len=2),
        StressRequest(arrival=0.25, prompt_token_ids=(6,) * SHORT_LEN, output_len=2),
    ]

    stats = run_stress(engine, requests, label="sparse")

    assert stats.completed == 2
    assert stats.seconds >= 0.25
    # The late one waited for the wall clock, not for the engine.
    assert max(stats.ttft) < 0.05


@pytest.mark.cuda
@pytest.mark.slow
def test_three_thousand_requests_leak_no_blocks(engine_parts):
    """The volume test the plan asks for: no leaks, no OOM, everything completes.

    All at once (`rate=0`) against a pool of 96 pages, which is a few hundred tokens —
    small enough that admission is refused constantly and preemption is routine, and
    the accounting has thousands of chances to lose a page. `run_stress` ends with
    `check_no_leaks`, so a single leaked block anywhere in the run fails this.
    """
    engine = build(engine_parts, num_blocks=96, max_batched_tokens=BUDGET, chunk_size=CHUNK, **CHUNKED)
    requests = stress_requests(
        num_requests=3000,
        rate=0.0,
        vocab_size=engine_parts[1].vocab_size,
        short_len=SHORT_LEN,
        long_len=64,
        long_every=50,
        output_len=3,
        seed=11,
    )

    stats = run_stress(engine, requests, label="volume")

    assert stats.completed == 3000
    assert stats.generated_tokens == 9000
    engine.manager.check_no_leaks()
    assert engine.manager.num_free_blocks == engine.manager.num_blocks


@pytest.mark.cuda
@pytest.mark.slow
def test_a_pool_that_forces_preemption_still_finishes_everything(engine_parts):
    """Under real pressure the engine preempts, recomputes, and still lands.

    Preemption is the path that only runs when memory is tight, which is where nobody is
    watching; this pins it at volume. A recomputed sequence re-prefills its prompt *and*
    its output so far, so the guarantee being checked is progress — that the loop cannot
    livelock trading pages between two sequences that both need them.
    """
    engine = build(engine_parts, num_blocks=40, max_batched_tokens=BUDGET, chunk_size=CHUNK, **CHUNKED)
    requests = stress_requests(
        num_requests=400,
        rate=0.0,
        vocab_size=engine_parts[1].vocab_size,
        short_len=8,
        long_len=96,
        long_every=10,
        output_len=6,
        seed=5,
    )

    stats = run_stress(engine, requests, label="pressure")

    assert stats.completed == 400
    assert stats.preemptions > 0, "the pool was not small enough to prove anything"
    engine.manager.check_no_leaks()
