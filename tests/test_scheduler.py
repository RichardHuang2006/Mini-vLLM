"""Step 4.3 — continuous batching.

Two kinds of test, and the split matters:

* **Policy**, with no model and no GPU — admission under a budget, the iteration
  boundary, preemption. These are the interesting cases and they run in
  milliseconds, which is only possible because `Scheduler` holds no tensors.
* **Identity**, with a real (tiny) model — the invariant that a scheduling decision
  changes timing and never output. A sequence run beside fifteen others, or
  preempted and recomputed, must produce the tokens it produces alone.

The second kind is what makes the first kind worth anything: a scheduler that
passes every policy test and perturbs one token is broken.
"""

from __future__ import annotations

import pytest
import torch
from conftest import config_from_hf, weights_from_hf

from mini_vllm.model.qwen3_cached import Qwen3Cached
from mini_vllm.sampler import SamplingParams
from mini_vllm.serve.scheduler import (
    DenseModelRunner,
    Scheduler,
    SchedulerConfig,
    SchedulerOutput,
)
from mini_vllm.serve.sequence import Sequence, SequenceStatus


# Every identity test is greedy on purpose. `SamplingParams()` defaults to
# temperature 1.0, and comparing two random draws would fail for reasons that have
# nothing to do with scheduling — the claim being tested is that the *logits* a
# sequence sees do not depend on what else is in the batch, and greedy decoding is
# how that becomes an exact assertion rather than a distributional one.
GREEDY = SamplingParams(temperature=0.0)

# Chunking off, for the tests that are about admission under a tight budget rather
# than about splitting. Spelled as kwargs so each use reads as a deviation from the
# default the engine actually ships with.
WHOLE = {"enable_chunked_prefill": False}


def make(prompt_len: int, max_tokens: int = 4, **kwargs) -> Sequence:
    kwargs.setdefault("sampling_params", GREEDY)
    return Sequence(
        prompt_token_ids=list(range(1, prompt_len + 1)), max_tokens=max_tokens, **kwargs
    )


# --------------------------------------------------------------------- the policy


def test_a_new_scheduler_has_no_work():
    scheduler = Scheduler()

    assert not scheduler.has_work
    assert scheduler.schedule().is_empty


def test_admission_is_first_come_first_served():
    scheduler = Scheduler(SchedulerConfig(max_sequences=2))
    first, second, third = make(4), make(4), make(4)
    scheduler.add_all([first, second, third])

    output = scheduler.schedule()

    assert output.sequences == [first, second]
    assert third.status is SequenceStatus.WAITING


def test_admission_stops_at_the_sequence_limit():
    scheduler = Scheduler(SchedulerConfig(max_sequences=3))
    scheduler.add_all([make(2) for _ in range(10)])

    assert len(scheduler.schedule().scheduled) == 3
    assert len(scheduler.running) == 3


def test_admission_stops_at_the_token_budget():
    """Two 300-token prompts do not fit in a 512-token iteration; one does.

    Whole prompts, because chunking is off here: admitting the second for its
    leftover 212 tokens would be a prefill split, and a split needs the position
    and mask handling that `test_chunked_prefill.py` covers.
    """
    scheduler = Scheduler(SchedulerConfig(max_batched_tokens=512, max_sequences=8, **WHOLE))
    scheduler.add_all([make(300), make(300)])

    output = scheduler.schedule()

    assert output.total_tokens == 300
    assert len(scheduler.waiting) == 1


def test_a_prompt_larger_than_the_budget_runs_alone_and_overruns_it():
    """Refusing it would deadlock the queue, so the budget yields instead.

    This is the head-of-line stall in its purest form: a 2000-token prompt takes a
    whole iteration to itself and every decode-phase request waits for it. It is
    reachable only with chunking off, and it is kept reachable because it is the
    behaviour Step 4.4 exists to remove — `test_chunked_prefill.py` runs the same
    prompt with chunking on and gets four bounded iterations instead.
    """
    scheduler = Scheduler(SchedulerConfig(max_batched_tokens=512, **WHOLE))
    huge = make(2000)
    scheduler.add(huge)

    output = scheduler.schedule()

    assert output.total_tokens == 2000, "the only runnable sequence must run"
    assert output.sequences == [huge]


def test_running_sequences_are_scheduled_before_new_arrivals():
    """An admitted request has memory committed and a caller watching it."""
    scheduler = Scheduler(SchedulerConfig(max_batched_tokens=10, max_sequences=8))
    established = make(4)
    scheduler.add(established)
    scheduler.commit(scheduler.schedule(), [42])

    scheduler.add(make(20))
    output = scheduler.schedule()

    assert output.sequences[0] is established


def test_a_decode_step_costs_one_token():
    scheduler = Scheduler()
    sequence = make(7)
    scheduler.add(sequence)
    scheduler.commit(scheduler.schedule(), [42])

    output = scheduler.schedule()

    assert output.scheduled == [(sequence, 1)]


# --------------------------------------------------------------- the iteration edge


def test_a_finished_sequence_is_replaced_in_the_same_iteration():
    """The point of the whole step: no drain between one request and the next.

    With `max_sequences=1`, sequence A finishes on some iteration and B has to be
    running on the *next* one — not after an idle iteration, and not after the queue
    is otherwise empty. Asserting on the batch composition at that boundary is the
    only way to tell continuous batching from static batching, since both produce
    correct output.
    """
    scheduler = Scheduler(SchedulerConfig(max_sequences=1))
    a, b = make(3, max_tokens=1), make(3, max_tokens=1)
    scheduler.add_all([a, b])

    prefill = scheduler.schedule()
    assert prefill.sequences == [a], "B must wait: only one slot"

    finished = scheduler.commit(prefill, [99])
    assert finished == [a], "A emitted its one token and is done"

    next_iteration = scheduler.schedule()
    assert next_iteration.sequences == [b], "B runs immediately, with no idle iteration"


def test_finishing_is_reported_from_the_iteration_it_happened_in():
    scheduler = Scheduler()
    sequence = make(3, max_tokens=2)
    scheduler.add(sequence)

    assert scheduler.commit(scheduler.schedule(), [7]) == []
    assert scheduler.commit(scheduler.schedule(), [8]) == [sequence]

    assert sequence.status is SequenceStatus.FINISHED
    assert sequence.output_token_ids == [7, 8]
    assert not scheduler.has_work


def test_a_mid_prefill_chunk_takes_no_token():
    """Sampling from a position in the middle of a prompt would invent a token.

    The next chunk of the prompt then contradicts it, so the sequence would carry a
    token its caller never asked for and the model never predicted from a complete
    context. `commit` throws that row away, and this pins it.
    """
    scheduler = Scheduler(SchedulerConfig(enable_chunked_prefill=True, chunk_size=4))
    sequence = make(10)
    scheduler.add(sequence)

    scheduler.commit(scheduler.schedule(), [777])

    assert sequence.output_token_ids == []
    assert sequence.num_computed_tokens == 4
    assert sequence.is_prefill()


def test_the_last_prefill_chunk_takes_its_token():
    scheduler = Scheduler(SchedulerConfig(enable_chunked_prefill=True, chunk_size=4))
    sequence = make(8)
    scheduler.add(sequence)

    scheduler.commit(scheduler.schedule(), [1])
    scheduler.commit(scheduler.schedule(), [2])

    assert sequence.output_token_ids == [2]
    assert not sequence.is_prefill()


def test_commit_checks_the_token_count():
    scheduler = Scheduler()
    scheduler.add_all([make(2), make(2)])
    output = scheduler.schedule()

    with pytest.raises(ValueError, match="for 2 scheduled sequences"):
        scheduler.commit(output, [1])


def test_a_running_sequence_with_nothing_to_do_is_an_error():
    """The alternative is an engine that spins with work outstanding and no output.

    It happens when `commit` is called without the token sampled for a sequence that
    just finished its prompt: nothing to feed it, nothing to finish it. Saying so
    here costs one comparison per sequence per iteration and turns a hang into a
    message that names the caller's mistake.
    """
    scheduler = Scheduler()
    scheduler.add(make(4))
    scheduler.commit(scheduler.schedule())  # no token: the bug being caught

    with pytest.raises(ValueError, match="nothing to compute"):
        scheduler.schedule()


def test_a_sequence_cannot_be_queued_twice():
    scheduler = Scheduler()
    sequence = make(2)
    scheduler.add(sequence)
    scheduler.commit(scheduler.schedule())

    with pytest.raises(ValueError, match="already running"):
        scheduler.add(sequence)


# -------------------------------------------------------------------- preemption


def test_preemption_returns_a_sequence_to_the_front_of_the_queue():
    scheduler = Scheduler()
    sequence = make(4)
    scheduler.add(sequence)
    scheduler.commit(scheduler.schedule(), [5])

    scheduler.preempt(sequence)

    assert sequence.status is SequenceStatus.PREEMPTED
    assert scheduler.waiting[0] is sequence
    assert sequence not in scheduler.running


def test_a_preempted_sequence_re_prefills_over_its_output():
    scheduler = Scheduler()
    sequence = make(4)
    scheduler.add(sequence)
    scheduler.commit(scheduler.schedule(), [5])
    scheduler.preempt(sequence)

    output = scheduler.schedule()

    assert output.scheduled == [(sequence, 5)], "4 prompt + 1 emitted token, recomputed"


def test_preempting_something_not_running_is_refused():
    scheduler = Scheduler()
    sequence = make(4)
    scheduler.add(sequence)

    with pytest.raises(ValueError, match="is not running"):
        scheduler.preempt(sequence)


# ------------------------------------------------------------------ the batch seam


def test_the_output_builds_a_ragged_batch():
    scheduler = Scheduler()
    long_prompt, short_prompt = make(6), make(2)
    scheduler.add_all([long_prompt, short_prompt])

    batch = scheduler.schedule().batch()

    assert batch.cu_seqlens_q.tolist() == [0, 6, 8]
    assert batch.seq_lens.tolist() == [6, 2]


def test_config_rejects_nonsense():
    for kwargs in ({"max_batched_tokens": 0}, {"max_sequences": 0}, {"chunk_size": 0}):
        with pytest.raises(ValueError, match="must be >= 1"):
            SchedulerConfig(**kwargs)


def test_empty_output_reports_itself():
    assert SchedulerOutput().is_empty
    assert SchedulerOutput().total_tokens == 0


# ---------------------------------------------------------------------- identity


@pytest.fixture
def tiny_model(tiny_qwen3, device):
    theirs = tiny_qwen3.to(device=device, dtype=torch.float32)
    return Qwen3Cached(config_from_hf(theirs), weights_from_hf(theirs), use_cuda=False)


def run_to_completion(scheduler: Scheduler, runner: DenseModelRunner) -> dict[int, list[int]]:
    """Drive the engine loop until every request is done. This is Step 4.9 in miniature."""
    while scheduler.has_work:
        output = scheduler.schedule()
        assert not output.is_empty, "the scheduler stalled with work outstanding"

        logits = runner.execute(output)
        tokens = runner.sample_tokens(output, logits)
        for finished in scheduler.commit(output, tokens):
            runner.free(finished)

    return {sequence.seq_id: sequence.output_token_ids for sequence in scheduler.finished}


def alone(model, prompt: list[int], max_tokens: int) -> list[int]:
    """The reference: one request, one scheduler, nothing else in flight."""
    scheduler = Scheduler()
    scheduler.add(
        Sequence(prompt_token_ids=prompt, max_tokens=max_tokens, sampling_params=GREEDY)
    )
    return next(iter(run_to_completion(scheduler, DenseModelRunner(model)).values()))


@pytest.mark.cuda
def test_one_sequence_matches_the_generate_loop(tiny_model, device):
    """First, agree with Phase 1's generator, so the harness itself is trusted."""
    from mini_vllm.generate import generate_ids_cached

    prompt = [3, 9, 4, 1, 7]
    expected = generate_ids_cached(
        tiny_model, torch.tensor([prompt], device=device), max_tokens=8
    )

    got = alone(tiny_model, prompt, max_tokens=8)

    assert got == expected[0, len(prompt) :].tolist()


@pytest.mark.cuda
def test_a_batched_run_is_token_identical_to_running_each_alone(tiny_model):
    """The invariant. Six requests of different lengths, interleaved.

    Their prefills and decodes end up in whatever iterations the budget allows, so
    every sequence is stepped alongside a different mix at every point in its life.
    None of that may reach the tokens.
    """
    prompts = [[1, 2, 3], [5], [7, 7, 7, 7, 7, 7, 7], [2, 4], [9, 8, 7, 6], [1]]
    expected = {index: alone(tiny_model, prompt, 6) for index, prompt in enumerate(prompts)}

    scheduler = Scheduler(SchedulerConfig(max_batched_tokens=16, max_sequences=3))
    sequences = [Sequence(prompt_token_ids=prompt, max_tokens=6, sampling_params=GREEDY)
        for prompt in prompts]
    scheduler.add_all(sequences)

    got = run_to_completion(scheduler, DenseModelRunner(tiny_model))

    for index, sequence in enumerate(sequences):
        assert got[sequence.seq_id] == expected[index], f"prompt {index} changed under batching"


@pytest.mark.cuda
def test_a_preempted_sequence_produces_the_same_tokens(tiny_model):
    """Recomputation must be indistinguishable from never having been evicted."""
    prompt = [4, 2, 7, 1]
    expected = alone(tiny_model, prompt, max_tokens=6)

    scheduler = Scheduler()
    runner = DenseModelRunner(tiny_model)
    sequence = Sequence(prompt_token_ids=prompt, max_tokens=6, sampling_params=GREEDY)
    scheduler.add(sequence)

    # Run two iterations, evict, then let it finish.
    for _ in range(2):
        output = scheduler.schedule()
        scheduler.commit(output, runner.sample_tokens(output, runner.execute(output)))
    scheduler.preempt(sequence)

    got = run_to_completion(scheduler, runner)

    assert got[sequence.seq_id] == expected


@pytest.mark.cuda
def test_per_row_sampling_parameters_are_honoured(tiny_model):
    """A greedy request and a sampled one in the same batch, in one sampler call."""
    torch.manual_seed(0)
    scheduler = Scheduler()
    greedy = Sequence(
        prompt_token_ids=[1, 2, 3], max_tokens=4, sampling_params=SamplingParams(temperature=0.0)
    )
    hot = Sequence(
        prompt_token_ids=[1, 2, 3], max_tokens=4, sampling_params=SamplingParams(temperature=2.0)
    )
    scheduler.add_all([greedy, hot])

    got = run_to_completion(scheduler, DenseModelRunner(tiny_model))

    assert got[greedy.seq_id] == alone(tiny_model, [1, 2, 3], 4)
    assert len(got[hot.seq_id]) == 4


@pytest.mark.cuda
def test_the_dense_cache_cannot_actually_batch(tiny_model):
    """The wall, asserted rather than described.

    Every scheduled sequence gets its own cache and therefore its own forward pass,
    so a batch of eight costs eight launches. Continuous batching has bought
    fairness and nothing else — the GPU is doing exactly the Phase 2 work it did
    before, and this is the measurement that motivates paging.
    """
    scheduler = Scheduler(SchedulerConfig(max_sequences=8))
    runner = DenseModelRunner(tiny_model)
    scheduler.add_all([Sequence(prompt_token_ids=[1, 2, 3], max_tokens=2, sampling_params=GREEDY)
                for _ in range(8)])

    output = scheduler.schedule()
    runner.execute(output)

    assert len(output.scheduled) == 8
    assert len(runner.caches) == 8, "eight separate caches, so eight separate forward passes"
    offsets = {seq_id: caches[0].offset for seq_id, caches in runner.caches.items()}
    assert set(offsets.values()) == {3}
