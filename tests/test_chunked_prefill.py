"""Chunked prefill and piggyback decoding.

Two claims, and they fail in different ways:

* **Splitting a prefill does not change it.** A 2000-token prompt fed in 512-token
  chunks must produce what it produces in one pass. This is where a position or
  mask error lives: the model sees a `(512, 1024)` query-by-key rectangle on the
  second chunk, and if RoPE starts at 0 again, or the causal diagonal is not shifted
  right by `S - L`, the logits are wrong in a way that still reads as fluent text.
* **Mixing prefill and decode in one pass does not change either.** A chunk beside a
  dozen single-token decodes is one ragged forward pass, and every sequence in it
  must produce what it produces alone.

The policy tests run without a model, because that is where the scheduling arithmetic
is; the identity tests carry the weight.
"""

from __future__ import annotations

import pytest
import torch
from conftest import assert_allclose

# This file extends the scheduler tests' harness rather than reimplementing it: the
# engine loop, the greedy parameters and the single-sequence reference are exactly the
# ones the continuous-batching tests use, and a chunked run has to match *that*
# reference to mean anything.
from test_scheduler import GREEDY, WHOLE, alone, make, run_to_completion

from mini_vllm.serve.scheduler import DenseModelRunner, Scheduler, SchedulerConfig
from mini_vllm.serve.sequence import Sequence


# ------------------------------------------------------------------ splitting


def test_a_long_prompt_is_split_into_chunks():
    """2000 tokens, 512 to an iteration: 512, 512, 512, 464."""
    scheduler = Scheduler(SchedulerConfig(max_batched_tokens=2048, chunk_size=512))
    sequence = make(2000, max_tokens=1)
    scheduler.add(sequence)

    counts = []
    while sequence.is_prefill():
        output = scheduler.schedule()
        counts.append(output.tokens_for(sequence))
        scheduler.commit(output)

    assert counts == [512, 512, 512, 464]
    assert sequence.num_computed_tokens == 2000


def test_a_chunk_is_bounded_by_the_budget_as_well_as_the_chunk_size():
    """Whichever is smaller. A chunk that overran the budget would defeat both."""
    scheduler = Scheduler(SchedulerConfig(max_batched_tokens=100, chunk_size=512))
    scheduler.add(make(2000))

    assert scheduler.schedule().total_tokens == 100


def test_no_token_is_emitted_until_the_prompt_is_finished():
    """A mid-prompt chunk's last position is not the position that predicts anything.

    Its logits describe the token *after* one the caller already sent, so sampling
    from them would invent a token the prompt then contradicts. Only the chunk that
    reaches the end of the prompt produces output.
    """
    scheduler = Scheduler(SchedulerConfig(max_batched_tokens=64, chunk_size=20))
    sequence = make(50)
    scheduler.add(sequence)

    for expected_outputs in (0, 0, 1):
        output = scheduler.schedule()
        # The runner samples a row for every scheduled sequence, mid-prefill or not;
        # `commit` is what decides whether it counts.
        scheduler.commit(output, [7] * len(output.scheduled))
        assert sequence.num_output_tokens == expected_outputs

    assert sequence.output_token_ids == [7]


def test_the_head_of_line_stall_is_gone():
    """The same prompt that monopolizes an iteration without chunking, now bounded.

    With chunking off it takes 2000 tokens in one pass and everything else waits;
    with it on, no iteration exceeds the budget it was given.
    """
    stalling = Scheduler(SchedulerConfig(max_batched_tokens=512, **WHOLE))
    stalling.add(make(2000))
    assert stalling.schedule().total_tokens == 2000

    chunked = Scheduler(SchedulerConfig(max_batched_tokens=512, chunk_size=512))
    chunked.add(make(2000))
    assert chunked.schedule().total_tokens == 512


# ------------------------------------------------------------------ piggyback


def decoding_scheduler(config: SchedulerConfig, count: int) -> tuple[Scheduler, list[Sequence]]:
    """A scheduler with `count` sequences past their prompts and into decode."""
    scheduler = Scheduler(config)
    sequences = [make(2, max_tokens=8) for _ in range(count)]
    scheduler.add_all(sequences)
    scheduler.commit(scheduler.schedule(), [5] * count)
    assert all(not sequence.is_prefill() for sequence in sequences)
    return scheduler, sequences


def test_decodes_ride_along_with_a_prefill_chunk():
    """One forward pass carrying a 300-token chunk and three single-token decodes.

    This is stall-free piggyback decoding, and the ragged `ForwardBatch` exists for
    exactly this shape.
    """
    scheduler, decoders = decoding_scheduler(
        SchedulerConfig(max_batched_tokens=1024, chunk_size=300), count=3
    )
    prompt = make(2000)
    scheduler.add(prompt)

    output = scheduler.schedule()

    assert output.total_tokens == 303
    assert output.num_decodes == 3
    assert [output.tokens_for(sequence) for sequence in decoders] == [1, 1, 1]
    assert output.tokens_for(prompt) == 300

    batch = output.batch()
    assert not batch.is_pure_decode
    assert batch.num_prefill_tokens == 300
    assert batch.total_tokens == 303


def test_a_decode_is_never_stalled_by_a_long_prefill():
    """The point of piggybacking, stated as a latency claim.

    A sequence in decode advances one token every iteration while a 2000-token
    prompt is being chunked beside it. Without piggybacking it would advance on none
    of them, which is a request that stops streaming for as long as someone else's
    prompt takes.
    """
    scheduler, (decoder,) = decoding_scheduler(
        SchedulerConfig(max_batched_tokens=600, chunk_size=512), count=1
    )
    scheduler.add(make(2000, max_tokens=1))

    before = decoder.num_output_tokens
    for _ in range(3):
        output = scheduler.schedule()
        assert output.tokens_for(decoder) == 1, "the decode was left out of an iteration"
        scheduler.commit(output, [5] * len(output.scheduled))

    assert decoder.num_output_tokens == before + 3


def test_the_chunk_takes_what_the_decodes_leave():
    """Decodes are scheduled first, so the chunk shrinks rather than the budget growing."""
    scheduler, _ = decoding_scheduler(
        SchedulerConfig(max_batched_tokens=10, chunk_size=512), count=4
    )
    prompt = make(2000)
    scheduler.add(prompt)

    output = scheduler.schedule()

    assert output.total_tokens == 10
    assert output.tokens_for(prompt) == 6


def test_a_prefill_that_has_started_finishes_before_a_new_one_is_admitted():
    """Its cache is already committed and will not be released until it is done."""
    scheduler = Scheduler(SchedulerConfig(max_batched_tokens=512, chunk_size=512))
    started = make(2000)
    scheduler.add(started)
    scheduler.commit(scheduler.schedule())

    scheduler.add(make(4))
    output = scheduler.schedule()

    assert output.sequences == [started], "a new arrival stole the whole budget"


def test_a_decode_that_arrives_mid_chunk_is_picked_up_immediately():
    """Admission is per-iteration, so a request does not wait for the chunking to end."""
    scheduler = Scheduler(SchedulerConfig(max_batched_tokens=600, chunk_size=512))
    scheduler.add(make(2000))
    scheduler.commit(scheduler.schedule())

    arrival = make(4)
    scheduler.add(arrival)

    assert scheduler.schedule().tokens_for(arrival) == 4


# --------------------------------------------------------- the ragged batch


def test_the_batch_describes_each_sequence_at_its_own_length():
    scheduler, decoders = decoding_scheduler(
        SchedulerConfig(max_batched_tokens=1024, chunk_size=8), count=2
    )
    prompt = make(20)
    scheduler.add(prompt)

    batch = scheduler.schedule().batch()

    # L differs per sequence (1 for a decode, 8 for the chunk) and S is larger than L
    # for the decodes, whose prefixes are cached. Those two facts together are the
    # entire reason the kernels take offsets instead of a shape.
    assert batch.seq_lens.tolist() == [1, 1, 8]
    assert batch.context_lens.tolist() == [3, 3, 8]
    assert batch.cu_seqlens_q.tolist() == [0, 1, 2, 10]
    assert batch.num_sequences == 3
    assert batch.describe() == (
        f"batch[10 tokens] {decoders[0].seq_id}:decode/3 "
        f"{decoders[1].seq_id}:decode/3 {prompt.seq_id}:prefill 8/8"
    )


def test_positions_resume_where_the_previous_chunk_stopped():
    """The chunk-boundary bug, caught at the metadata rather than in the logits.

    RoPE takes explicit positions precisely so the second chunk can start at 8
    instead of at 0. An `arange(L)` here would be invisible until the text came out
    subtly wrong.
    """
    scheduler = Scheduler(SchedulerConfig(max_batched_tokens=64, chunk_size=8))
    sequence = make(24)
    scheduler.add(sequence)

    seen = []
    while sequence.is_prefill():
        output = scheduler.schedule()
        seen.append(output.batch().positions.tolist())
        scheduler.commit(output)

    assert seen == [list(range(0, 8)), list(range(8, 16)), list(range(16, 24))]


# ------------------------------------------------------------------ identity


@pytest.fixture
def tiny_model(tiny_qwen3):
    """The tiny model on the CPU. Chunking is arithmetic; it needs no GPU to be wrong."""
    from conftest import config_from_hf, weights_from_hf

    from mini_vllm.model.qwen3_cached import Qwen3Cached

    return Qwen3Cached(config_from_hf(tiny_qwen3), weights_from_hf(tiny_qwen3), use_cuda=False)


def logits_in_chunks(model, prompt: list[int], chunk: int) -> torch.Tensor:
    """Feed `prompt` a chunk at a time, returning the last position's logits."""
    caches = model.create_kv_cache()
    tokens = torch.tensor([prompt], dtype=torch.int64)
    last = None
    for start in range(0, len(prompt), chunk):
        piece = tokens[:, start : start + chunk]
        positions = torch.arange(start, start + piece.shape[1])
        last = model(piece, caches, positions=positions, last_only=True)
    return last


def test_chunked_prefill_gives_the_same_logits_as_one_pass(tiny_model):
    """The headline claim, checked on the logits themselves.

    Not bitwise: the chunked run multiplies differently-shaped matrices, so cuBLAS
    (or MKL) is free to accumulate in a different order. What must hold is agreement
    to fp32 tolerance and, downstream, the same greedy token.
    """
    prompt = torch.randint(0, 512, (200,)).tolist()
    whole = tiny_model(torch.tensor([prompt]), tiny_model.create_kv_cache(), last_only=True)

    for chunk in (1, 7, 64, 199, 200):
        chunked = logits_in_chunks(tiny_model, prompt, chunk)
        assert_allclose(chunked, whole, msg=f"chunk size {chunk}")
        assert chunked.argmax(-1).item() == whole.argmax(-1).item(), f"chunk size {chunk}"


def test_a_wrong_position_at_a_chunk_boundary_is_caught(tiny_model):
    """The mutation the previous test is protecting against, shown to be detectable.

    Restarting RoPE at 0 on the second chunk is the single most likely chunked-prefill
    bug, and it produces a model that still emits plausible tokens. If this test ever
    passes, the comparison above has stopped measuring anything.
    """
    prompt = torch.randint(0, 512, (40,)).tolist()
    whole = tiny_model(torch.tensor([prompt]), tiny_model.create_kv_cache(), last_only=True)

    caches = tiny_model.create_kv_cache()
    tokens = torch.tensor([prompt])
    for start in (0, 20):
        piece = tokens[:, start : start + 20]
        wrong = tiny_model(piece, caches, positions=torch.arange(20), last_only=True)

    assert not torch.allclose(wrong, whole, rtol=1e-3, atol=1e-3), (
        "restarting RoPE at zero changed nothing, so the positions are not being used"
    )


def test_a_chunked_run_is_token_identical_to_an_unchunked_one(tiny_model):
    """Through the engine loop this time, tokens rather than logits."""
    prompt = torch.randint(0, 512, (37,)).tolist()
    expected = alone(tiny_model, prompt, max_tokens=6)

    scheduler = Scheduler(SchedulerConfig(max_batched_tokens=8, chunk_size=8))
    sequence = Sequence(prompt_token_ids=prompt, max_tokens=6, sampling_params=GREEDY)
    scheduler.add(sequence)

    got = run_to_completion(scheduler, DenseModelRunner(tiny_model))

    assert got[sequence.seq_id] == expected


def test_a_piggybacked_batch_is_token_identical_to_running_each_alone(tiny_model):
    """Chunked prefills and decodes interleaved, every sequence unaffected.

    The budget is deliberately tight relative to the prompts, so the long ones are
    chunked, the short ones decode alongside them, and no sequence sees the same
    batch composition twice.
    """
    prompts = [
        torch.randint(0, 512, (n,)).tolist() for n in (33, 5, 17, 1, 40, 2)
    ]
    expected = [alone(tiny_model, prompt, 5) for prompt in prompts]

    scheduler = Scheduler(
        SchedulerConfig(max_batched_tokens=12, max_sequences=4, chunk_size=8)
    )
    sequences = [
        Sequence(prompt_token_ids=prompt, max_tokens=5, sampling_params=GREEDY)
        for prompt in prompts
    ]
    scheduler.add_all(sequences)

    got = run_to_completion(scheduler, DenseModelRunner(tiny_model))

    for index, sequence in enumerate(sequences):
        assert got[sequence.seq_id] == expected[index], (
            f"prompt {index} ({len(prompts[index])} tokens) changed under chunking"
        )


def test_chunking_does_not_change_the_number_of_tokens_computed(tiny_model):
    """A chunk boundary is not an off-by-one waiting to happen.

    Every token of prompt and output goes through the model exactly once, whatever
    the chunk size — a sequence that recomputed a boundary token would produce the
    same text and a slower engine, which is the kind of bug that survives for months.
    """
    prompt = torch.randint(0, 512, (25,)).tolist()

    for chunk_size in (1, 4, 25, 512):
        scheduler = Scheduler(SchedulerConfig(max_batched_tokens=64, chunk_size=chunk_size))
        sequence = Sequence(prompt_token_ids=prompt, max_tokens=4, sampling_params=GREEDY)
        scheduler.add(sequence)
        runner = DenseModelRunner(tiny_model)

        computed = 0
        while scheduler.has_work:
            output = scheduler.schedule()
            computed += output.total_tokens
            scheduler.commit(output, runner.sample_tokens(output, runner.execute(output)))

        assert computed == len(prompt) + 3, f"chunk size {chunk_size}"
