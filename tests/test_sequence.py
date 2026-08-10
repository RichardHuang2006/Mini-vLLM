"""Step 4.1 — sequence state.

No GPU and no model: this is integer bookkeeping, which is precisely why it is
worth testing exhaustively here rather than discovering it through a scheduler bug
later. The two things the tests are really about:

* `num_computed_tokens` versus `len(sequence)`, whose difference is chunked
  prefill and whose conflation is the bug that produces a duplicated or skipped
  token at every chunk boundary;
* the status machine, where the transition that matters is preemption — the only
  way a sequence goes backwards.
"""

from __future__ import annotations

import pytest

from mini_vllm.sampler import SamplingParams
from mini_vllm.serve.sequence import Sequence, SequenceStatus


def make(prompt=(1, 2, 3), **kwargs) -> Sequence:
    return Sequence(prompt_token_ids=list(prompt), **kwargs)


# ------------------------------------------------------------------- the counts


def test_a_new_sequence_has_computed_nothing():
    sequence = make((1, 2, 3, 4))

    assert len(sequence) == 4
    assert sequence.num_computed_tokens == 0
    assert sequence.num_uncomputed_tokens == 4
    assert sequence.status is SequenceStatus.WAITING


def test_token_ids_is_prompt_then_output():
    sequence = make((1, 2, 3))
    sequence.append_token(9)
    sequence.append_token(8)

    assert sequence.token_ids == [1, 2, 3, 9, 8]
    assert sequence.num_prompt_tokens == 3
    assert sequence.num_output_tokens == 2
    assert len(sequence) == 5


def test_seq_ids_are_unique():
    assert make().seq_id != make().seq_id


def test_an_empty_prompt_is_refused():
    """There is no forward pass to run, so the engine has nothing to do with it."""
    with pytest.raises(ValueError, match="at least one prompt token"):
        Sequence(prompt_token_ids=[])


def test_advance_cannot_outrun_the_tokens():
    sequence = make((1, 2, 3))

    sequence.advance(2)
    assert sequence.num_computed_tokens == 2

    with pytest.raises(ValueError, match="cannot compute"):
        sequence.advance(2)


def test_appending_a_token_does_not_make_it_computed():
    """The sampled token has no key or value in the cache yet.

    This is the distinction that keeps the *next* forward pass correct: it must be
    given that token as its input, which means it must still count as uncomputed.
    Advancing here instead would skip a position, and the symptom is a dropped
    token rather than an exception.
    """
    sequence = make((1, 2, 3))
    sequence.advance(3)

    sequence.append_token(9)

    assert sequence.num_computed_tokens == 3
    assert sequence.num_uncomputed_tokens == 1


# ------------------------------------------------------------------- the phases


def test_prefill_ends_exactly_when_the_prompt_is_computed():
    sequence = make((1, 2, 3, 4, 5))

    for computed in range(5):
        assert sequence.is_prefill(), f"still prefill at {computed}"
        sequence.advance(1)

    assert not sequence.is_prefill()


def test_prefill_survives_chunking():
    """Three chunks of a five-token prompt, ending in the same place as one pass."""
    sequence = make((1, 2, 3, 4, 5))

    sequence.advance(2)
    assert sequence.is_prefill() and sequence.num_uncomputed_tokens == 3

    sequence.advance(2)
    assert sequence.is_prefill() and sequence.num_uncomputed_tokens == 1

    sequence.advance(1)
    assert not sequence.is_prefill() and sequence.num_uncomputed_tokens == 0


def test_output_tokens_do_not_reopen_prefill():
    """Decode leaves exactly one uncomputed token, which is not a prefill.

    A sequence that is decoding always has one token outstanding — the one just
    sampled — so a definition of "prefill" based on `num_uncomputed_tokens > 0`
    would say every decoding sequence is prefilling. It is the *prompt* boundary
    that matters, which is why `is_prefill` compares against the prompt length.
    """
    sequence = make((1, 2, 3))
    sequence.advance(3)
    sequence.append_token(9)

    assert not sequence.is_prefill()
    assert sequence.num_uncomputed_tokens == 1


# ---------------------------------------------------------------- the stopping


def test_max_tokens_stops_the_sequence():
    sequence = make(max_tokens=2)

    assert not sequence.is_done()
    sequence.append_token(9)
    assert not sequence.is_done()
    sequence.append_token(8)
    assert sequence.is_done()


def test_eos_stops_the_sequence():
    sequence = make(max_tokens=100, eos_token_id=7)

    sequence.append_token(9)
    assert not sequence.is_done()

    sequence.append_token(7)
    assert sequence.is_done()


def test_eos_only_counts_as_the_last_token():
    """An EOS id sampled mid-sequence is not how stopping works.

    It cannot happen from greedy decoding, but it can from sampling with a
    temperature, and a sequence that checked `in output_token_ids` would then
    finish at the wrong length and truncate the caller's text.
    """
    sequence = make(max_tokens=100, eos_token_id=7)
    sequence.append_token(7)
    sequence.output_token_ids.append(9)

    assert not sequence.is_done()


def test_no_eos_configured_means_length_is_the_only_stop():
    sequence = make(max_tokens=3, eos_token_id=None)

    for token in (7, 7):
        sequence.append_token(token)
    assert not sequence.is_done()


def test_max_tokens_must_be_positive():
    with pytest.raises(ValueError, match="max_tokens must be >= 1"):
        make(max_tokens=0)


# ------------------------------------------------------------------ the machine


@pytest.mark.parametrize(
    "start,target",
    [
        (SequenceStatus.WAITING, SequenceStatus.RUNNING),
        (SequenceStatus.RUNNING, SequenceStatus.PREEMPTED),
        (SequenceStatus.RUNNING, SequenceStatus.FINISHED),
        (SequenceStatus.PREEMPTED, SequenceStatus.RUNNING),
        (SequenceStatus.WAITING, SequenceStatus.FINISHED),
    ],
)
def test_legal_transitions(start, target):
    sequence = make()
    sequence.status = start

    sequence.set_status(target)

    assert sequence.status is target


@pytest.mark.parametrize(
    "start,target",
    [
        (SequenceStatus.WAITING, SequenceStatus.PREEMPTED),
        (SequenceStatus.FINISHED, SequenceStatus.RUNNING),
        (SequenceStatus.FINISHED, SequenceStatus.WAITING),
        (SequenceStatus.PREEMPTED, SequenceStatus.WAITING),
    ],
)
def test_illegal_transitions_raise(start, target):
    """Including every way out of FINISHED, which is terminal.

    A finished sequence has had its blocks returned to the pool, so resuming it
    would attend over memory now owned by somebody else — a wrong answer rather
    than a crash, which is why this is refused rather than merely discouraged.
    """
    sequence = make()
    sequence.status = start

    with pytest.raises(ValueError, match="cannot go from"):
        sequence.set_status(target)


def test_setting_the_current_status_is_a_no_op():
    """Even for FINISHED, whose legal-transition set is empty."""
    sequence = make()
    sequence.set_status(SequenceStatus.RUNNING)
    sequence.set_status(SequenceStatus.FINISHED)

    sequence.set_status(SequenceStatus.FINISHED)

    assert sequence.status is SequenceStatus.FINISHED


def test_a_finished_sequence_refuses_more_tokens():
    sequence = make()
    sequence.set_status(SequenceStatus.RUNNING)
    sequence.set_status(SequenceStatus.FINISHED)

    with pytest.raises(ValueError, match="is finished"):
        sequence.append_token(9)


# ---------------------------------------------------------------- the preemption


def test_preemption_keeps_tokens_and_drops_the_cache():
    """The trade: throw away computed keys and values, never emitted tokens.

    A caller streaming this sequence has already seen those output tokens, so they
    have to survive. What is safe to lose is the cache, which is exactly why
    recompute-preemption is expressible without a swap path.
    """
    sequence = make((1, 2, 3))
    sequence.set_status(SequenceStatus.RUNNING)
    sequence.advance(3)
    sequence.append_token(9)
    sequence.advance(1)

    sequence.reset_for_recompute()

    assert sequence.status is SequenceStatus.PREEMPTED
    assert sequence.token_ids == [1, 2, 3, 9]
    assert sequence.num_computed_tokens == 0
    assert sequence.block_table is None


def test_a_preempted_sequence_re_prefills_over_its_output():
    """It comes back as a prefill of 4 tokens, not of the original 3.

    Which is why `is_prefill` cannot be "has no output yet": this sequence has
    output *and* is prefilling. The recomputed prefill covers prompt and output
    together, and the position of the next token has to come out the same as if it
    had never been preempted.
    """
    sequence = make((1, 2, 3))
    sequence.set_status(SequenceStatus.RUNNING)
    sequence.advance(3)
    sequence.append_token(9)

    sequence.reset_for_recompute()

    assert sequence.is_prefill()
    assert sequence.num_uncomputed_tokens == 4


def test_preempting_from_waiting_is_refused():
    """Nothing has run, so there is nothing to preempt — this is a scheduler bug."""
    with pytest.raises(ValueError, match="cannot go from waiting to preempted"):
        make().reset_for_recompute()


# ----------------------------------------------------------------------- extras


def test_sampling_params_default_to_greedy_length_capped():
    sequence = make()

    assert sequence.sampling_params == SamplingParams()
    assert sequence.max_tokens == 16


def test_repr_shows_progress():
    sequence = make((1, 2, 3), max_tokens=8)
    sequence.advance(2)

    assert "2/3 computed" in repr(sequence)
    assert "0/8 out" in repr(sequence)
