"""The bookkeeping speculative decoding needs: propose, accept, and roll back.

Speculation writes tokens into the cache before it knows whether they are wanted, making
rollback a correctness requirement. Rollback is also where a paged engine leaks: a rejected
proposal that spilled into a fresh page must return it, and a page dropped without a decref
is an out-of-memory a thousand requests later with nothing pointing at the cause.

These tests therefore check two numbers agreeing — how many slots a sequence thinks it has
computed, and how many the block manager has reserved for it — across every outcome the
verifier can produce: everything accepted, nothing accepted, a tail rejected, and a stop
token landing in the middle of an accepted run.
"""

from __future__ import annotations

import pytest

from mini_vllm.block.block_manager import BlockManager
from mini_vllm.block.block_table import BlockTable
from mini_vllm.sampler import SamplingParams
from mini_vllm.serve.sequence import Sequence, SequenceStatus

GREEDY = SamplingParams(temperature=0.0)


def decoding_sequence(prompt: int = 4, outputs: int = 1, **kwargs) -> Sequence:
    """A sequence past prefill, with one uncommitted token — the state a decode step sees.

    `num_computed_tokens` is one short of the length on purpose: the last sampled token
    has been chosen but not yet run through the model, which is the invariant the whole
    engine maintains between iterations.
    """
    sequence = Sequence(prompt_token_ids=list(range(1, prompt + 1)), max_tokens=64, **kwargs)
    sequence.set_status(SequenceStatus.RUNNING)
    sequence.output_token_ids = [100 + index for index in range(outputs)]
    sequence.num_computed_tokens = len(sequence) - 1
    return sequence


# ------------------------------------------------------------------ proposals


def test_proposals_lengthen_the_sequence_without_joining_the_output():
    """A proposal needs a position and a slot, but it is not yet an emitted token."""
    sequence = decoding_sequence()
    length_before, output_before = len(sequence), list(sequence.output_token_ids)

    sequence.propose([201, 202, 203])

    assert len(sequence) == length_before + 3, "proposals must be forwarded, so they count"
    assert sequence.token_ids[-3:] == [201, 202, 203]
    assert sequence.output_token_ids == output_before, "a proposal is not output"
    assert sequence.num_uncomputed_tokens == 4, "the pending token plus three proposals"


def test_a_proposed_stop_token_does_not_finish_the_sequence():
    """The reason proposals live outside the output at all.

    If a speculated end-of-text could set `is_done`, a request would be returned to its
    caller on the strength of a guess the target model was about to reject.
    """
    sequence = decoding_sequence(eos_token_id=999)

    sequence.propose([999])

    assert not sequence.is_done(), "an unverified proposal ended the sequence"
    assert sequence.finish_reason is None


def test_a_sequence_cannot_speculate_during_prefill():
    """There is nothing to speculate from until the prompt has been computed."""
    sequence = Sequence(prompt_token_ids=[1, 2, 3, 4])
    sequence.set_status(SequenceStatus.RUNNING)
    with pytest.raises(ValueError, match="still in prefill"):
        sequence.propose([5])


def test_proposals_cannot_stack():
    sequence = decoding_sequence()
    sequence.propose([201])
    with pytest.raises(ValueError, match="unverified proposals"):
        sequence.propose([202])


# ----------------------------------------------------------------- acceptance


def test_accepting_everything_commits_the_run_and_the_bonus():
    """`k` accepted proposals plus the bonus: `k + 1` tokens out of one forward pass."""
    sequence = decoding_sequence()
    computed_before = sequence.num_computed_tokens
    sequence.propose([201, 202, 203])

    to_trim = sequence.accept([201, 202, 203, 204], num_accepted=3)

    assert to_trim == 0, "nothing was rejected, so nothing is given back"
    assert sequence.output_token_ids[-4:] == [201, 202, 203, 204]
    assert sequence.proposed_token_ids == []
    # The forward computed the pending token and all three proposals; the bonus is not
    # computed, which leaves the usual one-uncommitted-token invariant.
    assert sequence.num_computed_tokens == computed_before + 4
    assert sequence.num_uncomputed_tokens == 1


def test_rejecting_the_tail_gives_back_exactly_the_rejected_slots():
    sequence = decoding_sequence()
    computed_before = sequence.num_computed_tokens
    sequence.propose([201, 202, 203])

    # The verifier kept one proposal and replaced the second with its own token.
    to_trim = sequence.accept([201, 777], num_accepted=1)

    assert to_trim == 2, "two proposals were rejected, so two slots go back"
    assert sequence.output_token_ids[-2:] == [201, 777]
    assert sequence.num_computed_tokens == computed_before + 2
    assert sequence.num_uncomputed_tokens == 1


def test_rejecting_everything_still_emits_one_token():
    """Total rejection is not a stall: the residual draw is always a real token."""
    sequence = decoding_sequence()
    computed_before = sequence.num_computed_tokens
    sequence.propose([201, 202])

    to_trim = sequence.accept([777], num_accepted=0)

    assert to_trim == 2
    assert sequence.output_token_ids[-1] == 777
    # Only the previously-pending token was worth computing.
    assert sequence.num_computed_tokens == computed_before + 1
    assert sequence.num_uncomputed_tokens == 1


def test_a_stop_token_mid_run_truncates_the_rest():
    """Tokens after the stop were computed, but the sequence ended before them."""
    sequence = decoding_sequence(eos_token_id=999)
    sequence.propose([201, 999, 203])

    to_trim = sequence.accept([201, 999, 203, 204], num_accepted=3)

    assert sequence.output_token_ids[-2:] == [201, 999], "output ran past the stop token"
    assert sequence.is_done() and sequence.finish_reason == "stop"
    # One proposal survived past the pending token; the other two go back.
    assert to_trim == 1


def test_accept_rejects_tokens_that_were_never_proposed():
    """A mismatch means verification and proposal have gone out of step."""
    sequence = decoding_sequence()
    sequence.propose([201, 202])
    with pytest.raises(ValueError, match="not the proposals"):
        sequence.accept([888, 999], num_accepted=1)


def test_accept_wants_one_more_token_than_it_accepted():
    sequence = decoding_sequence()
    sequence.propose([201, 202])
    with pytest.raises(ValueError, match="plus one"):
        sequence.accept([201], num_accepted=1)


def test_preemption_drops_unverified_proposals():
    """They never belonged to the request, so a recomputed prefill must not include them."""
    sequence = decoding_sequence()
    sequence.propose([201, 202])
    length_with_proposals = len(sequence)

    sequence.reset_for_recompute()

    assert sequence.proposed_token_ids == []
    assert len(sequence) < length_with_proposals


# -------------------------------------------------------------- table rollback


def test_trim_tokens_reports_the_blocks_that_fell_empty():
    """Occupancy moves here; only the manager may hand pages back."""
    table = BlockTable(4, block_ids=[10, 11, 12], num_tokens=9)

    # 9 tokens fill two blocks and one slot of the third; dropping 2 empties the third.
    assert table.trim_tokens(2) == 1
    assert table.num_tokens == 7
    assert table.num_blocks == 3, "the blocks stay until the manager drops them"


def test_trim_within_the_partial_block_empties_nothing():
    table = BlockTable(4, block_ids=[10, 11], num_tokens=7)
    assert table.trim_tokens(2) == 0
    assert table.num_tokens == 5


def test_a_table_will_not_trim_more_than_it_holds():
    table = BlockTable(4, block_ids=[10], num_tokens=3)
    with pytest.raises(ValueError, match="cannot trim"):
        table.trim_tokens(4)


def test_drop_last_block_refuses_a_block_still_holding_tokens():
    """Dropping an occupied block would unmap KV the sequence is still attending over."""
    table = BlockTable(4, block_ids=[10, 11], num_tokens=6)
    with pytest.raises(ValueError, match="still holds tokens"):
        table.drop_last_block()


# ------------------------------------------------------------ manager rollback


def test_trimming_a_spilled_proposal_returns_the_page():
    """The case rollback exists for: proposals that crossed a block boundary."""
    manager = BlockManager(num_blocks=8, block_size=4)
    sequence = Sequence(prompt_token_ids=[1, 2, 3], max_tokens=32)
    sequence.set_status(SequenceStatus.RUNNING)

    manager.allocate(sequence, 3)  # 3 tokens, one block
    free_after_prompt = manager.num_free_blocks

    # Five more tokens: a pending token plus four proposals, spilling into two more blocks.
    manager.allocate(sequence, 5)
    assert manager.num_free_blocks < free_after_prompt

    released = manager.trim(sequence, 5)

    assert released == 1, "the pages the rejected tail occupied come back"
    assert manager.num_free_blocks == free_after_prompt
    assert manager.table(sequence).num_tokens == 3


def test_trimming_nothing_is_free_and_touches_no_pages():
    manager = BlockManager(num_blocks=8, block_size=4)
    sequence = Sequence(prompt_token_ids=[1, 2, 3], max_tokens=32)
    sequence.set_status(SequenceStatus.RUNNING)
    manager.allocate(sequence, 3)

    free_before = manager.num_free_blocks
    assert manager.trim(sequence, 0) == 0
    assert manager.num_free_blocks == free_before


def test_a_full_speculative_round_trip_leaks_no_blocks():
    """Propose, reject most of it, roll back, and end where the pool started.

    The end-to-end statement about memory: a speculative step that goes badly costs
    nothing permanent. Run repeatedly, because a leak of one page per step is invisible
    once and fatal a thousand times.
    """
    manager = BlockManager(num_blocks=32, block_size=4)
    sequence = Sequence(prompt_token_ids=[1, 2, 3, 4], max_tokens=256)
    sequence.set_status(SequenceStatus.RUNNING)
    manager.allocate(sequence, 4)
    sequence.num_computed_tokens = 4
    sequence.append_token(100)  # the pending token every decode step starts from

    free_at_start = manager.num_free_blocks
    blocks_at_start = manager.table(sequence).num_blocks

    for round_index in range(12):
        proposals = [200 + round_index, 201 + round_index, 202 + round_index]
        sequence.propose(proposals)
        manager.allocate(sequence, sequence.num_uncomputed_tokens)

        # Keep one proposal, reject the rest — the interesting case for rollback.
        to_trim = sequence.accept([proposals[0], 900 + round_index], num_accepted=1)
        manager.trim(sequence, to_trim)

        # The manager's reservation and the sequence's own count must not drift apart:
        # the table holds exactly the computed positions, and one token is pending.
        assert manager.table(sequence).num_tokens == sequence.num_computed_tokens
        assert sequence.num_uncomputed_tokens == 1

    # The sequence grew, so it legitimately holds more pages than it started with — but
    # only as many as its length needs, and every rejected page came back.
    grown = manager.table(sequence).num_blocks - blocks_at_start
    assert manager.num_free_blocks == free_at_start - grown

    manager.free(sequence)
    assert manager.num_free_blocks == free_at_start + blocks_at_start
