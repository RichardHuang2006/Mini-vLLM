"""One request's state, from arrival to completion.

Two lengths are tracked separately:

* ``len(token_ids)`` — how many tokens the sequence has.
* ``num_computed_tokens`` — how many have been through a forward pass and therefore
  have keys and values in the cache.

Their difference is chunked prefill. A sequence with 2000 prompt tokens and 512
computed is mid-prefill, and everything needed to resume it — the RoPE position
offset, the next chunk's token count, the causal mask's key-axis length — derives
from that one counter, which keeps chunked prefill a scheduler concern rather than a
change to this class.

The status machine exists for preemption: a sequence may return to the waiting queue
after running, discarding its computed tokens, so that transition is validated
against an explicit table rather than being reachable from any state.
"""

from __future__ import annotations

import enum
import itertools
from collections.abc import Iterable
from dataclasses import dataclass, field

from mini_vllm.block.block_table import BlockTable
from mini_vllm.sampler import SamplingParams

__all__ = ["Sequence", "SequenceStatus"]


class SequenceStatus(enum.Enum):
    """Where a request is in its life cycle.

    ``PREEMPTED`` is distinct from ``WAITING`` although both queue for admission: a
    preempted sequence has output tokens already emitted to its caller and must be
    recomputed over prompt *plus* that output, while a waiting one has nothing behind
    it. Collapsing the two loses tokens the caller has already seen.
    """

    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


# Legal transitions, as data rather than as `if` statements spread through the
# scheduler, so an illegal move raises from one place.
_LEGAL_TRANSITIONS: dict[SequenceStatus, frozenset[SequenceStatus]] = {
    SequenceStatus.WAITING: frozenset({SequenceStatus.RUNNING, SequenceStatus.FINISHED}),
    SequenceStatus.RUNNING: frozenset(
        {SequenceStatus.PREEMPTED, SequenceStatus.FINISHED, SequenceStatus.WAITING}
    ),
    SequenceStatus.PREEMPTED: frozenset({SequenceStatus.RUNNING, SequenceStatus.FINISHED}),
    # Terminal: a finished sequence's blocks have been freed, so it can never resume.
    SequenceStatus.FINISHED: frozenset(),
}

_next_seq_id = itertools.count()


@dataclass
class Sequence:
    """A request in flight.

    ::

        prompt_token_ids   what the caller sent
        output_token_ids   what has been sampled so far
        token_ids          the concatenation — what the model has to attend over
        num_computed_tokens  how much of `token_ids` is already in the KV cache

    Mutated in exactly two places: the scheduler advances `num_computed_tokens` and
    the engine appends sampled tokens. Everything else is derived.
    """

    prompt_token_ids: list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    max_tokens: int = 16
    eos_token_id: int | None = None
    # Qwen3 ends a turn with either `<|im_end|>` or `<|endoftext|>`, so one id is not
    # enough: honouring only `tokenizer.eos_token_id` generates past the end of a turn.
    stop_token_ids: tuple[int, ...] = ()
    seq_id: int = field(default_factory=lambda: next(_next_seq_id))

    output_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    status: SequenceStatus = SequenceStatus.WAITING

    # Speculative decoding: tokens the draft model has proposed but the target has not
    # yet verified. Kept out of `output_token_ids` so a proposal cannot finish a
    # sequence — a speculated end-of-text token would otherwise set `is_done` and
    # return the request on a guess the target was about to reject.
    proposed_token_ids: list[int] = field(default_factory=list)

    # Parallel sampling: a forked branch records the id of the request it split from,
    # so the engine can group a prompt's `n` completions back together. None for an
    # ordinary request, which is its own group of one.
    parent_id: int | None = None

    # The sequence's pages, and the single home for them: the block manager reads and
    # writes this field rather than keeping a registry of its own, so there is only one
    # record of which blocks a sequence holds and when they may be freed.
    block_table: BlockTable | None = None

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("a sequence needs at least one prompt token to run a forward pass")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")

    # ------------------------------------------------------------------ lengths

    @property
    def token_ids(self) -> list[int]:
        """Prompt, output, then any live proposals: what the model attends over.

        Proposals are included because the verifying forward pass has to score them,
        so they need positions, slots and KV like any other token. They do not become
        output until :meth:`accept` records which survived.
        """
        return self.prompt_token_ids + self.output_token_ids + self.proposed_token_ids

    def __len__(self) -> int:
        return (
            len(self.prompt_token_ids)
            + len(self.output_token_ids)
            + len(self.proposed_token_ids)
        )

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_uncomputed_tokens(self) -> int:
        """How many tokens still need a forward pass — the next chunk's ceiling."""
        return len(self) - self.num_computed_tokens

    # -------------------------------------------------------------------- phase

    def is_prefill(self) -> bool:
        """True while any prompt token has yet to be computed.

        This is a statement about the prompt, not about the output being empty, which
        is the form that survives chunking: the first decode step happens once
        `num_computed_tokens == num_prompt_tokens`, however many chunks it took.
        """
        return self.num_computed_tokens < self.num_prompt_tokens

    def is_done(self) -> bool:
        """A stop token emitted, or the output length reached."""
        if self.num_output_tokens >= self.max_tokens:
            return True
        return bool(self.output_token_ids) and self.output_token_ids[-1] in self.stop_ids

    @property
    def stop_ids(self) -> frozenset[int]:
        """Every token that ends this sequence."""
        extra = () if self.eos_token_id is None else (self.eos_token_id,)
        return frozenset(self.stop_token_ids + extra)

    @property
    def finish_reason(self) -> str | None:
        """Why generation stopped: `"stop"`, `"length"`, or None if it has not."""
        if self.output_token_ids and self.output_token_ids[-1] in self.stop_ids:
            return "stop"
        return "length" if self.num_output_tokens >= self.max_tokens else None

    # ------------------------------------------------------------------ mutation

    def append_token(self, token_id: int) -> None:
        """Record a sampled token.

        Does not advance `num_computed_tokens`: the token has been chosen but no
        forward pass has consumed it, so its key and value are not yet in the cache.
        Conflating the two makes a decode step skip a position, which surfaces as a
        repeated or dropped token rather than a crash.
        """
        if self.status is SequenceStatus.FINISHED:
            raise ValueError(f"sequence {self.seq_id} is finished and cannot take more tokens")
        self.output_token_ids.append(token_id)

    @property
    def num_proposed_tokens(self) -> int:
        return len(self.proposed_token_ids)

    def propose(self, token_ids: Iterable[int]) -> None:
        """Attach a draft model's speculated tokens, pending verification.

        They lengthen the sequence immediately, so the scheduler reserves slots for
        them and the runner forwards them, but they stay out of the output until
        :meth:`accept`.
        """
        if self.proposed_token_ids:
            raise ValueError(
                f"sequence {self.seq_id} already holds {len(self.proposed_token_ids)} "
                "unverified proposals; accept or discard them first"
            )
        if self.is_prefill():
            raise ValueError(
                f"sequence {self.seq_id} is still in prefill; there is nothing to speculate "
                "from until its prompt is computed"
            )
        self.proposed_token_ids = [int(token) for token in token_ids]

    def discard_proposals(self) -> int:
        """Drop the proposals without committing any, returning how many were dropped.

        For paths that abandon a speculative step rather than verifying it: preemption,
        or shutdown mid-flight.
        """
        dropped = len(self.proposed_token_ids)
        self.proposed_token_ids = []
        return dropped

    def accept(self, token_ids: list[int], num_accepted: int) -> int:
        """Commit a verified step's tokens, returning how many cache slots to give back.

        ``token_ids`` is the rejection sampler's output: the accepted proposals followed
        by one more token, the bonus token if nothing was rejected or the residual draw
        if something was. ``num_accepted`` is how many leading tokens were surviving
        proposals, which distinguishes them from that final token: the survivors are
        already in the cache, having been computed by the verifying forward pass, while
        the final token is not.

        Hence the return value. The forward pass computed all ``k + 1`` positions but
        only ``num_accepted + 1`` remain computed, so ``k - num_accepted`` slots go back
        to the block manager. The sequence ends in the same state an ordinary decode
        step leaves it: one uncommitted trailing token whose KV is not in the cache.

        A stop token among the accepted tokens truncates here. Later tokens are dropped:
        they were computed, but the sequence ended before them, so returning them would
        be output the model never chose to produce.
        """
        num_proposed = len(self.proposed_token_ids)
        if not 0 <= num_accepted <= num_proposed:
            raise ValueError(
                f"cannot accept {num_accepted} of {num_proposed} proposals"
            )
        if len(token_ids) != num_accepted + 1:
            raise ValueError(
                f"{num_accepted} accepted proposals should come with {num_accepted + 1} "
                f"tokens (the accepted run plus one), got {len(token_ids)}"
            )
        if list(token_ids[:num_accepted]) != self.proposed_token_ids[:num_accepted]:
            raise ValueError(
                f"sequence {self.seq_id}: the accepted tokens are not the proposals that "
                "were made; verification and proposal have gone out of step"
            )

        computed_before = self.num_computed_tokens
        self.proposed_token_ids = []

        # Append until a stop token or the length limit ends the sequence, counting the
        # surviving proposals: only those are already in the cache.
        kept_proposals = 0
        for index, token in enumerate(token_ids):
            self.output_token_ids.append(int(token))
            if index < num_accepted:
                kept_proposals += 1
            if self.is_done():
                break

        # The forward pass computed the one previously-uncommitted token plus every
        # proposal; what remains computed is that token plus the survivors.
        self.num_computed_tokens = computed_before + 1 + kept_proposals
        return num_proposed - kept_proposals

    def advance(self, num_tokens: int) -> None:
        """Record that `num_tokens` more tokens have been through the model."""
        if num_tokens < 0:
            raise ValueError(f"cannot un-compute tokens (got {num_tokens})")
        if self.num_computed_tokens + num_tokens > len(self):
            raise ValueError(
                f"sequence {self.seq_id} has {len(self)} tokens; cannot compute "
                f"{self.num_computed_tokens} + {num_tokens} of them"
            )
        self.num_computed_tokens += num_tokens

    def set_status(self, status: SequenceStatus) -> None:
        """Move to `status`, refusing transitions the life cycle does not allow."""
        if status is self.status:
            return
        if status not in _LEGAL_TRANSITIONS[self.status]:
            raise ValueError(
                f"sequence {self.seq_id} cannot go from {self.status.value} to {status.value}"
            )
        self.status = status

    def fork(self, first_output_token: int | None = None) -> Sequence:
        """A new branch that shares this sequence's prompt, for parallel sampling.

        Copies the prompt and the computed-token count, so the child starts with the
        same prefix already in the cache; the physical pages are shared by the block
        manager's `fork`, which copies no KV. The child takes its own first output
        token, since `n > 1` means n independent continuations of one prompt. Passing
        None leaves it with none, for a caller that will sample it separately.

        The child does not copy the block table: that is the block manager's to set,
        incrementing refcounts as it points the child at the same pages.
        """
        child = Sequence(
            prompt_token_ids=list(self.prompt_token_ids),
            sampling_params=self.sampling_params,
            max_tokens=self.max_tokens,
            eos_token_id=self.eos_token_id,
            stop_token_ids=self.stop_token_ids,
            parent_id=self.seq_id if self.parent_id is None else self.parent_id,
        )
        child.num_computed_tokens = self.num_computed_tokens
        if first_output_token is not None:
            child.output_token_ids = [int(first_output_token)]
        child.set_status(SequenceStatus.RUNNING)
        return child

    def reset_for_recompute(self) -> None:
        """Drop everything cached, keeping the tokens. Used when preempting.

        Preemption is by recomputation rather than by swapping to host memory: blocks
        go back to the pool and prefill restarts over prompt *and* output. That trades
        the compute already spent for having no swap path in the engine.

        Dropping the table is not the same as releasing the blocks, and this refuses to
        do the first without the second: losing the pointer to pages the pool still
        counts as held leaks them, surfacing much later as an engine that admits
        nothing after a few hundred requests.
        """
        if self.block_table is not None:
            raise ValueError(
                f"sequence {self.seq_id} still holds {self.block_table.num_blocks} blocks; "
                "free them through the block manager before resetting it"
            )
        self.set_status(SequenceStatus.PREEMPTED)
        self.num_computed_tokens = 0
        # Unverified proposals do not survive a preemption: their KV went back to the
        # pool with the rest, and they were never part of the request, so keeping them
        # would make the recomputed prefill longer than what the caller asked for.
        self.proposed_token_ids = []

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, {self.status.value}, "
            f"{self.num_computed_tokens}/{len(self)} computed, "
            f"{self.num_output_tokens}/{self.max_tokens} out)"
        )
