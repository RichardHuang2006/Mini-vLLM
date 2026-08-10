"""Step 4.1 — one request's state, from arrival to completion.

The whole file exists to make two numbers distinguishable:

* ``len(token_ids)`` — how many tokens the sequence *has*.
* ``num_computed_tokens`` — how many the model has actually run through and whose
  keys and values are therefore in the cache.

In a naive engine those are the same number and the distinction looks like
bookkeeping for its own sake. It is not: their difference **is** chunked prefill
(:doc:`DESIGN.md` §7.2). A sequence with 2000 prompt tokens and 512 computed is
mid-prefill, and everything the scheduler needs to know to resume it — where RoPE
positions start, how many tokens to feed, how long the causal mask's key axis is —
follows from that one counter. Modelling it now means [Step 4.4] is a scheduler
change rather than a rewrite of this class.

The status machine is here for a different reason: preemption. A sequence can go
back to waiting after having run, and its already-computed tokens are then thrown
away, which is a transition that has to be *deliberate* rather than reachable by
accident from anywhere.
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field

from mini_vllm.block.block_table import BlockTable
from mini_vllm.sampler import SamplingParams

__all__ = ["Sequence", "SequenceStatus"]


class SequenceStatus(enum.Enum):
    """Where a request is in its life.

    ``PREEMPTED`` is distinct from ``WAITING`` even though both queue for
    admission, because they are not the same situation: a preempted sequence has
    output tokens already emitted to its caller and must be recomputed from its
    prompt *plus* that output, while a waiting one has nothing behind it. Collapsing
    the two loses the tokens the caller has already seen.
    """

    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


# Which transitions are legal. Written as data rather than as `if` statements
# scattered through the scheduler, so an illegal move is a single raise in one
# place instead of a state that silently makes sense to nobody later.
_LEGAL_TRANSITIONS: dict[SequenceStatus, frozenset[SequenceStatus]] = {
    SequenceStatus.WAITING: frozenset({SequenceStatus.RUNNING, SequenceStatus.FINISHED}),
    SequenceStatus.RUNNING: frozenset(
        {SequenceStatus.PREEMPTED, SequenceStatus.FINISHED, SequenceStatus.WAITING}
    ),
    SequenceStatus.PREEMPTED: frozenset({SequenceStatus.RUNNING, SequenceStatus.FINISHED}),
    # Terminal. A finished sequence's blocks are gone, so "resume" is not a thing
    # that could be made to work — it is a bug in the caller.
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

    Mutable by design, and mutated in exactly two places: the scheduler advances
    `num_computed_tokens` and the engine appends sampled tokens. Everything else is
    derived.
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

    # The sequence's pages, set by the block manager ([Step 4.7]) and the single home
    # for them: the manager reads and writes this field rather than keeping a registry
    # of its own, because two places recording which blocks a sequence holds is two
    # places that can disagree about when to free them.
    block_table: BlockTable | None = None

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("a sequence needs at least one prompt token to run a forward pass")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")

    # ------------------------------------------------------------------ lengths

    @property
    def token_ids(self) -> list[int]:
        """Prompt then output: the sequence the model actually attends over."""
        return self.prompt_token_ids + self.output_token_ids

    def __len__(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

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

        Note what this is *not*: it is not "no output tokens yet". A sequence is in
        prefill exactly while its prompt is not fully computed, which is the
        condition that survives chunking. The first decode step happens when
        `num_computed_tokens == num_prompt_tokens`, whatever route got it there.
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

        Deliberately does *not* advance `num_computed_tokens`: the token has been
        chosen but no forward pass has consumed it yet, so its key and value are not
        in the cache. Conflating the two is how a decode step ends up skipping a
        position, and the symptom is a repeated or dropped token rather than a crash.
        """
        if self.status is SequenceStatus.FINISHED:
            raise ValueError(f"sequence {self.seq_id} is finished and cannot take more tokens")
        self.output_token_ids.append(token_id)

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

    def reset_for_recompute(self) -> None:
        """Drop everything cached, keeping the tokens. Used when preempting.

        Preemption by recomputation rather than by swapping to host memory: the
        blocks go back to the pool and the sequence starts its prefill again, over
        prompt *and* output this time. It costs the compute already spent, and it
        keeps the engine free of a swap path — the trade [Step 4.3] makes explicit.

        Dropping the table is *not* the same as releasing the blocks, and this refuses
        to do the first without the second. Forgetting the pointer to pages the pool
        still believes are held is a leak that shows up much later, as an engine that
        runs a few hundred requests and then cannot admit anything.
        """
        if self.block_table is not None:
            raise ValueError(
                f"sequence {self.seq_id} still holds {self.block_table.num_blocks} blocks; "
                "free them through the block manager before resetting it"
            )
        self.set_status(SequenceStatus.PREEMPTED)
        self.num_computed_tokens = 0

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, {self.status.value}, "
            f"{self.num_computed_tokens}/{len(self)} computed, "
            f"{self.num_output_tokens}/{self.max_tokens} out)"
        )
