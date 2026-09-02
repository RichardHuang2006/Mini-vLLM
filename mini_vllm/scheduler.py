"""Continuous batching: sequences, ragged batches, and the per-iteration policy.

What this file teaches
    How an engine decides *what to run next*, in five layers:

    1. `SequenceStatus` / `Sequence` — one request's state, and the two lengths
       (present vs computed) whose difference makes chunked prefill and
       preemption expressible.
    2. `ForwardBatch` — a ragged batch: sequences of different lengths
       flattened onto one token axis instead of padded into a rectangle.
    3. `SchedulerConfig` / `SchedulerOutput` — the per-iteration budgets and
       the decision they produce.
    4. `Scheduler` — continuous batching (Orca-style iteration-level
       scheduling), chunked prefill, piggyback decoding, admission control,
       and preemption by recomputation.
    5. `DenseModelRunner` — the dense-cache oracle that shows what paging buys:
       without paging, a scheduled batch degenerates to one forward pass per
       sequence.

Inputs and outputs
    In: `Sequence` objects from the engine and free-page counts from
    `cache.BlockManager`. Out: a `SchedulerOutput` — `(sequence, token count)`
    pairs — which `ForwardBatch.from_scheduled` turns into the tensors one
    ragged forward pass consumes.

Read next
    `kernels.py` — how the ragged batch metadata reaches the CUDA kernels.

One invariant
    A scheduling decision may change *timing*, never *output*. A sequence run
    alone, run beside fifteen others, chunked, piggybacked, or preempted and
    recomputed must emit identical tokens; the test suite asserts this by
    token-for-token comparison against single-sequence runs.

On "stall-free" piggyback decoding
    Decodes are scheduled before prefill chunks in every iteration, so a
    decode step is never displaced by prompt work within an iteration's token
    budget: what is bounded is head-of-line blocking of decodes behind a long
    prompt (a decode waits at most one bounded chunk, never an unbounded whole
    prompt). This is a scheduling-order guarantee, not an absolute latency
    guarantee — a decode still shares each iteration's wall clock with the
    chunk it rides beside, and the scheduler benchmark measures exactly that
    trade.
"""

from __future__ import annotations

import enum
import itertools
from collections import deque
from collections.abc import Iterable, Sequence as SequenceABC
from dataclasses import dataclass, field

import torch

from mini_vllm.cache import BlockTable, KvCache, OutOfBlocks
from mini_vllm.config import SamplingParams, SchedulerConfig
from mini_vllm.ops import sample

__all__ = [
    "Sequence",
    "SequenceStatus",
    "ForwardBatch",
    "SchedulerConfig",
    "SchedulerOutput",
    "Scheduler",
    "DenseModelRunner",
]


# ---------------------------------------------------------- 1. sequence states


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


# ------------------------------------------------------- 2. per-request state


@dataclass
class Sequence:
    """A request in flight.

    ::

        prompt_token_ids   what the caller sent
        output_token_ids   what has been sampled so far
        token_ids          the concatenation — what the model has to attend over
        num_computed_tokens  how much of `token_ids` is already in the KV cache

    Two lengths are tracked separately: ``len(token_ids)`` is how many tokens the
    sequence *has*, ``num_computed_tokens`` how many have been through a forward pass
    and therefore have keys and values in the cache. Their difference is chunked
    prefill. A sequence with 2000 prompt tokens and 512 computed is mid-prefill, and
    everything needed to resume it — the RoPE position offset, the next chunk's token
    count, the causal mask's key-axis length — derives from that one counter, which
    keeps chunked prefill a scheduler concern rather than a change to this class.

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
            raise ValueError(f"cannot accept {num_accepted} of {num_proposed} proposals")
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


# --------------------------------------------------- 3. ragged forward batches

# What an unused block-table entry holds. A kernel bounded by `context_lens` never
# reads it, so a read is a bug, and -1 is an impossible block id rather than a silent
# alias for block 0.
PADDING_BLOCK = -1


def _check_metadata(
    num_sequences: int,
    num_params: int,
    seq_lens: list[int],
    context_lens: list[int],
    offsets: list[int],
    num_tokens: int,
    num_positions: int,
    num_slots: int | None,
    num_tables: int | None,
    empty_tables: list[int],
) -> None:
    """Check one batch's metadata for consistency, in plain integers.

    Integers rather than tensors, which is why this is a free function. Each check
    compares one field against another, so on device tensors each would cost a
    device-to-host read, and reading a CUDA tensor waits for everything queued on the
    stream — the previous iteration's twenty-eight layers. That would put several
    pipeline drains on every iteration's critical path to recover numbers the scheduler
    already held as `int`.

    The builder therefore calls this with what it already knows, and
    `ForwardBatch.__post_init__` calls it for a batch assembled by hand.
    """
    if not num_sequences:
        raise ValueError("a forward batch needs at least one sequence")

    for name, actual, expected in (
        ("seq_lens", len(seq_lens), num_sequences),
        ("context_lens", len(context_lens), num_sequences),
        ("cu_seqlens_q", len(offsets), num_sequences + 1),
    ):
        if actual != expected:
            raise ValueError(f"{name} has {actual} entries, expected {expected}")

    if num_params != num_sequences:
        raise ValueError(f"got {num_params} sampling params for {num_sequences} sequences")
    if num_positions != num_tokens:
        raise ValueError(
            f"input_ids has {num_tokens} tokens but positions has {num_positions}; "
            "they index the same tokens"
        )

    # The offsets must describe the token axis they index into. This catches a scheduler
    # that admitted a sequence but did not extend the flattened ids, which would
    # otherwise read whatever tokens sit at the end of the batch.
    if offsets[0] != 0:
        raise ValueError("cu_seqlens_q must start at 0")
    if offsets[-1] != num_tokens:
        raise ValueError(f"cu_seqlens_q ends at {offsets[-1]} but there are {num_tokens} tokens")

    lengths = [after - before for before, after in zip(offsets, offsets[1:], strict=False)]
    if lengths != seq_lens:
        raise ValueError(f"cu_seqlens_q differences {lengths} disagree with seq_lens {seq_lens}")

    # S >= L follows from causality: a pass cannot compute more tokens than it may
    # attend over, and a kernel given S < L masks every query in the overhang to
    # nothing.
    if any(context < length for context, length in zip(context_lens, seq_lens, strict=True)):
        raise ValueError(f"context_lens {context_lens} must be >= seq_lens {seq_lens} elementwise")
    if any(length < 1 for length in seq_lens):
        raise ValueError(f"every sequence must contribute a token, got {seq_lens}")

    if num_slots is not None and num_slots != num_tokens:
        raise ValueError(
            f"slot_mapping has {num_slots} entries for {num_tokens} tokens; "
            "every token written this pass needs a slot"
        )
    if num_tables is not None and num_tables != num_sequences:
        raise ValueError(f"block_tables has {num_tables} rows for {num_sequences} sequences")
    if empty_tables:
        raise ValueError(f"sequences {empty_tables} attend over cached tokens with no blocks")


def _pad_tables(tables: list[tuple[int, ...]], device: torch.device | str) -> torch.Tensor:
    """Stack per-sequence block tables into one rectangle, right-padded.

    Rectangular because the kernel indexes it as `block_tables[seq, logical_block]`;
    a ragged tensor of pointers would add a second indirection inside the inner loop.
    The width is the widest table in this batch rather than the maximum sequence
    length, so a batch of short sequences carries a small tensor.
    """
    widest = max((len(table) for table in tables), default=0)
    padded = [list(table) + [PADDING_BLOCK] * (widest - len(table)) for table in tables]
    return torch.tensor(padded, dtype=torch.int32, device=device).reshape(len(tables), widest)


@dataclass(frozen=True)
class ForwardBatch:
    """Everything the model and the kernels need for one ragged forward pass.

    One forward pass, one `ForwardBatch`. It describes a ragged batch: sequences of
    different lengths, some prefilling and some decoding, flattened into a single token
    axis with offsets rather than padded into a rectangle.

    ::

        for a mixed batch [prefill(A, 300), decode(B, 1), decode(C, 1)]:
          input_ids     300 + 1 + 1 = 302 tokens, concatenated
          cu_seqlens_q  [0, 300, 301, 302]   query-token offsets
          seq_lens      [300, 1, 1]          new tokens per sequence   (L)
          context_lens  [300, 512, 47]       total attended tokens     (S)

    Flattened rather than padded: padding a 300-token prefill beside two decode steps
    into a `3 x 300` rectangle wastes 598 token-slots of compute, and the waste grows
    with the length spread — the argument paging makes about memory, applied to
    compute. The paged attention kernels read `cu_seqlens_q` and `context_lens` and
    serve the whole ragged batch in one launch with no host-side per-sequence
    branching, so this object is built in their layout rather than reshaped later.

    `seq_lens` and `context_lens` are separate for the reason `Sequence` separates
    `num_computed_tokens` from `len(sequence)`: `L` is how many tokens this pass
    computes, `S` how many it attends over. They differ whenever a prefix is already
    cached, which is every decode step and every chunk after the first.

    A validated dataclass rather than a bag of tensors: every field indexes another
    one, and getting that wrong yields silently wrong text rather than an exception.
    The invariants are checked once at construction, which is cheap next to a forward
    pass.
    """

    input_ids: torch.Tensor  # int64 [total_tokens], flattened across sequences
    positions: torch.Tensor  # int64 [total_tokens], each sequence's own RoPE positions
    cu_seqlens_q: torch.Tensor  # int32 [num_sequences + 1], exclusive prefix sums
    seq_lens: torch.Tensor  # int32 [num_sequences], L per sequence
    context_lens: torch.Tensor  # int32 [num_sequences], S per sequence
    seq_ids: tuple[int, ...]
    sampling_params: tuple[SamplingParams, ...]

    # Paging metadata from the block manager. Optional: the dense runner has no use for
    # it, and pure-scheduling tests should not need a block manager to build a batch.
    #
    #   slot_mapping  int32 [total_tokens]        where each new K/V is written
    #   block_tables  int32 [num_sequences, max]  right-padded with -1
    #
    # Padding is -1 rather than 0 so an out-of-bounds read is visible: a kernel bounds
    # its walk by `context_lens` and never reads the padding, so a read that does
    # happen is a bug rather than a quiet read of whichever sequence owns block 0.
    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

    # The longest `L` and `S` in the batch. They size the prefill kernel's grid and the
    # decode kernel's split count, so they must be host integers: reading them off
    # `seq_lens` synchronizes, and 28 layers asking means 28 syncs per iteration.
    # `from_scheduled` fills them from the lists it built the batch from; left unset,
    # they are computed once here.
    max_query_len: int | None = None
    max_context_len: int | None = None

    # Set by `from_scheduled`, which already checked the same invariants against its own
    # Python integers. See `_check_metadata` for why that is worth a field.
    checked: bool = False

    def __post_init__(self) -> None:
        if not self.checked:
            count = len(self.seq_ids)
            offsets = self.cu_seqlens_q.tolist()
            # Whether a row holds enough blocks for its context needs the block size,
            # which belongs to the manager, and the attention path checks it against
            # the pool it is about to read. Checkable here: a sequence with a context
            # holds at least one page.
            tables = [] if self.block_tables is None else self.block_tables.tolist()
            _check_metadata(
                num_sequences=count,
                num_params=len(self.sampling_params),
                seq_lens=self.seq_lens.tolist(),
                context_lens=self.context_lens.tolist(),
                offsets=offsets,
                num_tokens=int(self.input_ids.shape[0]),
                num_positions=int(self.positions.shape[0]),
                num_slots=None if self.slot_mapping is None else int(self.slot_mapping.shape[0]),
                num_tables=None if self.block_tables is None else len(tables),
                empty_tables=[
                    index
                    for index, row in enumerate(tables)
                    if not any(block >= 0 for block in row)
                ],
            )

        if self.max_query_len is None:
            object.__setattr__(self, "max_query_len", int(self.seq_lens.max()))
        if self.max_context_len is None:
            object.__setattr__(self, "max_context_len", int(self.context_lens.max()))
        # Computed on the device, so no synchronization: the row indices stay on the
        # GPU, where `index_select` wants them.
        object.__setattr__(self, "_last_row_indices", self.cu_seqlens_q[1:].to(torch.int64) - 1)

    # ---------------------------------------------------------------- properties

    @property
    def last_row_indices(self) -> torch.Tensor:
        """Row of each sequence's last computed token, in the flattened token axis.

        Where the LM head is applied and where a sampled token comes from: the only
        row for a decode step, the end of the chunk for a prefill, which is why a
        mid-prompt chunk's logits are computed and discarded.
        """
        return self._last_row_indices  # type: ignore[attr-defined]

    @property
    def num_sequences(self) -> int:
        return len(self.seq_ids)

    @property
    def total_tokens(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def is_pure_decode(self) -> bool:
        """Every sequence contributes exactly one token.

        The common case: the batch is a rectangle, so the decode kernel's
        `B x H x 1 x D` shape applies with no ragged handling.
        """
        return bool((self.seq_lens == 1).all())

    @property
    def num_prefill_tokens(self) -> int:
        """Tokens belonging to sequences contributing more than one: the chunk work."""
        return int(self.seq_lens[self.seq_lens > 1].sum().item())

    def slice_of(self, index: int) -> slice:
        """Where sequence `index`'s tokens live in the flattened axis."""
        return slice(int(self.cu_seqlens_q[index]), int(self.cu_seqlens_q[index + 1]))

    def describe(self) -> str:
        parts = []
        for i, seq_id in enumerate(self.seq_ids):
            length = int(self.seq_lens[i])
            phase = "decode" if length == 1 else f"prefill {length}"
            parts.append(f"{seq_id}:{phase}/{int(self.context_lens[i])}")
        return f"batch[{self.total_tokens} tokens] " + " ".join(parts)

    # ------------------------------------------------------------------- builders

    @classmethod
    def from_scheduled(
        cls,
        scheduled: Iterable[tuple[Sequence, int]],
        device: torch.device | str = "cpu",
        manager: object | None = None,
    ) -> ForwardBatch:
        """Build from `(sequence, tokens to compute now)` pairs, in one pass.

        The token count is the scheduler's decision, not the sequence's: a 2000-token
        prompt admitted under a 512-token budget contributes 512 here and carries the
        rest in `num_computed_tokens`. Chunking therefore changes the caller, not this
        object.

        Positions start at each sequence's `num_computed_tokens`, which is why RoPE
        takes an explicit position tensor: a chunk's tokens sit at positions 512..1023
        and nothing in the tensor shapes records that.

        Pass `manager` (a `cache.BlockManager`) to fill in the paging metadata.
        Optional so a scheduling test can build a batch without a pool, and typed
        loosely because only two duck-typed methods are used (`slots` and `table`).
        """
        input_ids: list[int] = []
        positions: list[int] = []
        offsets = [0]
        seq_lens: list[int] = []
        context_lens: list[int] = []
        seq_ids: list[int] = []
        params: list[SamplingParams] = []
        slots: list[int] = []
        tables: list[tuple[int, ...]] = []

        for sequence, count in scheduled:
            if count < 1:
                raise ValueError(f"sequence {sequence.seq_id} was scheduled {count} tokens")
            start = sequence.num_computed_tokens
            if start + count > len(sequence):
                raise ValueError(
                    f"sequence {sequence.seq_id} has {len(sequence)} tokens; cannot schedule "
                    f"{count} starting at {start}"
                )

            tokens = sequence.token_ids
            input_ids.extend(tokens[start : start + count])
            positions.extend(range(start, start + count))
            offsets.append(offsets[-1] + count)
            seq_lens.append(count)
            context_lens.append(start + count)
            seq_ids.append(sequence.seq_id)
            params.append(sequence.sampling_params)

            if manager is not None:
                slots.extend(manager.slots(sequence, count))
                tables.append(manager.table(sequence).block_ids)

        _check_metadata(
            num_sequences=len(seq_ids),
            num_params=len(params),
            seq_lens=seq_lens,
            context_lens=context_lens,
            offsets=offsets,
            num_tokens=len(input_ids),
            num_positions=len(positions),
            num_slots=None if manager is None else len(slots),
            num_tables=None if manager is None else len(tables),
            empty_tables=[index for index, table in enumerate(tables) if not table],
        )

        as_int64 = {"dtype": torch.int64, "device": device}
        as_int32 = {"dtype": torch.int32, "device": device}
        return cls(
            input_ids=torch.tensor(input_ids, **as_int64),
            positions=torch.tensor(positions, **as_int64),
            cu_seqlens_q=torch.tensor(offsets, **as_int32),
            seq_lens=torch.tensor(seq_lens, **as_int32),
            context_lens=torch.tensor(context_lens, **as_int32),
            seq_ids=tuple(seq_ids),
            sampling_params=tuple(params),
            slot_mapping=None if manager is None else torch.tensor(slots, **as_int32),
            block_tables=None if manager is None else _pad_tables(tables, device),
            max_query_len=max(seq_lens),
            max_context_len=max(context_lens),
            checked=True,
        )

    @classmethod
    def from_sequences(
        cls,
        sequences: SequenceABC[Sequence],
        device: torch.device | str = "cpu",
    ) -> ForwardBatch:
        """Every sequence contributes all of its uncomputed tokens. No chunking."""
        return cls.from_scheduled(
            ((sequence, sequence.num_uncomputed_tokens) for sequence in sequences), device
        )


# ---------------------------------------- 4. scheduler config and output
#
# `SchedulerConfig` is defined in `config.py` with the other configuration dataclasses
# and re-exported here: the policy it parameterizes lives in this file.


@dataclass
class SchedulerOutput:
    """What one iteration decided to run.

    ``scheduled`` pairs each sequence with how many of its tokens this pass covers:
    1 for a decode step, up to `chunk_size` for a prefill chunk. The pair is the unit
    the rest of the engine works in, since the sequence list alone does not describe a
    ragged batch.
    """

    scheduled: list[tuple[Sequence, int]] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled

    @property
    def total_tokens(self) -> int:
        return sum(count for _, count in self.scheduled)

    @property
    def sequences(self) -> list[Sequence]:
        return [sequence for sequence, _ in self.scheduled]

    @property
    def num_decodes(self) -> int:
        """Scheduled sequences taking a single token — the piggybacked ones."""
        return sum(1 for _, count in self.scheduled if count == 1)

    def tokens_for(self, sequence: Sequence) -> int:
        """This iteration's token count for one sequence, or 0 if it is not in it.

        A linear scan over at most `max_sequences` entries. `Sequence` is a mutable
        dataclass and therefore unhashable, so a dict keyed by it is unavailable, and
        keying by `seq_id` would add a second index to keep consistent.
        """
        for scheduled, count in self.scheduled:
            if scheduled is sequence:
                return count
        return 0

    def batch(self, device: torch.device | str = "cpu") -> ForwardBatch:
        return ForwardBatch.from_scheduled(self.scheduled, device)


# --------------------------------------------- 5. continuous-batching policy


class Scheduler:
    """FCFS waiting and running queues, re-decided every iteration.

    Iteration-level scheduling, following Orca: the batch is re-formed every iteration
    rather than held fixed for a request's lifetime. Under static batching every
    request waits for the longest one, so a batch of sixteen where fifteen want 20
    tokens and one wants 500 spends 96% of its iterations mostly idle. Here a finished
    sequence is replaced by a waiting one at the iteration boundary it finished on.

    No tensors, no model, no device: this answers only what runs next, which keeps
    admission, replacement and preemption testable in milliseconds.
    """

    def __init__(self, config: SchedulerConfig | None = None, manager=None) -> None:
        """`manager` is a `cache.BlockManager`, or None for a scheduler with no memory
        limit.

        Optional so fairness, chunking and budget behaviour can be tested without a
        pool, and therefore without a GPU. With a manager the same policy runs under a
        second admission constraint: a decode step needs a slot, and a slot can be
        unavailable.
        """
        self.config = config or SchedulerConfig()
        self.manager = manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []

    # ------------------------------------------------------------------- queueing

    def add(self, sequence: Sequence) -> None:
        """Enqueue a request. FCFS, so arrival order is service order."""
        if sequence.status is not SequenceStatus.WAITING:
            raise ValueError(f"sequence {sequence.seq_id} is already {sequence.status.value}")
        self.waiting.append(sequence)

    def add_all(self, sequences: list[Sequence]) -> None:
        for sequence in sequences:
            self.add(sequence)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    @property
    def num_unfinished(self) -> int:
        return len(self.waiting) + len(self.running)

    # ------------------------------------------------------------------ scheduling

    def schedule(self) -> SchedulerOutput:
        """Decide this iteration's batch: decodes, then prefill chunks, then arrivals.

        Three passes, in the priority order that defines the fairness policy:

        1. Decodes. Every running sequence past its prompt takes one token. All of
           them together cost less than a single chunk, and they are the requests
           whose caller is already reading output, so queueing them behind a prefill
           is the stall piggyback decoding removes.
        2. Prefill chunks for already-admitted sequences, filling what is left. A
           prefill in progress holds cache until it completes, so finishing it takes
           priority over starting another.
        3. Admissions from the waiting queue, FCFS, with the remaining budget.

        Passes 1 and 2 together form the piggyback: one ragged forward pass carrying a
        300-token chunk beside a dozen single-token decodes, which is what
        `ForwardBatch`'s per-sequence query lengths encode.

        Starvation is bounded without a special case: decodes can only crowd out a
        prefill while `max_sequences` sequences are decoding, each retiring at one
        token per iteration, and with nothing running the budget is untouched, so the
        chunk is never zero-sized and the queue cannot deadlock.

        With a block manager attached there is a second budget behaving differently
        from the token one: exhausting tokens postpones a sequence, exhausting blocks
        requires someone to release theirs. A decode that cannot obtain its next slot
        preempts the newest running sequence and retries, so the oldest requests keep
        making progress under pressure.

        `prefill_priority` runs the same three passes in the opposite order. It is a
        baseline rather than an alternative: a prompt finishes marginally sooner at the
        cost of every decode behind it.
        """
        output = SchedulerOutput()
        budget = self.config.max_batched_tokens

        # Blocks this iteration has already promised, per sequence. Pages are not taken
        # until the runner reserves them, so without this every sequence in the pass
        # would be checked against the same free count: two sequences needing three
        # pages each would both be admitted against four free, and the second would
        # fail inside the forward pass with the first already committed.
        promised: dict[int, int] = {}

        decoding, prefilling = [], []
        for sequence in self.running:
            (prefilling if sequence.is_prefill() else decoding).append(sequence)

        if self.config.prefill_priority:
            budget = self._advance(prefilling, output, budget, promised)
            budget = self._admit(output, budget, promised)
            budget = self._advance(decoding, output, budget, promised)
        else:
            budget = self._advance(decoding, output, budget, promised)
            budget = self._advance(prefilling, output, budget, promised)
            budget = self._admit(output, budget, promised)

        if not output.scheduled and self.has_work and self.manager is not None:
            # No other empty iteration is reachable: the token budget is at least 1 and
            # a queue is non-empty, so something would have been scheduled. This case
            # means the pool cannot back a single sequence of this length, and an empty
            # batch would leave the engine spinning on it indefinitely.
            raise OutOfBlocks(
                f"nothing can run: {len(self.running)} running and {len(self.waiting)} waiting "
                f"sequences with {self.manager.num_free_blocks} of "
                f"{self.manager.num_blocks} blocks free. The pool is too small for even one "
                "sequence at this length — raise num_blocks or shorten the request."
            )
        return output

    # ----------------------------------------------------------------- the passes

    def _advance(
        self,
        sequences: list[Sequence],
        output: SchedulerOutput,
        budget: int,
        promised: dict[int, int],
    ) -> int:
        """Give each already-running sequence its next tokens, in the order given.

        Returns what is left of the budget. It can grow: preempting a victim already
        scheduled this iteration hands its tokens back, hence the budget is threaded
        through rather than kept as a field.
        """
        for sequence in sequences:
            if not sequence.is_prefill() and sequence.num_uncomputed_tokens == 0:
                # Nothing to feed and not finished, so the loop would spin forever.
                # The cause is `commit` being called without the sampled token for a
                # sequence whose prompt was complete; raising here beats an engine that
                # makes no progress.
                raise ValueError(
                    f"sequence {sequence.seq_id} is running with nothing to compute; "
                    "commit() needs the token sampled for it"
                )
            count = self._tokens_for(sequence, budget)
            if count == 0:
                # The budget ran out mid-batch. The sequence keeps its cache and its
                # place; it simply does not advance this iteration.
                continue
            refund = self._make_room_for(sequence, count, output, promised)
            if refund is None:
                continue
            self._schedule(output, sequence, count, promised)
            budget += refund - count
        return budget

    def _admit(self, output: SchedulerOutput, budget: int, promised: dict[int, int]) -> int:
        """Take arrivals off the front of the waiting queue while they fit."""
        while self.waiting and len(self.running) < self.config.max_sequences:
            candidate = self.waiting[0]
            # A prefix-cache hit is resolved here, before the prefill is sized, so the
            # reused prefix is already out of `num_uncomputed_tokens` and the chunk
            # this iteration schedules is only the part the cache did not hold.
            if self.manager is not None:
                self.manager.maybe_apply_prefix_cache(candidate)
            count = self._tokens_for(candidate, budget)

            if count == 0:
                if output.scheduled:
                    break  # no room left this iteration; it goes next time
                # Chunking is off, nothing else is running, and the prompt does not fit
                # whole in the budget. Refusing it would deadlock the queue, so it runs
                # alone and overruns the budget: the head-of-line stall chunking
                # removes, kept reachable so the tests can exhibit it.
                count = candidate.num_uncomputed_tokens

            # Admission declines memory pressure rather than resolving it: a request
            # that has not started holds nothing and costs nothing to leave queued, so
            # preempting a running sequence for it would discard completed work.
            if not self._fits(candidate, count, promised):
                break

            self.waiting.popleft()
            candidate.set_status(SequenceStatus.RUNNING)
            self.running.append(candidate)
            self._schedule(output, candidate, count, promised)
            budget -= count
        return budget

    # ------------------------------------------------------------ memory pressure

    def _schedule(
        self, output: SchedulerOutput, sequence: Sequence, count: int, promised: dict[int, int]
    ) -> None:
        """Put a sequence in the batch and record the pages it will take."""
        output.scheduled.append((sequence, count))
        if self.manager is not None:
            promised[sequence.seq_id] = self.manager.blocks_needed(sequence, count)

    def _fits(self, sequence: Sequence, count: int, promised: dict[int, int]) -> bool:
        """Whether the pool can back `count` more tokens of this sequence.

        Measured against the free count minus this iteration's outstanding promises,
        which is what the runner will find when it reserves them.
        """
        if self.manager is None:
            return True
        available = self.manager.num_free_blocks - sum(promised.values())
        return self.manager.blocks_needed(sequence, count) <= available

    def _make_room_for(
        self, sequence: Sequence, count: int, output: SchedulerOutput, promised: dict[int, int]
    ) -> int | None:
        """Preempt until `count` tokens of `sequence` fit, or give up on it.

        Victims come from the back of the running list, most recently admitted first,
        and each releases every block it holds. Returns the token budget freed by
        un-scheduling them, or None when the only remaining victim is `sequence`
        itself: the pool cannot hold one sequence of this length, which the caller
        raises rather than looping on.
        """
        refund = 0
        while not self._fits(sequence, count, promised):
            victim = next((s for s in reversed(self.running) if s is not sequence), None)
            if victim is None:
                return None
            # A victim already scheduled this iteration must come back out: it is about
            # to lose the blocks the forward pass would have written into.
            refund += output.tokens_for(victim)
            promised.pop(victim.seq_id, None)
            output.scheduled = [pair for pair in output.scheduled if pair[0] is not victim]
            self.preempt(victim)
            output.preempted.append(victim)
        return refund

    def _tokens_for(self, sequence: Sequence, budget: int) -> int:
        """How many of this sequence's tokens fit in what is left of the budget.

        With chunking on, a partial count is valid and the sequence resumes next
        iteration from `num_computed_tokens`. With it off the answer is all or nothing,
        since a partial count is chunking: splitting a prefill without matching
        positions and a shifted mask produces wrong text, not slow text. Zero defers
        the sequence to a later iteration.
        """
        wanted = sequence.num_uncomputed_tokens
        if sequence.proposed_token_ids:
            # A speculative group is atomic: the pending token and all `k` proposals go
            # through one pass or none do. Splitting it is wrong rather than slow — the
            # draft's `q` was computed for one specific context, and the verifier reads
            # a sequence's `k + 1` distributions as consecutive rows of a single
            # forward, so a partial group misaligns both.
            return wanted if wanted <= budget else 0
        if not self.config.enable_chunked_prefill:
            return wanted if wanted <= budget else 0
        return min(wanted, self.config.chunk_size, max(budget, 0))

    # ------------------------------------------------------------------ committing

    def commit(
        self,
        output: SchedulerOutput,
        tokens: list[int] | None = None,
        verified: frozenset[int] | set[int] = frozenset(),
    ) -> list[Sequence]:
        """Record an iteration's results and retire whatever finished.

        Called with the sampled token per scheduled sequence, or with none at all for a
        prefill chunk that did not reach the end of its prompt. Still-prefilling
        sequences never receive a token: their forward pass produced logits for a
        mid-prompt position, and sampling from those would invent a token the prompt
        then contradicts.

        ``verified`` names sequences a speculative step has already settled. Those
        emitted between 1 and `k + 1` tokens rather than exactly one, and their
        computed count already reflects a rollback this method has no view of, so their
        tokens and counters are left untouched; only the finish check applies.

        Returns the sequences that finished, in the iteration they finished in, which
        frees their slot for a waiting request immediately.
        """
        sampled = tokens if tokens is not None else [None] * len(output.scheduled)
        if len(sampled) != len(output.scheduled):
            raise ValueError(
                f"got {len(sampled)} tokens for {len(output.scheduled)} scheduled sequences"
            )

        for (sequence, count), token in zip(output.scheduled, sampled, strict=True):
            if sequence.seq_id in verified:
                # A verified speculative step has already done both halves:
                # `Sequence.accept` appended the surviving tokens and set the computed
                # count to match, since only it knows how many proposals were kept.
                # Advancing again would double-count them.
                continue
            sequence.advance(count)
            if sequence.is_prefill():
                continue  # mid-prompt: no token to take, and none was sampled
            if token is not None:
                sequence.append_token(int(token))

        just_finished = [sequence for sequence in self.running if sequence.is_done()]
        for sequence in just_finished:
            sequence.set_status(SequenceStatus.FINISHED)
            self.running.remove(sequence)
            self.finished.append(sequence)

        return just_finished

    def preempt(self, sequence: Sequence) -> None:
        """Evict a running sequence, to be recomputed when it is readmitted.

        Recompute rather than swap-to-host: blocks return to the pool immediately and
        the sequence re-prefills over its prompt and its output so far. That costs the
        compute already spent and keeps the engine free of a swap path; see
        `Sequence.reset_for_recompute`.

        Requeued at the front, not the back. It has been served before and its caller
        has seen output, so placing it behind requests that have never run would make
        preemption a demotion, and under sustained pressure a sequence could be
        preempted and requeued repeatedly without finishing.
        """
        if sequence not in self.running:
            raise ValueError(f"sequence {sequence.seq_id} is not running")
        self.running.remove(sequence)
        if self.manager is not None:
            self.manager.free(sequence)
        sequence.reset_for_recompute()
        self.waiting.appendleft(sequence)

    def __repr__(self) -> str:
        return f"Scheduler(waiting={len(self.waiting)}, running={len(self.running)})"


# ------------------------------------------------------ 6. dense-cache oracle


class DenseModelRunner:
    """Executes a `SchedulerOutput` against the dense per-sequence cache.

    A `DenseKvCache` is one contiguous `B x H x S x D` tensor per sequence, and two
    sequences at different lengths cannot share one, so a scheduled batch of `n`
    sequences becomes `n` separate forward passes. Continuous batching then buys
    scheduling fairness and no GPU efficiency: the device sees the same
    one-sequence-at-a-time work as an unbatched dense-cache run, plus `n` launches.

    Two ways out, of which one is viable:

    * Pad every sequence to the longest in the batch. A 300-token prefill beside two
      decode steps becomes a `3 x 300` rectangle — 598 wasted token slots, and the
      waste grows with the length spread.
    * Page the cache, so K and V live in fixed-size blocks that any sequence can hold
      in any order, and pass the kernel `cu_seqlens_q` and `context_lens` so one launch
      covers the ragged batch. This is what the paged attention kernel does.

    This class is retained beside the paged runner as the oracle it is diffed against:
    the scheduler tests assert that batching, chunking and preemption change no tokens,
    and this runner is the slow, per-sequence executor those identities are measured
    on. It also demonstrates the motivation for paging, measured in the scheduler
    benchmark.
    """

    def __init__(self, model, device: torch.device | str | None = None) -> None:
        self.model = model
        self.device = torch.device(device) if device else model.weights["embedding"].device
        self.caches: dict[int, list[KvCache]] = {}

    def caches_for(self, sequence: Sequence) -> list[KvCache]:
        """One cache set per sequence, created on first sight."""
        if sequence.seq_id not in self.caches:
            self.caches[sequence.seq_id] = self.model.create_kv_cache()
        return self.caches[sequence.seq_id]

    def free(self, sequence: Sequence) -> None:
        self.caches.pop(sequence.seq_id, None)

    def execute(self, output: SchedulerOutput) -> torch.Tensor:
        """Run every scheduled sequence and return one logits row each.

        ::

            returns: num_scheduled x V   (the last position of each sequence)

        The row is the last position the sequence computed: its only position for a
        decode step, the end of the chunk for a prefill. Only the final chunk's row is
        ever sampled from, but returning one uniformly means the caller need not know
        which chunk it is looking at.
        """
        rows = []
        for sequence, count in output.scheduled:
            caches = self.caches_for(sequence)
            start = sequence.num_computed_tokens

            if start == 0 and caches[0].offset > 0:
                # Readmitted after preemption: the sequence recomputes from token zero,
                # so the stale cache must be dropped. This is the concrete cost of
                # recompute-based preemption.
                for cache in caches:
                    cache.reset()
            elif caches[0].offset != start:
                raise ValueError(
                    f"sequence {sequence.seq_id} has {start} computed tokens but its cache "
                    f"holds {caches[0].offset}; the two have drifted apart"
                )

            tokens = sequence.token_ids[start : start + count]
            input_ids = torch.tensor([tokens], dtype=torch.int64, device=self.device)
            positions = torch.arange(start, start + count, device=self.device)

            logits = self.model(input_ids, caches, positions=positions, last_only=True)
            rows.append(logits[0, -1, :])

        return torch.stack(rows)

    def sample_tokens(self, output: SchedulerOutput, logits: torch.Tensor) -> list[int]:
        """Sample one token per scheduled sequence, honouring per-row parameters.

        Sequences still mid-prefill get a token sampled and discarded by
        `Scheduler.commit`: one wasted row of a `V`-wide sample instead of a branch in
        the hot path, which keeps this a single batched call.
        """
        params = [sequence.sampling_params for sequence, _ in output.scheduled]
        if all(parameter.is_greedy for parameter in params):
            return logits.argmax(dim=-1).tolist()
        return sample(logits, params).tolist()
