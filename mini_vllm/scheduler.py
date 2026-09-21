"""Continuous batching: sequences, ragged batches, and the per-iteration policy."""

from __future__ import annotations

import enum
import itertools
from collections import deque
from collections.abc import Iterable
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field

import torch

from mini_vllm.cache import BlockTable, KvCache, OutOfBlocks
from mini_vllm.config import SamplingParams, SchedulerConfig
from mini_vllm.ops import sample

__all__ = [
    "DenseModelRunner",
    "ForwardBatch",
    "Scheduler",
    "SchedulerConfig",
    "SchedulerOutput",
    "Sequence",
    "SequenceStatus",
]


class SequenceStatus(enum.Enum):
    """Where a request is in its life cycle."""

    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


# Legal transitions as data rather than `if` statements spread through the scheduler.
_LEGAL_TRANSITIONS: dict[SequenceStatus, frozenset[SequenceStatus]] = {
    SequenceStatus.WAITING: frozenset({SequenceStatus.RUNNING, SequenceStatus.FINISHED}),
    SequenceStatus.RUNNING: frozenset(
        {SequenceStatus.PREEMPTED, SequenceStatus.FINISHED, SequenceStatus.WAITING}
    ),
    SequenceStatus.PREEMPTED: frozenset({SequenceStatus.RUNNING, SequenceStatus.FINISHED}),
    SequenceStatus.FINISHED: frozenset(),  # terminal: its blocks have been freed
}

_next_seq_id = itertools.count()


@dataclass
class Sequence:
    """A request in flight: its tokens, how much of them is cached, and its pages.

    `token_ids` is prompt + output + proposals, and `num_computed_tokens` says how much
    of it the KV cache holds; their difference is what chunked prefill works in.
    """

    prompt_token_ids: list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    max_tokens: int = 16
    eos_token_id: int | None = None
    # Qwen3 ends a turn with `<|im_end|>` or `<|endoftext|>`, so one id is not enough.
    stop_token_ids: tuple[int, ...] = ()
    seq_id: int = field(default_factory=lambda: next(_next_seq_id))

    output_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    status: SequenceStatus = SequenceStatus.WAITING

    # Speculated but unverified: kept out of the output so a guess cannot finish a request.
    proposed_token_ids: list[int] = field(default_factory=list)

    # The request a parallel-sampling branch split from; None for an ordinary request.
    parent_id: int | None = None

    # The single home for the sequence's pages: the block manager keeps no registry.
    block_table: BlockTable | None = None

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("a sequence needs at least one prompt token to run a forward pass")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")

    @property
    def token_ids(self) -> list[int]:
        """Prompt, output, then any live proposals: what the model attends over."""
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

    def is_prefill(self) -> bool:
        """True while any prompt token has yet to be computed, chunked or not."""
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

    def append_token(self, token_id: int) -> None:
        """Record a sampled token; its KV is not cached, so num_computed_tokens stands."""
        if self.status is SequenceStatus.FINISHED:
            raise ValueError(f"sequence {self.seq_id} is finished and cannot take more tokens")
        self.output_token_ids.append(token_id)

    @property
    def num_proposed_tokens(self) -> int:
        return len(self.proposed_token_ids)

    def propose(self, token_ids: Iterable[int]) -> None:
        """Attach a draft model's speculated tokens, pending verification."""
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
        """Drop the proposals without committing any, returning how many were dropped."""
        dropped = len(self.proposed_token_ids)
        self.proposed_token_ids = []
        return dropped

    def accept(self, token_ids: list[int], num_accepted: int) -> int:
        """Commit a verified step's tokens, returning how many cache slots to give back.

        The verifying pass computed all k + 1 positions, but only the accepted proposals
        plus the previously pending token remain computed.
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

        # Append until a stop token or the limit ends it, counting surviving proposals.
        kept_proposals = 0
        for index, token in enumerate(token_ids):
            self.output_token_ids.append(int(token))
            if index < num_accepted:
                kept_proposals += 1
            if self.is_done():
                break

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
        """A new branch sharing this sequence's prompt and computed count, for n > 1."""
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
        """Drop everything cached, keeping the tokens: preemption is by recomputation."""
        if self.block_table is not None:
            raise ValueError(
                f"sequence {self.seq_id} still holds {self.block_table.num_blocks} blocks; "
                "free them through the block manager before resetting it"
            )
        self.set_status(SequenceStatus.PREEMPTED)
        self.num_computed_tokens = 0
        # Unverified proposals were never part of the request, so they do not survive.
        self.proposed_token_ids = []

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, {self.status.value}, "
            f"{self.num_computed_tokens}/{len(self)} computed, "
            f"{self.num_output_tokens}/{self.max_tokens} out)"
        )


# What an unused block-table entry holds: impossible, not a silent alias for block 0.
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
    """Check one batch's metadata for consistency, in plain integers rather than tensors."""
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

    # The offsets must describe the token axis they index, or a short batch reads garbage.
    if offsets[0] != 0:
        raise ValueError("cu_seqlens_q must start at 0")
    if offsets[-1] != num_tokens:
        raise ValueError(f"cu_seqlens_q ends at {offsets[-1]} but there are {num_tokens} tokens")

    lengths = [after - before for before, after in itertools.pairwise(offsets)]
    if lengths != seq_lens:
        raise ValueError(f"cu_seqlens_q differences {lengths} disagree with seq_lens {seq_lens}")

    # S >= L is causality: a pass cannot compute more tokens than it may attend over.
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
    """Stack per-sequence block tables into one right-padded rectangle for the kernel."""
    widest = max((len(table) for table in tables), default=0)
    padded = [list(table) + [PADDING_BLOCK] * (widest - len(table)) for table in tables]
    return torch.tensor(padded, dtype=torch.int32, device=device).reshape(len(tables), widest)


@dataclass(frozen=True)
class ForwardBatch:
    """Everything the model and the kernels need for one ragged forward pass.

    For a mixed batch [prefill(A, 300), decode(B, 1), decode(C, 1)]: 302 flattened
    tokens, cu_seqlens_q [0, 300, 301, 302], seq_lens [300, 1, 1] and context_lens
    [300, 512, 47]. Flattened rather than padded, in the paged kernels' own layout.
    """

    input_ids: torch.Tensor  # int64 [total_tokens], flattened across sequences
    positions: torch.Tensor  # int64 [total_tokens], each sequence's own RoPE positions
    cu_seqlens_q: torch.Tensor  # int32 [num_sequences + 1], exclusive prefix sums
    seq_lens: torch.Tensor  # int32 [num_sequences], L per sequence
    context_lens: torch.Tensor  # int32 [num_sequences], S per sequence
    seq_ids: tuple[int, ...]
    sampling_params: tuple[SamplingParams, ...]

    # Paging metadata: the slot per token, and block tables padded with PADDING_BLOCK.
    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

    # The longest L and S: host integers, since reading a device tensor would synchronize.
    max_query_len: int | None = None
    max_context_len: int | None = None

    # Set by `from_scheduled`, which already checked these invariants on plain integers.
    checked: bool = False

    def __post_init__(self) -> None:
        if not self.checked:
            count = len(self.seq_ids)
            offsets = self.cu_seqlens_q.tolist()
            # Only emptiness is checkable here; sufficiency needs the manager's block size.
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
        # Computed on the device, so no synchronization and no host round trip.
        object.__setattr__(self, "_last_row_indices", self.cu_seqlens_q[1:].to(torch.int64) - 1)

    @property
    def last_row_indices(self) -> torch.Tensor:
        """Row of each sequence's last computed token: where the LM head is applied."""
        return self._last_row_indices  # type: ignore[attr-defined]

    @property
    def num_sequences(self) -> int:
        return len(self.seq_ids)

    @property
    def total_tokens(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def is_pure_decode(self) -> bool:
        """Every sequence contributes exactly one token: the common case."""
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

    @classmethod
    def from_scheduled(
        cls,
        scheduled: Iterable[tuple[Sequence, int]],
        device: torch.device | str = "cpu",
        manager: object | None = None,
    ) -> ForwardBatch:
        """Build from (sequence, tokens to compute now) pairs, in one pass.

        Positions start at each sequence's num_computed_tokens; pass a manager to fill in
        the paging metadata.
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


@dataclass
class SchedulerOutput:
    """What one iteration decided to run: each sequence with how many tokens it takes."""

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
        """This iteration's token count for one sequence, or 0 if it is not in it."""
        for scheduled, count in self.scheduled:
            if scheduled is sequence:
                return count
        return 0

    def batch(self, device: torch.device | str = "cpu") -> ForwardBatch:
        return ForwardBatch.from_scheduled(self.scheduled, device)


class Scheduler:
    """FCFS waiting and running queues, re-decided every iteration, Orca-style.

    No tensors, no model, no device: a scheduling decision may change timing, never
    output.
    """

    def __init__(self, config: SchedulerConfig | None = None, manager=None) -> None:
        """manager is a cache.BlockManager, or None for a scheduler with no memory limit."""
        self.config = config or SchedulerConfig()
        self.manager = manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []

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

    def schedule(self) -> SchedulerOutput:
        """Decide this iteration's batch: decodes, then prefill chunks, then arrivals.

        Decodes first because together they cost less than one chunk and their callers
        are already reading output; `prefill_priority` runs the same passes in the
        opposite order as the pre-chunking baseline.
        """
        output = SchedulerOutput()
        budget = self.config.max_batched_tokens

        # Blocks promised so far, so two sequences cannot be admitted against one page.
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
            # The budget is >= 1 and a queue is non-empty, so only the pool can be at fault.
            raise OutOfBlocks(
                f"nothing can run: {len(self.running)} running and {len(self.waiting)} waiting "
                f"sequences with {self.manager.num_free_blocks} of "
                f"{self.manager.num_blocks} blocks free. The pool is too small for even one "
                "sequence at this length — raise num_blocks or shorten the request."
            )
        return output

    def _advance(
        self,
        sequences: list[Sequence],
        output: SchedulerOutput,
        budget: int,
        promised: dict[int, int],
    ) -> int:
        """Give each already-running sequence its next tokens, returning what is left of
        the budget; it can grow, since preempting a scheduled victim hands its tokens back."""
        for sequence in sequences:
            if not sequence.is_prefill() and sequence.num_uncomputed_tokens == 0:
                # Nothing to feed and not finished, so the loop would spin forever.
                raise ValueError(
                    f"sequence {sequence.seq_id} is running with nothing to compute; "
                    "commit() needs the token sampled for it"
                )
            count = self._tokens_for(sequence, budget)
            if count == 0:
                continue  # the budget ran out; the sequence keeps its cache and its place
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
            # Resolved before the prefill is sized, so only the uncached part is scheduled.
            if self.manager is not None:
                self.manager.maybe_apply_prefix_cache(candidate)
            count = self._tokens_for(candidate, budget)

            if count == 0:
                if output.scheduled:
                    break  # no room left this iteration; it goes next time
                # Chunking off and nothing running: it overruns the budget, not deadlocks.
                count = candidate.num_uncomputed_tokens

            # A queued request holds nothing, so preempting for it would discard work.
            if not self._fits(candidate, count, promised):
                break

            self.waiting.popleft()
            candidate.set_status(SequenceStatus.RUNNING)
            self.running.append(candidate)
            self._schedule(output, candidate, count, promised)
            budget -= count
        return budget

    def _schedule(
        self, output: SchedulerOutput, sequence: Sequence, count: int, promised: dict[int, int]
    ) -> None:
        """Put a sequence in the batch and record the pages it will take."""
        output.scheduled.append((sequence, count))
        if self.manager is not None:
            promised[sequence.seq_id] = self.manager.blocks_needed(sequence, count)

    def _fits(self, sequence: Sequence, count: int, promised: dict[int, int]) -> bool:
        """Whether the pool can back `count` more tokens, net of outstanding promises."""
        if self.manager is None:
            return True
        available = self.manager.num_free_blocks - sum(promised.values())
        return self.manager.blocks_needed(sequence, count) <= available

    def _make_room_for(
        self, sequence: Sequence, count: int, output: SchedulerOutput, promised: dict[int, int]
    ) -> int | None:
        """Preempt from the back of the running list until count tokens fit, returning the
        budget that freed, or None when the only remaining victim is sequence itself."""
        refund = 0
        while not self._fits(sequence, count, promised):
            victim = next((s for s in reversed(self.running) if s is not sequence), None)
            if victim is None:
                return None
            # A victim scheduled this iteration must come back out with its blocks.
            refund += output.tokens_for(victim)
            promised.pop(victim.seq_id, None)
            output.scheduled = [pair for pair in output.scheduled if pair[0] is not victim]
            self.preempt(victim)
            output.preempted.append(victim)
        return refund

    def _tokens_for(self, sequence: Sequence, budget: int) -> int:
        """How many of this sequence's tokens fit in what is left of the budget."""
        wanted = sequence.num_uncomputed_tokens
        if sequence.proposed_token_ids:
            # A speculative group is atomic: the verifier reads its k + 1 rows together.
            return wanted if wanted <= budget else 0
        if not self.config.enable_chunked_prefill:
            return wanted if wanted <= budget else 0
        return min(wanted, self.config.chunk_size, max(budget, 0))

    def commit(
        self,
        output: SchedulerOutput,
        tokens: list[int] | None = None,
        verified: frozenset[int] | set[int] = frozenset(),
    ) -> list[Sequence]:
        """Record an iteration's results and retire whatever finished.

        Still-prefilling sequences never receive a token, and `verified` names sequences
        a speculative step already settled, so only the finish check applies to them.
        """
        sampled = tokens if tokens is not None else [None] * len(output.scheduled)
        if len(sampled) != len(output.scheduled):
            raise ValueError(
                f"got {len(sampled)} tokens for {len(output.scheduled)} scheduled sequences"
            )

        for (sequence, count), token in zip(output.scheduled, sampled, strict=True):
            if sequence.seq_id in verified:
                continue  # `Sequence.accept` already appended and advanced this one
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
        """Evict a running sequence, requeued at the front to be recomputed later."""
        if sequence not in self.running:
            raise ValueError(f"sequence {sequence.seq_id} is not running")
        self.running.remove(sequence)
        if self.manager is not None:
            self.manager.free(sequence)
        sequence.reset_for_recompute()
        self.waiting.appendleft(sequence)

    def __repr__(self) -> str:
        return f"Scheduler(waiting={len(self.waiting)}, running={len(self.running)})"


class DenseModelRunner:
    """Executes a SchedulerOutput against the dense per-sequence cache: one pass per
    sequence, and the oracle the paged runner is diffed against."""

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
        """Run every scheduled sequence, returning num_scheduled x V: its last position."""
        rows = []
        for sequence, count in output.scheduled:
            caches = self.caches_for(sequence)
            start = sequence.num_computed_tokens

            if start == 0 and caches[0].offset > 0:
                # Readmitted after preemption: drop the stale cache and recompute.
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
        """Sample one token per scheduled sequence, honouring per-row parameters."""
        params = [sequence.sampling_params for sequence, _ in output.scheduled]
        if all(parameter.is_greedy for parameter in params):
            return logits.argmax(dim=-1).tolist()
        return sample(logits, params).tolist()
