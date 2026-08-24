"""Continuous batching, chunked prefill, piggyback decoding.

Iteration-level scheduling, following Orca: the batch is re-formed every iteration
rather than held fixed for a request's lifetime. Under static batching every request
waits for the longest one, so a batch of sixteen where fifteen want 20 tokens and one
wants 500 spends 96% of its iterations mostly idle. Here a finished sequence is
replaced by a waiting one at the iteration boundary it finished on.

Governing invariant: a scheduling decision may change timing, never output. A
sequence run alone, run beside fifteen others, or preempted and recomputed must emit
identical tokens; `test_scheduler.py` asserts that against single-sequence runs.

Two policies make the batch genuinely ragged: a prompt longer than `chunk_size` is
spread over iterations, and whatever budget a chunk leaves is filled with pending
decode steps from other sequences. Both are decided in :meth:`Scheduler.schedule`
alone, since the model takes explicit RoPE positions and an offset causal mask.

Two objects, split along the policy/execution line:

* :class:`Scheduler` is pure policy — queues, budget, status — with no tensors, so
  admission under a tight budget, same-iteration replacement, and preemption are
  testable without a GPU.
* :class:`DenseModelRunner` executes those decisions and is what `PagedModelRunner`
  replaces. A dense per-sequence cache cannot be batched, so it issues one forward
  pass per scheduled sequence and the batch buys no GPU efficiency; that is the
  motivation for paging, measured in
  `test_scheduler.py::test_the_dense_cache_cannot_actually_batch`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch

from mini_vllm.block.block_pool import OutOfBlocks
from mini_vllm.kv_cache import KvCache
from mini_vllm.sampler import sample
from mini_vllm.serve.batch import ForwardBatch
from mini_vllm.serve.sequence import Sequence, SequenceStatus

__all__ = ["DenseModelRunner", "Scheduler", "SchedulerConfig", "SchedulerOutput"]


@dataclass(frozen=True)
class SchedulerConfig:
    """Admission limits for one iteration.

    ``max_batched_tokens`` is the compute budget: how many token-positions one
    forward pass may cover. ``max_sequences`` is the memory-and-overhead budget on
    how many requests may be in flight. ``chunk_size`` bounds a single prefill's
    share of an iteration.

    ``enable_chunked_prefill`` defaults on; turning it off gives the tests their
    reference, the same prompt in a single pass, which must produce the same logits as
    the chunked run.

    ``prefill_priority`` inverts the pass order to prompts before decode steps, the
    policy vLLM shipped before chunked prefill and the baseline the benchmarks measure
    against. Off by default: a prompt large enough to consume the budget then leaves
    nothing for sequences a caller is already reading, so their next token waits for
    the whole prompt. Kept reachable so the cost is measurable.
    """

    max_batched_tokens: int = 2048
    max_sequences: int = 16
    chunk_size: int = 512
    enable_chunked_prefill: bool = True
    prefill_priority: bool = False

    def __post_init__(self) -> None:
        if self.max_batched_tokens < 1:
            raise ValueError(f"max_batched_tokens must be >= 1, got {self.max_batched_tokens}")
        if self.max_sequences < 1:
            raise ValueError(f"max_sequences must be >= 1, got {self.max_sequences}")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")


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


class Scheduler:
    """FCFS waiting and running queues, re-decided every iteration.

    No tensors, no model, no device: this answers only what runs next, which keeps
    admission, replacement and preemption testable in milliseconds.
    """

    def __init__(self, config: SchedulerConfig | None = None, manager=None) -> None:
        """`manager` is a `BlockManager`, or None for a scheduler with no memory limit.

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

    This class is retained beside the paged runner as the oracle it is diffed against.
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
