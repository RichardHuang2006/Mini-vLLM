"""Steps 4.3 and 4.4 — continuous batching, chunked prefill, piggyback decoding.

The idea, from Orca and [§7.1](./DESIGN.md#71-iteration-level-scheduling): re-form
the batch **every iteration** instead of holding it fixed for a request's lifetime.
Static batching makes every request in a batch wait for the longest one, so a batch
of sixteen where fifteen want 20 tokens and one wants 500 spends 96% of its
iterations mostly idle. Here a finished sequence is replaced by a waiting one at the
same iteration boundary it finished on.

The invariant that makes it testable: **a scheduling decision may change timing,
never output.** Whether a sequence ran alone, or beside fifteen others, or was
preempted and recomputed, its tokens must come out identical. Every test in
`test_scheduler.py` asserts that against a single-sequence run.

[Step 4.4] adds the two policies that make the batch genuinely ragged: a prompt
longer than `chunk_size` is spread over iterations ([§7.2](./DESIGN.md#72-chunked-prefill)),
and whatever budget a chunk leaves is filled with pending decode steps from other
sequences ([§7.3](./DESIGN.md#73-stall-free-piggyback-decoding)). Both are decided
in :meth:`Scheduler.schedule` alone — the model already took explicit RoPE positions
and an offset causal mask from Step 2.2, which is what made this a scheduler change
instead of a rewrite.

Two objects live here, deliberately split:

* :class:`Scheduler` is pure policy — queues, budget, status — with no tensors in
  it at all, so its decisions can be tested without a GPU.
* :class:`DenseModelRunner` executes those decisions, and is the part
  [Step 4.7](../block/block_manager.py) replaces. It is also where this step's
  wall is: a dense per-sequence cache **cannot be batched**, so it runs one forward
  pass per sequence and the "batch" buys nothing. That is not a shortcoming of the
  scheduler; it is the reason paging exists, and it is measured in
  `test_scheduler.py::test_the_dense_cache_cannot_actually_batch`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch

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
    share of an iteration ([§7.2](./DESIGN.md#72-chunked-prefill)).

    ``enable_chunked_prefill`` defaults on as of [Step 4.4] and exists as a flag
    because turning it off is how the tests get their reference: the same prompt in
    one pass, which must produce the same logits as the chunked run.
    """

    max_batched_tokens: int = 2048
    max_sequences: int = 16
    chunk_size: int = 512
    enable_chunked_prefill: bool = True

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
    1 for a decode step, up to `chunk_size` for a prefill chunk. The pairing is the
    unit the rest of the engine works in, because "which sequences" alone is not
    enough to build a ragged batch.
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
        dataclass and therefore unhashable, so a dict keyed by it is not available —
        and keying by `seq_id` would be a second index to keep true.
        """
        for scheduled, count in self.scheduled:
            if scheduled is sequence:
                return count
        return 0

    def batch(self, device: torch.device | str = "cpu") -> ForwardBatch:
        return ForwardBatch.from_scheduled(self.scheduled, device)


class Scheduler:
    """FCFS waiting and running queues, re-decided every iteration.

    No tensors, no model, no device. The scheduler's job is to answer "what runs
    next", and keeping it free of execution is what makes the interesting cases —
    admission under a tight budget, a finished sequence replaced in the same
    iteration, preemption — testable in milliseconds.
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()
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

        Three passes, in a priority order that is the whole of the fairness story:

        1. **Decodes.** Every running sequence past its prompt takes one token. They
           are one token each, so all of them together cost less than a single
           chunk, and they are the requests with a caller already reading output —
           making them wait behind a prefill is what
           [§7.3](./DESIGN.md#73-stall-free-piggyback-decoding) calls the stall.
        2. **Prefill chunks** for sequences already admitted, filling what is left.
           A prefill that has begun holds cache and will not release it until it
           finishes, so finishing it beats starting another.
        3. **Admissions** from the waiting queue, FCFS, with whatever budget remains.

        Passes 1 and 2 together are the piggyback: one ragged forward pass carrying
        a 300-token chunk beside a dozen single-token decodes, which is where
        `ForwardBatch`'s per-sequence query lengths stop being decoration.

        Starvation is bounded without a special case. Decodes can only crowd out a
        prefill while `max_sequences` sequences are decoding, each of which is
        finishing at one token per iteration; and when nothing is running the budget
        is untouched, so the chunk is never zero-sized and the queue cannot deadlock.
        """
        output = SchedulerOutput()
        budget = self.config.max_batched_tokens

        decoding, prefilling = [], []
        for sequence in self.running:
            (prefilling if sequence.is_prefill() else decoding).append(sequence)

        for sequence in decoding:
            if sequence.num_uncomputed_tokens == 0:
                # Nothing to feed it and it has not finished, so the loop would spin
                # forever. It means `commit` was called without the sampled token for
                # a sequence that had finished its prompt — worth saying plainly here
                # rather than as an engine that quietly makes no progress.
                raise ValueError(
                    f"sequence {sequence.seq_id} is running with nothing to compute; "
                    "commit() needs the token sampled for it"
                )
            count = self._tokens_for(sequence, budget)
            if count == 0:
                # The budget ran out mid-batch. The sequence keeps its cache and its
                # place; it simply does not advance this iteration.
                continue
            output.scheduled.append((sequence, count))
            budget -= count

        for sequence in prefilling:
            count = self._tokens_for(sequence, budget)
            if count == 0:
                continue
            output.scheduled.append((sequence, count))
            budget -= count

        while self.waiting and len(self.running) < self.config.max_sequences:
            candidate = self.waiting[0]
            count = self._tokens_for(candidate, budget)

            if count == 0:
                if output.scheduled:
                    break  # no room left this iteration; it goes next time
                # Nothing else is running and this prompt does not fit whole in the
                # budget, with chunking off. Refusing it would deadlock the queue, so
                # it runs alone and overruns — the head-of-line stall that chunking
                # exists to remove, kept reachable so the tests can show it.
                count = candidate.num_uncomputed_tokens

            self.waiting.popleft()
            candidate.set_status(SequenceStatus.RUNNING)
            self.running.append(candidate)
            output.scheduled.append((candidate, count))
            budget -= count

        return output

    def _tokens_for(self, sequence: Sequence, budget: int) -> int:
        """How many of this sequence's tokens fit in what is left of the budget.

        With chunking on, a partial count is the answer and the sequence resumes
        next iteration from `num_computed_tokens`. With it off the answer is all or
        nothing, because handing back a partial count *is* chunking: a prefill split
        without matching positions and a shifted mask produces wrong text, not slow
        text. Zero means "not this iteration".
        """
        wanted = sequence.num_uncomputed_tokens
        if not self.config.enable_chunked_prefill:
            return wanted if wanted <= budget else 0
        return min(wanted, self.config.chunk_size, max(budget, 0))

    # ------------------------------------------------------------------ committing

    def commit(self, output: SchedulerOutput, tokens: list[int] | None = None) -> list[Sequence]:
        """Record an iteration's results and retire whatever finished.

        Called with the sampled token per *scheduled* sequence, or with none at all
        for a prefill chunk that did not reach the end of its prompt. Sequences that
        are still prefilling never receive a token: their forward pass produced
        logits for a position in the middle of the prompt, and sampling from those
        would invent a token the caller never asked for and the prompt then
        contradicts.

        Returns the sequences that finished, in the same iteration they finished in,
        which is what frees their slot for a waiting request immediately.
        """
        sampled = tokens if tokens is not None else [None] * len(output.scheduled)
        if len(sampled) != len(output.scheduled):
            raise ValueError(
                f"got {len(sampled)} tokens for {len(output.scheduled)} scheduled sequences"
            )

        for (sequence, count), token in zip(output.scheduled, sampled, strict=True):
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

        Recompute rather than swap-to-host: the blocks go straight back to the pool
        and the sequence re-prefills over its prompt *and* its output so far. It
        costs the compute already spent on it, and it keeps the engine free of a
        swap path — see [Step 4.1]'s `reset_for_recompute`.
        """
        if sequence not in self.running:
            raise ValueError(f"sequence {sequence.seq_id} is not running")
        self.running.remove(sequence)
        sequence.reset_for_recompute()
        self.waiting.appendleft(sequence)

    def __repr__(self) -> str:
        return f"Scheduler(waiting={len(self.waiting)}, running={len(self.running)})"


class DenseModelRunner:
    """Executes a `SchedulerOutput` against Phase 2's per-sequence dense cache.

    **This is the wall.** A `DenseKvCache` is one contiguous `B x H x S x D` tensor
    per sequence, and two sequences at different lengths cannot share one. So a
    scheduled "batch" of `n` sequences becomes `n` separate forward passes here, and
    continuous batching buys scheduling fairness while buying **no** GPU efficiency
    at all: the GPU sees the same one-sequence-at-a-time work it saw in Phase 2, plus
    `n` times the launch overhead.

    Two ways out, and only one of them works:

    * *Pad* every sequence to the longest in the batch. A 300-token prefill beside
      two decode steps becomes a `3 x 300` rectangle: 598 wasted token-slots, and
      the waste grows with the length spread.
    * *Page* the cache, so K and V live in fixed-size blocks that any sequence can
      hold in any order, and hand the kernel `cu_seqlens_q` and `context_lens` so one
      launch covers the ragged batch. That is [Step 4.8].

    Keeping this class around after paging lands is deliberate: it is the oracle the
    paged runner is diffed against.
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

        The row is the *last* position the sequence computed, which for a decode step
        is its only position and for a prefill chunk is the end of the chunk. Only
        the final chunk's row is ever sampled from, but returning it uniformly keeps
        the caller from having to know which chunk it is looking at.
        """
        rows = []
        for sequence, count in output.scheduled:
            caches = self.caches_for(sequence)
            start = sequence.num_computed_tokens

            if start == 0 and caches[0].offset > 0:
                # Readmitted after preemption: the sequence recomputes from token
                # zero, so its stale cache has to go. This is the cost of
                # recompute-preemption made concrete.
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

        Sequences still mid-prefill get a token sampled and thrown away by
        `Scheduler.commit`. That is one wasted row of a `V`-wide sample rather than a
        branch in the hot path, and it keeps this a single batched call.
        """
        params = [sequence.sampling_params for sequence, _ in output.scheduled]
        if all(parameter.is_greedy for parameter in params):
            return logits.argmax(dim=-1).tolist()
        return sample(logits, params).tolist()
