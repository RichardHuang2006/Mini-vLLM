"""Continuous batching: requests, and the per-iteration policy that decides who runs and
for how many tokens: admission, a token budget, chunked prefill, and preemption."""

from __future__ import annotations

import enum
import itertools
from collections import deque
from dataclasses import dataclass, field

from mini_vllm.paged_kv_cache import BlockManager, BlockTable
from mini_vllm.sampler import SamplingParams

__all__ = ["Request", "RequestStatus", "Scheduler", "SchedulerConfig", "SchedulerOutput"]


class RequestStatus(enum.Enum):
    WAITING = "waiting"    # queued, or preempted and queued again
    RUNNING = "running"
    FINISHED = "finished"


_next_request_id = itertools.count()


@dataclass(eq=False)
class Request:
    """A request in flight: its tokens, how many of them the KV cache holds, its pages.

    token_ids is prompt + output, and num_computed_tokens says how much of it is cached;
    the difference is what the next iteration can compute.
    """

    prompt_token_ids: list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    max_tokens: int = 16
    stop_token_ids: frozenset[int] = frozenset()
    request_id: int = field(default_factory=lambda: next(_next_request_id))

    output_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    status: RequestStatus = RequestStatus.WAITING
    block_table: BlockTable | None = None

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    def __len__(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_uncomputed_tokens(self) -> int:
        return len(self) - self.num_computed_tokens

    def is_prefill(self) -> bool:
        """True while any prompt token is still uncomputed, chunked or not."""
        return self.num_computed_tokens < len(self.prompt_token_ids)

    def is_done(self) -> bool:
        if len(self.output_token_ids) >= self.max_tokens:
            return True
        return bool(self.output_token_ids) and self.output_token_ids[-1] in self.stop_token_ids


@dataclass(frozen=True)
class SchedulerConfig:
    max_batched_tokens: int = 2048       # compute budget: token positions per forward pass
    max_sequences: int = 16              # requests in flight at once
    chunk_size: int = 512                # one prefill's share of an iteration
    enable_chunked_prefill: bool = True  # off: a prompt is prefilled in one pass
    prefill_priority: bool = False       # prompts before decodes: the pre-chunking baseline


@dataclass
class SchedulerOutput:
    """One iteration's batch: each request with how many tokens it computes."""

    scheduled: list[tuple[Request, int]] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)

    @property
    def requests(self) -> list[Request]:
        return [request for request, _ in self.scheduled]

    @property
    def total_tokens(self) -> int:
        return sum(count for _, count in self.scheduled)

    def tokens_for(self, request: Request) -> int:
        return next((count for scheduled, count in self.scheduled if scheduled is request), 0)


class Scheduler:
    """FCFS waiting and running queues, re-decided every iteration, Orca-style.

    It owns the blocks' life cycle: schedule() allocates what it admits, and commit() and
    preempt() free. A scheduling decision may change timing, never output.
    """

    def __init__(self, config: SchedulerConfig, manager: BlockManager) -> None:
        self.config = config
        self.manager = manager
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    def add(self, request: Request) -> None:
        self.waiting.append(request)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput:
        """Decide this iteration's batch: decodes, then prefill chunks, then arrivals.

        Decodes go first because together they cost less than one chunk and their callers
        are already reading output; prefill_priority runs the passes the other way round.
        """
        output = SchedulerOutput()
        budget = self.config.max_batched_tokens
        # Blocks promised so far this iteration, so two requests cannot claim one page.
        promised: dict[int, int] = {}

        decoding = [request for request in self.running if not request.is_prefill()]
        prefilling = [request for request in self.running if request.is_prefill()]

        if self.config.prefill_priority:
            budget = self._advance(prefilling, output, budget, promised)
            budget = self._admit(output, budget, promised)
            self._advance(decoding, output, budget, promised)
        else:
            budget = self._advance(decoding, output, budget, promised)
            budget = self._advance(prefilling, output, budget, promised)
            self._admit(output, budget, promised)

        for request, count in output.scheduled:
            self.manager.allocate(request, count)
        return output

    def commit(self, output: SchedulerOutput, tokens: list[int] | None = None) -> list[Request]:
        """Record an iteration: tokens holds one sampled token per scheduled request.
        Returns the requests that finished, whose blocks are already freed."""
        tokens = tokens or [None] * len(output.scheduled)
        for (request, count), token in zip(output.scheduled, tokens, strict=True):
            request.num_computed_tokens += count
            # Only a request whose every token is now computed takes one. A chunk that stopped
            # mid-prompt, or mid-recompute after preemption, sampled a row predicting a token
            # the request already has.
            if request.num_computed_tokens == len(request) and token is not None:
                request.output_token_ids.append(token)

        finished = [request for request in self.running if request.is_done()]
        for request in finished:
            request.status = RequestStatus.FINISHED
            self.running.remove(request)
            self.manager.free(request)
        return finished

    def preempt(self, request: Request) -> None:
        """Evict a running request: its pages go back now, and it is requeued at the front
        to be recomputed over its prompt and its output so far."""
        self.running.remove(request)
        self.manager.free(request)
        request.num_computed_tokens = 0
        request.status = RequestStatus.WAITING
        self.waiting.appendleft(request)

    def _advance(
        self,
        requests: list[Request],
        output: SchedulerOutput,
        budget: int,
        promised: dict[int, int],
    ) -> int:
        """Give each running request its next tokens, returning the budget left; it can
        grow, since preempting an already-scheduled victim hands its tokens back."""
        for request in requests:
            if request.status is not RequestStatus.RUNNING:
                continue  # preempted earlier in this pass to make room for another
            count = self._tokens_for(request, budget)
            if count == 0:
                continue  # the budget ran out: it keeps its cache and its place
            refund = self._make_room_for(request, count, output, promised)
            if refund is None:
                continue  # it was the newest, so it was preempted itself
            self._schedule(output, request, count, promised)
            budget += refund - count
        return budget

    def _admit(self, output: SchedulerOutput, budget: int, promised: dict[int, int]) -> int:
        """Take arrivals off the front of the queue while they fit."""
        while self.waiting and len(self.running) < self.config.max_sequences:
            candidate = self.waiting[0]
            # Before the prefill is sized, so only the uncached part is scheduled.
            self.manager.apply_prefix_cache(candidate)
            count = self._tokens_for(candidate, budget)
            if count == 0 and not output.scheduled:
                # Chunking off and nothing else running: it overruns the budget rather
                # than deadlocking the queue.
                count = candidate.num_uncomputed_tokens

            # A queued request waits rather than preempting for room. It also hands back the
            # cached pages it just matched: held while it waits, they would shrink the pool
            # under blocks already promised to the requests scheduled ahead of it.
            if count == 0 or not self._fits(candidate, count, promised):
                self.manager.free(candidate)
                candidate.num_computed_tokens = 0
                break

            self.waiting.popleft()
            candidate.status = RequestStatus.RUNNING
            self.running.append(candidate)
            self._schedule(output, candidate, count, promised)
            budget -= count
        return budget

    def _schedule(
        self, output: SchedulerOutput, request: Request, count: int, promised: dict[int, int]
    ) -> None:
        output.scheduled.append((request, count))
        promised[request.request_id] = self.manager.blocks_needed(request, count)

    def _fits(self, request: Request, count: int, promised: dict[int, int]) -> bool:
        """Whether the pool can back count more tokens, net of this iteration's promises."""
        available = self.manager.pool.num_free - sum(promised.values())
        return self.manager.blocks_needed(request, count) <= available

    def _make_room_for(
        self,
        request: Request,
        count: int,
        output: SchedulerOutput,
        promised: dict[int, int],
    ) -> int | None:
        """Preempt from the back of the running list, newest first, until count tokens fit.
        Returns the budget that freed, or None once request itself was the newest left and
        so was preempted: an older request is never evicted for a newer one."""
        refund = 0
        while not self._fits(request, count, promised):
            victim = self.running[-1]
            if victim is request:
                self.preempt(request)
                output.preempted.append(request)
                return None
            # A victim already scheduled this iteration comes back out with its tokens.
            refund += output.tokens_for(victim)
            promised.pop(victim.request_id, None)
            output.scheduled = [pair for pair in output.scheduled if pair[0] is not victim]
            self.preempt(victim)
            output.preempted.append(victim)
        return refund

    def _tokens_for(self, request: Request, budget: int) -> int:
        """How many of this request's tokens fit in what is left of the budget."""
        wanted = request.num_uncomputed_tokens
        if not self.config.enable_chunked_prefill:
            return wanted if wanted <= budget else 0
        return min(wanted, self.config.chunk_size, max(budget, 0))
