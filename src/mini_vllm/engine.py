"""The engine: requests in, the continuous-batching loop, parallel sampling, and the LLM
API with streaming."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from mini_vllm.batch import ForwardBatch
from mini_vllm.generate import GREEDY
from mini_vllm.models import DEFAULT_MODEL, load
from mini_vllm.paged_kv_cache import BlockManager, PagedKvPool
from mini_vllm.qwen3 import Qwen3Model
from mini_vllm.sampler import SamplingParams, sample
from mini_vllm.scheduler import Request, RequestStatus, Scheduler, SchedulerConfig, SchedulerOutput
from mini_vllm.speculative import DraftProposer, SpeculativeDecoder, self_draft

__all__ = ["LLM", "Completion", "EngineConfig", "StreamUpdate", "blocks_that_fit"]


@dataclass(frozen=True)
class EngineConfig:
    """Everything the engine takes besides the model and tokenizer."""

    # KV memory.
    num_blocks: int | None = None   # None: kv_fraction of the memory left once the weights are in
    block_size: int = 16            # tokens per page
    kv_fraction: float = 0.5
    fp8_kv_cache: bool = False      # e4m3 pages: half the bytes, so twice the tokens in flight
    enable_prefix_caching: bool = False

    # Scheduling.
    max_batched_tokens: int = 2048
    max_sequences: int = 32
    chunk_size: int = 512
    enable_chunked_prefill: bool = True
    prefill_priority: bool = False

    # Speculative decoding: k > 0 drafts k tokens per decode step for the target to verify.
    num_speculative_tokens: int = 0
    draft_model: str | None = None        # a separate checkpoint sharing the tokenizer, or
    num_draft_layers: int | None = None   # None: a self-draft of the target's first N layers
    draft_blocks: int | None = None       # None: the target's pool plus room for k per request

    seed: int | None = None  # seeds MLX's global RNG; greedy decoding is deterministic anyway

    def scheduler_config(self) -> SchedulerConfig:
        return SchedulerConfig(
            max_batched_tokens=self.max_batched_tokens,
            max_sequences=self.max_sequences,
            chunk_size=self.chunk_size,
            enable_chunked_prefill=self.enable_chunked_prefill,
            prefill_priority=self.prefill_priority,
        )


def blocks_that_fit(model: Qwen3Model, config: EngineConfig) -> int:
    """How many pages fit in kv_fraction of the memory left once the weights are in.

    Unified memory has no free-versus-total split, so "left" is MLX's recommended working
    set (37.4 GB on a 48 GB M5 Pro) minus what is already active.
    """
    c = model.config
    element = 1 if config.fp8_kv_cache else model.embedding.weight.dtype.size
    per_block = 2 * c.num_hidden_layers * config.block_size * c.num_key_value_heads * c.head_dim * element
    left = mx.device_info()["max_recommended_working_set_size"] - mx.get_active_memory()
    return int(left * config.kv_fraction) // per_block


def _as_prompt_list(prompts: str | list[int] | Sequence[str | list[int]]) -> list[str | list[int]]:
    """One prompt, as text or token ids, or a list of them."""
    return [prompts] if isinstance(prompts, str) or isinstance(prompts[0], int) else list(prompts)


@dataclass(frozen=True)
class StreamUpdate:
    """One token as it is produced. index is the prompt's position in the request list and
    sample_index which of its n completions this is; text is the newly decodable text."""

    index: int
    sample_index: int
    token_id: int
    text: str
    finish_reason: str | None  # "stop" or "length" on the request's last update, else None


@dataclass(frozen=True)
class Completion:
    prompt: str | list[int]
    sample_index: int
    text: str
    token_ids: list[int]
    finish_reason: str


class LLM:
    """A paged-attention inference engine for Qwen3.

        llm = LLM.from_pretrained("Qwen/Qwen3-0.6B", enable_prefix_caching=True)
        completions = llm.generate(["The capital of France is"], max_tokens=32)
    """

    def __init__(self, model: Qwen3Model, tokenizer: Any, config: EngineConfig | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config = config or EngineConfig()
        self.stop_token_ids = frozenset(tokenizer.eos_token_ids)

        c = model.config
        self.kv = PagedKvPool(
            c.num_hidden_layers,
            config.num_blocks or blocks_that_fit(model, config),
            config.block_size,
            c.num_key_value_heads,
            c.head_dim,
            dtype=model.embedding.weight.dtype,
            fp8=config.fp8_kv_cache,
        )
        mx.eval(self.kv.keys, self.kv.values)  # take the whole pool now, not on the first write
        self.manager = BlockManager(self.kv, config.enable_prefix_caching)
        self.scheduler = Scheduler(config.scheduler_config(), self.manager)
        self.spec = self._build_speculation() if config.num_speculative_tokens else None

        if config.seed is not None:
            mx.random.seed(config.seed)

    def _build_speculation(self) -> SpeculativeDecoder:
        """The draft and the pool of its own it caches in: as many layers as the draft has."""
        config = self.config
        if config.draft_model is not None:
            draft = load(config.draft_model)[0]
        else:
            draft = self_draft(self.model, config.num_draft_layers or self.model.config.num_hidden_layers)
        k = config.num_speculative_tokens
        # It caches every target token plus k proposals, for every request in flight.
        per_request = -(-k // config.block_size) + 1
        dc = draft.config
        kv = PagedKvPool(
            dc.num_hidden_layers,
            config.draft_blocks or self.kv.num_blocks + config.max_sequences * per_request,
            config.block_size,
            dc.num_key_value_heads,
            dc.head_dim,
            dtype=draft.embedding.weight.dtype,
            fp8=config.fp8_kv_cache,
        )
        mx.eval(kv.keys, kv.values)
        return SpeculativeDecoder(DraftProposer(draft, kv, k), self.manager)

    @classmethod
    def from_pretrained(cls, model: str = DEFAULT_MODEL, **config: Any) -> LLM:
        qwen3, tokenizer = load(model)
        return cls(qwen3, tokenizer, EngineConfig(**config))

    def add_request(
        self,
        prompt: str | list[int],
        params: SamplingParams | None = None,
        max_tokens: int = 16,
        ignore_eos: bool = False,
    ) -> Request:
        """Queue a prompt, as text or token ids. Greedy unless params say otherwise;
        ignore_eos runs to max_tokens, for benchmarks that must not stop early."""
        request = Request(
            prompt_token_ids=self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt),
            sampling_params=params or GREEDY,
            max_tokens=max_tokens,
            stop_token_ids=frozenset() if ignore_eos else self.stop_token_ids,
        )
        self.scheduler.add(request)
        return request

    def step(self) -> tuple[list[tuple[Request, int]], list[tuple[Request, Request]]]:
        """One iteration for everything in flight. Returns the (request, token) pairs it
        emitted, in order, and the (leader, branch) pairs parallel sampling forked."""
        # Proposals come first: they lengthen a request, and schedule() reserves their slots.
        decoding = []
        if self.spec is not None:
            running = self.scheduler.running
            decoding = [r for r in running if r.output_token_ids and r.num_uncomputed_tokens == 1]
            self.spec.propose(decoding)
        output = self.scheduler.schedule()
        scheduled = {request.request_id for request in output.requests}
        for request in decoding:
            if request.request_id not in scheduled:
                self.spec.discard(request)

        # A speculated request needs logits for every row it computed, to verify each
        # proposal; everything else needs only its last row.
        rows, spans, start = [], [], 0
        for request, count in output.scheduled:
            first = len(rows)
            rows.extend(range(start, start + count) if request.proposed_token_ids else [start + count - 1])
            spans.append(slice(first, len(rows)))
            start += count
        batch = ForwardBatch.from_scheduled(output.scheduled, self.manager)
        caches = self.kv.caches(batch)
        logits = self.model(batch.input_ids[None], batch.positions, caches, rows=mx.array(rows))[0]
        last = logits[mx.array([span.stop - 1 for span in spans])]
        tokens = sample(last, [request.sampling_params for request in output.requests])

        # The one synchronization per step. The tokens are needed on the host anyway, and
        # evaluating every layer's pages with them ends this step's graph there: the next
        # step starts from concrete buffers, and the old pool buffers are released rather
        # than kept alive by a graph that still reaches back to them.
        mx.eval(tokens, self.kv.keys, self.kv.values)
        tokens = tokens.tolist()

        # Read before commit advances them: a request takes a token only once caught up.
        emitting = [
            request
            for request, count in output.scheduled
            if request.num_computed_tokens + count == len(request) and not request.proposed_token_ids
        ]
        runs = {
            request.request_id: self.spec.verify(request, logits[span])
            for (request, _), span in zip(output.scheduled, spans, strict=True)
            if request.proposed_token_ids
        }
        # Before commit, which frees a leader that finishes on its first token.
        forked = self._fork_after_prefill(output, last)
        finished = self.scheduler.commit(output, tokens, verified=runs.keys())

        emitted = [(request, request.output_token_ids[-1]) for request in emitting]
        for request in output.requests:
            emitted += [(request, token) for token in runs.get(request.request_id, [])]
        emitted += [(branch, branch.output_token_ids[-1]) for _, branch in forked]

        # A preempted request recomputes, so its draft cache is released rather than repaired.
        if self.spec is not None:
            for request in finished + output.preempted:
                self.spec.release(request)
        return emitted, forked

    def _fork_after_prefill(
        self, output: SchedulerOutput, logits: mx.array
    ) -> list[tuple[Request, Request]]:
        """When an n > 1 request finishes its prompt, sample its first-token row n - 1 more
        times and start a branch from each, sharing every prompt page through fork."""
        forked = []
        for row, (leader, count) in enumerate(output.scheduled):
            params = leader.sampling_params
            done_prompt = leader.num_computed_tokens + count == len(leader.prompt_token_ids)
            if params.n == 1 or leader.output_token_ids or not done_prompt:
                continue
            firsts = sample(mx.repeat(logits[row : row + 1], params.n - 1, axis=0), params).tolist()
            for token in firsts:
                branch = Request(
                    prompt_token_ids=leader.prompt_token_ids,
                    sampling_params=params,
                    max_tokens=leader.max_tokens,
                    stop_token_ids=leader.stop_token_ids,
                    output_token_ids=[token],
                    num_computed_tokens=leader.num_computed_tokens + count,
                    status=RequestStatus.RUNNING,
                )
                self.manager.fork(leader, branch)
                self.scheduler.running.append(branch)
                forked.append((leader, branch))
        return forked

    def generate_stream(
        self,
        prompts: str | list[int] | Sequence[str | list[int]],
        params: SamplingParams | Sequence[SamplingParams] | None = None,
        max_tokens: int = 16,
        ignore_eos: bool = False,
    ) -> Iterator[StreamUpdate]:
        """Yield every token as it is produced, across all prompts, interleaved."""
        prompts = _as_prompt_list(prompts)
        params = params if isinstance(params, Sequence) else [params] * len(prompts)
        leaders = [
            self.add_request(prompt, sp, max_tokens, ignore_eos)
            for prompt, sp in zip(prompts, params, strict=True)
        ]

        # (prompt index, sample index) for each request; a branch takes its leader's index.
        where = {leader.request_id: (index, 0) for index, leader in enumerate(leaders)}
        samples = [1] * len(leaders)
        # Per request: how many of its tokens, and how many characters of text, were yielded.
        yielded = {leader.request_id: 0 for leader in leaders}
        shown = {leader.request_id: 0 for leader in leaders}
        owned = list(leaders)
        try:
            while any(request.status is not RequestStatus.FINISHED for request in owned):
                emitted, forked = self.step()
                for leader, branch in forked:
                    index = where[leader.request_id][0]
                    where[branch.request_id] = (index, samples[index])
                    samples[index] += 1
                    yielded[branch.request_id] = shown[branch.request_id] = 0
                    owned.append(branch)

                for request, token in emitted:
                    if request.request_id not in where:
                        continue  # another caller's request, sharing the engine
                    index, sample_index = where[request.request_id]
                    # A speculative step emits a run of tokens at once; each update covers
                    # the output up to its own token, and only the run's last can finish.
                    yielded[request.request_id] += 1
                    upto = request.output_token_ids[: yielded[request.request_id]]
                    is_last = len(upto) == len(request.output_token_ids)
                    finished = request.status is RequestStatus.FINISHED and is_last
                    stopped = token in request.stop_token_ids
                    # Decode the output so far and yield what is new: a token is not a character,
                    # and a multi-byte character split across tokens decodes to U+FFFD until its
                    # last byte arrives. A stop token ends the text rather than appearing in it.
                    text = self.tokenizer.decode(upto[:-1] if stopped else upto)
                    delta = ""
                    if finished or not text.endswith("\ufffd"):
                        delta = text[shown[request.request_id] :]
                        shown[request.request_id] = len(text)
                    reason = ("stop" if stopped else "length") if finished else None
                    yield StreamUpdate(index, sample_index, token, delta, reason)
        finally:
            # A caller that stops reading closes the generator here, with pages still held.
            for request in owned:
                if request.status is not RequestStatus.FINISHED:
                    if request in self.scheduler.running:
                        self.scheduler.running.remove(request)
                    else:
                        self.scheduler.waiting.remove(request)
                    self.manager.free(request)
                if self.spec is not None:
                    self.spec.release(request)

    def generate(
        self,
        prompts: str | list[int] | Sequence[str | list[int]],
        params: SamplingParams | Sequence[SamplingParams] | None = None,
        max_tokens: int = 16,
        ignore_eos: bool = False,
    ) -> list[Completion]:
        """Run every prompt to completion: generate_stream drained, so the two cannot drift.
        Completions come in prompt order, and a prompt's n samples in sample order."""
        prompt_list = _as_prompt_list(prompts)
        texts, tokens, reasons = {}, {}, {}
        for update in self.generate_stream(prompt_list, params, max_tokens, ignore_eos):
            key = (update.index, update.sample_index)
            texts[key] = texts.get(key, "") + update.text
            tokens.setdefault(key, []).append(update.token_id)
            if update.finish_reason is not None:
                reasons[key] = update.finish_reason
        return [
            Completion(prompt_list[key[0]], key[1], texts[key], tokens[key], reasons[key])
            for key in sorted(texts)
        ]
