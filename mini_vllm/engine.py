"""The engine: request admission, the iteration loop, and the `LLM` API."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Iterator, Sequence as SequenceABC
from dataclasses import dataclass, replace
from typing import NamedTuple

import torch
from transformers import AutoTokenizer

from mini_vllm.cache import BlockManager, PagedKvPool
from mini_vllm.config import (
    CPU_BLOCKS,
    DEFAULT_MODEL_ID,
    EngineConfig,
    SamplingParams,
)
from mini_vllm.model import Qwen3, Qwen3Cached, Qwen3Paged, load_weights, resolve_model_path
from mini_vllm.ops import sample
from mini_vllm.scheduler import ForwardBatch, Scheduler, Sequence
from mini_vllm.speculative import DraftProposer, SpeculativeDecoder

__all__ = [
    "Loaded",
    "eos_token_ids_for",
    "load",
    "generate_ids",
    "generate_ids_cached",
    "PagedModelRunner",
    "Completion",
    "StreamUpdate",
    "EngineStats",
    "LLM",
]

DEFAULT_MAX_TOKENS = 32


class Loaded(NamedTuple):
    """A model, its tokenizer, and the stop tokens that go with them."""

    model: Qwen3 | Qwen3Cached
    tokenizer: object
    eos_token_ids: tuple[int, ...]
    pad_token_id: int


def eos_token_ids_for(model_path) -> tuple[int, ...]:
    """The stop tokens from `generation_config.json`; Qwen3 lists two, hence a tuple."""
    config_path = model_path / "generation_config.json"
    if not config_path.is_file():
        return ()

    stated = json.loads(config_path.read_text()).get("eos_token_id")
    if stated is None:
        return ()
    return (stated,) if isinstance(stated, int) else tuple(stated)


def load(
    model: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
    cached: bool = True,
    use_cuda_kernels: bool = False,
) -> Loaded:
    """Load a reference model, tokenizer and stop tokens together."""
    path = resolve_model_path(model)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if cached:
        loaded_model: Qwen3 | Qwen3Cached = Qwen3Cached.from_pretrained(
            path, device=device, use_cuda=use_cuda_kernels
        )
    else:
        loaded_model = Qwen3.from_pretrained(path, device=device)

    return Loaded(
        model=loaded_model,
        tokenizer=AutoTokenizer.from_pretrained(path),
        eos_token_ids=eos_token_ids_for(path),
        pad_token_id=json.loads((path / "generation_config.json").read_text()).get(
            "pad_token_id", 0
        ),
    )


@torch.no_grad()
def generate_ids(
    model: Qwen3,
    input_ids: torch.Tensor,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    eos_token_ids: SequenceABC[int] = (),
    pad_token_id: int | None = None,
) -> torch.Tensor:
    """Greedy decode with no cache: input_ids [B, L] -> [B, L + generated].

    Quadratic in the output length, and the oracle `generate_ids_cached` is diffed
    against. Finished rows are padded to keep the batch rectangular, matching HuggingFace.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"expected B x L input ids, got shape {tuple(input_ids.shape)}")

    stop_tokens = set(eos_token_ids)
    if pad_token_id is None:
        pad_token_id = next(iter(stop_tokens), 0)

    tokens = input_ids
    finished = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)

    for _ in range(max_tokens):
        logits = model(tokens)[:, -1, :]  # the whole prefix, recomputed every step
        next_tokens = logits.argmax(dim=-1)

        next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_token_id), next_tokens)
        tokens = torch.cat([tokens, next_tokens.unsqueeze(1)], dim=1)

        for stop in stop_tokens:
            finished |= next_tokens == stop
        if bool(finished.all()):
            break

    return tokens


@torch.no_grad()
def generate_ids_cached(
    model: Qwen3Cached,
    input_ids: torch.Tensor,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    eos_token_ids: SequenceABC[int] = (),
    pad_token_id: int | None = None,
    caches: list | None = None,
) -> torch.Tensor:
    """Greedy decode with a KV cache: prefill the prompt, then one token per step."""
    if input_ids.ndim != 2:
        raise ValueError(f"expected B x L input ids, got shape {tuple(input_ids.shape)}")

    stop_tokens = set(eos_token_ids)
    if pad_token_id is None:
        pad_token_id = next(iter(stop_tokens), 0)

    if caches is None:
        caches = model.create_kv_cache()

    tokens = input_ids
    finished = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
    step_input = input_ids

    for _ in range(max_tokens):
        # Only the tokens the model has not seen; the caches carry the position.
        logits = model(step_input, caches, last_only=True)[:, -1, :]
        next_tokens = logits.argmax(dim=-1)

        next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_token_id), next_tokens)
        tokens = torch.cat([tokens, next_tokens.unsqueeze(1)], dim=1)
        step_input = next_tokens.unsqueeze(1)

        for stop in stop_tokens:
            finished |= next_tokens == stop
        if bool(finished.all()):
            break

    return tokens


class PagedModelRunner:
    """Runs one iteration: reserve pages, build the ragged batch, forward, sample."""

    def __init__(self, model, manager: BlockManager, device: torch.device | str | None = None):
        self.model = model
        self.manager = manager
        self.device = torch.device(device) if device else manager.kv.device

    def execute(self, output, all_rows: bool = False) -> torch.Tensor:
        """One forward pass over the scheduled batch: num_scheduled x V, or T x V."""
        # Blocks first: slot_mapping holds addresses, which need the pages to exist.
        for sequence, count in output.scheduled:
            self.manager.allocate(sequence, count)

        batch = self.build(output)
        return self.model(batch, all_rows=all_rows)

    def build(self, output):
        """The batch for an already-reserved iteration. Split out for the tests."""
        return ForwardBatch.from_scheduled(output.scheduled, self.device, manager=self.manager)

    def sample_tokens(
        self,
        output,
        logits: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> list[int]:
        """One token per scheduled sequence, honouring per-row parameters.

        Mid-prompt rows are sampled and discarded by `Scheduler.commit`: one wasted row
        instead of a device synchronization to slice them out.
        """
        params = [sequence.sampling_params for sequence, _ in output.scheduled]
        if all(parameter.is_greedy for parameter in params):
            return logits.argmax(dim=-1).tolist()
        return sample(logits, params, generator=generator).tolist()

    def free(self, sequence: Sequence) -> None:
        """Return a finished sequence's pages to the pool."""
        self.manager.free(sequence)


@dataclass(frozen=True)
class Completion:
    """One finished request."""

    prompt: str
    text: str
    token_ids: tuple[int, ...]
    finish_reason: str
    seq_id: int
    sample_index: int = 0  # position among a prompt's `n` completions

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class StreamUpdate:
    """One token as it is produced; `index` is the prompt's position in the request list."""

    index: int
    seq_id: int
    text: str
    token_id: int
    finished: bool
    finish_reason: str | None = None
    sample_index: int = 0  # which of a prompt's `n` parallel samples this token is


@dataclass
class EngineStats:
    """Counters for the whole engine's life, read by the benchmarks."""

    iterations: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    preemptions: int = 0
    cached_tokens: int = 0
    elapsed: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of prompt tokens served from the prefix cache rather than recomputed."""
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.elapsed if self.elapsed else 0.0

    @property
    def tokens_per_iteration(self) -> float:
        total = self.prompt_tokens + self.generated_tokens
        return total / self.iterations if self.iterations else 0.0


class LLM:
    """A paged-attention inference engine for Qwen3.

        llm = LLM("Qwen/Qwen3-0.6B")
        completions = llm.generate(prompts, max_tokens=64)

    Configuration is one `config.EngineConfig`; keyword arguments build one.
    """

    def __init__(
        self,
        model: str | None = None,
        config: EngineConfig | None = None,
        **overrides,
    ) -> None:
        if config is None:
            if model is not None:
                overrides["model"] = model
            config = EngineConfig(**overrides)
        elif model is not None or overrides:
            raise ValueError("pass either config= or keyword arguments, not both")
        self.engine_config = config

        device = config.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        path = resolve_model_path(config.model)
        weights, model_config = load_weights(path, device=device)
        if config.dtype is not None and config.dtype != model_config.dtype:
            # fp32 is for tests: in bf16 a step's top two logits are often a tie.
            weights = {name: tensor.to(config.dtype) for name, tensor in weights.items()}
            model_config = replace(model_config, dtype=config.dtype)
        self.config = model_config
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.stop_token_ids = eos_token_ids_for(path) or (self.tokenizer.eos_token_id,)

        self.kv_dtype = config.kv_dtype
        num_blocks = config.num_blocks
        if num_blocks is None:
            num_blocks = self.blocks_that_fit(
                model_config, config.block_size, self.device, config.kv_fraction, self.kv_dtype
            )
        self.manager = BlockManager(
            num_blocks=num_blocks,
            block_size=config.block_size,
            num_layers=model_config.num_hidden_layers,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            dtype=model_config.dtype,
            device=self.device,
            enable_prefix_caching=config.enable_prefix_caching,
            kv_dtype=self.kv_dtype,
        )

        self.model = Qwen3Paged(model_config, weights, self.manager, use_cuda=config.use_cuda_kernels)
        self.scheduler = Scheduler(config.scheduler_config(), manager=self.manager)
        self.runner = PagedModelRunner(self.model, self.manager, self.device)
        self.stats = EngineStats()

        # A seed makes stochastic sampling replayable without touching global RNG state.
        self.generator = (
            torch.Generator(device=self.device).manual_seed(config.seed)
            if config.seed is not None
            else None
        )

        # Parallel sampling: leaders already forked, and the branches forked this step.
        self._forked: set[int] = set()
        self._newly_forked: list[Sequence] = []

        self.spec = self._build_speculation(config) if config.num_speculative_tokens > 0 else None

    def _build_speculation(self, config: EngineConfig) -> SpeculativeDecoder:
        """Stand up the draft model and the KV pool it needs of its own.

        A separate checkpoint is the arrangement that can be faster; a self-draft (the
        default) reuses the target's first `num_draft_layers` layers and their weights,
        which exercises the mechanism where there is no room for a second model.
        """
        if config.draft_model is not None:
            weights, draft_config = load_weights(
                resolve_model_path(config.draft_model), device=str(self.device)
            )
            if draft_config.dtype != self.config.dtype:
                weights = {name: tensor.to(self.config.dtype) for name, tensor in weights.items()}
                draft_config = replace(draft_config, dtype=self.config.dtype)
            if draft_config.vocab_size != self.config.vocab_size:
                # Rejection sampling compares p and q token by token.
                raise ValueError(
                    f"the draft's vocabulary ({draft_config.vocab_size}) differs from the "
                    f"target's ({self.config.vocab_size}); they cannot verify each other"
                )
            layers = draft_config.num_hidden_layers
            kv_heads, head_dim = draft_config.num_key_value_heads, draft_config.head_dim
        else:
            draft_config = None
            layers = config.num_draft_layers or self.config.num_hidden_layers
            kv_heads, head_dim = self.config.num_key_value_heads, self.config.head_dim

        # The draft caches the target's tokens plus k more, in a shallower, cheaper pool.
        k = config.num_speculative_tokens
        proposal_pages = -(-k // self.manager.block_size) + 1
        default_blocks = (
            self.manager.num_blocks + self.scheduler.config.max_sequences * proposal_pages
        )

        draft_manager = BlockManager(
            num_blocks=config.draft_blocks or default_blocks,
            block_size=self.manager.block_size,
            num_layers=layers,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            dtype=self.config.dtype,
            device=self.device,
            kv_dtype=self.kv_dtype,
        )
        if draft_config is not None:
            drafter = Qwen3Paged(draft_config, weights, draft_manager, use_cuda=self.model.use_cuda)
        else:
            drafter = self.model.self_draft(layers, draft_manager)

        proposer = DraftProposer(drafter, draft_manager, num_speculative_tokens=k)
        return SpeculativeDecoder(proposer, self.model, self.manager, runner=self.runner)

    @staticmethod
    def blocks_that_fit(
        config,
        block_size: int,
        device: torch.device,
        fraction: float,
        kv_dtype: torch.dtype | None = None,
    ) -> int:
        """How many pages fit in `fraction` of the memory free once the weights are in."""
        per_block = PagedKvPool.bytes_for(
            num_layers=config.num_hidden_layers,
            num_blocks=1,
            block_size=block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=kv_dtype or config.dtype,
        )
        if device.type != "cuda":
            return CPU_BLOCKS

        free, _total = torch.cuda.mem_get_info(device)
        return max(1, int(free * fraction) // per_block)

    @property
    def kv_cache_bytes(self) -> int:
        return PagedKvPool.bytes_for(
            self.config.num_hidden_layers,
            self.manager.num_blocks,
            self.manager.block_size,
            self.config.num_key_value_heads,
            self.config.head_dim,
            self.manager.kv.kv_dtype,
        )

    def reconfigure(self, **changes) -> None:
        """Swap the scheduling policy between runs, keeping the weights and the pool."""
        if self.scheduler.num_unfinished:
            raise ValueError(
                f"{self.scheduler.num_unfinished} sequences are still in flight; "
                "the policy can only change between runs"
            )
        self.scheduler = Scheduler(replace(self.scheduler.config, **changes), manager=self.manager)

    def add_request(
        self,
        prompt: str | SequenceABC[int],
        sampling_params: SamplingParams | None = None,
        max_tokens: int = 16,
        ignore_eos: bool = False,
    ) -> Sequence:
        """Tokenize (if needed), wrap in a Sequence, and enqueue it.

        `ignore_eos` is for the benchmarks: a run where some requests quit at token 9 and
        others at 64 measures the prompts rather than the engine.
        """
        if isinstance(prompt, str):
            token_ids = self.tokenizer(prompt).input_ids
        else:
            token_ids = list(prompt)
        if not token_ids:
            raise ValueError("an empty prompt has nothing to forward")

        sequence = Sequence(
            prompt_token_ids=list(token_ids),
            sampling_params=sampling_params or SamplingParams(temperature=0.0),
            max_tokens=max_tokens,
            stop_token_ids=() if ignore_eos else tuple(self.stop_token_ids),
        )
        self.scheduler.add(sequence)
        self.stats.prompt_tokens += len(token_ids)
        return sequence

    def step(self) -> list[tuple[Sequence, int]]:
        """Advance every sequence in flight by one iteration, returning what it emitted."""
        if self.spec is not None:
            return self._speculative_step()

        output = self.scheduler.schedule()
        logits = self.runner.execute(output)
        tokens = self.runner.sample_tokens(output, logits, generator=self.generator)
        finished = self.scheduler.commit(output, tokens)

        # A chunk that stopped mid-prompt produced no token to emit.
        emitted = [
            (sequence, sequence.output_token_ids[-1])
            for sequence, _ in output.scheduled
            if not sequence.is_prefill() and sequence.output_token_ids
        ]

        # A leader forks the moment it finishes its prompt, and those tokens count here.
        self._newly_forked = self._fork_after_prefill(output, logits)
        emitted.extend(
            (child, child.output_token_ids[-1])
            for child in self._newly_forked
            if child.output_token_ids
        )

        for sequence in finished:
            self.runner.free(sequence)

        self.stats.iterations += 1
        self.stats.generated_tokens += len(emitted)
        self.stats.preemptions += len(output.preempted)
        self.stats.cached_tokens = self.manager.cached_tokens
        return emitted

    def _speculative_step(self) -> list[tuple[Sequence, int]]:
        """One iteration with a draft model in front of the target.

        Proposals are drafted before `schedule`, which must reserve slots for them; the
        target runs with all_rows=True; and a sequence emits between 1 and k + 1 tokens.
        """
        spec = self.spec
        assert spec is not None  # only reached when speculation is configured

        candidates = spec.candidates(list(self.scheduler.running))
        proposals = spec.propose(candidates) if candidates else {}

        output = self.scheduler.schedule()
        scheduled_ids = {sequence.seq_id for sequence, _ in output.scheduled}
        # Unscheduled proposals are discarded: q was computed for this context only.
        for sequence in candidates:
            if sequence.seq_id not in scheduled_ids and sequence.proposed_token_ids:
                sequence.discard_proposals()

        logits = self.runner.execute(output, all_rows=True)

        # The last row of each sequence, for the ones not being verified.
        last_rows, row = [], 0
        for _sequence, count in output.scheduled:
            row += count
            last_rows.append(row - 1)
        last_row_logits = logits.index_select(
            0, torch.tensor(last_rows, device=logits.device, dtype=torch.int64)
        )

        verified_pairs = spec.verify(output.scheduled, proposals, logits, generator=self.generator)
        verified = {sequence.seq_id for sequence, _ in verified_pairs}

        tokens = self.runner.sample_tokens(output, last_row_logits, generator=self.generator)
        finished = self.scheduler.commit(output, tokens, verified=verified)

        emitted: list[tuple[Sequence, int]] = []
        for sequence, _count in output.scheduled:
            if sequence.seq_id in verified:
                continue
            if not sequence.is_prefill() and sequence.output_token_ids:
                emitted.append((sequence, sequence.output_token_ids[-1]))
        # A verified sequence emits its whole accepted run, in order.
        for sequence, run in verified_pairs:
            emitted.extend((sequence, token) for token in run)

        self._newly_forked = self._fork_after_prefill(output, last_row_logits)
        emitted.extend(
            (child, child.output_token_ids[-1])
            for child in self._newly_forked
            if child.output_token_ids
        )

        # A preempted sequence recomputes, so its draft cache is released, not repaired.
        for sequence in output.preempted:
            spec.release(sequence)

        for sequence in finished:
            spec.release(sequence)
            self.runner.free(sequence)

        self.stats.iterations += 1
        self.stats.generated_tokens += len(emitted)
        self.stats.preemptions += len(output.preempted)
        self.stats.cached_tokens = self.manager.cached_tokens
        return emitted

    def _fork_after_prefill(self, output, logits: torch.Tensor) -> list[Sequence]:
        """Spawn a parallel-sampling request's remaining branches once it has prefilled.

        The row that gave the leader its first token is sampled n - 1 more times, and each
        branch shares the prompt's pages through `BlockManager.fork`.
        """
        children: list[Sequence] = []
        for row, (sequence, _count) in enumerate(output.scheduled):
            params = sequence.sampling_params
            if params.n <= 1 or sequence.parent_id is not None:
                continue
            if sequence.seq_id in self._forked or sequence.is_prefill():
                continue

            self._forked.add(sequence.seq_id)
            parent_logits = logits[row : row + 1]  # 1 x V, the first-token distribution
            for _ in range(params.n - 1):
                if params.is_greedy:
                    token = int(parent_logits.argmax(dim=-1))
                else:
                    token = int(sample(parent_logits, params, generator=self.generator))
                child = sequence.fork(first_output_token=token)
                self.manager.fork(sequence, child)
                self.scheduler.running.append(child)
                children.append(child)
        return children

    def generate_stream(
        self,
        prompts: SequenceABC[str] | str,
        sampling_params: SamplingParams | SequenceABC[SamplingParams] | None = None,
        max_tokens: int = 16,
        skip_special_tokens: bool = True,
        ignore_eos: bool = False,
    ) -> Iterator[StreamUpdate]:
        """Yield each token as it is produced, across all prompts, interleaved.

        An update's text is the delta of the whole decoded output rather than the single
        token decoded alone, since a token is not a character.
        """
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        sequences = self._admit(prompt_list, sampling_params, max_tokens, ignore_eos)

        # Branches share their prompt's index; the leader is sample 0, the rest fork later.
        index_of = {sequence.seq_id: index for index, sequence in enumerate(sequences)}
        sample_of = {sequence.seq_id: 0 for sequence in sequences}
        next_sample = [1] * len(sequences)
        emitted_chars: dict[int, int] = {sequence.seq_id: 0 for sequence in sequences}
        done_seen: set[int] = set()
        owned = list(sequences)

        self._forked.clear()
        started = time.perf_counter()
        remaining = len(sequences)
        try:
            while remaining and self.scheduler.num_unfinished:
                emitted = self.step()

                # Register branches forked this step before their first token is read.
                for child in self._newly_forked:
                    index = index_of[child.parent_id]
                    index_of[child.seq_id] = index
                    sample_of[child.seq_id] = next_sample[index]
                    next_sample[index] += 1
                    emitted_chars[child.seq_id] = 0
                    owned.append(child)
                    remaining += 1

                # A speculative run appears several times; only the last update is final.
                final_update = {
                    sequence.seq_id: position for position, (sequence, _) in enumerate(emitted)
                }

                for position, (sequence, token_id) in enumerate(emitted):
                    index = index_of.get(sequence.seq_id)
                    if index is None:
                        continue  # someone else's request, sharing the engine
                    text = self.tokenizer.decode(
                        sequence.output_token_ids, skip_special_tokens=skip_special_tokens
                    )
                    delta = text[emitted_chars[sequence.seq_id] :]
                    emitted_chars[sequence.seq_id] = len(text)

                    is_last = position == final_update[sequence.seq_id]
                    finished = sequence.is_done() and is_last
                    if finished and sequence.seq_id not in done_seen:
                        done_seen.add(sequence.seq_id)
                        remaining -= 1
                    yield StreamUpdate(
                        index=index,
                        seq_id=sequence.seq_id,
                        text=delta,
                        token_id=token_id,
                        finished=finished,
                        finish_reason=sequence.finish_reason,
                        sample_index=sample_of[sequence.seq_id],
                    )
        finally:
            # A caller that stops reading closes the generator here, pages still held.
            self._release(owned)
            self.stats.elapsed += time.perf_counter() - started

    def generate(
        self,
        prompts: SequenceABC[str] | str,
        sampling_params: SamplingParams | SequenceABC[SamplingParams] | None = None,
        max_tokens: int = 16,
        skip_special_tokens: bool = True,
        ignore_eos: bool = False,
    ) -> list[Completion]:
        """Run every prompt to completion and return them in the order given.

        A prompt requested with n > 1 yields n completions in sample order. The body is
        the streaming API drained, so the two cannot drift apart.
        """
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)

        # Keyed by (prompt index, sample index), since branches appear as they fork.
        texts: dict[tuple[int, int], list[str]] = {}
        tokens: dict[tuple[int, int], list[int]] = {}
        reasons: dict[tuple[int, int], str] = {}
        ids: dict[tuple[int, int], int] = {}

        for update in self.generate_stream(
            prompt_list, sampling_params, max_tokens, skip_special_tokens, ignore_eos
        ):
            key = (update.index, update.sample_index)
            texts.setdefault(key, []).append(update.text)
            tokens.setdefault(key, []).append(update.token_id)
            ids[key] = update.seq_id
            if update.finish_reason is not None:
                reasons[key] = update.finish_reason

        completions: list[Completion] = []
        for index, prompt in enumerate(prompt_list):
            samples = sorted(sample_index for (i, sample_index) in texts if i == index)
            if not samples:
                samples = [0]  # nothing came out at all: a fully aborted request
            for sample_index in samples:
                key = (index, sample_index)
                completions.append(
                    Completion(
                        prompt=prompt,
                        text="".join(texts.get(key, [])),
                        token_ids=tuple(tokens.get(key, [])),
                        finish_reason=reasons.get(key, "aborted"),
                        seq_id=ids.get(key, -1),
                        sample_index=sample_index,
                    )
                )
        return completions

    def _admit(
        self,
        prompts: Iterable[str],
        sampling_params: SamplingParams | SequenceABC[SamplingParams] | None,
        max_tokens: int,
        ignore_eos: bool = False,
    ) -> list[Sequence]:
        prompt_list = list(prompts)
        if sampling_params is None or isinstance(sampling_params, SamplingParams):
            params = [sampling_params] * len(prompt_list)
        else:
            params = list(sampling_params)
            if len(params) != len(prompt_list):
                raise ValueError(f"got {len(params)} sampling params for {len(prompt_list)} prompts")
        return [
            self.add_request(prompt, parameter, max_tokens, ignore_eos)
            for prompt, parameter in zip(prompt_list, params, strict=True)
        ]

    def _release(self, sequences: SequenceABC[Sequence]) -> None:
        """Drop the given sequences from every in-flight structure, blocks included.

        Finished ones are forgotten too: `Scheduler.finished` is read only by the tests,
        and retaining it would hold every prompt of a long run in memory.
        """
        for sequence in sequences:
            if sequence in self.scheduler.running:
                self.scheduler.running.remove(sequence)
            elif sequence in self.scheduler.waiting:
                self.scheduler.waiting.remove(sequence)
            if sequence in self.scheduler.finished:
                self.scheduler.finished.remove(sequence)
            if self.spec is not None:
                self.spec.release(sequence)
            self.manager.free(sequence)

    def __repr__(self) -> str:
        return (
            f"LLM({self.config.num_hidden_layers} layers, {self.device.type}, "
            f"{self.manager.num_blocks} blocks of {self.manager.block_size} "
            f"= {self.kv_cache_bytes / 2**30:.2f} GiB of KV cache)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate with the Mini-vLLM engine.",
        epilog='example: python -m mini_vllm.engine "The capital of France is" --stream',
    )
    parser.add_argument("prompt", nargs="?", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 means greedy")
    parser.add_argument("--stream", action="store_true", help="print tokens as they arrive")
    parser.add_argument("--prefix-caching", action="store_true")
    parser.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8"])
    parser.add_argument("--speculative-tokens", type=int, default=0)
    parser.add_argument("--no-cuda-kernels", action="store_true")
    args = parser.parse_args()

    print(f"loading {args.model} ...")
    llm = LLM(
        args.model,
        enable_prefix_caching=args.prefix_caching,
        kv_cache_dtype=args.kv_cache_dtype,
        num_speculative_tokens=args.speculative_tokens,
        use_cuda_kernels=not args.no_cuda_kernels,
    )
    print(llm)
    params = SamplingParams(temperature=args.temperature)

    started = time.perf_counter()
    if args.stream:
        print(f"\n{args.prompt}", end="", flush=True)
        for update in llm.generate_stream(args.prompt, params, max_tokens=args.max_tokens):
            print(update.text, end="", flush=True)
        print()
        generated = llm.stats.generated_tokens
    else:
        completion = llm.generate(args.prompt, params, max_tokens=args.max_tokens)[0]
        print(f"\n{args.prompt}\033[1m{completion.text}\033[0m")
        generated = completion.num_tokens
    elapsed = time.perf_counter() - started

    print(f"\n{generated} tokens in {elapsed:.2f}s ({generated / max(elapsed, 1e-9):.1f} tok/s)")
    if llm.spec is not None:
        stats = llm.spec.stats
        print(
            f"speculation: acceptance {stats.acceptance_rate:.3f}, "
            f"{stats.tokens_per_step:.2f} tokens per target pass"
        )


if __name__ == "__main__":
    main()
