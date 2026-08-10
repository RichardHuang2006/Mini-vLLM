"""Step 4.9 — the engine, and the one import a caller needs.

::

    from mini_vllm import LLM

    llm = LLM()
    for completion in llm.generate(["The capital of France is"], max_tokens=32):
        print(completion.text)

Everything below this line has been built already; this file is the wiring, and it is
short on purpose. The pieces and what each contributes to one iteration:

* `Scheduler` decides which sequences run and how many tokens each contributes.
* `BlockManager` backs that decision with pages, and refuses it when it cannot.
* `PagedModelRunner` turns it into a single ragged forward pass.
* `Qwen3Paged` runs the 28 layers against the pool.
* `sample` picks a token per sequence; the tokenizer turns them back into text.

The loop is `step()`, and every request in flight advances by exactly one iteration of
it. That is what continuous batching means concretely: there is no per-request loop
anywhere in this file, because a request is not a unit of execution — an iteration is.

:meth:`LLM.generate` is written on top of :meth:`LLM.generate_stream` rather than
beside it. Two loops that must agree token for token is a bug waiting to happen, and
"the batch API is the streaming API, drained" is a claim `test_engine.py` can check.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Sequence as SequenceABC
from dataclasses import dataclass, replace

import torch

from mini_vllm.block.block_manager import BlockManager
from mini_vllm.block.kv_pool import PagedKvPool
from mini_vllm.generate import eos_token_ids_for
from mini_vllm.model.loader import DEFAULT_MODEL_ID, load_weights, resolve_model_path
from mini_vllm.model.qwen3_paged import Qwen3Paged
from mini_vllm.sampler import SamplingParams
from mini_vllm.serve.runner import PagedModelRunner
from mini_vllm.serve.scheduler import Scheduler, SchedulerConfig
from mini_vllm.serve.sequence import Sequence

__all__ = ["LLM", "Completion", "EngineStats", "StreamUpdate"]

# What fraction of the memory still free *after the weights are resident* goes to the
# KV pool. Deliberately not 0.9: activations for a 2048-token batch, the logits tensor
# at 151k vocabulary, and cuBLAS workspaces all come out of what is left, and a pool
# sized to the last byte turns a long prompt into an out-of-memory error rather than a
# queued request.
DEFAULT_KV_FRACTION = 0.5

# Blocks to allocate when there is no device memory to measure. Enough for a handful of
# short sequences, which is all a CPU run is ever going to want.
CPU_BLOCKS = 512


@dataclass(frozen=True)
class Completion:
    """One finished request."""

    prompt: str
    text: str
    token_ids: tuple[int, ...]
    finish_reason: str
    seq_id: int

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class StreamUpdate:
    """One token, as it is produced.

    `index` is the position of the prompt in the list handed to `generate_stream`, not
    the sequence id: with continuous batching, requests finish out of order and in
    interleaved pieces, so the caller needs to know which of *its* prompts a token
    belongs to.
    """

    index: int
    seq_id: int
    text: str
    token_id: int
    finished: bool
    finish_reason: str | None = None


@dataclass
class EngineStats:
    """Counters for the whole engine's life, read by the Phase 5 benchmarks."""

    iterations: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    preemptions: int = 0
    elapsed: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.elapsed if self.elapsed else 0.0

    @property
    def tokens_per_iteration(self) -> float:
        total = self.prompt_tokens + self.generated_tokens
        return total / self.iterations if self.iterations else 0.0


class LLM:
    """A paged-attention inference engine for Qwen3.

    ::

        llm = LLM("Qwen/Qwen3-0.6B")
        completions = llm.generate(prompts, max_tokens=64)

    The constructor's defaults are the interesting part, because they are the two
    numbers that decide how many requests can be in flight:

    * `num_blocks` defaults to as many pages as fit in `kv_fraction` of the memory
      free once the weights are loaded. At 16 tokens a page and Qwen3-0.6B's 8 KV
      heads over 28 layers, a page is 448 KB — so the pool is thousands of pages, and
      the request count is bounded by the pool rather than by a configured maximum.
    * `max_batched_tokens` is the compute budget per iteration, and `chunk_size`
      bounds any single prefill's share of it.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        num_blocks: int | None = None,
        block_size: int = 16,
        max_batched_tokens: int = 2048,
        max_sequences: int = 32,
        chunk_size: int = 512,
        enable_chunked_prefill: bool = True,
        use_cuda_kernels: bool = True,
        kv_fraction: float = DEFAULT_KV_FRACTION,
    ) -> None:
        from transformers import AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        path = resolve_model_path(model)
        weights, config = load_weights(path, device=device)
        if dtype is not None and dtype != config.dtype:
            # Serving is bf16; fp32 is for the tests that want an exact answer. In bf16
            # the top two logits of a Qwen3 step are often one rounding apart, so greedy
            # decoding has genuine ties and two correct implementations pick differently
            # — `test_engine.py` uses fp32 to take that ambiguity off the table.
            weights = {name: tensor.to(dtype) for name, tensor in weights.items()}
            config = replace(config, dtype=dtype)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.stop_token_ids = eos_token_ids_for(path) or (self.tokenizer.eos_token_id,)

        if num_blocks is None:
            num_blocks = self.blocks_that_fit(config, block_size, self.device, kv_fraction)
        self.manager = BlockManager(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=config.dtype,
            device=self.device,
        )

        self.model = Qwen3Paged(config, weights, self.manager, use_cuda=use_cuda_kernels)
        self.scheduler = Scheduler(
            SchedulerConfig(
                max_batched_tokens=max_batched_tokens,
                max_sequences=max_sequences,
                chunk_size=chunk_size,
                enable_chunked_prefill=enable_chunked_prefill,
            ),
            manager=self.manager,
        )
        self.runner = PagedModelRunner(self.model, self.manager, self.device)
        self.stats = EngineStats()

    # ------------------------------------------------------------------- sizing

    @staticmethod
    def blocks_that_fit(config, block_size: int, device: torch.device, fraction: float) -> int:
        """How many pages fit in `fraction` of what is free right now.

        Measured *after* the weights are resident, so this is the memory actually
        available rather than the card's capacity — the difference is 1.2 GB for this
        model, which on an 8 GB laptop GPU is most of the answer.
        """
        per_block = PagedKvPool.bytes_for(
            num_layers=config.num_hidden_layers,
            num_blocks=1,
            block_size=block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=config.dtype,
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
            self.config.dtype,
        )

    # ---------------------------------------------------------------- admission

    def add_request(
        self,
        prompt: str | SequenceABC[int],
        sampling_params: SamplingParams | None = None,
        max_tokens: int = 16,
    ) -> Sequence:
        """Tokenize (if needed), wrap in a `Sequence`, and enqueue it."""
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
            stop_token_ids=tuple(self.stop_token_ids),
        )
        self.scheduler.add(sequence)
        self.stats.prompt_tokens += len(token_ids)
        return sequence

    # --------------------------------------------------------------- the loop

    def step(self) -> list[tuple[Sequence, int]]:
        """Advance every sequence in flight by one iteration.

        Returns the `(sequence, token)` pairs that produced a token, which is not the
        same as the sequences that ran: a prefill chunk that stopped in the middle of a
        prompt computed logits for a position the prompt itself already answers, and
        sampling from those would invent a token the caller never asked for.
        """
        output = self.scheduler.schedule()
        logits = self.runner.execute(output)
        tokens = self.runner.sample_tokens(output, logits)
        finished = self.scheduler.commit(output, tokens)

        emitted = [
            (sequence, sequence.output_token_ids[-1])
            for sequence, _ in output.scheduled
            if not sequence.is_prefill() and sequence.output_token_ids
        ]

        for sequence in finished:
            self.runner.free(sequence)

        self.stats.iterations += 1
        self.stats.generated_tokens += len(emitted)
        self.stats.preemptions += len(output.preempted)
        return emitted

    def generate_stream(
        self,
        prompts: SequenceABC[str] | str,
        sampling_params: SamplingParams | SequenceABC[SamplingParams] | None = None,
        max_tokens: int = 16,
        skip_special_tokens: bool = True,
    ) -> Iterator[StreamUpdate]:
        """Yield each token as it is produced, across all prompts, interleaved.

        Interleaved is the honest shape: the engine runs all of the prompts at once, so
        a stream that promised prompt 0's tokens before prompt 1's would have to buffer
        one to deliver the other. Each update says which prompt it belongs to.

        The text of an update is the *delta* — what the decoded output grew by. It is
        computed by re-decoding the whole output and taking the new suffix, rather than
        decoding the single token, because a token is not a character: the two-token
        sequence for a byte pair in an emoji decodes to nothing then to the emoji, and
        decoding tokens independently would emit a replacement character forever.
        """
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        sequences = self._admit(prompt_list, sampling_params, max_tokens)
        index_of = {sequence.seq_id: index for index, sequence in enumerate(sequences)}
        emitted_chars = [0] * len(sequences)

        started = time.perf_counter()
        remaining = len(sequences)
        try:
            while remaining and self.scheduler.num_unfinished:
                for sequence, token_id in self.step():
                    index = index_of.get(sequence.seq_id)
                    if index is None:
                        continue  # someone else's request, sharing the engine
                    text = self.tokenizer.decode(
                        sequence.output_token_ids, skip_special_tokens=skip_special_tokens
                    )
                    delta, emitted_chars[index] = text[emitted_chars[index] :], len(text)
                    finished = sequence.is_done()
                    remaining -= int(finished)
                    yield StreamUpdate(
                        index=index,
                        seq_id=sequence.seq_id,
                        text=delta,
                        token_id=token_id,
                        finished=finished,
                        finish_reason=sequence.finish_reason,
                    )
        finally:
            # A caller that stops reading — `break` out of the loop, or an exception —
            # closes the generator here, and the sequences it left in flight still hold
            # pages. Without this the pool leaks a request's worth of blocks per
            # abandoned stream, and the engine dies a few hundred requests later with an
            # OutOfBlocks that has nothing to do with the request that hit it.
            self._release(sequences)
            self.stats.elapsed += time.perf_counter() - started

    def generate(
        self,
        prompts: SequenceABC[str] | str,
        sampling_params: SamplingParams | SequenceABC[SamplingParams] | None = None,
        max_tokens: int = 16,
        skip_special_tokens: bool = True,
    ) -> list[Completion]:
        """Run every prompt to completion and return them in the order given.

        In the order given, though they did not finish in that order: a 20-token
        request admitted beside a 500-token one finishes 480 iterations earlier, and
        making the caller reassemble that is what a batch API exists to avoid.

        The body is the streaming API, drained. Reassembling deltas is exactly what a
        caller of the stream would do, and doing it here rather than writing a second
        loop is what makes "they agree token for token" true by construction rather
        than by testing.
        """
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        texts: list[list[str]] = [[] for _ in prompt_list]
        tokens: list[list[int]] = [[] for _ in prompt_list]
        reasons: list[str] = ["aborted"] * len(prompt_list)
        ids: list[int] = [-1] * len(prompt_list)

        for update in self.generate_stream(
            prompt_list, sampling_params, max_tokens, skip_special_tokens
        ):
            texts[update.index].append(update.text)
            tokens[update.index].append(update.token_id)
            ids[update.index] = update.seq_id
            if update.finish_reason is not None:
                reasons[update.index] = update.finish_reason

        return [
            Completion(
                prompt=prompt,
                text="".join(texts[index]),
                token_ids=tuple(tokens[index]),
                finish_reason=reasons[index],
                seq_id=ids[index],
            )
            for index, prompt in enumerate(prompt_list)
        ]

    # ------------------------------------------------------------------ helpers

    def _admit(
        self,
        prompts: Iterable[str],
        sampling_params: SamplingParams | SequenceABC[SamplingParams] | None,
        max_tokens: int,
    ) -> list[Sequence]:
        prompt_list = list(prompts)
        if sampling_params is None or isinstance(sampling_params, SamplingParams):
            params = [sampling_params] * len(prompt_list)
        else:
            params = list(sampling_params)
            if len(params) != len(prompt_list):
                raise ValueError(
                    f"got {len(params)} sampling params for {len(prompt_list)} prompts"
                )
        return [
            self.add_request(prompt, parameter, max_tokens)
            for prompt, parameter in zip(prompt_list, params, strict=True)
        ]

    def _release(self, sequences: SequenceABC[Sequence]) -> None:
        """Drop anything of ours still in flight, blocks included.

        Also forgets the finished ones. `Scheduler.finished` is a record of the
        iteration a request completed in, useful to a test and to nobody else; keeping
        it for the engine's whole life would hold every prompt and completion of a
        million-request run in memory.
        """
        for sequence in sequences:
            if sequence in self.scheduler.running:
                self.scheduler.running.remove(sequence)
            elif sequence in self.scheduler.waiting:
                self.scheduler.waiting.remove(sequence)
            if sequence in self.scheduler.finished:
                self.scheduler.finished.remove(sequence)
            self.manager.free(sequence)

    def __repr__(self) -> str:
        return (
            f"LLM({self.config.num_hidden_layers} layers, {self.device.type}, "
            f"{self.manager.num_blocks} blocks of {self.manager.block_size} "
            f"= {self.kv_cache_bytes / 2**30:.2f} GiB of KV cache)"
        )
