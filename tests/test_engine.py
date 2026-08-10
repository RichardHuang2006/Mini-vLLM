"""Step 4.9 — the engine, end to end.

Three kinds of test, in order of how much they can tell you when they fail:

* **Cross-implementation identity.** The paged engine against the dense one from
  Step 4.3, on the same weights and the same prompts. Everything between them is
  different — ragged batch, paged cache, a kernel that gathers K and V through a block
  table — so if the tokens agree, the whole of Phase 4 agrees with the model Phase 2
  established. This is the test to reach for first when something breaks.
* **Pressure.** A pool small enough to force preemption, and one too small to run
  anything at all. Preemption is the path an engine takes only under load, which is
  exactly where nobody is watching it, so it is worth pinning at leisure.
* **The API**, on real weights: token-identical to `transformers.generate` for sixteen
  varied prompts, which is the milestone this step exists to reach.

The tiny-model tests deliberately do not go through `LLM`. `LLM` loads a checkpoint and
a tokenizer, and a test that needs 1.2 GB of weights to check that a preempted sequence
resumes correctly is a test nobody runs.
"""

from __future__ import annotations

import pytest
import torch
from conftest import assert_tokens_equal, config_from_hf, weights_from_hf

from mini_vllm.block.block_manager import BlockManager, OutOfBlocks
from mini_vllm.model.loader import resolve_model_path
from mini_vllm.model.qwen3_cached import Qwen3Cached
from mini_vllm.model.qwen3_paged import Qwen3Paged
from mini_vllm.sampler import SamplingParams
from mini_vllm.serve.runner import PagedModelRunner
from mini_vllm.serve.scheduler import DenseModelRunner, Scheduler, SchedulerConfig
from mini_vllm.serve.sequence import Sequence, SequenceStatus

GREEDY = SamplingParams(temperature=0.0)

# Small enough that a handful of short sequences cross block boundaries several times,
# so the page-boundary paths are exercised by every test rather than by a special one.
BLOCK_SIZE = 8

PROMPTS = [
    [1, 2, 3],
    [5],
    [7, 7, 7, 7, 7, 7, 7, 7, 7],
    [2, 4],
    [9, 8, 7, 6, 5],
    [1],
]


def make(prompt: list[int], max_tokens: int = 6) -> Sequence:
    return Sequence(prompt_token_ids=list(prompt), max_tokens=max_tokens, sampling_params=GREEDY)


class Engine:
    """The Step 4.9 loop over a tiny model, without a tokenizer or a checkpoint.

    Exactly what `LLM.step` does — schedule, execute, sample, commit, free — and it is
    spelled out here rather than reused so that a change to the engine's loop has to be
    reflected in a test that reads like the loop it is checking.
    """

    def __init__(self, model, manager: BlockManager, **config) -> None:
        self.manager = manager
        self.scheduler = Scheduler(SchedulerConfig(**config), manager=manager)
        self.runner = PagedModelRunner(model, manager)
        self.preemptions = 0
        self.iterations = 0

    def run(self, sequences: list[Sequence], max_iterations: int = 500) -> dict[int, list[int]]:
        self.scheduler.add_all(sequences)
        while self.scheduler.has_work:
            self.iterations += 1
            assert self.iterations <= max_iterations, "the engine is not making progress"

            output = self.scheduler.schedule()
            logits = self.runner.execute(output)
            tokens = self.runner.sample_tokens(output, logits)
            self.preemptions += len(output.preempted)
            for finished in self.scheduler.commit(output, tokens):
                self.runner.free(finished)

        return {sequence.seq_id: sequence.output_token_ids for sequence in sequences}


@pytest.fixture
def weights(tiny_qwen3, device):
    """The tiny model's weights and config, in fp32 on the GPU."""
    theirs = tiny_qwen3.to(device=device, dtype=torch.float32)
    return weights_from_hf(theirs), config_from_hf(theirs)


def paged(weights_and_config, num_blocks: int = 256, use_cuda: bool = True):
    """A paged model and its manager, sized by block count."""
    weights, config = weights_and_config
    manager = BlockManager(
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=config.dtype,
        device=weights["embedding"].device,
    )
    return Qwen3Paged(config, weights, manager, use_cuda=use_cuda), manager


def dense_reference(weights_and_config, prompts: list[list[int]], max_tokens: int) -> list[list[int]]:
    """What Step 4.3's dense runner produces for each prompt, run alone.

    One sequence at a time, one forward pass per token, no paging anywhere: the slowest
    path in the project and the one with the fewest moving parts.
    """
    weights, config = weights_and_config
    model = Qwen3Cached(config, weights, use_cuda=False)
    outputs = []
    for prompt in prompts:
        scheduler = Scheduler()
        runner = DenseModelRunner(model)
        sequence = make(prompt, max_tokens)
        scheduler.add(sequence)
        while scheduler.has_work:
            output = scheduler.schedule()
            logits = runner.execute(output)
            scheduler.commit(output, runner.sample_tokens(output, logits))
        outputs.append(sequence.output_token_ids)
    return outputs


# --------------------------------------------------------------------- the batch


@pytest.mark.cuda
def test_the_whole_iteration_is_one_forward_pass(weights):
    """Six sequences, one launch — the thing the dense runner could not do.

    `DenseModelRunner.execute` loops and calls the model once per sequence.
    `PagedModelRunner.execute` calls it once, and the logits come back with one row per
    sequence, so the assertion is on the shape of the result rather than on a counter.
    """
    model, manager = paged(weights)
    engine = Engine(model, manager)
    sequences = [make(prompt) for prompt in PROMPTS]
    engine.scheduler.add_all(sequences)

    output = engine.scheduler.schedule()
    logits = engine.runner.execute(output)

    assert logits.shape == (len(PROMPTS), model.config.vocab_size)
    assert engine.runner.build(output).total_tokens == sum(len(p) for p in PROMPTS)


@pytest.mark.cuda
def test_the_logits_row_is_each_sequence_s_last_position(weights):
    """Not the last row of the batch, and not the first: each sequence's own end.

    Getting this wrong is the subtlest bug in a ragged batch, because with equal-length
    sequences every wrong answer is also the right one. The check is against the model
    run over one sequence alone, which has only one row to choose from.
    """
    model, manager = paged(weights)
    engine = Engine(model, manager)
    sequences = [make(prompt) for prompt in PROMPTS]
    engine.scheduler.add_all(sequences)

    output = engine.scheduler.schedule()
    batched = engine.runner.execute(output)

    for index, prompt in enumerate(PROMPTS):
        alone_model, alone_manager = paged(weights)
        alone_engine = Engine(alone_model, alone_manager)
        alone_engine.scheduler.add(make(prompt))
        row = alone_engine.runner.execute(alone_engine.scheduler.schedule())

        torch.testing.assert_close(batched[index], row[0], msg=f"prompt {index}")


# ---------------------------------------------------------------------- identity


@pytest.mark.cuda
def test_the_paged_engine_is_token_identical_to_the_dense_one(weights):
    """Phase 4 against Phase 2, on the same weights.

    Six requests of different lengths in one engine, against each run alone through the
    dense cache. Between the two: a ragged batch, a paged cache with sequences
    interleaved across pages, and a kernel that resolves every key's address through a
    block table. None of it may reach the tokens.
    """
    expected = dense_reference(weights, PROMPTS, max_tokens=6)

    model, manager = paged(weights)
    engine = Engine(model, manager, max_batched_tokens=16, max_sequences=3)
    sequences = [make(prompt) for prompt in PROMPTS]

    got = engine.run(sequences)

    for index, sequence in enumerate(sequences):
        assert_tokens_equal(
            got[sequence.seq_id], expected[index], msg=f"prompt {index} changed under paging"
        )


@pytest.mark.cuda
def test_a_chunked_prefill_is_token_identical_too(weights):
    """The same claim with the prompts split across iterations.

    A 9-token prompt at `chunk_size=4` is three chunks, and the last one is what emits
    the first token. If positions or the causal offset were wrong at a chunk boundary
    the tokens would differ here and nowhere else.
    """
    prompts = [[7, 7, 7, 7, 7, 7, 7, 7, 7], [9, 8, 7, 6, 5]]
    expected = dense_reference(weights, prompts, max_tokens=5)

    model, manager = paged(weights)
    engine = Engine(model, manager, chunk_size=4, max_batched_tokens=6)
    sequences = [make(prompt, 5) for prompt in prompts]

    got = engine.run(sequences)

    assert engine.iterations > len(prompts), "the prompts must actually have been chunked"
    for index, sequence in enumerate(sequences):
        assert_tokens_equal(got[sequence.seq_id], expected[index], msg=f"prompt {index}")


def test_the_oracle_path_gives_the_same_tokens_without_a_gpu(tiny_qwen3):
    """`use_cuda=False` on the CPU, gathering each sequence's cache the slow way.

    Same tokens, no kernel, no device: the engine's correctness does not depend on
    Phase 3, which is what keeps the CUDA path honest — the two are compared against
    each other, not each against itself.
    """
    theirs = tiny_qwen3.to(dtype=torch.float32)
    bundle = (weights_from_hf(theirs), config_from_hf(theirs))
    expected = dense_reference(bundle, PROMPTS[:3], max_tokens=4)

    model, manager = paged(bundle, num_blocks=64, use_cuda=False)
    engine = Engine(model, manager, max_batched_tokens=12, max_sequences=2)
    sequences = [make(prompt, 4) for prompt in PROMPTS[:3]]

    got = engine.run(sequences)

    for index, sequence in enumerate(sequences):
        assert_tokens_equal(got[sequence.seq_id], expected[index], msg=f"prompt {index}")


# ---------------------------------------------------------------------- pressure


@pytest.mark.cuda
def test_every_block_comes_back(weights):
    """A finished request's pages return to the pool, all of them.

    The failure this catches is invisible for one request and fatal for a thousand: the
    engine serves a few hundred and then cannot admit anything, with the error landing
    on a request that did nothing wrong.
    """
    model, manager = paged(weights, num_blocks=64)
    engine = Engine(model, manager, max_batched_tokens=16, max_sequences=3)

    engine.run([make(prompt) for prompt in PROMPTS])

    manager.check_no_leaks()


@pytest.mark.cuda
def test_a_small_pool_forces_preemption_and_changes_nothing(weights):
    """The load path, and the invariant that survives it.

    Six requests through a pool that cannot hold them all at once, so the scheduler
    evicts the newest running sequence to keep the oldest moving. A preempted sequence
    loses its pages and re-prefills over its prompt *and* the output it had already
    emitted — and must arrive at exactly the tokens it would have produced alone.
    """
    expected = dense_reference(weights, PROMPTS, max_tokens=12)

    # Five pages of 8 slots against six requests that grow into fourteen between them.
    # Admission fills the pool with the first four; the preemptions come from the
    # *decodes* after that, when a running sequence crosses a page boundary and there is
    # nothing free — which is the shape this actually takes under load.
    model, manager = paged(weights, num_blocks=5)
    engine = Engine(model, manager, max_batched_tokens=16, max_sequences=6)
    sequences = [make(prompt, 12) for prompt in PROMPTS]

    got = engine.run(sequences)

    assert engine.preemptions > 0, "this pool is too big to be testing preemption"
    for index, sequence in enumerate(sequences):
        assert_tokens_equal(
            got[sequence.seq_id], expected[index], msg=f"prompt {index} changed under preemption"
        )
    manager.check_no_leaks()


@pytest.mark.cuda
def test_a_preempted_sequence_gives_its_blocks_back_immediately(weights):
    """Eviction is only useful if the pages are free the same iteration.

    Preemption that dropped the block table without decrefing the pool would look
    correct — the sequence re-prefills, the tokens come out right — and would free
    nothing, so the scheduler would preempt again, and again, and never make room.
    """
    model, manager = paged(weights, num_blocks=32)
    engine = Engine(model, manager, max_sequences=4)
    sequences = [make(prompt) for prompt in PROMPTS[:2]]
    engine.scheduler.add_all(sequences)

    output = engine.scheduler.schedule()
    engine.scheduler.commit(output, engine.runner.sample_tokens(output, engine.runner.execute(output)))
    held = manager.num_blocks - manager.num_free_blocks

    engine.scheduler.preempt(sequences[1])

    assert manager.num_free_blocks > manager.num_blocks - held
    assert sequences[1].block_table is None
    assert sequences[1].status is SequenceStatus.PREEMPTED


@pytest.mark.cuda
def test_dropping_a_table_without_freeing_the_blocks_is_refused(weights):
    """The leak from the previous test, caught at the sequence rather than found later.

    `reset_for_recompute` is the only way a sequence forgets its pages, so refusing to
    do it while the pool still believes they are held makes the leak unreachable
    instead of merely tested for.
    """
    model, manager = paged(weights, num_blocks=32)
    sequence = make([1, 2, 3])
    manager.allocate(sequence)

    with pytest.raises(ValueError, match="still holds"):
        sequence.reset_for_recompute()

    manager.free(sequence)


@pytest.mark.cuda
def test_a_pool_too_small_for_one_sequence_says_so(weights):
    """Not a hang, and not a wrong answer: a message naming the fix.

    With one block of 8 slots and a 9-token prompt there is nothing to preempt and
    nothing that fits, and an engine returning an empty batch here would spin forever
    with a request outstanding.
    """
    model, manager = paged(weights, num_blocks=1)
    engine = Engine(model, manager)

    with pytest.raises(OutOfBlocks, match="nothing can run"):
        engine.run([make([7] * 9)])


@pytest.mark.cuda
def test_admission_waits_rather_than_preempting_for_a_new_request(weights):
    """A queued request holds nothing, so evicting a running one to admit it is a loss.

    The asymmetry is the whole of the policy: memory pressure preempts *for a sequence
    already in flight*, and merely postpones a sequence that has not started.
    """
    # Three of the four pages go to the first request, so the second cannot have its two.
    model, manager = paged(weights, num_blocks=4)
    engine = Engine(model, manager, max_sequences=8)
    running, queued = make([1] * 24), make([9] * 16)
    engine.scheduler.add_all([running, queued])

    first = engine.scheduler.schedule()

    assert first.sequences == [running]
    assert not first.preempted
    assert queued.status is SequenceStatus.WAITING


# -------------------------------------------------------------------- the API


def real_engine(**kwargs):
    """An engine on the real checkpoint, skipping when it is not available."""
    from mini_vllm import LLM

    path = resolve_model_path()
    if not (path / "model.safetensors").is_file():
        pytest.skip("Qwen3-0.6B weights are not downloaded")
    if not torch.cuda.is_available():
        pytest.skip("the engine's own tests want the kernels")

    return LLM(**kwargs)


@pytest.fixture(scope="module")
def llm():
    """One fp32 engine for the whole module: the weights are 2.4 GB in fp32.

    **fp32 rather than the bf16 the engine actually serves in**, and the reason is the
    only interesting caveat in this file. In bf16 the top two logits of a Qwen3 step are
    frequently one rounding apart — a gap of 0.125 at a logit magnitude of 20 — so
    greedy decoding has genuine ties, and two correct implementations that accumulate in
    a different order pick differently. Six of these sixteen prompts diverge from
    `transformers` in bf16 for exactly that reason, all of them at a tie.

    Testing against that would mean a test that fails for a legitimate reason, which is
    worse than no test. fp32 removes the ambiguity — every prompt then agrees to the
    token — and `test_bf16_only_disagrees_at_a_tie` covers the bf16 path by showing that
    its disagreements are ties and nothing else.
    """
    return real_engine(dtype=torch.float32, num_blocks=192, max_sequences=16)


REAL_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time",
    "The three primary colors are",
    "Water boils at",
    "2 + 2 =",
    "The largest planet in the solar system is",
    "Shakespeare wrote",
    "In Python, a list comprehension",
    "The speed of light is approximately",
    "A neural network learns by",
    "The Pacific Ocean is",
    "To sort a list in Python you can",
    "The first president of the United States was",
    "Photosynthesis converts",
    "A paged attention kernel exists because",
]


@pytest.mark.oracle
def test_greedy_output_matches_transformers_for_sixteen_prompts(llm):
    """**The milestone.** Sixteen varied prompts, batched, token for token.

    Sixteen at once through one engine, against `transformers.generate` one prompt at a
    time: continuous batching, chunked prefill, a paged cache and hand-written kernels
    on one side, and a reference implementation running each request alone on the other.
    Greedy decoding makes the comparison exact, so there is no tolerance to argue
    about — either the engine is right or it is not.

    One prompt at a time on the reference side deliberately. Batching it would need
    left-padding, and a padded position is a position: it changes the reference's own
    answer, which would make a disagreement here ambiguous. The whole value of this test
    is that a disagreement is not.
    """
    from transformers import AutoModelForCausalLM

    theirs = AutoModelForCausalLM.from_pretrained(resolve_model_path(), dtype=llm.config.dtype)
    theirs = theirs.to(llm.device).eval()
    try:
        completions = llm.generate(REAL_PROMPTS, max_tokens=24)

        for prompt, completion in zip(REAL_PROMPTS, completions, strict=True):
            ids = llm.tokenizer(prompt, return_tensors="pt").input_ids.to(llm.device)
            reference = theirs.generate(
                ids,
                max_new_tokens=24,
                do_sample=False,
                pad_token_id=llm.tokenizer.eos_token_id,
            )
            assert_tokens_equal(
                completion.token_ids,
                reference[0, ids.shape[1] :],
                msg=f"prompt: {prompt!r}\nours:   {completion.text!r}",
            )
    finally:
        # Both models fp32 on one card is 5 GB, so the reference does not get to outlive
        # the test that needed it.
        del theirs
        torch.cuda.empty_cache()


@pytest.mark.oracle
def test_bf16_only_disagrees_at_a_tie(llm):
    """The dtype the engine actually serves in, and what it costs.

    bf16 keeps 8 bits of mantissa, so a logit near 20 is quantized to steps of 0.125 and
    two candidates can end up on adjacent representable values. When that happens the
    argmax is decided by rounding, and the fp32 engine and the bf16 one disagree without
    either being wrong.

    What would *not* be a tie is a bf16 run that picks a token the fp32 run does not rank
    second — which is what this pins down, and it is the honest form of the milestone for
    the dtype that ships.
    """
    exact = llm.generate(REAL_PROMPTS, max_tokens=24)
    fast = real_engine(dtype=torch.bfloat16, num_blocks=192, max_sequences=16)
    try:
        approximate = fast.generate(REAL_PROMPTS, max_tokens=24)
    finally:
        del fast
        torch.cuda.empty_cache()

    agreed = 0
    for prompt, ours, theirs in zip(REAL_PROMPTS, exact, approximate, strict=True):
        if ours.token_ids == theirs.token_ids:
            agreed += 1
            continue

        split = next(
            index
            for index in range(min(len(ours.token_ids), len(theirs.token_ids)))
            if ours.token_ids[index] != theirs.token_ids[index]
        )
        # Rank the bf16 choice under the fp32 model at the point they parted. A tie means
        # it was the runner-up; a bug means it was nowhere.
        prefix = list(llm.tokenizer(prompt).input_ids) + list(ours.token_ids[:split])
        logits = engine_logits(llm, prefix)
        ranked = logits.argsort(descending=True)[:2].tolist()

        assert theirs.token_ids[split] in ranked, (
            f"bf16 picked {theirs.token_ids[split]} at {split} for {prompt!r}, which fp32 "
            f"ranks below its top two {ranked} — that is not a rounding tie"
        )

    assert agreed >= len(REAL_PROMPTS) // 2, f"only {agreed} prompts survived bf16 unchanged"


def engine_logits(llm, token_ids: list[int]) -> torch.Tensor:
    """The next-token logits for a prefix, through one prefill iteration of the engine."""
    sequence = llm.add_request(token_ids, max_tokens=1)
    output = llm.scheduler.schedule()
    logits = llm.runner.execute(output)
    llm.scheduler.commit(output, [0])
    llm._release([sequence])
    return logits[0].float()


@pytest.mark.oracle
def test_the_batch_api_is_the_streaming_api_drained(llm):
    """Two entry points, one loop. The deltas must reassemble into the completion."""
    prompts = ["The capital of France is", "def fibonacci(n):"]
    batched = llm.generate(prompts, max_tokens=16)

    streamed: dict[int, str] = {0: "", 1: ""}
    tokens: dict[int, list[int]] = {0: [], 1: []}
    for update in llm.generate_stream(prompts, max_tokens=16):
        streamed[update.index] += update.text
        tokens[update.index].append(update.token_id)

    for index, completion in enumerate(batched):
        assert streamed[index] == completion.text
        assert_tokens_equal(tokens[index], completion.token_ids)


@pytest.mark.oracle
def test_a_stream_is_interleaved_across_prompts(llm):
    """Not prompt 0 to completion and then prompt 1: they run together.

    A stream that delivered one request at a time would be a batch API with extra
    steps, and would mean the engine was not batching.
    """
    updates = list(llm.generate_stream(["The capital of France is", "def fibonacci(n):"],
                                       max_tokens=8))
    indices = [update.index for update in updates]

    assert set(indices) == {0, 1}
    assert indices != sorted(indices), "the two requests must be interleaved"


@pytest.mark.oracle
def test_a_completion_says_why_it_stopped(llm):
    """`length` when it ran out of budget, `stop` when the model ended the turn."""
    capped = llm.generate("Once upon a time", max_tokens=4)[0]
    assert capped.finish_reason == "length"
    assert capped.num_tokens == 4

    # A chat turn the model finishes on its own well inside the budget.
    prompt = llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say the single word: hello"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ended = llm.generate(prompt, max_tokens=64)[0]

    assert ended.finish_reason == "stop", f"got {ended.text!r}"
    assert ended.num_tokens < 64
    assert llm.tokenizer.eos_token not in ended.text, "special tokens must not reach the text"


@pytest.mark.oracle
def test_an_abandoned_stream_returns_its_blocks(llm):
    """A caller that stops reading must not cost the engine a request's worth of pages.

    The `finally` in `generate_stream` is the only thing standing between a web server
    whose clients disconnect and an engine that stops admitting anything after a few
    hundred of them.
    """
    before = llm.manager.num_free_blocks

    for _update in llm.generate_stream(["Once upon a time"] * 4, max_tokens=64):
        break  # the disconnect

    assert llm.manager.num_free_blocks == before
    assert llm.scheduler.num_unfinished == 0


@pytest.mark.oracle
def test_sampling_parameters_are_per_request(llm):
    """One greedy request and one hot one, in the same iterations."""
    prompts = ["The capital of France is"] * 2
    params = [SamplingParams(temperature=0.0), SamplingParams(temperature=1.5, top_p=0.9)]

    completions = llm.generate(prompts, sampling_params=params, max_tokens=12)

    assert completions[0].text == llm.generate(prompts[:1], max_tokens=12)[0].text
    assert completions[1].num_tokens == 12


@pytest.mark.oracle
def test_the_engine_reports_what_it_did(llm):
    """The counters Phase 5 measures with, sanity-checked on a known run."""
    before = llm.stats.iterations
    llm.generate(["Once upon a time"] * 4, max_tokens=8)

    assert llm.stats.iterations - before <= 9, "4 prompts should share their iterations"
    assert llm.stats.tokens_per_second > 0
    assert llm.kv_cache_bytes > 0
