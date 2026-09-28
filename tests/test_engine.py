"""engine.py: the serving loop against the dense single-request loop, the stream against the
batch API, prefix caching, parallel sampling, stop tokens, and cleanup."""

import mlx.core as mx
import pytest
from utils import with_and_without_metal

from mini_vllm.engine import LLM, EngineConfig
from mini_vllm.generate import generate_with_kv_cache
from mini_vllm.models import from_mlx
from mini_vllm.sampler import SamplingParams


class LetterTokenizer:
    """Enough of a tokenizer for the tiny model's 512-token vocabulary: token i is one letter."""

    def __init__(self, eos_token_ids=()):
        self.eos_token_ids = set(eos_token_ids)

    def encode(self, text):
        return [ord(c) % 512 for c in text]

    def decode(self, ids):
        return "".join(chr(ord("a") + i % 26) for i in ids)


def engine(tiny_qwen3, eos_token_ids=(), use_metal=False, **config) -> LLM:
    config.setdefault("num_blocks", 64)
    config.setdefault("block_size", 4)
    return LLM(from_mlx(tiny_qwen3, use_metal), LetterTokenizer(eos_token_ids), EngineConfig(**config))


def assert_nothing_held(llm: LLM) -> None:
    assert not llm.scheduler.has_work
    assert llm.manager.pool.num_free == llm.kv.num_blocks, "blocks leaked"


PROMPTS = [[1, 2, 3], [5], [7] * 9, [2, 4], [9, 8, 7, 6, 5], [1], list(range(30, 70))]


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"max_batched_tokens": 12, "max_sequences": 3, "chunk_size": 8},  # chunked, piggybacked
        # 14 pages: the longest request (40 + 8 tokens) fits alone, and the seven together
        # (~30 pages) force steady preemption.
        {"num_blocks": 14, "max_batched_tokens": 16},
    ],
)
@with_and_without_metal
def test_every_request_matches_running_it_alone(tiny_qwen3, config, use_metal):
    llm = engine(tiny_qwen3, use_metal=use_metal, **config)
    completions = llm.generate(PROMPTS, max_tokens=8)

    # Against the dense loop on the pure-MLX model, so the kernels are checked too.
    pure = from_mlx(tiny_qwen3)
    for prompt, completion in zip(PROMPTS, completions, strict=True):
        assert completion.token_ids == generate_with_kv_cache(pure, prompt, 8)
        assert completion.finish_reason == "length"
    # A continuation that repeats one token would match whatever the history held.
    assert sum(len(set(c.token_ids)) for c in completions) > 3 * len(PROMPTS), "degenerate outputs"
    assert_nothing_held(llm)


def test_generate_is_the_stream_drained(tiny_qwen3):
    llm = engine(tiny_qwen3)
    streamed = {}
    for update in llm.generate_stream(PROMPTS, max_tokens=6):
        streamed.setdefault(update.index, []).append(update)

    completions = llm.generate(PROMPTS, max_tokens=6)
    for index, completion in enumerate(completions):
        updates = streamed[index]
        assert [u.token_id for u in updates] == completion.token_ids
        assert "".join(u.text for u in updates) == completion.text
        assert completion.text == llm.tokenizer.decode(completion.token_ids)
        assert [u.finish_reason for u in updates] == [None] * 5 + ["length"]


def test_prefix_caching_changes_nothing_and_hits(tiny_qwen3):
    shared = list(range(100, 124))
    prompts = [[*shared, tail] for tail in range(5)]
    plain = engine(tiny_qwen3).generate(prompts, max_tokens=6)

    llm = engine(tiny_qwen3, enable_prefix_caching=True, max_sequences=1)
    cached = llm.generate(prompts, max_tokens=6)

    assert [c.token_ids for c in cached] == [c.token_ids for c in plain]
    assert llm.manager.cache.num_cached_blocks >= len(shared) // 4, "the shared prefix was never cached"
    assert_nothing_held(llm)


def test_parallel_sampling_forks_after_the_prompt_and_shares_its_pages(tiny_qwen3):
    llm = engine(tiny_qwen3)
    prompt = list(range(1, 11))  # 10 tokens: 3 pages of 4, the last one partial
    llm.add_request(prompt, SamplingParams(temperature=0.0, n=3), max_tokens=5)

    emitted, forked = llm.step()

    assert len(forked) == 2 and len(emitted) == 3, "leader and both branches emit a first token"
    leader = forked[0][0]
    pages = list(leader.block_table.block_ids)
    assert all(branch.block_table.block_ids == pages for _, branch in forked), "not the leader's pages"
    assert all(llm.manager.pool.ref_counts[page] == 3 for page in pages)
    assert llm.manager.pool.num_free == 64 - 3, "the branches hold the prompt's pages, not copies"

    llm.step()  # the leader keeps the partial page; each branch copies it before writing
    assert llm.manager.pool.num_free == 64 - 3 - 2
    assert all(llm.manager.pool.ref_counts[page] == 3 for page in pages[:2]), "full pages stay shared"
    assert llm.manager.pool.ref_counts[pages[2]] == 1


def test_greedy_branches_are_identical_and_complete(tiny_qwen3):
    llm = engine(tiny_qwen3)
    prompts = [[1, 2, 3], list(range(20, 33))]
    completions = llm.generate(prompts, SamplingParams(temperature=0.0, n=3), max_tokens=6)

    assert [(c.prompt, c.sample_index) for c in completions] == [
        (p, s) for p in prompts for s in range(3)
    ], "prompt order, then sample order"
    for c in completions:
        assert c.token_ids == generate_with_kv_cache(llm.model, c.prompt, 6)
        assert len(set(c.token_ids)) > 1, "a repeated token would match whatever the branch attended to"
    assert_nothing_held(llm)


def test_sampled_branches_differ_and_replay_under_a_seed(tiny_qwen3):
    params = SamplingParams(temperature=1.0, n=4)
    first = engine(tiny_qwen3, seed=7).generate([[1, 2, 3]], params, max_tokens=8)
    again = engine(tiny_qwen3, seed=7).generate([[1, 2, 3]], params, max_tokens=8)

    assert [c.token_ids for c in first] == [c.token_ids for c in again]
    assert len({tuple(c.token_ids) for c in first}) > 1, "four samples at temperature 1 all agreed"


def test_a_stop_token_ends_a_request_and_stays_out_of_its_text(tiny_qwen3):
    free_run = engine(tiny_qwen3).generate([[4, 5, 6]], max_tokens=10)[0].token_ids
    stop = free_run[3]
    expected = free_run[: free_run.index(stop) + 1]

    llm = engine(tiny_qwen3, eos_token_ids={stop})
    completion = llm.generate([[4, 5, 6]], max_tokens=10)[0]

    assert completion.token_ids == expected and completion.finish_reason == "stop"
    assert completion.text == llm.tokenizer.decode(expected[:-1])
    ignored = llm.generate([[4, 5, 6]], max_tokens=10, ignore_eos=True)[0]
    assert ignored.token_ids == free_run and ignored.finish_reason == "length"


def test_abandoning_a_stream_releases_everything(tiny_qwen3):
    llm = engine(tiny_qwen3, max_sequences=2)
    stream = llm.generate_stream(PROMPTS, max_tokens=8)
    next(stream)
    assert llm.scheduler.has_work

    stream.close()

    assert_nothing_held(llm)


@with_and_without_metal
def test_an_fp8_kv_cache_serves_to_completion(tiny_qwen3, use_metal):
    llm = engine(tiny_qwen3, fp8_kv_cache=True, use_metal=use_metal)
    assert llm.kv.keys[0].dtype == mx.uint8
    completions = llm.generate(PROMPTS, max_tokens=8)
    assert all(len(c.token_ids) == 8 for c in completions)
    # The kernels quantize the same bytes and dequantize in fp32, as this fp32 model's oracle does.
    pure = engine(tiny_qwen3, fp8_kv_cache=True).generate(PROMPTS, max_tokens=8)
    assert [c.token_ids for c in completions] == [c.token_ids for c in pure]
    assert_nothing_held(llm)


@pytest.fixture(scope="module")
def real_llm(real_qwen3):
    mlx_model, tokenizer = real_qwen3
    return LLM(from_mlx(mlx_model), tokenizer, EngineConfig(num_blocks=512))


def test_the_real_model_serves_text(real_llm):
    prompts = ["The capital of France is", "def fibonacci(n):", "Paged attention stores the KV cache in"]
    completions = real_llm.generate(prompts, max_tokens=24)

    assert completions[0].text.startswith(" Paris"), "the first token keeps its leading space"
    for completion in completions:
        assert completion.text == real_llm.tokenizer.decode(completion.token_ids)
    streamed = ["", "", ""]
    for update in real_llm.generate_stream(prompts, max_tokens=24):
        streamed[update.index] += update.text
    assert streamed == [c.text for c in completions], "the same batch must decode the same way"
    assert_nothing_held(real_llm)


def test_default_pool_sizing_uses_the_memory_left(real_qwen3):
    from mini_vllm.engine import blocks_that_fit

    model = from_mlx(real_qwen3[0])
    c = model.config
    per_block = 2 * c.num_hidden_layers * 16 * c.num_key_value_heads * c.head_dim * 2
    left = mx.device_info()["max_recommended_working_set_size"] - mx.get_active_memory()

    assert blocks_that_fit(model, EngineConfig(kv_fraction=0.1)) == int(left * 0.1) // per_block
    assert blocks_that_fit(model, EngineConfig(kv_fraction=0.1, fp8_kv_cache=True)) == int(left * 0.1) // (
        per_block // 2
    ), "fp8 pages take half the bytes"
