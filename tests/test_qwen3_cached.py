"""Step 2.2 — the cached model against the uncached one.

The cache is an optimization, so the bar is not "close" but **identical**: for the
same tokens, the cached model must produce the same logits as Step 1.8's model and
the same greedy tokens as Step 1.9's loop. Anything less means it changed the
model's behaviour, which is the one thing an optimization may not do.

Most of these run on `tiny_qwen3` and need no download. The `oracle` ones close the
loop against `transformers.generate`.
"""

from __future__ import annotations

import pytest
import torch
from conftest import (
    assert_allclose,
    assert_relative_error_below,
    assert_tokens_equal,
    config_from_hf,
    qwen3_from_hf,
    weights_from_hf,
)

from mini_vllm.generate import eos_token_ids_for, generate_ids, generate_ids_cached
from mini_vllm.kernels import ops
from mini_vllm.kv_cache import DenseKvCache
from mini_vllm.model.loader import resolve_model_path
from mini_vllm.model.qwen3_cached import Qwen3Cached


@pytest.fixture
def pair(tiny_qwen3):
    """The uncached model and the cached one, sharing weights."""
    naive = qwen3_from_hf(tiny_qwen3)
    cached = Qwen3Cached(config_from_hf(tiny_qwen3), weights_from_hf(tiny_qwen3))
    return naive, cached, tiny_qwen3.config.vocab_size


# ---------------------------------------------------------------------- setup


def test_creates_one_cache_per_layer(pair):
    _naive, cached, _vocab = pair
    caches = cached.create_kv_cache()

    assert len(caches) == cached.config.num_hidden_layers
    assert all(isinstance(cache, DenseKvCache) for cache in caches)
    assert all(cache.offset == 0 for cache in caches)


def test_each_layer_gets_a_distinct_cache(pair):
    """Sharing one cache across layers would mix unrelated keys together."""
    _naive, cached, _vocab = pair
    caches = cached.create_kv_cache()
    assert len({id(cache) for cache in caches}) == len(caches)


def test_rejects_the_wrong_number_of_caches(pair):
    _naive, cached, vocab = pair
    with pytest.raises(ValueError, match="one per layer"):
        cached(torch.randint(0, vocab, (1, 3)), cached.create_kv_cache()[:1])


def test_the_uncached_model_is_untouched(pair):
    """The fork must not have changed Step 1.8's model, which is the oracle."""
    from mini_vllm.model.qwen3 import Qwen3

    naive, _cached, _vocab = pair
    assert isinstance(naive, Qwen3)
    assert not hasattr(naive, "create_kv_cache")


# ------------------------------------------------- prefill matches the oracle


@pytest.mark.parametrize(("batch", "length"), [(1, 1), (1, 8), (3, 12)])
def test_prefill_logits_match_the_uncached_model(pair, batch, length):
    """One full-prompt pass must equal the uncached forward exactly."""
    naive, cached, vocab = pair
    ids = torch.randint(0, vocab, (batch, length))

    assert_allclose(cached(ids, cached.create_kv_cache()), naive(ids))


def test_prefill_fills_every_cache_to_the_prompt_length(pair):
    _naive, cached, vocab = pair
    caches = cached.create_kv_cache()

    cached(torch.randint(0, vocab, (2, 9)), caches)

    assert all(cache.offset == 9 for cache in caches)
    for cache in caches:
        assert cache.keys.shape[-2] == 9
        assert cache.keys.shape[1] == cached.config.num_key_value_heads


def test_last_only_returns_just_the_final_position(pair):
    """The LM head shortcut must not change the value it produces."""
    _naive, cached, vocab = pair
    ids = torch.randint(0, vocab, (2, 7))

    everything = cached(ids, cached.create_kv_cache())
    final = cached(ids, cached.create_kv_cache(), last_only=True)

    assert final.shape == (2, 1, cached.config.vocab_size)
    assert_allclose(final, everything[:, -1:, :])


# --------------------------------------------------- decode matches the oracle


def test_one_decode_step_matches_a_full_recompute(pair):
    """The core claim: token `n+1` from the cache equals it from scratch."""
    naive, cached, vocab = pair
    prompt = torch.randint(0, vocab, (1, 6))
    next_token = torch.randint(0, vocab, (1, 1))

    caches = cached.create_kv_cache()
    cached(prompt, caches)
    from_cache = cached(next_token, caches)

    whole = torch.cat([prompt, next_token], dim=1)
    assert_allclose(from_cache, naive(whole)[:, -1:, :])


@pytest.mark.parametrize("steps", [1, 3, 10])
def test_many_decode_steps_match_a_full_recompute(pair, steps):
    naive, cached, vocab = pair
    prompt = torch.randint(0, vocab, (2, 5))
    extra = torch.randint(0, vocab, (2, steps))

    caches = cached.create_kv_cache()
    cached(prompt, caches)
    for step in range(steps):
        from_cache = cached(extra[:, step : step + 1], caches)

    whole = torch.cat([prompt, extra], dim=1)
    assert_allclose(from_cache, naive(whole)[:, -1:, :], msg=f"after {steps} decode steps")
    assert all(cache.offset == 5 + steps for cache in caches)


def test_token_by_token_matches_one_prefill(pair):
    """Feeding a prompt one token at a time must equal prefilling it at once.

    A strong test of the position bookkeeping specifically: the arithmetic is
    identical either way, so only `offset` and the mask can differ.
    """
    _naive, cached, vocab = pair
    ids = torch.randint(0, vocab, (1, 7))

    at_once = cached(ids, cached.create_kv_cache())

    caches = cached.create_kv_cache()
    for position in range(ids.shape[1]):
        one_at_a_time = cached(ids[:, position : position + 1], caches)

    assert_allclose(one_at_a_time, at_once[:, -1:, :])


def test_chunked_prefill_matches_one_prefill(pair):
    """Ragged chunks, which is exactly what Step 4.4's scheduler will produce."""
    _naive, cached, vocab = pair
    ids = torch.randint(0, vocab, (1, 12))

    at_once = cached(ids, cached.create_kv_cache())

    caches = cached.create_kv_cache()
    start = 0
    for size in (5, 1, 4, 2):
        chunk = cached(ids[:, start : start + size], caches)
        start += size

    assert_allclose(chunk[:, -1:, :], at_once[:, -1:, :])


# ----------------------------------------------------------------- positions


def test_positions_come_from_the_cache_offset(pair):
    """Explicit positions matching the offset must be a no-op."""
    _naive, cached, vocab = pair
    prompt = torch.randint(0, vocab, (1, 4))
    token = torch.randint(0, vocab, (1, 1))

    caches = cached.create_kv_cache()
    cached(prompt, caches)
    implicit = cached(token, caches)

    caches = cached.create_kv_cache()
    cached(prompt, caches, positions=torch.arange(0, 4))
    explicit = cached(token, caches, positions=torch.arange(4, 5))

    assert_allclose(implicit, explicit)


def test_wrong_positions_change_the_result(pair):
    """Confirms the offset is actually used rather than incidentally correct."""
    _naive, cached, vocab = pair
    prompt = torch.randint(0, vocab, (1, 4))
    token = torch.randint(0, vocab, (1, 1))

    caches = cached.create_kv_cache()
    cached(prompt, caches)
    correct = cached(token, caches)

    caches = cached.create_kv_cache()
    cached(prompt, caches)
    wrong = cached(token, caches, positions=torch.arange(0, 1))

    assert not torch.allclose(correct, wrong)


def test_a_decode_token_attends_to_the_whole_cache(pair):
    """No mask is built for `L == 1`, so this checks nothing got masked away.

    Changing an early prompt token must move the decode logits. If the `(L, S)`
    mask offset were wrong, the token would see only part of its cache and this
    would not change at all.
    """
    _naive, cached, vocab = pair
    prompt = torch.randint(0, vocab, (1, 6))
    token = torch.randint(0, vocab, (1, 1))

    caches = cached.create_kv_cache()
    cached(prompt, caches)
    original = cached(token, caches)

    altered = prompt.clone()
    altered[0, 0] = (altered[0, 0] + 1) % vocab
    caches = cached.create_kv_cache()
    cached(altered, caches)
    changed = cached(token, caches)

    assert not torch.allclose(original, changed)


# -------------------------------------------------------------- cache reuse


def test_resetting_caches_allows_a_second_sequence(pair):
    _naive, cached, vocab = pair
    caches = cached.create_kv_cache()
    ids = torch.randint(0, vocab, (1, 5))

    first = cached(ids, caches)
    for cache in caches:
        cache.reset()
    second = cached(ids, caches)

    assert_allclose(first, second)


# ------------------------------------------------------------- generation


def test_cached_greedy_matches_the_naive_loop(pair):
    """Step 2.2's loop against Step 1.9's, token for token."""
    naive, cached, vocab = pair
    ids = torch.randint(0, vocab, (1, 5))

    assert_tokens_equal(
        generate_ids_cached(cached, ids, max_tokens=12),
        generate_ids(naive, ids, max_tokens=12),
    )


def test_cached_greedy_matches_the_naive_loop_batched(pair):
    naive, cached, vocab = pair
    ids = torch.randint(0, vocab, (3, 6))

    assert_tokens_equal(
        generate_ids_cached(cached, ids, max_tokens=8),
        generate_ids(naive, ids, max_tokens=8),
    )


def test_cached_generation_stops_at_eos(pair):
    naive, cached, vocab = pair
    ids = torch.randint(0, vocab, (1, 4))

    first_generated = int(generate_ids_cached(cached, ids, max_tokens=6)[0, 4])
    stopped = generate_ids_cached(
        cached, ids, max_tokens=6, eos_token_ids=[first_generated]
    )

    assert stopped.shape[1] == 5


def test_cached_loop_rejects_unbatched_input(pair):
    _naive, cached, _vocab = pair
    with pytest.raises(ValueError, match="expected B x L"):
        generate_ids_cached(cached, torch.tensor([1, 2, 3]), max_tokens=2)


def test_caches_can_be_supplied_by_the_caller(pair):
    """Phase 4's engine will own the caches, so the loop must accept them."""
    _naive, cached, vocab = pair
    ids = torch.randint(0, vocab, (1, 4))
    caches = cached.create_kv_cache()

    tokens = generate_ids_cached(cached, ids, max_tokens=5, caches=caches)

    assert all(cache.offset == tokens.shape[1] - 1 for cache in caches), (
        "the cache should hold every token except the last generated one, "
        "which has not been fed back in"
    )


# --------------------------------------------------------------- dispatch


def test_every_claimed_kernel_is_callable():
    """A name in `CUDA_KERNELS` must be a kernel the extension actually exports.

    This replaces Step 2.2's "no kernels are claimed yet". Phase 3 flips these
    flags one at a time, and the flag is what the benchmark reports and what the
    dispatch trusts, so a flag flipped ahead of the `.cu` would silently make the
    report a lie. Asserting the symbol exists keeps them honest as the list grows.
    """
    claimed = ops.cuda_kernel_names()
    if not claimed:
        pytest.skip("no kernels claimed yet")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device to load the extension")

    from mini_vllm.kernels.extension import load_extension

    module = load_extension()
    missing = [name for name in claimed if not hasattr(module, name)]
    assert not missing, f"claimed but not exported by csrc/: {missing}"


def test_use_cuda_falls_back_on_cpu_tensors(tiny_qwen3):
    """`use_cuda=True` must fall back rather than fail when there is no GPU tensor.

    The same model object serves CPU tests and GPU serving, so the flag means "use
    kernels where they apply", not "assume a device".
    """
    config, weights = config_from_hf(tiny_qwen3), weights_from_hf(tiny_qwen3)
    ids = torch.randint(0, tiny_qwen3.config.vocab_size, (1, 5))

    torch_path = Qwen3Cached(config, weights, use_cuda=False)
    cuda_path = Qwen3Cached(config, weights, use_cuda=True)

    assert_allclose(
        cuda_path(ids, cuda_path.create_kv_cache()),
        torch_path(ids, torch_path.create_kv_cache()),
    )


def test_dispatch_report_names_every_op_and_its_step():
    report = ops.dispatch_report(use_cuda=True)

    for name, (_implemented, step) in ops.CUDA_KERNELS.items():
        assert name in report
        assert step in report


@pytest.mark.cuda
def test_runs_on_the_gpu(tiny_qwen3, device):
    theirs = tiny_qwen3.to(device)
    cached = Qwen3Cached(config_from_hf(theirs), weights_from_hf(theirs))
    naive = qwen3_from_hf(theirs)
    ids = torch.randint(0, theirs.config.vocab_size, (2, 6), device=device)

    assert_allclose(cached(ids, cached.create_kv_cache()), naive(ids))


# ------------------------------------------------------- the real weights


@pytest.fixture(scope="module")
def real_models():
    """Qwen3-0.6B at a requested dtype: cached, uncached, and HF, weights shared.

    Both dtypes are needed, and for a sharper reason than in Step 1.8. A decode
    step computes `q` of length 1 against a cache of length `S`; a full recompute
    computes length `S` against length `S`. Same arithmetic, different matmul
    shapes — so cuBLAS reduces them in a different order, and in bf16 the results
    differ in the last bit. **The cached and uncached models therefore cannot be
    bitwise identical in bf16, however correct both are.** fp32 has the headroom to
    absorb that, so it is where token identity is asserted.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = resolve_model_path()
    if not (path / "model.safetensors").is_file():
        pytest.skip("Qwen3-0.6B weights are not downloaded")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(path)
    cache: dict[torch.dtype, tuple] = {}

    def load(dtype: torch.dtype):
        if dtype not in cache:
            hf = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
            config, weights = config_from_hf(hf), weights_from_hf(hf)
            cache[dtype] = (
                Qwen3Cached(config, weights),
                qwen3_from_hf(hf),
                hf,
                tokenizer,
                device,
            )
        return cache[dtype]

    yield load

    cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.oracle
def test_real_prefill_matches_the_uncached_model(real_models):
    """Prefill is the same shape either way, so bf16 agreement is exact here."""
    cached, naive, _hf, tokenizer, device = real_models(torch.bfloat16)
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.to(device)

    got = cached(ids, cached.create_kv_cache())
    expected = naive(ids)

    assert torch.equal(got, expected), (
        "a prefill runs identical shapes through identical ops, so it should be "
        "bitwise equal to the uncached forward"
    )


@pytest.mark.oracle
def test_real_cached_greedy_matches_our_naive_loop_in_fp32(real_models):
    """The load-bearing test for this step: cache versus no cache, nothing else varying.

    In fp32 this is exact over 24 tokens, which is a complete statement about the
    cache: every decode step reproduced what recomputing the whole prefix would
    have produced, including the position bookkeeping and the `(L, S)` mask.
    """
    cached, naive, _hf, tokenizer, device = real_models(torch.float32)
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.to(device)

    assert_tokens_equal(
        generate_ids_cached(cached, ids, max_tokens=24),
        generate_ids(naive, ids, max_tokens=24),
    )


@pytest.mark.oracle
def test_real_cached_greedy_matches_transformers_in_fp32(real_models):
    """Closes the loop to the external oracle, on the prompt from the plan."""
    cached, _naive, hf, tokenizer, device = real_models(torch.float32)
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.to(device)

    expected = hf.generate(
        ids, max_new_tokens=32, do_sample=False, pad_token_id=hf.generation_config.pad_token_id
    )
    got = generate_ids_cached(
        cached,
        ids,
        max_tokens=32,
        eos_token_ids=eos_token_ids_for(resolve_model_path()),
        pad_token_id=hf.generation_config.pad_token_id,
    )

    assert_tokens_equal(got, expected)


@pytest.mark.oracle
def test_real_batch_of_three_prompts_in_fp32(real_models):
    """A padded batch of different prompts, as the plan asks for.

    Left-padding is what makes this work: the prompts are aligned at their *right*
    edge so every sequence's last token sits at the same index and one position
    vector serves the whole batch. That alignment tax — padding shorter prompts to
    the longest — is the reason Phase 4 abandons rectangular batches for a ragged
    one.
    """
    cached, naive, _hf, tokenizer, device = real_models(torch.float32)
    prompts = ["The capital of France is", "def fibonacci(n):", "Once upon a time"]

    encoded = [tokenizer(p, return_tensors="pt").input_ids[0] for p in prompts]
    width = max(len(ids) for ids in encoded)
    padded = torch.stack(
        [
            torch.cat([torch.full((width - len(ids),), ids[0].item()), ids])
            for ids in encoded
        ]
    ).to(device)

    assert_tokens_equal(
        generate_ids_cached(cached, padded, max_tokens=16),
        generate_ids(naive, padded, max_tokens=16),
    )


@pytest.mark.oracle
def test_a_bf16_decode_step_agrees_with_a_full_recompute_to_within_rounding(real_models):
    """Quantifies what bf16 costs, so the fp32-only identity above is justified.

    A decode step and a full recompute of the same position are the same function
    of the same inputs, so they must agree to bf16 rounding — 2.05% relative here,
    the same order as the 1.7% this model drifts from HuggingFace anyway, and with
    the same argmax. What they cannot do is agree bitwise, because the matmul
    shapes differ.

    The prefill comparison beside it is the control that pins the cause down:
    prefill runs the *same* shapes as the uncached forward and comes out bitwise
    equal, so the decode-shaped matmul is the only thing introducing a difference.
    """
    cached, naive, _hf, tokenizer, device = real_models(torch.bfloat16)
    ids = tokenizer("The capital of France is Paris. The capital of", return_tensors="pt")
    ids = ids.input_ids.to(device)

    from_scratch = naive(ids)[:, -1:, :]

    caches = cached.create_kv_cache()
    cached(ids[:, :-1], caches)
    from_decode = cached(ids[:, -1:], caches, last_only=True)

    assert_relative_error_below(from_decode, from_scratch)
    assert int(from_decode.argmax()) == int(from_scratch.argmax())

    from_prefill = cached(ids, cached.create_kv_cache(), last_only=True)
    assert torch.equal(from_prefill, from_scratch), (
        "prefill uses the same shapes as the uncached forward, so any difference "
        "here would be a real bug rather than a reduction-order artefact"
    )


@pytest.mark.oracle
def test_any_bf16_greedy_divergence_is_a_near_tie(real_models):
    """The bf16 counterpart of the fp32 identity test, using Step 1.9's argument.

    When the cached and naive loops disagree in bf16, they must disagree only about
    the order of two tokens that bf16 cannot separate. On this prompt they split at
    step 10 over ' Italy' (16.625) and ' France' (16.5) — one ULP apart at that
    magnitude. A cache bug looks nothing like this: it picks a token the other path
    does not rank at all.
    """
    from test_generate import bf16_ulp

    cached, naive, _hf, tokenizer, device = real_models(torch.bfloat16)
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.to(device)

    from_cache = generate_ids_cached(cached, ids, max_tokens=24)
    from_scratch = generate_ids(naive, ids, max_tokens=24)
    if from_cache.tolist() == from_scratch.tolist():
        pytest.skip("no divergence to characterize on this build")

    index = next(
        i for i in range(from_cache.shape[1]) if from_cache[0, i] != from_scratch[0, i]
    )
    prefix = from_scratch[:, :index]

    with torch.no_grad():
        logits = naive(prefix)[0, -1].float()

    top = logits.topk(2)
    candidates = {int(from_cache[0, index]), int(from_scratch[0, index])}

    assert candidates == set(top.indices.tolist()), (
        "the two paths disagreed over tokens that are not the top two, "
        "which is a bug rather than a tie"
    )
    gap = (top.values[0] - top.values[1]).item()
    limit = 2 * bf16_ulp(top.values[0].item())
    assert gap <= limit, (
        f"the preferred token led by {gap:.4f}, more than {limit:.4f} (two bf16 ULP), "
        "so this divergence is not explained by rounding"
    )
