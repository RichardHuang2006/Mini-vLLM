"""The reference ops, the weight loader, and the three Qwen3 variants.

The oracle chain, tested link by link:

    HuggingFace transformers  ─▶  Qwen3 (dense)  ─▶  Qwen3Cached  ─▶  Qwen3Paged

* The ops are checked against torch and against HuggingFace's own modules.
* The loader's name mapping is checked total in both directions, and (with weights
  downloaded) bitwise against the transformers state dict.
* The dense model must match HF logits and greedy tokens on a tiny random model, with
  no download; the cached model must match the dense one; the paged model must match
  the cached one — all exactly, in fp32, on greedy tokens.

Everything here runs on the CPU unless marked; `oracle` tests need the real
Qwen3-0.6B checkpoint.
"""

from __future__ import annotations

import pytest
import torch
from conftest import (
    BF16_DRIFT_LIMIT,
    assert_allclose,
    assert_relative_error_below,
    assert_tokens_equal,
    config_from_hf,
    qwen3_from_hf,
    weights_from_hf,
)

from mini_vllm import ops
from mini_vllm.cache import BlockManager, DenseKvCache
from mini_vllm.config import ModelConfig, SamplingParams
from mini_vllm.engine import generate_ids, generate_ids_cached
from mini_vllm.model import (
    Qwen3,
    Qwen3Cached,
    Qwen3Paged,
    expected_names,
    expected_shape,
    map_name,
    resolve_model_path,
)
from mini_vllm.scheduler import ForwardBatch, Sequence

GREEDY = SamplingParams(temperature=0.0)


# ------------------------------------------------------------- reference ops


def test_linear_matches_torch():
    x, w, bias = torch.randn(3, 5, 8), torch.randn(4, 8), torch.randn(4)
    assert_allclose(ops.linear(x, w, bias), torch.nn.functional.linear(x, w, bias))


def test_silu_matches_torch():
    x = torch.randn(64)
    assert_allclose(ops.silu(x), torch.nn.functional.silu(x))


def test_softmax_is_stable_at_large_logits():
    """exp overflows near 88 in fp32; subtracting the row max keeps this finite."""
    x = torch.tensor([[1000.0, 1000.0, 999.0]])
    got = ops.softmax(x)
    assert torch.isfinite(got).all()
    assert_allclose(got, torch.softmax(x, dim=-1))


def test_rms_norm_matches_hf_semantics():
    """fp32 reduction, cast back before the weight multiply — HF's exact order."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    x = torch.randn(2, 7, 64, dtype=torch.bfloat16)
    theirs = Qwen3RMSNorm(64, eps=1e-6)
    theirs.weight.data = torch.randn(64, dtype=torch.bfloat16)

    got = ops.rms_norm(x, theirs.weight.data, eps=1e-6)

    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, theirs(x))


def test_rope_matches_hf_tables():
    """Same frequencies, same rotate-half convention, row for row."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    from conftest import make_tiny_qwen3

    hf_model = make_tiny_qwen3()
    head_dim = hf_model.config.head_dim
    # Read theta through our own config parser: transformers 5.x makes `rope_theta` a
    # per-layer attribute that raises on global access.
    theta = config_from_hf(hf_model).rope_theta
    rope = ops.RoPE(head_dim, 128, theta=theta)

    positions = torch.arange(32).unsqueeze(0)
    theirs = Qwen3RotaryEmbedding(config=hf_model.config)
    cos, sin = theirs(torch.zeros(1, 32, head_dim), positions)

    assert_allclose(rope.cos[:32], cos[0])
    assert_allclose(rope.sin[:32], sin[0])


def test_rope_positions_are_explicit_not_assumed():
    """The same tokens at different absolute positions rotate differently."""
    rope = ops.RoPE(32, 128)
    x = torch.randn(1, 4, 2, 32)
    at_zero = rope(x, torch.arange(0, 4))
    at_fifty = rope(x, torch.arange(50, 54))
    assert not torch.allclose(at_zero, at_fifty)


def test_grouped_attention_equals_repeating_the_kv_heads():
    """GQA by broadcasting must equal GQA by materializing G copies of K and V."""
    q = torch.randn(2, 8, 5, 16)
    k = torch.randn(2, 2, 5, 16)
    v = torch.randn(2, 2, 5, 16)

    got = ops.scaled_dot_product_attention_grouped(q, k, v, mask="causal")

    repeated = ops.scaled_dot_product_attention_grouped(
        q, k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1), mask="causal"
    )
    assert_allclose(got, repeated)


def test_grouped_attention_matches_torch_sdpa():
    q = torch.randn(1, 4, 6, 32)
    kv = torch.randn(1, 4, 6, 32)
    got = ops.scaled_dot_product_attention_grouped(q, kv, kv, mask="causal")
    want = torch.nn.functional.scaled_dot_product_attention(q, kv, kv, is_causal=True)
    assert_allclose(got, want)


def test_causal_mask_shifts_for_a_decode_step():
    """With L < S the diagonal moves right: a decode token sees its whole cache."""
    mask = ops.causal_mask(1, 5)
    assert (mask == 0).all(), "a single query attends over everything cached"

    prefill = ops.causal_mask(3, 3)
    assert prefill[0, 1] == float("-inf") and prefill[2, 2] == 0


def test_paged_attention_gathered_rejects_causality_violations():
    with pytest.raises(ValueError, match="causality"):
        ops.paged_attention_gathered(
            torch.randn(4, 2, 8),
            torch.randn(2, 4, 1, 8),
            torch.randn(2, 4, 1, 8),
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([0, 4], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),  # S=2 < L=4
        )


# ------------------------------------------------------------------ sampling


def test_greedy_rows_are_one_hot():
    logits = torch.randn(3, 50)
    probabilities = ops.sampling_probabilities(logits, GREEDY)
    assert torch.equal(probabilities.argmax(-1), logits.argmax(-1))
    assert torch.equal(probabilities.max(-1).values, torch.ones(3))


def test_top_k_keeps_exactly_k_tokens():
    logits = torch.randn(1, 100)
    probabilities = ops.sampling_probabilities(logits, SamplingParams(top_k=5))
    assert int((probabilities > 0).sum()) == 5
    assert probabilities.sum().item() == pytest.approx(1.0)


def test_top_p_keeps_the_smallest_prefix_reaching_p():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    probabilities = ops.sampling_probabilities(logits, SamplingParams(top_p=0.7))
    # 0.5 < 0.7, so the boundary token 0.3 is included; 0.15 and 0.05 are not.
    assert (probabilities[0, :2] > 0).all() and (probabilities[0, 2:] == 0).all()


def test_per_row_parameters_apply_per_row():
    logits = torch.randn(2, 40)
    rows = [GREEDY, SamplingParams(temperature=1.0)]
    probabilities = ops.sampling_probabilities(logits, rows)
    assert probabilities[0].max() == 1.0, "row 0 is greedy"
    assert probabilities[1].max() < 1.0, "row 1 is a full distribution"


def test_a_seeded_generator_reproduces_the_draw():
    logits = torch.randn(4, 100)
    params = SamplingParams(temperature=0.8, top_p=0.9)
    first = ops.sample(logits, params, generator=torch.Generator().manual_seed(7))
    second = ops.sample(logits, params, generator=torch.Generator().manual_seed(7))
    assert torch.equal(first, second)


def test_dispatch_falls_back_on_cpu_tensors():
    """`use_cuda=True` on CPU tensors quietly runs the reference: one model object
    serves both CPU tests and GPU runs."""
    from mini_vllm import kernels

    x, weight = torch.randn(4, 64), torch.randn(64)
    assert_allclose(kernels.rmsnorm(x, weight, use_cuda=True), ops.rms_norm(x, weight))


# ----------------------------------------------------------------- the loader


def test_the_name_mapping_is_total_in_both_directions(tiny_qwen3):
    """Every HF tensor is consumed or deliberately dropped; every local name is filled."""
    config = config_from_hf(tiny_qwen3)
    produced = set()
    for hf_name in tiny_qwen3.state_dict():
        ours = map_name(hf_name)  # raises on anything unrecognized
        if ours is not None:
            produced.add(ours)
    assert produced == expected_names(config)


def test_unmapped_weights_raise_rather_than_vanish():
    with pytest.raises(KeyError, match="unmapped"):
        map_name("model.layers.0.self_attn.rotary_emb.inv_freq")


def test_expected_shapes_match_the_hf_tensors(tiny_qwen3):
    config = config_from_hf(tiny_qwen3)
    for name, tensor in weights_from_hf(tiny_qwen3).items():
        assert tuple(tensor.shape) == expected_shape(name, config), name


def test_model_config_reads_nested_rope_theta():
    raw = {
        "num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 32, "intermediate_size": 128,
        "vocab_size": 512, "rms_norm_eps": 1e-6, "max_position_embeddings": 256,
        "rope_parameters": {"rope_theta": 1e6},
    }
    assert ModelConfig.from_dict(raw).rope_theta == 1e6


@pytest.mark.oracle
def test_every_real_tensor_is_bitwise_equal_to_transformers():
    """The loader against the transformers state dict, tensor for tensor."""
    from transformers import AutoModelForCausalLM

    from mini_vllm.model import load_weights

    path = resolve_model_path()
    if not (path / "model.safetensors").is_file():
        pytest.skip("Qwen3-0.6B weights are not downloaded")

    ours, config = load_weights(path)
    theirs = AutoModelForCausalLM.from_pretrained(path, dtype=config.dtype)
    reference = weights_from_hf(theirs)

    assert ours.keys() == reference.keys()
    for name in ours:
        assert torch.equal(ours[name], reference[name]), f"{name} differs from transformers"


# ----------------------------------------------------- dense model vs HF


def test_logits_match_hf_on_the_tiny_model(tiny_qwen3):
    """The dense model against HuggingFace's, same random weights, fp32, exact-ish."""
    model = qwen3_from_hf(tiny_qwen3)
    ids = torch.randint(0, 512, (2, 12))

    with torch.no_grad():
        theirs = tiny_qwen3(ids).logits

    assert_allclose(model(ids), theirs, kind="model")


def test_greedy_tokens_match_hf_on_the_tiny_model(tiny_qwen3):
    model = qwen3_from_hf(tiny_qwen3)
    ids = torch.randint(0, 512, (1, 6))

    with torch.no_grad():
        theirs = tiny_qwen3.generate(
            ids, max_new_tokens=8, do_sample=False, pad_token_id=0
        )

    ours = generate_ids(model, ids, max_tokens=8)
    assert_tokens_equal(ours[0], theirs[0])


@pytest.mark.oracle
def test_real_logits_sit_within_the_bf16_drift_limit():
    """The real checkpoint, real text: within 5% of HF, where a broken model sits at 14%."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = resolve_model_path()
    if not (path / "model.safetensors").is_file():
        pytest.skip("Qwen3-0.6B weights are not downloaded")

    tokenizer = AutoTokenizer.from_pretrained(path)
    ids = tokenizer("The capital of France is Paris.", return_tensors="pt").input_ids

    model = Qwen3.from_pretrained(path)
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        theirs = hf(ids).logits

    assert_relative_error_below(model(ids), theirs, BF16_DRIFT_LIMIT)


# ------------------------------------------------------- cached vs dense


def test_the_dense_cache_appends_like_concat():
    cache = DenseKvCache()
    first = torch.randn(1, 2, 3, 8)
    second = torch.randn(1, 2, 1, 8)

    keys, _values, offset = cache.update_and_fetch(first, first)
    assert offset == 0 and keys.shape[-2] == 3

    keys, _values, offset = cache.update_and_fetch(second, second)
    assert offset == 3, "the offset is the length before the append"
    assert torch.equal(keys, torch.cat([first, second], dim=-2))


def test_cached_prefill_matches_the_dense_forward(tiny_qwen3):
    config, weights = config_from_hf(tiny_qwen3), weights_from_hf(tiny_qwen3)
    dense, cached = Qwen3(config, weights), Qwen3Cached(config, weights)
    ids = torch.randint(0, 512, (1, 10))

    assert_allclose(cached(ids, cached.create_kv_cache()), dense(ids))


def test_cached_decode_matches_recomputing_from_scratch(tiny_qwen3):
    """The whole point of the cache: one token forwarded, same logits as a full pass."""
    config, weights = config_from_hf(tiny_qwen3), weights_from_hf(tiny_qwen3)
    dense, cached = Qwen3(config, weights), Qwen3Cached(config, weights)

    prompt = torch.randint(0, 512, (1, 9))
    caches = cached.create_kv_cache()
    cached(prompt, caches)

    step = torch.randint(0, 512, (1, 1))
    got = cached(step, caches)
    want = dense(torch.cat([prompt, step], dim=1))[:, -1:]

    assert_allclose(got, want)
    assert got.argmax(-1).item() == want.argmax(-1).item()


def test_chunked_prefill_matches_one_pass(tiny_qwen3):
    """Feeding the prompt in pieces must give the last position the same logits.

    This is the model-side half of chunked prefill: explicit positions and the offset
    causal mask. Chunk sizes straddle every edge — single tokens, an uneven split, all
    but one, and the whole prompt.
    """
    config, weights = config_from_hf(tiny_qwen3), weights_from_hf(tiny_qwen3)
    cached = Qwen3Cached(config, weights)
    prompt = torch.randint(0, 512, (1, 50))

    whole = cached(prompt, cached.create_kv_cache(), last_only=True)

    for chunk in (1, 7, 49, 50):
        caches = cached.create_kv_cache()
        last = None
        for start in range(0, 50, chunk):
            piece = prompt[:, start : start + chunk]
            positions = torch.arange(start, start + piece.shape[1])
            last = cached(piece, caches, positions=positions, last_only=True)
        assert_allclose(last, whole, msg=f"chunk size {chunk}")
        assert last.argmax(-1).item() == whole.argmax(-1).item(), f"chunk size {chunk}"


def test_a_wrong_position_at_a_chunk_boundary_is_caught(tiny_qwen3):
    """Restarting RoPE at zero on the second chunk must change the logits.

    This is the mutation the test above protects against; if it ever stops being
    detectable, that comparison has stopped measuring anything.
    """
    config, weights = config_from_hf(tiny_qwen3), weights_from_hf(tiny_qwen3)
    cached = Qwen3Cached(config, weights)
    prompt = torch.randint(0, 512, (1, 40))

    whole = cached(prompt, cached.create_kv_cache(), last_only=True)

    caches = cached.create_kv_cache()
    wrong = None
    for start in (0, 20):
        piece = prompt[:, start : start + 20]
        wrong = cached(piece, caches, positions=torch.arange(20), last_only=True)

    assert not torch.allclose(wrong, whole, rtol=1e-3, atol=1e-3), (
        "restarting RoPE at zero changed nothing, so the positions are not being used"
    )


def test_the_generation_loops_agree(tiny_qwen3):
    """The naive quadratic loop and the cached loop, token-identical."""
    config, weights = config_from_hf(tiny_qwen3), weights_from_hf(tiny_qwen3)
    dense, cached = Qwen3(config, weights), Qwen3Cached(config, weights)
    ids = torch.randint(0, 512, (2, 5))

    assert_tokens_equal(
        generate_ids_cached(cached, ids, max_tokens=8),
        generate_ids(dense, ids, max_tokens=8),
    )


# -------------------------------------------------------- paged vs cached


def paged_from(tiny_qwen3, num_blocks: int = 64, block_size: int = 4) -> Qwen3Paged:
    config, weights = config_from_hf(tiny_qwen3), weights_from_hf(tiny_qwen3)
    manager = BlockManager(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=config.dtype,
    )
    return Qwen3Paged(config, weights, manager, use_cuda=False)


def test_paged_prefill_matches_the_dense_model(tiny_qwen3):
    """One ragged sequence through the paged pool, against the dense forward."""
    dense = qwen3_from_hf(tiny_qwen3)
    paged = paged_from(tiny_qwen3)

    prompt = torch.randint(0, 512, (14,)).tolist()
    sequence = Sequence(prompt_token_ids=prompt, sampling_params=GREEDY)
    paged.manager.allocate(sequence)
    batch = ForwardBatch.from_scheduled([(sequence, len(prompt))], manager=paged.manager)

    got = paged(batch)  # 1 x V: the last position
    want = dense(torch.tensor([prompt]))[:, -1, :]

    assert_allclose(got, want)
    assert got.argmax(-1).item() == want.argmax(-1).item()


def test_paged_ragged_batch_matches_each_dense_run(tiny_qwen3):
    """Three sequences of different lengths in one forward pass, rows independent."""
    dense = qwen3_from_hf(tiny_qwen3)
    paged = paged_from(tiny_qwen3)

    prompts = [torch.randint(0, 512, (n,)).tolist() for n in (11, 3, 7)]
    scheduled = []
    for prompt in prompts:
        sequence = Sequence(prompt_token_ids=prompt, sampling_params=GREEDY)
        paged.manager.allocate(sequence)
        scheduled.append((sequence, len(prompt)))
    batch = ForwardBatch.from_scheduled(scheduled, manager=paged.manager)

    got = paged(batch)  # 3 x V

    for index, prompt in enumerate(prompts):
        want = dense(torch.tensor([prompt]))[0, -1, :]
        assert_allclose(got[index], want, msg=f"sequence {index}")


def test_a_self_draft_shares_weights_and_truncates_layers(tiny_qwen3):
    paged = paged_from(tiny_qwen3)
    draft_manager = BlockManager(
        num_blocks=16, block_size=4,
        num_layers=1, num_kv_heads=paged.config.num_key_value_heads,
        head_dim=paged.config.head_dim, dtype=paged.config.dtype,
    )

    draft = paged.self_draft(1, draft_manager)

    assert len(draft.blocks) == 1
    assert draft.blocks[0] is paged.blocks[0], "the layers must be shared, not copied"
    assert draft.manager is draft_manager


# ------------------------------------------------------------ real weights


@pytest.mark.oracle
def test_real_greedy_generation_matches_transformers():
    """The cached model against `transformers.generate`, fp32, token for token."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = resolve_model_path()
    if not (path / "model.safetensors").is_file():
        pytest.skip("Qwen3-0.6B weights are not downloaded")

    tokenizer = AutoTokenizer.from_pretrained(path)
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids

    hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    config, weights = config_from_hf(hf), weights_from_hf(hf)
    cached = Qwen3Cached(config, weights)

    with torch.no_grad():
        theirs = hf.generate(ids, max_new_tokens=16, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    ours = generate_ids_cached(cached, ids, max_tokens=16)

    assert_tokens_equal(ours[0, : theirs.shape[1]], theirs[0])
