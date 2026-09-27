"""qwen3.py and models.py: the transplanted model against mlx_lm, and the cache ladder."""

import mlx.core as mx
import pytest
from conftest import make_tiny_qwen3
from utils import assert_allclose

from mini_vllm.kv_cache import KvFullCache
from mini_vllm.models import from_mlx


def caches_for(model):
    return [KvFullCache() for _ in model.layers]


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_logits_match_mlx_lm(tie_word_embeddings):
    reference = make_tiny_qwen3(tie_word_embeddings=tie_word_embeddings)
    model = from_mlx(reference)
    ids = mx.random.randint(0, 512, (2, 10))

    assert_allclose(model(ids, mx.arange(10)), reference(ids))


def test_cached_prefill_matches_the_dense_forward(tiny_qwen3):
    model = from_mlx(tiny_qwen3)
    ids = mx.random.randint(0, 512, (1, 10))

    assert_allclose(model(ids, mx.arange(10), caches_for(model)), model(ids, mx.arange(10)))


def test_cached_decode_matches_recomputing_from_scratch(tiny_qwen3):
    """The whole point of the cache: one token forwarded, same logits as a full pass."""
    model = from_mlx(tiny_qwen3)
    prompt = mx.random.randint(0, 512, (1, 9))
    step = mx.random.randint(0, 512, (1, 1))

    caches = caches_for(model)
    model(prompt, mx.arange(9), caches)
    got = model(step, mx.array([9]), caches)
    want = model(mx.concatenate([prompt, step], axis=1), mx.arange(10))[:, -1:]

    assert_allclose(got, want)


@pytest.mark.parametrize("chunk", [1, 7, 49, 50])  # single tokens, uneven, all but one, all
def test_chunked_prefill_matches_one_pass(tiny_qwen3, chunk):
    """Feeding the prompt in pieces must give every position the same logits."""
    model = from_mlx(tiny_qwen3)
    prompt = mx.random.randint(0, 512, (1, 50))
    whole = model(prompt, mx.arange(50))

    caches = caches_for(model)
    pieces = [
        model(prompt[:, start : start + chunk], mx.arange(start, min(start + chunk, 50)), caches)
        for start in range(0, 50, chunk)
    ]
    assert_allclose(mx.concatenate(pieces, axis=1), whole)


def test_a_wrong_position_at_a_chunk_boundary_is_caught(tiny_qwen3):
    """Restarting RoPE at zero on the second chunk must change the logits."""
    model = from_mlx(tiny_qwen3)
    prompt = mx.random.randint(0, 512, (1, 40))
    whole = model(prompt, mx.arange(40))[:, -1]

    caches = caches_for(model)
    model(prompt[:, :20], mx.arange(20), caches)
    wrong = model(prompt[:, 20:], mx.arange(20), caches)[:, -1]

    assert not mx.allclose(wrong, whole, rtol=1e-3, atol=1e-3).item(), (
        "restarting RoPE at zero changed nothing, so the positions are not being used"
    )


def test_real_logits_sit_close_to_mlx_lm(real_qwen3):
    reference, tokenizer = real_qwen3
    model = from_mlx(reference)
    ids = mx.array([tokenizer.encode("Paged attention stores the KV cache in fixed-size blocks")])
    length = ids.shape[1]

    ours = model(ids, mx.arange(length)).astype(mx.float32)
    theirs = reference(ids).astype(mx.float32)

    # BF16 through 28 layers: bound the relative error, not each element.
    relative = (mx.linalg.norm(ours - theirs) / mx.linalg.norm(theirs)).item()
    assert relative < 0.05, relative
    assert mx.array_equal(mx.argmax(ours, axis=-1), mx.argmax(theirs, axis=-1)).item()
