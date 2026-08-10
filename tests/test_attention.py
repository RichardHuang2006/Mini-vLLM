"""Steps 1.2 and 1.4 — attention against torch's SDPA and MultiheadAttention.

Test names carry `simple`, `mha`, `causal` or `grouped` so each step can run its
own slice with `-k`.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from conftest import assert_allclose

from mini_vllm.attention import (
    SimpleMultiHeadAttention,
    causal_mask,
    scaled_dot_product_attention_grouped,
    scaled_dot_product_attention_simple,
)

BATCH_SHAPES = [(2,), (2, 3), (2, 4, 3)]


# ------------------------------------------------------- Step 1.2: simple SDPA


@pytest.mark.parametrize("batch", BATCH_SHAPES)
def test_simple_attention_matches_torch(batch):
    q = torch.randn(*batch, 6, 16)
    k = torch.randn(*batch, 6, 16)
    v = torch.randn(*batch, 6, 16)
    assert_allclose(
        scaled_dot_product_attention_simple(q, k, v),
        F.scaled_dot_product_attention(q, k, v),
    )


def test_simple_attention_with_different_query_and_source_lengths():
    """`L != S`, the shape decode produces once there is a cache."""
    q = torch.randn(2, 3, 1, 16)
    k = torch.randn(2, 3, 9, 16)
    v = torch.randn(2, 3, 9, 16)

    got = scaled_dot_product_attention_simple(q, k, v)

    assert got.shape == (2, 3, 1, 16)
    assert_allclose(got, F.scaled_dot_product_attention(q, k, v))


def test_simple_attention_honours_explicit_scale():
    q, k, v = (torch.randn(2, 5, 8) for _ in range(3))
    scale = 0.137
    assert_allclose(
        scaled_dot_product_attention_simple(q, k, v, scale=scale),
        F.scaled_dot_product_attention(q, k, v, scale=scale),
    )


def test_simple_attention_default_scale_is_one_over_sqrt_d():
    q, k, v = (torch.randn(2, 5, 16) for _ in range(3))
    assert_allclose(
        scaled_dot_product_attention_simple(q, k, v),
        scaled_dot_product_attention_simple(q, k, v, scale=1.0 / math.sqrt(16)),
    )


def test_simple_attention_with_additive_mask():
    q, k, v = (torch.randn(2, 4, 6, 16) for _ in range(3))
    mask = torch.zeros(6, 6).masked_fill(torch.rand(6, 6) < 0.3, float("-inf"))
    # Guarantee no fully-masked row, which is legitimately nan for both.
    mask.fill_diagonal_(0.0)

    assert_allclose(
        scaled_dot_product_attention_simple(q, k, v, mask=mask),
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask),
    )


def test_simple_attention_masked_positions_contribute_nothing():
    """A `-inf` key must be exactly ignored, not merely down-weighted.

    Checked by making the masked value enormous: if any weight leaked onto it,
    the output would be dominated by it.
    """
    q = torch.randn(1, 2, 8)
    k = torch.randn(1, 3, 8)
    v = torch.randn(1, 3, 8)
    v[:, 2, :] = 1e6

    mask = torch.zeros(2, 3)
    mask[:, 2] = float("-inf")

    got = scaled_dot_product_attention_simple(q, k, v, mask=mask)
    want = scaled_dot_product_attention_simple(q, k[:, :2], v[:, :2])
    assert_allclose(got, want)


def test_simple_attention_rejects_unknown_mask_shorthand():
    q, k, v = (torch.randn(1, 4, 8) for _ in range(3))
    with pytest.raises(ValueError, match="unknown mask shorthand"):
        scaled_dot_product_attention_simple(q, k, v, mask="bidirectional")


# --------------------------------------------------------------- Step 1.2: MHA


def _torch_mha(hidden_size, num_heads, wq, wk, wv, wo):
    """torch's MultiheadAttention wired up with our weights."""
    reference = torch.nn.MultiheadAttention(
        hidden_size, num_heads, bias=False, batch_first=True
    )
    with torch.no_grad():
        # torch packs the three input projections into one (3E, E) matrix.
        reference.in_proj_weight.copy_(torch.cat([wq, wk, wv], dim=0))
        reference.out_proj.weight.copy_(wo)
    return reference.eval()


@pytest.mark.parametrize("num_heads", [1, 2, 4])
def test_mha_matches_torch_multihead_attention(num_heads):
    hidden_size, length, batch = 32, 6, 2
    wq, wk, wv, wo = (torch.randn(hidden_size, hidden_size) * 0.1 for _ in range(4))
    x = torch.randn(batch, length, hidden_size)

    ours = SimpleMultiHeadAttention(hidden_size, num_heads, wq, wk, wv, wo)(x)
    with torch.no_grad():
        theirs, _ = _torch_mha(hidden_size, num_heads, wq, wk, wv, wo)(
            x, x, x, need_weights=False
        )

    assert_allclose(ours, theirs)


def test_mha_matches_torch_with_a_causal_mask():
    hidden_size, num_heads, length = 32, 4, 7
    wq, wk, wv, wo = (torch.randn(hidden_size, hidden_size) * 0.1 for _ in range(4))
    x = torch.randn(2, length, hidden_size)

    ours = SimpleMultiHeadAttention(hidden_size, num_heads, wq, wk, wv, wo)(x, mask="causal")
    with torch.no_grad():
        theirs, _ = _torch_mha(hidden_size, num_heads, wq, wk, wv, wo)(
            x, x, x, need_weights=False, attn_mask=causal_mask(length, length)
        )

    assert_allclose(ours, theirs)


def test_mha_output_shape_and_head_dim_derivation():
    """`H·D` need not equal `E`, as in the real Qwen3-0.6B."""
    hidden_size, num_heads, head_dim = 16, 4, 8  # H·D = 32 != E = 16
    wq, wk, wv = (torch.randn(num_heads * head_dim, hidden_size) for _ in range(3))
    wo = torch.randn(hidden_size, num_heads * head_dim)

    attention = SimpleMultiHeadAttention(hidden_size, num_heads, wq, wk, wv, wo)
    assert attention.head_dim == head_dim

    out = attention(torch.randn(2, 5, hidden_size))
    assert out.shape == (2, 5, hidden_size)


def test_mha_rejects_head_count_that_does_not_divide_the_projection():
    with pytest.raises(ValueError, match="must be a multiple of num_heads"):
        SimpleMultiHeadAttention(
            16, 3, torch.randn(32, 16), torch.randn(32, 16), torch.randn(32, 16), torch.randn(16, 32)
        )


# ------------------------------------------------------ Step 1.4: causal masks


@pytest.mark.parametrize("length", [1, 2, 5])
def test_causal_mask_is_lower_triangular_when_l_equals_s(length):
    mask = causal_mask(length, length)
    assert mask.shape == (length, length)

    allowed = mask == 0.0
    assert torch.equal(allowed, torch.ones(length, length, dtype=torch.bool).tril())


def test_causal_mask_offsets_when_l_is_less_than_s():
    """The decode case: `L` queries are the *last* `L` positions of `S`."""
    mask = causal_mask(2, 5)

    # Query 0 is absolute position 3, so it sees keys 0..3 but not 4.
    # Query 1 is absolute position 4, so it sees everything.
    expected_allowed = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]
    )
    assert torch.equal(mask == 0.0, expected_allowed)


def test_causal_mask_for_a_single_decode_token_sees_the_whole_cache():
    """The bug this offset exists to prevent.

    One new token attending against a cache of 9 must see all 10 positions. A
    mask built without the `S - L` shift would let it see only key 0.
    """
    mask = causal_mask(1, 10)
    assert (mask == 0.0).all()


def test_causal_mask_matches_torch_is_causal():
    q, k, v = (torch.randn(2, 4, 6, 16) for _ in range(3))
    assert_allclose(
        scaled_dot_product_attention_simple(q, k, v, mask="causal"),
        F.scaled_dot_product_attention(q, k, v, is_causal=True),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_causal_mask_respects_dtype(dtype):
    mask = causal_mask(4, 4, dtype)
    assert mask.dtype == dtype
    assert torch.isneginf(mask[0, 1])


# ------------------------------------------------- Step 1.4: grouped attention


@pytest.mark.parametrize(
    ("num_query_heads", "num_kv_heads"),
    [
        (16, 8),  # Qwen3-0.6B: G = 2
        (4, 2),  # the tiny fixture: G = 2
        (8, 2),  # G = 4
        (4, 1),  # multi-query attention
        (4, 4),  # degenerate: G = 1, plain multi-head
    ],
)
def test_grouped_attention_matches_torch(num_query_heads, num_kv_heads):
    batch, query_len, source_len, head_dim = 2, 6, 6, 32
    q = torch.randn(batch, num_query_heads, query_len, head_dim)
    k = torch.randn(batch, num_kv_heads, source_len, head_dim)
    v = torch.randn(batch, num_kv_heads, source_len, head_dim)

    assert_allclose(
        scaled_dot_product_attention_grouped(q, k, v),
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True),
    )


def test_grouped_attention_matches_torch_in_decode_shape():
    """`L = 1` against a long cache, the shape decode spends all its time in."""
    q = torch.randn(2, 16, 1, 32)
    k = torch.randn(2, 8, 137, 32)
    v = torch.randn(2, 8, 137, 32)

    assert_allclose(
        scaled_dot_product_attention_grouped(q, k, v),
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True),
    )


def test_grouped_attention_matches_torch_with_causal_mask():
    q = torch.randn(2, 16, 6, 32)
    k = torch.randn(2, 8, 6, 32)
    v = torch.randn(2, 8, 6, 32)

    assert_allclose(
        scaled_dot_product_attention_grouped(q, k, v, mask="causal"),
        F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True),
    )


def test_grouped_attention_accepts_a_per_head_mask():
    q = torch.randn(2, 8, 5, 16)
    k = torch.randn(2, 2, 5, 16)
    v = torch.randn(2, 2, 5, 16)
    mask = causal_mask(5, 5).expand(2, 8, 5, 5)

    assert_allclose(
        scaled_dot_product_attention_grouped(q, k, v, mask=mask),
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True),
    )


def test_grouped_attention_equals_repeating_the_kv_heads():
    """The broadcast shortcut must equal the obvious `repeat_interleave` form.

    Worth asserting separately from the torch comparison: it is what proves the
    (kv_head, group) reshape lines each query head up with *its own* KV head
    rather than transposing the grouping.
    """
    group_size = 4
    q = torch.randn(2, 8, 5, 16)
    k = torch.randn(2, 2, 5, 16)
    v = torch.randn(2, 2, 5, 16)

    expanded = scaled_dot_product_attention_simple(
        q,
        k.repeat_interleave(group_size, dim=1),
        v.repeat_interleave(group_size, dim=1),
        mask="causal",
    )
    assert_allclose(scaled_dot_product_attention_grouped(q, k, v, mask="causal"), expanded)


def test_grouped_attention_rejects_indivisible_head_counts():
    q = torch.randn(1, 5, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    with pytest.raises(ValueError, match="must be a multiple"):
        scaled_dot_product_attention_grouped(q, k, v)


def test_grouped_attention_rejects_ambiguous_3d_mask():
    q = torch.randn(2, 8, 5, 16)
    k = torch.randn(2, 2, 5, 16)
    v = torch.randn(2, 2, 5, 16)
    with pytest.raises(ValueError, match="ambiguous 3-D mask"):
        scaled_dot_product_attention_grouped(q, k, v, mask=torch.zeros(2, 5, 5))
