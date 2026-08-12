"""RoPE against HuggingFace's own rotary embedding.

HF applies rotation to `B x H x L x D` tensors with `cos`/`sin` of shape
`B x L x D`; ours takes `B x L x H x D` and gathers the tables itself from a
position tensor. The comparisons below transpose between the two layouts, so a
mismatch means the *rotation* differs, not the axis order.
"""

from __future__ import annotations

import pytest
import torch
from conftest import TINY_QWEN3_DIMS, assert_allclose
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers.models.qwen3.modeling_qwen3 import rotate_half as hf_rotate_half

from mini_vllm.positional_encoding import RoPE, rotate_half

HEAD_DIM = TINY_QWEN3_DIMS["head_dim"]
MAX_SEQ_LEN = TINY_QWEN3_DIMS["max_position_embeddings"]
THETA = TINY_QWEN3_DIMS["rope_theta"]


@pytest.fixture
def hf_rotary():
    """HF's rotary embedding for the tiny config, the oracle for these tests."""
    return Qwen3RotaryEmbedding(Qwen3Config(**TINY_QWEN3_DIMS))


def hf_rotate(hf_rotary, x_blhd: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Rotate a `B x L x H x D` tensor via HF, returning the same layout."""
    x_bhld = x_blhd.transpose(1, 2)
    position_ids = positions.unsqueeze(0) if positions.ndim == 1 else positions
    position_ids = position_ids.expand(x_blhd.shape[0], -1)

    cos, sin = hf_rotary(x_bhld, position_ids)
    rotated, _ = apply_rotary_pos_emb(x_bhld, x_bhld, cos, sin)
    return rotated.transpose(1, 2)


# ------------------------------------------------------------------ rotate_half


def test_rotate_half_matches_hf():
    x = torch.randn(2, 3, 4, HEAD_DIM)
    assert_allclose(rotate_half(x), hf_rotate_half(x))


def test_rotate_half_splits_halves_not_pairs():
    """Explicitly pins the convention, since the other one also 'works'."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert_allclose(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_rotate_half_applied_four_times_is_identity():
    """Four 90-degree rotations return to the start."""
    x = torch.randn(2, 8)
    assert_allclose(rotate_half(rotate_half(rotate_half(rotate_half(x)))), x)


# ------------------------------------------------------------- tables and setup


def test_tables_have_expected_shape_and_range():
    rope = RoPE(HEAD_DIM, MAX_SEQ_LEN, THETA)

    assert rope.cos.shape == (MAX_SEQ_LEN, HEAD_DIM)
    assert rope.sin.shape == (MAX_SEQ_LEN, HEAD_DIM)
    assert rope.cos.dtype == torch.float32
    assert rope.cos.abs().max() <= 1.0 and rope.sin.abs().max() <= 1.0


def test_position_zero_is_the_identity_rotation():
    rope = RoPE(HEAD_DIM, MAX_SEQ_LEN, THETA)
    x = torch.randn(1, 1, 2, HEAD_DIM)
    assert_allclose(rope(x, torch.zeros(1, dtype=torch.int64)), x)


def test_tables_are_duplicated_not_interleaved():
    """`cos[:, i] == cos[:, i + D/2]`, which is what pairs with `rotate_half`."""
    rope = RoPE(HEAD_DIM, MAX_SEQ_LEN, THETA)
    half = HEAD_DIM // 2
    assert_allclose(rope.cos[:, :half], rope.cos[:, half:])
    assert_allclose(rope.sin[:, :half], rope.sin[:, half:])


def test_rejects_odd_head_dim():
    with pytest.raises(ValueError, match="must be even"):
        RoPE(15, MAX_SEQ_LEN, THETA)


def test_rejects_float_positions():
    rope = RoPE(HEAD_DIM, MAX_SEQ_LEN, THETA)
    with pytest.raises(ValueError, match="must be integer"):
        rope(torch.randn(1, 2, 1, HEAD_DIM), torch.tensor([0.0, 1.0]))


def test_rejects_positions_past_the_table():
    rope = RoPE(HEAD_DIM, 8, THETA)
    with pytest.raises(ValueError, match="beyond the precomputed table"):
        rope(torch.randn(1, 1, 1, HEAD_DIM), torch.tensor([8]))


# ------------------------------------------------------------- against HF


@pytest.mark.parametrize("length", [1, 3, 16])
def test_matches_hf_for_contiguous_positions(hf_rotary, length):
    x = torch.randn(2, length, 4, HEAD_DIM)
    positions = torch.arange(length)

    rope = RoPE(HEAD_DIM, MAX_SEQ_LEN, THETA)
    assert_allclose(rope(x, positions), hf_rotate(hf_rotary, x, positions))


def test_matches_hf_for_offset_positions(hf_rotary):
    """The `[5, 6, 7]` case: a chunk that does not start at zero.

    This is what chunked prefill produces for every chunk after the first, so it
    is the case an implicit `arange` would silently get wrong.
    """
    x = torch.randn(2, 3, 4, HEAD_DIM)
    positions = torch.tensor([5, 6, 7])

    rope = RoPE(HEAD_DIM, MAX_SEQ_LEN, THETA)
    assert_allclose(rope(x, positions), hf_rotate(hf_rotary, x, positions))


def test_matches_hf_for_a_single_decode_position(hf_rotary):
    """One token at an arbitrary position, the decode-step shape."""
    x = torch.randn(1, 1, 4, HEAD_DIM)
    positions = torch.tensor([137])

    rope = RoPE(HEAD_DIM, MAX_SEQ_LEN, THETA)
    assert_allclose(rope(x, positions), hf_rotate(hf_rotary, x, positions))


def test_matches_hf_for_non_monotonic_positions(hf_rotary):
    """Positions need not be sorted or contiguous.

    A ragged batch interleaves tokens from different sequences, so the position
    tensor really can look like this.
    """
    x = torch.randn(1, 5, 2, HEAD_DIM)
    positions = torch.tensor([9, 0, 4, 4, 1])

    rope = RoPE(HEAD_DIM, MAX_SEQ_LEN, THETA)
    assert_allclose(rope(x, positions), hf_rotate(hf_rotary, x, positions))


def test_per_sequence_positions(hf_rotary):
    """A `B x L` position tensor, each sequence at its own offset."""
    x = torch.randn(2, 3, 4, HEAD_DIM)
    positions = torch.tensor([[0, 1, 2], [10, 11, 12]])

    rope = RoPE(HEAD_DIM, MAX_SEQ_LEN, THETA)
    got = rope(x, positions)

    # Each row must equal that row rotated on its own.
    for row in range(2):
        assert_allclose(got[row : row + 1], rope(x[row : row + 1], positions[row]))


# ----------------------------------------------------------- properties, dtypes


def test_rotation_preserves_norm():
    """RoPE is a rotation, so it cannot change a head vector's length."""
    rope = RoPE(HEAD_DIM, MAX_SEQ_LEN, THETA)
    x = torch.randn(2, 7, 3, HEAD_DIM)

    rotated = rope(x, torch.arange(7))
    assert_allclose(rotated.norm(dim=-1), x.norm(dim=-1))


def test_relative_position_is_what_survives_the_dot_product():
    """The property RoPE exists for.

    `<RoPE(q, m), RoPE(k, n)>` depends only on `m - n`. Two pairs with the same
    separation must therefore produce the same score, which is what lets a model
    trained on short contexts generalize to longer ones.
    """
    rope = RoPE(HEAD_DIM, MAX_SEQ_LEN, THETA)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)

    def score(query_position: int, key_position: int) -> torch.Tensor:
        rotated_q = rope(q, torch.tensor([query_position]))
        rotated_k = rope(k, torch.tensor([key_position]))
        return (rotated_q * rotated_k).sum()

    assert_allclose(score(3, 1), score(12, 10))
    assert not torch.allclose(score(3, 1), score(3, 2))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_returns_input_dtype_but_rotates_in_fp32(dtype):
    rope = RoPE(HEAD_DIM, MAX_SEQ_LEN, THETA)
    x = torch.randn(2, 4, 3, HEAD_DIM, dtype=dtype)

    got = rope(x, torch.arange(4))

    assert got.dtype == dtype
    assert_allclose(got, rope(x.float(), torch.arange(4)).to(dtype))


@pytest.mark.cuda
def test_matches_cpu_on_gpu(device):
    rope = RoPE(HEAD_DIM, MAX_SEQ_LEN, THETA)
    x = torch.randn(2, 6, 3, HEAD_DIM)
    positions = torch.arange(6)

    on_cpu = rope(x, positions)
    on_gpu = rope(x.to(device), positions.to(device))

    assert_allclose(on_gpu.cpu(), on_cpu)
