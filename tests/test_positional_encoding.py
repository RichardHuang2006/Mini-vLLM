"""positional_encoding.py against mx.fast.rope."""

import math

import mlx.core as mx
import pytest
from utils import DTYPES, assert_allclose

from mini_vllm.positional_encoding import RoPE

THETA = 1_000_000.0


def fast_rope(x: mx.array, offset: int) -> mx.array:
    """mx.fast.rope on a B x L x H x D input: it wants positions on the second-to-last axis."""
    rotated = mx.fast.rope(
        x.swapaxes(1, 2), x.shape[-1], traditional=False, base=THETA, scale=1.0, offset=offset
    )
    return rotated.swapaxes(1, 2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("offset", [0, 17])
def test_rope_matches_mx_fast_rope(dtype, offset):
    rope = RoPE(head_dim=128, max_seq_len=64, theta=THETA)
    x = mx.random.normal((2, 9, 4, 128)).astype(dtype)

    assert_allclose(rope(x, mx.arange(offset, offset + 9)), fast_rope(x, offset))


@pytest.mark.parametrize("position", [4000, 30000])
def test_long_positions_stay_within_the_fp32_angle_error(position):
    # An fp32 angle pos * theta^(-2i/D) is off by ~pos * 2^-24, so neither this nor
    # mx.fast.rope can meet the op tolerance far out; both are held to an exact reference.
    head_dim = 128
    x = mx.random.normal((1, 1, 1, head_dim))
    values, half = x.reshape(-1).tolist(), head_dim // 2

    exact = [0.0] * head_dim
    for i in range(half):
        angle = position * THETA ** (-2 * i / head_dim)
        cos, sin = math.cos(angle), math.sin(angle)
        exact[i] = values[i] * cos - values[i + half] * sin
        exact[i + half] = values[i + half] * cos + values[i] * sin

    bound = position * 2**-20 * max(abs(v) for v in values)
    for rotated in (RoPE(head_dim, position + 1, THETA)(x, mx.array([position])), fast_rope(x, position)):
        worst = max(abs(a - b) for a, b in zip(rotated.reshape(-1).tolist(), exact, strict=True))
        assert worst < bound, f"{worst:.3g} >= {bound:.3g}"


def test_per_row_positions_rotate_each_row_independently():
    rope = RoPE(head_dim=64, max_seq_len=256, theta=THETA)
    x = mx.random.normal((3, 5, 2, 64))
    offsets = [0, 40, 200]
    positions = mx.array([[offset + i for i in range(5)] for offset in offsets])

    out = rope(x, positions)
    for row in range(len(offsets)):
        assert mx.array_equal(out[row : row + 1], rope(x[row : row + 1], positions[row])).item()


def test_positions_are_explicit_not_assumed():
    # A gap in the positions must rotate each token by its own position, not its index.
    rope = RoPE(head_dim=64, max_seq_len=256, theta=THETA)
    x = mx.random.normal((1, 4, 2, 64))
    positions = [0, 1, 2, 7]

    out = rope(x, mx.array(positions))
    for i, position in enumerate(positions):
        assert_allclose(out[:, i : i + 1], fast_rope(x[:, i : i + 1], position))
