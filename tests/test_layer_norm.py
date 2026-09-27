"""layer_norm.py against mx.fast.rms_norm."""

import mlx.core as mx
import pytest
from utils import DTYPES, assert_allclose

from mini_vllm.layer_norm import RMSNorm


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [128, 1024])  # Qwen3-0.6B's per-head q/k norm and its hidden size
def test_rms_norm_matches_mx_fast(dtype, dim):
    weight = (1 + 0.1 * mx.random.normal((dim,))).astype(dtype)
    x = (3 * mx.random.normal((2, 5, dim))).astype(dtype)

    assert_allclose(RMSNorm(dim, weight)(x), mx.fast.rms_norm(x, weight, 1e-6))


def test_rms_norm_casts_back_before_the_weight_multiply():
    weight = (1 + 0.1 * mx.random.normal((64,))).astype(mx.bfloat16)
    x = mx.random.normal((4, 64)).astype(mx.bfloat16)

    x32 = x.astype(mx.float32)
    normalized = x32 * mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + 1e-6)
    expected = weight * normalized.astype(mx.bfloat16)

    assert mx.array_equal(RMSNorm(64, weight)(x), expected).item()


def test_a_misshaped_weight_is_rejected():
    with pytest.raises(ValueError, match="weight must have shape"):
        RMSNorm(64, mx.ones((32,)))
