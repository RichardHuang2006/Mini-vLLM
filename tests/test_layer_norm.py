"""layer_norm.py against mx.fast.rms_norm."""

import mlx.core as mx
import pytest
from utils import DTYPES, assert_allclose, requires_metal

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


@requires_metal
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [128, 1024, 3000])  # a q/k head, the hidden size, a strided row
def test_the_rms_norm_kernel_matches_the_oracle(dtype, dim):
    """In bf16 and fp16 the kernel is bit-exact, which is what shows it rounds before the
    weight multiply as the oracle does. In fp32 it sums the squares in a different order,
    so a fraction of outputs move by an ulp."""
    weight = (1 + 0.1 * mx.random.normal((dim,))).astype(dtype)
    x = (3 * mx.random.normal((4, 3, dim))).astype(dtype)
    norm, metal = RMSNorm(dim, weight), RMSNorm(dim, weight, use_metal=True)
    for rows in (x, x.swapaxes(0, 1)):  # contiguous, then a strided view
        if dtype == mx.float32:
            assert_allclose(metal(rows), norm(rows))
        else:
            assert mx.array_equal(metal(rows), norm(rows)).item()
