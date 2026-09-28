"""basics.py against mlx.core and mlx.nn."""

import mlx.core as mx
import mlx.nn as nn
import pytest
from utils import DTYPES, assert_allclose, requires_metal

from mini_vllm.basics import linear, silu, softmax, swiglu


@pytest.mark.parametrize("dtype", DTYPES)
def test_linear_matches_nn_linear(dtype):
    layer = nn.Linear(48, 32, bias=True)
    layer.set_dtype(dtype)
    x = mx.random.normal((3, 5, 48)).astype(dtype)

    assert_allclose(linear(x, layer.weight, layer.bias), layer(x))
    assert_allclose(linear(x, layer.weight), x @ layer.weight.T)


@pytest.mark.parametrize("dtype", DTYPES)
def test_silu_and_swiglu_match_nn_silu(dtype):
    gate = (4 * mx.random.normal((7, 64))).astype(dtype)
    up = mx.random.normal((7, 64)).astype(dtype)

    assert_allclose(silu(gate), nn.silu(gate))
    assert_allclose(swiglu(gate, up), nn.silu(gate) * up)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("axis", [-1, 0])
def test_softmax_matches_mx_softmax(dtype, axis):
    x = (3 * mx.random.normal((6, 33))).astype(dtype)
    assert_allclose(softmax(x, axis=axis), mx.softmax(x, axis=axis))


def test_softmax_is_stable_at_large_logits():
    x = mx.array([[1e4, 1e4 - 1.0, -1e4], [88.0, 89.0, 90.0]])
    out = softmax(x)

    assert not mx.any(mx.isnan(out)).item()
    assert_allclose(out, mx.softmax(x, axis=-1, precise=True))


@requires_metal
@pytest.mark.parametrize("dtype", DTYPES)
def test_the_swiglu_kernel_matches_the_oracle(dtype):
    gate = (4 * mx.random.normal((5, 3072))).astype(dtype)
    up = mx.random.normal((5, 3072)).astype(dtype)
    assert_allclose(swiglu(gate, up, use_metal=True), swiglu(gate, up))
    # A transposed view is made contiguous before the kernel reads it.
    assert_allclose(swiglu(gate.T, up.T, use_metal=True), swiglu(gate.T, up.T))
