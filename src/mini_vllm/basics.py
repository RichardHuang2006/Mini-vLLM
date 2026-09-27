"""The primitives every other operator is built from: linear, softmax, silu, swiglu."""

from __future__ import annotations

import mlx.core as mx

__all__ = ["linear", "silu", "softmax", "swiglu"]


def linear(x: mx.array, w: mx.array, bias: mx.array | None = None) -> mx.array:
    """y = x @ w.T (+ bias), with w stored O x I as the checkpoint keeps it."""
    out = x @ w.T
    if bias is not None:
        out = out + bias
    return out


def softmax(x: mx.array, axis: int = -1) -> mx.array:
    """Softmax along axis, computed in fp32 and returned in the input dtype."""
    x32 = x.astype(mx.float32)
    # Subtract the row max: exp overflows around 88 in fp32 and the factor cancels.
    x32 = x32 - mx.max(x32, axis=axis, keepdims=True)
    exp = mx.exp(x32)
    return (exp / mx.sum(exp, axis=axis, keepdims=True)).astype(x.dtype)


def silu(x: mx.array) -> mx.array:
    """x * sigmoid(x), the activation inside Qwen3's SwiGLU MLP."""
    return x * mx.sigmoid(x)


def swiglu(gate: mx.array, up: mx.array) -> mx.array:
    """silu(gate) * up: the MLP's two up-projections merged before the down-projection."""
    return silu(gate) * up
