"""RMSNorm: the pre-attention, pre-MLP, final and per-head q/k norms."""

from __future__ import annotations

import mlx.core as mx

__all__ = ["RMSNorm"]


class RMSNorm:
    """x * rsqrt(mean(x^2) + eps) * weight over the last axis, reduced in fp32."""

    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-6) -> None:
        if weight.shape != (dim,):
            raise ValueError(f"weight must have shape ({dim},), got {tuple(weight.shape)}")
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        input_dtype = x.dtype

        x32 = x.astype(mx.float32)
        mean_square = mx.mean(mx.square(x32), axis=-1, keepdims=True)
        normalized = x32 * mx.rsqrt(mean_square + self.eps)

        # The cast back happens before the weight multiply, matching HuggingFace exactly.
        return self.weight * normalized.astype(input_dtype)
