"""FP8 (e4m3) KV quantization with static per-tensor scales, stored as uint8."""

from __future__ import annotations

import mlx.core as mx

__all__ = ["FP8_MAX", "dequantize_fp8", "quantize_fp8"]

# The largest finite e4m3 value; mx.to_fp8 saturates anything beyond it.
FP8_MAX = 448.0


def quantize_fp8(x: mx.array, scale: float = 1.0) -> mx.array:
    """x / scale as e4m3 bits in uint8: half the bytes of bf16, 3 mantissa bits."""
    return mx.to_fp8(x.astype(mx.float32) / scale)


def dequantize_fp8(x: mx.array, scale: float = 1.0, dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """The inverse of quantize_fp8: decode the uint8 bits, multiply the scale back in."""
    return (mx.from_fp8(x, mx.float32) * scale).astype(dtype)
