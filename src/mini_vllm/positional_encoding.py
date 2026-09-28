"""RoPE: precomputed rotary tables, applied at explicit absolute positions."""

from __future__ import annotations

import mlx.core as mx

__all__ = ["RoPE", "rotate_half"]


def rotate_half(x: mx.array) -> mx.array:
    """[x1, x2] -> [-x2, x1], pairing element i with element i + D/2 as Qwen3 was trained."""
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


class RoPE:
    """Rotate x [B, L, H, D] by the angles at positions [B, L] or [L]."""

    def __init__(
        self, head_dim: int, max_seq_len: int, theta: float = 1_000_000.0, use_metal: bool = False
    ) -> None:
        self.use_metal = use_metal
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Frequency i decays as theta^(-2i/D): early pairs are local, late ones long-range.
        exponents = mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim
        inverse_frequencies = 1.0 / (theta**exponents)

        positions = mx.arange(max_seq_len, dtype=mx.float32)
        angles = mx.outer(positions, inverse_frequencies)  # max_seq_len x D/2

        # Duplicated rather than interleaved, matching `rotate_half`.
        angles = mx.concatenate([angles, angles], axis=-1)  # max_seq_len x D

        # Tables stay fp32: a bf16 cosine near a zero crossing would shift tokens.
        self.cos = mx.cos(angles)
        self.sin = mx.sin(angles)

    def __call__(self, x: mx.array, positions: mx.array) -> mx.array:
        if self.use_metal:
            import mini_vllm_ext

            # The same fp32 tables, gathered in the kernel.
            return mini_vllm_ext.rope(x, positions, self.cos, self.sin)

        # The head axis is inserted so one table row applies to every head of its token.
        cos = mx.expand_dims(self.cos[positions], -2)
        sin = mx.expand_dims(self.sin[positions], -2)

        x32 = x.astype(mx.float32)
        return (x32 * cos + rotate_half(x32) * sin).astype(x.dtype)
