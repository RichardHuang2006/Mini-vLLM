"""The dense KV cache: one layer's keys and values, concatenated along the sequence axis."""

from __future__ import annotations

import mlx.core as mx

__all__ = ["KvFullCache"]


class KvFullCache:
    """Grows by concatenation every step: O(S) copy per append, and the paged cache's oracle."""

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0  # how many positions are cached

    def update_and_fetch(self, key: mx.array, value: mx.array) -> tuple[mx.array, mx.array]:
        """Append key/value [B, H_k, L, D], returning the whole history [B, H_k, S, D]."""
        if self.keys is None:
            self.keys, self.values = key, value
        else:
            self.keys = mx.concatenate([self.keys, key], axis=-2)
            self.values = mx.concatenate([self.values, value], axis=-2)
        self.offset += key.shape[-2]
        return self.keys, self.values
