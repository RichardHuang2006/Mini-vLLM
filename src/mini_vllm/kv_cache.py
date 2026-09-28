"""The dense KV cache: one layer's keys and values, concatenated along the sequence axis."""

from __future__ import annotations

import mlx.core as mx

from mini_vllm.attention import scaled_dot_product_attention_grouped

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

    def attend(self, q: mx.array, k: mx.array, v: mx.array, use_metal: bool = False) -> mx.array:
        """Append this step's keys and values, then attend causally over the whole history.
        With S > L, the causal mask's S - L offset lines the new tokens up after it."""
        keys, values = self.update_and_fetch(k, v)
        return scaled_dot_product_attention_grouped(q, keys, values, mask="causal", use_metal=use_metal)
