"""The token embedding table, read as a lookup or, for the tied LM head, as a projection."""

from __future__ import annotations

import mlx.core as mx

from mini_vllm.basics import linear

__all__ = ["Embedding"]


class Embedding:
    """A V x E table: row lookup on the way in, h @ weight.T on the way out."""

    def __init__(self, vocab_size: int, dim: int, weight: mx.array) -> None:
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = weight

    def __call__(self, ids: mx.array) -> mx.array:
        """Gather one row per token id."""
        return self.weight[ids]

    def as_linear(self, h: mx.array) -> mx.array:
        """h @ weight.T: the tied LM head, and the most expensive op in a decode step."""
        return linear(h, self.weight)
