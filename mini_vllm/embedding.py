"""The embedding table, used in both directions.

Qwen3-0.6B sets ``tie_word_embeddings=true``, so one matrix serves as both the
input embedding and the output projection. That is why this is a single object
with two methods rather than two separate layers: `__call__` reads rows out of it,
`as_linear` multiplies by its transpose to produce logits.

Worth noticing how much of the model this one matrix is. At `V x E` =
151936 x 1024 it is about 155M of the 596M parameters, and untying it would add
another 155M for no benefit at this scale.
"""

from __future__ import annotations

import torch

from mini_vllm.basics import linear

__all__ = ["Embedding"]


class Embedding:
    """A ``V x E`` table, readable as a lookup or as a linear projection.

    ::

        weight:       V x E
        __call__(ids):  B x L      (int64)  ->  B x L x E
        as_linear(h):   B x L x E            ->  B x L x V
    """

    def __init__(self, vocab_size: int, dim: int, weight: torch.Tensor) -> None:
        if weight.shape != (vocab_size, dim):
            raise ValueError(
                f"weight must have shape ({vocab_size}, {dim}), got {tuple(weight.shape)}"
            )
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = weight

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        """Gather one row per token id."""
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"ids must be integer, got {ids.dtype}")
        return self.weight[ids]

    def as_linear(self, h: torch.Tensor) -> torch.Tensor:
        """``h @ weightᵀ`` — the tied LM head.

        A vocabulary-wide matmul, so this is the single most expensive op in a
        decode step: `E x V` work to produce logits for one token. The serving
        layer only ever runs it on the *last* position of each sequence for that
        reason.
        """
        if h.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {h.shape[-1]}")
        return linear(h, self.weight)
