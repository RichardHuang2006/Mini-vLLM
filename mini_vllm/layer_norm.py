"""Step 1.5 — RMSNorm.

Qwen3 uses this in three places, and the third is easy to miss: before attention,
before the MLP, and *inside* attention as QK-norm, applied over the head
dimension of `q` and `k` (DESIGN.md §3). The same function serves all three; only
the axis width differs.

Kept forever as the oracle for the CUDA kernel in Step 3.1.
"""

from __future__ import annotations

import torch

__all__ = ["RMSNorm", "rms_norm"]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``x * rsqrt(mean(x²) + eps) * weight``, reducing over the last dimension.

    ::

        x:      N.. x dim
        weight: dim
        out:    N.. x dim

    Unlike LayerNorm there is no mean subtraction and no bias: the vector is
    rescaled but not recentred, which is cheaper and works as well in practice.

    **The reduction is fp32 even when ``x`` is bf16, and that is not optional.**
    A bf16 sum of 1024 squares carries roughly three decimal digits, and the
    error feeds straight into a multiplicative rescale of the whole residual
    stream. Across 28 layers it moves greedy tokens, which presents as a vague
    "the model is a bit wrong" rather than as a dtype bug.

    The cast back to the input dtype happens *before* the weight multiply, which
    is what HuggingFace does. Matching that ordering keeps this exactly
    comparable to the oracle rather than approximately.
    """
    input_dtype = x.dtype

    x32 = x.float()
    mean_square = x32.pow(2).mean(dim=-1, keepdim=True)
    normalized = x32 * torch.rsqrt(mean_square + eps)

    return weight * normalized.to(input_dtype)


class RMSNorm:
    """`rms_norm` bound to a weight and an epsilon."""

    def __init__(self, dim: int, weight: torch.Tensor, eps: float = 1e-6) -> None:
        if weight.shape != (dim,):
            raise ValueError(f"weight must have shape ({dim},), got {tuple(weight.shape)}")
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.shape[-1]}")
        return rms_norm(x, self.weight, self.eps)
