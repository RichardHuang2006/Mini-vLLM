"""RMSNorm.

Qwen3 uses this in three places: before attention, before the MLP, and inside attention
as QK-norm over the head dimension of `q` and `k`. The same function serves all three,
differing only in the width of the reduced axis.

Kept as the oracle for the CUDA RMSNorm kernel.
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

    Unlike LayerNorm there is no mean subtraction and no bias: the vector is rescaled
    but not recentred.

    The reduction is fp32 even when ``x`` is bf16, and that is required rather than
    conservative. A bf16 sum of 1024 squares carries roughly three decimal digits, and
    the error feeds a multiplicative rescale of the whole residual stream; across 28
    layers it moves greedy tokens.

    The cast back to the input dtype happens before the weight multiply, matching
    HuggingFace, so this is exactly rather than approximately comparable to the oracle.
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
