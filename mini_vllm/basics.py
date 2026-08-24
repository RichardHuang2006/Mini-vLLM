"""The three primitives everything else is built from.

These are the reference implementations the faster paths are diffed against: `softmax` is
what the online-softmax decode attention kernel must agree with, and `silu` what the
fused SwiGLU kernel must agree with. Written for clarity rather than speed.

Shapes are written with `N..` for any number of leading batch dimensions.
"""

from __future__ import annotations

import torch

__all__ = ["linear", "silu", "softmax"]


def linear(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``y = x @ w.T (+ bias)``.

    ::

        x:    N.. x I
        w:    O x I        (transposed, the HuggingFace storage convention)
        bias: O
        out:  N.. x O

    The weight is stored as ``O x I`` rather than ``I x O`` because that is the
    checkpoint convention. Matching it here means the weight loader never transposes,
    and a transposed weight surfaces as a shape error rather than wrong numbers.
    """
    out = x @ w.transpose(-2, -1)
    if bias is not None:
        out = out + bias
    return out


def silu(x: torch.Tensor) -> torch.Tensor:
    """``x * sigmoid(x)``, the activation inside Qwen3's SwiGLU MLP."""
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax along ``dim``, computed in fp32 and returned in the input dtype.

    Two properties that recur in every attention kernel:

    Subtract the row max first. ``exp`` overflows to ``inf`` around 88 in fp32 and
    attention logits routinely exceed that; subtracting the max makes the largest
    exponent exactly ``exp(0) == 1`` without changing the result, since the shared factor
    cancels between numerator and denominator. This is the basis of the online-softmax
    recurrence in the decode attention kernel, where the max arrives incrementally and
    the running total is rescaled as it changes.

    Reduce in fp32. Summing bf16 exponentials loses enough precision to move greedy
    tokens a few layers downstream.
    """
    x32 = x.float()
    x32 = x32 - x32.max(dim=dim, keepdim=True).values
    exp = torch.exp(x32)
    return (exp / exp.sum(dim=dim, keepdim=True)).to(x.dtype)
