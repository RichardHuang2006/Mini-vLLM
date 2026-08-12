"""The three primitives everything else is built from.

Readable by design. These are the reference implementations that the faster
paths are diffed against: `softmax` here is what the online-softmax decode
attention kernel must agree with, and `silu` is what the fused SwiGLU kernel
must agree with. So the goal is obviousness, not speed.

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

    The weight is stored as ``O x I`` rather than ``I x O`` because that is how
    every checkpoint stores it; matching the convention here means the weight
    loader never has to transpose, and a transposed weight shows up as a shape
    error instead of as silently wrong numbers.
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

    Two decisions worth understanding, because both recur in every attention
    kernel:

    **Subtract the max first.** ``exp`` overflows to ``inf`` around 88 in fp32,
    and attention logits routinely exceed that. Subtracting the row max makes
    the largest exponent exactly ``exp(0) == 1`` without changing the result,
    since a shared factor cancels between numerator and denominator. This is the
    seed of the online-softmax recurrence in the decode attention kernel: there
    the max arrives incrementally, so the running total has to be rescaled as it
    changes.

    **Reduce in fp32.** Summing bf16 exponentials loses enough precision to move
    greedy tokens a few layers downstream, which reads as a model bug rather
    than a dtype bug.
    """
    x32 = x.float()
    x32 = x32 - x32.max(dim=dim, keepdim=True).values
    exp = torch.exp(x32)
    return (exp / exp.sum(dim=dim, keepdim=True)).to(x.dtype)
