"""Step 1.3 — rotary position embedding, driven by an explicit position tensor.

The one design decision that matters here is the interface, not the arithmetic:
:meth:`RoPE.__call__` takes the absolute position of every token as a tensor and
never assumes ``arange(L)``.

That costs nothing now and is what makes Phase 4 possible. A decode step for a
sequence at position 500 and a prefill chunk covering positions 0-511 have to
share a single forward pass (DESIGN.md §7.3), so position is a property of each
token rather than of the batch. An implicit ``arange`` here would have to be torn
out later, and the resulting bug — correct prefill, subtly wrong continuation —
is among the hardest in the project to see.
"""

from __future__ import annotations

import torch

__all__ = ["RoPE", "apply_rope", "rotate_half"]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[x1, x2] -> [-x2, x1]``, splitting the head dimension in half.

    This is the "rotate halves" convention, which pairs element ``i`` with
    element ``i + D/2``. Note that the original RoFormer paper pairs *adjacent*
    elements instead. The two are related by a permutation of the head
    dimension, so both are self-consistent, but they are not interchangeable
    against a given checkpoint: Qwen3's weights were trained with this one, and
    using the other produces fluent nonsense rather than an error.
    """
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Rotate ``x`` using precomputed tables, gathered at ``positions``.

    ::

        x:         B x L x H x D
        positions: L  or  B x L
        cos, sin:  max_seq_len x D
        out:       same shape and dtype as x

    Split out from :class:`RoPE` so it can take the tables as plain tensors: the
    fused kernel in Step 3.2 has the same signature, which lets
    `mini_vllm.kernels.ops.rope` dispatch between the two without either side
    knowing about the other.
    """
    # positions is L or B x L, so the gather yields L x D or B x L x D; the head
    # axis is inserted so one row applies to every head of its token.
    gathered_cos = cos.to(x.device)[positions].unsqueeze(-2)
    gathered_sin = sin.to(x.device)[positions].unsqueeze(-2)

    rotated = x.float() * gathered_cos + rotate_half(x.float()) * gathered_sin
    return rotated.to(x.dtype)


class RoPE:
    """Precomputed rotary embedding tables, applied at explicit positions.

    ::

        x:         B x L x H x D   (or any N.. x H x D with positions to match)
        positions: L  or  B x L    (int64, absolute position of each token)
        out:       same shape as x

    ``cos`` and ``sin`` are exposed as attributes because the fused RoPE kernel
    in Step 3.2 reads these same tables directly, and it must agree with this
    implementation row for row.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 1_000_000.0,
        device: torch.device | str | None = None,
    ) -> None:
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Frequency i decays as theta^(-2i/D): the first pairs rotate quickly and
        # encode local offsets, the last rotate slowly and encode long-range
        # position. Qwen3's theta of 1e6 is what stretches the slow end far
        # enough to cover a long context.
        exponents = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
        inverse_frequencies = 1.0 / (theta**exponents)

        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = torch.outer(positions, inverse_frequencies)  # max_seq_len x D/2

        # Duplicated rather than interleaved, to match `rotate_half` above: the
        # rotation angle for element i and element i + D/2 is the same.
        angles = torch.cat((angles, angles), dim=-1)  # max_seq_len x D

        # Tables stay fp32 even when activations are bf16; a bf16 cosine near
        # zero crossing loses enough precision to shift tokens.
        self.cos = angles.cos()
        self.sin = angles.sin()

    def __call__(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate ``x`` in place-of-position, returning ``x``'s dtype."""
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"positions must be integer, got {positions.dtype}")

        maximum = int(positions.max()) if positions.numel() else -1
        if maximum >= self.max_seq_len:
            raise ValueError(
                f"position {maximum} is beyond the precomputed table "
                f"(max_seq_len={self.max_seq_len})"
            )

        return apply_rope(x, positions, self.cos, self.sin)
