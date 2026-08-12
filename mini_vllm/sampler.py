"""Greedy, temperature, top-k and top-p sampling.

Everything here is **vectorized across the batch with per-row parameters**, which
is the one design constraint that matters. A server batches whatever requests
happen to arrive together, and they will not agree on temperature: row 0 may be
greedy while row 1 wants `temperature=1.2, top_p=0.9`. Looping over rows to honour
that would put a Python loop inside the decode step — the single hottest path in
the engine — so instead every row is masked and scaled in parallel, and greedy is
handled as `temperature == 0` rather than as a separate code path.

The cost is one sort of the vocabulary axis per step, which is what makes per-row
top-k and top-p expressible as pure tensor ops.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

__all__ = ["SamplingParams", "sample", "sampling_probabilities"]


@dataclass(frozen=True)
class SamplingParams:
    """One request's sampling configuration.

    ``temperature=0`` means greedy. ``top_k=0`` and ``top_p=1.0`` both mean
    "disabled", so the default is plain temperature-1 sampling over the full
    distribution.
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0


def _as_columns(
    params: SamplingParams | Sequence[SamplingParams],
    batch: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Broadcast per-request parameters into three ``B x 1`` tensors."""
    rows = [params] * batch if isinstance(params, SamplingParams) else list(params)
    if len(rows) != batch:
        raise ValueError(f"got {len(rows)} sampling params for a batch of {batch}")

    def column(values, dtype) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device=device).unsqueeze(1)

    return (
        column([row.temperature for row in rows], torch.float32),
        column([row.top_k for row in rows], torch.int64),
        column([row.top_p for row in rows], torch.float32),
    )


def sampling_probabilities(
    logits: torch.Tensor,
    params: SamplingParams | Sequence[SamplingParams],
) -> torch.Tensor:
    """The distribution each row will actually be sampled from.

    ::

        logits: B x V   ->   probabilities: B x V   (fp32, rows sum to 1)

    Exposed separately from :func:`sample` because it is what makes the sampler
    testable: a truncation rule is much easier to verify by inspecting the
    distribution it produces than by drawing from it. Greedy rows come back as a
    one-hot row.

    Temperature is applied first, then top-k and top-p together on the scaled
    distribution — the truncation has to see the same probabilities the draw will.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected B x V logits, got shape {tuple(logits.shape)}")

    batch, vocab = logits.shape
    temperature, top_k, top_p = _as_columns(params, batch, logits.device)

    greedy = temperature == 0.0
    # Divide by 1.0 on greedy rows to keep this branch-free and finite; those rows
    # are overwritten with their one-hot below.
    scaled = logits.float() / torch.where(greedy, torch.ones_like(temperature), temperature)

    descending, order = scaled.sort(dim=-1, descending=True)
    probabilities = descending.softmax(dim=-1)

    # top-k: keep ranks [0, k). k == 0 disables it.
    rank = torch.arange(vocab, device=logits.device).unsqueeze(0)
    effective_k = torch.where(top_k == 0, torch.full_like(top_k, vocab), top_k)
    keep = rank < effective_k

    # top-p: keep a token when the probability mass strictly *before* it is still
    # below p. That yields the smallest prefix whose total reaches p, and includes
    # the boundary token that crosses it rather than stopping short of p.
    mass_before = probabilities.cumsum(dim=-1) - probabilities
    keep &= mass_before < top_p

    # The most likely token is always kept, so no row can be fully masked however
    # small p is.
    keep[:, 0] = True

    truncated = probabilities * keep
    truncated = truncated / truncated.sum(dim=-1, keepdim=True)

    # Back to vocabulary order.
    result = torch.zeros_like(truncated).scatter_(dim=-1, index=order, src=truncated)

    if bool(greedy.any()):
        one_hot = torch.zeros_like(result).scatter_(
            dim=-1, index=logits.argmax(dim=-1, keepdim=True), value=1.0
        )
        result = torch.where(greedy, one_hot, result)

    return result


def sample(
    logits: torch.Tensor,
    params: SamplingParams | Sequence[SamplingParams] = SamplingParams(),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one token per row.

    ::

        logits: B x V   ->   tokens: B   (int64)

    Pass a ``generator`` to make a draw reproducible without disturbing global
    RNG state, which is what lets a server replay one request.
    """
    probabilities = sampling_probabilities(logits, params)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(1)
