"""Per-request sampling: greedy, temperature, top-k and top-p, vectorized over the batch."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx

from mini_vllm.basics import softmax

__all__ = ["SamplingParams", "sample", "sampling_probabilities"]


@dataclass(frozen=True)
class SamplingParams:
    """One request's sampling configuration; temperature=0 is greedy, top_k=0 and top_p=1
    disable truncation."""

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0


# SamplingParams is frozen, so every caller can share one default instance.
_DEFAULT_SAMPLING = SamplingParams()


def _as_columns(
    params: SamplingParams | Sequence[SamplingParams],
    batch: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Broadcast per-request parameters into three B x 1 arrays."""
    rows = [params] * batch if isinstance(params, SamplingParams) else list(params)

    def column(values, dtype) -> mx.array:
        return mx.array(values, dtype=dtype)[:, None]

    return (
        column([row.temperature for row in rows], mx.float32),
        column([row.top_k for row in rows], mx.int32),
        column([row.top_p for row in rows], mx.float32),
    )


def sampling_probabilities(
    logits: mx.array,
    params: SamplingParams | Sequence[SamplingParams],
) -> mx.array:
    """The fp32 [B, V] distribution each row will be sampled from, vectorized per row."""
    batch, vocab = logits.shape
    temperature, top_k, top_p = _as_columns(params, batch)

    greedy = temperature == 0.0
    # Divide by 1.0 on greedy rows to stay branch-free; they are overwritten below.
    scaled = logits.astype(mx.float32) / mx.where(greedy, 1.0, temperature)

    order = mx.argsort(-scaled, axis=-1)
    probabilities = softmax(mx.take_along_axis(scaled, order, axis=-1), axis=-1)

    # top-k: keep ranks [0, k). k == 0 disables it.
    rank = mx.arange(vocab)[None, :]
    keep = rank < mx.where(top_k == 0, vocab, top_k)

    # top-p: keep a token while the mass strictly before it is below p, boundary included.
    mass_before = mx.cumsum(probabilities, axis=-1) - probabilities
    keep = keep & (mass_before < top_p)

    # The most likely token is always kept, so no row is fully masked however small p is.
    keep = keep | (rank == 0)

    truncated = mx.where(keep, probabilities, 0.0)
    truncated = truncated / mx.sum(truncated, axis=-1, keepdims=True)

    # Back to vocabulary order: argsort of a permutation is its inverse.
    result = mx.take_along_axis(truncated, mx.argsort(order, axis=-1), axis=-1)

    one_hot = (rank == mx.argmax(logits, axis=-1, keepdims=True)).astype(mx.float32)
    return mx.where(greedy, one_hot, result)


def sample(
    logits: mx.array,
    params: SamplingParams | Sequence[SamplingParams] = _DEFAULT_SAMPLING,
    key: mx.array | None = None,
) -> mx.array:
    """Draw one token per row: logits [B, V] -> int32 tokens [B]; greedy rows are exact."""
    probabilities = sampling_probabilities(logits, params)
    # log(0) = -inf, so truncated tokens and the zeros of a greedy one-hot are never drawn.
    return mx.random.categorical(mx.log(probabilities), axis=-1, key=key).astype(mx.int32)
