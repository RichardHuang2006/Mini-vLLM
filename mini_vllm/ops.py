"""Reference implementations: every operation the engine runs, in its slow, correct form."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from mini_vllm.config import SamplingParams

__all__ = [
    "Embedding",
    "RoPE",
    "apply_rope",
    "causal_mask",
    "linear",
    "paged_attention_gathered",
    "rms_norm",
    "rotate_half",
    "sample",
    "sampling_probabilities",
    "scaled_dot_product_attention_grouped",
    "silu",
    "softmax",
]

# SamplingParams is frozen, so every caller can share one default instance.
_DEFAULT_SAMPLING = SamplingParams()


def linear(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """y = x @ w.T (+ bias), with w stored O x I as the checkpoint keeps it."""
    out = x @ w.transpose(-2, -1)
    if bias is not None:
        out = out + bias
    return out


class Embedding:
    """A V x E table, read as a lookup or, for the tied LM head, as a linear projection."""

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
        """h @ weight.T: the tied LM head, and the most expensive op in a decode step."""
        if h.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {h.shape[-1]}")
        return linear(h, self.weight)


def silu(x: torch.Tensor) -> torch.Tensor:
    """x * sigmoid(x), the activation inside Qwen3's SwiGLU MLP."""
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax along dim, computed in fp32 and returned in the input dtype."""
    x32 = x.float()
    # Subtract the row max: exp overflows around 88 in fp32 and the factor cancels.
    x32 = x32 - x32.max(dim=dim, keepdim=True).values
    exp = torch.exp(x32)
    return (exp / exp.sum(dim=dim, keepdim=True)).to(x.dtype)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x * rsqrt(mean(x^2) + eps) * weight over the last axis, reduced in fp32."""
    input_dtype = x.dtype

    x32 = x.float()
    mean_square = x32.pow(2).mean(dim=-1, keepdim=True)
    normalized = x32 * torch.rsqrt(mean_square + eps)

    # The cast back happens before the weight multiply, matching HuggingFace exactly.
    return weight * normalized.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[x1, x2] -> [-x2, x1], pairing element i with element i + D/2 as Qwen3 was trained."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Rotate x [B, L, H, D] using cos/sin tables [max_seq_len, D] gathered at positions."""
    # The head axis is inserted so one table row applies to every head of its token.
    gathered_cos = cos.to(x.device)[positions].unsqueeze(-2)
    gathered_sin = sin.to(x.device)[positions].unsqueeze(-2)

    rotated = x.float() * gathered_cos + rotate_half(x.float()) * gathered_sin
    return rotated.to(x.dtype)


class RoPE:
    """Precomputed rotary tables, applied at explicit absolute positions."""

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

        # Frequency i decays as theta^(-2i/D): early pairs are local, late ones long-range.
        exponents = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
        inverse_frequencies = 1.0 / (theta**exponents)

        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = torch.outer(positions, inverse_frequencies)  # max_seq_len x D/2

        # Duplicated rather than interleaved, matching `rotate_half`.
        angles = torch.cat((angles, angles), dim=-1)  # max_seq_len x D

        # Tables stay fp32: a bf16 cosine near a zero crossing would shift tokens.
        self.cos = angles.cos()
        self.sin = angles.sin()

    def __call__(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate ``x`` at ``positions``, returning ``x``'s dtype."""
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"positions must be integer, got {positions.dtype}")

        maximum = int(positions.max()) if positions.numel() else -1
        if maximum >= self.max_seq_len:
            raise ValueError(
                f"position {maximum} is beyond the precomputed table "
                f"(max_seq_len={self.max_seq_len})"
            )

        return apply_rope(x, positions, self.cos, self.sin)


def scaled_dot_product_attention_grouped(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:
    """Grouped-query attention: q is B x H_q x L x D, k and v are B x H_k x S x D."""
    *batch, num_query_heads, query_len, head_dim = q.shape
    num_kv_heads = k.shape[-3]

    if num_query_heads % num_kv_heads != 0:
        raise ValueError(f"H_q ({num_query_heads}) must be a multiple of H_k ({num_kv_heads})")
    group_size = num_query_heads // num_kv_heads
    source_len = k.shape[-2]

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Split the query heads into (kv_head, group) so K and V broadcast over the group.
    q = q.reshape(*batch, num_kv_heads, group_size, query_len, head_dim)
    k = k.unsqueeze(-3)
    v = v.unsqueeze(-3)

    scores = (q @ k.transpose(-2, -1)) * scale

    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError(f"unknown mask shorthand {mask!r}, expected 'causal'")
        mask = causal_mask(query_len, source_len, q.dtype, q.device)
    if mask is not None:
        if mask.ndim == 4:
            # A per-head mask needs the same (kv_head, group) split as the query.
            mask = mask.reshape(*mask.shape[:-3], num_kv_heads, group_size, query_len, source_len)
        elif mask.ndim == 3:
            # (B, L, S) and (H_q, L, S) are indistinguishable from the shape alone.
            raise ValueError(
                "ambiguous 3-D mask: pass L x S to share across heads, "
                "or B x H_q x L x S to vary per head"
            )
        scores = scores + mask

    out = softmax(scores, dim=-1) @ v
    return out.reshape(*batch, num_query_heads, query_len, head_dim)


def causal_mask(
    query_len: int,
    source_len: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """An additive [query_len, source_len] mask: query i may attend to key j <= S - L + i."""
    offset = source_len - query_len
    rows = torch.arange(query_len, device=device).unsqueeze(1)
    columns = torch.arange(source_len, device=device).unsqueeze(0)

    allowed = columns <= rows + offset
    return torch.zeros(query_len, source_len, dtype=dtype, device=device).masked_fill(
        ~allowed, float("-inf")
    )


def paged_attention_gathered(
    q: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float | None = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> torch.Tensor:
    """Causal grouped attention over a paged cache, gathering each sequence's pages first.

    q is T x H_q x D, the pools are num_blocks x P x H_k x D, and the int32 metadata is
    block_tables (N x max_blocks, -1 padded), cu_seqlens_q (N + 1) and context_lens (N).
    """
    if q.dim() != 3:
        raise ValueError(f"expected q shaped T x H_q x D, got {tuple(q.shape)}")
    if key_pool.dim() != 4 or key_pool.shape != value_pool.shape:
        raise ValueError(
            f"expected matching pools shaped num_blocks x P x H_k x D, got "
            f"{tuple(key_pool.shape)} and {tuple(value_pool.shape)}"
        )

    num_sequences = context_lens.shape[0]
    if block_tables.shape[0] != num_sequences or cu_seqlens_q.shape[0] != num_sequences + 1:
        raise ValueError(
            f"metadata disagrees on the sequence count: block_tables "
            f"{tuple(block_tables.shape)}, cu_seqlens_q {tuple(cu_seqlens_q.shape)}, "
            f"context_lens {tuple(context_lens.shape)}"
        )
    if int(cu_seqlens_q[-1]) != q.shape[0]:
        raise ValueError(
            f"cu_seqlens_q ends at {int(cu_seqlens_q[-1])} but q has {q.shape[0]} rows"
        )

    block_size, num_kv_heads, head_dim = key_pool.shape[1:]
    out = torch.empty_like(q)

    for index in range(num_sequences):
        start, end = int(cu_seqlens_q[index]), int(cu_seqlens_q[index + 1])
        query_len = end - start
        context_len = int(context_lens[index])
        if context_len < query_len:
            raise ValueError(
                f"sequence {index} attends over {context_len} tokens but computes "
                f"{query_len}; S >= L is causality, not convention"
            )

        blocks_used = -(-context_len // block_size)
        table = block_tables[index, :blocks_used]
        if int(table.min()) < 0:
            raise ValueError(
                f"sequence {index} needs {blocks_used} blocks for {context_len} tokens "
                f"but its table is padded there: {block_tables[index].tolist()}"
            )

        # index_select on the block axis, then flatten block and offset into one axis.
        ids = table.to(dtype=torch.int64, device=key_pool.device)
        keys = key_pool.index_select(0, ids).reshape(-1, num_kv_heads, head_dim)
        values = value_pool.index_select(0, ids).reshape(-1, num_kv_heads, head_dim)

        # Dequantize an FP8 cache before the math; a no-op cast for a matching pool.
        if keys.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            keys = keys.float().mul_(k_scale).to(q.dtype)
            values = values.float().mul_(v_scale).to(q.dtype)

        keys = keys[:context_len].permute(1, 0, 2).unsqueeze(0)  # 1 x H_k x S x D
        values = values[:context_len].permute(1, 0, 2).unsqueeze(0)
        queries = q[start:end].permute(1, 0, 2).unsqueeze(0)  # 1 x H_q x L x D

        attended = scaled_dot_product_attention_grouped(
            queries, keys, values, scale=scale, mask="causal"
        )
        out[start:end] = attended.squeeze(0).permute(1, 0, 2)

    return out


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
    """The fp32 [B, V] distribution each row will be sampled from, vectorized per row."""
    if logits.ndim != 2:
        raise ValueError(f"expected B x V logits, got shape {tuple(logits.shape)}")

    batch, vocab = logits.shape
    temperature, top_k, top_p = _as_columns(params, batch, logits.device)

    greedy = temperature == 0.0
    # Divide by 1.0 on greedy rows to stay branch-free; they are overwritten below.
    scaled = logits.float() / torch.where(greedy, torch.ones_like(temperature), temperature)

    descending, order = scaled.sort(dim=-1, descending=True)
    probabilities = descending.softmax(dim=-1)

    # top-k: keep ranks [0, k). k == 0 disables it.
    rank = torch.arange(vocab, device=logits.device).unsqueeze(0)
    effective_k = torch.where(top_k == 0, torch.full_like(top_k, vocab), top_k)
    keep = rank < effective_k

    # top-p: keep a token while the mass strictly before it is below p, boundary included.
    mass_before = probabilities.cumsum(dim=-1) - probabilities
    keep &= mass_before < top_p

    # The most likely token is always kept, so no row is fully masked however small p is.
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
    params: SamplingParams | Sequence[SamplingParams] = _DEFAULT_SAMPLING,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one token per row: logits [B, V] -> tokens [B]; greedy rows are one-hot."""
    probabilities = sampling_probabilities(logits, params)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(1)
