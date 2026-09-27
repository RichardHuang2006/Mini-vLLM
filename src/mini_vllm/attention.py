"""Grouped-query attention, dense and over a paged KV cache."""

from __future__ import annotations

import math

import mlx.core as mx

from mini_vllm.basics import softmax
from mini_vllm.quantize import dequantize_fp8

__all__ = ["causal_mask", "paged_attention", "scaled_dot_product_attention_grouped"]


def causal_mask(query_len: int, source_len: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    """An additive [L, S] mask: query i may attend to key j <= S - L + i."""
    # The k = S - L offset is what makes a chunk's rectangular mask line up with its history.
    allowed = mx.tril(mx.ones((query_len, source_len), dtype=mx.bool_), k=source_len - query_len)
    return mx.where(allowed, mx.array(0.0, dtype), mx.array(-mx.inf, dtype))


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    """query is B x H_q x L x D, key and value are B x H_k x S x D; mask is "causal" or
    additive and broadcastable to B x H_q x L x S."""
    *batch, num_query_heads, query_len, head_dim = query.shape
    num_kv_heads, source_len = key.shape[-3], key.shape[-2]

    if num_query_heads % num_kv_heads != 0:
        raise ValueError(f"H_q ({num_query_heads}) must be a multiple of H_k ({num_kv_heads})")
    group_size = num_query_heads // num_kv_heads

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Split the query heads into (kv_head, group) so K and V broadcast over the group
    # instead of being repeated G times.
    query = query.reshape(*batch, num_kv_heads, group_size, query_len, head_dim)
    key = mx.expand_dims(key, -3)
    value = mx.expand_dims(value, -3)

    scores = (query @ key.swapaxes(-2, -1)) * scale

    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError(f"unknown mask shorthand {mask!r}, expected 'causal'")
        if query_len > source_len:
            raise ValueError(f"causal attention needs S >= L, got L={query_len}, S={source_len}")
        scores = scores + causal_mask(query_len, source_len, scores.dtype)
    elif mask is not None:
        mask = mx.broadcast_to(mask, (*batch, num_query_heads, query_len, source_len))
        scores = scores + mask.reshape(*batch, num_kv_heads, group_size, query_len, source_len)

    out = softmax(scores, axis=-1) @ value
    return out.reshape(*batch, num_query_heads, query_len, head_dim)


def paged_attention(
    query: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_tables: mx.array,
    cu_seqlens_q: mx.array,
    context_lens: mx.array,
    scale: float | None = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> mx.array:
    """Causal grouped attention over a paged cache, gathering each sequence's pages first.

    query is T x H_q x D, the pages are num_blocks x P x H_k x D (uint8 for fp8), and the
    int32 metadata is block_tables (N x max_blocks, -1 padded), cu_seqlens_q (N + 1) and
    context_lens (N).
    """
    if query.ndim != 3:
        raise ValueError(f"expected query shaped T x H_q x D, got {tuple(query.shape)}")
    if key_pages.ndim != 4 or key_pages.shape != value_pages.shape:
        raise ValueError(
            f"expected matching pages shaped num_blocks x P x H_k x D, got "
            f"{tuple(key_pages.shape)} and {tuple(value_pages.shape)}"
        )
    is_fp8 = key_pages.dtype == mx.uint8
    if not is_fp8 and key_pages.dtype != query.dtype:
        raise ValueError(f"pages are {key_pages.dtype} but query is {query.dtype}")

    # The metadata is small and needed on the host to slice, so read it back once.
    tables = block_tables.tolist()
    starts = cu_seqlens_q.tolist()
    lengths = context_lens.tolist()

    num_sequences = len(lengths)
    if len(tables) != num_sequences or len(starts) != num_sequences + 1:
        raise ValueError(
            f"metadata disagrees on the sequence count: block_tables "
            f"{tuple(block_tables.shape)}, cu_seqlens_q {tuple(cu_seqlens_q.shape)}, "
            f"context_lens {tuple(context_lens.shape)}"
        )
    if starts[-1] != query.shape[0]:
        raise ValueError(f"cu_seqlens_q ends at {starts[-1]} but query has {query.shape[0]} rows")

    block_size, num_kv_heads, head_dim = key_pages.shape[1:]
    outputs = []

    for index in range(num_sequences):
        start, end = starts[index], starts[index + 1]
        query_len = end - start
        context_len = lengths[index]
        if context_len < query_len:
            raise ValueError(
                f"sequence {index} attends over {context_len} tokens but computes "
                f"{query_len}; S >= L is causality, not convention"
            )

        blocks_used = -(-context_len // block_size)
        table = tables[index][:blocks_used]
        if len(table) < blocks_used or min(table, default=0) < 0:
            raise ValueError(
                f"sequence {index} needs {blocks_used} blocks for {context_len} tokens "
                f"but its table does not cover them: {tables[index]}"
            )

        # Gather on the block axis, then flatten block and offset into one token axis.
        ids = mx.array(table, dtype=mx.int32)
        keys = key_pages[ids].reshape(-1, num_kv_heads, head_dim)[:context_len]
        values = value_pages[ids].reshape(-1, num_kv_heads, head_dim)[:context_len]

        # Dequantize only the pages this sequence reads, never the whole pool.
        if is_fp8:
            keys = dequantize_fp8(keys, k_scale, query.dtype)
            values = dequantize_fp8(values, v_scale, query.dtype)

        attended = scaled_dot_product_attention_grouped(
            query[start:end].swapaxes(0, 1),  # H_q x L x D
            keys.swapaxes(0, 1),              # H_k x S x D
            values.swapaxes(0, 1),
            scale=scale,
            mask="causal",
        )
        outputs.append(attended.swapaxes(0, 1))

    return mx.concatenate(outputs, axis=0)
