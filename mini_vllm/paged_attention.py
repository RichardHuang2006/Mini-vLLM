"""Paged attention, written to be obviously correct rather than fast.

The oracle for the paged attention kernels. For each sequence it walks the block table,
copies that sequence's keys and values out of the pool into a contiguous tensor, and
calls the reference attention on it.

That is a full copy of the cache every iteration, the traffic paging exists to avoid and
worse than the dense cache it replaces. It is also correct by construction: with
`scaled_dot_product_attention_grouped` as the reference, a correct gather gives a correct
answer, which is what the kernels are diffed against.

::

    q:            T x H_q x D        every scheduled token, sequences concatenated
    block_tables: int32 N x max_blocks   -1 padded
    cu_seqlens_q: int32 N + 1        where each sequence's queries start
    context_lens: int32 N            how many cached tokens each attends over
    out:          T x H_q x D

The flattened token axis is what lets one call serve a mixed batch: a 300-token prefill
chunk followed by a dozen single-token decodes is 312 rows here, distinguished only by
`cu_seqlens_q`.
"""

from __future__ import annotations

import torch

from mini_vllm.attention import scaled_dot_product_attention_grouped

__all__ = ["paged_attention_gathered"]


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
    """Grouped-query causal attention over a paged cache, one sequence at a time.

    `key_pool` and `value_pool` are one layer's pages, `num_blocks x P x H_k x D`. Each
    sequence is masked causally with its own `(L, S)` offset: a decode step's single query
    sees the whole context, and a prefill chunk's queries are the last `L` positions of
    `S`, so their diagonal is shifted right by `S - L`.

    With FP8 pools the gather also dequantizes, casting each cached key and value up and
    multiplying by its scale before the math, so the oracle attends in the activation
    dtype exactly as the kernel does after dequantizing in registers. The scales default
    to 1.0, the identity for an unquantized pool.
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

        # The gather: `index_select` on the block axis, then flatten block and offset
        # back into one logical axis, which works because the two are adjacent. The
        # kernel does the same arithmetic per element instead of per block.
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
