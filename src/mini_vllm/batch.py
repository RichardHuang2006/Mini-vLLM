"""The ragged forward batch: every scheduled token flattened onto one axis, with the
metadata the paged cache writes and attends through."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from mini_vllm.paged_kv_cache import BlockManager
    from mini_vllm.scheduler import Request

__all__ = ["PADDING_BLOCK", "ForwardBatch"]

# What an unused block-table entry holds: an impossible id, not a silent alias for block 0.
PADDING_BLOCK = -1


@dataclass(frozen=True)
class ForwardBatch:
    """For [prefill(A, 300), decode(B), decode(C)]: 302 tokens, cu_seqlens_q
    [0, 300, 301, 302], and context_lens [300, 512, 47]. Flattened rather than padded,
    in the paged kernel's own layout."""

    input_ids: mx.array     # int32 [T]
    positions: mx.array     # int32 [T], each sequence's own RoPE positions
    cu_seqlens_q: mx.array  # int32 [N + 1], where each sequence's tokens start
    context_lens: mx.array  # int32 [N], S: how many tokens each sequence attends over
    slot_mapping: mx.array  # int32 [T], the pool slot each token's K/V is written to
    block_tables: mx.array  # int32 [N, max_blocks], padded with PADDING_BLOCK

    @property
    def last_rows(self) -> mx.array:
        """Each sequence's last token row: the one whose logits are sampled."""
        return self.cu_seqlens_q[1:] - 1

    @classmethod
    def from_scheduled(cls, scheduled: list[tuple[Request, int]], manager: BlockManager) -> ForwardBatch:
        """Build from (request, tokens to compute now) pairs whose slots are allocated.

        Each request's tokens start at its num_computed_tokens, so a chunk's positions
        resume exactly where the previous chunk stopped.
        """
        input_ids, positions, offsets, context_lens, slots, tables = [], [], [0], [], [], []
        for request, count in scheduled:
            start = request.num_computed_tokens
            input_ids.extend(request.token_ids[start : start + count])
            positions.extend(range(start, start + count))
            offsets.append(offsets[-1] + count)
            context_lens.append(start + count)
            slots.extend(manager.slots(request, count))
            tables.append(request.block_table.block_ids)

        width = max(len(table) for table in tables)
        padded = [table + [PADDING_BLOCK] * (width - len(table)) for table in tables]
        return cls(
            input_ids=mx.array(input_ids, dtype=mx.int32),
            positions=mx.array(positions, dtype=mx.int32),
            cu_seqlens_q=mx.array(offsets, dtype=mx.int32),
            context_lens=mx.array(context_lens, dtype=mx.int32),
            slot_mapping=mx.array(slots, dtype=mx.int32),
            block_tables=mx.array(padded, dtype=mx.int32),
        )
