"""Step 4.7 — the block manager: capacity, growth, sharing, and release.

The layer that turns [Step 4.5]'s integers and [Step 4.6]'s arithmetic into something
a scheduler can talk to ([§6.4](../DESIGN.md#64-block-manager-api)):

::

    can_allocate(seq, n)   may this sequence take n more tokens?  (admission control)
    allocate(seq)          reserve blocks for everything uncomputed
    append_slot(seq)       grow by one token, adding a block only if needed
    fork(parent, child)    share the parent's blocks, copying nothing
    free(seq)              drop a holder on every block it owns

It owns the block pool, the KV pool tensors, and — through the sequences themselves —
the block tables. It has to own all three together because copy-on-write is the one
operation that touches all of them at once: a refcount, a table entry, and a page of
actual keys and values ([§6.3](../DESIGN.md#63-copy-on-write-sharing)).

Copy-on-write is worth being precise about, because it is easy to implement as
"copy the shared blocks" and that would be wrong in both directions. Only the **last
partial block** is ever copied, and only when it is shared. Earlier blocks of a shared
prefix are full and immutable — nobody will ever write into them again — so they stay
shared for the lifetime of both sequences. The partial one is the only page two
sequences would both write into, and the copy costs `P` tokens of K and V, not a
prefix of any length.
"""

from __future__ import annotations

import torch

from mini_vllm.block.block_pool import BlockPool, OutOfBlocks
from mini_vllm.block.block_table import BlockTable
from mini_vllm.block.kv_pool import PagedKvPool
from mini_vllm.serve.sequence import Sequence

# `OutOfBlocks` is re-exported because the scheduler catches it around this API and
# should not have to reach past it into the pool to name the exception.
__all__ = ["BlockManager", "OutOfBlocks"]


class BlockManager:
    """Allocates pages to sequences, and shares them when it can.

    Sizing is by block count rather than by bytes: how many pages fit in the memory
    budget is a question for the engine, which knows the model and the device, and
    `PagedKvPool.bytes_for` is what answers it.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        num_layers: int = 1,
        num_kv_heads: int = 1,
        head_dim: int = 1,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.block_size = block_size
        self.pool = BlockPool(num_blocks)
        self.kv = PagedKvPool(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

    # -------------------------------------------------------------- introspection

    @property
    def num_free_blocks(self) -> int:
        return self.pool.num_free

    @property
    def num_blocks(self) -> int:
        return self.pool.num_blocks

    def table(self, sequence: Sequence) -> BlockTable:
        """The sequence's table, raising if it was never allocated.

        A missing table means the scheduler ran a sequence the manager has not seen,
        which would otherwise show up as attention over an empty cache — fluent
        output with no relationship to the prompt.
        """
        if sequence.block_table is None:
            raise ValueError(f"sequence {sequence.seq_id} has no block table; allocate it first")
        return sequence.block_table

    def has_table(self, sequence: Sequence) -> bool:
        return sequence.block_table is not None

    # ---------------------------------------------------------- admission control

    def blocks_needed(self, sequence: Sequence, num_tokens: int) -> int:
        """Fresh blocks required to hold `num_tokens` more of this sequence.

        Includes the copy-on-write block: a shared partial page has to be duplicated
        before it can be written, so it costs one allocation that a private page does
        not. Leaving it out is how admission control passes and the write then fails
        mid-iteration, with half the batch already committed.
        """
        table = sequence.block_table
        if table is None:
            return -(-num_tokens // self.block_size)  # ceiling division

        needed = table.blocks_needed_for(num_tokens)
        if num_tokens and self._needs_copy(table):
            needed += 1
        return needed

    def can_allocate(self, sequence: Sequence, num_tokens: int | None = None) -> bool:
        """Whether `num_tokens` more tokens of this sequence would fit.

        Defaults to everything the sequence has left to compute, which is the
        admission question for a new request. The scheduler asks with a chunk size
        instead when chunking is on, and with 1 before a decode step
        ([§7.4](../DESIGN.md#74-admission-control)).
        """
        if num_tokens is None:
            num_tokens = sequence.num_uncomputed_tokens
        return self.blocks_needed(sequence, num_tokens) <= self.pool.num_free

    # ------------------------------------------------------------------- lifetime

    def allocate(self, sequence: Sequence, num_tokens: int | None = None) -> None:
        """Reserve capacity for `num_tokens` of this sequence, creating its table.

        Idempotent in the useful sense: calling it again for a sequence that already
        has a table extends that table rather than replacing it, which is what a
        chunked prefill does on every iteration.
        """
        if num_tokens is None:
            num_tokens = sequence.num_uncomputed_tokens
        if num_tokens < 0:
            raise ValueError(f"cannot reserve {num_tokens} tokens")

        if sequence.block_table is None:
            sequence.block_table = BlockTable(self.block_size)

        table = sequence.block_table
        if num_tokens:
            self._resolve_copy_on_write(sequence)
        for block_id in self.pool.allocate_many(table.blocks_needed_for(num_tokens)):
            table.append_block(block_id)
        table.append_tokens(num_tokens)

    def append_slot(self, sequence: Sequence) -> None:
        """Grow by exactly one token — the decode step.

        Usually free: the last block has room, so this is one increment. It costs an
        allocation once every `P` tokens, and a copy the first time a forked sequence
        writes into a shared page.
        """
        self.allocate(sequence, 1)

    def fork(self, parent: Sequence, child: Sequence) -> None:
        """Point `child` at every one of `parent`'s blocks, copying nothing.

        The refcount bump is the whole operation. A 4000-token prompt shared by eight
        samples costs 250 blocks instead of 2000, and the eight branches only start
        diverging in physical memory when one of them writes.
        """
        table = self.table(parent)
        if child.block_table is not None:
            raise ValueError(f"sequence {child.seq_id} already has a block table")

        for block_id in table.block_ids:
            self.pool.incref(block_id)
        child.block_table = table.copy()

    def free(self, sequence: Sequence) -> int:
        """Drop a holder on every block the sequence owns.

        Returns how many blocks that actually released — fewer than it holds when a
        fork is still alive. Safe to call twice: a finished request being freed again
        is a scheduler retry, not a corruption, and double-freeing the *pool* would be.
        """
        table = sequence.block_table
        if table is None:
            return 0

        sequence.block_table = None
        return self.pool.decref_many(list(table.block_ids))

    # -------------------------------------------------------------- copy-on-write

    def _needs_copy(self, table: BlockTable) -> bool:
        """Whether the next write into this table would land on a shared page.

        Only the last block can be written to, and only a partial block is written to
        at all — a full one is finished forever. So this is the only page
        copy-on-write ever has to consider.
        """
        if not table.num_blocks or table.num_empty_slots == 0:
            return False
        return self.pool.ref_count(table.block_ids[-1]) > 1

    def _resolve_copy_on_write(self, sequence: Sequence) -> int | None:
        """Give the sequence a private copy of its last page, if it is sharing one.

        Returns the new block id, or None when nothing had to be copied. The order
        matters: allocate, copy, repoint, *then* decref. Decrefing first could return
        the source page to the free list and hand it to another sequence before its
        contents were read.
        """
        table = self.table(sequence)
        if not self._needs_copy(table):
            return None

        fresh = self.pool.allocate()
        source = table.block_ids[-1]
        self.kv.copy_block(source, fresh)
        displaced = table.replace_block(table.num_blocks - 1, fresh)
        self.pool.decref(displaced)
        return fresh

    # ------------------------------------------------------------- the GPU handoff

    def slots(self, sequence: Sequence, num_tokens: int) -> list[int]:
        """Where this iteration's `num_tokens` tokens are written, as integers.

        The last `num_tokens` positions of the sequence, because `allocate` has
        already been called for them: occupancy runs ahead of computation by exactly
        this iteration's worth.
        """
        table = self.table(sequence)
        start = table.num_tokens - num_tokens
        if start < 0:
            raise ValueError(
                f"sequence {sequence.seq_id} holds {table.num_tokens} tokens; "
                f"cannot map {num_tokens}"
            )

        slots = table.slots(range(start, table.num_tokens))
        # Bounded here, while they are still integers. `PagedKvPool.write` used to do it
        # and it cost a device read per layer per iteration; the arithmetic that produces
        # a slot is the same either way, so the only question is whether it is checked on
        # the host or on the critical path.
        for slot in slots:
            if not 0 <= slot < self.kv.num_slots:
                raise ValueError(
                    f"sequence {sequence.seq_id} maps to slot {slot}, outside a pool of "
                    f"{self.kv.num_slots}"
                )
        return slots

    def slot_mapping(self, sequence: Sequence, num_tokens: int) -> torch.Tensor:
        """`slots`, as the int32 tensor the pool's write path scatters through."""
        return torch.tensor(
            self.slots(sequence, num_tokens), dtype=torch.int32, device=self.kv.device
        )

    def check_no_leaks(self) -> None:
        """Assert every block is back in the pool. For the end of a test or a run."""
        self.pool.check_consistency()
        if self.pool.num_free != self.pool.num_blocks:
            raise AssertionError(
                f"{self.pool.num_allocated} of {self.pool.num_blocks} blocks leaked: "
                f"{self.pool.allocated_ids()}"
            )

    def __repr__(self) -> str:
        return (
            f"BlockManager(blocks={self.pool.num_blocks}, free={self.pool.num_free}, "
            f"block_size={self.block_size})"
        )
