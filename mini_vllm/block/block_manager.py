"""The block manager: capacity, growth, sharing, and release.

The layer that turns the block pool's integers and the block table's arithmetic into
something a scheduler can talk to:

::

    can_allocate(seq, n)   may this sequence take n more tokens?  (admission control)
    allocate(seq)          reserve blocks for everything uncomputed
    append_slot(seq)       grow by one token, adding a block only if needed
    fork(parent, child)    share the parent's blocks, copying nothing
    free(seq)              drop a holder on every block it owns

It owns the block pool, the KV pool tensors, and — through the sequences themselves —
the block tables. All three together, because copy-on-write is the one operation that
touches a refcount, a table entry, and a page of keys and values at once.

Copy-on-write copies only the last partial block, and only when that block is shared.
Earlier blocks of a shared prefix are full and immutable, so they stay shared for the
lifetime of both sequences; the partial block is the only page two sequences would both
write into. The copy therefore costs `P` tokens of K and V rather than a prefix of
arbitrary length.
"""

from __future__ import annotations

import torch

from mini_vllm.block.block_pool import BlockPool, OutOfBlocks
from mini_vllm.block.block_table import BlockTable
from mini_vllm.block.kv_pool import PagedKvPool
from mini_vllm.block.prefix_cache import PrefixCache
from mini_vllm.serve.sequence import Sequence

# Re-exported: the scheduler catches `OutOfBlocks` around this API and should not have
# to reach past it into the pool to name the exception.
__all__ = ["BlockManager", "OutOfBlocks"]


class BlockManager:
    """Allocates pages to sequences, and shares them when it can.

    Sized by block count rather than by bytes: how many pages fit in a memory budget is
    the engine's question, since it knows the model and the device, and
    `PagedKvPool.bytes_for` answers it.
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
        enable_prefix_caching: bool = False,
        kv_dtype: torch.dtype | None = None,
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
            kv_dtype=kv_dtype,
        )

        # Prefix caching, off by default. When on, the pool calls back into the tree as
        # it reclaims a cached page, and `cached_tokens` counts tokens served from a hit
        # instead of recomputed.
        self.cache: PrefixCache | None = None
        self.cached_tokens = 0
        # Fresh pages taken from the free list over this manager's life. A prefix hit
        # reuses pages rather than taking new ones, so the saving shows up here and not
        # in the free count: a reused page still leaves the free list, it is just not a
        # newly reserved one.
        self.blocks_allocated = 0
        if enable_prefix_caching:
            self.cache = PrefixCache(block_size)
            self.pool.on_evict = self.cache.evict

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
        which would otherwise surface as attention over an empty cache: fluent output
        unrelated to the prompt.
        """
        if sequence.block_table is None:
            raise ValueError(f"sequence {sequence.seq_id} has no block table; allocate it first")
        return sequence.block_table

    def has_table(self, sequence: Sequence) -> bool:
        return sequence.block_table is not None

    # ---------------------------------------------------------- admission control

    def blocks_needed(self, sequence: Sequence, num_tokens: int) -> int:
        """Fresh blocks required to hold `num_tokens` more of this sequence.

        Includes the copy-on-write block: a shared partial page must be duplicated before
        it can be written, costing one allocation a private page does not. Omitting it
        lets admission control pass and the write fail mid-iteration, with half the batch
        already committed.
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

        Defaults to everything the sequence has left to compute, the admission question
        for a new request. The scheduler passes a chunk size when chunking is on, and 1
        before a decode step.
        """
        if num_tokens is None:
            num_tokens = sequence.num_uncomputed_tokens
        return self.blocks_needed(sequence, num_tokens) <= self.pool.num_free

    # ------------------------------------------------------------------- lifetime

    def allocate(self, sequence: Sequence, num_tokens: int | None = None) -> None:
        """Reserve capacity for `num_tokens` of this sequence, creating its table.

        Calling it again for a sequence that already has a table extends that table
        rather than replacing it, which is what a chunked prefill does every iteration.
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
        fresh = self.pool.allocate_many(table.blocks_needed_for(num_tokens))
        self.blocks_allocated += len(fresh)
        for block_id in fresh:
            table.append_block(block_id)
        table.append_tokens(num_tokens)

    def append_slot(self, sequence: Sequence) -> None:
        """Grow by exactly one token: the decode step.

        Usually one increment, since the last block has room. It costs an allocation
        once every `P` tokens, and a copy the first time a forked sequence writes into a
        shared page.
        """
        self.allocate(sequence, 1)

    def maybe_apply_prefix_cache(self, sequence: Sequence) -> int:
        """Match this sequence's prompt against the cache and reuse what hits.

        Called once, when a sequence is first admitted and before the scheduler sizes its
        prefill, so a hit is reflected in `num_computed_tokens` and the engine forwards
        only the part the cache did not hold. Returns the number of tokens reused, zero
        when caching is off or nothing matched. A sequence that already has a table is
        left alone, so a second call — or a chunked prefill's later iterations — is a
        no-op.

        The match never covers the whole sequence: a request hitting on every block would
        have nothing left to forward and no logits to sample from, so at least one block
        is always left to recompute.
        """
        if self.cache is None or sequence.block_table is not None:
            return 0

        sequence.block_table = BlockTable(self.block_size)
        if sequence.num_computed_tokens != 0 or sequence.num_output_tokens != 0:
            return 0  # only a fresh prompt is a prefix worth matching

        matched = self.cache.match(sequence.token_ids)
        while matched and len(matched) * self.block_size >= len(sequence):
            matched.pop()
        if not matched:
            return 0

        table = sequence.block_table
        for block_id in matched:
            self.pool.acquire_cached(block_id)
            table.append_block(block_id)
        reused = len(matched) * self.block_size
        table.append_tokens(reused)
        sequence.num_computed_tokens = reused
        self.cached_tokens += reused
        return reused

    def _cache_full_blocks(self, sequence: Sequence) -> None:
        """Register a sequence's completed full blocks in the prefix tree.

        Called from `free`, before the blocks are decref'd back onto the pool. Only whole
        computed blocks are cached: `table.num_tokens` is occupancy, so flooring it to a
        block boundary gives exactly the immutable prefix, and the tokens are read from
        the sequence in the order the blocks hold their KV.
        """
        if self.cache is None:
            return
        table = sequence.block_table
        if table is None:
            return
        num_full = table.num_tokens // self.block_size
        if num_full == 0:
            return
        block_ids = list(table.block_ids[:num_full])
        tokens = sequence.token_ids[: num_full * self.block_size]
        for block_id in self.cache.insert(tokens, block_ids):
            self.pool.mark_cached(block_id)

    def fork(self, parent: Sequence, child: Sequence) -> None:
        """Point `child` at every one of `parent`'s blocks, copying nothing.

        The refcount bump is the whole operation. A 4000-token prompt shared by eight
        samples costs 250 blocks instead of 2000, and the branches diverge in physical
        memory only when one of them writes.
        """
        table = self.table(parent)
        if child.block_table is not None:
            raise ValueError(f"sequence {child.seq_id} already has a block table")

        for block_id in table.block_ids:
            self.pool.incref(block_id)
        child.block_table = table.copy()

    def trim(self, sequence: Sequence, num_tokens: int) -> int:
        """Give back the last `num_tokens` slots, releasing any pages they emptied.

        The rollback half of speculative decoding. A speculative step writes `k` proposed
        tokens into the cache before the target has verified them, because the verifying
        forward pass must attend over them. When the target rejects a tail, that tail's
        slots go back, along with any fresh page the proposals spilled into.

        Returns the number of blocks actually released to the pool: trimming to a block
        boundary releases pages, trimming within the last partial block is a decrement
        and releases none.
        """
        if num_tokens < 0:
            raise ValueError(f"cannot trim {num_tokens} tokens")
        if num_tokens == 0:
            return 0

        table = self.table(sequence)
        emptied = table.trim_tokens(num_tokens)

        released = 0
        for _ in range(emptied):
            block_id = table.drop_last_block()
            released += int(self.pool.decref(block_id))
        return released

    def free(self, sequence: Sequence) -> int:
        """Drop a holder on every block the sequence owns.

        Returns how many blocks were actually released, fewer than it holds when a fork
        is still alive. Safe to call twice: freeing a finished request again is a
        scheduler retry, whereas double-decrefing the pool would be corruption.
        """
        table = sequence.block_table
        if table is None:
            return 0

        # Cache the immutable prefix before releasing the blocks. The pages then sit on
        # the free list at reference count zero, still referenced by the tree, and are
        # reclaimed only when the pool runs short.
        self._cache_full_blocks(sequence)

        sequence.block_table = None
        return self.pool.decref_many(list(table.block_ids))

    # -------------------------------------------------------------- copy-on-write

    def _needs_copy(self, table: BlockTable) -> bool:
        """Whether the next write into this table would land on a shared page.

        Only the last block is ever written to, and only while it is partial, since a
        full block is immutable. That makes it the only page copy-on-write considers.
        """
        if not table.num_blocks or table.num_empty_slots == 0:
            return False
        return self.pool.ref_count(table.block_ids[-1]) > 1

    def _resolve_copy_on_write(self, sequence: Sequence) -> int | None:
        """Give the sequence a private copy of its last page, if it is sharing one.

        Returns the new block id, or None when nothing had to be copied. Ordering is
        load-bearing: allocate, copy, repoint, then decref. Decrefing first could return
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

        The last `num_tokens` positions of the sequence, since `allocate` has already run
        for them: occupancy leads computation by exactly this iteration's worth.
        """
        table = self.table(sequence)
        start = table.num_tokens - num_tokens
        if start < 0:
            raise ValueError(
                f"sequence {sequence.seq_id} holds {table.num_tokens} tokens; "
                f"cannot map {num_tokens}"
            )

        slots = table.slots(range(start, table.num_tokens))
        # Bounds-checked here, while the slots are still integers. Doing it in
        # `PagedKvPool.write` cost a device read per layer per iteration for the same
        # arithmetic; the only difference is host versus critical path.
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
