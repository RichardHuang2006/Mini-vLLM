"""The KV-cache memory hierarchy: from a dense cache to paged, shared, quantized
storage.

What this file teaches
    How an inference engine stores keys and values, built up in six layers:

    1. `DenseKvCache` — the obvious cache, and why it fragments memory.
    2. `BlockPool` — physical pages as reference-counted integers.
    3. `BlockTable` — one sequence's logical-to-physical mapping.
    4. `PagedKvPool` — the pre-allocated tensors the block ids index into,
       optionally stored in FP8.
    5. `PrefixCache` — a radix tree that lets a new prompt reuse the pages of
       a prompt already served.
    6. `BlockManager` — allocation, copy-on-write, trimming, and release,
       tying the previous four together for the scheduler.

Inputs and outputs
    The pool and table speak integers; `PagedKvPool` speaks tensors; the
    manager speaks `Sequence` objects (defined in `scheduler.py`) and hands the
    model slot mappings and block tables.

Read next
    `scheduler.py` — the policy that decides which sequences get pages.

One invariant
    Every physical block is accounted for at all times: it is either on the
    free list at reference count zero, or held by one or more block tables at
    a positive count. `BlockManager.check_no_leaks` asserts this, and the test
    suite calls it after every scenario — a leaked page surfaces as an
    out-of-memory thousands of requests later, so it is checked eagerly.

Five words that must not blur together
    *token position*  — an index into one sequence (0, 1, 2, ...).
    *logical block*   — position // block_size: which entry of the sequence's
                        block table covers it.
    *physical block*  — the pool page that entry names; any page, any order.
    *slot offset*     — position % block_size: where inside the page it sits.
    *reference count* — how many block tables currently name a physical block;
                        sharing is a count above one, freeing is a count of zero.

::

    logical position p ─▶ block = block_ids[p // P],  slot = block · P + (p % P)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, NamedTuple

import torch

if TYPE_CHECKING:
    # `cache.py` sits below `scheduler.py` in the module graph, but the manager's API
    # takes the scheduler's `Sequence`. At runtime only attributes are read
    # (`block_table`, `token_ids`, `num_computed_tokens`), so the import is for type
    # checkers only and the graph stays acyclic.
    from mini_vllm.scheduler import Sequence

__all__ = [
    "KvCache",
    "DenseKvCache",
    "Block",
    "BlockPool",
    "BlockPoolError",
    "OutOfBlocks",
    "BlockTable",
    "PagedKvPool",
    "RadixNode",
    "PrefixCache",
    "BlockManager",
]


# ----------------------------------------------------------- 1. dense KV cache


class KvCache(ABC):
    """One layer's worth of cached keys and values.

    Attention at position `t` needs the keys and values of every position `0..t`, and
    those do not change once computed. An uncached model recomputes all of them every
    step, making decode quadratic in the sequence length; caching them makes it linear.

    One cache per layer, not one per model: each layer's keys and values are
    independent, and the model holds a list of them. The interface is one method, so
    the paged implementation is indistinguishable to the model.
    """

    @property
    @abstractmethod
    def offset(self) -> int:
        """How many positions are currently cached."""

    @abstractmethod
    def update_and_fetch(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Append ``key``/``value`` and return everything cached, plus the write offset.

        ::

            key, value (incoming):  B x H_k x L x D
            returns full_key/value: B x H_k x S x D     where S = offset + L
            returns offset:         the length *before* this call

        The returned offset is the position this update was written at, not the length
        afterwards, so ``S == offset + L`` and ``self.offset`` afterwards equals ``S``.
        The caller builds a causal mask from it, and being off by ``L`` there lets a
        token attend to its own future.

        Keys must already have RoPE applied: positions are baked into the cached
        tensors, which is what makes a cache entry reusable.
        """

    @abstractmethod
    def reset(self) -> None:
        """Forget everything, so the cache can serve a new sequence."""


class DenseKvCache(KvCache):
    """A cache that simply concatenates along the sequence dimension.

    The obvious implementation, with two costs the rest of this file exists to remove:

    * Every decode step reallocates. ``torch.cat`` cannot extend a tensor in place, so
      appending one token to a cache of `S` copies all `S` positions into a new buffer:
      quadratic memory traffic to store linear data.
    * One contiguous allocation per sequence. A batch of differing lengths must be padded
      to the longest, and a sequence that might reach 40960 tokens has to be budgeted as
      if it will — the fragmentation PagedAttention removes.

    It is correct, which makes it the oracle for the paged version.
    """

    def __init__(self) -> None:
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self._offset = 0

    @property
    def offset(self) -> int:
        return self._offset

    def update_and_fetch(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                f"expected B x H_k x L x D keys and values, got {tuple(key.shape)} "
                f"and {tuple(value.shape)}"
            )
        if key.shape != value.shape:
            raise ValueError(
                f"key and value must have the same shape, got {tuple(key.shape)} "
                f"and {tuple(value.shape)}"
            )

        written_at = self._offset

        if self.keys is None:
            self.keys, self.values = key, value
        else:
            if key.shape[:2] != self.keys.shape[:2] or key.shape[3] != self.keys.shape[3]:
                raise ValueError(
                    f"cannot append {tuple(key.shape)} to a cache of "
                    f"{tuple(self.keys.shape)}: only the sequence dimension may differ"
                )
            self.keys = torch.cat([self.keys, key], dim=-2)
            self.values = torch.cat([self.values, value], dim=-2)

        self._offset += key.shape[-2]
        return self.keys, self.values, written_at

    def reset(self) -> None:
        self.keys = None
        self.values = None
        self._offset = 0


# ------------------------------------------------------- 2. physical block pool


class BlockPoolError(RuntimeError):
    """Base for block pool misuse. These indicate bugs, not conditions."""


class OutOfBlocks(BlockPoolError):
    """The pool is exhausted.

    Unlike the other errors here this is an expected runtime condition rather than a
    bug: it signals to preempt or to leave a request waiting. It inherits from
    `BlockPoolError` so a caller can catch everything from the pool at once, but the
    scheduler catches it on its own.
    """


class Block(NamedTuple):
    """A read-only view of one block's state, for tests and debugging.

    The pool does not store these; it stores a flat list of counts, since a pool holds
    tens of thousands of blocks and an object per block would be pure overhead. One of
    these is materialized on request.
    """

    block_id: int
    ref_count: int

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0


class BlockPool:
    """A fixed pool of physical blocks, handed out by id and refcounted.

    The bottom of the paged memory hierarchy. The pool knows nothing about sequences,
    tokens, or tensors: it hands out integer ids and counts holders. Block tables,
    copy-on-write and admission control are all built on those two operations, none of
    which needs a GPU to test.

    ::

        allocate() -> id        take a free block, at ref_count 1
        incref(id)              add a holder (fork, prefix reuse)
        decref(id) -> bool      drop a holder; True if that freed the block

    Reference counting is what allows sharing without copying: forking a sequence or
    reusing a cached prefix increments counts instead of duplicating a page of K and V,
    and a block returns to circulation when its last holder releases it.

    Two non-obvious choices:

    * The free list is FIFO rather than LIFO. Recycling the most recently freed block
      has better cache locality, but it also means a use-after-free usually reads back
      the data it just released and appears correct. Cycling the whole pool makes that
      class of bug fail loudly, which is worth more than the locality.
    * Sized once at startup: paging reserves GPU memory up front and partitions it, so
      growing the pool later would mean a cudaMalloc inside a decode step.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        self._num_blocks = num_blocks
        self._ref_counts = [0] * num_blocks
        self._free: deque[int] = deque(range(num_blocks))

        # Prefix caching, inert unless a manager wires it up. `_cached[id]` is True while
        # a block is registered in the prefix tree, free or held, since a held block
        # stays matchable; `on_evict` is how the pool tells the tree to release a block
        # it is about to repurpose.
        self._cached = [False] * num_blocks
        self.on_evict: Callable[[int], None] | None = None

    # ---------------------------------------------------------------- inspection

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_allocated(self) -> int:
        return self._num_blocks - len(self._free)

    def ref_count(self, block_id: int) -> int:
        self._check_id(block_id)
        return self._ref_counts[block_id]

    def block(self, block_id: int) -> Block:
        return Block(block_id, self.ref_count(block_id))

    def allocated_ids(self) -> list[int]:
        """Ids currently held by someone. Sorted, so output is reproducible."""
        return [i for i, count in enumerate(self._ref_counts) if count]

    # ---------------------------------------------------------------- allocation

    def allocate(self) -> int:
        """Take one block from the free list at reference count 1.

        Raises:
            OutOfBlocks: if nothing is free. The caller should preempt a running
                sequence or stop admitting rather than retry.
        """
        if not self._free:
            raise OutOfBlocks(
                f"all {self._num_blocks} blocks are in use; "
                "the caller should preempt or stop admitting rather than retry"
            )

        block_id = self._free.popleft()
        # A block carrying cached KV is being repurposed: unlink it from the prefix tree
        # first, so a later match cannot return a page now holding another sequence's
        # tokens. FIFO over the free list is LRU over cache entries, so the free list
        # already supplies the right eviction order.
        if self._cached[block_id]:
            if self.on_evict is not None:
                self.on_evict(block_id)
            self._cached[block_id] = False
        self._ref_counts[block_id] = 1
        return block_id

    def allocate_many(self, count: int) -> list[int]:
        """Take `count` blocks, or none at all.

        All-or-nothing: a half-allocated sequence forces the caller to unwind, and the
        unwind path is where block leaks come from. Anything taken is returned before
        the exception propagates.
        """
        if count < 0:
            raise ValueError(f"cannot allocate {count} blocks")
        if count > self.num_free:
            raise OutOfBlocks(
                f"asked for {count} blocks with {self.num_free} free of {self._num_blocks}"
            )
        return [self.allocate() for _ in range(count)]

    def incref(self, block_id: int) -> int:
        """Add a holder. Used by fork and by prefix-cache hits."""
        self._check_id(block_id)
        if self._ref_counts[block_id] == 0:
            raise BlockPoolError(
                f"block {block_id} is free; it must be allocated before it can be shared"
            )

        self._ref_counts[block_id] += 1
        return self._ref_counts[block_id]

    def decref(self, block_id: int) -> bool:
        """Drop a holder, returning the block to the pool at zero.

        Returns:
            True if this call freed the block, False if holders remain. Copy-on-write
            needs the distinction: a write to a block with holders left must copy, a
            write to one just released need not.
        """
        self._check_id(block_id)
        if self._ref_counts[block_id] == 0:
            raise BlockPoolError(
                f"block {block_id} is already free (double free); some holder released it twice"
            )

        self._ref_counts[block_id] -= 1
        if self._ref_counts[block_id]:
            return False

        self._free.append(block_id)
        return True

    def decref_many(self, block_ids: list[int]) -> int:
        """Drop a holder on each, returning how many blocks that freed."""
        return sum(self.decref(block_id) for block_id in block_ids)

    # -------------------------------------------------------------- prefix cache

    def mark_cached(self, block_id: int) -> None:
        """Record that a block is now registered in the prefix tree.

        Called after the cache inserts a finishing sequence's full blocks. The reference
        count is unchanged: caching does not add a holder, it makes the page matchable
        while still reclaimable.
        """
        self._check_id(block_id)
        self._cached[block_id] = True

    def is_cached(self, block_id: int) -> bool:
        self._check_id(block_id)
        return self._cached[block_id]

    def acquire_cached(self, block_id: int) -> None:
        """Take a matched block for a new holder, from wherever it currently sits.

        A prefix-cache hit is not a fresh allocation: the block exists and already holds
        the right KV, so this adds a holder rather than taking a page from the free list.
        A cached block at reference count zero comes off the free list here; one already
        held by a running sequence is a plain incref. Either way it stays in the tree,
        matchable by the next request.
        """
        self._check_id(block_id)
        if self._ref_counts[block_id] == 0:
            # A cached free block: pull it off the free list by hand. It stays cached,
            # so a later reuse of the same page still evicts its node.
            self._free.remove(block_id)
            self._ref_counts[block_id] = 1
        else:
            self._ref_counts[block_id] += 1

    def is_free_cached(self, block_id: int) -> bool:
        """A cached block at reference count zero. Reusing it consumes a free page while
        matching a held one does not, a distinction admission control accounts for."""
        self._check_id(block_id)
        return self._cached[block_id] and self._ref_counts[block_id] == 0

    # --------------------------------------------------------------- consistency

    def check_consistency(self) -> None:
        """Assert the free list and the reference counts still agree.

        Cheap enough to call from test teardown, where it attributes a refcount bug to
        the test that introduced it rather than letting it surface as an out-of-memory
        thousands of iterations later.
        """
        free = list(self._free)

        if len(set(free)) != len(free):
            duplicates = sorted({i for i in free if free.count(i) > 1})
            raise BlockPoolError(f"blocks appear twice in the free list: {duplicates}")

        expected_free = {i for i, count in enumerate(self._ref_counts) if count == 0}
        if set(free) != expected_free:
            raise BlockPoolError(
                f"free list {sorted(free)} disagrees with the zero-refcount blocks "
                f"{sorted(expected_free)}"
            )

        negative = [i for i, count in enumerate(self._ref_counts) if count < 0]
        if negative:
            raise BlockPoolError(f"negative reference counts at blocks {negative}")

    def _check_id(self, block_id: int) -> None:
        if not 0 <= block_id < self._num_blocks:
            raise BlockPoolError(
                f"block id {block_id} out of range for a pool of {self._num_blocks}"
            )

    def __repr__(self) -> str:
        return (
            f"BlockPool(num_blocks={self._num_blocks}, "
            f"free={self.num_free}, allocated={self.num_allocated})"
        )


# -------------------------------------------------------- 3. logical block table


class BlockTable:
    """The physical block ids backing one sequence, in logical order.

    The whole paging indirection: a list of block ids, a shift, and a mask. Non-
    contiguous physical storage removes per-sequence contiguous reservation and its
    fragmentation at a cost of one integer division and one modulo per token, both of
    which reduce to bit operations because `P` is a power of two. The same arithmetic
    reappears inside the paged attention kernel.

    Two distinct sizes are tracked:

    * :attr:`num_slots` — capacity, always a multiple of `block_size`.
    * :attr:`num_tokens` — occupancy, usually not.

    Their difference is the internal fragmentation in the final block, bounded by `P - 1`
    tokens per sequence regardless of its length, and it tells the block manager whether
    appending a token needs a new block or fits in place.

    The one output that leaves Python is :meth:`slot_mapping`, the flat index vector the
    paged write path scatters new keys and values through.
    """

    def __init__(
        self,
        block_size: int,
        block_ids: Iterable[int] | None = None,
        num_tokens: int = 0,
    ) -> None:
        if block_size <= 0 or block_size & (block_size - 1):
            raise ValueError(f"block_size must be a positive power of two, got {block_size}")

        self._block_size = block_size
        # Power of two, so `p // block_size` is `p >> shift` and `p % block_size` is
        # `p & mask`. The only reason the block size is constrained.
        self._shift = block_size.bit_length() - 1
        self._mask = block_size - 1

        self._block_ids: list[int] = [] if block_ids is None else list(block_ids)
        for block_id in self._block_ids:
            if block_id < 0:
                raise ValueError(f"block ids must be non-negative, got {block_id}")

        if not 0 <= num_tokens <= self.num_slots:
            raise ValueError(
                f"num_tokens={num_tokens} does not fit in {len(self._block_ids)} "
                f"blocks of {block_size} ({self.num_slots} slots)"
            )
        self._num_tokens = num_tokens

    # ------------------------------------------------------------------ geometry

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def block_ids(self) -> tuple[int, ...]:
        """The table itself, logical order first. A copy, so callers cannot alias it."""
        return tuple(self._block_ids)

    @property
    def num_blocks(self) -> int:
        return len(self._block_ids)

    @property
    def num_slots(self) -> int:
        """Token capacity of this sequence.

        Distinct from the range of :meth:`physical_slot`, which indexes the pool's whole
        flat cache: a one-block table holding block id 900 has 16 slots but addresses
        slot 14400.
        """
        return len(self._block_ids) * self._block_size

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def num_empty_slots(self) -> int:
        """Room left in the final block. Zero when the last block is full."""
        return self.num_slots - self._num_tokens

    def blocks_needed_for(self, num_new_tokens: int) -> int:
        """How many fresh blocks appending `num_new_tokens` would require.

        The admission-control question as arithmetic. Space left in the partial block is
        used first, so this is often zero, which is why a decode step usually allocates
        nothing.
        """
        if num_new_tokens < 0:
            raise ValueError(f"num_new_tokens must be non-negative, got {num_new_tokens}")

        deficit = num_new_tokens - self.num_empty_slots
        if deficit <= 0:
            return 0
        return (deficit + self._mask) >> self._shift

    # -------------------------------------------------------------------- growth

    def append_block(self, block_id: int) -> None:
        """Extend capacity by one block. The id comes from the pool."""
        if block_id < 0:
            raise ValueError(f"block ids must be non-negative, got {block_id}")
        self._block_ids.append(block_id)

    def append_tokens(self, count: int = 1) -> None:
        """Mark `count` more slots occupied.

        Capacity first, occupancy second: this raises rather than growing the table,
        since only the block manager may take blocks from the pool. A self-allocating
        table would be a second place blocks can leak.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count > self.num_empty_slots:
            raise ValueError(
                f"cannot append {count} tokens: {self.num_empty_slots} free slots in "
                f"{self.num_blocks} blocks; append_block first"
            )
        self._num_tokens += count

    def trim_tokens(self, count: int = 1) -> int:
        """Give back `count` of the occupied slots, returning how many blocks fell empty.

        The mirror of :meth:`append_tokens`, with one caller: speculative decoding writes
        `k` proposed tokens' KV into the cache before knowing whether the target accepts
        them, and the rejected tail must be unwritten.

        Only occupancy moves. The blocks stay in the table and the count of wholly empty
        trailing blocks is reported rather than released, because only the block manager
        hands pages back to the pool; that keeps exactly one place a block can be freed.

        KV in the trimmed slots is left in place: nothing reads past `num_tokens`, and a
        later token written into a reclaimed slot overwrites it, so zeroing would only
        hide data that is already unreachable.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count > self._num_tokens:
            raise ValueError(f"cannot trim {count} tokens from a table holding {self._num_tokens}")

        blocks_before = self.num_blocks
        self._num_tokens -= count
        # Blocks still needed for what is left: the empty tail is everything past it.
        blocks_still_used = -(-self._num_tokens // self._block_size)
        return blocks_before - max(blocks_still_used, 0)

    def drop_last_block(self) -> int:
        """Remove the final block from the table and return its id, for the manager.

        Paired with :meth:`trim_tokens`, which reports how many trailing blocks emptied:
        the manager detaches them one at a time so it can decref each. Raises if the
        block still holds tokens, since dropping an occupied block would unmap KV the
        sequence is still attending over.
        """
        if not self._block_ids:
            raise IndexError("no blocks to drop")
        if self._num_tokens > (self.num_blocks - 1) * self._block_size:
            raise ValueError(
                f"the last block still holds tokens: {self._num_tokens} tokens in "
                f"{self.num_blocks} blocks of {self._block_size}"
            )
        return self._block_ids.pop()

    def replace_block(self, index: int, block_id: int) -> int:
        """Repoint one table entry, returning the id it displaced.

        The copy-on-write step. The displaced id is returned rather than dropped so the
        caller cannot forget to decref it; a forgotten decref surfaces only as an
        out-of-memory much later.
        """
        if not 0 <= index < len(self._block_ids):
            raise IndexError(f"block index {index} out of range for {len(self._block_ids)} blocks")
        if block_id < 0:
            raise ValueError(f"block ids must be non-negative, got {block_id}")

        displaced = self._block_ids[index]
        self._block_ids[index] = block_id
        return displaced

    def copy(self) -> BlockTable:
        """An independent table over the same blocks: the fork half of copy-on-write.

        No block data is copied and no refcounts are touched; the caller increfs. The
        independence is of the list, so appending to one branch cannot alter the other's
        mapping.
        """
        return BlockTable(self._block_size, self._block_ids, self._num_tokens)

    # ------------------------------------------------------------ the indirection

    def block_index(self, position: int) -> int:
        """Which entry of this table covers `position`."""
        self._check_position(position)
        return position >> self._shift

    def block_offset(self, position: int) -> int:
        """Where inside its block `position` sits."""
        self._check_position(position)
        return position & self._mask

    def physical_slot(self, position: int) -> int:
        """Flat index of a logical position in the pool's cache.

        ``block_id * block_size + offset``, written as a shift and an or because the
        offset is strictly narrower than the shift.
        """
        self._check_position(position)
        block_id = self._block_ids[position >> self._shift]
        return (block_id << self._shift) | (position & self._mask)

    def slots(self, positions: Iterable[int]) -> list[int]:
        """The physical slots of `positions`, as plain integers.

        The form the engine needs: it concatenates several sequences' slots into one
        vector per iteration, and per-sequence tensors would have to be read back off the
        device to do that — one synchronization per sequence per iteration, for numbers
        that were already Python integers.
        """
        return [self.physical_slot(position) for position in positions]

    def slot_mapping(
        self,
        positions: Iterable[int],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """The `int32` slot vector the paged write path scatters through.

        One entry per token written this iteration: the whole chunk for a prefill, a
        single element for a decode step. `int32` is what the kernel indexes with, and a
        pool large enough to overflow it would not fit in any current GPU.
        """
        return torch.tensor(self.slots(positions), dtype=torch.int32, device=device)

    def _check_position(self, position: int) -> None:
        # Checked against occupancy rather than capacity: a slot that exists but holds
        # no token is not addressable, so a scheduler off-by-one fails here instead of
        # reading whichever sequence used the slot last.
        if not 0 <= position < self._num_tokens:
            raise IndexError(
                f"position {position} out of range for a sequence of {self._num_tokens} tokens"
            )

    def __repr__(self) -> str:
        return (
            f"BlockTable(block_size={self._block_size}, block_ids={self._block_ids}, "
            f"num_tokens={self._num_tokens}/{self.num_slots})"
        )


# --------------------------------------------------------- 4. paged KV storage


class PagedKvPool:
    """Pre-allocated paged storage for every layer's keys and values.

    ::

        keys:   num_layers x num_blocks x P x H_k x D
        values: num_layers x num_blocks x P x H_k x D

    Allocated once at startup and never grown: a `cudaMalloc` in the middle of a decode
    step would stall every sequence in flight, and a pool that can be exhausted but not
    fragmented turns memory pressure into a scheduling problem.

    The layer axis is part of one tensor rather than a list of per-layer tensors, so the
    whole cache is a single allocation whose size is knowable up front. That is what
    makes :meth:`bytes_for` answerable, which is how an engine picks `num_blocks` from a
    memory budget.

    Two asymmetric operations:

    * :meth:`write` scatters this iteration's new keys and values into the slots the
      block tables name, one indexed copy per layer. Physical order is irrelevant.
    * :meth:`gather` collects one sequence's cache back into a contiguous tensor. This
      is the slow reference path the paged attention kernel is diffed against, not a
      serving path: doing it for real would copy the entire cache every iteration.
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        kv_dtype: torch.dtype | None = None,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> None:
        for name, value in (
            ("num_layers", num_layers),
            ("num_blocks", num_blocks),
            ("block_size", block_size),
            ("num_kv_heads", num_kv_heads),
            ("head_dim", head_dim),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if block_size & (block_size - 1):
            raise ValueError(f"block_size must be a power of two, got {block_size}")

        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # `dtype` is the activation dtype: what keys and values arrive as and what a
        # gather returns. `kv_dtype` is the storage dtype, identical unless the cache is
        # quantized. Separating them is what FP8 amounts to here: the model still
        # computes in bf16 and only the resident cache shrinks.
        self.dtype = dtype
        self.kv_dtype = kv_dtype or dtype
        self.is_fp8 = self.kv_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        # Static scales, one per tensor. A key or value is divided by its scale before
        # the cast down and multiplied back after, which is how e4m3's ±448 range is
        # made to cover activations outside it. Qwen3's post-norm, post-RoPE keys are
        # near unit scale, so 1.0 is a safe default; the hook exists for models whose
        # are not.
        self.k_scale = float(k_scale)
        self.v_scale = float(v_scale)
        self.device = torch.device(device)

        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.keys = torch.zeros(shape, dtype=self.kv_dtype, device=self.device)
        self.values = torch.zeros(shape, dtype=self.kv_dtype, device=self.device)

    # ------------------------------------------------------------------- sizing

    @staticmethod
    def bytes_for(
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> int:
        """How much GPU memory a pool of this shape would take, keys and values.

        ``dtype`` is the storage dtype: pass ``torch.float8_e4m3fn`` to size an FP8 pool,
        where a page of the same geometry costs half as much and the engine fits twice
        as many.
        """
        elements = num_layers * num_blocks * block_size * num_kv_heads * head_dim
        return 2 * elements * torch.empty((), dtype=dtype).element_size()

    @property
    def num_slots(self) -> int:
        """Token slots per layer: the range `physical_slot` addresses."""
        return self.num_blocks * self.block_size

    # -------------------------------------------------------------------- access

    def layer_keys(self, layer: int) -> torch.Tensor:
        """One layer's key pool, `num_blocks x P x H_k x D`. A view, not a copy."""
        return self.keys[layer]

    def layer_values(self, layer: int) -> torch.Tensor:
        return self.values[layer]

    def flat(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer's pools flattened to `num_slots x H_k x D`.

        The layout `slot_mapping` indexes into. `block_id * P + offset` is a flat slot
        number because the block and offset axes are adjacent and contiguous, so this
        view is free.
        """
        shape = (self.num_slots, self.num_kv_heads, self.head_dim)
        return self.keys[layer].view(shape), self.values[layer].view(shape)

    # --------------------------------------------------------------------- write

    def write(
        self,
        layer: int,
        slot_mapping: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Scatter this iteration's keys and values into their slots.

        ::

            slot_mapping: int32 [T]     where each token goes, from the block tables
            key, value:   [T, H_k, D]   flattened across sequences

        `T` spans the whole ragged batch — a prefill chunk's tokens and a dozen decode
        steps' single tokens in one call — because the slots are already absolute. Nothing
        here needs to know which sequence a token belongs to, which is what makes one
        launch sufficient.

        Slot values are not checked here. This is the innermost call in the engine, 28
        layers per iteration, and `slot_mapping.max()` on a CUDA tensor is a
        device-to-host read that waits for everything queued behind it: two per layer
        measured 7 ms per iteration, a third of the model's time, to re-check integers
        `BlockManager.slots` already bounds-checked on the host. Shapes are checked
        because that is free.
        """
        self._check_layer(layer)
        if key.shape != value.shape:
            raise ValueError(
                f"key and value must match, got {tuple(key.shape)} and {tuple(value.shape)}"
            )
        expected = (slot_mapping.shape[0], self.num_kv_heads, self.head_dim)
        if tuple(key.shape) != expected:
            raise ValueError(f"expected keys shaped {expected}, got {tuple(key.shape)}")

        flat_keys, flat_values = self.flat(layer)
        if self.is_fp8:
            # The fused kernel quantizes and scatters in one pass; off the GPU it falls
            # back to the two-pass PyTorch that serves as its oracle. Either way the
            # trailing dimensions must match the pool's, which the reshape enforces.
            from mini_vllm import kernels

            kernels.quantize_scatter(
                key.reshape(-1, self.num_kv_heads, self.head_dim),
                value.reshape(-1, self.num_kv_heads, self.head_dim),
                flat_keys,
                flat_values,
                slot_mapping,
                self.k_scale,
                self.v_scale,
                use_cuda=self.device.type == "cuda",
            )
        else:
            index = slot_mapping.to(device=flat_keys.device, dtype=torch.int64)
            flat_keys.index_copy_(0, index, key.to(flat_keys.dtype))
            flat_values.index_copy_(0, index, value.to(flat_values.dtype))

    # -------------------------------------------------------------------- gather

    def gather(
        self,
        layer: int,
        block_ids: tuple[int, ...] | list[int],
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One sequence's cache, contiguous in logical order.

        ::

            returns: 1 x H_k x num_tokens x D, keys and values

        The shape the reference attention implementation takes, which is what makes a
        paged cache testable against a dense one. It is a copy — the gather paged kernels
        exist to avoid — and is present only as the oracle.
        """
        self._check_layer(layer)
        needed = -(-num_tokens // self.block_size)  # ceiling division
        if needed > len(block_ids):
            raise ValueError(
                f"{num_tokens} tokens need {needed} blocks, but the table holds {len(block_ids)}"
            )

        index = torch.tensor(list(block_ids[:needed]), dtype=torch.int64, device=self.device)
        gathered = []
        for pool, scale in ((self.keys[layer], self.k_scale), (self.values[layer], self.v_scale)):
            blocks = pool.index_select(0, index)  # needed x P x H_k x D
            flat = blocks.reshape(-1, self.num_kv_heads, self.head_dim)[:num_tokens]
            contiguous = flat.permute(1, 0, 2).unsqueeze(0).contiguous()
            if self.is_fp8:
                # Dequantize back to the activation dtype: the oracle attention runs in
                # the model's precision, so the cast and the scale are undone here rather
                # than leaving FP8 in the math.
                contiguous = (contiguous.float() * scale).to(self.dtype)
            gathered.append(contiguous)
        return gathered[0], gathered[1]

    # ---------------------------------------------------------------------- copy

    def copy_block(self, source: int, destination: int) -> None:
        """Duplicate one page across every layer: the copy in copy-on-write.

        Every layer at once, because a block id names the same page in all of them — the
        block table is per sequence, not per sequence and layer. Copying one layer's page
        and not the others would leave a sequence attending over a prefix that is correct
        in layer 0 and stale in layer 1.
        """
        for block_id in (source, destination):
            if not 0 <= block_id < self.num_blocks:
                raise ValueError(f"block {block_id} is outside a pool of {self.num_blocks}")
        if source == destination:
            return

        # Copy raw bytes: a uint8 view is dtype-agnostic and works for every storage
        # type this pool can hold, FP8 included.
        keys, values = self.keys.view(torch.uint8), self.values.view(torch.uint8)
        keys[:, destination].copy_(keys[:, source])
        values[:, destination].copy_(values[:, source])

    def _check_layer(self, layer: int) -> None:
        if not 0 <= layer < self.num_layers:
            raise ValueError(f"layer {layer} is outside a pool of {self.num_layers} layers")

    def __repr__(self) -> str:
        stored = f"{self.kv_dtype} <- {self.dtype}" if self.is_fp8 else f"{self.dtype}"
        return (
            f"PagedKvPool(layers={self.num_layers}, blocks={self.num_blocks}, "
            f"block_size={self.block_size}, heads={self.num_kv_heads}, dim={self.head_dim}, "
            f"{stored}, {self.device.type})"
        )


# ------------------------------------------------- 5. radix-tree prefix cache


class RadixNode:
    """One cached block, and the edge of tokens that reaches it.

    The root is the only node with ``block_id is None``: it spells the empty prefix and
    owns no page. Every other node stands for one physical block whose KV is the block's
    worth of tokens on the edge from its parent.
    """

    __slots__ = ("parent", "token_key", "block_id", "children")

    def __init__(
        self,
        parent: RadixNode | None = None,
        token_key: tuple[int, ...] | None = None,
        block_id: int | None = None,
    ) -> None:
        self.parent = parent
        self.token_key = token_key
        self.block_id = block_id
        self.children: dict[tuple[int, ...], RadixNode] = {}


class PrefixCache:
    """A radix tree over block-aligned token prefixes, mapping them to blocks.

    Two requests beginning with the same tokens compute the same keys and values for
    that prefix. Served workloads are full of this — a shared system prompt, a few-shot
    preamble, a conversation replayed with one more turn — and recomputing it is the
    largest single waste available to an inference engine.

    The structure is a radix tree whose edges are one block of tokens. A path from the
    root spells a token prefix in block-sized steps, and the node ending each edge names
    the physical block holding that block's KV::

        root
         ├─ (t0..t15)  -> block 7
         │                 └─ (t16..t31) -> block 12
         └─ (u0..u15)  -> block 3

    Matching a prompt walks the tree as far as its tokens agree, one block at a time,
    and the blocks along the matched path are shared rather than recomputed. Keys are
    exact token tuples, so unlike a hash cache there are no collisions: two prefixes
    share a node exactly when their tokens are identical.

    ::

        match(tokens)          -> physical blocks for the longest cached prefix
        insert(tokens, blocks) -> register full blocks, returns the newly cached ones
        evict(block_id)        -> unlink a block the pool is reclaiming

    The tree does not own memory, and it never touches reference counts. A cached block
    sits on the block pool's free list at reference count zero, still referenced by its
    node, and is reclaimed as soon as the pool needs it — :class:`BlockPool` calls
    :meth:`evict` as it hands the block out. Prefix caching therefore costs nothing when
    there are no hits: a cached block is a free block that remembers its contents until
    the page is needed.
    """

    def __init__(self, block_size: int) -> None:
        if block_size <= 0 or block_size & (block_size - 1):
            raise ValueError(f"block_size must be a positive power of two, got {block_size}")
        self.block_size = block_size
        self.root = RadixNode()
        # Reverse index so eviction is O(depth) rather than a tree walk: the pool names
        # a block id, and this finds the node standing for it.
        self._node_of_block: dict[int, RadixNode] = {}

    # ------------------------------------------------------------------- matching

    def match(self, token_ids: list[int]) -> list[int]:
        """The blocks of the longest cached prefix of ``token_ids``, in order.

        Only whole blocks match. A partial trailing block is never cached: a block still
        being written to is not immutable, so sharing it would be a data race. The caller
        increfs whatever comes back.
        """
        matched: list[int] = []
        node = self.root
        num_full = len(token_ids) // self.block_size
        for index in range(num_full):
            start = index * self.block_size
            key = tuple(token_ids[start : start + self.block_size])
            child = node.children.get(key)
            if child is None:
                break
            matched.append(child.block_id)  # type: ignore[arg-type]
            node = child
        return matched

    # ------------------------------------------------------------------ insertion

    def insert(self, token_ids: list[int], block_ids: list[int]) -> list[int]:
        """Register ``block_ids`` as the cache of ``token_ids``'s full blocks.

        Returns the newly cached block ids, which the pool must mark so it calls
        :meth:`evict` before reusing them. A block whose prefix another block already
        caches is left out and freed normally: two sequences that prefilled the same
        prompt without a hit between them each computed it into their own page, and the
        tree keeps the first.
        """
        node = self.root
        newly: list[int] = []
        for index, block_id in enumerate(block_ids):
            start = index * self.block_size
            key = tuple(token_ids[start : start + self.block_size])
            if len(key) < self.block_size:
                break  # a partial block is not cacheable
            child = node.children.get(key)
            if child is None:
                child = RadixNode(parent=node, token_key=key, block_id=block_id)
                node.children[key] = child
                self._node_of_block[block_id] = child
                newly.append(block_id)
            node = child
        return newly

    # ------------------------------------------------------------------- eviction

    def evict(self, block_id: int) -> None:
        """Unlink the block the pool is about to reuse, and orphan its subtree.

        Called by :class:`BlockPool` as it pops a cached block off the free list. The
        block's descendants become unreachable once it is gone, and since a held block
        also holds its ancestors, a free block's cached descendants are themselves free:
        dropping them loses nothing in use. They keep their pages until the pool reuses
        them in turn, at which point their own eviction is a no-op.
        """
        node = self._node_of_block.pop(block_id, None)
        if node is None:
            return  # already orphaned by an ancestor's eviction
        if node.parent is not None and node.token_key is not None:
            node.parent.children.pop(node.token_key, None)
        self._orphan(node)

    def _orphan(self, node: RadixNode) -> None:
        """Drop a node's whole subtree from the reverse index."""
        stack = list(node.children.values())
        node.children.clear()
        while stack:
            child = stack.pop()
            self._node_of_block.pop(child.block_id, None)  # type: ignore[arg-type]
            stack.extend(child.children.values())
            child.children.clear()

    # ---------------------------------------------------------------- inspection

    @property
    def num_cached_blocks(self) -> int:
        """How many blocks the tree currently points at. For tests and stats."""
        return len(self._node_of_block)


# ------------------------------------------------------------ 6. block manager


class BlockManager:
    """Capacity, growth, sharing, and release: pages made usable by a scheduler.

    The layer that turns the block pool's integers and the block table's arithmetic
    into something a scheduler can talk to:

    ::

        can_allocate(seq, n)   may this sequence take n more tokens?  (admission control)
        allocate(seq)          reserve blocks for everything uncomputed
        append_slot(seq)       grow by one token, adding a block only if needed
        fork(parent, child)    share the parent's blocks, copying nothing
        trim(seq, n)           roll back n tokens, releasing pages they emptied
        free(seq)              drop a holder on every block it owns

    It owns the block pool, the KV pool tensors, and — through the sequences themselves
    — the block tables. All three together, because copy-on-write is the one operation
    that touches a refcount, a table entry, and a page of keys and values at once.

    Copy-on-write copies only the last partial block, and only when that block is
    shared. Earlier blocks of a shared prefix are full and immutable, so they stay
    shared for the lifetime of both sequences; the partial block is the only page two
    sequences would both write into. The copy therefore costs `P` tokens of K and V
    rather than a prefix of arbitrary length.

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
        k_scale: float = 1.0,
        v_scale: float = 1.0,
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
            k_scale=k_scale,
            v_scale=v_scale,
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
