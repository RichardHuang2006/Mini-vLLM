"""The physical block pool: allocation and reference counting.

The bottom of the paged memory hierarchy. The pool knows nothing about sequences,
tokens, or tensors: it hands out integer ids and counts holders. Block tables,
copy-on-write and admission control are all built on those two operations, none of
which needs a GPU to test.

Reference counting is what allows sharing without copying: forking a sequence or
reusing a cached prefix increments counts instead of duplicating a page of K and V, and
a block returns to circulation when its last holder releases it.

Two non-obvious choices:

* The free list is FIFO rather than LIFO. Recycling the most recently freed block has
  better cache locality, but it also means a use-after-free usually reads back the data
  it just released and appears correct. Cycling the whole pool makes that class of bug
  fail loudly, which is worth more than the locality.
* :class:`OutOfBlocks` is a typed exception rather than a `None` return. Exhaustion is
  the condition that drives admission control and preemption in the block manager, so
  the caller must handle it; a sentinel return is easy to ignore.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import NamedTuple

__all__ = ["Block", "BlockPool", "BlockPoolError", "OutOfBlocks"]


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

    ::

        allocate() -> id        take a free block, at ref_count 1
        incref(id)              add a holder (fork, prefix reuse)
        decref(id) -> bool      drop a holder; True if that freed the block

    Sized once at startup: paging reserves GPU memory up front and partitions it, so
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
