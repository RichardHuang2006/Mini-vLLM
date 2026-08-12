"""The physical block pool: allocation and reference counting.

The bottom of the paged memory hierarchy. The pool knows nothing about sequences,
tokens, or tensors: it hands out integer ids and counts how many holders each one has.
Everything above it — block tables, copy-on-write, admission control — is built from
those two operations, and none of it needs a GPU to be tested, which is why paging
starts here rather than in a kernel.

Reference counting is what makes sharing possible without copying. Forking a sequence
or reusing a cached prefix increments counts instead of duplicating a page of K and V,
and a block returns to circulation exactly when its last holder lets go.

Two decisions worth recording, because both are the less obvious choice:

* The free list is **FIFO, not LIFO**. Recycling the most recently freed block has
  better cache locality, but it also means a use-after-free usually reads back the
  data it just released and *looks correct*. Cycling the whole pool makes that class
  of bug fail loudly, which is worth more here than the locality.
* :class:`OutOfBlocks` is a **typed exception, not a `None` return**. Exhaustion is
  the condition that drives admission control and preemption in the block manager, so
  the caller has to handle it; a sentinel return is too easy to drop on the floor.
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

__all__ = ["Block", "BlockPool", "BlockPoolError", "OutOfBlocks"]


class BlockPoolError(RuntimeError):
    """Base for block pool misuse. These indicate bugs, not conditions."""


class OutOfBlocks(BlockPoolError):
    """The pool is exhausted.

    Unlike the other errors here this is an expected runtime *condition* rather than
    a bug: it is the signal to preempt or to leave a request waiting. It inherits
    from `BlockPoolError` so a caller can catch everything from the pool at once, but
    it is caught on its own in the scheduler.
    """


class Block(NamedTuple):
    """A read-only view of one block's state, for tests and debugging.

    The pool does not store these — it stores a flat list of counts, because a pool
    holds tens of thousands of blocks and an object per block would be pure overhead.
    This is what you get when you ask about one.
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

    The pool is sized once at startup, because the whole point of paging is that GPU
    memory is reserved up front and then partitioned — growing it later would mean a
    cudaMalloc in the middle of a decode step.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        self._num_blocks = num_blocks
        self._ref_counts = [0] * num_blocks
        self._free: deque[int] = deque(range(num_blocks))

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
            OutOfBlocks: if nothing is free. The caller's move is to preempt a
                running sequence or to stop admitting, not to retry.
        """
        if not self._free:
            raise OutOfBlocks(
                f"all {self._num_blocks} blocks are in use; "
                "the caller should preempt or stop admitting rather than retry"
            )

        block_id = self._free.popleft()
        self._ref_counts[block_id] = 1
        return block_id

    def allocate_many(self, count: int) -> list[int]:
        """Take `count` blocks, or none at all.

        All-or-nothing because a half-allocated sequence is worse than a rejected
        one: the caller would have to unwind, and the unwinding is exactly the code
        path that leaks blocks when it is wrong. Whatever was taken goes back before
        the exception leaves.
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
            needs that distinction: a write to a block with holders left has to copy,
            and a write to one it just released does not.
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

    # --------------------------------------------------------------- consistency

    def check_consistency(self) -> None:
        """Assert the free list and the reference counts still agree.

        Cheap enough to call from test teardown, which is where it earns its keep: a
        refcount bug is caught by the test that introduced it rather than surfacing
        as an out-of-memory three steps later, when the leak is thousands of
        iterations old and belongs to whoever ran last.
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
