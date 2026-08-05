"""The physical block pool: allocation and reference counting.

This is the bottom of the memory hierarchy described in DESIGN.md 3. The pool
knows nothing about sequences, tokens, or tensors -- it hands out integer ids
and counts how many holders each one has. Everything above it (block tables,
copy-on-write, the radix prefix cache) is built from those two operations.

Reference counting is what makes DESIGN.md 3.3 and 3.4 possible: forking a
sequence or reusing a cached prefix increments counts instead of copying data,
and a block returns to circulation exactly when its last holder lets go.
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

__all__ = ["Block", "BlockPool", "BlockPoolError", "OutOfBlocks"]


class BlockPoolError(RuntimeError):
    """Base for block pool misuse. These indicate bugs, not conditions."""


class OutOfBlocks(BlockPoolError):
    """The pool is exhausted.

    Unlike the other errors here this is an expected runtime *condition*, not a
    bug: it is the signal that drives admission control (DESIGN.md 4.5) and
    preemption (DESIGN.md 4.2). It is a typed exception rather than a ``None``
    return so a caller cannot accidentally ignore it.
    """


class Block(NamedTuple):
    """A read-only view of one block's state, for tests and debugging."""

    block_id: int
    ref_count: int

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0


class BlockPool:
    """A fixed pool of physical blocks, handed out by id and refcounted.

    Internally this is a free list plus a flat array of reference counts. The
    counts are a plain ``list[int]`` rather than per-block objects because the
    pool holds tens of thousands of entries that are pure bookkeeping; the
    :class:`Block` view exists for when you want to look at one.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        self._num_blocks = num_blocks
        self._ref_counts = [0] * num_blocks

        # FIFO, not LIFO. Reusing the most recently freed block would give
        # slightly better cache locality, but it also means a use-after-free
        # usually reads back the data it just released and looks correct.
        # Cycling through the whole pool instead makes that class of bug fail
        # loudly, which matters more here than locality.
        self._free: deque[int] = deque(range(num_blocks))

    # -- introspection -----------------------------------------------------

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
        """Ids currently held by someone. Ordered, for reproducible output."""
        return [i for i, count in enumerate(self._ref_counts) if count]

    # -- allocation --------------------------------------------------------

    def allocate(self) -> int:
        """Take one block from the free list at reference count 1.

        Raises:
            OutOfBlocks: if nothing is free.
        """
        if not self._free:
            raise OutOfBlocks(
                f"all {self._num_blocks} blocks are in use; "
                "the caller should preempt or evict rather than retry"
            )

        block_id = self._free.popleft()
        self._ref_counts[block_id] = 1
        return block_id

    def incref(self, block_id: int) -> int:
        """Add a holder. Used by fork (3.3) and prefix cache hits (3.4)."""
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
            True if this call freed the block, False if holders remain. The
            radix tree (3.4) uses this to know when a node becomes evictable.
        """
        self._check_id(block_id)
        if self._ref_counts[block_id] == 0:
            raise BlockPoolError(
                f"block {block_id} is already free (double free); "
                "some holder released it twice"
            )

        self._ref_counts[block_id] -= 1
        if self._ref_counts[block_id]:
            return False

        self._free.append(block_id)
        return True

    # -- consistency -------------------------------------------------------

    def check_consistency(self) -> None:
        """Assert the free list and the reference counts still agree.

        Cheap enough to call from test teardown, which is where it earns its
        keep: it catches a refcount bug in the test that introduced it rather
        than three phases later.
        """
        free = list(self._free)

        if len(set(free)) != len(free):
            duplicates = sorted({i for i in free if free.count(i) > 1})
            raise BlockPoolError(f"blocks appear twice in the free list: {duplicates}")

        expected_free = {i for i, count in enumerate(self._ref_counts) if count == 0}
        if set(free) != expected_free:
            raise BlockPoolError(
                f"free list {sorted(free)} disagrees with zero-refcount blocks "
                f"{sorted(expected_free)}"
            )

        if any(count < 0 for count in self._ref_counts):
            negative = [i for i, count in enumerate(self._ref_counts) if count < 0]
            raise BlockPoolError(f"negative reference counts at blocks {negative}")

    def _check_id(self, block_id: int) -> None:
        if not 0 <= block_id < self._num_blocks:
            raise BlockPoolError(
                f"block id {block_id} out of range for a pool of {self._num_blocks}"
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(num_blocks={self._num_blocks}, "
            f"free={self.num_free}, allocated={self.num_allocated})"
        )
