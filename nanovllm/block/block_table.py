"""One sequence's logical-to-physical token mapping (DESIGN.md 3.2).

This is the entire paging indirection, and it is worth noticing how little there
is to it: a list of block ids, a shift, and a mask. Non-contiguous physical
storage -- the thing that removes the need to reserve a contiguous KV region
per sequence -- costs one integer division and one modulo per token, both of
which reduce to bit operations because ``block_size`` is a power of two.

The one output that leaves Python is :meth:`BlockTable.slot_mapping`, the flat
index vector the Step 3.2 write kernel scatters new K/V through.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

__all__ = ["BlockTable"]


class BlockTable:
    """The physical block ids backing one sequence, in logical order.

    Two sizes are tracked and they are not the same thing:

    * :attr:`num_slots` -- capacity, always a multiple of ``block_size``.
    * :attr:`num_tokens` -- occupancy, usually not.

    The difference is internal fragmentation in the final block, and it is also
    what tells the block manager whether appending a token needs a new block
    (DESIGN.md 3.3) or fits where it stands.
    """

    def __init__(
        self,
        block_size: int,
        block_ids: Iterable[int] | None = None,
        num_tokens: int = 0,
    ) -> None:
        if block_size <= 0 or block_size & (block_size - 1):
            raise ValueError(
                f"block_size must be a positive power of two, got {block_size}"
            )

        self._block_size = block_size
        # Power of two, so `p // block_size` is `p >> shift` and
        # `p % block_size` is `p & mask`. This is the only reason Config
        # validates that block_size is a power of two.
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

    # -- geometry ----------------------------------------------------------

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def block_ids(self) -> tuple[int, ...]:
        """The table itself, logical order first. A copy: callers cannot alias it."""
        return tuple(self._block_ids)

    @property
    def num_blocks(self) -> int:
        return len(self._block_ids)

    @property
    def num_slots(self) -> int:
        """Token capacity of this sequence.

        Not to be confused with the range of :meth:`physical_slot`, which
        indexes the *pool's* whole flat cache. A one-block table holding block
        id 900 has 16 slots but addresses slot 14400.
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
        """How many fresh blocks appending ``num_new_tokens`` would require.

        The admission-control question of DESIGN.md 4.5, asked as arithmetic.
        Partial-block space is used first, so this is often zero.
        """
        if num_new_tokens < 0:
            raise ValueError(f"num_new_tokens must be non-negative, got {num_new_tokens}")

        deficit = num_new_tokens - self.num_empty_slots
        if deficit <= 0:
            return 0
        return (deficit + self._mask) >> self._shift

    # -- growth ------------------------------------------------------------

    def append_block(self, block_id: int) -> None:
        """Extend capacity by one block. The id comes from the pool."""
        if block_id < 0:
            raise ValueError(f"block ids must be non-negative, got {block_id}")
        self._block_ids.append(block_id)

    def append_tokens(self, count: int = 1) -> None:
        """Mark ``count`` more slots occupied.

        Capacity first, occupancy second: this raises rather than growing the
        table, because only the block manager may take blocks from the pool.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count > self.num_empty_slots:
            raise ValueError(
                f"cannot append {count} tokens: {self.num_empty_slots} free slots in "
                f"{self.num_blocks} blocks; append_block first"
            )
        self._num_tokens += count

    def replace_block(self, index: int, block_id: int) -> int:
        """Repoint one table entry, returning the id it displaced.

        The copy-on-write step of DESIGN.md 3.3. The displaced id is returned
        rather than dropped so the caller cannot forget to decref it.
        """
        if not 0 <= index < len(self._block_ids):
            raise IndexError(
                f"block index {index} out of range for {len(self._block_ids)} blocks"
            )
        if block_id < 0:
            raise ValueError(f"block ids must be non-negative, got {block_id}")

        displaced = self._block_ids[index]
        self._block_ids[index] = block_id
        return displaced

    def copy(self) -> BlockTable:
        """An independent table over the same blocks -- the fork of DESIGN.md 3.3.

        No block data is copied and no refcounts are touched here; the caller
        increfs. Independence is of the *list*, so that appending to one branch
        cannot alter the other's mapping.
        """
        return BlockTable(self._block_size, self._block_ids, self._num_tokens)

    # -- the indirection ---------------------------------------------------

    def block_index(self, position: int) -> int:
        """Which entry of this table covers ``position``."""
        self._check_position(position)
        return position >> self._shift

    def block_offset(self, position: int) -> int:
        """Where inside its block ``position`` sits."""
        self._check_position(position)
        return position & self._mask

    def physical_slot(self, position: int) -> int:
        """Flat index of a logical position in the pool's cache.

        ``block_id * block_size + offset``, written as a shift and an or since
        the offset is strictly narrower than the shift.
        """
        self._check_position(position)
        block_id = self._block_ids[position >> self._shift]
        return (block_id << self._shift) | (position & self._mask)

    def slot_mapping(self, positions: Iterable[int]) -> torch.Tensor:
        """The ``int32`` slot vector consumed by the Step 3.2 write kernel.

        One entry per token written this iteration -- the whole prompt for a
        prefill chunk, a single element for a decode step. ``int32`` because
        that is what the kernel indexes with, and a pool large enough to
        overflow it would not fit in any current GPU.
        """
        return torch.tensor(
            [self.physical_slot(position) for position in positions], dtype=torch.int32
        )

    def _check_position(self, position: int) -> None:
        # Deliberately checked against occupancy, not capacity. A slot that
        # exists but holds no token is not addressable, so a scheduler
        # off-by-one fails here instead of quietly reading stale cache.
        if not 0 <= position < self._num_tokens:
            raise IndexError(
                f"position {position} out of range for a sequence of "
                f"{self._num_tokens} tokens"
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(block_size={self._block_size}, "
            f"block_ids={self._block_ids}, num_tokens={self._num_tokens}/{self.num_slots})"
        )
