"""One sequence's logical-to-physical token mapping.

The whole paging indirection: a list of block ids, a shift, and a mask.

::

    logical position p ─▶ block = block_ids[p // P],  slot = block · P + (p % P)

Non-contiguous physical storage removes per-sequence contiguous reservation and its
fragmentation at a cost of one integer division and one modulo per token, both of which
reduce to bit operations because `P` is a power of two. The same arithmetic reappears
inside the paged attention kernel.

The one output that leaves Python is :meth:`BlockTable.slot_mapping`, the flat index
vector the paged write path scatters new keys and values through.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

__all__ = ["BlockTable"]


class BlockTable:
    """The physical block ids backing one sequence, in logical order.

    Two distinct sizes are tracked:

    * :attr:`num_slots` — capacity, always a multiple of `block_size`.
    * :attr:`num_tokens` — occupancy, usually not.

    Their difference is the internal fragmentation in the final block, bounded by `P - 1`
    tokens per sequence regardless of its length, and it tells the block manager whether
    appending a token needs a new block or fits in place.
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
            raise ValueError(
                f"cannot trim {count} tokens from a table holding {self._num_tokens}"
            )

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
