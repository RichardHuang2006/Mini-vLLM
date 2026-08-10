"""Step 4.6 — the logical-to-physical block table.

The implementation maps positions with shifts and masks, so the tests compare it
against a naive `//` and `%` reference. That is the whole point of the step: the
arithmetic is easy to write two ways, only one of them is fast, so the slow one gets
to be the oracle.

Block ids are deliberately non-monotonic throughout. Physical order matching logical
order is precisely the case that hides an indirection bug, and after a few thousand
allocations the pool hands out nothing so tidy.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from mini_vllm.block.block_table import BlockTable

BLOCK_SIZES = [1, 2, 4, 8, 16, 32]


def naive_physical_slot(block_ids, block_size: int, position: int) -> int:
    """[§6.2](../DESIGN.md#62-block-table-indirection), transcribed literally."""
    return block_ids[position // block_size] * block_size + position % block_size


# --------------------------------------------------------------- construction


def test_a_new_table_is_empty():
    table = BlockTable(block_size=16)

    assert table.num_blocks == 0
    assert table.num_slots == 0
    assert table.num_tokens == 0
    assert table.num_empty_slots == 0
    assert table.block_ids == ()


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_a_power_of_two_block_size_is_accepted(block_size):
    assert BlockTable(block_size).block_size == block_size


@pytest.mark.parametrize("block_size", [0, -16, 3, 24, 100])
def test_the_block_size_must_be_a_power_of_two(block_size):
    """Because the mapping is a shift and a mask, not a division and a modulo."""
    with pytest.raises(ValueError, match="power of two"):
        BlockTable(block_size)


def test_construction_refuses_more_tokens_than_capacity():
    with pytest.raises(ValueError, match="does not fit"):
        BlockTable(4, block_ids=[5, 1], num_tokens=9)


def test_construction_refuses_a_negative_block_id():
    with pytest.raises(ValueError, match="non-negative"):
        BlockTable(4, block_ids=[5, -1])


def test_the_caller_cannot_alias_the_block_ids():
    """A shared list would let a caller corrupt the mapping behind our back."""
    ids = [5, 1]
    table = BlockTable(4, block_ids=ids, num_tokens=8)

    ids.append(99)

    assert table.block_ids == (5, 1)
    assert table.num_blocks == 2


# ---------------------------------------------------- capacity vs. occupancy


def test_capacity_is_block_aligned_and_occupancy_is_not():
    table = BlockTable(16, block_ids=[7, 42], num_tokens=20)

    assert table.num_slots == 32, "capacity is always a multiple of the block size"
    assert table.num_tokens == 20
    assert table.num_empty_slots == 12, "internal fragmentation, last block only"


def test_a_full_last_block_leaves_no_empty_slots():
    table = BlockTable(16, block_ids=[7, 42], num_tokens=32)
    assert table.num_empty_slots == 0


def test_appending_a_block_grows_capacity_but_not_occupancy():
    table = BlockTable(4)
    table.append_block(9)

    assert table.num_blocks == 1
    assert table.num_slots == 4
    assert table.num_tokens == 0


def test_appending_tokens_fills_the_partial_block_first():
    table = BlockTable(4, block_ids=[9], num_tokens=1)
    table.append_tokens(3)

    assert table.num_tokens == 4
    assert table.num_empty_slots == 0


def test_appending_past_capacity_raises_rather_than_allocating():
    """The table never touches the pool: only the block manager may."""
    table = BlockTable(4, block_ids=[9], num_tokens=3)

    with pytest.raises(ValueError, match="append_block first"):
        table.append_tokens(2)

    assert table.num_tokens == 3, "the failed append changed the occupancy"


@pytest.mark.parametrize(
    ("num_tokens", "num_new", "expected"),
    [
        (0, 0, 0),  # nothing to do
        (4, 1, 0),  # room in the partial block
        (4, 4, 0),  # exactly fills it
        (4, 5, 1),  # one token over the boundary costs a whole block
        (4, 12, 1),  # 16 tokens total: two blocks, one already held
        (4, 13, 2),  # 17 tokens total: three blocks
        (8, 1, 1),  # last block full, so any token at all needs a new one
        (8, 8, 1),
        (8, 9, 2),
    ],
)
def test_blocks_needed_for(num_tokens, num_new, expected):
    table = BlockTable(8, block_ids=[3], num_tokens=num_tokens)
    assert table.blocks_needed_for(num_new) == expected


def test_blocks_needed_for_refuses_a_negative_count():
    with pytest.raises(ValueError, match="non-negative"):
        BlockTable(8).blocks_needed_for(-1)


# ------------------------------------------------------- the indirection itself


def test_physical_slot_worked_by_hand():
    """block_size=4, table [5, 1, 6]: three ascending runs that jump backwards."""
    table = BlockTable(4, block_ids=[5, 1, 6], num_tokens=10)

    assert [table.physical_slot(p) for p in range(10)] == [
        20, 21, 22, 23,  # logical 0-3  -> block 5
        4, 5, 6, 7,      # logical 4-7  -> block 1
        24, 25,          # logical 8-9  -> block 6, partially filled
    ]  # fmt: skip


def test_block_index_and_offset_decompose_the_position():
    table = BlockTable(4, block_ids=[5, 1, 6], num_tokens=10)

    assert (table.block_index(9), table.block_offset(9)) == (2, 1)
    assert table.physical_slot(9) == 6 * 4 + 1


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_physical_slot_matches_the_naive_reference(block_size):
    block_ids = [11, 0, 7, 3, 40]
    table = BlockTable(block_size, block_ids=block_ids, num_tokens=len(block_ids) * block_size)

    for position in range(table.num_tokens):
        assert table.physical_slot(position) == naive_physical_slot(
            block_ids, block_size, position
        )


def test_shuffling_the_blocks_does_not_disturb_logical_order():
    """The indirection's promise: physical order is irrelevant to the sequence."""
    ordered = BlockTable(4, block_ids=[0, 1, 2], num_tokens=12)
    shuffled = BlockTable(4, block_ids=[2, 0, 1], num_tokens=12)

    for table in (ordered, shuffled):
        slots = [table.physical_slot(p) for p in range(12)]
        # Whatever the physical layout, each block's four slots stay together and in
        # order; only the jumps between runs differ.
        for start in (0, 4, 8):
            run = slots[start : start + 4]
            assert run == list(range(run[0], run[0] + 4))

    assert ordered.physical_slot(0) == 0
    assert shuffled.physical_slot(0) == 8


def test_the_last_block_is_partial():
    table = BlockTable(16, block_ids=[7, 42], num_tokens=17)

    assert table.block_index(16) == 1, "the 17th token opened the second block"
    assert table.block_offset(16) == 0
    assert table.physical_slot(16) == 42 * 16

    # The 15 slots after it exist but hold nothing addressable.
    assert table.num_empty_slots == 15
    with pytest.raises(IndexError):
        table.physical_slot(17)


@pytest.mark.parametrize("method", ["physical_slot", "block_index", "block_offset"])
@pytest.mark.parametrize("position", [-1, 20, 100])
def test_an_unoccupied_position_raises(method, position):
    table = BlockTable(4, block_ids=[5, 1, 6], num_tokens=10)
    with pytest.raises(IndexError, match="out of range"):
        getattr(table, method)(position)


def test_reserved_but_unwritten_slots_are_not_addressable():
    """Capacity is not permission: 8 slots exist, 5 hold tokens.

    Checking positions against occupancy rather than capacity is what turns a
    scheduler off-by-one into an exception instead of a read of whatever the previous
    holder of that block left behind.
    """
    table = BlockTable(4, block_ids=[5, 1], num_tokens=5)

    assert table.num_slots == 8
    assert table.physical_slot(4) == 4, "first slot of block 1"
    with pytest.raises(IndexError):
        table.physical_slot(5)


# ------------------------------------------- slot_mapping: the handoff to CUDA


def test_slot_mapping_is_int32():
    table = BlockTable(4, block_ids=[5, 1, 6], num_tokens=10)
    mapping = table.slot_mapping(range(10))

    assert mapping.dtype is torch.int32
    assert mapping.shape == (10,)


def test_slot_mapping_agrees_with_physical_slot():
    table = BlockTable(4, block_ids=[5, 1, 6], num_tokens=10)
    mapping = table.slot_mapping(range(10))

    assert mapping.tolist() == [table.physical_slot(p) for p in range(10)]


def test_slot_mapping_of_one_position_is_a_decode_step():
    """A decode step writes exactly one KV entry, at the sequence's last position."""
    table = BlockTable(4, block_ids=[5, 1, 6], num_tokens=9)
    mapping = table.slot_mapping([table.num_tokens - 1])

    assert mapping.tolist() == [24], "block 6, offset 0"


def test_slot_mapping_of_a_range_is_a_prefill_chunk():
    """Positions 4-7 of a prompt whose first chunk is already computed."""
    table = BlockTable(4, block_ids=[5, 1, 6], num_tokens=10)
    assert table.slot_mapping(range(4, 8)).tolist() == [4, 5, 6, 7]


def test_slot_mapping_of_nothing_is_empty_rather_than_an_error():
    mapping = BlockTable(4).slot_mapping([])

    assert mapping.shape == (0,)
    assert mapping.dtype is torch.int32


def test_slot_mapping_refuses_unoccupied_positions():
    table = BlockTable(4, block_ids=[5], num_tokens=2)
    with pytest.raises(IndexError):
        table.slot_mapping(range(4))


def test_slot_mapping_round_trips_through_a_scatter():
    """Write one distinguishable value per token through the mapping, read back blocks.

    This is the paged write path in miniature, and it is the test that would catch a
    mapping which is self-consistent but points into the wrong blocks.
    """
    block_size, pool_blocks = 4, 8
    table = BlockTable(block_size, block_ids=[5, 1, 6], num_tokens=10)

    # Sized by the *pool*, not the table: `physical_slot` indexes global storage, so
    # block id 6 needs slots 24-27 to exist even though the table holds three blocks.
    # Conflating the two sizes is the trap `num_slots` warns about.
    cache = torch.zeros(pool_blocks * block_size, dtype=torch.int64)
    values = torch.arange(1, table.num_tokens + 1)  # token at position p holds p + 1
    cache.scatter_(0, table.slot_mapping(range(table.num_tokens)).long(), values)

    blocks = cache.view(pool_blocks, block_size)
    assert blocks[5].tolist() == [1, 2, 3, 4]  # logical 0-3
    assert blocks[1].tolist() == [5, 6, 7, 8]  # logical 4-7
    assert blocks[6].tolist() == [9, 10, 0, 0]  # logical 8-9, then unwritten

    untouched = [i for i in range(pool_blocks) if i not in (5, 1, 6)]
    assert blocks[untouched].count_nonzero() == 0, "wrote into somebody else's blocks"

    for position in range(table.num_tokens):
        assert cache[table.physical_slot(position)] == position + 1


# ------------------------------------------------------ copy-on-write support


def test_replacing_a_block_repoints_one_entry_and_hands_back_the_old_id():
    table = BlockTable(4, block_ids=[5, 1, 6], num_tokens=10)

    assert table.replace_block(2, 30) == 6
    assert table.block_ids == (5, 1, 30)
    assert table.physical_slot(8) == 120, "the mapping follows the new block"
    assert table.num_tokens == 10, "copy-on-write moves data, not occupancy"


@pytest.mark.parametrize("index", [-1, 3, 99])
def test_replacing_an_out_of_range_block_raises(index):
    table = BlockTable(4, block_ids=[5, 1, 6], num_tokens=10)
    with pytest.raises(IndexError, match="out of range"):
        table.replace_block(index, 30)


def test_a_copy_shares_blocks_but_not_the_mapping():
    """The fork invariant: appending to one branch cannot move the other's tokens."""
    parent = BlockTable(4, block_ids=[5, 1], num_tokens=7)
    child = parent.copy()

    assert child.block_ids == parent.block_ids, "a fork copies no block data"
    assert child.num_tokens == parent.num_tokens

    child.append_tokens(1)
    child.append_block(30)
    child.append_tokens(1)
    child.replace_block(1, 31)

    assert parent.block_ids == (5, 1)
    assert parent.num_tokens == 7
    assert [parent.physical_slot(p) for p in range(7)] == [20, 21, 22, 23, 4, 5, 6]


# ------------------------------------------ property-based: against the oracle


@st.composite
def block_tables(draw) -> BlockTable:
    """A random table. Ids are unique, as the pool guarantees for live blocks."""
    block_size = draw(st.sampled_from(BLOCK_SIZES))
    block_ids = draw(st.lists(st.integers(0, 63), max_size=6, unique=True))
    num_tokens = draw(st.integers(0, len(block_ids) * block_size))
    return BlockTable(block_size, block_ids=block_ids, num_tokens=num_tokens)


@given(block_tables())
def test_the_mapping_always_matches_the_naive_reference(table: BlockTable):
    ids = list(table.block_ids)
    for position in range(table.num_tokens):
        assert table.physical_slot(position) == naive_physical_slot(
            ids, table.block_size, position
        )


@given(block_tables())
def test_distinct_positions_never_share_a_slot(table: BlockTable):
    """Injectivity. A shift or mask error shows up here as two tokens colliding, which
    on the GPU would be one sequence silently overwriting another's keys."""
    slots = [table.physical_slot(p) for p in range(table.num_tokens)]
    assert len(set(slots)) == len(slots)


@given(block_tables())
def test_every_slot_lands_inside_a_block_this_table_owns(table: BlockTable):
    owned = set(table.block_ids)
    for position in range(table.num_tokens):
        slot = table.physical_slot(position)
        assert slot // table.block_size in owned
        assert slot % table.block_size == table.block_offset(position)


@given(block_tables(), st.integers(0, 40))
def test_blocks_needed_for_is_sufficient_and_minimal(table: BlockTable, num_new: int):
    """Enough blocks to fit the tokens, and never one more than enough."""
    needed = table.blocks_needed_for(num_new)
    for i in range(needed):
        table.append_block(100 + i)

    table.append_tokens(num_new)  # raises if the estimate was short

    if needed:
        # Minimality only means something when blocks were asked for at all: a table
        # that already had spare capacity keeps it, so `num_empty_slots` can exceed
        # the block size when `needed` is zero.
        assert table.num_empty_slots < table.block_size
