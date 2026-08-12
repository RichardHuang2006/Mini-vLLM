"""Block pool allocation and reference counting.

No GPU, no tensors, no model: every test here runs in microseconds, which is exactly
why this layer is worth testing exhaustively. A refcount bug that escapes this file
resurfaces in the benchmarks as an out-of-memory after four thousand requests, with
nothing left to say which one of them leaked.

The last section is a `hypothesis` state machine driving random allocate / incref /
decref sequences against a plain dict. It is the part that finds what hand-written
cases do not: hypothesis shrinks a failing sequence to the shortest one that still
fails, so a violation arrives as a two-line reproduction.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from mini_vllm.block.block_pool import Block, BlockPool, BlockPoolError, OutOfBlocks

POOL_SIZE = 8


@pytest.fixture
def pool() -> Iterator[BlockPool]:
    """A small pool, checked for internal consistency at teardown.

    The check is *consistency*, not "everything was freed": most tests here end
    mid-scenario with blocks deliberately held. The stricter everything-returned
    assertion belongs at engine level, where a finished request really should leave
    nothing behind — `test_engine.py` makes it.
    """
    pool = BlockPool(POOL_SIZE)
    yield pool
    pool.check_consistency()


# --------------------------------------------------------------- construction


def test_a_new_pool_is_entirely_free(pool: BlockPool):
    assert pool.num_blocks == POOL_SIZE
    assert pool.num_free == POOL_SIZE
    assert pool.num_allocated == 0
    assert pool.allocated_ids() == []


@pytest.mark.parametrize("num_blocks", [0, -1])
def test_a_pool_needs_at_least_one_block(num_blocks):
    with pytest.raises(ValueError, match="num_blocks must be positive"):
        BlockPool(num_blocks)


# ------------------------------------------------------------ allocate / free


def test_allocate_hands_out_a_live_block(pool: BlockPool):
    block_id = pool.allocate()

    assert pool.ref_count(block_id) == 1
    assert pool.num_free == POOL_SIZE - 1
    assert pool.allocated_ids() == [block_id]


def test_allocate_never_repeats_a_live_block(pool: BlockPool):
    """Two sequences handed the same page would silently overwrite each other's K and V."""
    ids = [pool.allocate() for _ in range(POOL_SIZE)]

    assert sorted(ids) == list(range(POOL_SIZE))
    assert pool.num_free == 0


def test_allocating_everything_then_freeing_it_restores_the_pool(pool: BlockPool):
    ids = [pool.allocate() for _ in range(POOL_SIZE)]
    for block_id in ids:
        assert pool.decref(block_id) is True

    assert pool.num_free == POOL_SIZE
    assert pool.num_allocated == 0
    assert pool.allocated_ids() == []


def test_exhaustion_raises_rather_than_returning_nothing(pool: BlockPool):
    for _ in range(POOL_SIZE):
        pool.allocate()

    with pytest.raises(OutOfBlocks, match="all 8 blocks are in use"):
        pool.allocate()


def test_out_of_blocks_is_catchable_as_a_pool_error():
    """The scheduler catches it specifically; keep the hierarchy intact for callers
    that just want to know the pool complained."""
    assert issubclass(OutOfBlocks, BlockPoolError)


def test_a_freed_block_returns_to_circulation(pool: BlockPool):
    first = pool.allocate()
    pool.decref(first)

    for _ in range(POOL_SIZE):
        pool.allocate()

    assert pool.num_free == 0


def test_the_free_list_is_first_in_first_out(pool: BlockPool):
    """A freed block goes to the back of the queue, not the front.

    Deliberate, and the reverse of what locality would suggest: reusing a
    just-released block means a use-after-free reads back its own stale data and the
    engine appears to work. Cycling the whole pool makes it fail instead.
    """
    a, b, c = pool.allocate(), pool.allocate(), pool.allocate()
    assert (a, b, c) == (0, 1, 2)

    pool.decref(a)

    assert pool.allocate() == 3, "the recycled block came back too soon"
    assert pool.allocate() == 4
    assert pool.allocate() == 5


# ---------------------------------------------------------- batch allocation


def test_allocate_many_takes_a_run_of_blocks(pool: BlockPool):
    ids = pool.allocate_many(3)

    assert len(set(ids)) == 3
    assert pool.num_free == POOL_SIZE - 3
    assert all(pool.ref_count(block_id) == 1 for block_id in ids)


def test_allocate_many_is_all_or_nothing(pool: BlockPool):
    """A half-allocated sequence is worse than a rejected one.

    The caller would have to unwind, and unwinding is the code path that leaks blocks
    when it is wrong — so the pool refuses before it takes anything.
    """
    pool.allocate_many(6)

    with pytest.raises(OutOfBlocks, match="asked for 4 blocks with 2 free"):
        pool.allocate_many(4)

    assert pool.num_free == 2, "a failed request took blocks with it"


def test_allocate_many_of_nothing_is_allowed(pool: BlockPool):
    """A sequence shorter than one block asks for zero, and that is not an error."""
    assert pool.allocate_many(0) == []
    assert pool.num_free == POOL_SIZE


def test_decref_many_reports_how_many_it_freed(pool: BlockPool):
    ids = pool.allocate_many(3)
    pool.incref(ids[0])  # one of them is shared, so it survives

    assert pool.decref_many(ids) == 2
    assert pool.num_free == POOL_SIZE - 1


# ------------------------------------------------------- reference counting


def test_a_shared_block_survives_until_its_last_holder(pool: BlockPool):
    """The mechanism behind copy-on-write sharing: fork and prefix reuse."""
    block_id = pool.allocate()
    pool.incref(block_id)
    pool.incref(block_id)
    assert pool.ref_count(block_id) == 3

    assert pool.decref(block_id) is False
    assert pool.decref(block_id) is False
    assert pool.num_free == POOL_SIZE - 1, "released while still held"

    assert pool.decref(block_id) is True
    assert pool.num_free == POOL_SIZE


def test_incref_returns_the_new_count(pool: BlockPool):
    block_id = pool.allocate()
    assert pool.incref(block_id) == 2


def test_sharing_costs_no_capacity(pool: BlockPool):
    """Forking is free in blocks, which is the entire point of refcounting."""
    block_id = pool.allocate()
    free_after_allocate = pool.num_free

    for _ in range(100):
        pool.incref(block_id)

    assert pool.num_free == free_after_allocate


# ------------------------------------------------------------------- misuse


def test_a_double_free_raises(pool: BlockPool):
    block_id = pool.allocate()
    pool.decref(block_id)

    with pytest.raises(BlockPoolError, match="already free"):
        pool.decref(block_id)


def test_increfing_a_free_block_raises(pool: BlockPool):
    """Sharing a block nobody owns means a stale id escaped somewhere."""
    with pytest.raises(BlockPoolError, match="must be allocated before"):
        pool.incref(0)


@pytest.mark.parametrize("bad_id", [-1, POOL_SIZE, POOL_SIZE + 100])
@pytest.mark.parametrize("method", ["incref", "decref", "ref_count"])
def test_an_out_of_range_id_raises(pool: BlockPool, method, bad_id):
    with pytest.raises(BlockPoolError, match="out of range"):
        getattr(pool, method)(bad_id)


def test_the_consistency_check_catches_a_corrupted_free_list(pool: BlockPool):
    """The check is only worth calling if it can fail, so make it fail once here."""
    pool.allocate()
    pool._free.append(0)  # the id is live, so it must not be on the free list

    with pytest.raises(BlockPoolError, match="disagrees with"):
        pool.check_consistency()

    pool._free.pop()  # leave the fixture's teardown check something valid


# -------------------------------------------------------------- the Block view


def test_the_block_view_reflects_state(pool: BlockPool):
    block_id = pool.allocate()

    assert pool.block(block_id) == Block(block_id=block_id, ref_count=1)
    assert not pool.block(block_id).is_free

    pool.decref(block_id)
    assert pool.block(block_id).is_free


def test_repr_says_how_full_it_is(pool: BlockPool):
    pool.allocate_many(3)
    assert repr(pool) == "BlockPool(num_blocks=8, free=5, allocated=3)"


# ------------------------------------------- property-based: against a shadow


class BlockPoolStateMachine(RuleBasedStateMachine):
    """Drive random allocate / incref / decref sequences against a plain dict.

    The invariant that matters, and the one a leak breaks:
    ``num_free + len(live) == num_blocks``. The pool is kept small so exhaustion is
    reached often rather than as a rare tail case.
    """

    POOL_SIZE = 4

    def __init__(self) -> None:
        super().__init__()
        self.pool = BlockPool(self.POOL_SIZE)
        self.live: dict[int, int] = {}  # block_id -> expected ref count

    @rule()
    def allocate(self) -> None:
        if self.pool.num_free == 0:
            with pytest.raises(OutOfBlocks):
                self.pool.allocate()
            return

        block_id = self.pool.allocate()
        assert block_id not in self.live, f"block {block_id} handed out while still live"
        self.live[block_id] = 1

    @precondition(lambda self: self.live)
    @rule(data=st.data())
    def incref(self, data) -> None:
        block_id = data.draw(st.sampled_from(sorted(self.live)))
        self.live[block_id] += 1
        assert self.pool.incref(block_id) == self.live[block_id]

    @precondition(lambda self: self.live)
    @rule(data=st.data())
    def decref(self, data) -> None:
        block_id = data.draw(st.sampled_from(sorted(self.live)))
        self.live[block_id] -= 1

        was_freed = self.pool.decref(block_id)
        if self.live[block_id] == 0:
            del self.live[block_id]
            assert was_freed, f"block {block_id} hit zero holders but was not released"
        else:
            assert not was_freed, f"block {block_id} released with holders remaining"

    @invariant()
    def accounting_balances(self) -> None:
        assert self.pool.num_free + len(self.live) == self.POOL_SIZE
        assert self.pool.allocated_ids() == sorted(self.live)
        for block_id, expected in self.live.items():
            assert self.pool.ref_count(block_id) == expected
        self.pool.check_consistency()


TestBlockPoolProperties = BlockPoolStateMachine.TestCase
