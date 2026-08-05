"""Step 1.1 -- block pool allocation and reference counting.

No GPU, no tensors, no model: every test here runs in microseconds, which is
what makes it worth testing this layer exhaustively. The refcount bugs that
would otherwise surface in Phase 5 as a mysterious OOM are cheap to find here.
"""

from __future__ import annotations

from typing import Iterator

import pytest
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from nanovllm.block.block_pool import Block, BlockPool, BlockPoolError, OutOfBlocks

POOL_SIZE = 8


@pytest.fixture
def pool() -> Iterator[BlockPool]:
    """A small pool, checked for internal consistency at teardown.

    The check is deliberately *consistency*, not "everything was freed": most
    tests here end mid-scenario with blocks deliberately held. The
    stricter everything-returned assertion belongs at engine level, where a
    finished request really should leave nothing behind.
    """
    pool = BlockPool(POOL_SIZE)
    yield pool
    pool.check_consistency()


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def test_new_pool_is_entirely_free(pool: BlockPool):
    assert pool.num_blocks == POOL_SIZE
    assert pool.num_free == POOL_SIZE
    assert pool.num_allocated == 0
    assert pool.allocated_ids() == []


@pytest.mark.parametrize("num_blocks", [0, -1])
def test_pool_size_must_be_positive(num_blocks):
    with pytest.raises(ValueError, match="num_blocks must be positive"):
        BlockPool(num_blocks)


# --------------------------------------------------------------------------
# allocate / free
# --------------------------------------------------------------------------


def test_allocate_hands_out_a_live_block(pool: BlockPool):
    block_id = pool.allocate()

    assert pool.ref_count(block_id) == 1
    assert pool.num_free == POOL_SIZE - 1
    assert pool.allocated_ids() == [block_id]


def test_allocate_never_repeats_a_live_block(pool: BlockPool):
    ids = [pool.allocate() for _ in range(POOL_SIZE)]

    assert sorted(ids) == list(range(POOL_SIZE))
    assert pool.num_free == 0


def test_allocate_all_then_free_all_restores_the_pool(pool: BlockPool):
    """PLAN.md Step 1.1 'Done when', first clause."""
    ids = [pool.allocate() for _ in range(POOL_SIZE)]
    for block_id in ids:
        assert pool.decref(block_id) is True

    assert pool.num_free == POOL_SIZE
    assert pool.num_allocated == 0
    assert pool.allocated_ids() == []


def test_exhaustion_raises_out_of_blocks(pool: BlockPool):
    for _ in range(POOL_SIZE):
        pool.allocate()

    with pytest.raises(OutOfBlocks, match="all 8 blocks are in use"):
        pool.allocate()


def test_out_of_blocks_is_catchable_as_pool_error():
    """Phase 4/5 catch OutOfBlocks specifically; keep the hierarchy intact."""
    assert issubclass(OutOfBlocks, BlockPoolError)


def test_freed_block_is_reusable(pool: BlockPool):
    first = pool.allocate()
    pool.decref(first)

    for _ in range(POOL_SIZE):
        pool.allocate()  # the freed block is back in circulation

    assert pool.num_free == 0


def test_free_list_is_fifo(pool: BlockPool):
    """A freed block goes to the back of the queue, not the front.

    Deliberate: immediately reusing a just-freed block would let a
    use-after-free read back its own stale data and appear to work.
    """
    a, b, c = pool.allocate(), pool.allocate(), pool.allocate()
    assert (a, b, c) == (0, 1, 2)

    pool.decref(a)

    assert pool.allocate() == 3  # next untouched block, not the recycled 0
    assert pool.allocate() == 4
    assert pool.allocate() == 5


# --------------------------------------------------------------------------
# reference counting
# --------------------------------------------------------------------------


def test_shared_block_survives_until_the_last_holder(pool: BlockPool):
    """The mechanism behind fork (DESIGN.md 3.3) and prefix reuse (3.4)."""
    block_id = pool.allocate()
    pool.incref(block_id)
    pool.incref(block_id)
    assert pool.ref_count(block_id) == 3

    assert pool.decref(block_id) is False
    assert pool.decref(block_id) is False
    assert pool.num_free == POOL_SIZE - 1  # still held

    assert pool.decref(block_id) is True
    assert pool.num_free == POOL_SIZE


def test_incref_returns_the_new_count(pool: BlockPool):
    block_id = pool.allocate()
    assert pool.incref(block_id) == 2


def test_sharing_does_not_consume_pool_capacity(pool: BlockPool):
    """Forking is free in blocks; that is the whole point of refcounting."""
    block_id = pool.allocate()
    free_after_allocate = pool.num_free

    for _ in range(100):
        pool.incref(block_id)

    assert pool.num_free == free_after_allocate


# --------------------------------------------------------------------------
# misuse
# --------------------------------------------------------------------------


def test_double_free_raises(pool: BlockPool):
    """PLAN.md Step 1.1 'Done when', second clause."""
    block_id = pool.allocate()
    pool.decref(block_id)

    with pytest.raises(BlockPoolError, match="already free"):
        pool.decref(block_id)


def test_incref_of_a_free_block_raises(pool: BlockPool):
    """Sharing a block nobody owns means a stale id escaped somewhere."""
    with pytest.raises(BlockPoolError, match="must be allocated before"):
        pool.incref(0)


@pytest.mark.parametrize("bad_id", [-1, POOL_SIZE, POOL_SIZE + 100])
@pytest.mark.parametrize("method", ["incref", "decref", "ref_count"])
def test_out_of_range_ids_raise(pool: BlockPool, method, bad_id):
    with pytest.raises(BlockPoolError, match="out of range"):
        getattr(pool, method)(bad_id)


# --------------------------------------------------------------------------
# the Block view
# --------------------------------------------------------------------------


def test_block_view_reflects_state(pool: BlockPool):
    block_id = pool.allocate()

    assert pool.block(block_id) == Block(block_id=block_id, ref_count=1)
    assert not pool.block(block_id).is_free

    pool.decref(block_id)
    assert pool.block(block_id).is_free


# --------------------------------------------------------------------------
# property-based: the pool against a shadow model
# --------------------------------------------------------------------------


class BlockPoolStateMachine(RuleBasedStateMachine):
    """Drive random allocate/incref/decref sequences against a plain dict.

    Hypothesis searches for an operation sequence that breaks an invariant and
    then shrinks it to the shortest one that still fails, so a violation
    arrives as a two-line reproduction rather than a puzzle. The pool is kept
    small so that exhaustion is reached often.
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
        """PLAN.md Step 1.1 'Done when', third clause."""
        assert self.pool.num_free + len(self.live) == self.POOL_SIZE
        assert self.pool.allocated_ids() == sorted(self.live)
        for block_id, expected in self.live.items():
            assert self.pool.ref_count(block_id) == expected
        self.pool.check_consistency()


TestBlockPoolProperties = BlockPoolStateMachine.TestCase
