"""Step 4.7 — the block manager, copy-on-write, and paged attention's oracle.

Three groups:

* **Bookkeeping** — capacity, growth, admission control. Integers, no tensors.
* **Copy-on-write** — the claim that forking an N-block sequence allocates zero blocks,
  and that writing afterwards copies exactly one page and leaves the other branch's
  mapping untouched.
* **Equivalence** — paged attention over a deliberately *shuffled* block table produces
  what a plain dense cache produces. That shuffle is the real test of the indirection:
  a wrong gather still passes when physical order happens to match logical order,
  which it does for the first sequence a fresh pool ever serves.

Every test that allocates ends with a leak check. A leaked block is invisible until
the pool runs dry thousands of iterations later, in whichever test ran last.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from conftest import assert_allclose

from mini_vllm.attention import scaled_dot_product_attention_grouped
from mini_vllm.block.block_manager import BlockManager, OutOfBlocks
from mini_vllm.paged_attention import paged_attention_gathered
from mini_vllm.sampler import SamplingParams
from mini_vllm.serve.batch import PADDING_BLOCK, ForwardBatch
from mini_vllm.serve.sequence import Sequence

GREEDY = SamplingParams(temperature=0.0)


def sequence(num_tokens: int, **kwargs) -> Sequence:
    return Sequence(
        prompt_token_ids=list(range(1, num_tokens + 1)), sampling_params=GREEDY, **kwargs
    )


@pytest.fixture
def manager() -> Iterator[BlockManager]:
    """A tiny pool: 8 blocks of 4 tokens. Exhaustion is reachable, which is the point."""
    manager = BlockManager(num_blocks=8, block_size=4)
    yield manager
    manager.pool.check_consistency()


# ----------------------------------------------------------------- bookkeeping


def test_allocating_reserves_whole_blocks(manager: BlockManager):
    request = sequence(6)
    manager.allocate(request)

    table = manager.table(request)
    assert table.num_tokens == 6
    assert table.num_blocks == 2, "6 tokens in blocks of 4"
    assert manager.num_free_blocks == 6
    assert table.num_empty_slots == 2, "and the last block is partial"


def test_a_sequence_without_a_table_is_an_error(manager: BlockManager):
    with pytest.raises(ValueError, match="no block table"):
        manager.table(sequence(4))


def test_appending_a_slot_is_usually_free(manager: BlockManager):
    request = sequence(6)
    manager.allocate(request)
    free_before = manager.num_free_blocks

    manager.append_slot(request)

    assert manager.table(request).num_tokens == 7
    assert manager.num_free_blocks == free_before, "a decode step allocated a block it did not need"


def test_appending_across_a_block_boundary_costs_one_block(manager: BlockManager):
    request = sequence(8)
    manager.allocate(request)
    assert manager.num_free_blocks == 6

    manager.append_slot(request)  # the 9th token opens a third block

    assert manager.table(request).num_blocks == 3
    assert manager.num_free_blocks == 5


def test_a_chunked_prefill_extends_the_same_table(manager: BlockManager):
    """`allocate` is called once per chunk, and must not start over."""
    request = sequence(10)
    manager.allocate(request, 4)
    manager.allocate(request, 4)
    manager.allocate(request, 2)

    table = manager.table(request)
    assert table.num_tokens == 10
    assert table.num_blocks == 3
    assert len(set(table.block_ids)) == 3, "a chunk reused another chunk's block"


def test_freeing_returns_every_block(manager: BlockManager):
    request = sequence(10)
    manager.allocate(request)

    assert manager.free(request) == 3
    assert manager.num_free_blocks == 8
    assert request.block_table is None
    manager.check_no_leaks()


def test_freeing_twice_is_harmless(manager: BlockManager):
    """A scheduler retiring a sequence twice is a retry; double-freeing the pool
    would be corruption, so the manager absorbs it rather than propagating it."""
    request = sequence(4)
    manager.allocate(request)

    assert manager.free(request) == 1
    assert manager.free(request) == 0
    manager.check_no_leaks()


def test_the_leak_check_notices_a_sequence_that_was_never_freed(manager: BlockManager):
    request = sequence(4)
    manager.allocate(request)

    with pytest.raises(AssertionError, match="blocks leaked"):
        manager.check_no_leaks()

    manager.free(request)


# ---------------------------------------------------------- admission control


def test_can_allocate_answers_for_a_whole_prompt(manager: BlockManager):
    assert manager.can_allocate(sequence(32)), "8 blocks of 4 is exactly enough"
    assert not manager.can_allocate(sequence(33))


def test_can_allocate_accounts_for_the_partial_block(manager: BlockManager):
    """A sequence with room in its last block needs nothing for the next token."""
    request = sequence(5)
    manager.allocate(request)
    while manager.num_free_blocks:
        manager.allocate(sequence(4))  # drain the pool

    assert manager.can_allocate(request, 1), "the partial block still had room"
    assert not manager.can_allocate(request, 4), "but not that much room"


def test_exhaustion_raises_out_of_blocks(manager: BlockManager):
    held = [sequence(32)]
    manager.allocate(held[0])

    with pytest.raises(OutOfBlocks):
        manager.allocate(sequence(4))

    manager.free(held[0])


def test_a_failed_allocation_takes_no_blocks(manager: BlockManager):
    """All-or-nothing, so a refused admission does not leave the pool poorer."""
    manager.allocate(sequence(20))  # 5 blocks, 3 left
    free_before = manager.num_free_blocks

    with pytest.raises(OutOfBlocks):
        manager.allocate(sequence(20))

    assert manager.num_free_blocks == free_before


# --------------------------------------------------------------- copy-on-write


def test_forking_allocates_nothing(manager: BlockManager):
    """The headline claim: sharing a prefix of any length is free in blocks."""
    parent = sequence(12)
    manager.allocate(parent)
    free_before = manager.num_free_blocks

    child = sequence(12)
    manager.fork(parent, child)

    assert manager.num_free_blocks == free_before
    assert manager.table(child).block_ids == manager.table(parent).block_ids
    assert all(manager.pool.ref_count(b) == 2 for b in manager.table(parent).block_ids)


def test_forking_a_sequence_that_already_has_pages_is_refused(manager: BlockManager):
    parent, child = sequence(4), sequence(4)
    manager.allocate(parent)
    manager.allocate(child)

    with pytest.raises(ValueError, match="already has a block table"):
        manager.fork(parent, child)


def test_writing_after_a_fork_copies_exactly_one_page(manager: BlockManager):
    """The other branch's mapping must not move, and only the partial page is copied.

    Blocks 0 and 1 are full and will never be written to again, so they stay shared
    for both sequences' lifetimes. Block 2 is the one both would write into.
    """
    parent = sequence(10)
    manager.allocate(parent)
    shared = manager.table(parent).block_ids
    child = sequence(10)
    manager.fork(parent, child)

    free_before = manager.num_free_blocks
    manager.append_slot(child)

    child_ids = manager.table(child).block_ids
    assert manager.table(parent).block_ids == shared, "the parent's mapping moved"
    assert child_ids[:2] == shared[:2], "a full block was copied for no reason"
    assert child_ids[2] != shared[2], "the shared partial block was written in place"
    assert manager.num_free_blocks == free_before - 1, "more than one page was copied"

    assert manager.pool.ref_count(shared[2]) == 1, "the parent is now the only holder"
    assert manager.pool.ref_count(shared[0]) == 2, "the full blocks are still shared"


def test_the_copy_carries_the_cached_keys_and_values():
    """Copy-on-write moves data, not just ids. If it did not, a forked sequence would
    attend over an uninitialized page and the divergence would look like sampling."""
    manager = BlockManager(num_blocks=4, block_size=4, num_layers=2, num_kv_heads=2, head_dim=8)
    parent = sequence(6)
    manager.allocate(parent)

    # Fill the parent's second (partial) page with something recognizable.
    page = manager.table(parent).block_ids[1]
    manager.kv.keys[:, page].fill_(3.5)
    manager.kv.values[:, page].fill_(-1.25)

    child = sequence(6)
    manager.fork(parent, child)
    manager.append_slot(child)

    copied = manager.table(child).block_ids[1]
    assert copied != page
    assert torch.equal(manager.kv.keys[:, copied], manager.kv.keys[:, page])
    assert torch.equal(manager.kv.values[:, copied], manager.kv.values[:, page])

    manager.free(parent)
    manager.free(child)
    manager.check_no_leaks()


def test_a_fork_whose_last_block_is_full_needs_no_copy(manager: BlockManager):
    """There is nothing to write into: the next token opens a fresh private block."""
    parent = sequence(8)
    manager.allocate(parent)
    child = sequence(8)
    manager.fork(parent, child)

    free_before = manager.num_free_blocks
    manager.append_slot(child)

    assert manager.num_free_blocks == free_before - 1, "one new block, and no copy"
    assert manager.table(child).block_ids[:2] == manager.table(parent).block_ids


def test_admission_control_counts_the_copy(manager: BlockManager):
    """A shared partial page costs a block to write into, and `can_allocate` says so.

    Leaving it out is how admission succeeds and the write then fails mid-iteration,
    with half the batch already committed.
    """
    parent = sequence(6)
    manager.allocate(parent)
    child = sequence(6)
    manager.fork(parent, child)

    private = sequence(6)
    manager.allocate(private)
    assert manager.blocks_needed(private, 1) == 0, "a private partial page is free to extend"
    assert manager.blocks_needed(child, 1) == 1, "a shared one costs the copy"


def test_both_branches_free_cleanly(manager: BlockManager):
    parent = sequence(10)
    manager.allocate(parent)
    child = sequence(10)
    manager.fork(parent, child)
    manager.append_slot(child)
    manager.append_slot(parent)

    manager.free(parent)
    manager.free(child)

    manager.check_no_leaks()


# -------------------------------------------------- paged attention equivalence


def paged_batch(manager: BlockManager, requests, counts) -> ForwardBatch:
    """Reserve pages and build the batch metadata for one iteration."""
    for request, count in zip(requests, counts, strict=True):
        manager.allocate(request, count)
    return ForwardBatch.from_scheduled(
        list(zip(requests, counts, strict=True)), manager=manager
    )


def write_and_attend(manager: BlockManager, batch: ForwardBatch, q, k, v):
    """Write this iteration's K/V into the pool, then attend over it."""
    manager.kv.write(0, batch.slot_mapping, k, v)
    return paged_attention_gathered(
        q,
        manager.kv.layer_keys(0),
        manager.kv.layer_values(0),
        batch.block_tables,
        batch.cu_seqlens_q,
        batch.context_lens,
    )


def test_paged_attention_matches_a_dense_cache():
    """One sequence, prefilled in one pass. The simplest case, and the baseline."""
    heads, kv_heads, dim, length = 4, 2, 8, 10
    manager = BlockManager(4, block_size=4, num_layers=1, num_kv_heads=kv_heads, head_dim=dim)

    request = sequence(length)
    batch = paged_batch(manager, [request], [length])
    q = torch.randn(length, heads, dim)
    k = torch.randn(length, kv_heads, dim)
    v = torch.randn(length, kv_heads, dim)

    got = write_and_attend(manager, batch, q, k, v)

    expected = scaled_dot_product_attention_grouped(
        q.permute(1, 0, 2).unsqueeze(0),
        k.permute(1, 0, 2).unsqueeze(0),
        v.permute(1, 0, 2).unsqueeze(0),
        mask="causal",
    )
    assert_allclose(got, expected.squeeze(0).permute(1, 0, 2))


def test_a_shuffled_block_table_changes_nothing():
    """The real test of the indirection ([§6.2](../DESIGN.md#62-block-table-indirection)).

    The same tokens, written through a table whose physical order is reversed. If the
    gather arithmetic is wrong this is where it shows: with an unshuffled table,
    logical order and physical order coincide and almost any indexing bug looks right.
    """
    heads, kv_heads, dim, length = 4, 2, 8, 12
    q = torch.randn(length, heads, dim)
    k = torch.randn(length, kv_heads, dim)
    v = torch.randn(length, kv_heads, dim)

    results = []
    for shuffle in (False, True):
        manager = BlockManager(8, block_size=4, num_layers=1, num_kv_heads=kv_heads, head_dim=dim)
        request = sequence(length)
        manager.allocate(request, length)
        table = manager.table(request)

        if shuffle:
            ids = list(reversed(table.block_ids))
            for index, block_id in enumerate(ids):
                table.replace_block(index, block_id)
            assert table.block_ids != tuple(range(table.num_blocks))

        batch = ForwardBatch.from_scheduled([(request, length)], manager=manager)
        results.append(write_and_attend(manager, batch, q, k, v))
        manager.free(request)

    assert_allclose(results[1], results[0], msg="the gather depends on physical order")


def test_a_mixed_batch_matches_each_sequence_alone():
    """A chunk beside two decodes, in one call, each row unaffected by its neighbours.

    This is the shape Step 4.8's kernel has to serve in a single launch, so the
    reference has to serve it too.
    """
    heads, kv_heads, dim = 4, 2, 8
    manager = BlockManager(16, block_size=4, num_layers=1, num_kv_heads=kv_heads, head_dim=dim)

    # Two sequences already have cache; a third arrives with a 6-token prompt.
    old = [sequence(7), sequence(5)]
    for request in old:
        manager.allocate(request, len(request))
        length = len(request)
        batch = ForwardBatch.from_scheduled([(request, length)], manager=manager)
        manager.kv.write(
            0, batch.slot_mapping, torch.randn(length, kv_heads, dim),
            torch.randn(length, kv_heads, dim),
        )
        request.advance(length)
        request.append_token(9)

    fresh = sequence(6)
    scheduled = [(old[0], 1), (old[1], 1), (fresh, 6)]
    for request, count in scheduled:
        manager.allocate(request, count)
    batch = ForwardBatch.from_scheduled(scheduled, manager=manager)

    total = batch.total_tokens
    q = torch.randn(total, heads, dim)
    k = torch.randn(total, kv_heads, dim)
    v = torch.randn(total, kv_heads, dim)
    together = write_and_attend(manager, batch, q, k, v)

    # Now each sequence on its own, reading the same pool it just wrote.
    for index in range(batch.num_sequences):
        rows = batch.slice_of(index)
        alone = paged_attention_gathered(
            q[rows],
            manager.kv.layer_keys(0),
            manager.kv.layer_values(0),
            batch.block_tables[index : index + 1],
            torch.tensor([0, rows.stop - rows.start], dtype=torch.int32),
            batch.context_lens[index : index + 1],
        )
        assert_allclose(together[rows], alone, msg=f"sequence {index} was affected by the batch")


def test_the_padding_is_never_read():
    """A short sequence in a batch with a long one carries padded table entries.

    They must not be reachable: `context_lens` bounds the walk, and the padding is -1
    so that a walk which overruns it fails loudly instead of reading block 0.
    """
    kv_heads, dim = 2, 8
    manager = BlockManager(16, block_size=4, num_layers=1, num_kv_heads=kv_heads, head_dim=dim)
    short, long = sequence(2), sequence(9)

    batch = paged_batch(manager, [short, long], [2, 9])

    assert batch.block_tables.shape == (2, 3), "the widest table sets the width"
    assert batch.block_tables[0, 1:].tolist() == [PADDING_BLOCK, PADDING_BLOCK]

    manager.kv.write(
        0, batch.slot_mapping, torch.randn(11, kv_heads, dim), torch.randn(11, kv_heads, dim)
    )
    out = paged_attention_gathered(
        torch.randn(11, 4, dim),
        manager.kv.layer_keys(0),
        manager.kv.layer_values(0),
        batch.block_tables,
        batch.cu_seqlens_q,
        batch.context_lens,
    )

    assert torch.isfinite(out).all()


def test_reading_past_the_table_is_refused():
    """The check that makes the padding safe rather than merely unlikely."""
    kv_heads, dim = 2, 8
    manager = BlockManager(16, block_size=4, num_layers=1, num_kv_heads=kv_heads, head_dim=dim)
    request = sequence(4)
    manager.allocate(request, 4)
    batch = ForwardBatch.from_scheduled([(request, 4)], manager=manager)

    with pytest.raises(ValueError, match="padded there"):
        paged_attention_gathered(
            torch.randn(4, 4, dim),
            manager.kv.layer_keys(0),
            manager.kv.layer_values(0),
            torch.tensor([[batch.block_tables[0, 0].item(), PADDING_BLOCK]], dtype=torch.int32),
            batch.cu_seqlens_q,
            torch.tensor([8], dtype=torch.int32),  # claims twice the cache it has
        )


def test_the_slot_mapping_covers_this_iteration_only():
    """A decode step writes one slot; the chunk before it wrote its own."""
    manager = BlockManager(8, block_size=4)
    request = sequence(6)
    manager.allocate(request, 4)
    first = ForwardBatch.from_scheduled([(request, 4)], manager=manager)
    request.advance(4)

    manager.allocate(request, 2)
    second = ForwardBatch.from_scheduled([(request, 2)], manager=manager)

    assert first.slot_mapping.tolist() == [0, 1, 2, 3]
    assert second.slot_mapping.tolist() == [4, 5], "the second chunk rewrote the first"
    assert second.block_tables.tolist() == [[0, 1]]
