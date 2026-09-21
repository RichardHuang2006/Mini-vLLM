"""The paged KV cache and the continuous-batching scheduler.

Four groups: bookkeeping (refcounts, slot arithmetic, capacity), copy-on-write and
radix-tree prefix caching, the scheduler's policy, and identity, where batched, chunked,
piggybacked and preempted runs are compared token-for-token against single-sequence runs.
Every test that allocates ends with a leak check.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from conftest import (
    FP8_MAX_ERROR,
    assert_allclose,
    assert_tokens_equal,
    config_from_hf,
    relative_error,
    tiny_kv_manager,
    weights_from_hf,
)

from mini_vllm import ops
from mini_vllm.cache import (
    BlockManager,
    BlockPool,
    BlockTable,
    OutOfBlocks,
    PagedKvPool,
    PrefixCache,
)
from mini_vllm.config import SamplingParams, SchedulerConfig
from mini_vllm.engine import PagedModelRunner, generate_ids_cached
from mini_vllm.model import Qwen3Cached, Qwen3Paged
from mini_vllm.scheduler import (
    PADDING_BLOCK,
    DenseModelRunner,
    ForwardBatch,
    Scheduler,
    SchedulerOutput,
    Sequence,
    SequenceStatus,
)

GREEDY = SamplingParams(temperature=0.0)
WHOLE = {"enable_chunked_prefill": False}


def make(prompt_len: int, max_tokens: int = 4, **kwargs) -> Sequence:
    kwargs.setdefault("sampling_params", GREEDY)
    return Sequence(prompt_token_ids=list(range(1, prompt_len + 1)), max_tokens=max_tokens, **kwargs)


def tokens(token_ids: list[int], **kwargs) -> Sequence:
    return Sequence(prompt_token_ids=list(token_ids), sampling_params=GREEDY, **kwargs)


def test_refcounts_allow_sharing_without_capacity():
    """Two holders of one block cost one block."""
    pool = BlockPool(num_blocks=4)
    block = pool.allocate()
    pool.incref(block)

    assert pool.ref_count(block) == 2
    assert pool.num_free == 3, "sharing consumed a second page"

    assert not pool.decref(block), "a holder remains"
    assert pool.decref(block), "the last holder frees it"
    assert pool.num_free == 4
    pool.check_consistency()


def test_the_free_list_is_fifo_so_use_after_free_fails_loudly():
    pool = BlockPool(num_blocks=3)
    first = pool.allocate()
    pool.decref(first)

    # The pool hands out the other blocks before recycling the freed one.
    assert pool.allocate() != first
    assert pool.allocate() != first
    assert pool.allocate() == first


def test_double_free_and_free_incref_are_errors():
    pool = BlockPool(num_blocks=2)
    block = pool.allocate()
    pool.decref(block)

    with pytest.raises(Exception, match="double free"):
        pool.decref(block)
    with pytest.raises(Exception, match="must be allocated"):
        pool.incref(block)


def test_allocate_many_is_all_or_nothing():
    pool = BlockPool(num_blocks=4)
    pool.allocate()

    with pytest.raises(OutOfBlocks):
        pool.allocate_many(4)
    assert pool.num_free == 3, "a failed group allocation took blocks"


def test_the_slot_arithmetic():
    """position -> (block, offset) -> flat slot, through an out-of-order table."""
    table = BlockTable(block_size=4, block_ids=[7, 2, 9], num_tokens=10)

    assert table.block_index(5) == 1 and table.block_offset(5) == 1
    assert table.physical_slot(0) == 7 * 4 + 0
    assert table.physical_slot(5) == 2 * 4 + 1
    assert table.physical_slot(9) == 9 * 4 + 1
    assert table.slots([0, 5, 9]) == [28, 9, 37]
    assert table.slot_mapping([0, 5]).dtype == torch.int32


def test_capacity_and_occupancy_are_distinct():
    table = BlockTable(block_size=4, block_ids=[1, 2], num_tokens=6)
    assert table.num_slots == 8 and table.num_tokens == 6
    assert table.num_empty_slots == 2
    assert table.blocks_needed_for(2) == 0, "the partial block still has room"
    assert table.blocks_needed_for(3) == 1
    with pytest.raises(IndexError):
        table.physical_slot(6)  # a slot that exists but holds no token


def test_trim_reports_emptied_blocks_but_releases_nothing():
    table = BlockTable(4, block_ids=[10, 11, 12], num_tokens=9)
    assert table.trim_tokens(2) == 1, "9 -> 7 tokens empties the third block"
    assert table.num_blocks == 3, "the blocks stay until the manager drops them"
    with pytest.raises(ValueError, match="still holds tokens"):
        BlockTable(4, block_ids=[1, 2], num_tokens=6).drop_last_block()


def test_copy_shares_blocks_without_touching_refcounts():
    table = BlockTable(4, block_ids=[3, 5], num_tokens=6)
    forked = table.copy()
    forked.append_block(9)
    assert table.block_ids == (3, 5), "appending to the fork moved the original"
    assert forked.block_ids == (3, 5, 9)


@pytest.fixture
def manager() -> Iterator[BlockManager]:
    """A tiny pool: 8 blocks of 4 tokens, so exhaustion is reachable."""
    manager = BlockManager(num_blocks=8, block_size=4)
    yield manager
    manager.pool.check_consistency()


def test_allocate_append_free_round_trip(manager: BlockManager):
    request = tokens(list(range(6)))
    manager.allocate(request)
    assert manager.table(request).num_blocks == 2 and manager.num_free_blocks == 6

    manager.append_slot(request)  # fits the partial block: free
    free_before = manager.num_free_blocks
    assert manager.num_free_blocks == free_before

    manager.allocate(request, 2)  # crosses the boundary: one fresh block
    assert manager.table(request).num_blocks == 3

    assert manager.free(request) == 3
    assert manager.free(request) == 0, "freeing twice is a harmless retry"
    manager.check_no_leaks()


def test_a_chunked_prefill_extends_the_same_table(manager: BlockManager):
    request = tokens(list(range(10)))
    manager.allocate(request, 4)
    manager.allocate(request, 4)
    manager.allocate(request, 2)

    table = manager.table(request)
    assert table.num_tokens == 10 and table.num_blocks == 3
    assert len(set(table.block_ids)) == 3, "a chunk reused another chunk's block"
    manager.free(request)


def test_exhaustion_raises_and_takes_nothing(manager: BlockManager):
    held = tokens(list(range(20)))  # 5 of 8 blocks
    manager.allocate(held)
    free_before = manager.num_free_blocks

    with pytest.raises(OutOfBlocks):
        manager.allocate(tokens(list(range(20))))

    assert manager.num_free_blocks == free_before, "a failed allocation kept blocks"
    manager.free(held)


def test_the_leak_check_notices_an_unfreed_sequence(manager: BlockManager):
    request = tokens([1, 2, 3, 4])
    manager.allocate(request)
    with pytest.raises(AssertionError, match="blocks leaked"):
        manager.check_no_leaks()
    manager.free(request)


def test_forking_allocates_nothing(manager: BlockManager):
    """The headline claim: sharing a prefix of any length is free in blocks."""
    parent = tokens(list(range(12)))
    manager.allocate(parent)
    free_before = manager.num_free_blocks

    child = tokens(list(range(12)))
    manager.fork(parent, child)

    assert manager.num_free_blocks == free_before
    assert manager.table(child).block_ids == manager.table(parent).block_ids
    assert all(manager.pool.ref_count(b) == 2 for b in manager.table(parent).block_ids)
    manager.free(parent), manager.free(child)


def test_writing_after_a_fork_copies_exactly_one_page(manager: BlockManager):
    """Blocks 0 and 1 are full and stay shared; block 2 is the one both would write into."""
    parent = tokens(list(range(10)))
    manager.allocate(parent)
    shared = manager.table(parent).block_ids
    child = tokens(list(range(10)))
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
    manager.free(parent), manager.free(child)


def test_the_copy_carries_the_cached_keys_and_values():
    """Copy-on-write moves data, not just ids; otherwise a fork attends over a blank page."""
    manager = BlockManager(num_blocks=4, block_size=4, num_layers=2, num_kv_heads=2, head_dim=8)
    parent = tokens(list(range(6)))
    manager.allocate(parent)

    page = manager.table(parent).block_ids[1]
    manager.kv.keys[:, page].fill_(3.5)
    manager.kv.values[:, page].fill_(-1.25)

    child = tokens(list(range(6)))
    manager.fork(parent, child)
    manager.append_slot(child)

    copied = manager.table(child).block_ids[1]
    assert copied != page
    assert torch.equal(manager.kv.keys[:, copied], manager.kv.keys[:, page])
    assert torch.equal(manager.kv.values[:, copied], manager.kv.values[:, page])

    manager.free(parent), manager.free(child)
    manager.check_no_leaks()


def test_a_fork_whose_last_block_is_full_needs_no_copy(manager: BlockManager):
    """There is nothing to write into: the next token opens a fresh private block."""
    parent = tokens(list(range(8)))
    manager.allocate(parent)
    child = tokens(list(range(8)))
    manager.fork(parent, child)

    free_before = manager.num_free_blocks
    manager.append_slot(child)

    assert manager.num_free_blocks == free_before - 1, "one new block, and no copy"
    assert manager.table(child).block_ids[:2] == manager.table(parent).block_ids
    manager.free(parent), manager.free(child)


def test_admission_control_counts_the_copy(manager: BlockManager):
    """A shared partial page costs a block to write into, and `blocks_needed` says so."""
    parent = tokens(list(range(6)))
    manager.allocate(parent)
    child = tokens(list(range(6)))
    manager.fork(parent, child)
    private = tokens(list(range(6)))
    manager.allocate(private)

    assert manager.blocks_needed(private, 1) == 0, "a private partial page is free to extend"
    assert manager.blocks_needed(child, 1) == 1, "a shared one costs the copy"
    manager.free(parent), manager.free(child), manager.free(private)


def test_speculative_trim_returns_spilled_pages(manager: BlockManager):
    """The rollback path: rejected proposals that crossed a block boundary."""
    request = tokens([1, 2, 3], max_tokens=32)
    request.set_status(SequenceStatus.RUNNING)
    manager.allocate(request, 3)
    free_after_prompt = manager.num_free_blocks

    manager.allocate(request, 5)  # a pending token plus four proposals: two more blocks
    assert manager.num_free_blocks < free_after_prompt

    released = manager.trim(request, 5)

    assert released == 1, "the pages the rejected tail occupied come back"
    assert manager.num_free_blocks == free_after_prompt
    assert manager.table(request).num_tokens == 3
    manager.free(request)


def paged_batch(manager: BlockManager, requests, counts) -> ForwardBatch:
    for request, count in zip(requests, counts, strict=True):
        manager.allocate(request, count)
    return ForwardBatch.from_scheduled(list(zip(requests, counts, strict=True)), manager=manager)


def write_and_attend(manager: BlockManager, batch: ForwardBatch, q, k, v):
    manager.kv.write(0, batch.slot_mapping, k, v)
    return ops.paged_attention_gathered(
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

    request = tokens(list(range(length)))
    batch = paged_batch(manager, [request], [length])
    q = torch.randn(length, heads, dim)
    k = torch.randn(length, kv_heads, dim)
    v = torch.randn(length, kv_heads, dim)

    got = write_and_attend(manager, batch, q, k, v)

    expected = ops.scaled_dot_product_attention_grouped(
        q.permute(1, 0, 2).unsqueeze(0),
        k.permute(1, 0, 2).unsqueeze(0),
        v.permute(1, 0, 2).unsqueeze(0),
        mask="causal",
    )
    assert_allclose(got, expected.squeeze(0).permute(1, 0, 2))


def test_a_shuffled_block_table_changes_nothing():
    """The real test of the indirection: with an unshuffled table, logical and physical
    order coincide and almost any indexing bug looks right."""
    heads, kv_heads, dim, length = 4, 2, 8, 12
    q = torch.randn(length, heads, dim)
    k = torch.randn(length, kv_heads, dim)
    v = torch.randn(length, kv_heads, dim)

    results = []
    for shuffle in (False, True):
        manager = BlockManager(8, block_size=4, num_layers=1, num_kv_heads=kv_heads, head_dim=dim)
        request = tokens(list(range(length)))
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
    """A chunk beside two decodes in one call: the shape the paged kernels must serve."""
    heads, kv_heads, dim = 4, 2, 8
    manager = BlockManager(16, block_size=4, num_layers=1, num_kv_heads=kv_heads, head_dim=dim)

    old = [tokens(list(range(7))), tokens(list(range(5)))]
    for request in old:
        manager.allocate(request, len(request))
        length = len(request)
        batch = ForwardBatch.from_scheduled([(request, length)], manager=manager)
        manager.kv.write(
            0, batch.slot_mapping,
            torch.randn(length, kv_heads, dim), torch.randn(length, kv_heads, dim),
        )
        request.advance(length)
        request.append_token(9)

    fresh = tokens(list(range(6)))
    scheduled = [(old[0], 1), (old[1], 1), (fresh, 6)]
    for request, count in scheduled:
        manager.allocate(request, count)
    batch = ForwardBatch.from_scheduled(scheduled, manager=manager)

    total = batch.total_tokens
    q = torch.randn(total, heads, dim)
    k = torch.randn(total, kv_heads, dim)
    v = torch.randn(total, kv_heads, dim)
    together = write_and_attend(manager, batch, q, k, v)

    for index in range(batch.num_sequences):
        rows = batch.slice_of(index)
        alone_result = ops.paged_attention_gathered(
            q[rows],
            manager.kv.layer_keys(0),
            manager.kv.layer_values(0),
            batch.block_tables[index : index + 1],
            torch.tensor([0, rows.stop - rows.start], dtype=torch.int32),
            batch.context_lens[index : index + 1],
        )
        assert_allclose(together[rows], alone_result, msg=f"sequence {index} affected by batch")


def test_padding_is_never_read_and_overruns_are_refused():
    kv_heads, dim = 2, 8
    manager = BlockManager(16, block_size=4, num_layers=1, num_kv_heads=kv_heads, head_dim=dim)
    short, long = tokens([1, 2]), tokens(list(range(9)))

    batch = paged_batch(manager, [short, long], [2, 9])
    assert batch.block_tables.shape == (2, 3), "the widest table sets the width"
    assert batch.block_tables[0, 1:].tolist() == [PADDING_BLOCK, PADDING_BLOCK]

    manager.kv.write(0, batch.slot_mapping,
                     torch.randn(11, kv_heads, dim), torch.randn(11, kv_heads, dim))
    out = ops.paged_attention_gathered(
        torch.randn(11, 4, dim),
        manager.kv.layer_keys(0), manager.kv.layer_values(0),
        batch.block_tables, batch.cu_seqlens_q, batch.context_lens,
    )
    assert torch.isfinite(out).all()

    with pytest.raises(ValueError, match="padded there"):
        ops.paged_attention_gathered(
            torch.randn(4, 4, dim),
            manager.kv.layer_keys(0), manager.kv.layer_values(0),
            torch.tensor([[batch.block_tables[1, 0].item(), PADDING_BLOCK]], dtype=torch.int32),
            torch.tensor([0, 4], dtype=torch.int32),
            torch.tensor([8], dtype=torch.int32),  # claims twice the cache it has
        )


def test_the_slot_mapping_covers_this_iteration_only():
    """A decode step writes one slot; the chunk before it wrote its own."""
    manager = BlockManager(8, block_size=4)
    request = tokens(list(range(6)))
    manager.allocate(request, 4)
    first = ForwardBatch.from_scheduled([(request, 4)], manager=manager)
    request.advance(4)

    manager.allocate(request, 2)
    second = ForwardBatch.from_scheduled([(request, 2)], manager=manager)

    assert first.slot_mapping.tolist() == [0, 1, 2, 3]
    assert second.slot_mapping.tolist() == [4, 5], "the second chunk rewrote the first"
    assert second.block_tables.tolist() == [[0, 1]]


def test_match_returns_the_longest_cached_prefix():
    cache = PrefixCache(block_size=4)
    prompt = [10, 11, 12, 13, 20, 21, 22, 23]
    cache.insert(prompt, [7, 9])

    assert cache.match([10, 11, 12, 13, 99, 98, 97, 96]) == [7]
    assert cache.match(prompt) == [7, 9]
    assert cache.match([1, 2, 3, 4]) == []


def test_only_whole_blocks_are_cacheable():
    cache = PrefixCache(block_size=4)
    newly = cache.insert([1, 2, 3, 4, 5, 6, 7], [3, 8])
    assert newly == [3], "the partial second block must not register"
    assert cache.match([1, 2, 3, 4, 5, 6, 7]) == [3]


def test_a_duplicate_prefix_keeps_the_first_block():
    cache = PrefixCache(block_size=4)
    cache.insert([1, 2, 3, 4], [5])
    assert cache.insert([1, 2, 3, 4], [6]) == [], "the tree keeps the first copy"
    assert cache.match([1, 2, 3, 4]) == [5]


def test_eviction_orphans_the_subtree():
    cache = PrefixCache(block_size=4)
    cache.insert([1, 2, 3, 4, 5, 6, 7, 8], [1, 2])
    assert cache.num_cached_blocks == 2

    cache.evict(1)
    assert cache.match([1, 2, 3, 4]) == []
    assert cache.num_cached_blocks == 0
    cache.evict(2)  # already orphaned: harmless


def test_a_cached_block_is_reclaimed_by_the_free_list():
    pool = BlockPool(num_blocks=4)
    evicted: list[int] = []
    pool.on_evict = evicted.append

    block = pool.allocate()
    pool.mark_cached(block)
    pool.decref(block)  # back on the free list, but still cached

    reused = [pool.allocate() for _ in range(4)]
    assert block in reused
    assert evicted == [block]
    assert not pool.is_cached(block)


def test_acquiring_a_cached_block_skips_the_free_list():
    pool = BlockPool(num_blocks=4)
    block = pool.allocate()
    pool.mark_cached(block)
    pool.decref(block)

    pool.acquire_cached(block)  # a hit: hold it again without a fresh allocation

    assert pool.num_free == 3 and pool.ref_count(block) == 1
    assert pool.is_cached(block), "a reused cached block stays matchable"

    pool.acquire_cached(block)  # a second holder of a still-held prefix
    assert pool.ref_count(block) == 2
    pool.decref(block), pool.decref(block)
    pool.check_consistency()


@pytest.fixture
def caching_manager() -> Iterator[BlockManager]:
    manager = tiny_kv_manager(enable_prefix_caching=True)
    yield manager
    manager.pool.check_consistency()


def _fill_kv(manager: BlockManager, seq: Sequence) -> None:
    """A distinct key/value per position, so a wrong gather shows the wrong number."""
    table = manager.table(seq)
    for position in range(table.num_tokens):
        slot = table.physical_slot(position)
        block, offset = slot // manager.block_size, slot % manager.block_size
        value = float(seq.token_ids[position])
        manager.kv.keys[0, block, offset] = value
        manager.kv.values[0, block, offset] = -value


def test_a_shared_prefix_reuses_the_same_physical_pages(caching_manager: BlockManager):
    manager = caching_manager
    prompt = list(range(100, 124))  # 24 tokens -> 6 blocks of 4
    first = tokens(prompt)
    manager.maybe_apply_prefix_cache(first)  # cold: no hit
    manager.allocate(first)
    original_blocks = manager.table(first).block_ids
    _fill_kv(manager, first)
    allocated_cold = manager.blocks_allocated
    manager.free(first)  # caches the full blocks

    second = tokens(prompt)
    manager.maybe_apply_prefix_cache(second)
    manager.allocate(second)
    allocated_warm = manager.blocks_allocated - allocated_cold

    reused = manager.table(second).block_ids
    assert reused[:5] == original_blocks[:5], "a hit must reuse the same physical pages"
    assert allocated_warm < allocated_cold
    assert manager.cached_tokens == 20, "five of six blocks reused, one left to recompute"
    manager.free(second)


def test_reused_kv_is_bit_for_bit_what_was_written(caching_manager: BlockManager):
    manager = caching_manager
    prompt = list(range(200, 224))
    first = tokens(prompt)
    manager.maybe_apply_prefix_cache(first)
    manager.allocate(first)
    _fill_kv(manager, first)
    keys_before = manager.kv.keys.clone()
    manager.free(first)

    second = tokens(prompt)
    manager.maybe_apply_prefix_cache(second)
    manager.allocate(second)

    for position in range(manager.cached_tokens):
        slot = manager.table(second).physical_slot(position)
        block, offset = slot // manager.block_size, slot % manager.block_size
        assert manager.kv.keys[0, block, offset].equal(keys_before[0, block, offset])
    manager.free(second)


def test_a_full_match_still_leaves_a_token_to_forward(caching_manager: BlockManager):
    manager = caching_manager
    prompt = list(range(300, 316))  # exactly 4 blocks
    first = tokens(prompt)
    manager.maybe_apply_prefix_cache(first)
    manager.allocate(first)
    manager.free(first)

    second = tokens(prompt)
    reused = manager.maybe_apply_prefix_cache(second)

    assert reused == len(prompt) - manager.block_size
    assert second.num_uncomputed_tokens >= manager.block_size
    manager.allocate(second)
    manager.free(second)


def test_eviction_under_pressure_beats_out_of_blocks(caching_manager: BlockManager):
    """A pool full of cached-but-free pages admits a new request by reclaiming them."""
    manager = caching_manager
    for base in range(0, 64, 8):
        seq = tokens(list(range(base, base + 8)))
        manager.maybe_apply_prefix_cache(seq)
        manager.allocate(seq)
        manager.free(seq)

    assert manager.pool.num_free == manager.num_blocks, "all cached pages are free"
    assert manager.cache.num_cached_blocks == manager.num_blocks, "and all are cached"

    fresh = tokens(list(range(1000, 1040)))  # 40 tokens: needs the whole pool
    manager.maybe_apply_prefix_cache(fresh)
    manager.allocate(fresh)
    assert manager.table(fresh).num_tokens == 40
    manager.free(fresh)


def test_no_leaks_across_two_thousand_cached_requests(caching_manager: BlockManager):
    manager = caching_manager
    for round_index in range(2000):
        shared = round_index % 2 == 0
        base = 500 if shared else 1000 + round_index
        seq = tokens(list(range(base, base + 12)), max_tokens=2)
        manager.maybe_apply_prefix_cache(seq)
        manager.allocate(seq)
        manager.free(seq)

    manager.check_no_leaks()


def test_an_fp8_pool_stores_e4m3_and_halves_the_bytes():
    fp8 = PagedKvPool(1, 4, 4, 2, 8, dtype=torch.bfloat16, kv_dtype=torch.float8_e4m3fn)
    assert fp8.is_fp8 and fp8.keys.dtype is torch.float8_e4m3fn

    bf16_bytes = PagedKvPool.bytes_for(28, 100, 16, 8, 128, torch.bfloat16)
    fp8_bytes = PagedKvPool.bytes_for(28, 100, 16, 8, 128, torch.float8_e4m3fn)
    assert bf16_bytes == 2 * fp8_bytes


def test_fp8_write_and_gather_round_trip_within_tolerance():
    """Quantize on write, dequantize on gather, with explicit scales."""
    pool = PagedKvPool(
        1, 4, 4, 2, 8, dtype=torch.float32, kv_dtype=torch.float8_e4m3fn,
        k_scale=0.5, v_scale=0.25,
    )
    key = torch.randn(6, 2, 8)
    value = torch.randn(6, 2, 8)
    slots = torch.arange(6, dtype=torch.int32)

    pool.write(0, slots, key, value)
    keys, values = pool.gather(0, [0, 1], num_tokens=6)

    assert keys.dtype == torch.float32, "the gather dequantizes to the activation dtype"
    assert relative_error(keys.squeeze(0).permute(1, 0, 2), key) < FP8_MAX_ERROR
    assert relative_error(values.squeeze(0).permute(1, 0, 2), value) < FP8_MAX_ERROR


def test_fp8_copy_on_write_copies_raw_bytes():
    manager = BlockManager(
        num_blocks=4, block_size=4, num_layers=1, num_kv_heads=2, head_dim=8,
        dtype=torch.float32, kv_dtype=torch.float8_e4m3fn,
    )
    parent = tokens(list(range(6)))
    manager.allocate(parent)
    page = manager.table(parent).block_ids[1]
    manager.kv.write(
        0,
        torch.tensor([4, 5], dtype=torch.int32),
        torch.randn(2, 2, 8),
        torch.randn(2, 2, 8),
    )

    child = tokens(list(range(6)))
    manager.fork(parent, child)
    manager.append_slot(child)

    copied = manager.table(child).block_ids[1]
    assert torch.equal(
        manager.kv.keys[:, copied].view(torch.uint8), manager.kv.keys[:, page].view(torch.uint8)
    )
    manager.free(parent), manager.free(child)
    manager.check_no_leaks()


def test_admission_is_fcfs_and_bounded():
    scheduler = Scheduler(SchedulerConfig(max_sequences=2))
    first, second, third = make(4), make(4), make(4)
    scheduler.add_all([first, second, third])

    output = scheduler.schedule()

    assert output.sequences == [first, second]
    assert third.status is SequenceStatus.WAITING


def test_admission_stops_at_the_token_budget():
    scheduler = Scheduler(SchedulerConfig(max_batched_tokens=512, max_sequences=8, **WHOLE))
    scheduler.add_all([make(300), make(300)])
    assert scheduler.schedule().total_tokens == 300
    assert len(scheduler.waiting) == 1


def test_a_prompt_larger_than_the_budget_runs_alone_and_overruns_it():
    """Refusing it would deadlock the queue: the head-of-line stall chunking removes."""
    scheduler = Scheduler(SchedulerConfig(max_batched_tokens=512, **WHOLE))
    huge = make(2000)
    scheduler.add(huge)
    output = scheduler.schedule()
    assert output.total_tokens == 2000 and output.sequences == [huge]


def test_a_finished_sequence_is_replaced_in_the_same_iteration():
    """The point of continuous batching: no drain between one request and the next."""
    scheduler = Scheduler(SchedulerConfig(max_sequences=1))
    a, b = make(3, max_tokens=1), make(3, max_tokens=1)
    scheduler.add_all([a, b])

    prefill = scheduler.schedule()
    assert prefill.sequences == [a], "B must wait: only one slot"
    assert scheduler.commit(prefill, [99]) == [a]

    assert scheduler.schedule().sequences == [b], "B runs immediately, no idle iteration"


def test_a_long_prompt_is_split_into_chunks():
    """2000 tokens, 512 to an iteration: 512, 512, 512, 464."""
    scheduler = Scheduler(SchedulerConfig(max_batched_tokens=2048, chunk_size=512))
    sequence = make(2000, max_tokens=1)
    scheduler.add(sequence)

    counts = []
    while sequence.is_prefill():
        output = scheduler.schedule()
        counts.append(output.tokens_for(sequence))
        scheduler.commit(output)

    assert counts == [512, 512, 512, 464]
    assert sequence.num_computed_tokens == 2000


def test_a_chunk_is_bounded_by_budget_and_chunk_size():
    scheduler = Scheduler(SchedulerConfig(max_batched_tokens=100, chunk_size=512))
    scheduler.add(make(2000))
    assert scheduler.schedule().total_tokens == 100

    chunked = Scheduler(SchedulerConfig(max_batched_tokens=512, chunk_size=512))
    chunked.add(make(2000))
    assert chunked.schedule().total_tokens == 512, "the head-of-line stall is bounded"


def test_a_mid_prefill_chunk_takes_no_token():
    """Sampling from a position in the middle of a prompt would invent a token."""
    scheduler = Scheduler(SchedulerConfig(enable_chunked_prefill=True, chunk_size=4))
    sequence = make(10)
    scheduler.add(sequence)

    scheduler.commit(scheduler.schedule(), [777])

    assert sequence.output_token_ids == []
    assert sequence.num_computed_tokens == 4 and sequence.is_prefill()


def decoding_scheduler(config: SchedulerConfig, count: int) -> tuple[Scheduler, list[Sequence]]:
    """A scheduler with `count` sequences past their prompts and into decode."""
    scheduler = Scheduler(config)
    sequences = [make(2, max_tokens=8) for _ in range(count)]
    scheduler.add_all(sequences)
    scheduler.commit(scheduler.schedule(), [5] * count)
    assert all(not sequence.is_prefill() for sequence in sequences)
    return scheduler, sequences


def test_decodes_ride_along_with_a_prefill_chunk():
    """One pass carrying a 300-token chunk and three decodes: piggyback decoding."""
    scheduler, decoders = decoding_scheduler(
        SchedulerConfig(max_batched_tokens=1024, chunk_size=300), count=3
    )
    prompt = make(2000)
    scheduler.add(prompt)

    output = scheduler.schedule()

    assert output.total_tokens == 303 and output.num_decodes == 3
    assert [output.tokens_for(sequence) for sequence in decoders] == [1, 1, 1]
    assert output.tokens_for(prompt) == 300

    batch = output.batch()
    assert not batch.is_pure_decode
    assert batch.num_prefill_tokens == 300 and batch.total_tokens == 303


def test_a_decode_is_never_stalled_by_a_long_prefill():
    """A decode advances one token every iteration while a 2000-token prompt chunks."""
    scheduler, (decoder,) = decoding_scheduler(
        SchedulerConfig(max_batched_tokens=600, chunk_size=512), count=1
    )
    scheduler.add(make(2000, max_tokens=1))

    before = decoder.num_output_tokens
    for _ in range(3):
        output = scheduler.schedule()
        assert output.tokens_for(decoder) == 1, "the decode was left out of an iteration"
        scheduler.commit(output, [5] * len(output.scheduled))

    assert decoder.num_output_tokens == before + 3


def test_the_chunk_takes_what_the_decodes_leave():
    """Decodes are scheduled first, so the chunk shrinks rather than the budget growing."""
    scheduler, _ = decoding_scheduler(SchedulerConfig(max_batched_tokens=10, chunk_size=512), 4)
    prompt = make(2000)
    scheduler.add(prompt)

    output = scheduler.schedule()

    assert output.total_tokens == 10
    assert output.tokens_for(prompt) == 6


def test_positions_resume_where_the_previous_chunk_stopped():
    """The chunk-boundary bug, caught at the metadata rather than in the logits."""
    scheduler = Scheduler(SchedulerConfig(max_batched_tokens=64, chunk_size=8))
    sequence = make(24)
    scheduler.add(sequence)

    seen = []
    while sequence.is_prefill():
        output = scheduler.schedule()
        seen.append(output.batch().positions.tolist())
        scheduler.commit(output)

    assert seen == [list(range(0, 8)), list(range(8, 16)), list(range(16, 24))]


def test_preemption_requeues_at_the_front_and_re_prefills_over_output():
    scheduler = Scheduler()
    sequence = make(4)
    scheduler.add(sequence)
    scheduler.commit(scheduler.schedule(), [5])

    scheduler.preempt(sequence)

    assert sequence.status is SequenceStatus.PREEMPTED
    assert scheduler.waiting[0] is sequence
    assert scheduler.schedule().scheduled == [(sequence, 5)], "4 prompt + 1 emitted, recomputed"


def test_reset_for_recompute_refuses_to_leak_a_table():
    manager = BlockManager(num_blocks=8, block_size=4)
    sequence = make(4)
    sequence.set_status(SequenceStatus.RUNNING)
    manager.allocate(sequence)

    with pytest.raises(ValueError, match="still holds"):
        sequence.reset_for_recompute()
    manager.free(sequence)


def test_scheduler_output_and_config_contracts():
    assert SchedulerOutput().is_empty and SchedulerOutput().total_tokens == 0
    for kwargs in ({"max_batched_tokens": 0}, {"max_sequences": 0}, {"chunk_size": 0}):
        with pytest.raises(ValueError, match="must be >= 1"):
            SchedulerConfig(**kwargs)


@pytest.fixture
def tiny_model(tiny_qwen3):
    """The tiny cached model on the CPU: scheduling is arithmetic, no GPU needed."""
    return Qwen3Cached(config_from_hf(tiny_qwen3), weights_from_hf(tiny_qwen3), use_cuda=False)


def run_to_completion(scheduler: Scheduler, runner) -> dict[int, list[int]]:
    """Drive the engine loop until every request is done, as the engine itself does."""
    while scheduler.has_work:
        output = scheduler.schedule()
        assert not output.is_empty, "the scheduler stalled with work outstanding"
        logits = runner.execute(output)
        tokens_out = runner.sample_tokens(output, logits)
        for finished in scheduler.commit(output, tokens_out):
            runner.free(finished)
    return {sequence.seq_id: sequence.output_token_ids for sequence in scheduler.finished}


def alone(model, prompt: list[int], max_tokens: int) -> list[int]:
    """The reference: one request, one scheduler, nothing else in flight."""
    scheduler = Scheduler()
    scheduler.add(Sequence(prompt_token_ids=prompt, max_tokens=max_tokens, sampling_params=GREEDY))
    return next(iter(run_to_completion(scheduler, DenseModelRunner(model)).values()))


def test_one_sequence_matches_the_generate_loop(tiny_model):
    """Agree with `generate_ids_cached`, so the harness itself is trusted."""
    prompt = [3, 9, 4, 1, 7]
    expected = generate_ids_cached(tiny_model, torch.tensor([prompt]), max_tokens=8)

    assert alone(tiny_model, prompt, max_tokens=8) == expected[0, len(prompt) :].tolist()


def test_a_batched_run_is_token_identical_to_running_each_alone(tiny_model):
    """The invariant: six requests of different lengths, interleaved, unchanged tokens."""
    prompts = [[1, 2, 3], [5], [7, 7, 7, 7, 7, 7, 7], [2, 4], [9, 8, 7, 6], [1]]
    expected = {index: alone(tiny_model, prompt, 6) for index, prompt in enumerate(prompts)}

    scheduler = Scheduler(SchedulerConfig(max_batched_tokens=16, max_sequences=3))
    sequences = [Sequence(prompt_token_ids=p, max_tokens=6, sampling_params=GREEDY) for p in prompts]
    scheduler.add_all(sequences)

    got = run_to_completion(scheduler, DenseModelRunner(tiny_model))

    for index, sequence in enumerate(sequences):
        assert got[sequence.seq_id] == expected[index], f"prompt {index} changed under batching"


def test_a_preempted_sequence_produces_the_same_tokens(tiny_model):
    """Recomputation must be indistinguishable from never having been evicted."""
    prompt = [4, 2, 7, 1]
    expected = alone(tiny_model, prompt, max_tokens=6)

    scheduler = Scheduler()
    runner = DenseModelRunner(tiny_model)
    sequence = Sequence(prompt_token_ids=prompt, max_tokens=6, sampling_params=GREEDY)
    scheduler.add(sequence)

    for _ in range(2):
        output = scheduler.schedule()
        scheduler.commit(output, runner.sample_tokens(output, runner.execute(output)))
    scheduler.preempt(sequence)

    got = run_to_completion(scheduler, runner)

    assert got[sequence.seq_id] == expected


def test_a_chunked_and_piggybacked_run_is_token_identical(tiny_model):
    """Chunked prefills and decodes interleaved under a tight budget, tokens unchanged."""
    prompts = [torch.randint(0, 512, (n,)).tolist() for n in (33, 5, 17, 1, 40, 2)]
    expected = [alone(tiny_model, prompt, 5) for prompt in prompts]

    scheduler = Scheduler(SchedulerConfig(max_batched_tokens=12, max_sequences=4, chunk_size=8))
    sequences = [Sequence(prompt_token_ids=p, max_tokens=5, sampling_params=GREEDY) for p in prompts]
    scheduler.add_all(sequences)

    got = run_to_completion(scheduler, DenseModelRunner(tiny_model))

    for index, sequence in enumerate(sequences):
        assert got[sequence.seq_id] == expected[index], (
            f"prompt {index} ({len(prompts[index])} tokens) changed under chunking"
        )


def test_chunking_does_not_change_the_number_of_tokens_computed(tiny_model):
    """Every token goes through the model exactly once, whatever the chunk size."""
    prompt = torch.randint(0, 512, (25,)).tolist()

    for chunk_size in (1, 4, 25, 512):
        scheduler = Scheduler(SchedulerConfig(max_batched_tokens=64, chunk_size=chunk_size))
        sequence = Sequence(prompt_token_ids=prompt, max_tokens=4, sampling_params=GREEDY)
        scheduler.add(sequence)
        runner = DenseModelRunner(tiny_model)

        computed = 0
        while scheduler.has_work:
            output = scheduler.schedule()
            computed += output.total_tokens
            scheduler.commit(output, runner.sample_tokens(output, runner.execute(output)))

        assert computed == len(prompt) + 3, f"chunk size {chunk_size}"


class Engine:
    """The engine loop over a tiny model, without a tokenizer or a checkpoint.

    Exactly what `LLM.step` does — schedule, execute, sample, commit, free — spelled out
    so a change to the engine's loop is reflected in a test that reads like it.
    """

    def __init__(self, model, manager: BlockManager, **config) -> None:
        self.manager = manager
        self.scheduler = Scheduler(SchedulerConfig(**config), manager=manager)
        self.runner = PagedModelRunner(model, manager)
        self.preemptions = 0
        self.iterations = 0

    def run(self, sequences: list[Sequence], max_iterations: int = 500) -> dict[int, list[int]]:
        self.scheduler.add_all(sequences)
        while self.scheduler.has_work:
            self.iterations += 1
            assert self.iterations <= max_iterations, "the engine is not making progress"

            output = self.scheduler.schedule()
            logits = self.runner.execute(output)
            tokens_out = self.runner.sample_tokens(output, logits)
            self.preemptions += len(output.preempted)
            for finished in self.scheduler.commit(output, tokens_out):
                self.runner.free(finished)

        return {sequence.seq_id: sequence.output_token_ids for sequence in sequences}


PROMPTS = [[1, 2, 3], [5], [7] * 9, [2, 4], [9, 8, 7, 6, 5], [1]]
BLOCK_SIZE = 8  # small, so short sequences cross page boundaries several times


def paged_engine(tiny_qwen3, num_blocks: int = 64, **config):
    weights, model_config = weights_from_hf(tiny_qwen3), config_from_hf(tiny_qwen3)
    manager = BlockManager(
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        num_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=model_config.head_dim,
        dtype=model_config.dtype,
    )
    model = Qwen3Paged(model_config, weights, manager, use_cuda=False)
    return Engine(model, manager, **config), manager


def dense_reference(tiny_qwen3, prompts: list[list[int]], max_tokens: int) -> list[list[int]]:
    model = Qwen3Cached(config_from_hf(tiny_qwen3), weights_from_hf(tiny_qwen3), use_cuda=False)
    return [alone(model, prompt, max_tokens) for prompt in prompts]


def test_the_paged_engine_is_token_identical_to_the_dense_one(tiny_qwen3):
    """The serving layer against the cached model, on the same weights.

    Between the two: a ragged batch, a paged cache with sequences interleaved across
    pages, and attention that resolves every key's address through a block table.
    """
    expected = dense_reference(tiny_qwen3, PROMPTS, max_tokens=6)

    engine, manager = paged_engine(tiny_qwen3, max_batched_tokens=16, max_sequences=3)
    sequences = [Sequence(prompt_token_ids=p, max_tokens=6, sampling_params=GREEDY) for p in PROMPTS]

    got = engine.run(sequences)

    for index, sequence in enumerate(sequences):
        assert_tokens_equal(got[sequence.seq_id], expected[index],
                            msg=f"prompt {index} changed under paging")
    manager.check_no_leaks()


def test_a_small_pool_forces_preemption_and_changes_nothing(tiny_qwen3):
    """Six requests through a pool that cannot hold them all: a preempted sequence
    re-prefills over its prompt and its own output to exactly the same tokens."""
    expected = dense_reference(tiny_qwen3, PROMPTS, max_tokens=12)

    engine, manager = paged_engine(tiny_qwen3, num_blocks=5,
                                   max_batched_tokens=16, max_sequences=6)
    sequences = [Sequence(prompt_token_ids=p, max_tokens=12, sampling_params=GREEDY)
                 for p in PROMPTS]

    got = engine.run(sequences)

    assert engine.preemptions > 0, "this pool is too big to be testing preemption"
    for index, sequence in enumerate(sequences):
        assert_tokens_equal(got[sequence.seq_id], expected[index],
                            msg=f"prompt {index} changed under preemption")
    manager.check_no_leaks()


def test_a_preempted_sequence_gives_its_blocks_back_immediately(tiny_qwen3):
    engine, manager = paged_engine(tiny_qwen3, num_blocks=32, max_sequences=4)
    sequences = [Sequence(prompt_token_ids=p, max_tokens=6, sampling_params=GREEDY)
                 for p in PROMPTS[:2]]
    engine.scheduler.add_all(sequences)

    output = engine.scheduler.schedule()
    engine.scheduler.commit(output, engine.runner.sample_tokens(output, engine.runner.execute(output)))
    held = manager.num_blocks - manager.num_free_blocks

    engine.scheduler.preempt(sequences[1])

    assert manager.num_free_blocks > manager.num_blocks - held
    assert sequences[1].block_table is None
    assert sequences[1].status is SequenceStatus.PREEMPTED


def test_a_pool_too_small_for_one_sequence_says_so(tiny_qwen3):
    """Not a hang, and not a wrong answer: a message naming the fix."""
    engine, _manager = paged_engine(tiny_qwen3, num_blocks=1)
    with pytest.raises(OutOfBlocks, match="nothing can run"):
        engine.run([Sequence(prompt_token_ids=[7] * 9, max_tokens=4, sampling_params=GREEDY)])


def test_admission_waits_rather_than_preempting_for_a_new_request(tiny_qwen3):
    """Memory pressure preempts for a sequence already in flight, and merely postpones
    one that has not started: a queued request holds nothing."""
    engine, _manager = paged_engine(tiny_qwen3, num_blocks=4, max_sequences=8)
    running = Sequence(prompt_token_ids=[1] * 24, max_tokens=4, sampling_params=GREEDY)
    queued = Sequence(prompt_token_ids=[9] * 16, max_tokens=4, sampling_params=GREEDY)
    engine.scheduler.add_all([running, queued])

    first = engine.scheduler.schedule()

    assert first.sequences == [running]
    assert not first.preempted
    assert queued.status is SequenceStatus.WAITING


@pytest.mark.slow
def test_a_deterministic_stress_run_completes_and_leaks_nothing(tiny_qwen3):
    """120 requests of mixed sizes through a pool that forces steady preemption."""
    engine, manager = paged_engine(tiny_qwen3, num_blocks=12,
                                   max_batched_tokens=32, max_sequences=6)
    generator = torch.Generator().manual_seed(0)
    sequences = []
    for _index in range(120):
        length = int(torch.randint(1, 30, (1,), generator=generator))
        sequences.append(Sequence(
            prompt_token_ids=torch.randint(0, 512, (length,), generator=generator).tolist(),
            max_tokens=4, sampling_params=GREEDY,
        ))

    got = engine.run(sequences, max_iterations=5000)

    assert all(len(got[s.seq_id]) == 4 for s in sequences), "a request finished short"
    manager.check_no_leaks()
