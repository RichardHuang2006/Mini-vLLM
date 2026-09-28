"""paged_kv_cache.py: refcounts and slot arithmetic, copy-on-write, prefix reuse, fp8 pages,
and attention through the pool against a dense cache."""

import mlx.core as mx
import pytest
from utils import assert_allclose, requires_metal, with_and_without_metal

from mini_vllm.attention import scaled_dot_product_attention_grouped
from mini_vllm.batch import ForwardBatch
from mini_vllm.paged_kv_cache import BlockManager, BlockPool, BlockTable, PagedKvCache, PagedKvPool
from mini_vllm.quantize import dequantize_fp8
from mini_vllm.scheduler import Request


def request(token_ids) -> Request:
    return Request(prompt_token_ids=list(token_ids))


def make_manager(num_blocks=8, block_size=4, prefix_caching=False, num_layers=1, num_kv_heads=1,
                 head_dim=1, **pool) -> BlockManager:
    kv = PagedKvPool(num_layers, num_blocks, block_size, num_kv_heads, head_dim, **pool)
    return BlockManager(kv, enable_prefix_caching=prefix_caching)


def assert_no_leaks(manager: BlockManager) -> None:
    pool = manager.pool
    assert pool.num_free == pool.num_blocks, f"leaked: {[b for b, c in enumerate(pool.ref_counts) if c]}"
    assert sorted(pool.free) == list(range(pool.num_blocks)), "the free list holds a block twice"


# --- the block pool -------------------------------------------------------------------------


def test_refcounts_allow_sharing_without_capacity():
    """Two holders of one block cost one block."""
    pool = BlockPool(num_blocks=4)
    block = pool.allocate()
    pool.incref(block)

    assert pool.ref_counts[block] == 2 and pool.num_free == 3
    assert not pool.decref(block), "a holder remains"
    assert pool.decref(block), "the last holder frees it"
    assert pool.num_free == 4


def test_the_free_list_is_fifo_so_use_after_free_fails_loudly():
    pool = BlockPool(num_blocks=3)
    first = pool.allocate()
    pool.decref(first)

    # The other blocks are handed out before the freed one is recycled.
    assert [pool.allocate() for _ in range(3)] == [1, 2, first]


def test_a_cached_block_is_reclaimed_by_the_free_list():
    pool = BlockPool(num_blocks=4)
    evicted = []
    pool.on_evict = evicted.append

    block = pool.allocate()
    pool.cached[block] = True
    pool.decref(block)  # back on the free list, still cached

    assert block in [pool.allocate() for _ in range(4)]
    assert evicted == [block] and not pool.cached[block]


def test_acquiring_a_cached_block_skips_the_free_list():
    pool = BlockPool(num_blocks=4)
    block = pool.allocate()
    pool.cached[block] = True
    pool.decref(block)

    pool.acquire_cached(block)  # a hit on a free cached block: no fresh allocation
    assert pool.num_free == 3 and pool.ref_counts[block] == 1 and pool.cached[block]

    pool.acquire_cached(block)  # a second holder of a prefix still in use
    assert pool.ref_counts[block] == 2


# --- block tables ---------------------------------------------------------------------------


def test_the_slot_arithmetic():
    """position -> (block, offset) -> flat slot, through an out-of-order table."""
    table = BlockTable(block_size=4, block_ids=[7, 2, 9], num_tokens=10)
    assert table.slots(0, 10) == [28, 29, 30, 31, 8, 9, 10, 11, 36, 37]
    assert table.slots(5, 6) == [9]


def test_capacity_and_occupancy_are_distinct():
    table = BlockTable(block_size=4, block_ids=[1, 2], num_tokens=6)
    assert table.num_slots == 8 and table.num_empty_slots == 2
    assert table.blocks_needed_for(2) == 0, "the partial block still has room"
    assert table.blocks_needed_for(3) == 1
    assert table.blocks_needed_for(7) == 2


def test_a_copied_table_is_independent():
    table = BlockTable(4, [3, 5], 6)
    forked = table.copy()
    forked.append_block(9)
    assert table.block_ids == [3, 5] and forked.block_ids == [3, 5, 9]


# --- the block manager ----------------------------------------------------------------------


def test_allocate_extend_free_round_trip():
    manager = make_manager()
    r = request(range(6))

    manager.allocate(r, 6)
    assert len(r.block_table.block_ids) == 2 and manager.pool.num_free == 6
    manager.allocate(r, 1)  # fits the partial block
    assert manager.pool.num_free == 6
    manager.allocate(r, 2)  # crosses the boundary
    assert len(r.block_table.block_ids) == 3

    assert manager.free(r) == 3 and r.block_table is None
    assert_no_leaks(manager)


def test_a_chunked_prefill_extends_the_same_table():
    manager = make_manager()
    r = request(range(10))
    for chunk in (4, 4, 2):
        manager.allocate(r, chunk)

    assert r.block_table.num_tokens == 10 and len(set(r.block_table.block_ids)) == 3
    assert manager.slots(r, 2) == r.block_table.slots(8, 10), "slots name the newest tokens"


def test_forking_allocates_nothing():
    """Sharing a prefix of any length is free in blocks."""
    manager = make_manager()
    parent, child = request(range(12)), request(range(12))
    manager.allocate(parent, 12)
    free_before = manager.pool.num_free

    manager.fork(parent, child)

    assert manager.pool.num_free == free_before
    assert child.block_table.block_ids == parent.block_table.block_ids
    assert all(manager.pool.ref_counts[b] == 2 for b in parent.block_table.block_ids)
    manager.free(parent), manager.free(child)
    assert_no_leaks(manager)


def test_writing_after_a_fork_copies_exactly_one_page():
    """Blocks 0 and 1 are full and stay shared; block 2 is the one both would write into."""
    manager = make_manager()
    parent, child = request(range(10)), request(range(10))
    manager.allocate(parent, 10)
    shared = list(parent.block_table.block_ids)
    manager.fork(parent, child)
    free_before = manager.pool.num_free

    manager.allocate(child, 1)

    mine = child.block_table.block_ids
    assert parent.block_table.block_ids == shared, "the parent's mapping moved"
    assert mine[:2] == shared[:2], "a full block was copied for no reason"
    assert mine[2] != shared[2], "the shared partial block was written in place"
    assert manager.pool.num_free == free_before - 1, "more than one page was copied"
    assert manager.pool.ref_counts[shared[2]] == 1 and manager.pool.ref_counts[shared[0]] == 2
    manager.free(parent), manager.free(child)
    assert_no_leaks(manager)


@pytest.mark.parametrize("fp8", [False, True])
def test_the_copy_carries_the_cached_keys_and_values(fp8):
    """Copy-on-write moves data, not just ids; otherwise a fork attends over a blank page.
    An fp8 page is copied as its raw e4m3 bytes."""
    manager = make_manager(num_blocks=4, num_layers=2, num_kv_heads=2, head_dim=8, dtype=mx.float32, fp8=fp8)
    parent, child = request(range(6)), request(range(6))
    manager.allocate(parent, 6)
    for layer in range(2):
        slots = mx.array(parent.block_table.slots(0, 6))
        manager.kv.write(layer, slots, mx.random.normal((6, 2, 8)), mx.random.normal((6, 2, 8)))

    manager.fork(parent, child)
    manager.allocate(child, 1)

    page, copied = parent.block_table.block_ids[1], child.block_table.block_ids[1]
    assert copied != page
    for layer in range(2):
        keys, values = manager.kv.pages(layer)
        assert mx.array_equal(keys[copied], keys[page]).item()
        assert mx.array_equal(values[copied], values[page]).item()


def test_a_fork_whose_last_block_is_full_needs_no_copy():
    manager = make_manager()
    parent, child = request(range(8)), request(range(8))
    manager.allocate(parent, 8)
    manager.fork(parent, child)
    free_before = manager.pool.num_free

    manager.allocate(child, 1)

    assert manager.pool.num_free == free_before - 1, "one new block, and no copy"
    assert child.block_table.block_ids[:2] == parent.block_table.block_ids


def test_admission_control_counts_the_copy():
    """A shared partial page costs a block to write into, and blocks_needed says so."""
    manager = make_manager()
    parent, child, private = request(range(6)), request(range(6)), request(range(6))
    manager.allocate(parent, 6)
    manager.fork(parent, child)
    manager.allocate(private, 6)

    assert manager.blocks_needed(private, 1) == 0, "a private partial page is free to extend"
    assert manager.blocks_needed(child, 1) == 1, "a shared one costs the copy"
    assert manager.blocks_needed(request(range(9)), 9) == 3, "a request with no table yet"


# --- prefix reuse through the manager -------------------------------------------------------


def serve(manager: BlockManager, r: Request) -> int:
    """What the scheduler does for one request, minus the forward pass: match, allocate
    what is left, free. Returns how many tokens the prefix cache supplied."""
    reused = manager.apply_prefix_cache(r)
    manager.allocate(r, r.num_uncomputed_tokens)
    r.num_computed_tokens = len(r)
    manager.free(r)
    return reused


def test_a_shared_prefix_reuses_the_same_physical_pages():
    manager = make_manager(num_blocks=16, prefix_caching=True)
    prompt = list(range(100, 124))  # 24 tokens: 6 blocks of 4
    first = request(prompt)
    manager.allocate(first, 24)
    original = list(first.block_table.block_ids)
    manager.free(first)  # caches the full blocks

    second = request(prompt)
    reused = manager.apply_prefix_cache(second)

    assert reused == 20, "five of six blocks reused; the last is recomputed so it has logits"
    assert second.num_computed_tokens == 20
    assert second.block_table.block_ids == original[:5], "a hit must reuse the same physical pages"
    manager.allocate(second, 4)
    manager.free(second)
    assert_no_leaks(manager)


def test_reused_kv_is_bit_for_bit_what_was_written():
    manager = make_manager(num_blocks=16, prefix_caching=True, num_kv_heads=2, head_dim=8, dtype=mx.float32)
    prompt = list(range(200, 224))
    first = request(prompt)
    manager.allocate(first, 24)
    written = mx.random.normal((24, 2, 8))
    manager.kv.write(0, mx.array(first.block_table.slots(0, 24)), written, -written)
    manager.free(first)

    second = request(prompt)
    manager.apply_prefix_cache(second)
    manager.allocate(second, second.num_uncomputed_tokens)

    slots = mx.array(second.block_table.slots(0, 20))
    assert mx.array_equal(manager.kv.keys[0][slots], written[:20]).item()
    assert mx.array_equal(manager.kv.values[0][slots], -written[:20]).item()


def test_a_full_match_still_leaves_a_token_to_forward():
    manager = make_manager(num_blocks=16, prefix_caching=True)
    prompt = list(range(300, 316))  # exactly 4 blocks
    serve(manager, request(prompt))

    second = request(prompt)
    assert manager.apply_prefix_cache(second) == 12
    assert second.num_uncomputed_tokens == 4


def test_eviction_under_pressure_reclaims_cached_pages():
    """A pool full of cached-but-free pages admits a new request by reusing them."""
    manager = make_manager(num_blocks=10, prefix_caching=True)
    for base in range(0, 64, 8):
        serve(manager, request(range(base, base + 8)))

    assert manager.pool.num_free == 10 and manager.cache.num_cached_blocks == 10

    fresh = request(range(1000, 1040))  # 40 tokens: the whole pool
    manager.allocate(fresh, 40)
    assert manager.cache.num_cached_blocks == 0, "every cached page was reclaimed"
    manager.free(fresh)


def test_no_leaks_across_two_thousand_cached_requests():
    manager = make_manager(num_blocks=16, prefix_caching=True)
    hits = 0
    for index in range(2000):
        base = 500 if index % 2 == 0 else 1000 + index
        hits += serve(manager, request(range(base, base + 12))) > 0

    assert hits == 999, "every repeat of the shared prompt after the first should hit"
    assert_no_leaks(manager)


# --- fp8 pages ------------------------------------------------------------------------------


def test_an_fp8_pool_stores_e4m3_bits_in_half_the_bytes():
    bf16 = PagedKvPool(2, 8, 16, 8, 128, dtype=mx.bfloat16)
    fp8 = PagedKvPool(2, 8, 16, 8, 128, dtype=mx.bfloat16, fp8=True)
    assert fp8.keys[0].dtype == mx.uint8
    assert sum(a.nbytes for a in bf16.keys + bf16.values) == 2 * sum(a.nbytes for a in fp8.keys + fp8.values)


def test_fp8_pages_round_trip_through_their_scales():
    pool = PagedKvPool(1, 4, 4, 2, 8, dtype=mx.float32, fp8=True, k_scale=0.5, v_scale=0.25)
    key, value = mx.random.normal((6, 2, 8)), mx.random.normal((6, 2, 8))
    pool.write(0, mx.arange(6), key, value)

    keys, values = (pages.reshape(-1, 2, 8)[:6] for pages in pool.pages(0))
    # e4m3 rounds to within 2^-4 relative, plus half its 2^-9 subnormal step times the scale.
    for stored, original, scale in ((keys, key, 0.5), (values, value, 0.25)):
        back = dequantize_fp8(stored, scale, mx.float32)
        assert mx.all(mx.abs(back - original) <= 2**-4 * mx.abs(original) + scale * 2**-10).item()


# --- attention through the pool -------------------------------------------------------------

HEADS, KV_HEADS, DIM = 4, 2, 8


def nan_manager(num_blocks=16, block_size=4) -> BlockManager:
    """A pool filled with NaN, so a read of any slot that was not written poisons the output."""
    manager = make_manager(num_blocks, block_size, num_kv_heads=KV_HEADS, head_dim=DIM, dtype=mx.float32)
    kv = manager.kv
    kv.keys = [mx.full(pool.shape, mx.nan) for pool in kv.keys]
    kv.values = [mx.full(pool.shape, mx.nan) for pool in kv.values]
    return manager


def attend(manager: BlockManager, scheduled, q, k, v, use_metal=False) -> mx.array:
    """Allocate, build the batch, and attend through layer 0 on [T, H, D] inputs."""
    for r, count in scheduled:
        manager.allocate(r, count)
    batch = ForwardBatch.from_scheduled(scheduled, manager)
    cache = PagedKvCache(manager.kv, 0, batch)
    as_row = lambda x: x.swapaxes(0, 1)[None]  # noqa: E731
    out = cache.attend(as_row(q), as_row(k), as_row(v), use_metal)
    for r, count in scheduled:
        r.num_computed_tokens += count
    return out[0].swapaxes(0, 1)


def dense(q, k, v) -> mx.array:
    """Causal attention over one sequence's contiguous [S, H, D] history."""
    out = scaled_dot_product_attention_grouped(
        q.swapaxes(0, 1), k.swapaxes(0, 1), v.swapaxes(0, 1), mask="causal"
    )
    return out.swapaxes(0, 1)


def qkv(length):
    return (
        mx.random.normal((length, HEADS, DIM)),
        mx.random.normal((length, KV_HEADS, DIM)),
        mx.random.normal((length, KV_HEADS, DIM)),
    )


@with_and_without_metal
def test_attending_through_the_pool_matches_a_dense_cache(use_metal):
    manager = nan_manager()
    q, k, v = qkv(10)
    assert_allclose(attend(manager, [(request(range(10)), 10)], q, k, v, use_metal), dense(q, k, v))


def test_a_shuffled_block_table_changes_nothing():
    """With an unshuffled table, logical and physical order coincide and almost any
    indexing bug looks right."""
    manager = nan_manager()
    r = request(range(12))
    manager.allocate(r, 12)
    ids = list(r.block_table.block_ids)
    for index, block_id in enumerate(reversed(ids)):
        r.block_table.replace_block(index, block_id)
    r.block_table.num_tokens = 0  # re-reserve the same 12 slots through the shuffled table

    q, k, v = qkv(12)
    assert_allclose(attend(manager, [(r, 12)], q, k, v), dense(q, k, v))


@with_and_without_metal
def test_a_mixed_batch_matches_each_sequence_alone(use_metal):
    """Two decodes beside a fresh prefill in one call: the shape the paged kernel serves.
    Block size 4 against lengths 7 and 5, so both decodes land mid-page."""
    manager = nan_manager()
    old = [request(range(7)), request(range(5))]
    history = {}
    for r in old:
        q, k, v = qkv(len(r))
        attend(manager, [(r, len(r))], q, k, v, use_metal)
        history[r.request_id] = (k, v)
        r.output_token_ids.append(9)

    fresh = request(range(6))
    scheduled = [(old[0], 1), (old[1], 1), (fresh, 6)]
    q, k, v = qkv(8)
    together = attend(manager, scheduled, q, k, v, use_metal)

    rows = {old[0].request_id: slice(0, 1), old[1].request_id: slice(1, 2), fresh.request_id: slice(2, 8)}
    for r, _ in scheduled:
        mine = rows[r.request_id]
        past_k, past_v = history.get(r.request_id, (k[:0], v[:0]))
        expected = dense(q[mine], mx.concatenate([past_k, k[mine]]), mx.concatenate([past_v, v[mine]]))
        assert_allclose(together[mine], expected)


@requires_metal
@pytest.mark.parametrize("fp8", [False, True])
def test_the_metal_write_is_the_pure_write_in_place(fp8):
    """Both pools start from the same random contents, so a slot the kernel should not have
    touched would show. fp8 pages must match to the bit: the kernel quantizes as mx.to_fp8 does."""
    scales = {"k_scale": 0.5, "v_scale": 0.25}
    pools = [PagedKvPool(2, 8, 4, 2, 8, dtype=mx.bfloat16, fp8=fp8, **scales) for _ in range(2)]
    start = mx.random.randint(0, 255, pools[0].keys[0].shape).astype(pools[0].keys[0].dtype)
    for pool in pools:
        pool.keys = [start + 0 for _ in pool.keys]
        pool.values = [start + 1 for _ in pool.values]

    slots = mx.array([5, 17, 3, 30], dtype=mx.int32)
    key = mx.random.normal((4, 2, 8)).astype(mx.bfloat16)
    value = mx.random.normal((4, 2, 8)).astype(mx.bfloat16)
    pools[0].write(1, slots, key, value)
    pools[1].write(1, slots, key, value, use_metal=True)

    for layer in range(2):
        assert mx.array_equal(pools[1].keys[layer], pools[0].keys[layer]).item()
        assert mx.array_equal(pools[1].values[layer], pools[0].values[layer]).item()
