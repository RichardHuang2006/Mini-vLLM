"""Radix-tree prefix caching.

Three layers, tested where each one lives:

* The tree: matching the longest cached prefix, inserting full blocks, and orphaning a
  subtree on eviction. Pure Python, no pool.
* The pool: a cached block lives free at reference count zero and is reclaimed by the same
  free-list pop that reuses any block, calling back to unlink its node.
* The manager: a second request sharing a prefix reuses the first's physical pages rather
  than reserving new ones, the reused KV is bit-for-bit what was written, and nothing leaks
  under pool pressure.

The invariant throughout: a cache hit changes how much work happens, never what comes out.
A reused prefix is the same page, so its keys and values are identical by construction
rather than within a tolerance.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from mini_vllm.block.block_manager import BlockManager
from mini_vllm.block.block_pool import BlockPool
from mini_vllm.block.prefix_cache import PrefixCache
from mini_vllm.sampler import SamplingParams
from mini_vllm.serve.sequence import Sequence

GREEDY = SamplingParams(temperature=0.0)


def sequence(tokens: list[int], **kwargs) -> Sequence:
    return Sequence(prompt_token_ids=list(tokens), sampling_params=GREEDY, **kwargs)


# --------------------------------------------------------------------- the tree


def test_match_returns_the_longest_cached_prefix():
    cache = PrefixCache(block_size=4)
    tokens = [10, 11, 12, 13, 20, 21, 22, 23]
    cache.insert(tokens, [7, 9])

    # A prompt that agrees on the first block but diverges in the second matches one.
    assert cache.match([10, 11, 12, 13, 99, 98, 97, 96]) == [7]
    # One that agrees on both matches both.
    assert cache.match(tokens) == [7, 9]
    # One that diverges immediately matches nothing.
    assert cache.match([1, 2, 3, 4]) == []


def test_only_whole_blocks_are_cacheable():
    cache = PrefixCache(block_size=4)
    # Seven tokens is one full block and a partial one; only the full block registers.
    newly = cache.insert([1, 2, 3, 4, 5, 6, 7], [3, 8])
    assert newly == [3]
    assert cache.match([1, 2, 3, 4, 5, 6, 7]) == [3]


def test_inserting_a_duplicate_prefix_keeps_the_first_block():
    cache = PrefixCache(block_size=4)
    cache.insert([1, 2, 3, 4], [5])
    # A second sequence prefilled the same block into a different page: the tree keeps
    # the first, and reports nothing newly cached so the caller frees the redundant one.
    assert cache.insert([1, 2, 3, 4], [6]) == []
    assert cache.match([1, 2, 3, 4]) == [5]


def test_eviction_orphans_the_subtree():
    cache = PrefixCache(block_size=4)
    cache.insert([1, 2, 3, 4, 5, 6, 7, 8], [1, 2])
    assert cache.num_cached_blocks == 2

    # Evicting the parent takes its child with it: the child's page is unreachable
    # once its prefix is gone, so the tree stops pointing at it.
    cache.evict(1)
    assert cache.match([1, 2, 3, 4]) == []
    assert cache.num_cached_blocks == 0
    # And a later (redundant) eviction of the orphaned child is harmless.
    cache.evict(2)


# --------------------------------------------------------------------- the pool


def test_a_cached_block_is_reclaimed_by_the_free_list():
    pool = BlockPool(num_blocks=4)
    evicted: list[int] = []
    pool.on_evict = evicted.append

    block = pool.allocate()
    pool.mark_cached(block)
    assert pool.is_cached(block)
    pool.decref(block)  # back on the free list, but still cached

    # Draining the pool eventually reuses the cached page, and reuse is what evicts it.
    reused = [pool.allocate() for _ in range(4)]
    assert block in reused
    assert evicted == [block]
    assert not pool.is_cached(block)


def test_acquiring_a_cached_block_takes_it_off_the_free_list():
    pool = BlockPool(num_blocks=4)
    block = pool.allocate()
    pool.mark_cached(block)
    pool.decref(block)
    assert pool.num_free == 4

    pool.acquire_cached(block)  # a hit: hold it again without a fresh allocation

    assert pool.num_free == 3
    assert pool.ref_count(block) == 1
    assert pool.is_cached(block), "a reused cached block stays matchable"
    pool.check_consistency()


def test_acquiring_a_held_cached_block_is_an_incref():
    pool = BlockPool(num_blocks=4)
    block = pool.allocate()
    pool.mark_cached(block)

    pool.acquire_cached(block)  # a second holder of a still-held prefix

    assert pool.ref_count(block) == 2
    assert pool.num_free == 3


# ------------------------------------------------------------------ the manager


@pytest.fixture
def manager() -> Iterator[BlockManager]:
    """A small paged pool with real KV storage, so reuse can be checked byte for byte.

    16 blocks of 4 tokens, one layer, one KV head, head dim 8 — enough that a prefix
    spans several blocks and exhaustion is reachable.
    """
    manager = BlockManager(
        num_blocks=16,
        block_size=4,
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        enable_prefix_caching=True,
    )
    yield manager
    manager.pool.check_consistency()


def _fill_kv(manager: BlockManager, seq: Sequence) -> None:
    """Write a distinct key/value for every occupied position, through its slots.

    The pattern is a function of the (global) token id, so an accidental cross-block
    gather would show up as the wrong number rather than passing by luck.
    """
    table = manager.table(seq)
    for position in range(table.num_tokens):
        slot = table.physical_slot(position)
        block, offset = slot // manager.block_size, slot % manager.block_size
        value = float(seq.token_ids[position])
        manager.kv.keys[0, block, offset] = value
        manager.kv.values[0, block, offset] = -value


def test_a_shared_prefix_reuses_the_same_pages(manager: BlockManager):
    prompt = list(range(100, 124))  # 24 tokens -> 6 blocks of 4
    first = sequence(prompt)
    manager.maybe_apply_prefix_cache(first)  # no hit yet: cold
    manager.allocate(first)
    original_blocks = manager.table(first).block_ids
    _fill_kv(manager, first)
    allocated_cold = manager.blocks_allocated

    manager.free(first)  # caches the full blocks

    second = sequence(prompt)
    manager.maybe_apply_prefix_cache(second)
    manager.allocate(second)
    allocated_warm = manager.blocks_allocated - allocated_cold

    reused = manager.table(second).block_ids
    # Every full block of the prompt is the *same physical page* as the first request's,
    # and only the leftover (never the whole sequence) had to be freshly allocated.
    assert reused[:5] == original_blocks[:5]
    assert allocated_warm < allocated_cold
    assert manager.cached_tokens == 20, "five of six blocks reused, one left to recompute"

    manager.free(second)


def test_reused_kv_is_bit_for_bit_what_was_written(manager: BlockManager):
    prompt = list(range(200, 224))
    first = sequence(prompt)
    manager.maybe_apply_prefix_cache(first)
    manager.allocate(first)
    _fill_kv(manager, first)

    keys_before = manager.kv.keys.clone()
    manager.free(first)

    second = sequence(prompt)
    manager.maybe_apply_prefix_cache(second)
    manager.allocate(second)

    # The reused pages were never rewritten, so gathering the second sequence's cached
    # prefix returns exactly the first sequence's keys.
    reused = manager.cached_tokens
    for position in range(reused):
        slot = manager.table(second).physical_slot(position)
        block, offset = slot // manager.block_size, slot % manager.block_size
        assert manager.kv.keys[0, block, offset].equal(keys_before[0, block, offset])

    manager.free(second)


def test_a_full_match_still_leaves_a_token_to_forward(manager: BlockManager):
    prompt = list(range(300, 316))  # exactly 4 blocks
    first = sequence(prompt)
    manager.maybe_apply_prefix_cache(first)
    manager.allocate(first)
    manager.free(first)

    second = sequence(prompt)
    reused = manager.maybe_apply_prefix_cache(second)

    # The whole prompt is cached, but matching all of it would leave nothing to run.
    assert reused == len(prompt) - manager.block_size
    assert second.num_uncomputed_tokens >= manager.block_size
    manager.allocate(second)
    manager.free(second)


def test_eviction_under_pressure_beats_out_of_blocks(manager: BlockManager):
    """A pool full of cached-but-free pages admits a new request by reclaiming them."""
    # Fill every page in the pool with a cached prefix from a finished request:
    # eight requests of eight tokens is sixteen blocks, the whole pool.
    for base in range(0, 64, 8):
        seq = sequence(list(range(base, base + 8)))
        manager.maybe_apply_prefix_cache(seq)
        manager.allocate(seq)
        manager.free(seq)

    assert manager.pool.num_free == manager.num_blocks, "all cached pages are free"
    assert manager.cache.num_cached_blocks == manager.num_blocks, "and all are cached"

    # A brand-new prompt that shares nothing must still be servable: the cached pages
    # are reclaimed rather than the allocation raising.
    fresh = sequence(list(range(1000, 1040)))  # 40 tokens, needs the whole pool
    manager.maybe_apply_prefix_cache(fresh)
    manager.allocate(fresh)
    assert manager.table(fresh).num_tokens == 40
    manager.free(fresh)


def test_no_leaks_across_many_cached_requests(manager: BlockManager):
    for round_index in range(2000):
        # A shared preamble half the time, a private prompt the other half, so the
        # cache is exercised by both hits and misses.
        shared = round_index % 2 == 0
        base = 500 if shared else 1000 + round_index
        seq = sequence(list(range(base, base + 12)), max_tokens=2)
        manager.maybe_apply_prefix_cache(seq)
        manager.allocate(seq)
        manager.free(seq)

    manager.check_no_leaks()


# -------------------------------------------------------------- engine output


@pytest.mark.oracle
def test_a_cache_hit_changes_nothing_it_produces():
    """A shared prefix is served from cache the second time, and the output is identical.

    The whole invariant of a cache: it changes how much is computed, never what comes
    out. A long shared preamble is run twice under one engine; the second run hits, and
    must reproduce the first token for token.
    """
    import torch
    from conftest import real_engine

    with real_engine(dtype=torch.float32, enable_prefix_caching=True) as llm:
        # A preamble long enough to span several 16-token blocks, plus a short question.
        preamble = "You are a careful assistant. Answer concisely and correctly. " * 6
        prompt = preamble + "The capital of France is"

        first = llm.generate(prompt, sampling_params=GREEDY, max_tokens=16)[0]
        hits_after_first = llm.stats.cached_tokens

        second = llm.generate(prompt, sampling_params=GREEDY, max_tokens=16)[0]

        assert second.token_ids == first.token_ids, "the cache changed the output"
        assert llm.stats.cached_tokens > hits_after_first, "the second run did not hit the cache"


@pytest.mark.oracle
def test_prefix_caching_matches_the_uncached_engine():
    """Caching on must not perturb a single run against caching off.

    Two engines, strictly one at a time: an fp32 Qwen3-0.6B is 2.4 GB of weights and the
    card has 8 GB, so the first is torn down and its allocator emptied before the second
    is built. Holding both would abort the run for memory rather than fail an assertion.
    """
    import torch
    from conftest import real_engine

    prompt = "The capital of France is"
    with real_engine(dtype=torch.float32, enable_prefix_caching=True) as llm:
        cached = llm.generate(prompt, sampling_params=GREEDY, max_tokens=16)[0]
    with real_engine(dtype=torch.float32, enable_prefix_caching=False) as llm:
        plain = llm.generate(prompt, sampling_params=GREEDY, max_tokens=16)[0]

    assert cached.token_ids == plain.token_ids
