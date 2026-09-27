"""prefix_cache.py: longest-prefix match, whole blocks only, first copy wins, eviction."""

from mini_vllm.prefix_cache import PrefixCache


def test_match_returns_the_longest_cached_prefix():
    cache = PrefixCache(block_size=4)
    prompt = [10, 11, 12, 13, 20, 21, 22, 23]
    cache.insert(prompt, [7, 9])

    assert cache.match([10, 11, 12, 13, 99, 98, 97, 96]) == [7]
    assert cache.match(prompt) == [7, 9]
    assert cache.match([*prompt, 1, 2]) == [7, 9], "a trailing partial block is not matched"
    assert cache.match([1, 2, 3, 4]) == []


def test_only_whole_blocks_are_cacheable():
    cache = PrefixCache(block_size=4)
    assert cache.insert([1, 2, 3, 4, 5, 6, 7], [3, 8]) == [3], "the partial block must not register"
    assert cache.match([1, 2, 3, 4, 5, 6, 7, 8]) == [3]


def test_a_duplicate_prefix_keeps_the_first_block():
    cache = PrefixCache(block_size=4)
    cache.insert([1, 2, 3, 4], [5])
    assert cache.insert([1, 2, 3, 4], [6]) == [], "the tree keeps the first copy"
    assert cache.match([1, 2, 3, 4]) == [5]


def test_eviction_orphans_the_subtree():
    cache = PrefixCache(block_size=4)
    cache.insert([1, 2, 3, 4, 5, 6, 7, 8], [1, 2])
    cache.insert([9, 9, 9, 9], [3])
    assert cache.num_cached_blocks == 3

    cache.evict(1)
    assert cache.match([1, 2, 3, 4, 5, 6, 7, 8]) == []
    assert cache.num_cached_blocks == 1, "block 2 hung under block 1, so it went too"
    assert cache.match([9, 9, 9, 9]) == [3], "an unrelated branch is untouched"
    cache.evict(2)  # already orphaned: nothing to do
