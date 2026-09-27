"""kv_cache.py: the dense cache appends like concatenation."""

import mlx.core as mx

from mini_vllm.kv_cache import KvFullCache


def test_the_dense_cache_appends_like_concat():
    cache = KvFullCache()
    first = mx.random.normal((1, 2, 3, 8))
    second = mx.random.normal((1, 2, 1, 8))

    keys, values = cache.update_and_fetch(first, -first)
    assert cache.offset == 3 and keys.shape[-2] == 3

    keys, values = cache.update_and_fetch(second, -second)
    assert cache.offset == 4
    assert mx.array_equal(keys, mx.concatenate([first, second], axis=-2)).item()
    assert mx.array_equal(values, -keys).item()
