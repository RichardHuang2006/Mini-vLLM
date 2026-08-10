"""Step 2.1 — the dense KV cache.

The oracle is `torch.cat`: whatever order tokens arrive in, the cache must hold
exactly what concatenating them all at once would have produced. That is the whole
correctness claim, and it is what lets Step 2.2 trust the cached model.
"""

from __future__ import annotations

import pytest
import torch
from conftest import assert_allclose

from mini_vllm.kv_cache import DenseKvCache, KvCache

BATCH, KV_HEADS, HEAD_DIM = 2, 2, 32


def kv(length: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.randn(BATCH, KV_HEADS, length, HEAD_DIM),
        torch.randn(BATCH, KV_HEADS, length, HEAD_DIM),
    )


# --------------------------------------------------------------------- interface


def test_dense_cache_satisfies_the_interface():
    """The ABC is the contract the paged cache will have to meet in Step 4.7."""
    assert isinstance(DenseKvCache(), KvCache)


def test_the_base_class_cannot_be_instantiated():
    with pytest.raises(TypeError):
        KvCache()


def test_starts_empty():
    cache = DenseKvCache()
    assert cache.offset == 0
    assert cache.keys is None and cache.values is None


# ------------------------------------------------------------------ one update


def test_first_update_returns_what_it_was_given():
    cache = DenseKvCache()
    key, value = kv(5)

    full_key, full_value, offset = cache.update_and_fetch(key, value)

    assert offset == 0, "the first write lands at position 0"
    assert_allclose(full_key, key)
    assert_allclose(full_value, value)
    assert cache.offset == 5


def test_returned_offset_is_the_write_position_not_the_new_length():
    """The distinction that keeps the causal mask honest.

    `S == offset + L` must hold, so `offset` is the length *before* the call. If
    this returned the post-call length instead, a mask built as `(L, offset + L)`
    would be `L` columns too wide and a token could attend past its own position.
    """
    cache = DenseKvCache()
    cache.update_and_fetch(*kv(7))

    full_key, _full_value, offset = cache.update_and_fetch(*kv(3))

    assert offset == 7
    assert full_key.shape[-2] == offset + 3 == 10
    assert cache.offset == 10


# ---------------------------------------------------- against a single concat


@pytest.mark.parametrize("steps", [1, 2, 8, 33])
def test_appending_one_token_at_a_time_matches_one_concat(steps):
    """The plan's criterion: `S` single-token appends equal one `concat`."""
    pieces = [kv(1) for _ in range(steps)]

    cache = DenseKvCache()
    for key, value in pieces:
        full_key, full_value, _offset = cache.update_and_fetch(key, value)

    assert_allclose(full_key, torch.cat([k for k, _ in pieces], dim=-2))
    assert_allclose(full_value, torch.cat([v for _, v in pieces], dim=-2))
    assert cache.offset == steps


def test_prefill_then_decode_matches_one_concat():
    """The shape generation actually produces: one long write, then many of size 1."""
    prompt = kv(12)
    steps = [kv(1) for _ in range(6)]

    cache = DenseKvCache()
    cache.update_and_fetch(*prompt)
    for key, value in steps:
        full_key, full_value, _ = cache.update_and_fetch(key, value)

    assert_allclose(full_key, torch.cat([prompt[0]] + [k for k, _ in steps], dim=-2))
    assert_allclose(full_value, torch.cat([prompt[1]] + [v for _, v in steps], dim=-2))
    assert cache.offset == 18


def test_chunked_writes_match_one_concat():
    """Ragged chunk sizes, as chunked prefill will produce in Phase 4."""
    chunks = [kv(n) for n in (5, 1, 1, 9, 2)]

    cache = DenseKvCache()
    for key, value in chunks:
        full_key, full_value, _ = cache.update_and_fetch(key, value)

    assert_allclose(full_key, torch.cat([k for k, _ in chunks], dim=-2))
    assert cache.offset == 18


def test_offsets_form_a_running_total():
    cache = DenseKvCache()
    observed = [cache.update_and_fetch(*kv(n))[2] for n in (4, 3, 1, 1, 6)]
    assert observed == [0, 4, 7, 8, 9]
    assert cache.offset == 15


# ---------------------------------------------------------------------- reset


def test_reset_makes_the_cache_reusable():
    cache = DenseKvCache()
    cache.update_and_fetch(*kv(9))

    cache.reset()
    assert cache.offset == 0 and cache.keys is None

    key, value = kv(4)
    full_key, _full_value, offset = cache.update_and_fetch(key, value)
    assert offset == 0
    assert_allclose(full_key, key)


# ------------------------------------------------------------------ validation


def test_rejects_wrong_rank():
    cache = DenseKvCache()
    with pytest.raises(ValueError, match="expected B x H_k x L x D"):
        cache.update_and_fetch(torch.randn(2, 4, 8), torch.randn(2, 4, 8))


def test_rejects_mismatched_key_and_value():
    cache = DenseKvCache()
    with pytest.raises(ValueError, match="same shape"):
        cache.update_and_fetch(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 5, 4))


def test_rejects_an_append_that_changes_a_non_sequence_dimension():
    """A head-count or head-dim change means the caller mixed up two layers."""
    cache = DenseKvCache()
    cache.update_and_fetch(*kv(3))

    with pytest.raises(ValueError, match="only the sequence dimension may differ"):
        cache.update_and_fetch(
            torch.randn(BATCH, KV_HEADS + 1, 1, HEAD_DIM),
            torch.randn(BATCH, KV_HEADS + 1, 1, HEAD_DIM),
        )


# ----------------------------------------------------------------- properties


def test_cached_tensors_keep_dtype_and_device():
    cache = DenseKvCache()
    key, value = kv(3)
    full_key, _, _ = cache.update_and_fetch(key.bfloat16(), value.bfloat16())
    assert full_key.dtype == torch.bfloat16


def test_earlier_entries_are_never_modified():
    """A cached key is immutable once written — that is the premise of caching."""
    cache = DenseKvCache()
    first_key, first_value = kv(4)
    cache.update_and_fetch(first_key, first_value)

    for _ in range(3):
        cache.update_and_fetch(*kv(1))

    assert_allclose(cache.keys[:, :, :4], first_key)


@pytest.mark.cuda
def test_works_on_the_gpu(device):
    cache = DenseKvCache()
    pieces = [kv(1) for _ in range(4)]

    for key, value in pieces:
        full_key, _, _ = cache.update_and_fetch(key.to(device), value.to(device))

    assert full_key.device.type == "cuda"
    assert_allclose(full_key.cpu(), torch.cat([k for k, _ in pieces], dim=-2))
