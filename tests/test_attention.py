"""attention.py against mx.fast.scaled_dot_product_attention and its own dense path."""

import random

import mlx.core as mx
import pytest
from utils import DTYPES, assert_allclose

from mini_vllm.attention import causal_mask, paged_attention, scaled_dot_product_attention_grouped
from mini_vllm.quantize import dequantize_fp8, quantize_fp8

HEAD_DIM = 64


def qkv(num_query_heads, num_kv_heads, query_len, source_len, dtype, batch=2):
    q = mx.random.normal((batch, num_query_heads, query_len, HEAD_DIM)).astype(dtype)
    k = mx.random.normal((batch, num_kv_heads, source_len, HEAD_DIM)).astype(dtype)
    v = mx.random.normal((batch, num_kv_heads, source_len, HEAD_DIM)).astype(dtype)
    return q, k, v


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads", [(4, 2), (16, 8), (8, 8)])  # (H_q, H_k): G = 2 is Qwen3's
@pytest.mark.parametrize("lengths", [(7, 7), (3, 11), (1, 19)])  # prefill, chunk, decode
@pytest.mark.parametrize("mask", [None, "causal"])
def test_grouped_attention_matches_mx_fast(dtype, heads, lengths, mask):
    q, k, v = qkv(*heads, *lengths, dtype)
    scale = HEAD_DIM**-0.5

    assert_allclose(
        scaled_dot_product_attention_grouped(q, k, v, scale=scale, mask=mask),
        mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask),
    )


def test_bf16_attention_rounds_only_its_output():
    """bf16 in, fp32 inside: the only error left is the final cast, however sharp the scores."""
    q = (3 * mx.random.normal((2, 16, 9, 128))).astype(mx.bfloat16)
    k = (3 * mx.random.normal((2, 8, 40, 128))).astype(mx.bfloat16)
    v = mx.random.normal((2, 8, 40, 128)).astype(mx.bfloat16)

    out = scaled_dot_product_attention_grouped(q, k, v, mask="causal").astype(mx.float32)
    exact = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), scale=128**-0.5, mask="causal"
    )

    # One bf16 rounding is ~2^-9 relative; rounding the scores to bf16 as well measures ~0.01.
    relative = (mx.linalg.norm(out - exact) / mx.linalg.norm(exact)).item()
    assert relative < 2**-8, relative


@pytest.mark.parametrize("mask_shape", [(5, 9), (2, 4, 5, 9)])  # shared, per batch and head
def test_an_additive_mask_broadcasts_like_mx_fast(mask_shape):
    q, k, v = qkv(4, 2, 5, 9, mx.float32)
    mask = mx.random.normal(mask_shape)
    scale = HEAD_DIM**-0.5

    assert_allclose(
        scaled_dot_product_attention_grouped(q, k, v, scale=scale, mask=mask),
        mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask),
    )


def test_causal_mask_shifts_for_a_decode_step():
    # One new token sees the whole history; a 3-token chunk over 5 sees up to S - L + i.
    assert mx.array_equal(causal_mask(1, 5), mx.zeros((1, 5))).item()
    allowed = (causal_mask(3, 5) == 0).tolist()
    assert allowed == [
        [True, True, True, False, False],
        [True, True, True, True, False],
        [True, True, True, True, True],
    ]


# (query_len, context_len) per sequence: a fresh prefill, a decode step, a later chunk.
RAGGED_BATCH = [(5, 5), (1, 16), (7, 23)]
BLOCK_SIZE = 4
NUM_KV_HEADS = 2
NUM_QUERY_HEADS = 4


def build_paged_batch(dtype, fill):
    """Scatter each sequence's dense K/V into shuffled pages; every other slot holds `fill`."""
    rng = random.Random(0)
    blocks_needed = [-(-context // BLOCK_SIZE) for _, context in RAGGED_BATCH]
    num_blocks = sum(blocks_needed) + 3  # spare pages that no sequence owns
    physical = list(range(num_blocks))
    rng.shuffle(physical)

    page_shape = (num_blocks * BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    key_slots = mx.full(page_shape, fill, dtype=dtype)
    value_slots = mx.full(page_shape, fill, dtype=dtype)

    sequences, tables = [], []
    for (query_len, context_len), needed in zip(RAGGED_BATCH, blocks_needed, strict=True):
        q = mx.random.normal((query_len, NUM_QUERY_HEADS, HEAD_DIM)).astype(dtype)
        k = mx.random.normal((context_len, NUM_KV_HEADS, HEAD_DIM)).astype(dtype)
        v = mx.random.normal((context_len, NUM_KV_HEADS, HEAD_DIM)).astype(dtype)
        table = [physical.pop() for _ in range(needed)]

        slots = mx.array([table[t // BLOCK_SIZE] * BLOCK_SIZE + t % BLOCK_SIZE for t in range(context_len)])
        key_slots[slots] = k
        value_slots[slots] = v
        sequences.append((q, k, v))
        tables.append(table + [-1] * (max(blocks_needed) - needed))

    starts = [0]
    for query_len, _ in RAGGED_BATCH:
        starts.append(starts[-1] + query_len)

    pages = (num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    return (
        sequences,
        mx.concatenate([q for q, _, _ in sequences], axis=0),
        key_slots.reshape(pages),
        value_slots.reshape(pages),
        mx.array(tables, dtype=mx.int32),
        mx.array(starts, dtype=mx.int32),
        mx.array([context for _, context in RAGGED_BATCH], dtype=mx.int32),
    )


def dense_reference(sequences):
    """Each sequence attended on its own, over contiguous K/V."""
    outputs = []
    for q, k, v in sequences:
        attended = scaled_dot_product_attention_grouped(
            q.swapaxes(0, 1), k.swapaxes(0, 1), v.swapaxes(0, 1), mask="causal"
        )
        outputs.append(attended.swapaxes(0, 1))
    return mx.concatenate(outputs, axis=0)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_paged_attention_matches_each_dense_run_and_never_reads_padding(dtype):
    sequences, q, key_pages, value_pages, tables, starts, contexts = build_paged_batch(dtype, mx.nan)

    out = paged_attention(q, key_pages, value_pages, tables, starts, contexts)
    assert_allclose(out, dense_reference(sequences))


def test_paged_attention_reads_fp8_pages_through_their_scales():
    k_scale, v_scale = 0.02, 0.03
    sequences, q, key_pages, value_pages, tables, starts, contexts = build_paged_batch(mx.bfloat16, mx.nan)

    # 0x7F is NaN in e4m3, so padding read by mistake still poisons the output.
    unused = mx.isnan(key_pages)
    key_bits = mx.where(unused, 0x7F, quantize_fp8(key_pages, k_scale)).astype(mx.uint8)
    value_bits = mx.where(unused, 0x7F, quantize_fp8(value_pages, v_scale)).astype(mx.uint8)

    out = paged_attention(q, key_bits, value_bits, tables, starts, contexts, k_scale=k_scale, v_scale=v_scale)

    def round_trip(x, scale):
        return dequantize_fp8(quantize_fp8(x, scale), scale, x.dtype)

    expected = [(q_i, round_trip(k_i, k_scale), round_trip(v_i, v_scale)) for q_i, k_i, v_i in sequences]
    assert_allclose(out, dense_reference(expected))
