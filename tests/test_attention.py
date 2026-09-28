"""attention.py against mx.fast.scaled_dot_product_attention and its own dense path."""

import random

import mlx.core as mx
import pytest
from utils import DTYPES, assert_allclose, requires_metal

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


def build_paged_batch(dtype, fill, ragged=RAGGED_BATCH):
    """Scatter each sequence's dense K/V into shuffled pages; every other slot holds `fill`."""
    rng = random.Random(0)
    blocks_needed = [-(-context // BLOCK_SIZE) for _, context in ragged]
    num_blocks = sum(blocks_needed) + 3  # spare pages that no sequence owns
    physical = list(range(num_blocks))
    rng.shuffle(physical)

    page_shape = (num_blocks * BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    key_slots = mx.full(page_shape, fill, dtype=dtype)
    value_slots = mx.full(page_shape, fill, dtype=dtype)

    sequences, tables = [], []
    for (query_len, context_len), needed in zip(ragged, blocks_needed, strict=True):
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
    for query_len, _ in ragged:
        starts.append(starts[-1] + query_len)

    pages = (num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    return (
        sequences,
        mx.concatenate([q for q, _, _ in sequences], axis=0),
        key_slots.reshape(pages),
        value_slots.reshape(pages),
        mx.array(tables, dtype=mx.int32),
        mx.array(starts, dtype=mx.int32),
        mx.array([context for _, context in ragged], dtype=mx.int32),
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


# --- the Metal kernels ----------------------------------------------------------------------


def dense_qkv(dtype, query_len, source_len, head_dim, spread=1.0):
    q = (spread * mx.random.normal((2, 16, query_len, head_dim))).astype(dtype)
    k = (spread * mx.random.normal((2, 8, source_len, head_dim))).astype(dtype)
    v = mx.random.normal((2, 8, source_len, head_dim)).astype(dtype)
    return q, k, v


@requires_metal
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("query_len", "source_len"), [(1, 1), (1, 37), (3, 200), (8, 8)])
@pytest.mark.parametrize("mask", [None, "causal"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_decode_attention_matches_the_oracle(dtype, query_len, source_len, mask, head_dim):
    q, k, v = dense_qkv(dtype, query_len, source_len, head_dim)
    assert_allclose(
        scaled_dot_product_attention_grouped(q, k, v, mask=mask, use_metal=True),
        scaled_dot_product_attention_grouped(q, k, v, mask=mask),
    )


@requires_metal
@pytest.mark.parametrize(("query_len", "source_len"), [(9, 9), (64, 64), (45, 45), (40, 300), (33, 70)])
@pytest.mark.parametrize("mask", [None, "causal"])
def test_flash_prefill_is_as_accurate_as_the_oracle(query_len, source_len, mask):
    """bf16 probabilities feed the P @ V multiply, so one output can sit a bf16 step from the
    oracle's; held to the oracle's own relative error against exact fp32, not elementwise."""
    q, k, v = dense_qkv(mx.bfloat16, query_len, source_len, 128, spread=3.0)
    exact = mx.fast.scaled_dot_product_attention(
        *(x.astype(mx.float32) for x in (q, k, v)), scale=128**-0.5, mask=mask
    )

    def relative(out):
        return (mx.linalg.norm(out.astype(mx.float32) - exact) / mx.linalg.norm(exact)).item()

    oracle = relative(scaled_dot_product_attention_grouped(q, k, v, mask=mask))
    assert relative(scaled_dot_product_attention_grouped(q, k, v, mask=mask, use_metal=True)) < 1.1 * oracle


@requires_metal
def test_an_additive_mask_takes_the_pure_path():
    q, k, v = dense_qkv(mx.float32, 5, 9, 64)
    mask = mx.random.normal((5, 9))
    assert mx.array_equal(
        scaled_dot_product_attention_grouped(q, k, v, mask=mask, use_metal=True),
        scaled_dot_product_attention_grouped(q, k, v, mask=mask),
    ).item()


def spiked(query_len, source_len, where):
    """One key aligned with every query and 30x longer: its score dwarfs the rest, early or late."""
    q, k, v = dense_qkv(mx.float32, query_len, source_len, 128)
    q = mx.broadcast_to(q[:, :, :1], q.shape)
    k[:, :, where] = 30 * q[:, ::2, 0]
    return q, k, v


@requires_metal
@pytest.mark.parametrize("where", [0, -1])
@pytest.mark.parametrize("query_len", [1, 40])
def test_online_softmax_survives_a_spike_and_overflow(where, query_len):
    """A key whose score is ~1e3 above the rest, first or last, and raw scores near 1e4: a
    running max updated wrongly or late shows up as NaN or as the wrong mixture. In bf16, so
    one query token takes decode_attention, 40 take flash_prefill, and all take the paged kernel."""
    for case in (spiked(query_len, 64, where), dense_qkv(mx.float32, query_len, 64, 128, spread=30.0)):
        q, k, v = (x.astype(mx.bfloat16) for x in case)
        dense = scaled_dot_product_attention_grouped(q, k, v, use_metal=True)
        assert mx.all(mx.isfinite(dense)).item()
        assert_allclose(dense, scaled_dot_product_attention_grouped(q, k, v))

        # The same keys as one sequence in 16 pages of 4, every query token in one batch.
        rows = mx.contiguous(q[0].swapaxes(0, 1))  # L x H_q x D
        pages = mx.contiguous(k[0].swapaxes(0, 1)).reshape(16, 4, 8, 128)
        values = mx.contiguous(v[0].swapaxes(0, 1)).reshape(16, 4, 8, 128)
        args = (rows, pages, values, mx.arange(16, dtype=mx.int32)[None],
                mx.array([0, query_len], dtype=mx.int32), mx.array([64], dtype=mx.int32))
        paged = paged_attention(*args, use_metal=True)
        assert mx.all(mx.isfinite(paged)).item()
        assert_allclose(paged, paged_attention(*args))


@requires_metal
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize(
    "ragged",
    [
        RAGGED_BATCH,                    # a prefill, a decode, a later chunk: all on the paged kernel
        [(1, 300), (40, 300), (1, 1)],   # 75 pages deep, a short context, and a chunk split off it
        [(12, 12), (9, 30)],             # only chunks longer than 8: every sequence gathers its pages
    ],
)
def test_the_paged_kernel_matches_the_oracle_and_never_reads_padding(dtype, ragged):
    _, q, key_pages, value_pages, tables, starts, contexts = build_paged_batch(dtype, mx.nan, ragged)
    got = paged_attention(q, key_pages, value_pages, tables, starts, contexts, use_metal=True)
    assert mx.all(mx.isfinite(got)).item()
    assert_allclose(got, paged_attention(q, key_pages, value_pages, tables, starts, contexts))


@requires_metal
def test_the_paged_kernel_reads_fp8_pages():
    """The kernel dequantizes in fp32 where the oracle rounds keys to bf16 first, so it is
    the more precise of the two: held to the oracle's error against the fp32 dequantization."""
    k_scale, v_scale = 0.02, 0.03
    _, q, key_pages, value_pages, tables, starts, contexts = build_paged_batch(mx.bfloat16, mx.nan)
    unused = mx.isnan(key_pages)
    key_bits = mx.where(unused, 0x7F, quantize_fp8(key_pages, k_scale)).astype(mx.uint8)
    value_bits = mx.where(unused, 0x7F, quantize_fp8(value_pages, v_scale)).astype(mx.uint8)
    args = (q, key_bits, value_bits, tables, starts, contexts)

    got = paged_attention(*args, k_scale=k_scale, v_scale=v_scale, use_metal=True)
    oracle = paged_attention(*args, k_scale=k_scale, v_scale=v_scale)
    exact = paged_attention(q.astype(mx.float32), *args[1:], k_scale=k_scale, v_scale=v_scale)

    def relative(out):
        return (mx.linalg.norm(out.astype(mx.float32) - exact) / mx.linalg.norm(exact)).item()

    assert relative(got) <= relative(oracle) * 1.05
