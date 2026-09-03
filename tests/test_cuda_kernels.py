"""Every CUDA kernel against the PyTorch reference it replaces.

The comparisons are differential: each kernel is pinned to the exact ops.py
expression it replaces, on the boundary shapes where kernels break -- tile
boundaries and late maxima for the online softmax, shuffled block tables for
the paged gather, proof the flash prefill cannot see the future, and
bit-identity for FP8 quantize-scatter. The whole file carries the cuda marker
and skips cleanly without a GPU; oracle tests additionally need the real
weights.
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import (
    KERNEL_DRIFT_LIMIT,
    TINY_QWEN3_DIMS,
    assert_allclose,
    assert_relative_error_below,
    assert_tokens_equal,
    config_from_hf,
    weights_from_hf,
)

from mini_vllm import kernels, ops
from mini_vllm.engine import generate_ids_cached
from mini_vllm.model import Qwen3Cached

pytestmark = pytest.mark.cuda

QWEN3_HEADS = (16, 8, 128)  # H_q, H_k, D for Qwen3-0.6B


@pytest.fixture
def kernel(device):
    return kernels.load_extension()


# --- The extension -----------------------------------------------------------

def test_every_claimed_kernel_is_callable(kernel):
    """The dispatch table cannot claim a kernel that is missing or misspelled."""
    for name in kernels.cuda_kernel_names():
        assert hasattr(kernel, name), f"{name} is claimed but not exported"


def test_the_dispatch_report_names_every_op(kernel):
    report = kernels.dispatch_report(use_cuda=True)
    for name in kernels.CUDA_KERNELS:
        assert name in report
    assert "flash_prefill" in kernels.NOT_YET_FASTER
    line = next(row for row in report.splitlines() if "flash_prefill" in row)
    assert "torch" in line, "a kernel measured slower must not be the default"


# --- RMSNorm -----------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_rmsnorm_matches_the_oracle(kernel, dtype):
    x = torch.randn(33, 1024, device="cuda", dtype=dtype)
    weight = torch.randn(1024, device="cuda", dtype=dtype)

    assert_allclose(kernel.rmsnorm(x, weight, 1e-6), ops.rms_norm(x, weight, 1e-6))


def test_rmsnorm_over_the_head_dimension(kernel):
    """QK-norm's shape: the reduced axis is D, not E."""
    x = torch.randn(2, 7, 16, 128, device="cuda")
    weight = torch.randn(128, device="cuda")
    assert_allclose(kernel.rmsnorm(x, weight, 1e-6), ops.rms_norm(x, weight, 1e-6))


def test_rmsnorm_dispatch_declines_mixed_dtypes(kernel):
    """PyTorch would promote; returning a different dtype than the oracle is worse
    than declining the kernel, so the wrapper falls back."""
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(64, device="cuda", dtype=torch.float32)

    got = kernels.rmsnorm(x, weight, use_cuda=True)
    assert_allclose(got, ops.rms_norm(x, weight))


# --- RoPE --------------------------------------------------------------------

def test_rope_matches_the_oracle_at_explicit_positions(kernel):
    """Positions are per token and non-contiguous: the ragged-batch case."""
    tables = ops.RoPE(128, 512, device="cuda")
    x = torch.randn(1, 7, 16, 128, device="cuda")
    positions = torch.tensor([0, 5, 6, 100, 101, 102, 511], device="cuda")

    got = kernel.rope(x, positions, tables.cos, tables.sin)
    assert_allclose(got, ops.apply_rope(x, positions, tables.cos, tables.sin))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rope_keeps_low_precision_activations(kernel, dtype):
    tables = ops.RoPE(64, 128, device="cuda")
    x = torch.randn(1, 33, 4, 64, device="cuda", dtype=dtype)
    positions = torch.arange(33, device="cuda")

    got = kernel.rope(x, positions, tables.cos, tables.sin)
    assert got.dtype == dtype
    assert_allclose(got, ops.apply_rope(x, positions, tables.cos, tables.sin))


# --- SwiGLU ------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_swiglu_matches_the_oracle(kernel, dtype):
    gate = torch.randn(65, 3072, device="cuda", dtype=dtype)
    up = torch.randn_like(gate)

    assert_allclose(kernel.swiglu(gate, up), ops.silu(gate) * up)


def test_swiglu_fp32_is_bitwise(kernel):
    """Same expression, same order: fp32 should agree exactly, not merely closely."""
    gate = torch.randn(16, 128, device="cuda")
    up = torch.randn_like(gate)
    assert torch.equal(kernel.swiglu(gate, up), ops.silu(gate) * up)


# --- Decode attention --------------------------------------------------------

def attend(kernel, q, k, v):
    return kernel.decode_attention(q, k, v, 1.0 / math.sqrt(q.shape[-1]))


def triple(batch, num_query_heads, num_kv_heads, source_len, head_dim, dtype):
    q = torch.randn(batch, num_query_heads, 1, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, num_kv_heads, source_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, num_kv_heads, source_len, head_dim, device="cuda", dtype=dtype)
    return q, k, v


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_decode_matches_the_oracle_at_qwen3_shapes(kernel, dtype):
    q, k, v = triple(2, *QWEN3_HEADS[:2], 200, QWEN3_HEADS[2], dtype)
    assert_allclose(attend(kernel, q, k, v), ops.scaled_dot_product_attention_grouped(q, k, v))


@pytest.mark.parametrize("source_len", [1, 63, 64, 65, 127, 129, 513, 1000])
def test_decode_at_every_tile_boundary(kernel, source_len):
    """The tile is 64 keys wide, so these are the lengths that break kernels: a partial
    final tile is where a loop bound reads one key too many or one too few."""
    q, k, v = triple(1, *QWEN3_HEADS[:2], source_len, QWEN3_HEADS[2], torch.float32)
    assert_allclose(attend(kernel, q, k, v), ops.scaled_dot_product_attention_grouped(q, k, v))


def test_decode_survives_a_late_spike(kernel):
    """The maximum arrives in the last tile, so every accumulator must be rescaled.

    The recurrence's classic failure: a kernel that rescales the running sum `l` but
    not the running output `O` still produces weights that sum to one — a plausible
    convex combination, just the wrong one. Putting the spike last maximizes the
    already-accumulated `O` a missing correction would leave un-rescaled.
    """
    head_dim = 64
    q = torch.ones(1, 1, 1, head_dim, device="cuda")
    k = torch.full((1, 1, 200, head_dim), 0.01, device="cuda")
    v = torch.randn(1, 1, 200, head_dim, device="cuda")
    k[0, 0, 199] = 2.0  # scores 16 against 0.08 for the rest: takes all the weight

    got = attend(kernel, q, k, v)

    assert_allclose(got, ops.scaled_dot_product_attention_grouped(q, k, v))
    assert_relative_error_below(got, v[0, 0, 199].view(1, 1, 1, head_dim), 0.01)


def test_decode_survives_an_early_spike(kernel):
    """The mirror image: a kernel applying the correction with the wrong sign is
    correct on this input and wrong on the one above, so both are needed."""
    head_dim = 64
    q = torch.ones(1, 1, 1, head_dim, device="cuda")
    k = torch.full((1, 1, 200, head_dim), 0.01, device="cuda")
    v = torch.randn(1, 1, 200, head_dim, device="cuda")
    k[0, 0, 0] = 1.0

    assert_allclose(attend(kernel, q, k, v), ops.scaled_dot_product_attention_grouped(q, k, v))


@pytest.mark.parametrize("source_len", [2049, 4096, 8192])
def test_decode_splits_long_contexts_across_blocks(kernel, source_len):
    """Past ~512 keys the cache is cut into pieces with their own (m, l, O) and merged:
    new code that short contexts never touch."""
    q, k, v = triple(1, *QWEN3_HEADS[:2], source_len, QWEN3_HEADS[2], torch.float32)
    assert_allclose(attend(kernel, q, k, v), ops.scaled_dot_product_attention_grouped(q, k, v))


def test_the_split_merge_rescales_across_splits(kernel):
    """A spike in the last split, which every earlier split must be rescaled by — the
    late-spike test one level up, at the merge."""
    head_dim = 64
    q = torch.ones(1, 16, 1, head_dim, device="cuda")
    k = torch.full((1, 8, 4096, head_dim), 0.01, device="cuda")
    v = torch.randn(1, 8, 4096, head_dim, device="cuda")
    k[:, :, 4095] = 2.0

    got = attend(kernel, q, k, v)

    assert_allclose(got, ops.scaled_dot_product_attention_grouped(q, k, v))
    assert_relative_error_below(got[:, 0], v[:, 0, 4095].view(1, 1, head_dim), 0.01)


def test_decode_handles_logits_that_would_overflow(kernel):
    """Scores near fp32's exp limit (~88): the running max must be subtracted."""
    head_dim = 128
    q = torch.full((1, 1, 1, head_dim), 30.0, device="cuda")
    k = torch.full((1, 1, 100, head_dim), 30.0, device="cuda")
    v = torch.randn(1, 1, 100, head_dim, device="cuda")

    got = attend(kernel, q, k, v)

    assert torch.isfinite(got).all(), "the running max is not being subtracted"
    assert_allclose(got, ops.scaled_dot_product_attention_grouped(q, k, v))


def test_decode_output_is_a_convex_combination_of_values(kernel):
    """Softmax weights are non-negative and sum to one, so every output element lies
    between the smallest and largest value — a property no shared-wrong-reference
    comparison can fake."""
    q, k, v = triple(1, 4, 2, 300, 64, torch.float32)

    got = attend(kernel, q, k, v)

    low = v.amin(dim=2, keepdim=True).repeat_interleave(2, dim=1)
    high = v.amax(dim=2, keepdim=True).repeat_interleave(2, dim=1)
    assert (got >= low - 1e-5).all() and (got <= high + 1e-5).all()


def test_decode_reads_strided_views_without_copying(kernel):
    """A query sliced from a prefill tensor and a cache narrowed from a bigger buffer:
    making them contiguous would copy the whole cache every step."""
    num_query_heads, num_kv_heads, head_dim = QWEN3_HEADS
    prefill = torch.randn(2, num_query_heads, 8, head_dim, device="cuda")
    q = prefill[:, :, -1:, :]
    reserved = torch.randn(2, num_kv_heads, 512, head_dim, device="cuda")
    k, v = reserved[:, :, :100], reserved[:, :, 200:300]

    assert not q.is_contiguous() and not k.is_contiguous()
    assert_allclose(attend(kernel, q, k, v), ops.scaled_dot_product_attention_grouped(q, k, v))


def test_decode_refuses_shapes_it_cannot_serve(kernel):
    q, k, v = triple(1, 4, 2, 100, 64, torch.float32)
    with pytest.raises(RuntimeError, match="L == 1"):
        attend(kernel, torch.randn(1, 4, 2, 64, device="cuda"), k, v)
    empty = torch.randn(1, 2, 0, 64, device="cuda")
    with pytest.raises(RuntimeError, match="empty"):
        attend(kernel, q, empty, empty)


# --- Flash prefill -----------------------------------------------------------

def prefill(kernel, q, k, v):
    return kernel.flash_prefill(q, k, v, 1.0 / math.sqrt(q.shape[-1]))


@pytest.mark.parametrize("length", [1, 16, 17, 63, 64, 65, 128, 200])
def test_flash_prefill_matches_the_causal_oracle(kernel, length):
    """Tiled causal attention at every tile-boundary size, including lengths that are
    not multiples of the tile."""
    num_query_heads, num_kv_heads, head_dim = 8, 4, 64
    q = torch.randn(1, num_query_heads, length, head_dim, device="cuda")
    k = torch.randn(1, num_kv_heads, length, head_dim, device="cuda")
    v = torch.randn_like(k)

    assert_allclose(
        prefill(kernel, q, k, v),
        ops.scaled_dot_product_attention_grouped(q, k, v, mask="causal"),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_flash_prefill_in_low_precision(kernel, dtype):
    """Aggregate relative error rather than elementwise: the kernel keeps fp32 to the
    store while the oracle rounds intermediates to the storage dtype, so a handful of
    near-zero elements land one rounding apart without either being wrong."""
    q = torch.randn(1, 16, 128, 128, device="cuda", dtype=dtype)
    k = torch.randn(1, 8, 128, 128, device="cuda", dtype=dtype)
    v = torch.randn_like(k)

    assert_relative_error_below(
        prefill(kernel, q, k, v),
        ops.scaled_dot_product_attention_grouped(q, k, v, mask="causal"),
        KERNEL_DRIFT_LIMIT,
    )


def test_flash_prefill_cannot_see_the_future(kernel):
    """Change a key and value the causal mask forbids, and nothing may move.

    The kernel skips tiles above the diagonal rather than masking them, so this is the
    test that the skip is exactly the mask.
    """
    q = torch.randn(1, 4, 65, 64, device="cuda")
    k = torch.randn(1, 2, 65, 64, device="cuda")
    v = torch.randn_like(k)

    before = prefill(kernel, q, k, v)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 64], v2[:, :, 64] = 99.0, -99.0  # visible only to the last query row
    after = prefill(kernel, q, k2, v2)

    assert torch.equal(before[:, :, :64], after[:, :, :64]), "an earlier row saw the future"
    assert not torch.allclose(before[:, :, 64], after[:, :, 64]), "the last row must see it"


def test_flash_prefill_serves_a_decode_shaped_chunk_tail(kernel):
    """L < S: the queries are the *last* L positions, so the diagonal is shifted."""
    q = torch.randn(1, 4, 5, 64, device="cuda")
    k = torch.randn(1, 2, 37, 64, device="cuda")
    v = torch.randn_like(k)

    assert_allclose(
        prefill(kernel, q, k, v),
        ops.scaled_dot_product_attention_grouped(q, k, v, mask="causal"),
    )


def test_flash_prefill_stays_off_the_dispatch_path(kernel):
    """Correct but measured slower than cuBLAS, so `use_cuda=True` still routes prefill
    to the reference — the kernel earns the dispatch by being faster, not by existing."""
    q = torch.randn(1, 16, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 8, 32, 128, device="cuda", dtype=torch.bfloat16)

    got = kernels.attention(q, k, k, mask="causal", use_cuda=True)
    want = ops.scaled_dot_product_attention_grouped(q, k, k, mask="causal")
    assert torch.equal(got, want), "prefill dispatch left the reference path"


# --- Paged attention ---------------------------------------------------------

def paged_setup(batch, context_len, block_size=16, dtype=torch.bfloat16,
                kv_dtype=None, query_len=1):
    """A shuffled paged pool with `batch` sequences of `context_len` tokens each.

    Shuffled deliberately: a pool in logical order gives the kernel sequential reads
    an aged allocator never would, and hides gather bugs besides.
    """
    num_query_heads, num_kv_heads, head_dim = QWEN3_HEADS
    blocks_each = -(-context_len // block_size)
    pool_blocks = batch * blocks_each

    storage = kv_dtype or dtype
    source_keys = torch.randn(pool_blocks, block_size, num_kv_heads, head_dim, device="cuda",
                              dtype=dtype)
    source_values = torch.randn_like(source_keys)
    keys = source_keys.to(storage)
    values = source_values.to(storage)

    shuffled = torch.randperm(pool_blocks, device="cuda", dtype=torch.int32)
    block_tables = shuffled.reshape(batch, blocks_each).contiguous()

    total = batch * query_len
    q = torch.randn(total, num_query_heads, head_dim, device="cuda", dtype=dtype)
    cu_seqlens = torch.arange(0, total + 1, query_len, device="cuda", dtype=torch.int32)
    contexts = torch.full((batch,), context_len, device="cuda", dtype=torch.int32)
    lengths = torch.full((batch,), query_len, device="cuda", dtype=torch.int32)
    return q, keys, values, block_tables, cu_seqlens, contexts, lengths


def run_both(kernel, setup, query_len, context_len, k_scale=1.0, v_scale=1.0):
    q, keys, values, tables, cu, contexts, lengths = setup
    scale = 1.0 / math.sqrt(q.shape[-1])
    got = kernel.paged_attention(q, keys, values, tables, cu, contexts, lengths,
                                 query_len, context_len, scale, k_scale, v_scale)
    want = ops.paged_attention_gathered(q, keys, values, tables, cu, contexts, scale,
                                        k_scale=k_scale, v_scale=v_scale)
    return got, want


@pytest.mark.parametrize("batch,context_len", [(1, 40), (4, 100), (16, 250)])
def test_paged_decode_matches_the_gather_oracle(kernel, batch, context_len):
    got, want = run_both(kernel, paged_setup(batch, context_len), 1, context_len)
    assert_relative_error_below(got, want, KERNEL_DRIFT_LIMIT)


def test_paged_decode_on_partial_final_blocks(kernel):
    """A context that is not a multiple of the block size: the last page is partial and
    the walk must stop inside it."""
    got, want = run_both(kernel, paged_setup(3, 37), 1, 37)
    assert_relative_error_below(got, want, KERNEL_DRIFT_LIMIT)


def test_paged_decode_splits_a_long_context(kernel):
    got, want = run_both(kernel, paged_setup(1, 4096), 1, 4096)
    assert_relative_error_below(got, want, KERNEL_DRIFT_LIMIT)


def test_paged_prefill_matches_the_gather_oracle(kernel):
    """The multi-token path: a chunk's queries against a longer cached context."""
    got, want = run_both(kernel, paged_setup(2, 128, query_len=32), 32, 128)
    assert_relative_error_below(got, want, KERNEL_DRIFT_LIMIT)


def test_a_ragged_paged_batch_matches_each_sequence_alone(kernel):
    """A prefill chunk beside decodes in one launch, rows independent."""
    num_query_heads, num_kv_heads, head_dim = QWEN3_HEADS
    block_size = 16
    dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(head_dim)

    # Sequence 0: an 8-token chunk over 24 of context; sequences 1, 2: decodes.
    contexts_py = [24, 40, 17]
    lengths_py = [8, 1, 1]
    blocks_each = [-(-c // block_size) for c in contexts_py]
    pool_blocks = sum(blocks_each)
    keys = torch.randn(pool_blocks, block_size, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    values = torch.randn_like(keys)

    shuffled = torch.randperm(pool_blocks, dtype=torch.int32).tolist()
    tables, taken = [], 0
    for count in blocks_each:
        tables.append(shuffled[taken : taken + count])
        taken += count
    widest = max(blocks_each)
    padded = torch.tensor(
        [row + [-1] * (widest - len(row)) for row in tables], dtype=torch.int32, device="cuda"
    )

    total = sum(lengths_py)
    q = torch.randn(total, num_query_heads, head_dim, device="cuda", dtype=dtype)
    cu = torch.tensor([0, 8, 9, 10], dtype=torch.int32, device="cuda")
    contexts = torch.tensor(contexts_py, dtype=torch.int32, device="cuda")
    lengths = torch.tensor(lengths_py, dtype=torch.int32, device="cuda")

    together = kernel.paged_attention(q, keys, values, padded, cu, contexts, lengths,
                                      8, max(contexts_py), scale, 1.0, 1.0)

    for index in range(3):
        rows = slice(int(cu[index]), int(cu[index + 1]))
        alone = kernel.paged_attention(
            q[rows].contiguous(), keys, values, padded[index : index + 1].contiguous(),
            torch.tensor([0, lengths_py[index]], dtype=torch.int32, device="cuda"),
            contexts[index : index + 1], lengths[index : index + 1],
            lengths_py[index], contexts_py[index], scale, 1.0, 1.0,
        )
        assert_allclose(together[rows], alone, msg=f"sequence {index} affected by the batch")


def test_paged_attention_reads_an_fp8_pool(kernel):
    """e4m3 storage with explicit scales: the key scale folds into the softmax scale
    and the value scale rides on the output, dequantized in registers."""
    from conftest import FP8_MAX_ERROR

    setup = paged_setup(4, 64, dtype=torch.bfloat16, kv_dtype=torch.float8_e4m3fn)
    got, want = run_both(kernel, setup, 1, 64, k_scale=0.5, v_scale=0.25)
    assert_relative_error_below(got, want, FP8_MAX_ERROR)


def test_paged_dispatch_reaches_the_kernel(kernel):
    """`kernels.paged_attention(use_cuda=True)` and the raw kernel agree exactly."""
    setup = paged_setup(2, 48)
    q, keys, values, tables, cu, contexts, lengths = setup
    scale = 1.0 / math.sqrt(q.shape[-1])

    via_dispatch = kernels.paged_attention(
        q, keys, values, tables, cu, contexts, lengths, 1, 48, scale, use_cuda=True
    )
    direct = kernel.paged_attention(q, keys, values, tables, cu, contexts, lengths,
                                    1, 48, scale, 1.0, 1.0)
    assert torch.equal(via_dispatch, direct)


# --- FP8 quantize-scatter ----------------------------------------------------

def test_quantize_scatter_is_bitwise_identical_to_the_two_pass_reference(kernel):
    """Both divide by the scale in fp32 and round to nearest even, so the stored FP8
    bytes must be identical rather than merely close — compared as uint8 so no
    tolerance can hide a rounding disagreement."""
    num_slots, num_kv_heads, head_dim, count = 64, 2, 64, 20
    k_scale, v_scale = 0.5, 0.25
    key = torch.randn(count, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    # Scattered, non-monotonic slots: the paged case. A kernel assuming contiguity
    # would pass on arange.
    slots = torch.randperm(num_slots, device="cuda")[:count].to(torch.int32)

    def pools():
        shape = (num_slots, num_kv_heads, head_dim)
        return (torch.zeros(shape, device="cuda", dtype=torch.float8_e4m3fn),
                torch.zeros(shape, device="cuda", dtype=torch.float8_e4m3fn))

    kernel_keys, kernel_values = pools()
    kernels.quantize_scatter(key, value, kernel_keys, kernel_values, slots,
                             k_scale, v_scale, use_cuda=True)
    reference_keys, reference_values = pools()
    kernels.quantize_scatter(key, value, reference_keys, reference_values, slots,
                             k_scale, v_scale, use_cuda=False)

    assert torch.equal(kernel_keys.view(torch.uint8), reference_keys.view(torch.uint8))
    assert torch.equal(kernel_values.view(torch.uint8), reference_values.view(torch.uint8))


def test_quantize_scatter_round_trips_within_fp8_tolerance(kernel):
    from conftest import FP8_MAX_ERROR, relative_error

    count = 16
    key = torch.randn(count, 2, 64, device="cuda", dtype=torch.bfloat16)
    pool = torch.zeros(count, 2, 64, device="cuda", dtype=torch.float8_e4m3fn)
    slots = torch.arange(count, device="cuda", dtype=torch.int32)

    kernels.quantize_scatter(key, key, pool, pool.clone(), slots, 1.0, 1.0, use_cuda=True)

    assert relative_error(pool.float(), key.float()) < FP8_MAX_ERROR


# --- The model, kernels on/off -----------------------------------------------

def tiny_pair(tiny_qwen3, device, dtype):
    theirs = tiny_qwen3.to(device=device, dtype=dtype)
    config, weights = config_from_hf(theirs), weights_from_hf(theirs)
    return Qwen3Cached(config, weights, use_cuda=False), Qwen3Cached(config, weights, use_cuda=True)


def test_model_greedy_output_is_token_identical_with_kernels_on(tiny_qwen3, device):
    """The property that matters more than the value: the kernels changed the speed,
    not the text. Prefill and 24 decode steps, fp32, exact."""
    torch_path, cuda_path = tiny_pair(tiny_qwen3, device, torch.float32)
    ids = torch.randint(0, TINY_QWEN3_DIMS["vocab_size"], (1, 5), device=device)

    assert_tokens_equal(
        generate_ids_cached(cuda_path, ids, max_tokens=24),
        generate_ids_cached(torch_path, ids, max_tokens=24),
    )


def test_model_bf16_drift_stays_under_one_ulp(tiny_qwen3, device):
    torch_path, cuda_path = tiny_pair(tiny_qwen3, device, torch.bfloat16)
    ids = torch.randint(0, TINY_QWEN3_DIMS["vocab_size"], (1, 5), device=device)

    reference_cache, cache = torch_path.create_kv_cache(), cuda_path.create_kv_cache()
    torch_path(ids, reference_cache)
    cuda_path(ids, cache)

    step = torch.tensor([[7]], device=device)
    assert_relative_error_below(
        cuda_path(step, cache), torch_path(step, reference_cache), 2e-2
    )


@pytest.mark.oracle
def test_real_model_is_token_identical_with_kernels_on():
    """The real 0.6B checkpoint, 48 greedy tokens, kernels on vs off. Longer than the
    tiny-model runs on purpose: decode attention's work grows with the context, so a
    rescaling bug invisible in one tile appears once decode walks past 64 cached keys."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mini_vllm.model import resolve_model_path

    path = resolve_model_path()
    if not (path / "model.safetensors").is_file():
        pytest.skip("Qwen3-0.6B weights are not downloaded")

    hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to("cuda").eval()
    config, weights = config_from_hf(hf), weights_from_hf(hf)
    tokenizer = AutoTokenizer.from_pretrained(path)
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.to("cuda")

    torch_path = Qwen3Cached(config, weights, use_cuda=False)
    cuda_path = Qwen3Cached(config, weights, use_cuda=True)

    assert_tokens_equal(
        generate_ids_cached(cuda_path, ids, max_tokens=48),
        generate_ids_cached(torch_path, ids, max_tokens=48),
    )
