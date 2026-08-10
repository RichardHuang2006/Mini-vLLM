"""Step 3.4 — decode attention (online softmax) against the Step 1.4 oracle.

The oracle materializes the whole `1 x S` score row and reads it twice. This
kernel never materializes it, carrying `(m, l, O)` in FP32 registers instead, so
the tests here are aimed squarely at the recurrence:

* **tile boundaries** — every off-by-one in the rescaling hides in a partial
  final tile, so `S` is tested at, one below, and one above multiples of 64;
* **the rescaling factor itself** — `test_survives_a_late_spike` constructs a
  sequence whose maximum arrives in the *last* tile, which is the case that
  rescales every accumulator that came before it and the case that a kernel
  rescaling only `l` and not `O` passes at low precision and fails here;
* **saturation** — logits large enough that a non-shifted softmax would return
  `inf/inf`.
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import (
    TINY_QWEN3_DIMS,
    assert_allclose,
    assert_relative_error_below,
    assert_tokens_equal,
    config_from_hf,
    weights_from_hf,
)

from mini_vllm.attention import scaled_dot_product_attention_grouped as oracle
from mini_vllm.generate import generate_ids_cached
from mini_vllm.kernels import ops
from mini_vllm.kernels.extension import load_extension
from mini_vllm.model.qwen3_cached import Qwen3Cached

pytestmark = pytest.mark.cuda

QWEN3_HEADS = (16, 8, 128)  # H_q, H_k, D for Qwen3-0.6B


@pytest.fixture
def kernel(device):
    return load_extension()


def attend(kernel, q, k, v):
    return kernel.decode_attention(q, k, v, 1.0 / math.sqrt(q.shape[-1]))


def triple(batch, num_query_heads, num_kv_heads, source_len, head_dim, dtype, scale=1.0):
    q = torch.randn(batch, num_query_heads, 1, head_dim, device="cuda", dtype=dtype) * scale
    k = torch.randn(batch, num_kv_heads, source_len, head_dim, device="cuda", dtype=dtype) * scale
    v = torch.randn(batch, num_kv_heads, source_len, head_dim, device="cuda", dtype=dtype)
    return q, k, v


# ------------------------------------------------------------------ the values


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_matches_the_oracle_at_qwen3_shapes(kernel, dtype):
    num_query_heads, num_kv_heads, head_dim = QWEN3_HEADS
    q, k, v = triple(2, num_query_heads, num_kv_heads, 200, head_dim, dtype)

    assert_allclose(attend(kernel, q, k, v), oracle(q, k, v))


@pytest.mark.parametrize("source_len", [1, 2, 63, 64, 65, 127, 128, 129, 512, 513, 1000])
def test_every_tile_boundary(kernel, source_len):
    """The tile is 64 keys wide, so these are the lengths that break kernels.

    A partial final tile is where a loop bound reads one key too many (garbage
    from the next sequence) or one too few (a token silently ignored). `S = 1` is
    the same edge from the other side: a single tile that is almost entirely
    partial, and the first-iteration `exp(-inf - m)` path.
    """
    num_query_heads, num_kv_heads, head_dim = QWEN3_HEADS
    q, k, v = triple(1, num_query_heads, num_kv_heads, source_len, head_dim, torch.float32)

    assert_allclose(attend(kernel, q, k, v), oracle(q, k, v))


def test_survives_a_late_spike(kernel):
    """The maximum arrives in the last tile, so every accumulator is rescaled.

    This is the test the plan's warning is about. A kernel that rescales the
    running sum `l` but not the running output `O` still produces weights that
    sum to one, so the output looks like a plausible convex combination and
    nothing raises — it is simply the wrong combination. Putting the spike last
    maximizes the amount of already-accumulated `O` that a missing correction
    would leave un-rescaled, and comparing against the oracle in fp32 catches it.
    """
    head_dim = 64
    q = torch.ones(1, 1, 1, head_dim, device="cuda")
    k = torch.full((1, 1, 200, head_dim), 0.01, device="cuda")
    v = torch.randn(1, 1, 200, head_dim, device="cuda")

    # A key that aligns with the query far better than any before it: its score is
    # 16 against 0.08 for the other 199, so it takes essentially all the weight.
    k[0, 0, 199] = 2.0

    got = attend(kernel, q, k, v)

    assert_allclose(got, oracle(q, k, v))
    # And the answer really is that last value, so this test would also notice a
    # kernel that dropped the spike rather than mis-rescaling around it.
    assert_relative_error_below(got, v[0, 0, 199].view(1, 1, 1, head_dim), 0.01)


@pytest.mark.parametrize("source_len", [1024, 2049, 4096, 8192])
def test_long_contexts_are_split_across_blocks(kernel, source_len):
    """Past ~512 keys the cache is cut into pieces and merged, which is new code.

    One block per (sequence, head) is 16 blocks for 36 SMs, so long contexts run
    the flash-decoding split instead: several blocks per row, each with its own
    `(m, l, O)`, merged afterwards. These lengths straddle the split boundary in
    both directions, including one that leaves the last split partial.
    """
    num_query_heads, num_kv_heads, head_dim = QWEN3_HEADS
    q, k, v = triple(1, num_query_heads, num_kv_heads, source_len, head_dim, torch.float32)

    assert_allclose(attend(kernel, q, k, v), oracle(q, k, v))


def test_the_merge_rescales_across_splits(kernel):
    """A spike in the *last* split, which every earlier split must be rescaled by.

    `test_survives_a_late_spike` is this test one level down, inside a single
    block's tile loop. The merge repeats that recurrence across blocks, so it can
    be wrong in exactly the same way and independently: rescale the summed weights
    but not the summed outputs and the result is still a convex combination, just
    of the wrong thing. Short contexts never touch this path, which is why the
    long version has to be tested separately rather than assumed.
    """
    head_dim = 64
    q = torch.ones(1, 16, 1, head_dim, device="cuda")
    k = torch.full((1, 8, 4096, head_dim), 0.01, device="cuda")
    v = torch.randn(1, 8, 4096, head_dim, device="cuda")
    k[:, :, 4095] = 2.0

    got = attend(kernel, q, k, v)

    assert_allclose(got, oracle(q, k, v))
    assert_relative_error_below(got[:, 0], v[:, 0, 4095].view(1, 1, head_dim), 0.01)


def test_survives_an_early_spike(kernel):
    """The mirror image: the max arrives first, so the correction is always 1.

    Worth having as a pair, because a kernel that applies the correction with the
    wrong sign — `exp(m_new - m_old)` — is *correct* on this input and wrong on
    the one above.
    """
    head_dim = 64
    q = torch.ones(1, 1, 1, head_dim, device="cuda")
    k = torch.full((1, 1, 200, head_dim), 0.01, device="cuda")
    v = torch.randn(1, 1, 200, head_dim, device="cuda")
    k[0, 0, 0] = 1.0

    assert_allclose(attend(kernel, q, k, v), oracle(q, k, v))


def test_handles_logits_that_would_overflow(kernel):
    """Scores near fp32's `exp` limit, which is around 88."""
    head_dim = 128
    q = torch.full((1, 1, 1, head_dim), 30.0, device="cuda")
    k = torch.full((1, 1, 100, head_dim), 30.0, device="cuda")
    v = torch.randn(1, 1, 100, head_dim, device="cuda")

    got = attend(kernel, q, k, v)

    assert torch.isfinite(got).all(), "the running max is not being subtracted"
    assert_allclose(got, oracle(q, k, v))


def test_output_is_a_convex_combination_of_values(kernel):
    """A property that holds for any correct attention, oracle or not.

    Softmax weights are non-negative and sum to one, so every output element must
    lie between the smallest and largest value at that dimension. This catches a
    missing normalization that a tolerance comparison against a *shared* wrong
    reference would not.
    """
    q, k, v = triple(1, 4, 2, 300, 64, torch.float32)

    got = attend(kernel, q, k, v)

    low = v.amin(dim=2, keepdim=True).repeat_interleave(2, dim=1)
    high = v.amax(dim=2, keepdim=True).repeat_interleave(2, dim=1)
    assert (got >= low - 1e-5).all() and (got <= high + 1e-5).all()


@pytest.mark.parametrize("head_dim", [32, 64, 128, 192])
def test_head_dimensions(kernel, head_dim):
    q, k, v = triple(1, 4, 2, 100, head_dim, torch.float32)

    assert_allclose(attend(kernel, q, k, v), oracle(q, k, v))


@pytest.mark.parametrize("group_size", [1, 2, 8, 16])
def test_group_sizes(kernel, group_size):
    """Each KV head serving `G` query heads, including `G = 1` (plain MHA)."""
    num_kv_heads = 2
    q, k, v = triple(2, num_kv_heads * group_size, num_kv_heads, 100, 64, torch.float32)

    assert_allclose(attend(kernel, q, k, v), oracle(q, k, v))


def test_reads_a_strided_query_without_copying(kernel):
    """The last token's query, sliced out of a prefill-shaped tensor.

    Its head axis is strided by the *prompt* length rather than by `D`, so a kernel
    that assumed a contiguous layout would read a neighbouring token's query for
    every head but the first — and still return plausible numbers.
    """
    num_query_heads, num_kv_heads, head_dim = QWEN3_HEADS
    prefill = torch.randn(2, num_query_heads, 8, head_dim, device="cuda")
    q = prefill[:, :, -1:, :]
    _, k, v = triple(2, num_query_heads, num_kv_heads, 100, head_dim, torch.float32)

    assert not q.is_contiguous()
    assert_allclose(attend(kernel, q, k, v), oracle(q, k, v))


def test_reads_a_narrowed_cache_without_copying(kernel):
    """A paged or preallocated cache hands over a slice of a larger buffer.

    Making it contiguous would copy the entire cache on every decode step — the
    quadratic traffic the cache exists to avoid — so the kernel takes the stride.
    """
    num_query_heads, num_kv_heads, head_dim = QWEN3_HEADS
    reserved = torch.randn(1, num_kv_heads, 512, head_dim, device="cuda")
    k = reserved[:, :, :100]
    v = reserved[:, :, 200:300]
    q = torch.randn(1, num_query_heads, 1, head_dim, device="cuda")

    assert not k.is_contiguous()
    assert_allclose(attend(kernel, q, k, v), oracle(q, k, v))


# ---------------------------------------------------------------- the refusals


def test_rejects_a_multi_token_query(kernel):
    q = torch.randn(1, 4, 2, 64, device="cuda")
    _, k, v = triple(1, 4, 2, 100, 64, torch.float32)

    with pytest.raises(RuntimeError, match="L == 1 case"):
        attend(kernel, q, k, v)


def test_rejects_an_empty_cache(kernel):
    """Softmax over nothing is `0/0`, and returning a silent nan would be worse."""
    q = torch.randn(1, 4, 1, 64, device="cuda")
    empty = torch.randn(1, 2, 0, 64, device="cuda")

    with pytest.raises(RuntimeError, match="cache is empty"):
        attend(kernel, q, empty, empty)


def test_rejects_a_group_size_that_does_not_divide(kernel):
    q = torch.randn(1, 6, 1, 64, device="cuda")
    _, k, v = triple(1, 4, 4, 100, 64, torch.float32)

    with pytest.raises(RuntimeError, match="must be a multiple"):
        attend(kernel, q, k, v)


def test_rejects_mismatched_dtypes(kernel):
    q, k, v = triple(1, 4, 2, 100, 64, torch.float32)

    with pytest.raises(RuntimeError, match="must share a dtype"):
        attend(kernel, q, k, v.bfloat16())


def test_rejects_float64(kernel):
    q, k, v = triple(1, 4, 2, 100, 64, torch.float64)

    with pytest.raises(RuntimeError, match="float64 is not supported"):
        attend(kernel, q, k, v)


def test_rejects_a_cpu_tensor():
    kernel = load_extension()
    q = torch.randn(1, 4, 1, 64)

    with pytest.raises(RuntimeError, match="must be CUDA tensors"):
        kernel.decode_attention(q, q, q, 0.125)


# -------------------------------------------------------------------- the seam


def test_dispatch_claims_the_kernel():
    assert "decode_attention" in ops.cuda_kernel_names()

    report = ops.dispatch_report(use_cuda=True)
    line = next(row for row in report.splitlines() if "decode_attention" in row)
    assert line.split()[:2] == ["decode_attention", "cuda"], f"still routed to torch: {line!r}"


def test_dispatch_routes_decode_to_the_kernel(kernel):
    q, k, v = triple(1, 16, 8, 100, 128, torch.bfloat16)

    assert torch.equal(ops.attention(q, k, v, use_cuda=True), attend(kernel, q, k, v))


def test_dispatch_keeps_prefill_on_torch():
    """`L > 1` has no kernel until Step 3.5, and must not reach this one."""
    q = torch.randn(1, 16, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 8, 4, 128, device="cuda", dtype=torch.bfloat16)

    assert_allclose(
        ops.attention(q, k, k, mask="causal", use_cuda=True),
        oracle(q, k, k, mask="causal"),
    )


def test_dispatch_falls_back_on_an_explicit_mask(kernel):
    """A mask tensor can say anything, and this kernel attends to everything.

    The causal *shorthand* is fine for a single query token — it forbids nothing —
    but an arbitrary tensor has to go to the oracle rather than be ignored.
    """
    q, k, v = triple(1, 4, 2, 100, 64, torch.float32)
    mask = torch.zeros(1, 100, device="cuda")
    mask[0, 50:] = float("-inf")

    got = ops.attention(q, k, v, mask=mask, use_cuda=True)

    assert_allclose(got, oracle(q, k, v, mask=mask))
    assert not torch.allclose(got, attend(kernel, q, k, v)), "the mask was ignored"


def test_dispatch_accepts_the_causal_shorthand_for_decode(kernel):
    q, k, v = triple(1, 4, 2, 100, 64, torch.float32)

    assert torch.equal(ops.attention(q, k, v, mask="causal", use_cuda=True), attend(kernel, q, k, v))


def test_dispatch_falls_back_on_cpu_tensors():
    q = torch.randn(1, 4, 1, 64)
    k = torch.randn(1, 2, 100, 64)

    assert_allclose(ops.attention(q, k, k, use_cuda=True), oracle(q, k, k))


# ------------------------------------------------------------------ end to end


def test_model_decode_is_token_identical(tiny_qwen3, device):
    """The whole point: prefill on torch, decode on the kernel, same tokens."""
    theirs = tiny_qwen3.to(device=device, dtype=torch.float32)
    config, weights = config_from_hf(theirs), weights_from_hf(theirs)
    ids = torch.randint(0, TINY_QWEN3_DIMS["vocab_size"], (1, 5), device=device)

    torch_path = Qwen3Cached(config, weights, use_cuda=False)
    cuda_path = Qwen3Cached(config, weights, use_cuda=True)

    assert_tokens_equal(
        generate_ids_cached(cuda_path, ids, max_tokens=24),
        generate_ids_cached(torch_path, ids, max_tokens=24),
    )


def test_model_decode_agrees_in_bf16(tiny_qwen3, device):
    theirs = tiny_qwen3.to(device=device, dtype=torch.bfloat16)
    config, weights = config_from_hf(theirs), weights_from_hf(theirs)
    ids = torch.randint(0, TINY_QWEN3_DIMS["vocab_size"], (1, 5), device=device)

    torch_path = Qwen3Cached(config, weights, use_cuda=False)
    cuda_path = Qwen3Cached(config, weights, use_cuda=True)

    cache = cuda_path.create_kv_cache()
    reference_cache = torch_path.create_kv_cache()
    cuda_path(ids, cache)
    torch_path(ids, reference_cache)

    step = torch.tensor([[7]], device=device)
    assert_relative_error_below(cuda_path(step, cache), torch_path(step, reference_cache), 2e-2)


@pytest.mark.oracle
def test_real_model_is_token_identical():
    """Four kernels, the real 0.6B checkpoint, 48 greedy tokens. Same tokens.

    Longer than the earlier steps' runs on purpose: this is the first kernel whose
    work *grows* with the context, so a rescaling bug that is invisible inside one
    tile becomes visible once decode has walked past 64 cached keys.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mini_vllm.model.loader import resolve_model_path

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
