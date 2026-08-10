"""Step 3.5 — flash prefill against the Step 1.4 oracle.

Prefill adds one thing decode did not have: a mask. So most of these tests are
about the diagonal, because that is where the two failure modes live, and they are
not symmetric.

* Masking **too much** loses information and shows up as a tolerance failure —
  loud, and caught by any comparison.
* Masking **too little** lets a token attend to its own future. That is silent:
  the output is a perfectly well-formed convex combination, prefill logits look
  plausible, and the only symptom is a model that generates worse text than it
  should. `test_a_query_cannot_see_its_own_future` checks it directly rather than
  through a tolerance, by perturbing a later key and asserting an earlier query's
  output does not move at all.

The tile is 16 wide, so lengths are tested at 16, 17, 31, 32 and 33 — the plan's
"one token past a tile boundary" in both directions.
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
QUERY_TILE = 16


@pytest.fixture
def kernel(device):
    return load_extension()


def attend(kernel, q, k, v):
    return kernel.flash_prefill(q, k, v, 1.0 / math.sqrt(q.shape[-1]))


def triple(batch, num_query_heads, num_kv_heads, query_len, source_len, head_dim, dtype):
    q = torch.randn(batch, num_query_heads, query_len, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, num_kv_heads, source_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, num_kv_heads, source_len, head_dim, device="cuda", dtype=dtype)
    return q, k, v


# ------------------------------------------------------------------ the values


def test_matches_the_oracle_at_qwen3_shapes(kernel):
    num_query_heads, num_kv_heads, head_dim = QWEN3_HEADS
    q, k, v = triple(2, num_query_heads, num_kv_heads, 128, 128, head_dim, torch.float32)

    assert_allclose(attend(kernel, q, k, v), oracle(q, k, v, mask="causal"))


@pytest.mark.parametrize("dtype,one_ulp", [(torch.bfloat16, 2**-8), (torch.float16, 2**-11)])
def test_low_precision_disagrees_by_exactly_one_rounding(kernel, dtype, one_ulp):
    """The aggregate norm rather than an elementwise bound, and for a real reason.

    The oracle evaluates attention as separate ops, so it rounds the scores *and*
    the softmax weights back to `dtype` before `P·V`. The kernel never leaves fp32
    until the store. It is therefore slightly **more** accurate than its own
    reference, and the gap is not error to be tolerated but a measurable quantity:
    one ULP of the dtype, which is what this asserts.

    An elementwise `atol` would report something different and less useful. On a
    handful of the 524288 outputs, cancellation leaves a result near zero where a
    one-ULP difference in the inputs is a 10% *relative* difference — a fact about
    those elements' magnitude, not about either implementation.
    """
    num_query_heads, num_kv_heads, head_dim = QWEN3_HEADS
    q, k, v = triple(2, num_query_heads, num_kv_heads, 128, 128, head_dim, dtype)

    assert_relative_error_below(attend(kernel, q, k, v), oracle(q, k, v, mask="causal"), 1.5 * one_ulp)


@pytest.mark.parametrize("query_len", [1, 2, 15, 16, 17, 31, 32, 33, 64, 100, 129, 512])
def test_every_query_tile_boundary(kernel, query_len):
    """Square prefill at, around and far past the 16-wide query tile.

    A partial *last* query tile is the common case — prompts are not multiples of
    16 — and its threads have no query to own. They must contribute nothing rather
    than reading a neighbouring row or writing past the end of the output.
    """
    num_query_heads, num_kv_heads, head_dim = QWEN3_HEADS
    q, k, v = triple(1, num_query_heads, num_kv_heads, query_len, query_len, head_dim, torch.float32)

    assert_allclose(attend(kernel, q, k, v), oracle(q, k, v, mask="causal"))


@pytest.mark.parametrize("query_len,source_len", [(1, 64), (16, 64), (17, 100), (32, 33), (48, 512)])
def test_a_prefix_already_in_the_cache(kernel, query_len, source_len):
    """`S > L`: the queries are the *last* `L` positions, not the first.

    This is chunked prefill's shape (Step 4.4) and prefix reuse's shape. The
    diagonal shifts right by `S - L`, so a kernel that assumes a square lower
    triangle gets a mask that is far too tight and every one of these fails.
    """
    num_query_heads, num_kv_heads, head_dim = QWEN3_HEADS
    q, k, v = triple(
        1, num_query_heads, num_kv_heads, query_len, source_len, head_dim, torch.float32
    )

    assert_allclose(attend(kernel, q, k, v), oracle(q, k, v, mask="causal"))


def test_a_query_cannot_see_its_own_future(kernel):
    """The direct test of the mask, with no tolerance involved.

    Perturb key 20 and value 20. Query 20 and later must change; queries 0..19
    must be **bitwise** unchanged, because their weight on that position is
    exactly zero rather than merely small. A kernel that skips the diagonal tile's
    per-element mask — or computes the diagonal one column too far — fails here
    while still passing a tolerance comparison against its own wrong reference.
    """
    q, k, v = triple(1, 4, 2, 48, 48, 64, torch.float32)

    before = attend(kernel, q, k, v)
    k[:, :, 20] += 5.0
    v[:, :, 20] += 5.0
    after = attend(kernel, q, k, v)

    assert torch.equal(before[:, :, :20], after[:, :, :20]), "an earlier query saw a later key"
    assert not torch.equal(before[:, :, 20:], after[:, :, 20:])


def test_the_first_query_is_exactly_the_first_value(kernel):
    """Query 0 attends to key 0 alone, so softmax over one element is 1.0.

    Which makes the expected output not "close to" `v[0]` but equal to it, and
    makes this the one prefill case with no arithmetic in it to get wrong.
    """
    q, k, v = triple(1, 4, 2, 32, 32, 64, torch.float32)

    got = attend(kernel, q, k, v)

    assert_allclose(got[:, :, 0], v[:, :, 0].repeat_interleave(2, dim=1))


def test_matches_decode_for_the_last_position(kernel):
    """The last row of a prefill is the same computation a decode step does.

    Two kernels, two entirely different decompositions — tiled queries with a
    diagonal mask versus a split key axis with none — and they must agree on the
    one row where their problems coincide. Nothing else in the suite checks the
    two Phase 3 attention kernels against *each other*.
    """
    num_query_heads, num_kv_heads, head_dim = QWEN3_HEADS
    q, k, v = triple(1, num_query_heads, num_kv_heads, 40, 40, head_dim, torch.float32)

    prefilled = attend(kernel, q, k, v)
    decoded = kernel.decode_attention(
        q[:, :, -1:], k, v, 1.0 / math.sqrt(head_dim)
    )

    assert_relative_error_below(prefilled[:, :, -1:], decoded, 1e-5)


def test_output_is_a_convex_combination_of_visible_values(kernel):
    """Every output must lie inside the range of the values it may attend to."""
    q, k, v = triple(1, 4, 2, 64, 64, 64, torch.float32)

    got = attend(kernel, q, k, v)

    for position in (0, 1, 17, 63):
        visible = v[:, :, : position + 1]
        low = visible.amin(dim=2).repeat_interleave(2, dim=1)
        high = visible.amax(dim=2).repeat_interleave(2, dim=1)
        row = got[:, :, position]
        assert (row >= low - 1e-5).all() and (row <= high + 1e-5).all()


@pytest.mark.parametrize("head_dim", [32, 64, 128, 192])
def test_head_dimensions(kernel, head_dim):
    q, k, v = triple(1, 4, 2, 40, 40, head_dim, torch.float32)

    assert_allclose(attend(kernel, q, k, v), oracle(q, k, v, mask="causal"))


@pytest.mark.parametrize("group_size", [1, 2, 8, 16])
def test_group_sizes(kernel, group_size):
    num_kv_heads = 2
    q, k, v = triple(2, num_kv_heads * group_size, num_kv_heads, 40, 40, 64, torch.float32)

    assert_allclose(attend(kernel, q, k, v), oracle(q, k, v, mask="causal"))


def test_handles_logits_that_would_overflow(kernel):
    head_dim = 128
    q = torch.full((1, 2, 40, head_dim), 30.0, device="cuda")
    k = torch.full((1, 2, 40, head_dim), 30.0, device="cuda")
    v = torch.randn(1, 2, 40, head_dim, device="cuda")

    got = attend(kernel, q, k, v)

    assert torch.isfinite(got).all(), "the running max is not being subtracted"
    assert_allclose(got, oracle(q, k, v, mask="causal"))


def test_reads_a_strided_query_without_copying(kernel):
    """`B x L x H x D -> B x H x L x D` is how the model hands the query over."""
    num_query_heads, num_kv_heads, head_dim = QWEN3_HEADS
    q = torch.randn(1, 40, num_query_heads, head_dim, device="cuda").transpose(1, 2)
    _, k, v = triple(1, num_query_heads, num_kv_heads, 40, 40, head_dim, torch.float32)

    assert not q.is_contiguous()
    assert_allclose(attend(kernel, q, k, v), oracle(q, k, v, mask="causal"))


# ---------------------------------------------------------------- the refusals


def test_rejects_more_queries_than_keys(kernel):
    """`L > S` has no causal reading: some query would attend to nothing."""
    q, k, v = triple(1, 4, 2, 40, 20, 64, torch.float32)

    with pytest.raises(RuntimeError, match="S >= L"):
        attend(kernel, q, k, v)


def test_rejects_a_head_dim_it_cannot_stage(kernel):
    q, k, v = triple(1, 4, 2, 40, 40, 256, torch.float32)

    with pytest.raises(RuntimeError, match="exceeds 192"):
        attend(kernel, q, k, v)


def test_rejects_mismatched_dtypes(kernel):
    q, k, v = triple(1, 4, 2, 40, 40, 64, torch.float32)

    with pytest.raises(RuntimeError, match="must share a dtype"):
        attend(kernel, q, k, v.bfloat16())


def test_rejects_float64(kernel):
    q, k, v = triple(1, 4, 2, 40, 40, 64, torch.float64)

    with pytest.raises(RuntimeError, match="float64 is not supported"):
        attend(kernel, q, k, v)


def test_rejects_a_cpu_tensor():
    kernel = load_extension()
    q = torch.randn(1, 4, 8, 64)

    with pytest.raises(RuntimeError, match="must be CUDA tensors"):
        kernel.flash_prefill(q, q, q, 0.125)


# -------------------------------------------------------------------- the seam


def test_dispatch_deliberately_keeps_prefill_on_torch(kernel):
    """This kernel is correct and is **not** the default, on purpose.

    Its inner loop is scalar FMA against cuBLAS on tensor cores: 0.4x at L=512, so
    routing to it would make time-to-first-token worse in exchange for nothing.
    Step 3.6 replaces the loop with `mma.sync` and is what flips it. What this test
    pins is that the choice is *stated* — the op appears in the dispatch report as
    `torch` with the reason attached, rather than being quietly absent.
    """
    assert "flash_prefill" in ops.cuda_kernel_names(), "the kernel exists and is claimed"

    line = next(
        row for row in ops.dispatch_report(use_cuda=True).splitlines() if "flash_prefill" in row
    )
    assert line.split()[:2] == ["flash_prefill", "torch"]
    assert "slower" in line and "Step 3.6" in line, f"routed away with no reason: {line!r}"

    q, k, v = triple(1, 16, 8, 40, 40, 128, torch.bfloat16)
    assert_allclose(
        ops.attention(q, k, v, mask="causal", use_cuda=True), oracle(q, k, v, mask="causal")
    )


@pytest.fixture
def prefill_preferred(monkeypatch):
    """Route prefill to the kernel, as Step 3.6 will once it is worth preferring.

    Everything the kernel promises has to keep holding through the dispatch and
    through the model, or flipping that one entry later becomes an unbounded
    debugging session instead of a one-line change.
    """
    without = dict(ops.NOT_YET_FASTER)
    without.pop("flash_prefill")
    monkeypatch.setattr(ops, "NOT_YET_FASTER", without)


def test_dispatch_reaches_the_kernel_once_preferred(kernel, prefill_preferred):
    q, k, v = triple(1, 16, 8, 40, 40, 128, torch.bfloat16)

    assert torch.equal(ops.attention(q, k, v, mask="causal", use_cuda=True), attend(kernel, q, k, v))


def test_unmasked_prefill_stays_on_torch_even_when_preferred(kernel, prefill_preferred):
    """No mask at all is a *different* problem, not a simpler one.

    The kernel always masks causally, so an unmasked multi-token call has to reach
    the oracle whatever the routing says. This is the mirror of the decode case,
    where `None` and `"causal"` ask for the same thing and either may take the
    kernel.
    """
    q, k, v = triple(1, 4, 2, 40, 40, 64, torch.float32)

    got = ops.attention(q, k, v, use_cuda=True)

    assert_allclose(got, oracle(q, k, v))
    assert not torch.allclose(got, attend(kernel, q, k, v)), "the kernel masked an unmasked call"


def test_dispatch_falls_back_on_a_wide_head_dim():
    q, k, v = triple(1, 4, 2, 40, 40, 256, torch.float32)

    assert_allclose(
        ops.attention(q, k, v, mask="causal", use_cuda=True), oracle(q, k, v, mask="causal")
    )


def test_dispatch_falls_back_on_cpu_tensors():
    q = torch.randn(1, 4, 8, 64)
    k = torch.randn(1, 2, 8, 64)

    assert_allclose(
        ops.attention(q, k, k, mask="causal", use_cuda=True), oracle(q, k, k, mask="causal")
    )


# ------------------------------------------------------------------ end to end


def test_model_prefill_is_token_identical(tiny_qwen3, device, prefill_preferred):
    """All five kernels, prefill included, against the pure-PyTorch path.

    Uses `prefill_preferred`, since the engine's default routing sends prefill to
    cuBLAS: without it this would be a test of the other four kernels.
    """
    theirs = tiny_qwen3.to(device=device, dtype=torch.float32)
    config, weights = config_from_hf(theirs), weights_from_hf(theirs)
    ids = torch.randint(0, TINY_QWEN3_DIMS["vocab_size"], (2, 40), device=device)

    torch_path = Qwen3Cached(config, weights, use_cuda=False)
    cuda_path = Qwen3Cached(config, weights, use_cuda=True)

    assert_tokens_equal(
        generate_ids_cached(cuda_path, ids, max_tokens=16),
        generate_ids_cached(torch_path, ids, max_tokens=16),
    )


def test_a_chunked_prompt_matches_one_pass(tiny_qwen3, device, prefill_preferred):
    """Prefill in two chunks against prefill in one, which is Step 4.4's contract.

    The second chunk is a query length of 24 against a cache of 40 — the `S > L`
    shape — so this is where a diagonal that ignores the `S - L` offset shows up as
    a chunk boundary that ruins the continuation. Worth having here rather than
    only in Phase 4: if it fails, the kernel is the reason.
    """
    theirs = tiny_qwen3.to(device=device, dtype=torch.float32)
    config, weights = config_from_hf(theirs), weights_from_hf(theirs)
    ids = torch.randint(0, TINY_QWEN3_DIMS["vocab_size"], (1, 40), device=device)

    model = Qwen3Cached(config, weights, use_cuda=True)

    whole = model(ids, model.create_kv_cache())

    chunked_cache = model.create_kv_cache()
    first = model(ids[:, :16], chunked_cache)
    second = model(ids[:, 16:], chunked_cache)

    assert_allclose(torch.cat([first, second], dim=1), whole, kind="model")


@pytest.mark.oracle
def test_real_model_generates_the_same_text(prefill_preferred):
    """Qwen3-0.6B, every kernel live, 48 greedy tokens against the oracle path."""
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
