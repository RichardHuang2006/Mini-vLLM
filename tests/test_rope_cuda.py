"""Step 3.2 — the RoPE kernel against the PyTorch rotation it replaces.

The oracle is `mini_vllm.positional_encoding.apply_rope`, which Step 1.3 already
checked against HuggingFace's own rotary embedding. So the question here is only
whether the fused kernel gathers the same table rows and applies the same
rotation.

Position bookkeeping is what this step can get wrong, and it fails quietly: a
rotation by the wrong angle produces a plausible tensor, not an error. So the
tests lean on *positions the model will really pass* — a prefill's `arange`, a
decode step at a large offset, and positions that do not start at zero — plus the
one invariant the KV cache depends on: rotating a token at position `p` must give
the same answer whether it arrives in a prefill or as a single decode token.
"""

from __future__ import annotations

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

from mini_vllm.generate import generate_ids_cached
from mini_vllm.kernels import ops
from mini_vllm.kernels.extension import load_extension
from mini_vllm.positional_encoding import RoPE, apply_rope
from mini_vllm.model.qwen3_cached import Qwen3Cached

pytestmark = pytest.mark.cuda

HEAD_DIM = TINY_QWEN3_DIMS["head_dim"]
MAX_SEQ_LEN = 512
THETA = TINY_QWEN3_DIMS["rope_theta"]


@pytest.fixture
def kernel(device):
    return load_extension()


@pytest.fixture
def rope(device):
    """The Step 1.3 tables, on the GPU. The kernel reads these exact tensors."""
    return RoPE(HEAD_DIM, MAX_SEQ_LEN, THETA, device="cuda")


# --------------------------------------------------------------- the rotation


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [2, 8, 32, HEAD_DIM])
def test_matches_the_oracle(kernel, dtype, head_dim):
    tables = RoPE(head_dim, MAX_SEQ_LEN, THETA, device="cuda")
    x = torch.randn(2, 6, 4, head_dim, device="cuda", dtype=dtype)
    positions = torch.arange(6, device="cuda")

    assert_allclose(
        kernel.rope(x, positions, tables.cos, tables.sin),
        apply_rope(x, positions, tables.cos, tables.sin),
    )


@pytest.mark.parametrize("shape", [(1, 1, 16), (1, 128, 16), (4, 7, 8), (2, 3, 1)])
def test_matches_the_oracle_across_shapes(kernel, rope, shape):
    """`B x L x H x D`, including the two the engine actually runs.

    `(1, 1, 16)` is a decode step and `(1, 128, 16)` a prefill; the others exist
    to keep the index arithmetic honest when the head count is not a power of two
    or the batch is not one.
    """
    batch, length, heads = shape
    x = torch.randn(batch, length, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(length, device="cuda")

    assert_allclose(
        kernel.rope(x, positions, rope.cos, rope.sin),
        apply_rope(x, positions, rope.cos, rope.sin),
    )


# --------------------------------------------------------------- the positions


def test_positions_need_not_start_at_zero(kernel, rope):
    """A decode step at offset 500 is the normal case, not the exotic one."""
    x = torch.randn(1, 1, 16, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([500], device="cuda")

    assert_allclose(
        kernel.rope(x, positions, rope.cos, rope.sin),
        apply_rope(x, positions, rope.cos, rope.sin),
    )


def test_positions_need_not_be_contiguous(kernel, rope):
    """Shuffled positions prove the gather is a real lookup, not an `arange`.

    A kernel that quietly used its own token index would pass every test above
    and fail this one — and in Phase 4 the positions in a single batch genuinely
    are unordered, because a prefill chunk and several decodes share one pass.
    """
    x = torch.randn(2, 8, 4, HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.tensor([300, 0, 17, 511, 5, 128, 64, 2], device="cuda")

    assert_allclose(
        kernel.rope(x, positions, rope.cos, rope.sin),
        apply_rope(x, positions, rope.cos, rope.sin),
    )


def test_per_sequence_positions(kernel, rope):
    """`B x L` positions: each sequence sits at its own offset."""
    x = torch.randn(3, 4, 4, HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.tensor([[0, 1, 2, 3], [100, 101, 102, 103], [7, 9, 11, 13]], device="cuda")

    assert_allclose(
        kernel.rope(x, positions, rope.cos, rope.sin),
        apply_rope(x, positions, rope.cos, rope.sin),
    )


def test_positions_broadcast_over_the_batch(kernel, rope):
    """One `L`-long position vector applies to every sequence, as the oracle does."""
    x = torch.randn(4, 5, 2, HEAD_DIM, device="cuda", dtype=torch.float32)
    shared = torch.arange(5, device="cuda")

    from_kernel = kernel.rope(x, shared, rope.cos, rope.sin)
    per_sequence = kernel.rope(x, shared.expand(4, 5).contiguous(), rope.cos, rope.sin)

    assert torch.equal(from_kernel, per_sequence)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_accepts_both_integer_position_dtypes(kernel, rope, dtype):
    x = torch.randn(1, 4, 2, HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.arange(4, device="cuda", dtype=dtype)

    assert_allclose(
        kernel.rope(x, positions, rope.cos, rope.sin),
        apply_rope(x, positions, rope.cos, rope.sin),
    )


def test_a_decode_token_matches_its_place_in_a_prefill(kernel, rope):
    """The invariant the KV cache rests on.

    Keys are rotated before they are cached, so a token rotated once as part of a
    prefill and the same token arriving later as a single decode step must come
    out identical. If they differ, cached keys and fresh queries disagree about
    where they are, and the model degrades as the context grows rather than
    failing outright.
    """
    length = 12
    x = torch.randn(1, length, 4, HEAD_DIM, device="cuda", dtype=torch.float32)

    prefilled = kernel.rope(x, torch.arange(length, device="cuda"), rope.cos, rope.sin)
    for position in (0, 1, 7, length - 1):
        one_token = kernel.rope(
            x[:, position : position + 1],
            torch.tensor([position], device="cuda"),
            rope.cos,
            rope.sin,
        )
        assert torch.equal(one_token, prefilled[:, position : position + 1]), (
            f"position {position} rotates differently alone than in a prefill"
        )


# -------------------------------------------------------------- the properties


def test_position_zero_is_the_identity(kernel, rope):
    """Angle zero: cos is 1 and sin is 0, so the vector must come back unchanged."""
    x = torch.randn(1, 1, 4, HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.zeros(1, device="cuda", dtype=torch.int64)

    assert_allclose(kernel.rope(x, positions, rope.cos, rope.sin), x)


def test_rotation_preserves_pair_norms(kernel, rope):
    """It is a rotation, so every `(i, i + D/2)` pair keeps its length.

    This is the one property that holds independently of the oracle, which makes
    it the check that would survive a bug in the reference itself.
    """
    half = HEAD_DIM // 2
    x = torch.randn(2, 9, 4, HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.arange(9, device="cuda")

    rotated = kernel.rope(x, positions, rope.cos, rope.sin)

    before = x[..., :half].pow(2) + x[..., half:].pow(2)
    after = rotated[..., :half].pow(2) + rotated[..., half:].pow(2)
    assert_allclose(after, before)


def test_handles_a_non_contiguous_input(kernel, rope):
    """`B x H x L x D` transposed into `B x L x H x D` is not dense."""
    # B x H x L x D transposed to B x L x H x D: 6 tokens of 4 heads.
    x = torch.randn(2, 4, 6, HEAD_DIM, device="cuda", dtype=torch.float32).transpose(1, 2)
    positions = torch.arange(6, device="cuda")

    assert not x.is_contiguous()
    assert_allclose(
        kernel.rope(x, positions, rope.cos, rope.sin),
        apply_rope(x, positions, rope.cos, rope.sin),
    )


def test_handles_an_empty_input(kernel, rope):
    empty = torch.empty(0, 1, 4, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(1, device="cuda")

    assert kernel.rope(empty, positions, rope.cos, rope.sin).shape == empty.shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_returns_the_input_dtype(kernel, rope, dtype):
    x = torch.randn(1, 2, 4, HEAD_DIM, device="cuda", dtype=dtype)
    positions = torch.arange(2, device="cuda")

    assert kernel.rope(x, positions, rope.cos, rope.sin).dtype == dtype


# ---------------------------------------------------------------- the refusals


def test_rejects_float64(kernel, rope):
    x = torch.randn(1, 2, 4, HEAD_DIM, device="cuda", dtype=torch.float64)

    with pytest.raises(RuntimeError, match="float64 is not supported"):
        kernel.rope(x, torch.arange(2, device="cuda"), rope.cos, rope.sin)


def test_rejects_a_missing_head_axis(kernel, rope):
    """One position applies to every head of a token, so a head axis is required."""
    x = torch.randn(4, HEAD_DIM, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="a head axis is required"):
        kernel.rope(x, torch.arange(4, device="cuda"), rope.cos, rope.sin)


def test_rejects_an_odd_head_dimension(kernel):
    """Rotating halves needs halves."""
    tables = torch.randn(8, 6, device="cuda")
    x = torch.randn(1, 2, 2, 7, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="must be even"):
        kernel.rope(x, torch.arange(2, device="cuda"), tables, tables)


def test_rejects_bf16_tables(kernel, rope):
    """The tables stay fp32 even when activations do not — see Step 1.3."""
    x = torch.randn(1, 2, 4, HEAD_DIM, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="must be float32"):
        kernel.rope(
            x, torch.arange(2, device="cuda"), rope.cos.bfloat16(), rope.sin.bfloat16()
        )


def test_rejects_tables_of_the_wrong_width(kernel, rope):
    x = torch.randn(1, 2, 4, HEAD_DIM, device="cuda", dtype=torch.float32)
    narrow = rope.cos[:, : HEAD_DIM // 2].contiguous()

    with pytest.raises(RuntimeError, match="wide but the head dimension"):
        kernel.rope(x, torch.arange(2, device="cuda"), narrow, narrow)


def test_rejects_float_positions(kernel, rope):
    x = torch.randn(1, 2, 4, HEAD_DIM, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="positions must be int32 or int64"):
        kernel.rope(x, torch.zeros(2, device="cuda"), rope.cos, rope.sin)


def test_rejects_positions_that_cannot_broadcast(kernel, rope):
    """5 positions for 4 tokens is a bug, and silently ignoring one would hide it."""
    x = torch.randn(1, 4, 2, HEAD_DIM, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="cannot broadcast"):
        kernel.rope(x, torch.arange(5, device="cuda"), rope.cos, rope.sin)


def test_rejects_a_cpu_tensor(kernel):
    tables = RoPE(HEAD_DIM, MAX_SEQ_LEN, THETA)
    x = torch.randn(1, 2, 4, HEAD_DIM)

    with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
        kernel.rope(x, torch.arange(2), tables.cos, tables.sin)


# -------------------------------------------------------------------- the seam


def test_dispatch_claims_the_kernel():
    assert "rope" in ops.cuda_kernel_names()

    line = next(row for row in ops.dispatch_report(use_cuda=True).splitlines() if "rope" in row)
    assert line.split()[:2] == ["rope", "cuda"], f"the report still routes rope to torch: {line!r}"


def test_dispatch_routes_to_the_kernel(kernel, rope):
    x = torch.randn(2, 5, 4, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(5, device="cuda")

    assert torch.equal(
        ops.rope(x, positions, rope.cos, rope.sin, use_cuda=True),
        kernel.rope(x, positions, rope.cos, rope.sin),
    )


def test_dispatch_falls_back_on_cpu_tensors():
    tables = RoPE(HEAD_DIM, MAX_SEQ_LEN, THETA)
    x = torch.randn(1, 3, 2, HEAD_DIM)
    positions = torch.arange(3)

    assert_allclose(
        ops.rope(x, positions, tables.cos, tables.sin, use_cuda=True),
        apply_rope(x, positions, tables.cos, tables.sin),
    )


def test_dispatch_falls_back_on_non_fp32_tables(kernel, rope):
    """The kernel refuses bf16 tables, so the dispatch must not hand them over."""
    x = torch.randn(1, 3, 2, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(3, device="cuda")
    cos, sin = rope.cos.bfloat16(), rope.sin.bfloat16()

    assert_allclose(
        ops.rope(x, positions, cos, sin, use_cuda=True), apply_rope(x, positions, cos, sin)
    )


# ------------------------------------------------------------------ end to end


@pytest.fixture
def tiny_pair(tiny_qwen3, device):
    """The tiny model with kernels off and on, sharing weights."""

    def build(dtype: torch.dtype):
        theirs = tiny_qwen3.to(device=device, dtype=dtype)
        config, weights = config_from_hf(theirs), weights_from_hf(theirs)
        return (
            Qwen3Cached(config, weights, use_cuda=False),
            Qwen3Cached(config, weights, use_cuda=True),
        )

    return build


def test_model_greedy_output_is_token_identical(tiny_pair):
    """Now with two kernels live, so this is cumulative: RMSNorm *and* RoPE."""
    torch_path, cuda_path = tiny_pair(torch.float32)
    ids = torch.randint(0, TINY_QWEN3_DIMS["vocab_size"], (1, 5), device="cuda")

    assert_tokens_equal(
        generate_ids_cached(cuda_path, ids, max_tokens=16),
        generate_ids_cached(torch_path, ids, max_tokens=16),
    )


def test_model_logits_are_unchanged_in_bf16(tiny_pair):
    torch_path, cuda_path = tiny_pair(torch.bfloat16)
    ids = torch.randint(0, TINY_QWEN3_DIMS["vocab_size"], (2, 7), device="cuda")

    assert_relative_error_below(
        cuda_path(ids, cuda_path.create_kv_cache()),
        torch_path(ids, torch_path.create_kv_cache()),
        limit=KERNEL_DRIFT_LIMIT,
    )


@pytest.mark.oracle
def test_real_model_generates_the_same_text():
    """Qwen3-0.6B in fp32: 32 greedy tokens, identical, with both kernels on.

    RoPE is where a position bug would show up as a continuation that starts
    well and drifts, so a long-ish greedy run against the PyTorch path is the
    check that matters — much more than any tolerance on a single tensor.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mini_vllm.model.loader import resolve_model_path

    path = resolve_model_path()
    if not (path / "model.safetensors").is_file():
        pytest.skip("Qwen3-0.6B weights are not downloaded")

    hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to("cuda").eval()
    config, weights = config_from_hf(hf), weights_from_hf(hf)
    torch_path = Qwen3Cached(config, weights, use_cuda=False)
    cuda_path = Qwen3Cached(config, weights, use_cuda=True)

    tokenizer = AutoTokenizer.from_pretrained(path)
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.to("cuda")

    try:
        assert_tokens_equal(
            generate_ids_cached(cuda_path, ids, max_tokens=32),
            generate_ids_cached(torch_path, ids, max_tokens=32),
        )
    finally:
        del hf, weights, torch_path, cuda_path
        torch.cuda.empty_cache()
