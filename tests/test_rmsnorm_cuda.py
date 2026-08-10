"""Step 3.1 — the RMSNorm kernel against the PyTorch RMSNorm it replaces.

The oracle is `mini_vllm.layer_norm.rms_norm`, which Step 1.5 already checked
against HuggingFace. So this file never asks "is this the right formula" — that is
settled — only "does the kernel compute the same thing the settled formula does".
That is the whole differential-testing idea, and it is why the slow path is never
deleted.

Three classes of test earn their place here, because they are the three ways a
kernel of this shape goes wrong:

* **Widths.** The vectorized path needs the width to be a multiple of the vector
  width and the pointers 16-byte aligned; every awkward width must still be right
  through the fallback.
* **The reduction.** A block-wide sum has to survive rows narrower than a warp and
  rows wider than one chunk per thread, and it has to accumulate in fp32.
* **The seam.** A correct kernel that the model never actually calls is worth
  nothing, so the last section asserts the dispatch really routes to it and that
  the model's greedy output does not move.
"""

from __future__ import annotations

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

from mini_vllm.generate import generate_ids_cached
from mini_vllm.kernels import ops
from mini_vllm.kernels.extension import load_extension
from mini_vllm.layer_norm import rms_norm
from mini_vllm.model.qwen3_cached import Qwen3Cached

pytestmark = pytest.mark.cuda

EPS = 1e-6

# The widths Qwen3-0.6B actually norms over: `E` before attention and the MLP,
# `D` for QK-norm inside attention (DESIGN.md §3).
QWEN3_HIDDEN_SIZE = 1024
QWEN3_HEAD_DIM = 128


@pytest.fixture
def kernel(device):
    """The compiled extension, built on first use."""
    return load_extension()


def reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return rms_norm(x, weight, EPS)


# ------------------------------------------------------------------- the widths


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "dim",
    [
        1,  # narrower than one vector, and narrower than one lane of the reduction
        3,  # not a multiple of any vector width
        32,
        33,  # one past a warp's worth of scalars
        QWEN3_HEAD_DIM,
        QWEN3_HIDDEN_SIZE,
        3072,  # Qwen3's intermediate size
    ],
)
def test_matches_the_oracle(kernel, dtype, dim):
    x = torch.randn(8, dim, device="cuda", dtype=dtype)
    weight = torch.randn(dim, device="cuda", dtype=dtype)

    assert_allclose(kernel.rmsnorm(x, weight, EPS), reference(x, weight))


@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 3, 5), (2, 6, 4)])
def test_normalizes_over_the_last_dimension_only(kernel, shape):
    """Leading dimensions are just rows: `N.. x dim` must behave like `prod(N..)` rows.

    The 4-D case is the shape QK-norm passes in — `B x L x H x D` — so this is
    the real call, not a generalization for its own sake.
    """
    dim = QWEN3_HEAD_DIM
    x = torch.randn(*shape, dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)

    out = kernel.rmsnorm(x, weight, EPS)

    assert out.shape == x.shape
    assert_allclose(out, reference(x, weight))


def test_vectorized_and_fallback_paths_agree(kernel):
    """One width takes the 128-bit load path, the other cannot, on the same data.

    A bug in either path hides if only one is ever exercised, and which path runs
    is decided by a runtime check the caller cannot see.
    """
    wide = torch.randn(4, 1024, device="cuda", dtype=torch.bfloat16)
    narrow = wide[:, :1023].contiguous()

    for x in (wide, narrow):
        weight = torch.ones(x.shape[-1], device="cuda", dtype=torch.bfloat16)
        assert_allclose(kernel.rmsnorm(x, weight, EPS), reference(x, weight))


def test_handles_a_misaligned_input(kernel):
    """A view starting part way into its storage cannot take the 128-bit path.

    `h[:, -1:, :]` in the cached model produces exactly this, and a misaligned
    vector load does not merely run slowly — it faults. So the kernel checks the
    pointer rather than inferring alignment from the dtype and width.
    """
    dim = QWEN3_HIDDEN_SIZE
    storage = torch.randn(4 * dim + 1, device="cuda", dtype=torch.bfloat16)
    misaligned = storage[1:].view(4, dim)
    weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)

    assert misaligned.data_ptr() % 16 != 0, "this test needs a misaligned pointer"
    assert_allclose(kernel.rmsnorm(misaligned, weight, EPS), reference(misaligned, weight))


def test_handles_a_non_contiguous_input(kernel):
    """A transposed view must not be read as if it were dense."""
    dim = QWEN3_HEAD_DIM
    transposed = torch.randn(dim, 16, device="cuda", dtype=torch.float32).t()
    weight = torch.randn(dim, device="cuda", dtype=torch.float32)

    out = kernel.rmsnorm(transposed, weight, EPS)

    assert out.shape == transposed.shape
    assert_allclose(out, reference(transposed, weight))


def test_handles_an_empty_input(kernel):
    empty = torch.empty(0, QWEN3_HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(QWEN3_HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)

    assert kernel.rmsnorm(empty, weight, EPS).shape == empty.shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_returns_the_input_dtype(kernel, dtype):
    x = torch.randn(2, QWEN3_HIDDEN_SIZE, device="cuda", dtype=dtype)
    weight = torch.ones(QWEN3_HIDDEN_SIZE, device="cuda", dtype=dtype)

    assert kernel.rmsnorm(x, weight, EPS).dtype == dtype


# ---------------------------------------------------------------- the reduction


@pytest.mark.parametrize("dim", [16384, 20480])
def test_handles_rows_wider_than_one_chunk_per_thread(kernel, dim):
    """Past 1024 chunks a block runs out of threads and has to loop.

    That loop is also the path where the row no longer fits in registers and the
    second pass re-reads it, so both halves of the kernel change behaviour here.
    """
    x = torch.randn(2, dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)

    assert_allclose(kernel.rmsnorm(x, weight, EPS), reference(x, weight))


def test_many_rows(kernel):
    """One block per row means the row count is the grid; 4096 blocks must all run."""
    x = torch.randn(4096, QWEN3_HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(QWEN3_HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)

    assert_allclose(kernel.rmsnorm(x, weight, EPS), reference(x, weight))


def test_unit_weight_makes_rms_one(kernel):
    """The property, checked on the kernel rather than inferred from the oracle."""
    dim = QWEN3_HIDDEN_SIZE
    x = torch.randn(8, dim, device="cuda", dtype=torch.float32) * 17.0

    out = kernel.rmsnorm(x, torch.ones(dim, device="cuda"), EPS)

    assert_allclose(out.pow(2).mean(dim=-1).sqrt(), torch.ones(8, device="cuda"))


def test_a_zero_row_stays_finite(kernel):
    """`eps` is the only thing standing between a zero row and a division by zero."""
    dim = QWEN3_HIDDEN_SIZE
    zeros = torch.zeros(2, dim, device="cuda", dtype=torch.bfloat16)

    out = kernel.rmsnorm(zeros, torch.ones(dim, device="cuda", dtype=torch.bfloat16), EPS)

    assert torch.isfinite(out).all()
    assert_allclose(out, zeros)


def test_reduction_accumulates_in_fp32(kernel):
    """Summing 4096 bf16 squares in bf16 is visibly wrong, so it must not happen.

    The kernel's storage is bf16 and its accumulator is fp32 (DESIGN.md §5.3).
    This pins that down by showing the kernel agrees with the fp32 reduction and
    that the bf16 reduction is a different number — otherwise the test would pass
    for a kernel that reduced in the wrong precision.
    """
    dim = 4096
    x = torch.randn(4, dim, device="cuda", dtype=torch.bfloat16) * 3.0
    weight = torch.ones(dim, device="cuda", dtype=torch.bfloat16)

    ours = kernel.rmsnorm(x, weight, EPS)
    in_bf16 = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)

    assert_allclose(ours, reference(x, weight))
    assert not torch.equal(in_bf16, reference(x, weight)), (
        "the bf16 reduction happened to be exact here, so this test proves nothing"
    )


def test_eps_is_applied_inside_the_square_root(kernel):
    """A tiny row is where `eps`'s position in the expression shows up at all."""
    dim = 64
    x = torch.full((1, dim), 1e-4, device="cuda", dtype=torch.float32)
    weight = torch.ones(dim, device="cuda")

    for eps in (1e-6, 1e-2):
        assert_allclose(kernel.rmsnorm(x, weight, eps), rms_norm(x, weight, eps))


# ------------------------------------------------------------------ the refusals


def test_rejects_float64(kernel):
    """fp64 in an fp32 accumulator would be a silent precision loss, so it errors."""
    x = torch.randn(2, 8, device="cuda", dtype=torch.float64)
    weight = torch.ones(8, device="cuda", dtype=torch.float64)

    with pytest.raises(RuntimeError, match="float64 is not supported"):
        kernel.rmsnorm(x, weight, EPS)


def test_rejects_a_mismatched_weight_dtype(kernel):
    """PyTorch would promote; a kernel that guessed would return the wrong dtype."""
    x = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="but weight is"):
        kernel.rmsnorm(x, torch.ones(8, device="cuda", dtype=torch.float32), EPS)


def test_rejects_a_mismatched_weight_width(kernel):
    x = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="last dimension"):
        kernel.rmsnorm(x, torch.ones(9, device="cuda", dtype=torch.bfloat16), EPS)


def test_rejects_a_multi_dimensional_weight(kernel):
    x = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="weight must be 1-D"):
        kernel.rmsnorm(x, torch.ones(2, 8, device="cuda", dtype=torch.bfloat16), EPS)


def test_rejects_a_cpu_tensor(kernel):
    with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
        kernel.rmsnorm(torch.randn(2, 8), torch.ones(8), EPS)


# --------------------------------------------------------------------- the seam


def test_dispatch_claims_the_kernel():
    assert "rmsnorm" in ops.cuda_kernel_names()

    line = next(
        row for row in ops.dispatch_report(use_cuda=True).splitlines() if "rmsnorm" in row
    )
    assert line.split()[:2] == ["rmsnorm", "cuda"], (
        f"the report still routes rmsnorm to torch: {line!r}"
    )


def test_dispatch_routes_to_the_kernel(kernel):
    """`use_cuda=True` must produce the kernel's answer, bit for bit.

    Comparing against the oracle would pass even if the flag did nothing, which
    is the failure mode this catches.
    """
    dim = QWEN3_HIDDEN_SIZE
    x = torch.randn(4, dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)

    assert torch.equal(
        ops.rmsnorm(x, weight, EPS, use_cuda=True), kernel.rmsnorm(x, weight, EPS)
    )


def test_dispatch_falls_back_on_cpu_tensors():
    """The same model object serves CPU tests, so the flag cannot imply a device."""
    dim = 64
    x = torch.randn(4, dim)
    weight = torch.randn(dim)

    assert_allclose(ops.rmsnorm(x, weight, EPS, use_cuda=True), rms_norm(x, weight, EPS))


def test_dispatch_falls_back_on_a_mismatched_weight_dtype(kernel):
    """The kernel refuses mixed dtypes, so the dispatch must not hand them over."""
    dim = 64
    x = torch.randn(4, dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, device="cuda", dtype=torch.float32)

    assert_allclose(ops.rmsnorm(x, weight, EPS, use_cuda=True), rms_norm(x, weight, EPS))


# ------------------------------------------------------------------ end to end


@pytest.fixture
def tiny_pair(tiny_qwen3, device):
    """The tiny model twice over, sharing weights: kernels off and kernels on."""

    def build(dtype: torch.dtype):
        theirs = tiny_qwen3.to(device=device, dtype=dtype)
        config, weights = config_from_hf(theirs), weights_from_hf(theirs)
        return (
            Qwen3Cached(config, weights, use_cuda=False),
            Qwen3Cached(config, weights, use_cuda=True),
        )

    return build


def test_model_logits_are_unchanged(tiny_pair):
    """In fp32 there is headroom for the reduction order to differ and nothing else."""
    torch_path, cuda_path = tiny_pair(torch.float32)
    ids = torch.randint(0, TINY_QWEN3_DIMS["vocab_size"], (2, 7), device="cuda")

    assert_allclose(
        cuda_path(ids, cuda_path.create_kv_cache()),
        torch_path(ids, torch_path.create_kv_cache()),
    )


def test_model_greedy_output_is_token_identical(tiny_pair):
    """The claim Step 3.1 actually has to make: the kernel changed speed, not text."""
    torch_path, cuda_path = tiny_pair(torch.float32)
    ids = torch.randint(0, TINY_QWEN3_DIMS["vocab_size"], (1, 5), device="cuda")

    assert_tokens_equal(
        generate_ids_cached(cuda_path, ids, max_tokens=16),
        generate_ids_cached(torch_path, ids, max_tokens=16),
    )


@pytest.mark.oracle
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_model_generates_the_same_text(dtype):
    """Qwen3-0.6B, 28 layers deep, with the kernel doing all 57 of its norms.

    The tiny model can hide an error that only compounds: it has 2 layers, and
    RMSNorm runs five times in each plus once at the end. This is the version of
    the claim that counts, and it is run in both dtypes because they fail
    differently — fp32 has the headroom for token identity, while bf16 is where a
    reassociated sum can legally move the last bit (PLAN.md "Numerical
    tolerances"), so it gets the drift check instead.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mini_vllm.model.loader import resolve_model_path

    path = resolve_model_path()
    if not (path / "model.safetensors").is_file():
        pytest.skip("Qwen3-0.6B weights are not downloaded")

    hf = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to("cuda").eval()
    config, weights = config_from_hf(hf), weights_from_hf(hf)
    torch_path = Qwen3Cached(config, weights, use_cuda=False)
    cuda_path = Qwen3Cached(config, weights, use_cuda=True)

    tokenizer = AutoTokenizer.from_pretrained(path)
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.to("cuda")

    try:
        if dtype is torch.float32:
            assert_tokens_equal(
                generate_ids_cached(cuda_path, ids, max_tokens=32),
                generate_ids_cached(torch_path, ids, max_tokens=32),
            )
        else:
            assert_relative_error_below(
                cuda_path(ids, cuda_path.create_kv_cache()),
                torch_path(ids, torch_path.create_kv_cache()),
                limit=1e-2,
            )
    finally:
        del hf, weights, torch_path, cuda_path
        torch.cuda.empty_cache()


def test_model_logits_are_unchanged_in_bf16(tiny_pair):
    """bf16 gets the drift instrument, not the exact one.

    The kernel sums a row in a different order than PyTorch does, which is legal
    and unavoidable, so in bf16 the last bit may differ — see PLAN.md
    "Numerical tolerances". What must hold is that the difference stays at the
    rounding floor instead of growing.
    """
    torch_path, cuda_path = tiny_pair(torch.bfloat16)
    ids = torch.randint(0, TINY_QWEN3_DIMS["vocab_size"], (2, 7), device="cuda")

    assert_relative_error_below(
        cuda_path(ids, cuda_path.create_kv_cache()),
        torch_path(ids, torch_path.create_kv_cache()),
        limit=1e-3,
    )
