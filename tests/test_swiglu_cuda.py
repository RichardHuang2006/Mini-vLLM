"""The fused SwiGLU kernel against `silu(gate) * up`.

The op is trivial arithmetic, so the interesting content is not "is the formula
right" but **where the rounding happens**. PyTorch evaluates three separate
elementwise ops and each one lands back in the input dtype; a kernel that carried
fp32 through to the store would be slightly *more* accurate than its own oracle,
and a differential test cannot distinguish "better" from "wrong". Matching the
rounding points instead makes the comparison exact — which is why the headline
test here asserts `torch.equal` rather than a tolerance.
"""

from __future__ import annotations

import pytest
import torch
from conftest import (
    TINY_QWEN3_DIMS,
    assert_allclose,
    assert_tokens_equal,
    config_from_hf,
    weights_from_hf,
)

from mini_vllm.basics import silu
from mini_vllm.generate import generate_ids_cached
from mini_vllm.kernels import ops
from mini_vllm.kernels.extension import load_extension
from mini_vllm.model.qwen3_cached import Qwen3Cached

pytestmark = pytest.mark.cuda

QWEN3_INTERMEDIATE = 3072  # three times the hidden size: the widest activation


@pytest.fixture
def kernel(device):
    return load_extension()


# ------------------------------------------------------------------ the values


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, QWEN3_INTERMEDIATE), (128, QWEN3_INTERMEDIATE), (7, 13)])
def test_is_bitwise_identical_to_the_oracle(kernel, dtype, shape):
    """Not "within tolerance" — identical, because the rounding points match.

    This is the strongest form a kernel test can take, and it is available here
    only because the op has no reduction: nothing is summed, so there is no order
    to reassociate. A kernel that reduces gives this up and settles for a tolerance.
    """
    gate = torch.randn(*shape, device="cuda", dtype=dtype) * 3.0
    up = torch.randn(*shape, device="cuda", dtype=dtype)

    assert torch.equal(kernel.swiglu(gate, up), silu(gate) * up)


@pytest.mark.parametrize("shape", [(4,), (2, 3, 5), (2, 6, 4, 8)])
def test_shape_is_irrelevant(kernel, shape):
    """Pure elementwise: only the element count matters, never the layout."""
    gate = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    out = kernel.swiglu(gate, up)

    assert out.shape == gate.shape
    assert torch.equal(out, silu(gate) * up)


def test_handles_large_magnitudes(kernel):
    """`sigmoid` saturates rather than overflowing, at both ends.

    `exp(-x)` for very negative `x` is where a naive sigmoid returns `inf` and
    then `0/inf = nan`. The saturating values are the ones worth pinning, since
    real activations do reach them.
    """
    extreme = torch.tensor([[-100.0, -20.0, 0.0, 20.0, 100.0]], device="cuda")
    up = torch.ones_like(extreme)

    out = kernel.swiglu(extreme, up)

    assert torch.isfinite(out).all()
    assert_allclose(out, silu(extreme) * up)


def test_is_not_symmetric_in_its_arguments(kernel):
    """`gate` is activated and `up` is not, so the order of the two is meaningful.

    Swapping them is a real bug that every shape and finiteness check would miss.
    """
    gate = torch.randn(4, 64, device="cuda", dtype=torch.float32)
    up = torch.randn(4, 64, device="cuda", dtype=torch.float32)

    assert not torch.equal(kernel.swiglu(gate, up), kernel.swiglu(up, gate))


def test_vectorized_and_fallback_paths_agree(kernel):
    """A width that is a multiple of 8 takes the 128-bit path; 3071 cannot."""
    wide = torch.randn(4, 3072, device="cuda", dtype=torch.bfloat16)
    narrow = wide[:, :3071].contiguous()

    for gate in (wide, narrow):
        up = torch.randn_like(gate)
        assert torch.equal(kernel.swiglu(gate, up), silu(gate) * up)


def test_handles_a_misaligned_input(kernel):
    """An offset view cannot take the 128-bit path, and must still be right."""
    storage = torch.randn(4 * 3072 + 1, device="cuda", dtype=torch.bfloat16)
    gate = storage[1:].view(4, 3072)
    up = torch.randn_like(gate)

    assert gate.data_ptr() % 16 != 0, "this test needs a misaligned pointer"
    assert torch.equal(kernel.swiglu(gate, up), silu(gate) * up)


def test_handles_a_non_contiguous_input(kernel):
    transposed = torch.randn(64, 8, device="cuda", dtype=torch.float32).t()
    up = torch.randn(8, 64, device="cuda", dtype=torch.float32)

    assert torch.equal(kernel.swiglu(transposed, up), silu(transposed) * up)


def test_handles_an_empty_input(kernel):
    empty = torch.empty(0, QWEN3_INTERMEDIATE, device="cuda", dtype=torch.bfloat16)

    assert kernel.swiglu(empty, empty).shape == empty.shape


# ---------------------------------------------------------------- the refusals


def test_rejects_mismatched_shapes(kernel):
    gate = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="must match"):
        kernel.swiglu(gate, torch.randn(4, 65, device="cuda", dtype=torch.bfloat16))


def test_rejects_mismatched_dtypes(kernel):
    gate = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="but up is"):
        kernel.swiglu(gate, torch.randn(4, 64, device="cuda", dtype=torch.float32))


def test_rejects_float64(kernel):
    gate = torch.randn(4, 64, device="cuda", dtype=torch.float64)

    with pytest.raises(RuntimeError, match="float64 is not supported"):
        kernel.swiglu(gate, gate)


def test_rejects_a_cpu_tensor(kernel):
    with pytest.raises(RuntimeError, match="must be CUDA tensors"):
        kernel.swiglu(torch.randn(4, 8), torch.randn(4, 8))


# -------------------------------------------------------------------- the seam


def test_dispatch_claims_the_kernel():
    assert "swiglu" in ops.cuda_kernel_names()

    line = next(row for row in ops.dispatch_report(use_cuda=True).splitlines() if "swiglu" in row)
    assert line.split()[:2] == ["swiglu", "cuda"], f"still routed to torch: {line!r}"


def test_dispatch_routes_to_the_kernel(kernel):
    gate = torch.randn(4, QWEN3_INTERMEDIATE, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)

    assert torch.equal(ops.swiglu(gate, up, use_cuda=True), kernel.swiglu(gate, up))


def test_dispatch_falls_back_on_cpu_tensors():
    gate, up = torch.randn(4, 64), torch.randn(4, 64)

    assert_allclose(ops.swiglu(gate, up, use_cuda=True), silu(gate) * up)


# ------------------------------------------------------------------ end to end


def test_model_output_is_bitwise_unchanged(tiny_qwen3, device):
    """Alone among the elementwise kernels, this one is exact inside the model.

    RMSNorm and RoPE only promise token identity, because both reduce or gather
    and PyTorch reduces in a different order. SwiGLU has no reduction, so the
    whole model's logits must come out *bitwise* the same — a much sharper
    statement, and one no kernel that sums over the cache can make.
    """
    theirs = tiny_qwen3.to(device=device, dtype=torch.float32)
    config, weights = config_from_hf(theirs), weights_from_hf(theirs)
    ids = torch.randint(0, TINY_QWEN3_DIMS["vocab_size"], (2, 7), device=device)

    # Only SwiGLU on, so the comparison isolates it from RMSNorm and RoPE.
    plain = Qwen3Cached(config, weights, use_cuda=False)
    with_kernel = Qwen3Cached(config, weights, use_cuda=False)
    for block in with_kernel.blocks:
        block.mlp.use_cuda = True

    assert torch.equal(
        with_kernel(ids, with_kernel.create_kv_cache()), plain(ids, plain.create_kv_cache())
    )


def test_model_greedy_output_is_token_identical(tiny_qwen3, device):
    """All three kernels together, against the pure-PyTorch path."""
    theirs = tiny_qwen3.to(device=device, dtype=torch.float32)
    config, weights = config_from_hf(theirs), weights_from_hf(theirs)
    ids = torch.randint(0, TINY_QWEN3_DIMS["vocab_size"], (1, 5), device=device)

    torch_path = Qwen3Cached(config, weights, use_cuda=False)
    cuda_path = Qwen3Cached(config, weights, use_cuda=True)

    assert_tokens_equal(
        generate_ids_cached(cuda_path, ids, max_tokens=16),
        generate_ids_cached(torch_path, ids, max_tokens=16),
    )
