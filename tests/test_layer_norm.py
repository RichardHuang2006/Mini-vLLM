"""Step 1.5 — RMSNorm against HuggingFace's Qwen3RMSNorm."""

from __future__ import annotations

import pytest
import torch
from conftest import TINY_QWEN3_DIMS, assert_allclose
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from mini_vllm.layer_norm import RMSNorm, rms_norm

DIM = TINY_QWEN3_DIMS["hidden_size"]
EPS = TINY_QWEN3_DIMS["rms_norm_eps"]


def hf_rmsnorm(dim: int, weight: torch.Tensor, eps: float, dtype: torch.dtype) -> Qwen3RMSNorm:
    reference = Qwen3RMSNorm(dim, eps=eps)
    with torch.no_grad():
        reference.weight.copy_(weight)
    return reference.to(dtype).eval()


# ------------------------------------------------------------------ against HF


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("batch", [(4,), (2, 3), (2, 3, 5)])
def test_matches_hf(dtype, batch):
    x = torch.randn(*batch, DIM, dtype=dtype)
    weight = torch.randn(DIM, dtype=dtype)

    with torch.no_grad():
        expected = hf_rmsnorm(DIM, weight, EPS, dtype)(x)

    assert_allclose(rms_norm(x, weight, EPS), expected)


def test_matches_hf_over_a_head_dim_width():
    """QK-norm normalizes over `D`, not `E` — the same function, narrower axis."""
    head_dim = TINY_QWEN3_DIMS["head_dim"]
    x = torch.randn(2, 6, 4, head_dim)
    weight = torch.randn(head_dim)

    with torch.no_grad():
        expected = hf_rmsnorm(head_dim, weight, EPS, torch.float32)(x)

    assert_allclose(rms_norm(x, weight, EPS), expected)


# ------------------------------------------------------------------ properties


def test_unit_weight_makes_rms_one():
    """With weight 1, every row leaves with root-mean-square 1."""
    x = torch.randn(8, DIM) * 17.0
    out = rms_norm(x, torch.ones(DIM), EPS)
    assert_allclose(out.pow(2).mean(dim=-1).sqrt(), torch.ones(8))


def test_is_scale_invariant():
    """Scaling the input leaves the output unchanged, which is the point of it."""
    x = torch.randn(4, DIM)
    weight = torch.randn(DIM)
    assert_allclose(rms_norm(x * 100.0, weight, EPS), rms_norm(x, weight, EPS))


def test_does_not_recentre():
    """Unlike LayerNorm, the mean is not removed, so a shifted input differs."""
    x = torch.randn(4, DIM)
    weight = torch.ones(DIM)
    assert not torch.allclose(rms_norm(x + 5.0, weight, EPS), rms_norm(x, weight, EPS))


def test_weight_scales_each_channel():
    x = torch.randn(4, DIM)
    weight = torch.randn(DIM)
    assert_allclose(rms_norm(x, weight, EPS), rms_norm(x, torch.ones(DIM), EPS) * weight)


def test_all_zero_input_is_finite():
    """`eps` is what keeps a zero row from dividing by zero."""
    out = rms_norm(torch.zeros(2, DIM), torch.ones(DIM), EPS)
    assert torch.isfinite(out).all()
    assert_allclose(out, torch.zeros(2, DIM))


def test_reduction_happens_in_fp32():
    """A bf16 reduction would drift measurably at this width.

    Reducing in the input dtype gives a visibly different answer from the fp32
    reduction, which is exactly why the cast is not optional.
    """
    x = torch.randn(4, 1024, dtype=torch.bfloat16) * 3.0
    weight = torch.ones(1024, dtype=torch.bfloat16)

    ours = rms_norm(x, weight, EPS)
    naive_bf16 = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    exact = (x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + EPS)).to(
        torch.bfloat16
    )

    assert_allclose(ours, exact)
    assert not torch.equal(naive_bf16, exact), "bf16 reduction happened to be exact here"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_returns_input_dtype(dtype):
    x = torch.randn(2, DIM, dtype=dtype)
    assert rms_norm(x, torch.ones(DIM, dtype=dtype), EPS).dtype == dtype


# ----------------------------------------------------------------- the wrapper


def test_class_matches_function():
    x = torch.randn(3, DIM)
    weight = torch.randn(DIM)
    assert_allclose(RMSNorm(DIM, weight, EPS)(x), rms_norm(x, weight, EPS))


def test_class_rejects_mismatched_weight():
    with pytest.raises(ValueError, match="weight must have shape"):
        RMSNorm(DIM, torch.ones(DIM + 1), EPS)


def test_class_rejects_mismatched_input():
    norm = RMSNorm(DIM, torch.ones(DIM), EPS)
    with pytest.raises(ValueError, match="expected last dimension"):
        norm(torch.randn(2, DIM + 3))
