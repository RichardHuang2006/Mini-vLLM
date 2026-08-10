"""Step 1.1 — linear, silu, softmax against their torch builtins."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from conftest import assert_allclose

from mini_vllm.basics import linear, silu, softmax

# `N..` in the shape contract means any number of leading batch dims, so every
# op is checked across all of them rather than only the 2-D case.
BATCH_SHAPES = [(), (4,), (2, 3), (2, 3, 5)]


# -------------------------------------------------------------------- linear


@pytest.mark.parametrize("batch", BATCH_SHAPES)
def test_linear_matches_torch(batch):
    x = torch.randn(*batch, 16)
    w = torch.randn(8, 16)
    assert_allclose(linear(x, w), F.linear(x, w))


@pytest.mark.parametrize("batch", BATCH_SHAPES)
def test_linear_with_bias_matches_torch(batch):
    x = torch.randn(*batch, 16)
    w = torch.randn(8, 16)
    bias = torch.randn(8)
    assert_allclose(linear(x, w, bias), F.linear(x, w, bias))


def test_linear_output_shape():
    out = linear(torch.randn(2, 7, 16), torch.randn(8, 16))
    assert out.shape == (2, 7, 8)


def test_linear_expects_hf_weight_orientation():
    """An `I x O` weight must fail loudly rather than silently transpose."""
    with pytest.raises(RuntimeError):
        linear(torch.randn(2, 16), torch.randn(16, 8))


# ---------------------------------------------------------------------- silu


@pytest.mark.parametrize("batch", BATCH_SHAPES)
def test_silu_matches_torch(batch):
    x = torch.randn(*batch, 32)
    assert_allclose(silu(x), F.silu(x))


def test_silu_at_extremes():
    """Saturating inputs, where a sloppy formulation produces nan."""
    x = torch.tensor([-100.0, -10.0, 0.0, 10.0, 100.0])
    got = silu(x)
    assert_allclose(got, F.silu(x))
    assert torch.isfinite(got).all()


# ------------------------------------------------------------------- softmax


@pytest.mark.parametrize("batch", BATCH_SHAPES)
def test_softmax_matches_torch(batch):
    x = torch.randn(*batch, 24)
    assert_allclose(softmax(x), F.softmax(x, dim=-1))


@pytest.mark.parametrize("dim", [0, 1, 2, -1, -2])
def test_softmax_along_any_dim(dim):
    x = torch.randn(4, 5, 6)
    assert_allclose(softmax(x, dim=dim), F.softmax(x, dim=dim))


def test_softmax_sums_to_one():
    probabilities = softmax(torch.randn(3, 17))
    assert_allclose(probabilities.sum(dim=-1), torch.ones(3))


def test_softmax_survives_large_logits():
    """The max-subtraction trick, which is the whole point of writing this by hand.

    `exp(1000)` is `inf`, so a formulation without the shift produces `inf/inf`
    = `nan` here. Attention logits reach this range in practice.
    """
    x = torch.tensor([[1000.0, 1001.0, 999.0]])

    got = softmax(x)

    assert torch.isfinite(got).all(), "large logits overflowed; max subtraction missing"
    assert_allclose(got, F.softmax(x, dim=-1))
    # The answer only depends on differences between logits, so shifting the
    # whole row must leave the distribution untouched.
    assert_allclose(got, softmax(x - 1000.0))


def test_softmax_survives_very_negative_logits():
    x = torch.tensor([[-1000.0, -1001.0, -999.0]])
    got = softmax(x)
    assert torch.isfinite(got).all()
    assert_allclose(got, F.softmax(x, dim=-1))


def test_softmax_handles_partially_masked_rows():
    """Additive `-inf` masking, which is how every causal mask here works.

    The masked positions must come out at exactly zero, not merely small, so
    that a masked key contributes nothing to the weighted sum of values.
    """
    x = torch.tensor([[float("-inf"), 1.0, 2.0, float("-inf")]])

    got = softmax(x)

    assert_allclose(got, F.softmax(x, dim=-1))
    assert got[0, 0] == 0.0 and got[0, 3] == 0.0
    assert_allclose(got.sum(dim=-1), torch.ones(1))


def test_softmax_of_a_fully_masked_row_is_nan_as_in_torch():
    """A row that is `-inf` everywhere yields nan, matching torch exactly.

    Recorded rather than worked around: `0/0` is genuinely undefined, and torch
    makes the same choice. It matters because a causal mask must never produce
    such a row — query `i` can always attend to at least key `i` — so nan here
    is a useful signal that the mask offset is wrong rather than something to
    paper over.
    """
    x = torch.full((1, 4), float("-inf"))

    assert softmax(x).isnan().all()
    assert F.softmax(x, dim=-1).isnan().all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_softmax_reduces_in_fp32_and_returns_input_dtype(dtype):
    """Output dtype follows the input, but the reduction is fp32 internally.

    Checked against an fp32 reference rather than a same-dtype one: a bf16
    reduction would drift from this over 512 elements.
    """
    x = torch.randn(2, 512, dtype=dtype)

    got = softmax(x)

    assert got.dtype == dtype
    assert_allclose(got, F.softmax(x.float(), dim=-1).to(dtype))
