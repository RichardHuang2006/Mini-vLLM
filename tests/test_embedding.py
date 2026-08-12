"""The embedding table in both directions."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from conftest import assert_allclose

from mini_vllm.embedding import Embedding

VOCAB, DIM = 64, 16


@pytest.fixture
def table():
    return Embedding(VOCAB, DIM, torch.randn(VOCAB, DIM))


# --------------------------------------------------------------- lookup (input)


@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 3, 5)])
def test_lookup_matches_torch(table, shape):
    ids = torch.randint(0, VOCAB, shape)

    reference = torch.nn.Embedding(VOCAB, DIM)
    with torch.no_grad():
        reference.weight.copy_(table.weight)

    assert_allclose(table(ids), reference(ids))


def test_lookup_shape(table):
    assert table(torch.randint(0, VOCAB, (2, 7))).shape == (2, 7, DIM)


def test_lookup_returns_the_requested_rows(table):
    ids = torch.tensor([[3, 0, 63]])
    assert_allclose(table(ids)[0, 0], table.weight[3])
    assert_allclose(table(ids)[0, 2], table.weight[63])


def test_lookup_rejects_float_ids(table):
    with pytest.raises(ValueError, match="ids must be integer"):
        table(torch.tensor([1.0, 2.0]))


def test_lookup_rejects_out_of_range_ids(table):
    with pytest.raises(IndexError):
        table(torch.tensor([VOCAB]))


# --------------------------------------------------------- as_linear (LM head)


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
def test_as_linear_matches_torch(table, shape):
    h = torch.randn(*shape, DIM)
    assert_allclose(table.as_linear(h), F.linear(h, table.weight))


def test_as_linear_shape(table):
    assert table.as_linear(torch.randn(2, 7, DIM)).shape == (2, 7, VOCAB)


def test_as_linear_rejects_wrong_width(table):
    with pytest.raises(ValueError, match="expected last dimension"):
        table.as_linear(torch.randn(2, DIM + 1))


# ------------------------------------------------------------------ tying them


def test_tied_head_scores_a_token_by_similarity(table):
    """The consequence of tying: the logit for token `t` is `<h, embedding[t]>`.

    So feeding a token's own embedding back in makes that token the argmax, which
    is a compact way to prove the two directions use the same matrix in the same
    orientation. A transposed `as_linear` would fail this while still producing
    plausibly-shaped logits.
    """
    token = 42
    h = table(torch.tensor([[token]]))

    logits = table.as_linear(h)

    assert logits.shape == (1, 1, VOCAB)
    assert int(logits[0, 0].argmax()) == token


def test_as_linear_is_the_transpose_of_lookup(table):
    """`as_linear(one_hot @ W) == one_hot @ W @ Wᵀ`, spelled out per row."""
    ids = torch.arange(VOCAB)
    gram = table.as_linear(table(ids))
    assert_allclose(gram, table.weight @ table.weight.T)


# ---------------------------------------------------------------- construction


def test_rejects_mismatched_weight_shape():
    with pytest.raises(ValueError, match="weight must have shape"):
        Embedding(VOCAB, DIM, torch.randn(DIM, VOCAB))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_preserves_dtype(dtype):
    table = Embedding(VOCAB, DIM, torch.randn(VOCAB, DIM, dtype=dtype))
    assert table(torch.tensor([[1, 2]])).dtype == dtype
    assert table.as_linear(torch.randn(1, 2, DIM, dtype=dtype)).dtype == dtype
