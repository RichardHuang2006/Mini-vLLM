"""embedding.py against mlx.nn.Embedding."""

import mlx.core as mx
import mlx.nn as nn
import pytest
from utils import DTYPES, assert_allclose

from mini_vllm.embedding import Embedding


@pytest.mark.parametrize("dtype", DTYPES)
def test_lookup_and_tied_head_match_nn_embedding(dtype):
    reference = nn.Embedding(512, 64)
    reference.set_dtype(dtype)
    embedding = Embedding(512, 64, reference.weight)

    ids = mx.random.randint(0, 512, (3, 7))
    assert mx.array_equal(embedding(ids), reference(ids)).item()

    h = mx.random.normal((3, 7, 64)).astype(dtype)
    assert_allclose(embedding.as_linear(h), reference.as_linear(h))


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError, match="weight must have shape"):
        Embedding(512, 64, mx.zeros((64, 512)))

    embedding = Embedding(512, 64, mx.zeros((512, 64)))
    with pytest.raises(ValueError, match="integer"):
        embedding(mx.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="last dimension"):
        embedding.as_linear(mx.zeros((2, 32)))
