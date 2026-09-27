"""Test environment and fixtures shared by every test."""

import os

# The M5's neural accelerators run fp32 matmuls in TF32 by default, which puts fp32 results
# ~1e-3 off; that is 100x the fp32 tolerance. It must be set before MLX is imported.
os.environ["MLX_ENABLE_TF32"] = "0"

import mlx.core as mx
import mlx_lm
import pytest
from huggingface_hub import try_to_load_from_cache
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models import qwen3

from mini_vllm.models import DEFAULT_MODEL

SEED = 1234

# Two properties of Qwen3-0.6B that catch reshape bugs: H_q * D != E, and G = 2.
TINY_QWEN3 = {
    "model_type": "qwen3",
    "vocab_size": 512,
    "hidden_size": 64,           # E
    "num_hidden_layers": 2,
    "num_attention_heads": 4,    # H_q
    "num_key_value_heads": 2,    # H_k -> G = 2
    "head_dim": 32,              # D   -> H_q * D = 128 != E
    "intermediate_size": 128,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "tie_word_embeddings": True,
}


@pytest.fixture(autouse=True)
def seeded():
    """Seed MLX's global RNG before each test so a failure is reproducible."""
    mx.random.seed(SEED)


def make_tiny_qwen3(**overrides):
    """A randomly initialized fp32 mlx_lm Qwen3: the oracle our transplanted model must match."""
    model = qwen3.Model(qwen3.ModelArgs(**{**TINY_QWEN3, **overrides}))
    # nn.RMSNorm starts at all ones, which would hide a swapped or missing norm.
    parameters = [
        (name, 1 + 0.1 * mx.random.normal(value.shape) if name.endswith("norm.weight") else value)
        for name, value in tree_flatten(model.parameters())
    ]
    model.update(tree_unflatten(parameters))
    mx.eval(model.parameters())
    return model


@pytest.fixture
def tiny_qwen3():
    return make_tiny_qwen3()


@pytest.fixture(scope="session")
def real_qwen3():
    """The BF16 Qwen3-0.6B checkpoint as (mlx_lm model, tokenizer); skipped unless downloaded."""
    if not isinstance(try_to_load_from_cache(DEFAULT_MODEL, "model.safetensors"), str):
        pytest.skip(f"{DEFAULT_MODEL} is not in the Hugging Face cache")
    return mlx_lm.load(DEFAULT_MODEL)
