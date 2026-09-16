"""Fixtures and comparison helpers shared by every test."""

from __future__ import annotations

import contextlib
import gc
import random
from dataclasses import replace
from typing import Any

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from mini_vllm import LLM
from mini_vllm.cache import BlockManager
from mini_vllm.config import ModelConfig
from mini_vllm.model import Qwen3, map_name, resolve_model_path

# The shared numerical tolerances, keyed by dtype so no test picks its own threshold.
OP_TOLERANCES: dict[torch.dtype, float] = {
    torch.float64: 1e-5,
    torch.float32: 1e-5,
    torch.float16: 1e-2,
    torch.bfloat16: 1e-2,
}

# Full-model logits accumulate error across 28 layers, so they get a looser bound.
MODEL_TOLERANCE = 2e-2

# Measured ceiling for a bf16 forward pass against HuggingFace; see README §17.
BF16_DRIFT_LIMIT = 0.05

# How far the CUDA path may sit from the PyTorch path in bf16; see README §17.
KERNEL_DRIFT_LIMIT = 1e-2

# e4m3 keeps 3 mantissa bits (~2^-3 relative), so a round trip sits well under this.
FP8_MAX_ERROR = 0.07

SEED = 1234

# Two properties of Qwen3-0.6B that catch reshape bugs: H_q * D != E, and G = 2.
TINY_QWEN3_DIMS: dict[str, Any] = {
    "vocab_size": 512,
    "hidden_size": 64,          # E
    "num_hidden_layers": 2,
    "num_attention_heads": 4,   # H_q
    "num_key_value_heads": 2,   # H_k  -> G = 2
    "head_dim": 32,             # D    -> H_q * D = 128 != E
    "intermediate_size": 128,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "tie_word_embeddings": True,
}


@pytest.fixture(autouse=True)
def seeded():
    """Seed every RNG before each test so a failure is reproducible."""
    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


@pytest.fixture
def device() -> torch.device:
    """The CUDA device, skipping the test when there is not one.

    Requesting this fixture is what makes a test GPU-only; pair it with
    `@pytest.mark.cuda` so `-m "not cuda"` can deselect it without collecting.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    return torch.device("cuda")


def assert_allclose(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    kind: str = "op",
    msg: str = "",
) -> None:
    """Compare two tensors at the tolerance implied by their dtype.

    `kind="op"` is a single operator, `kind="model"` a full-model output. With mixed
    dtypes the lower-precision one sets the tolerance.
    """
    assert actual.shape == expected.shape, (
        f"shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}. {msg}"
    )

    if kind == "model":
        tolerance = MODEL_TOLERANCE
    elif kind == "op":
        tolerance = max(
            OP_TOLERANCES.get(actual.dtype, 1e-2),
            OP_TOLERANCES.get(expected.dtype, 1e-2),
        )
    else:
        raise ValueError(f"unknown kind {kind!r}, expected 'op' or 'model'")

    # Compare in fp32 so the comparison itself is not what loses precision.
    torch.testing.assert_close(
        actual.detach().float(),
        expected.detach().float(),
        rtol=tolerance,
        atol=tolerance,
        msg=lambda default: f"{default}\n{msg}" if msg else default,
    )


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """``‖actual − expected‖ / ‖expected‖`` over the whole tensor."""
    difference = actual.detach().float() - expected.detach().float()
    return (difference.norm() / expected.detach().float().norm()).item()


def assert_relative_error_below(
    actual: torch.Tensor,
    expected: torch.Tensor,
    limit: float = BF16_DRIFT_LIMIT,
    msg: str = "",
) -> None:
    """Compare by aggregate relative norm: the right instrument for accumulated bf16 error.

    Elementwise `atol` on a bf16 residual stream mostly reports magnitude, since one ULP
    at magnitude 512 is an absolute difference of 4. Use `assert_allclose` for single
    operators and fp32, and `assert_tokens_equal` wherever an exact check is available.
    """
    assert actual.shape == expected.shape, (
        f"shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}. {msg}"
    )

    error = relative_error(actual, expected)
    assert error < limit, (
        f"relative error {error:.4f} exceeds {limit} (‖actual − expected‖ / ‖expected‖). {msg}"
    )


def assert_tokens_equal(actual: Any, expected: Any, msg: str = "") -> None:
    """Assert two token-id sequences are identical, reporting where they first split."""
    got = [int(t) for t in (actual.flatten().tolist() if torch.is_tensor(actual) else actual)]
    want = [int(t) for t in (expected.flatten().tolist() if torch.is_tensor(expected) else expected)]

    if got == want:
        return

    limit = min(len(got), len(want))
    split = next((i for i in range(limit) if got[i] != want[i]), limit)
    lo, hi = max(0, split - 4), split + 4
    raise AssertionError(
        f"token sequences diverge at index {split} "
        f"(lengths {len(got)} vs {len(want)}).\n"
        f"  actual  [{lo}:{hi}] = {got[lo:hi]}\n"
        f"  expected[{lo}:{hi}] = {want[lo:hi]}\n{msg}"
    )


@pytest.fixture
def allclose():
    """`assert_allclose` as a fixture, for tests that prefer injection."""
    return assert_allclose


@pytest.fixture
def tokens_equal():
    """`assert_tokens_equal` as a fixture."""
    return assert_tokens_equal


# Enough pages for the short generations these tests do: 256 blocks of 16 tokens.
TEST_NUM_BLOCKS = 256


def _require_real_weights() -> None:
    path = resolve_model_path()
    if not (path / "model.safetensors").is_file():
        pytest.skip("Qwen3-0.6B weights are not downloaded")
    if not torch.cuda.is_available():
        pytest.skip("the engine's own tests want the kernels")


def free_cuda_memory() -> None:
    """Return everything the caching allocator is holding, so the next engine fits."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextlib.contextmanager
def real_engine(**kwargs: Any):
    """One engine on the real checkpoint, torn down on the way out.

    Every real-weight engine goes through here with `num_blocks` pinned, because an
    engine that sizes its own pool would claim the memory the next one needs.
    """
    _require_real_weights()

    kwargs.setdefault("dtype", torch.bfloat16)
    kwargs.setdefault("num_blocks", TEST_NUM_BLOCKS)
    engine = LLM(**kwargs)
    try:
        yield engine
    finally:
        del engine
        free_cuda_memory()


def make_tiny_qwen3(**overrides: Any):
    """Build a randomly-initialized HuggingFace Qwen3 small enough to be an oracle."""
    config = Qwen3Config(**(TINY_QWEN3_DIMS | overrides))
    return Qwen3ForCausalLM(config).eval()


@pytest.fixture
def tiny_qwen3():
    """A 2-layer Qwen3 with random weights, on the CPU in fp32; move it yourself."""
    return make_tiny_qwen3()


def config_from_hf(hf_model) -> Any:
    """A `mini_vllm` `ModelConfig`, parsed from a HuggingFace config as the loader does."""
    config = ModelConfig.from_dict(hf_model.config.to_dict())
    return replace(config, dtype=next(hf_model.parameters()).dtype)


def weights_from_hf(hf_model) -> dict[str, torch.Tensor]:
    """Rename a HuggingFace state dict into `mini_vllm` names, sharing the same tensors."""
    weights = {}
    for hf_name, tensor in hf_model.state_dict().items():
        ours = map_name(hf_name)
        if ours is not None:
            weights[ours] = tensor
    return weights


def qwen3_from_hf(hf_model):
    """Build a `mini_vllm` `Qwen3` from a HuggingFace `Qwen3ForCausalLM`, weights shared."""
    return Qwen3(config_from_hf(hf_model), weights_from_hf(hf_model))


def tiny_kv_manager(**overrides: Any):
    """A small paged pool with real KV storage: 16 blocks of 4 tokens, one layer."""
    defaults: dict[str, Any] = {
        "num_blocks": 16,
        "block_size": 4,
        "num_layers": 1,
        "num_kv_heads": 1,
        "head_dim": 8,
    }
    return BlockManager(**(defaults | overrides))
