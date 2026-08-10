"""Step 0.2 — fixtures and comparison helpers shared by every later test.

Three things live here, and each exists to make a specific class of bug cheap to
find later:

* **Seeding**, so any failure reproduces exactly.
* **Dtype-aware comparison**, so no test hardcodes a tolerance and drifts.
* **A tiny Qwen3**, so correctness tests run in milliseconds without the 1.2 GB
  of real weights. The real weights are reserved for `@pytest.mark.oracle`.
"""

from __future__ import annotations

import random
from typing import Any

import pytest
import torch

# PLAN.md "Numerical tolerances". Keyed by dtype so a test never picks its own
# threshold: a comparison that needs a looser bound than these is reporting a
# bug, not a tolerance problem.
OP_TOLERANCES: dict[torch.dtype, float] = {
    torch.float64: 1e-5,
    torch.float32: 1e-5,
    torch.float16: 1e-2,
    torch.bfloat16: 1e-2,
}

# Full-model logits accumulate error across 28 layers, so they get their own,
# looser bound. Prefer comparing greedy token ids instead where you can: that
# check is exact.
MODEL_TOLERANCE = 2e-2

# The ceiling for `assert_relative_error_below` on a full bf16 forward pass of
# Qwen3-0.6B, measured rather than guessed. On real text our logits sit 1.7% from
# HuggingFace's, reproducibly to five digits, while the subtlest broken model
# worth worrying about (Qwen2's `rope_theta` instead of Qwen3's) sits at 14%. This
# limit is ~3x above the first and ~3x below the second, and `test_qwen3.py`
# asserts both sides of that gap so it cannot quietly stop discriminating.
#
# Measure this on *real text*. Random token ids are chaotically amplified through
# 28 layers: typical drift is 3% but individual gibberish sequences reach 11%,
# which is a property of the input rather than of the implementation.
BF16_DRIFT_LIMIT = 0.05

# How far the CUDA path may sit from the PyTorch path on the *same* weights, in
# bf16. Not a tolerance for error: it is one bf16 ULP (2^-8 = 0.0039) with a little
# headroom, and the reason it is a whole ULP rather than a rounding crumb is the
# attention kernels. The oracle rounds the scores and the softmax weights back to
# bf16 between ops; the kernels keep them in fp32 to the store. So the kernel path
# is the *more* accurate of the two, by about one rounding — measured at 0.0041 for
# one attention call and 0.0058 through the tiny model, which is where this number
# comes from. Growth past it means a kernel is wrong, not imprecise; greedy token
# ids (`assert_tokens_equal`) remain the exact check and are preferred.
#
# For the real 28-layer checkpoint use `BF16_DRIFT_LIMIT` instead: the same single
# rounding compounds to ~1.9% by the last layer, which says nothing more than that
# 28 layers amplify one bit.
KERNEL_DRIFT_LIMIT = 1e-2

SEED = 1234

# The tiny model's dimensions. Deliberately keeps two properties of the real
# Qwen3-0.6B that catch reshape bugs (DESIGN.md §3):
#   * H_q * D != E, so the attention projection is *wider* than the hidden size
#     and code that conflates the two fails here rather than in Phase 4.
#   * G = H_q / H_k = 2, the real GQA group size, so the KV-head dimension is
#     genuinely exercised instead of degenerating to plain multi-head attention.
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


# ---------------------------------------------------------------- determinism


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


# ----------------------------------------------------------------- comparison


def assert_allclose(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    kind: str = "op",
    msg: str = "",
) -> None:
    """Compare two tensors at the tolerance implied by their dtype.

    `kind="op"` is a single operator, `kind="model"` a full-model output whose
    error has accumulated across every layer. When the two tensors have
    different dtypes the *lower*-precision one sets the tolerance, which is the
    useful behaviour when checking a bf16 implementation against an fp32
    reference.
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
    """Compare two tensors by aggregate relative norm rather than elementwise.

    The right instrument for a full bf16 forward pass, where `assert_allclose`
    is the wrong one — and note this is a *different* measurement, not a looser
    tolerance.

    Elementwise `atol` on a bf16 residual stream mostly reports magnitude. bf16
    keeps 8 bits of mantissa, so one ULP at magnitude 512 is an absolute
    difference of 4: a single-bit rounding disagreement between two correct
    implementations shows up as `atol=4` and says nothing about whether either is
    right. Dividing by the norm of the expected tensor removes that scale
    dependence, leaving a quantity that stays near the rounding floor when the
    implementation is correct and jumps by an order of magnitude when it is not.

    Use `assert_allclose` for single operators and fp32, `assert_tokens_equal`
    when an exact check is available, and this only for accumulated bf16 error.
    """
    assert actual.shape == expected.shape, (
        f"shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}. {msg}"
    )

    error = relative_error(actual, expected)
    assert error < limit, (
        f"relative error {error:.4f} exceeds {limit} "
        f"(‖actual − expected‖ / ‖expected‖). {msg}"
    )


def assert_tokens_equal(actual: Any, expected: Any, msg: str = "") -> None:
    """Assert two token-id sequences are *identical*, reporting where they split.

    The strongest check in the project: with greedy decoding two correct
    implementations must agree exactly, so there is no tolerance to argue about.
    When they do not agree, the position of the first difference is the thing
    worth knowing, since everything after it is downstream of one bad token.
    """
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


# ---------------------------------------------------------------- tiny model


def make_tiny_qwen3(**overrides: Any):
    """Build a randomly-initialized Qwen3 small enough to test against.

    This is HuggingFace's Qwen3 rather than ours, because ours does not exist
    until Step 1.8 — which is the right way round anyway: from Step 1.8 on this
    is the *oracle*, and our model is checked against it using these same random
    weights, no download required.
    """
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(**(TINY_QWEN3_DIMS | overrides))
    return Qwen3ForCausalLM(config).eval()


@pytest.fixture
def tiny_qwen3():
    """A 2-layer Qwen3 with random weights, on the CPU in fp32.

    Left on the CPU on purpose so tests that need no GPU can use it too; move it
    yourself with `tiny_qwen3.to(device)`. Weights are deterministic because the
    autouse `seeded` fixture runs first.
    """
    return make_tiny_qwen3()


def config_from_hf(hf_model) -> Any:
    """Our `ModelConfig`, filled from a HuggingFace config.

    Routed through `ModelConfig.from_dict` rather than reading attributes off the
    config object, so the tests exercise the same parsing the real loader uses.
    """
    from dataclasses import replace

    from mini_vllm.model.loader import ModelConfig

    config = ModelConfig.from_dict(hf_model.config.to_dict())
    return replace(config, dtype=next(hf_model.parameters()).dtype)


def weights_from_hf(hf_model) -> dict[str, torch.Tensor]:
    """Rename a HuggingFace state dict into ours, sharing the same tensors.

    Sharing rather than copying is deliberate: it removes "did the weights
    actually transfer" from the list of things a failing comparison could mean.
    """
    from mini_vllm.model.loader import map_name

    weights = {}
    for hf_name, tensor in hf_model.state_dict().items():
        ours = map_name(hf_name)
        if ours is not None:
            weights[ours] = tensor
    return weights


def qwen3_from_hf(hf_model):
    """Build our `Qwen3` from a HuggingFace `Qwen3ForCausalLM`, weights shared."""
    from mini_vllm.model.qwen3 import Qwen3

    return Qwen3(config_from_hf(hf_model), weights_from_hf(hf_model))
