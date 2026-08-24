"""Fixtures and comparison helpers shared by every test.

Three things live here, each making a class of bug cheap to find:

* Seeding, so any failure reproduces exactly.
* Dtype-aware comparison, so no test hardcodes a tolerance and drifts.
* A tiny Qwen3, so correctness tests run in milliseconds without the 1.2 GB of real
  weights. The real weights are reserved for `@pytest.mark.oracle`.
"""

from __future__ import annotations

import contextlib
import random
from typing import Any

import pytest
import torch

# The shared numerical tolerances, keyed by dtype so no test picks its own threshold. A
# comparison needing a looser bound than these is reporting a bug.
OP_TOLERANCES: dict[torch.dtype, float] = {
    torch.float64: 1e-5,
    torch.float32: 1e-5,
    torch.float16: 1e-2,
    torch.bfloat16: 1e-2,
}

# Full-model logits accumulate error across 28 layers, so they get a looser bound.
# Comparing greedy token ids is preferred where possible, since that check is exact.
MODEL_TOLERANCE = 2e-2

# The ceiling for `assert_relative_error_below` on a full bf16 forward pass of Qwen3-0.6B,
# measured rather than estimated. On real text these logits sit 1.7% from HuggingFace's,
# reproducibly to five digits, while the subtlest broken model worth catching (Qwen2's
# `rope_theta` instead of Qwen3's) sits at 14%. This limit is ~3x above the first and ~3x
# below the second, and `test_qwen3.py` asserts both sides of that gap so it cannot stop
# discriminating unnoticed.
#
# Measure on real text. Random token ids are chaotically amplified through 28 layers:
# typical drift is 3% but individual gibberish sequences reach 11%, which is a property of
# the input rather than of the implementation.
BF16_DRIFT_LIMIT = 0.05

# How far the CUDA path may sit from the PyTorch path on the same weights, in bf16. This
# is one bf16 ULP (2^-8 = 0.0039) with headroom rather than an error budget, and it is a
# whole ULP because of the attention kernels: the oracle rounds the scores and softmax
# weights back to bf16 between ops while the kernels keep them in fp32 to the store, making
# the kernel path the more accurate of the two by about one rounding. Measured at 0.0041
# for one attention call and 0.0058 through the tiny model. Exceeding it means a kernel is
# wrong rather than imprecise; greedy token ids (`assert_tokens_equal`) remain the exact
# check and are preferred.
#
# For the real 28-layer checkpoint use `BF16_DRIFT_LIMIT`: the same single rounding
# compounds to ~1.9% by the last layer, which reflects depth rather than a defect.
KERNEL_DRIFT_LIMIT = 1e-2

SEED = 1234

# The tiny model's dimensions, keeping two properties of the real Qwen3-0.6B that catch
# reshape bugs:
#   * H_q * D != E, so the attention projection is wider than the hidden size and code
#     that conflates the two fails here rather than in the serving layer.
#   * G = H_q / H_k = 2, the real GQA group size, so the KV-head dimension is exercised
#     instead of degenerating to plain multi-head attention.
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


# ------------------------------------------------------------- real engines
#
# The GPU of record has 8 GB and a bf16 Qwen3-0.6B is 1.2 GB of weights before its KV
# cache, so two engines do not fit at once. An engine that sizes its own pool takes a
# fraction of whatever is free, so the first one built claims the memory the second needs.
# Both failures appear as an out-of-memory abort with no failing assertion.
#
# Every real-weight engine therefore goes through here, with the rules enforced
# structurally: `num_blocks` is always pinned, and `real_engine` is a context manager that
# frees the engine and empties the caching allocator on exit. A test needing two engines
# gets them one after another.

# Enough pages for the short generations these tests do: 256 blocks of 16 tokens is 4096
# cached tokens, against prompts and outputs of a few dozen.
TEST_NUM_BLOCKS = 256


def _require_real_weights() -> None:
    from mini_vllm.model.loader import resolve_model_path

    path = resolve_model_path()
    if not (path / "model.safetensors").is_file():
        pytest.skip("Qwen3-0.6B weights are not downloaded")
    if not torch.cuda.is_available():
        pytest.skip("the engine's own tests want the kernels")


def free_cuda_memory() -> None:
    """Return everything the caching allocator is holding, so the next engine fits."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextlib.contextmanager
def real_engine(**kwargs: Any):
    """One engine on the real checkpoint, torn down on the way out.

    Defaults to bf16 — the dtype the engine actually serves in, and half the footprint of
    fp32. Pass ``dtype=torch.float32`` where a test needs greedy decoding to be free of
    bf16's ties, but expect to be the only engine on the card while it lives.
    """
    _require_real_weights()
    from mini_vllm import LLM

    kwargs.setdefault("dtype", torch.bfloat16)
    kwargs.setdefault("num_blocks", TEST_NUM_BLOCKS)
    engine = LLM(**kwargs)
    try:
        yield engine
    finally:
        del engine
        free_cuda_memory()


# ---------------------------------------------------------------- tiny model


def make_tiny_qwen3(**overrides: Any):
    """Build a randomly-initialized Qwen3 small enough to test against.

    This is HuggingFace's Qwen3 rather than ours, which is the right way round:
    it is the *oracle*, and `mini_vllm.model.qwen3.Qwen3` is checked against it
    using these same random weights, no download required.
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
    """A `mini_vllm` `ModelConfig`, filled from a HuggingFace config.

    Routed through `ModelConfig.from_dict` rather than reading attributes off the
    config object, so the tests exercise the same parsing the real loader uses.
    """
    from dataclasses import replace

    from mini_vllm.model.loader import ModelConfig

    config = ModelConfig.from_dict(hf_model.config.to_dict())
    return replace(config, dtype=next(hf_model.parameters()).dtype)


def weights_from_hf(hf_model) -> dict[str, torch.Tensor]:
    """Rename a HuggingFace state dict into `mini_vllm` names, sharing the same tensors.

    Sharing rather than copying removes weight transfer from the set of possible causes of
    a failing comparison.
    """
    from mini_vllm.model.loader import map_name

    weights = {}
    for hf_name, tensor in hf_model.state_dict().items():
        ours = map_name(hf_name)
        if ours is not None:
            weights[ours] = tensor
    return weights


def qwen3_from_hf(hf_model):
    """Build a `mini_vllm` `Qwen3` from a HuggingFace `Qwen3ForCausalLM`, weights shared."""
    from mini_vllm.model.qwen3 import Qwen3

    return Qwen3(config_from_hf(hf_model), weights_from_hf(hf_model))
