"""The test infrastructure itself is trustworthy.

Testing the test helpers looks like navel-gazing right up until a helper that
silently never fails lets a broken kernel through. `assert_allclose` is used by
almost every test in the suite, so its failure behaviour matters as much as its
success behaviour, and both directions are checked here.

`torch.testing.assert_close` passes when `|a - b| <= atol + rtol * |b|`, and
`assert_allclose` sets `atol == rtol == tolerance`, so on a tensor of ones the
effective threshold is `2 * tolerance`. The perturbations below are chosen with
that in mind.
"""

from __future__ import annotations

import random

import pytest
import torch

# pytest puts tests/ on sys.path (there is no tests/__init__.py), so this is the
# same module object pytest already loaded as the conftest, not a second copy.
from conftest import (
    MODEL_TOLERANCE,
    OP_TOLERANCES,
    SEED,
    TINY_QWEN3_DIMS,
    assert_allclose,
    assert_tokens_equal,
    make_tiny_qwen3,
)


# ---------------------------------------------------------------- determinism


def test_autouse_seed_is_active():
    """The first draw of a test is the seeded first draw, with no setup."""
    actual = torch.randn(3)
    torch.manual_seed(SEED)
    assert torch.equal(actual, torch.randn(3))


def test_autouse_seed_resets_every_test():
    """Identical to the test above on purpose.

    If the fixture seeded once per session rather than per test, the generator
    would have advanced by the draws above and this would fail.
    """
    actual = torch.randn(3)
    torch.manual_seed(SEED)
    assert torch.equal(actual, torch.randn(3))


def test_autouse_seed_covers_python_random():
    actual = random.randint(0, 10**6)
    random.seed(SEED)
    assert actual == random.randint(0, 10**6)


# ------------------------------------------------------- assert_allclose passes


def test_allclose_passes_on_identical_tensors():
    x = torch.randn(32, 16)
    assert_allclose(x, x.clone())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_allclose_passes_just_inside_tolerance(dtype):
    tolerance = OP_TOLERANCES[dtype]
    x = torch.ones(64, dtype=dtype)
    assert_allclose(x + tolerance, x)


def test_allclose_uses_the_looser_dtype():
    """A bf16 actual against an fp32 expected is held to the bf16 bound."""
    expected = torch.ones(64, dtype=torch.float32)
    actual = (expected + 5e-3).to(torch.bfloat16)

    assert_allclose(actual, expected)  # the bf16 bound tolerates it
    with pytest.raises(AssertionError):
        assert_allclose(actual.float(), expected)  # the fp32 bound does not


# ------------------------------------------------------- assert_allclose fails


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_allclose_fails_just_outside_tolerance(dtype):
    """The half of the contract that actually protects the suite."""
    tolerance = OP_TOLERANCES[dtype]
    x = torch.ones(64, dtype=dtype)
    with pytest.raises(AssertionError):
        assert_allclose(x + 10.0 * tolerance, x)


def test_allclose_fails_on_shape_mismatch():
    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_allclose(torch.zeros(4, 8), torch.zeros(8, 4))


def test_model_kind_is_looser_than_op_kind():
    """A full-model error passes the model bound while failing the op bound."""
    assert MODEL_TOLERANCE > OP_TOLERANCES[torch.bfloat16]

    x = torch.ones(64, dtype=torch.bfloat16)
    perturbed = x + 3e-2

    with pytest.raises(AssertionError):
        assert_allclose(perturbed, x, kind="op")
    assert_allclose(perturbed, x, kind="model")


def test_allclose_rejects_unknown_kind():
    x = torch.zeros(4)
    with pytest.raises(ValueError, match="unknown kind"):
        assert_allclose(x, x, kind="nonsense")


def test_allclose_message_survives():
    with pytest.raises(AssertionError, match="while checking layer 3"):
        assert_allclose(torch.zeros(4), torch.ones(4), msg="while checking layer 3")


# --------------------------------------------------------- token id comparison


def test_tokens_equal_accepts_lists_and_tensors():
    assert_tokens_equal([1, 2, 3], [1, 2, 3])
    assert_tokens_equal(torch.tensor([[1, 2, 3]]), [1, 2, 3])


def test_tokens_equal_reports_first_divergence():
    with pytest.raises(AssertionError, match="diverge at index 2"):
        assert_tokens_equal([5, 6, 7, 8], [5, 6, 99, 8])


def test_tokens_equal_reports_length_mismatch():
    with pytest.raises(AssertionError, match="diverge at index 3"):
        assert_tokens_equal([1, 2, 3], [1, 2, 3, 4])


# ---------------------------------------------------------------- tiny model


def test_tiny_qwen3_has_the_architecture_we_rebuild(tiny_qwen3):
    """The dimensions match what the reference implementation assumes, and the two traps."""
    config = tiny_qwen3.config

    assert config.num_hidden_layers == TINY_QWEN3_DIMS["num_hidden_layers"]
    assert config.tie_word_embeddings is True

    # GQA is real here, not degenerate.
    assert config.num_attention_heads > config.num_key_value_heads
    assert config.num_attention_heads % config.num_key_value_heads == 0

    # The attention projection is wider than the hidden size, as in Qwen3-0.6B.
    assert config.num_attention_heads * config.head_dim != config.hidden_size


def test_tiny_qwen3_forward_runs(tiny_qwen3):
    config = tiny_qwen3.config
    ids = torch.randint(0, config.vocab_size, (2, 8))

    with torch.no_grad():
        logits = tiny_qwen3(ids).logits

    assert logits.shape == (2, 8, config.vocab_size)
    assert torch.isfinite(logits).all()


def test_tiny_qwen3_is_deterministic():
    """Same seed, same weights — the property every differential test rests on."""
    torch.manual_seed(SEED)
    first = make_tiny_qwen3()
    torch.manual_seed(SEED)
    second = make_tiny_qwen3()

    for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters()):
        assert torch.equal(a, b), f"{name} differs between two seeded constructions"


def test_tiny_qwen3_is_small_enough_to_be_free(tiny_qwen3):
    """A few hundred thousand parameters, not a few hundred million."""
    assert sum(p.numel() for p in tiny_qwen3.parameters()) < 1_000_000


def test_tiny_qwen3_accepts_overrides():
    model = make_tiny_qwen3(num_hidden_layers=1, vocab_size=32)
    assert model.config.num_hidden_layers == 1
    assert model.config.vocab_size == 32


@pytest.mark.cuda
def test_tiny_qwen3_runs_on_the_gpu(tiny_qwen3, device):
    model = tiny_qwen3.to(device)
    ids = torch.randint(0, model.config.vocab_size, (2, 8), device=device)

    with torch.no_grad():
        logits = model(ids).logits

    assert logits.device.type == "cuda"
    assert torch.isfinite(logits).all()
