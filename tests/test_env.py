"""Step 0.1 -- assumptions every later phase is built on.

Each test here stands in for a whole phase that would otherwise fail in a much
more confusing way:

===================================  ====================================
this test                            the phase it de-risks
===================================  ====================================
``test_triton_kernel_runs``          Phase 3 and 6 (all custom kernels)
``test_fp8_tensor_roundtrips``       Phase 6.3 and 6.4 (FP8 KV cache)
``test_total_vram_matches_plan``     Phase 4.3 (KV cache sizing)
``test_autouse_seeding_is_active``   every test that compares tensors
===================================  ====================================
"""

from __future__ import annotations

import sys

import pytest
import torch
import triton
import triton.language as tl
from conftest import SEED

# The documented development machine (PLAN.md, "Environment & model choices").
# If you move to a different GPU, update both this constant and that table --
# the plan's memory arithmetic and model choices assume this hardware.
EXPECTED_CAPABILITY = (12, 0)  # Blackwell, sm_120
EXPECTED_TOTAL_VRAM_GIB = 8.0
FP8_MIN_CAPABILITY = (8, 9)  # E4M3 needs Ada or newer


# --------------------------------------------------------------------------
# interpreter and packages
# --------------------------------------------------------------------------


def test_python_is_recent_enough():
    """3.11+ for `X | None` annotations and modern builtin generics."""
    assert sys.version_info >= (3, 11), f"Python {sys.version} is too old"


def test_required_packages_import():
    """Fail here, with a clear name, rather than mid-phase."""
    import hypothesis  # noqa: F401
    import numpy  # noqa: F401
    import safetensors  # noqa: F401
    import scipy  # noqa: F401
    import transformers  # noqa: F401


def test_torch_is_a_cuda_build():
    assert torch.version.cuda is not None, (
        f"torch {torch.__version__} is a CPU-only build; reinstall from the CUDA wheel index"
    )


# --------------------------------------------------------------------------
# the GPU itself
# --------------------------------------------------------------------------


def test_cuda_is_available():
    """Deliberately *not* marked ``gpu``.

    This is the canary. Marking it would make it skip itself in exactly the
    situation it exists to report, which is a test that can never fail. The
    other GPU tests are marked, so a machine without a GPU produces one loud
    failure here instead of five confusing ones.
    """
    assert torch.cuda.is_available(), "no CUDA device visible to torch"


@pytest.mark.gpu
def test_compute_capability_matches_plan():
    capability = torch.cuda.get_device_capability(0)
    assert capability == EXPECTED_CAPABILITY, (
        f"expected sm_{EXPECTED_CAPABILITY[0]}{EXPECTED_CAPABILITY[1]} "
        f"({torch.cuda.get_device_name(0)!r} reports sm_{capability[0]}{capability[1]}). "
        "If you changed machines, update EXPECTED_CAPABILITY here and the "
        "environment table in PLAN.md."
    )


@pytest.mark.gpu
def test_total_vram_matches_plan():
    """Phase 4.3 sizes the KV cache from this number; the plan assumes ~8 GiB."""
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    assert total_gib == pytest.approx(EXPECTED_TOTAL_VRAM_GIB, abs=0.5), (
        f"GPU reports {total_gib:.2f} GiB, plan assumes {EXPECTED_TOTAL_VRAM_GIB} GiB; "
        "the model choices in PLAN.md may no longer apply"
    )


# --------------------------------------------------------------------------
# FP8, the prerequisite for Phase 6
# --------------------------------------------------------------------------


def test_fp8_dtype_exists():
    assert hasattr(torch, "float8_e4m3fn"), "this torch build has no FP8 E4M3 dtype"


@pytest.mark.gpu
def test_fp8_tensor_roundtrips():
    """The dtype existing is not enough -- it has to work on *this* GPU.

    Also pins down E4M3's precision, which is the tolerance Step 6.3 is
    written against: 3 mantissa bits, so ~6% worst-case relative error.
    """
    assert torch.cuda.get_device_capability(0) >= FP8_MIN_CAPABILITY, (
        "FP8 requires compute capability 8.9 (Ada) or newer"
    )

    original = torch.randn(256, device="cuda", dtype=torch.float32)
    scale = original.abs().max() / torch.finfo(torch.float8_e4m3fn).max

    quantized = (original / scale).to(torch.float8_e4m3fn)
    recovered = quantized.to(torch.float32) * scale

    assert quantized.dtype == torch.float8_e4m3fn
    assert quantized.nbytes == original.nbytes // 4
    relative_error = ((recovered - original).abs() / original.abs().clamp(min=1e-6)).max()
    assert relative_error < 0.10, f"FP8 roundtrip error {relative_error:.4f} is implausibly large"


# --------------------------------------------------------------------------
# Triton, the prerequisite for Phases 3 and 6
# --------------------------------------------------------------------------


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    """Vector add. Deliberately uses every primitive Step 3.2 needs."""
    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def test_triton_imports():
    assert triton.__version__ is not None


@pytest.mark.gpu
def test_triton_kernel_runs():
    """Compile and run a real kernel on this GPU.

    Blackwell is new enough that Triton support is the single largest
    unverified assumption in the plan. Finding out here costs a second;
    finding out in Step 3.3 costs an afternoon of debugging the wrong thing.

    ``n`` is deliberately not a multiple of ``BLOCK`` so the masking path is
    exercised -- reading past the end of a tile is the classic Triton bug, and
    every kernel in Phase 3 depends on getting it right.
    """
    n = 1000
    block = 256
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)

    _add_kernel[(triton.cdiv(n, block),)](x, y, out, n, BLOCK=block)

    torch.testing.assert_close(out, x + y)


# --------------------------------------------------------------------------
# the harness itself
# --------------------------------------------------------------------------


def test_markers_are_registered(pytestconfig: pytest.Config):
    """`--strict-markers` turns a typo'd marker into a collection error.

    That is only a safety net if the markers are actually declared, so check
    that the set in pyproject.toml still matches the plan.
    """
    declared = {line.split(":", 1)[0] for line in pytestconfig.getini("markers")}
    assert {"gpu", "oracle", "slow"} <= declared, f"missing markers, found {declared}"


def test_autouse_seeding_is_active():
    """The `seeded` fixture ran, and used the seed we think it did.

    A fresh generator seeded identically produces the same stream as the global
    RNG, so this compares the harness against ground truth rather than against
    a hard-coded constant that would drift between torch versions.
    """
    expected = torch.rand(4, generator=torch.Generator().manual_seed(SEED))
    torch.testing.assert_close(torch.rand(4), expected)


@pytest.mark.slow
def test_slow_marker_is_usable():
    """Placeholder proving `-m 'not slow'` deselects correctly."""


@pytest.mark.oracle
def test_oracle_marker_is_usable():
    """Placeholder proving oracle tests are skipped without NANOVLLM_RUN_ORACLE=1."""
