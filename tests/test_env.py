"""Step 0.1 — the environment and the CUDA extension pipeline are usable.

Everything in Phase 3 and 4 assumes a working `csrc/` build, so this file's job
is to fail loudly and early if the toolchain is not what the rest of the plan
expects, rather than letting a version mismatch surface later while a real
kernel is also being debugged.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from mini_vllm.kernels import extension

REPO_ROOT = Path(__file__).resolve().parents[1]

# DESIGN.md §10 fixes the hardware of record: RTX 5070 Laptop, Blackwell sm_120.
EXPECTED_CAPABILITY = (12, 0)


# ------------------------------------------------------------------ the basics


def test_cuda_is_available():
    assert torch.cuda.is_available(), "no CUDA device; the whole project needs a GPU"


def test_torch_version_matches_requirements():
    """The pinned torch is the one actually installed.

    A drifting torch version is worth catching here because it silently changes
    the CUDA major version the extension must be compiled against.
    """
    text = (REPO_ROOT / "requirements.txt").read_text()
    match = re.search(r"^torch==(\S+)$", text, re.MULTILINE)
    assert match, "requirements.txt does not pin torch"
    assert torch.__version__.split("+")[0] == match.group(1)


@pytest.mark.cuda
def test_device_is_the_hardware_of_record():
    assert torch.cuda.get_device_capability(0) == EXPECTED_CAPABILITY, (
        f"expected compute capability {EXPECTED_CAPABILITY} (sm_120); "
        f"if this GPU is different, update CUDA_ARCH in {extension.__name__}"
    )


def test_markers_are_registered(pytestconfig):
    """Every marker the plan uses is declared, so no test is silently skipped.

    An unregistered marker is a warning, not an error, which makes it exactly
    the kind of thing that rots quietly.
    """
    declared = {line.split(":", 1)[0].strip() for line in pytestconfig.getini("markers")}
    assert {"cuda", "oracle", "slow"} <= declared


# --------------------------------------------------------------- the toolchain


def test_resolved_nvcc_major_matches_torch():
    """The crux of Step 0.1.

    torch.utils.cpp_extension refuses to compile when nvcc's major version
    differs from the one torch was built against. The system nvcc on this
    machine is 12.8 against a cu130 torch, so this passing means the resolver
    found or assembled a CUDA 13 toolkit instead of falling back to the system.
    """
    home, how = extension.resolve_cuda_home()
    version = extension.nvcc_version(home / "bin" / "nvcc")

    assert version is not None, f"no runnable nvcc under {home} (found via {how})"
    assert version[0] == extension.torch_cuda_major(), (
        f"nvcc {version} at {home} does not match torch's CUDA "
        f"{torch.version.cuda}; extension builds will be rejected"
    )


def test_assembled_toolkit_has_what_nvcc_needs():
    """The assembled CUDA_HOME is complete.

    Each of these was a separate build failure on the way to a working
    extension, so each earns an assertion: a missing `nv/target` (the CCCL
    wheel) and a missing unversioned `libcudart.so` (the wheels ship only
    `libcudart.so.13`, but torch links with `-lcudart`).
    """
    home, _how = extension.resolve_cuda_home()

    assert (home / "bin" / "nvcc").is_file()
    assert (home / "include" / "cuda_runtime.h").is_file()
    assert (home / "include" / "nv" / "target").is_file(), "CCCL headers missing"
    assert (home / "lib64" / "libcudart.so").exists(), "unversioned cudart soname missing"


def test_toolchain_report_is_complete():
    report = extension.toolchain_report()
    for key in ("torch", "torch_cuda", "cuda_home", "nvcc", "arch", "sources"):
        assert report.get(key), f"toolchain_report() is missing {key}"
    assert "bindings.cpp" in report["sources"]


# ------------------------------------------------------- the extension is real


@pytest.mark.cuda
def test_extension_builds_and_loads():
    """It compiles, links, and the symbol is callable. The whole point of 0.1."""
    module = extension.load_extension()
    assert hasattr(module, "hello"), "extension built but does not export hello"


@pytest.mark.cuda
def test_extension_is_cached():
    assert extension.load_extension() is extension.load_extension()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hello_matches_torch(dtype):
    """y = a*x + b, against the same expression in PyTorch.

    This is the shape of every kernel test in Phase 3: the kernel is only
    correct insofar as it agrees with a readable implementation.
    """
    module = extension.load_extension()
    x = torch.randn(4096, device="cuda", dtype=dtype)
    a, b = 2.5, -1.25

    got = module.hello(x, a, b)
    want = a * x.float() + b

    assert got.dtype == dtype
    assert got.shape == x.shape
    # Compare in fp32: the kernel accumulates in fp32 and rounds once on store,
    # so a fp32 reference is the right oracle even for the half dtypes.
    tolerance = 1e-6 if dtype is torch.float32 else 1e-2
    torch.testing.assert_close(got.float(), want, rtol=tolerance, atol=tolerance)


@pytest.mark.cuda
def test_hello_handles_awkward_inputs():
    """Empty and non-contiguous tensors, the two shapes that break naive kernels."""
    module = extension.load_extension()

    empty = torch.empty(0, device="cuda")
    assert module.hello(empty, 1.0, 0.0).numel() == 0

    # A transposed view is non-contiguous; the kernel must not read it as if it
    # were dense. Every kernel in this project takes .contiguous() for exactly
    # this reason, and this asserts it actually happens.
    strided = torch.randn(64, 32, device="cuda").t()
    got = module.hello(strided, 3.0, 1.0)
    torch.testing.assert_close(got, 3.0 * strided + 1.0)
