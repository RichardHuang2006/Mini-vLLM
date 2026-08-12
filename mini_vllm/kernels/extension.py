"""JIT build and load of the CUDA sources in ``csrc/``.

Every hand-written kernel, and every part of the serving layer that runs one,
reaches the GPU through here, so this module owns one awkward problem:
``torch.utils.cpp_extension`` refuses to compile when nvcc's CUDA *major* version
differs from the one torch was built against, and on this machine they differ
(torch is a cu130 build; the system nvcc is 12.8).

The fix is to ignore the system toolkit and use the CUDA 13 compiler that ships
as pip wheels. Those wheels are scattered, though: nvcc and its nvvm backend in
one, the runtime headers and libraries in another, the CCCL headers that
``cuda_fp16.h`` pulls in (``nv/target``) in a third — and pip may install them
into different site-packages trees, so no single directory looks like a CUDA
installation. So this module assembles a symlink tree that does, under
``build/``, and points CUDA_HOME at it.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CSRC_DIR = REPO_ROOT / "csrc"
BUILD_DIR = REPO_ROOT / "build"
EXT_BUILD_DIR = BUILD_DIR / "torch_extensions"
TOOLKIT_DIR = BUILD_DIR / "cuda_toolkit"

EXT_NAME = "mini_vllm_C"

# Blackwell, the RTX 5070 Laptop of record (compute capability 12.0). Targeting
# the single architecture this project runs on keeps compiles fast.
CUDA_ARCH = "sm_120"

_extension: Any = None


class ToolchainError(RuntimeError):
    """Raised when no CUDA toolkit matching torch's CUDA version can be found."""


# --------------------------------------------------------------------- probing


def torch_cuda_major() -> int:
    """The CUDA major version torch was built against, e.g. 13 for ``2.11.0+cu130``."""
    if torch.version.cuda is None:
        raise ToolchainError(
            f"this torch ({torch.__version__}) is a CPU-only build; "
            "reinstall with `make setup` to get the CUDA 13 wheel"
        )
    return int(torch.version.cuda.split(".")[0])


def nvcc_version(nvcc: Path) -> tuple[int, int] | None:
    """Parse ``(major, minor)`` out of ``nvcc --version``, or None if it will not run."""
    try:
        out = subprocess.run(
            [str(nvcc), "--version"], capture_output=True, text=True, timeout=30, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"release (\d+)\.(\d+)", out)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _nvidia_wheel_roots() -> list[Path]:
    """The ``nvidia`` namespace-package directories, one per site-packages tree."""
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return []
    return [Path(p) for p in spec.submodule_search_locations]


class WheelToolkit:
    """The pieces of a CUDA toolkit as pip actually scatters them.

    There is no guarantee any single directory holds a usable toolkit. On this
    machine nvcc and nvvm come from one site-packages tree, the runtime headers
    and libraries from another, and the CCCL headers that ``cuda_fp16.h``
    includes (``nv/target``) from a third wheel in yet another tree. So the
    include and library search paths are lists, and get merged.
    """

    def __init__(self, compiler_root: Path, include_dirs: list[Path], lib_dirs: list[Path]) -> None:
        self.compiler_root = compiler_root
        self.include_dirs = include_dirs
        self.lib_dirs = lib_dirs


def _find_wheel_toolkit(major: int) -> WheelToolkit | None:
    """Collect the wheel-provided toolkit pieces for CUDA ``major``."""
    candidates: list[Path] = []
    for root in _nvidia_wheel_roots():
        # `cu13` is the current consolidated layout; the split
        # `cuda_nvcc` / `cuda_runtime` / `cuda_cccl` layout is what older wheels use.
        candidates += [
            root / f"cu{major}",
            root / "cuda_nvcc",
            root / "cuda_runtime",
            root / "cuda_cccl",
        ]

    compiler = next((c for c in candidates if (c / "bin" / "nvcc").is_file()), None)
    if compiler is None:
        return None

    include_dirs = [c / "include" for c in candidates if (c / "include").is_dir()]
    lib_dirs = [c / "lib" for c in candidates if (c / "lib").is_dir()]

    # Sort the tree holding cuda_runtime.h first so it wins any name collision
    # during the merge; it is the authoritative copy of the core headers.
    include_dirs.sort(key=lambda d: not (d / "cuda_runtime.h").is_file())
    if not include_dirs or not (include_dirs[0] / "cuda_runtime.h").is_file():
        return None

    return WheelToolkit(compiler, include_dirs, lib_dirs)


# ------------------------------------------------------------ toolkit assembly


def _relink(link: Path, target: Path) -> None:
    """Point ``link`` at ``target``, replacing whatever was there before."""
    if link.is_symlink():
        if link.readlink() == target:
            return
        link.unlink()
    elif link.is_dir():
        shutil.rmtree(link)
    elif link.exists():
        link.unlink()
    link.symlink_to(target)


def _ensure_real_dir(path: Path) -> None:
    """Make ``path`` a real directory, replacing a symlink left by an older layout.

    Without this, ``mkdir(exist_ok=True)`` on a path that is currently a symlink
    to a wheel's include directory would silently succeed, and the symlinks would
    then be scattered *inside site-packages*.
    """
    if path.is_symlink():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def synthesize_cuda_home(toolkit: WheelToolkit, dest: Path = TOOLKIT_DIR) -> Path:
    """Assemble a directory that looks enough like a CUDA toolkit for torch.

    Layout produced::

        dest/bin      -> compiler_root/bin     (nvcc, ptxas, cudafe++, crt/)
        dest/nvvm     -> compiler_root/nvvm    (cicc and libdevice)
        dest/include/ -> merged symlinks from every wheel include dir
        dest/lib64/   -> merged symlinks from every wheel lib dir, plus sonames
        dest/lib      -> dest/lib64

    ``bin`` is linked as a whole directory on purpose: nvcc locates its nvvm
    backend relative to the real path of the binary, so linking the individual
    executables would leave it unable to find ``cicc``.
    """
    dest.mkdir(parents=True, exist_ok=True)
    _relink(dest / "bin", toolkit.compiler_root / "bin")
    if (toolkit.compiler_root / "nvvm").is_dir():
        _relink(dest / "nvvm", toolkit.compiler_root / "nvvm")

    include = dest / "include"
    _ensure_real_dir(include)
    for source_dir in toolkit.include_dirs:
        for src in sorted(source_dir.iterdir()):
            link = include / src.name
            if link.is_symlink() and link.readlink() != src:
                continue  # an earlier, higher-priority include dir already won
            _relink(link, src)

    lib64 = dest / "lib64"
    _ensure_real_dir(lib64)
    for source_dir in toolkit.lib_dirs:
        for src in sorted(source_dir.iterdir()):
            if src.is_dir():
                continue
            _relink(lib64 / src.name, src)
            # The wheels ship only versioned sonames (libcudart.so.13), but torch
            # links with -lcudart, which needs the unversioned name to exist.
            if ".so." in src.name:
                bare = lib64 / (src.name.split(".so.")[0] + ".so")
                if not bare.is_symlink():
                    _relink(bare, src)
    _relink(dest / "lib", lib64)

    return dest


def _candidate_homes() -> list[tuple[Path, str]]:
    """Existing toolkits to try before falling back to assembling one."""
    candidates: list[tuple[Path, str]] = []
    for var in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(var):
            candidates.append((Path(os.environ[var]), f"${var}"))

    from torch.utils.cpp_extension import CUDA_HOME as TORCH_CUDA_HOME

    if TORCH_CUDA_HOME:
        candidates.append((Path(TORCH_CUDA_HOME), "torch's detected CUDA_HOME"))

    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append((Path(nvcc).resolve().parent.parent, "nvcc on PATH"))

    return candidates


def resolve_cuda_home() -> tuple[Path, str]:
    """Find a CUDA toolkit whose nvcc major version matches torch's.

    Returns ``(cuda_home, how_it_was_found)``.
    """
    want = torch_cuda_major()

    for home, how in _candidate_homes():
        version = nvcc_version(home / "bin" / "nvcc")
        if version is not None and version[0] == want:
            return home, how

    wheel = _find_wheel_toolkit(want)
    if wheel is not None:
        home = synthesize_cuda_home(wheel)
        version = nvcc_version(home / "bin" / "nvcc")
        if version is not None and version[0] == want:
            return home, f"assembled from pip wheels into {home.relative_to(REPO_ROOT)}"

    tried = "\n".join(
        f"  - {how}: {home} (nvcc {nvcc_version(home / 'bin' / 'nvcc')})"
        for home, how in _candidate_homes()
    )
    raise ToolchainError(
        f"no CUDA toolkit found with major version {want} to match torch {torch.__version__}.\n"
        f"Tried:\n{tried or '  (nothing)'}\n"
        "Fix it one of these ways:\n"
        f"  1. pip install nvidia-cuda-nvcc=={want}.* (preferred; see requirements.txt)\n"
        f"  2. install a full CUDA {want} toolkit and set CUDA_HOME to it\n"
        f"  3. reinstall torch built against the CUDA you already have"
    )


# --------------------------------------------------------------------- loading


def _sources() -> list[str]:
    """Every translation unit in csrc/, bindings first."""
    sources = sorted(CSRC_DIR.glob("*.cpp")) + sorted(CSRC_DIR.glob("*.cu"))
    if not sources:
        raise ToolchainError(f"no CUDA sources found in {CSRC_DIR}")
    return [str(p) for p in sources]


def load_extension(verbose: bool = False) -> Any:
    """Compile (if needed) and return the ``csrc/`` extension module, cached."""
    global _extension
    if _extension is not None:
        return _extension

    if not torch.cuda.is_available():
        raise ToolchainError("no CUDA device available; the csrc/ extension needs a GPU to build")

    home, _how = resolve_cuda_home()

    # Both of these matter. The environment variables are what nvcc and any
    # subprocess see; the module attribute is what torch itself consults, and it
    # is captured once at import time, so setting only the environment would be
    # silently ignored whenever cpp_extension was imported before this point.
    os.environ["CUDA_HOME"] = str(home)
    os.environ["CUDA_PATH"] = str(home)
    os.environ["PATH"] = f"{home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"

    from torch.utils import cpp_extension

    cpp_extension.CUDA_HOME = str(home)

    EXT_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    _extension = cpp_extension.load(
        name=EXT_NAME,
        sources=_sources(),
        build_directory=str(EXT_BUILD_DIR),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", f"-arch={CUDA_ARCH}", "--expt-relaxed-constexpr"],
        verbose=verbose,
    )
    return _extension


def rebuild(verbose: bool = True) -> Any:
    """Discard the build cache and compile from scratch.

    `make ext`. Worth reaching for whenever a kernel edit appears to have no
    effect, which is usually a stale object file rather than a wrong kernel.
    """
    global _extension
    _extension = None
    if EXT_BUILD_DIR.exists():
        shutil.rmtree(EXT_BUILD_DIR)
    return load_extension(verbose=verbose)


def toolchain_report() -> dict[str, Any]:
    """Everything worth knowing about how this build will be configured."""
    home, how = resolve_cuda_home()
    version = nvcc_version(home / "bin" / "nvcc")
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_home": str(home),
        "cuda_home_found_via": how,
        "nvcc": f"{version[0]}.{version[1]}" if version else None,
        "arch": CUDA_ARCH,
        "sources": [Path(s).name for s in _sources()],
    }
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        report["capability"] = torch.cuda.get_device_capability(0)
        report["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build the Mini-vLLM CUDA extension.")
    parser.add_argument("--rebuild", action="store_true", help="delete the build cache first")
    args = parser.parse_args()

    for key, value in toolchain_report().items():
        print(f"{key:>20}: {value}")
    print()

    module = rebuild() if args.rebuild else load_extension(verbose=True)

    x = torch.randn(1024, device="cuda")
    got, want = module.hello(x, 2.0, 3.0), 2.0 * x + 3.0
    ok = torch.allclose(got, want)
    print(f"\nhello(x, 2, 3) == 2x + 3 : {'ok' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
