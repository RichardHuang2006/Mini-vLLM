"""The CUDA kernels' front door: the JIT build of csrc/ and the dispatch seam."""

from __future__ import annotations

import importlib.util
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from mini_vllm import ops as reference

REPO_ROOT = Path(__file__).resolve().parents[1]
CSRC_DIR = REPO_ROOT / "csrc"
BUILD_DIR = REPO_ROOT / "build"
EXT_BUILD_DIR = BUILD_DIR / "torch_extensions"
TOOLKIT_DIR = BUILD_DIR / "cuda_toolkit"

EXT_NAME = "mini_vllm_C"

# Blackwell, the RTX 5070 Laptop of record. One architecture keeps compiles fast.
CUDA_ARCH = "sm_120"

# Translation units at once: ninja's one-job-per-core default OOMs on these kernels.
DEFAULT_MAX_JOBS = 4

# Memory budget for one nvcc invocation, used to lower the job count on a smaller machine.
BYTES_PER_JOB = 2 * 1024**3

_extension: Any = None

__all__ = [
    "CUDA_KERNELS",
    "FP8_KERNEL_DTYPE",
    "NOT_YET_FASTER",
    "ToolchainError",
    "attention",
    "cuda_kernel_names",
    "dispatch_report",
    "load_extension",
    "paged_attention",
    "quantize_scatter",
    "rebuild",
    "rmsnorm",
    "rope",
    "swiglu",
    "toolchain_report",
]


class ToolchainError(RuntimeError):
    """Raised when no CUDA toolkit matching torch's CUDA version can be found."""


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
    """The pieces of a CUDA toolkit as pip scatters them, hence lists of search paths."""

    def __init__(self, compiler_root: Path, include_dirs: list[Path], lib_dirs: list[Path]) -> None:
        self.compiler_root = compiler_root
        self.include_dirs = include_dirs
        self.lib_dirs = lib_dirs


def _find_wheel_toolkit(major: int) -> WheelToolkit | None:
    """Collect the wheel-provided toolkit pieces for CUDA ``major``."""
    candidates: list[Path] = []
    for root in _nvidia_wheel_roots():
        # `cu13` is the current consolidated layout; the split one is what older wheels use.
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

    # Sort the tree holding cuda_runtime.h first so it wins name collisions in the merge.
    include_dirs.sort(key=lambda d: not (d / "cuda_runtime.h").is_file())
    if not include_dirs or not (include_dirs[0] / "cuda_runtime.h").is_file():
        return None

    return WheelToolkit(compiler, include_dirs, lib_dirs)


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
    """Make ``path`` a real directory, replacing a symlink left by an older layout."""
    if path.is_symlink():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def synthesize_cuda_home(toolkit: WheelToolkit, dest: Path = TOOLKIT_DIR) -> Path:
    """Assemble a directory that looks enough like a CUDA toolkit for torch.

    bin and nvvm are linked whole because nvcc locates its backend relative to the real
    path of the binary; include and lib64 are merged symlinks from every wheel.
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
            # The wheels ship only versioned sonames, but torch links with -lcudart.
            if ".so." in src.name:
                bare = lib64 / (src.name.split(".so.")[0] + ".so")
                if not bare.is_symlink():
                    _relink(bare, src)
    _relink(dest / "lib", lib64)

    return dest


def _candidate_homes() -> list[tuple[Path, str]]:
    """Existing toolkits to try before falling back to assembling one."""
    # Imported here, not at module scope: importing cpp_extension probes for nvcc.
    from torch.utils import cpp_extension

    candidates: list[tuple[Path, str]] = []
    for var in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(var):
            candidates.append((Path(os.environ[var]), f"${var}"))

    if cpp_extension.CUDA_HOME:
        candidates.append((Path(cpp_extension.CUDA_HOME), "torch's detected CUDA_HOME"))

    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append((Path(nvcc).resolve().parent.parent, "nvcc on PATH"))

    return candidates


def resolve_cuda_home() -> tuple[Path, str]:
    """Find a CUDA toolkit whose nvcc major matches torch's: (cuda_home, how it was found)."""
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


def _available_bytes() -> int | None:
    """RAM the machine will give a compile, or None if it does not report it."""
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                # MemAvailable rather than MemFree: the page cache is reclaimable.
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def max_jobs() -> int:
    """How many compiler processes to run at once, respecting an explicit MAX_JOBS."""
    override = os.environ.get("MAX_JOBS")
    if override:
        return max(1, int(override))

    jobs = min(DEFAULT_MAX_JOBS, os.cpu_count() or 1)
    available = _available_bytes()
    if available is not None:
        jobs = min(jobs, max(1, available // BYTES_PER_JOB))
    return max(1, jobs)


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

    os.environ["CUDA_HOME"] = str(home)
    os.environ["CUDA_PATH"] = str(home)
    os.environ["PATH"] = f"{home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"

    # Bound the compile before torch reads it, or the build is OOM-killed instead.
    os.environ["MAX_JOBS"] = str(max_jobs())

    from torch.utils import cpp_extension

    # The environment is what nvcc sees; this attribute is what torch consults.
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
    """Discard the build cache and compile from scratch. Exposed as `make ext`."""
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


# Which ops have a kernel, keyed by the name the extension exports so tests can check.
CUDA_KERNELS: dict[str, tuple[bool, str]] = {
    "rmsnorm": (True, "csrc/rmsnorm.cu"),
    "rope": (True, "csrc/rope.cu"),
    "swiglu": (True, "csrc/swiglu.cu"),
    "decode_attention": (True, "csrc/decode_attention.cu"),
    "flash_prefill": (True, "csrc/flash_prefill.cu"),
    "paged_attention": (True, "csrc/paged_attention.cu"),
    "kv_quantize_scatter": (True, "csrc/kv_quantize.cu"),
}

# The one FP8 format the kernels accelerate; an e5m2 pool takes the reference path.
FP8_KERNEL_DTYPE = torch.float8_e4m3fn

# Correct, tested kernels that are not the default: the dispatch is earned by speed.
NOT_YET_FASTER: dict[str, str] = {
    "flash_prefill": "0.4x cuBLAS at L=512",
}

# The prefill kernel stages three tiles in shared memory, bounding the head dimension.
MAX_KERNEL_HEAD_DIM = 192

# The paged kernel holds the query and the accumulator in registers instead.
MAX_PAGED_HEAD_DIM = 256


def _use_kernel(name: str, use_cuda: bool, x: torch.Tensor) -> bool:
    """Whether ``name`` should run on the GPU for this call."""
    implemented, _source = CUDA_KERNELS[name]
    return use_cuda and implemented and name not in NOT_YET_FASTER and x.is_cuda


def cuda_kernel_names() -> list[str]:
    """The ops that currently have a working kernel."""
    return [name for name, (implemented, _) in CUDA_KERNELS.items() if implemented]


def dispatch_report(use_cuda: bool) -> str:
    """One line per op saying which implementation a run would use, and why."""
    lines = []
    for name, (implemented, source) in CUDA_KERNELS.items():
        if not implemented:
            note = f"no kernel yet — {source}"
        elif name in NOT_YET_FASTER:
            note = f"kernel from {source} is slower: {NOT_YET_FASTER[name]}"
        elif not use_cuda:
            note = f"kernel exists from {source}; use_cuda is off"
        else:
            lines.append(f"  {name:<18} cuda    (kernel from {source})")
            continue
        lines.append(f"  {name:<18} torch   ({note})")
    return "\n".join(lines)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    use_cuda: bool = False,
) -> torch.Tensor:
    """RMSNorm over the last dimension. Kernel: ``csrc/rmsnorm.cu``."""
    # The kernel needs a matching weight dtype; PyTorch would promote instead.
    if _use_kernel("rmsnorm", use_cuda, x) and weight.dtype == x.dtype:
        return load_extension().rmsnorm(x, weight, eps)
    return reference.rms_norm(x, weight, eps)


def rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    use_cuda: bool = False,
) -> torch.Tensor:
    """Rotary embedding at explicit positions. Kernel: ``csrc/rope.cu``."""
    # The kernel needs fp32 tables, which `ops.RoPE` always builds, and a head axis.
    if _use_kernel("rope", use_cuda, x) and cos.dtype == torch.float32 and x.dim() >= 3:
        return load_extension().rope(x, positions, cos, sin)
    return reference.apply_rope(x, positions, cos, sin)


def swiglu(gate: torch.Tensor, up: torch.Tensor, use_cuda: bool = False) -> torch.Tensor:
    """silu(gate) * up, the elementwise half of the MLP. Kernel: csrc/swiglu.cu."""
    if _use_kernel("swiglu", use_cuda, gate) and gate.dtype == up.dtype:
        return load_extension().swiglu(gate, up)
    return reference.silu(gate) * up


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | str | None = None,
    use_cuda: bool = False,
) -> torch.Tensor:
    """Grouped-query attention, q [B, H_q, L, D] and k/v [B, H_k, S, D].

    Decode and prefill are different problems rather than different sizes of one, hence
    csrc/decode_attention.cu and csrc/flash_prefill.cu.
    """
    is_decode = q.shape[-2] == 1
    name = "decode_attention" if is_decode else "flash_prefill"

    if _use_kernel(name, use_cuda, q) and _kernel_can_attend(q, k, v, mask, is_decode):
        scale = 1.0 / math.sqrt(q.shape[-1])
        if is_decode:
            return load_extension().decode_attention(q, k, v, scale)
        return load_extension().flash_prefill(q, k, v, scale)

    return reference.scaled_dot_product_attention_grouped(q, k, v, mask=mask)


def paged_attention(
    q: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    context_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    max_context_len: int,
    scale: float | None = None,
    use_cuda: bool = False,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> torch.Tensor:
    """Ragged attention over a paged cache: q [T, H_q, D]. Kernel: csrc/paged_attention.cu.

    max_query_len and max_context_len are host integers because they decide a grid shape,
    and reading them off a device tensor would synchronize on the critical path.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    if _use_kernel("paged_attention", use_cuda, q) and _kernel_can_page(
        q, key_pool, value_pool, block_tables
    ):
        return load_extension().paged_attention(
            q,
            key_pool,
            value_pool,
            block_tables,
            cu_seqlens_q,
            context_lens,
            seq_lens,
            int(max_query_len),
            int(max_context_len),
            scale,
            k_scale,
            v_scale,
        )

    return reference.paged_attention_gathered(
        q,
        key_pool,
        value_pool,
        block_tables,
        cu_seqlens_q,
        context_lens,
        scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )


def quantize_scatter(
    key: torch.Tensor,
    value: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: float,
    v_scale: float,
    use_cuda: bool = False,
) -> None:
    """Quantize a step's KV to FP8 and scatter it into the pool, in place.

    Kernel: csrc/kv_quantize.cu. The two-pass PyTorch below is its oracle, and the two
    agree bit for bit: both divide by the scale in fp32 and round to nearest even.
    """
    if _use_kernel("kv_quantize_scatter", use_cuda, key) and key_pool.dtype is FP8_KERNEL_DTYPE:
        load_extension().kv_quantize_scatter(
            key.contiguous(),
            value.contiguous(),
            key_pool,
            value_pool,
            slot_mapping.to(torch.int64),
            float(k_scale),
            float(v_scale),
        )
        return

    index = slot_mapping.to(device=key_pool.device, dtype=torch.int64)
    quant_keys = (key.float() / k_scale).to(key_pool.dtype)
    quant_values = (value.float() / v_scale).to(value_pool.dtype)
    key_pool.view(torch.uint8).index_copy_(0, index, quant_keys.view(torch.uint8))
    value_pool.view(torch.uint8).index_copy_(0, index, quant_values.view(torch.uint8))


def _kernel_can_page(
    q: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_tables: torch.Tensor,
) -> bool:
    """Whether the paged kernel can serve this call.

    It walks the pools by slot arithmetic rather than by stride, so they must be
    contiguous: a narrowed view would read the wrong pages rather than read slowly.
    """
    pools_match = key_pool.dtype == value_pool.dtype
    pool_is_fp8 = key_pool.dtype is FP8_KERNEL_DTYPE
    dtype_ok = pools_match and (key_pool.dtype == q.dtype or pool_is_fp8)
    if q.dim() != 3 or not dtype_ok:
        return False
    if q.dtype == torch.float64 or q.shape[-1] > MAX_PAGED_HEAD_DIM:
        return False
    if not (q.is_contiguous() and key_pool.is_contiguous() and value_pool.is_contiguous()):
        return False
    return block_tables.dtype == torch.int32 and block_tables.is_contiguous()


def _kernel_can_attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | str | None,
    is_decode: bool,
) -> bool:
    """Whether the attention kernels can serve this exact call.

    Both encode the causal structure in arithmetic rather than reading a mask tensor, so
    an explicit mask falls back to the oracle rather than being ignored.
    """
    is_causal_shorthand = isinstance(mask, str) and mask == "causal"
    if is_decode:
        # A single query attends over everything cached, so a causal mask forbids nothing.
        if mask is not None and not is_causal_shorthand:
            return False
    elif not is_causal_shorthand:
        # Prefill always masks causally, so it cannot serve an unmasked multi-token call.
        return False
    if q.dim() != 4 or q.dtype != k.dtype or q.dtype != v.dtype or q.dtype == torch.float64:
        return False
    if q.shape[-1] > MAX_KERNEL_HEAD_DIM or k.shape[-2] < q.shape[-2]:
        return False
    return k.shape[-2] > 0 and q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build the Mini-vLLM CUDA extension.")
    parser.add_argument("--rebuild", action="store_true", help="delete the build cache first")
    args = parser.parse_args()

    for key, value in toolchain_report().items():
        print(f"{key:>20}: {value}")
    print()

    module = rebuild() if args.rebuild else load_extension(verbose=True)

    # Verify the fresh build with a real kernel on a model-sized input.
    x = torch.randn(64, 1024, device="cuda", dtype=torch.float32)
    weight = torch.randn(1024, device="cuda", dtype=torch.float32)
    got = module.rmsnorm(x, weight, 1e-6)
    want = reference.rms_norm(x, weight, 1e-6)
    ok = torch.allclose(got, want, rtol=1e-5, atol=1e-5)
    print(f"\nrmsnorm(64x1024 fp32) matches ops.rms_norm : {'ok' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
