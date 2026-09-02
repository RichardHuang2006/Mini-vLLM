"""The CUDA kernels' front door: JIT build of ``csrc/`` and the dispatch seam.

What this file teaches
    How hand-written CUDA reaches a PyTorch program, in two halves:

    1. *The build* — `load_extension` JIT-compiles ``csrc/`` on first use,
       after `resolve_cuda_home` finds (or assembles from pip wheels) a CUDA
       toolkit whose major version matches torch's.
    2. *The dispatch* — one wrapper per op (`rmsnorm`, `rope`, `swiglu`,
       `attention`, `paged_attention`, `quantize_scatter`), each of which runs
       either the hand-written kernel or the reference implementation from
       `ops.py`. The model passes `use_cuda` down and never learns which ran.

Inputs and outputs
    The wrappers take and return plain tensors with the same signatures as
    their `ops.py` references. `dispatch_report` returns a human-readable
    statement of which path each op would take and why.

Read next
    `engine.py` — the serving loop these kernels accelerate. The kernels
    themselves are in ``csrc/``, one file per kernel, readable in the same
    order as the wrappers below.

One invariant
    `use_cuda=True` means "use kernels where they are implemented, correct,
    and *faster*". An op whose kernel measures slower than PyTorch is listed
    in `NOT_YET_FASTER` with the measured reason and keeps the reference path,
    so a benchmark can never silently route through a kernel that loses. The
    same wrapper with `use_cuda=False` is the oracle the kernel is diffed
    against, which is how the tests establish that a kernel changed the speed
    and not the output.

Runnable example
    ``python -m mini_vllm.kernels --rebuild`` — force a full rebuild, print
    the resolved toolchain, and verify the compiled rmsnorm kernel against
    `ops.rms_norm` on a real input. (Requires a CUDA GPU.)

The toolchain problem, for anyone hitting a build error
    ``torch.utils.cpp_extension`` refuses to compile when nvcc's CUDA major
    version differs from the one torch was built against, and on the machine
    of record they differ (torch is a cu130 build, the system nvcc is 12.8).
    The resolution is to use the CUDA 13 compiler shipped as pip wheels. Those
    wheels are split across packages — nvcc and its nvvm backend in one,
    runtime headers in another, the CCCL headers that ``cuda_fp16.h`` pulls in
    in a third — and pip may install them into different site-packages trees,
    so no single directory resembles a CUDA installation. `synthesize_cuda_home`
    assembles a symlink tree that does, under ``build/``, and points CUDA_HOME
    at it.
"""

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

# Blackwell, the RTX 5070 Laptop of record (compute capability 12.0). Targeting a single
# architecture keeps compiles fast.
CUDA_ARCH = "sm_120"

# Translation units to compile at once. Ninja defaults to one job per core, which is wrong
# here: nvcc on a template-heavy kernel peaks well past a gigabyte, this machine has 16
# cores, and WSL2 gives the VM about half the host's RAM. Sixteen concurrent jobs exhaust
# it and the build is OOM-killed, which presents as a crash with no failing test. Four
# fits the budget and costs little wall clock given the small number of sources.
DEFAULT_MAX_JOBS = 4

# Memory budget for one nvcc invocation, used to lower the job count on a machine with
# less memory than this one rather than assuming 15 GB.
BYTES_PER_JOB = 2 * 1024**3

_extension: Any = None

__all__ = [
    "ToolchainError",
    "load_extension",
    "rebuild",
    "toolchain_report",
    "CUDA_KERNELS",
    "NOT_YET_FASTER",
    "FP8_KERNEL_DTYPE",
    "cuda_kernel_names",
    "dispatch_report",
    "rmsnorm",
    "rope",
    "swiglu",
    "attention",
    "paged_attention",
    "quantize_scatter",
]


class ToolchainError(RuntimeError):
    """Raised when no CUDA toolkit matching torch's CUDA version can be found."""


# ======================================================================== build
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
    """The pieces of a CUDA toolkit as pip scatters them.

    No single directory is guaranteed to hold a usable toolkit. On this machine nvcc and
    nvvm come from one site-packages tree, the runtime headers and libraries from another,
    and the CCCL headers that ``cuda_fp16.h`` includes (``nv/target``) from a third wheel
    in a fourth tree. The include and library search paths are therefore lists, merged
    downstream.
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

    # Sort the tree holding cuda_runtime.h first so it wins name collisions during the
    # merge; it is the authoritative copy of the core headers.
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

    ``mkdir(exist_ok=True)`` on a path that is currently a symlink to a wheel's include
    directory succeeds, which would scatter the symlinks inside site-packages.
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

    ``bin`` is linked as a whole directory because nvcc locates its nvvm backend relative
    to the real path of the binary; linking individual executables leaves it unable to
    find ``cicc``.
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


def _available_bytes() -> int | None:
    """RAM the machine will give a compile, or None if it does not report it.

    ``MemAvailable`` rather than ``MemFree``: the page cache is reclaimable, so counting
    only free memory would throttle the build needlessly.
    """
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def max_jobs() -> int:
    """How many compiler processes to run at once, respecting an explicit override.

    ``MAX_JOBS`` is the variable ``torch.utils.cpp_extension`` reads, so an explicit
    setting passes through untouched.
    """
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

    # Both are required. The environment variables are what nvcc and any subprocess see;
    # the module attribute is what torch consults, and it is captured once at import time,
    # so setting only the environment is ignored when cpp_extension was imported earlier.
    os.environ["CUDA_HOME"] = str(home)
    os.environ["CUDA_PATH"] = str(home)
    os.environ["PATH"] = f"{home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"

    # Bound the compile before torch reads it; otherwise ninja fills every core and the
    # build is OOM-killed instead of failing for a diagnosable reason.
    os.environ["MAX_JOBS"] = str(max_jobs())

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

    Exposed as `make ext`. The usual cause of a kernel edit having no effect is a stale
    object file rather than a wrong kernel.
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


# ===================================================================== dispatch

# Which ops have a hand-written kernel, paired with the source file the dispatch report
# names beside each one.
#
# The keys are the names the extension exports rather than synonyms: the test suite
# looks each up in the compiled module, so a kernel cannot be claimed here while being
# missing, misspelled, or renamed.
CUDA_KERNELS: dict[str, tuple[bool, str]] = {
    "rmsnorm": (True, "csrc/rmsnorm.cu"),
    "rope": (True, "csrc/rope.cu"),
    "swiglu": (True, "csrc/swiglu.cu"),
    "decode_attention": (True, "csrc/decode_attention.cu"),
    "flash_prefill": (True, "csrc/flash_prefill.cu"),
    "paged_attention": (True, "csrc/paged_attention.cu"),
    "kv_quantize_scatter": (True, "csrc/kv_quantize.cu"),
}

# The one FP8 format the kernels accelerate. e5m2 remains a legal storage dtype for the
# pool and the PyTorch paths quantize and dequantize it correctly, but it does not justify
# a second set of template instantiations in every kernel, so a pool in it degrades to the
# oracle rather than being half-supported.
FP8_KERNEL_DTYPE = torch.float8_e4m3fn

# Kernels that are correct, tested, and not the default, with the reason measured by the
# benchmark. A kernel earns the dispatch by being faster, and routing to a slower one
# behind `use_cuda=True` is what `dispatch_report` exists to prevent.
#
# Prefill is listed because its inner loop is scalar FMA against cuBLAS on tensor cores,
# a gap tuning does not close; replacing the loop with `mma.sync` would flip this entry.
# The tests call the kernel directly through the extension, so its coverage does not
# depend on routing.
NOT_YET_FASTER: dict[str, str] = {
    "flash_prefill": "0.4x cuBLAS at L=512",
}

# The prefill kernel stages query, key and value tiles in shared memory simultaneously,
# which bounds the head dimension it can serve. Qwen3-0.6B uses 128; anything wider goes
# to the oracle.
MAX_KERNEL_HEAD_DIM = 192

# The paged kernel holds the query and the accumulator in registers instead, `D / 32`
# floats per lane of each, which is where its own ceiling comes from.
MAX_PAGED_HEAD_DIM = 256


def _use_kernel(name: str, use_cuda: bool, x: torch.Tensor) -> bool:
    """Whether ``name`` should run on the GPU for this call.

    Requires an implemented kernel that is the default and CUDA tensors: the same model
    object serves CPU tests and GPU runs.
    """
    implemented, _source = CUDA_KERNELS[name]
    return use_cuda and implemented and name not in NOT_YET_FASTER and x.is_cuda


def cuda_kernel_names() -> list[str]:
    """The ops that currently have a working kernel."""
    return [name for name, (implemented, _) in CUDA_KERNELS.items() if implemented]


def dispatch_report(use_cuda: bool) -> str:
    """One line per op saying which implementation a run would use.

    `use_cuda=True` means use kernels where they exist, so an op with no kernel falls
    back silently, as does any op called on CPU tensors. That silence is how a kernel
    that never ran ends up in a benchmark, so this states which path each op would take
    and the benchmark prints it.
    """
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


# ------------------------------------------------------------------------ ops


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    use_cuda: bool = False,
) -> torch.Tensor:
    """RMSNorm over the last dimension. Kernel: ``csrc/rmsnorm.cu``.

    The kernel requires ``weight`` to have the same dtype as ``x``. PyTorch would promote
    instead, and returning a different dtype than the oracle is worse than declining the
    kernel. The model never mixes them.
    """
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
    """Rotary embedding at explicit positions. Kernel: ``csrc/rope.cu``.

    The kernel requires fp32 tables, which `ops.RoPE` builds regardless of the activation
    dtype, and a head axis to broadcast the position across. Anything else goes to the
    oracle.
    """
    if _use_kernel("rope", use_cuda, x) and cos.dtype == torch.float32 and x.dim() >= 3:
        return load_extension().rope(x, positions, cos, sin)
    return reference.apply_rope(x, positions, cos, sin)


def swiglu(gate: torch.Tensor, up: torch.Tensor, use_cuda: bool = False) -> torch.Tensor:
    """``silu(gate) * up``, the elementwise half of the MLP. Kernel: ``csrc/swiglu.cu``.

    Takes the two projections rather than the input: the projections are ordinary matmuls
    that cuBLAS handles better than a hand-written kernel. What is worth fusing is this
    part, two full passes over a `B x L x intermediate` tensor for a few flops per
    element, which is pure memory traffic.
    """
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
    """Grouped-query attention, with a separate kernel for decode and for prefill.

    ::

        q:    B x H_q x L x D
        k, v: B x H_k x S x D
        out:  B x H_q x L x D

    Decode and prefill are different problems rather than different sizes of one. Decode
    has `L = 1`: no parallelism across queries, one long pass over the cache, entirely
    memory-bound. Prefill has `L = S`: a matrix multiply worth tiling. Hence separate
    kernels, ``csrc/decode_attention.cu`` and ``csrc/flash_prefill.cu``, with the routing
    here so the model does not have to choose.
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
    """Ragged attention over a paged cache. Kernel: ``csrc/paged_attention.cu``.

    ::

        q:    T x H_q x D              every scheduled token, sequences concatenated
        pools num_blocks x P x H_k x D
        out:  T x H_q x D

    The fallback differs in kind from the others here. The PyTorch path for `rmsnorm` is
    a slower way to do the same work; the PyTorch path for this copies every sequence's
    whole cache into a contiguous temporary first, which is the traffic paging exists to
    remove. It is the oracle rather than an alternative, and the engine's throughput
    numbers are only meaningful on the kernel path.

    `max_query_len` and `max_context_len` are plain integers rather than reads off
    `seq_lens` and `context_lens`. They decide a grid shape and a split count, so reading
    them from a device tensor would synchronize on every iteration's critical path, and
    the caller already holds them as Python integers from building the batch.

    With an FP8 e4m3 pool the kernel dequantizes in registers: `k_scale` folds into the
    softmax scale at the launch site and `v_scale` rides on the output, so the inner
    loops are unchanged and an FP8 element costs only its conversion on load.
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
    """Quantize a step's KV to FP8 and scatter it into the pool. Kernel: ``csrc/kv_quantize.cu``.

    ``key_pool`` and ``value_pool`` are one layer's pages flattened to
    ``num_slots x H_k x D``, the same view :func:`paged_attention` reads back. Writes in
    place, returns nothing. The oracle is the two-pass PyTorch it replaces — quantize into
    a temporary, then ``index_copy_`` — and the two agree bit for bit: both divide by the
    scale in fp32 and round to nearest even.
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


# ------------------------------------------------------------- dispatch guards


def _kernel_can_page(
    q: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_tables: torch.Tensor,
) -> bool:
    """Whether the paged kernel can serve this call.

    The kernel walks the pools by slot arithmetic rather than by stride, so they must be
    contiguous: a narrowed view would read the wrong pages rather than read slowly, which
    is why this is a condition and not a copy.

    FP8 pools are served as well, dequantized in registers, so the pools may differ from
    the query in dtype provided they agree with each other and use a storage type the
    kernel knows.
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

    Both kernels encode the causal structure in arithmetic rather than reading a mask
    tensor — decode attends to the whole cache, prefill compares each index against the
    diagonal — so `mask="causal"` is the only mask they can honour. An explicit tensor
    could express anything, so it falls back to the oracle rather than being ignored.
    """
    is_causal_shorthand = isinstance(mask, str) and mask == "causal"
    if is_decode:
        # A single query token may attend to every position cached before it, so the
        # causal mask forbids nothing and `None` requests the same thing.
        if mask is not None and not is_causal_shorthand:
            return False
    elif not is_causal_shorthand:
        # Prefill is the reverse: the kernel always masks causally, so it cannot serve
        # an unmasked multi-token call either.
        return False
    if q.dim() != 4 or q.dtype != k.dtype or q.dtype != v.dtype or q.dtype == torch.float64:
        return False
    if q.shape[-1] > MAX_KERNEL_HEAD_DIM or k.shape[-2] < q.shape[-2]:
        return False
    return k.shape[-2] > 0 and q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1


# ---------------------------------------------------------------- CLI: make ext


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build the Mini-vLLM CUDA extension.")
    parser.add_argument("--rebuild", action="store_true", help="delete the build cache first")
    args = parser.parse_args()

    for key, value in toolchain_report().items():
        print(f"{key:>20}: {value}")
    print()

    module = rebuild() if args.rebuild else load_extension(verbose=True)

    # Verify the freshly built extension with a real kernel against its reference:
    # rmsnorm on an actual model-sized input, rather than a synthetic smoke kernel.
    x = torch.randn(64, 1024, device="cuda", dtype=torch.float32)
    weight = torch.randn(1024, device="cuda", dtype=torch.float32)
    got = module.rmsnorm(x, weight, 1e-6)
    want = reference.rms_norm(x, weight, 1e-6)
    ok = torch.allclose(got, want, rtol=1e-5, atol=1e-5)
    print(f"\nrmsnorm(64x1024 fp32) matches ops.rms_norm : {'ok' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
