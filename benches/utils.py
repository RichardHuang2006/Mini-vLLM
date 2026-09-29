"""Shared by every bench: timing at mx.eval boundaries, the evidence header, the thermal verdict,
the models every engine in a run shares, and the plots written to results/."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import random
import re
import statistics
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx_lm
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mini_vllm import LLM, EngineConfig, StreamUpdate, from_mlx

try:
    import mini_vllm_ext  # noqa: F401

    METAL_BUILT = True
except ImportError:
    METAL_BUILT = False

RESULTS = Path(__file__).resolve().parent.parent / "results"


def per_call_us(fn: Callable[[], Any], calls: int = 50, repeats: int = 20) -> float:
    """Median microseconds per call of fn. MLX is lazy, so nothing runs until mx.eval: each
    timed eval takes `calls` calls at once, which amortizes the ~175 µs per-eval launch floor
    that would otherwise swamp a small kernel."""
    for _ in range(3):
        mx.eval([fn() for _ in range(calls)])
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval([fn() for _ in range(calls)])
        times.append((time.perf_counter() - start) / calls)
    return statistics.median(times) * 1e6


def percentile(values: list[float], p: int) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[p - 1]


def token_times(updates: Iterator[StreamUpdate]) -> dict[int, list[float]]:
    """Seconds from the start of a stream to each of its tokens, per prompt: a prompt's first
    time is its TTFT and the gaps between the rest are its inter-token latency. Pass the
    stream unstarted, since the clock starts here."""
    start = time.perf_counter()
    times = defaultdict(list)
    for update in updates:
        times[update.index].append(time.perf_counter() - start)
    return times


def _run(*command: str) -> str:
    return subprocess.run(command, capture_output=True, text=True).stdout


def _process_info(selector: str) -> int:
    """An integer property of NSProcessInfo, through the Objective-C runtime."""
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
    objc.objc_getClass.restype = objc.sel_registerName.restype = ctypes.c_void_p
    objc.objc_msgSend.restype = ctypes.c_long
    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    info = ctypes.c_void_p(objc.objc_msgSend(objc.objc_getClass(b"NSProcessInfo"),
                                             objc.sel_registerName(b"processInfo")))
    return objc.objc_msgSend(info, objc.sel_registerName(selector.encode()))


def throttle_reasons() -> list[str]:
    """Why the machine may be running slow right now; empty when it is not. macOS's thermal
    state covers the GPU too, and Low Power Mode caps it on its own."""
    reasons = []
    state = _process_info("thermalState")
    if state:
        reasons.append(f"thermal state {['nominal', 'fair', 'serious', 'critical'][state]}")
    if _process_info("isLowPowerModeEnabled") & 0xFF:  # a BOOL: only the low byte is defined
        reasons.append("Low Power Mode is on")
    return reasons


@contextmanager
def evidence(title: str) -> Iterator[None]:
    """Print what a number needs beside it to mean anything, run the bench, then say whether
    the machine was throttled at either end, which voids the numbers as absolute figures."""
    info = mx.device_info()
    commit = _run("git", "rev-parse", "--short", "HEAD").strip()
    dirty = " + uncommitted changes" if _run("git", "status", "--porcelain").strip() else ""
    power = re.search(r"'(.+?)'", _run("pmset", "-g", "ps"))
    print(f"# {title}\n")
    print(f"- {info['device_name']}, {info['memory_size'] / 2**30:.0f} GB unified memory, "
          f"macOS {platform.mac_ver()[0]}, on {power.group(1) if power else 'unknown power'}")
    print(f"- mini-vllm {commit}{dirty}; mlx {mx.__version__}, mlx-lm {version('mlx-lm')}, "
          f"Python {platform.python_version()}")
    print(f"- Metal extension {'built' if METAL_BUILT else 'not built'}; "
          f"MLX_ENABLE_TF32={os.environ.get('MLX_ENABLE_TF32', 'unset')}")
    print(f"- {time.strftime('%Y-%m-%d %H:%M')}\n")

    before = throttle_reasons()
    yield
    after = throttle_reasons()
    if before or after:
        print("\n**THROTTLED** — these numbers are invalid as absolute figures:")
        for reason in dict.fromkeys(before + after):
            print(f"- {reason}")
    else:
        print("\nThermal state nominal and Low Power Mode off, before and after the run.")


def load_mlx_lm(model: str) -> tuple[Any, Any]:
    """The mlx_lm model and tokenizer that the baselines run and every engine borrows weights
    from. EOS is disabled, so every engine runs every request to max_tokens: the same work."""
    mlx_model, tokenizer = mlx_lm.load(model)
    tokenizer.eos_token_ids = []
    return mlx_model, tokenizer


def engine(mlx_model: Any, tokenizer: Any, use_metal: bool, **config: Any) -> LLM:
    """An engine over mlx_lm's weights, warmed up: the first pass pays Metal pipeline builds."""
    llm = LLM(from_mlx(mlx_model, use_metal), tokenizer, EngineConfig(**config))
    llm.generate([[1] * 32, [2] * 8], max_tokens=4)
    return llm


def random_prompts(count: int, lengths: Iterable[int], vocab: int, seed: int = 0) -> list[list[int]]:
    """count token-id prompts cycling through lengths, the same on every run."""
    rng = random.Random(seed)
    cycle = list(lengths)
    return [[rng.randrange(vocab) for _ in range(cycle[i % len(cycle)])] for i in range(count)]


def variants() -> list[tuple[str, bool]]:
    """(label, use_metal) for each engine this machine can run."""
    return [("mini-vllm pure", False)] + ([("mini-vllm metal", True)] if METAL_BUILT else [])


def grouped_bars(axes: Axes, groups: list[str], series: dict[str, list[float]],
                 horizontal: bool = False) -> None:
    """One cluster of bars per group, one bar per series in each; a nan value draws no bar."""
    width = 0.8 / len(series)
    for i, (label, values) in enumerate(series.items()):
        positions = [group + i * width for group in range(len(groups))]
        if horizontal:
            axes.barh(positions, values, width, label=label)
        else:
            axes.bar(positions, values, width, label=label)
    centers = [group + width * (len(series) - 1) / 2 for group in range(len(groups))]
    if horizontal:
        axes.set_yticks(centers, groups)
    else:
        axes.set_xticks(centers, groups)
        axes.margins(y=0.4)  # headroom, so the legend sits above the bars
    axes.legend()


def save_plot(figure: Figure, name: str) -> None:
    """Write figure to results/<name>.png, replacing the previous run's."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nPlot: {path.relative_to(RESULTS.parent)}")
