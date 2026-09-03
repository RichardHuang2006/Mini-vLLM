"""Mini-vLLM — a paged-attention inference engine for Qwen3.

config, ops, model, cache, scheduler, kernels, engine, speculative, benchmark.
Five names are re-exported, lazily so a test of the reference ops does not pull
in transformers: LLM, EngineConfig, SamplingParams, Completion, StreamUpdate.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from mini_vllm.config import EngineConfig, SamplingParams

if TYPE_CHECKING:
    from mini_vllm.engine import LLM, Completion, StreamUpdate

__all__ = ["LLM", "EngineConfig", "SamplingParams", "Completion", "StreamUpdate"]

_ENGINE_EXPORTS = frozenset({"LLM", "Completion", "StreamUpdate"})


def __getattr__(name: str) -> Any:
    if name in _ENGINE_EXPORTS:
        return getattr(import_module("mini_vllm.engine"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
