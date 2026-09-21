"""Mini-vLLM — a paged-attention inference engine for Qwen3."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from mini_vllm.config import EngineConfig, SamplingParams

if TYPE_CHECKING:
    from mini_vllm.engine import LLM, Completion, StreamUpdate

__all__ = ["LLM", "Completion", "EngineConfig", "SamplingParams", "StreamUpdate"]

# Re-exported lazily so importing a reference op does not pull in transformers.
_ENGINE_EXPORTS = frozenset({"LLM", "Completion", "StreamUpdate"})


def __getattr__(name: str) -> Any:
    if name in _ENGINE_EXPORTS:
        return getattr(import_module("mini_vllm.engine"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
