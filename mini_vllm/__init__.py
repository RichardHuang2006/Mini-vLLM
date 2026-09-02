"""Mini-vLLM — a paged-attention LLM inference engine for Qwen3.

::

    from mini_vllm import LLM, SamplingParams

    llm = LLM()
    print(llm.generate("The capital of France is", max_tokens=16)[0].text)

The public API is five names: `LLM`, `EngineConfig`, `SamplingParams`,
`Completion`, and `StreamUpdate`. Everything else is an implementation detail,
importable from its module for study (`mini_vllm.ops`, `mini_vllm.cache`, ...)
but not re-exported here.

The engine names are imported lazily so that `import mini_vllm.ops`, in tests
of the reference implementations, does not pull in transformers and the model
loader.
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
