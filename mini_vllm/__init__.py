"""Mini-vLLM — a paged-attention LLM inference engine for Qwen3.

::

    from mini_vllm import LLM

    llm = LLM()
    print(llm.generate("The capital of France is", max_tokens=16)[0].text)

`LLM` lives in `mini_vllm.serve.engine`, which is where it belongs but not how it
should be spelled: the module path is an implementation detail of an engine assembled
from a scheduler, a block manager and a paged model, and the caller needs none of that.
Imported lazily so that `import mini_vllm.basics` in a test of the readable reference
implementation does not drag in transformers and a model loader.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mini_vllm.serve.engine import LLM, Completion, EngineStats, StreamUpdate

__all__ = ["LLM", "Completion", "EngineStats", "StreamUpdate"]

_ENGINE_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    if name in _ENGINE_EXPORTS:
        return getattr(import_module("mini_vllm.serve.engine"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
