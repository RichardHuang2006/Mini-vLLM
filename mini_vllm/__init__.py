"""Mini-vLLM — a paged-attention LLM inference engine for Qwen3.

::

    from mini_vllm import LLM

    llm = LLM()
    print(llm.generate("The capital of France is", max_tokens=16)[0].text)

`LLM` is defined in `mini_vllm.serve.engine` and re-exported here: the module path is an
implementation detail of an engine assembled from a scheduler, a block manager and a
paged model. The import is lazy so that `import mini_vllm.basics`, in tests of the
reference implementation, does not pull in transformers and the model loader.
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
