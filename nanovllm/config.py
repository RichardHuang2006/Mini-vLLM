"""Engine configuration and KV cache sizing arithmetic.

The interesting part of this module is not the dataclass, it is
:meth:`Config.kv_cache_bytes_per_block`. That number decides how many blocks
fit in GPU memory, which decides how many sequences can run concurrently,
which is the whole point of the engine. Step 4.3 sizes the block pool from it.

Nothing here imports ``transformers``. The HuggingFace config is read
structurally via :func:`kv_geometry`, so this module stays cheap to import and
trivial to test with a stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

__all__ = ["Config", "KVGeometry", "kv_geometry", "resolve_dtype"]


def resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Turn ``"float16"`` into :data:`torch.float16`, and pass dtypes through."""
    if isinstance(dtype, torch.dtype):
        return dtype
    resolved = getattr(torch, dtype, None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"unknown dtype {dtype!r}")
    return resolved


class KVGeometry(NamedTuple):
    """The three model dimensions that determine KV cache size.

    Note what is *absent*: ``num_attention_heads``. Query heads cost activation
    memory but no cache, because only K and V are stored. Under GQA that is a
    large saving -- Qwen2.5-0.5B has 14 query heads but caches only 2.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int


def kv_geometry(hf_config: Any) -> KVGeometry:
    """Extract KV cache dimensions from a HuggingFace config object.

    Two model-specific traps are handled here, both of which silently produce a
    wrongly-sized cache if you get them wrong:

    * ``num_key_value_heads`` is absent on models without GQA, where it equals
      ``num_attention_heads``.
    * ``head_dim`` is sometimes ``None`` (Qwen2.5) and must be derived, but is
      sometimes set explicitly to a value that is *not* the derived one (Qwen3
      uses 128 where ``hidden_size // num_attention_heads`` would give 64). So
      an explicit value always wins; deriving is only a fallback.
    """
    num_layers = _require(hf_config, "num_hidden_layers")
    num_heads = _require(hf_config, "num_attention_heads")

    num_kv_heads = getattr(hf_config, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = num_heads  # no GQA: every query head has its own KV head

    head_dim = getattr(hf_config, "head_dim", None)
    if head_dim is None:
        head_dim, remainder = divmod(_require(hf_config, "hidden_size"), num_heads)
        if remainder:
            raise ValueError(
                f"cannot derive head_dim: hidden_size={hf_config.hidden_size} is not "
                f"divisible by num_attention_heads={num_heads}"
            )

    if num_heads % num_kv_heads:
        raise ValueError(
            f"num_attention_heads={num_heads} is not divisible by "
            f"num_key_value_heads={num_kv_heads}; GQA grouping would be ragged"
        )

    return KVGeometry(num_layers, num_kv_heads, head_dim)


def _require(hf_config: Any, field: str) -> int:
    value = getattr(hf_config, field, None)
    if value is None:
        raise ValueError(f"model config is missing required field {field!r}")
    return int(value)


@dataclass(frozen=True)
class Config:
    """Engine configuration.

    ``dtype`` and ``kv_cache_dtype`` accept strings for convenience but are
    normalized to :class:`torch.dtype` during construction, so everything
    downstream can assume the resolved type.
    """

    model: str

    # DESIGN.md 3.1: 16 balances internal fragmentation against block-table size.
    block_size: int = 16

    max_seq_len: int = 4096
    max_num_seqs: int = 256

    # DESIGN.md 4.3: the per-iteration token budget shared by prefill chunks and
    # piggybacked decode steps.
    max_num_batched_tokens: int = 2048

    gpu_memory_utilization: float = 0.90

    dtype: str | torch.dtype = torch.float16
    kv_cache_dtype: str | torch.dtype | None = None  # None: follow `dtype`
    enable_prefix_caching: bool = True

    def __post_init__(self) -> None:
        # object.__setattr__ is the standard way to normalize fields on a frozen
        # dataclass: __post_init__ runs before the instance is handed out, so
        # this is initialization rather than mutation.
        dtype = resolve_dtype(self.dtype)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(
            self,
            "kv_cache_dtype",
            dtype if self.kv_cache_dtype is None else resolve_dtype(self.kv_cache_dtype),
        )
        self._validate()

    def _validate(self) -> None:
        if not self.model:
            raise ValueError("model must be a non-empty identifier")

        if self.block_size <= 0 or self.block_size & (self.block_size - 1):
            raise ValueError(
                f"block_size must be a positive power of two, got {self.block_size}. "
                "Position-to-block mapping uses shifts and masks."
            )

        # A block is the allocation unit, so an iteration that cannot hold one
        # block's worth of tokens cannot make progress on a prefill.
        if self.max_num_batched_tokens < self.block_size:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must be at "
                f"least block_size ({self.block_size})"
            )

        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")

        if self.max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be positive, got {self.max_num_seqs}")

        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}"
            )

        for name in ("dtype", "kv_cache_dtype"):
            dtype = getattr(self, name)
            if not dtype.is_floating_point:
                raise ValueError(f"{name} must be a floating-point type, got {dtype}")

    def kv_cache_bytes_per_block(self, hf_config: Any) -> int:
        """Bytes one block occupies across the *whole* model.

        ``2`` is K and V. ``num_layers`` is there because a block id is global:
        allocating block 7 reserves slot 7 in every layer's cache. Forgetting
        that factor is the classic sizing bug, and it is asymmetric -- too small
        and you OOM immediately, too large and the GPU just sits underused while
        throughput quietly disappoints.
        """
        geometry = kv_geometry(hf_config)
        assert isinstance(self.kv_cache_dtype, torch.dtype)  # normalized in __post_init__
        return (
            2
            * geometry.num_layers
            * self.block_size
            * geometry.num_kv_heads
            * geometry.head_dim
            * self.kv_cache_dtype.itemsize
        )

    def kv_cache_bytes_per_token(self, hf_config: Any) -> int:
        """Bytes one token of context costs. Handy for reasoning about capacity."""
        return self.kv_cache_bytes_per_block(hf_config) // self.block_size
