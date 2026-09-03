"""Every knob in the engine, in one file.

Four surfaces: ModelConfig (architecture, parsed from a checkpoint),
SchedulerConfig (per-iteration budgets), SamplingParams (per-request decoding),
and EngineConfig (memory, features, wiring). EngineConfig is the single source
of truth for defaults, and a wrong ModelConfig fails at load time rather than
as a matmul error 28 layers deep.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

__all__ = [
    "DEFAULT_MODEL_ID",
    "ModelConfig",
    "SchedulerConfig",
    "SamplingParams",
    "EngineConfig",
    "resolve_kv_dtype",
]

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"

# Fraction of the memory still free after the weights are resident that goes to the KV
# pool. Not 0.9: activations for a 2048-token batch, the logits tensor at 151k
# vocabulary, and cuBLAS workspaces all come out of the remainder, and a pool sized to
# the last byte turns a long prompt into an out-of-memory error rather than a queued
# request.
DEFAULT_KV_FRACTION = 0.5

# Blocks to allocate when there is no device memory to measure. Enough for a handful of
# short sequences, which is all a CPU run is ever going to want.
CPU_BLOCKS = 512

# KV-cache storage precision as a caller names it. "auto" keeps the model dtype; "fp8"
# halves the resident cache at the cost of a rounding on every stored key and value,
# dequantized inside the attention kernel. e4m3 is the only format offered because it
# is the only one the kernels accelerate; see `kernels.FP8_KERNEL_DTYPE`.
KV_CACHE_DTYPES: dict[str, torch.dtype | None] = {
    "auto": None,
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
}


def resolve_kv_dtype(name: str) -> torch.dtype | None:
    """Turn a kv_cache_dtype name into a storage dtype, or None for the model's."""
    if name not in KV_CACHE_DTYPES:
        raise ValueError(
            f"unknown kv_cache_dtype {name!r}; expected one of {sorted(KV_CACHE_DTYPES)}"
        )
    return KV_CACHE_DTYPES[name]


# --- 1. Model config ---------------------------------------------------------

def _rope_theta(raw: dict) -> float:
    """Read the RoPE base, wherever this config generation happens to keep it.

    Older configs put it at the top level, transformers 5.x nests it under
    rope_parameters, and for a time under rope_scaling. All three appear in the
    wild, and a wrong base silently changes every position encoding.
    """
    if raw.get("rope_theta") is not None:
        return float(raw["rope_theta"])

    for key in ("rope_parameters", "rope_scaling"):
        block = raw.get(key)
        if isinstance(block, dict) and block.get("rope_theta") is not None:
            return float(block["rope_theta"])

    raise KeyError("config specifies no rope_theta under any known key")


@dataclass(frozen=True)
class ModelConfig:
    """The subset of config.json this engine actually uses."""

    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    max_position_embeddings: int
    dtype: torch.dtype = torch.bfloat16

    @property
    def group_size(self) -> int:
        """G = H_q / H_k: how many query heads share each KV head."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def q_projection_size(self) -> int:
        """H_q * D, which is not E: it is twice E in Qwen3-0.6B."""
        return self.num_attention_heads * self.head_dim

    @property
    def kv_projection_size(self) -> int:
        """H_k * D."""
        return self.num_key_value_heads * self.head_dim

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"H_q ({self.num_attention_heads}) must be a multiple of "
                f"H_k ({self.num_key_value_heads})"
            )

    @classmethod
    def from_dict(cls, raw: dict) -> ModelConfig:
        hidden_size = raw["hidden_size"]
        num_heads = raw["num_attention_heads"]
        # head_dim is explicit in Qwen3; fall back to the usual assumption so configs
        # that omit it still load.
        head_dim = raw.get("head_dim") or hidden_size // num_heads

        stated_dtype = raw.get("torch_dtype") or raw.get("dtype") or "bfloat16"

        return cls(
            num_hidden_layers=raw["num_hidden_layers"],
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=head_dim,
            intermediate_size=raw["intermediate_size"],
            vocab_size=raw["vocab_size"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=_rope_theta(raw),
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            max_position_embeddings=raw["max_position_embeddings"],
            dtype=getattr(torch, stated_dtype) if isinstance(stated_dtype, str) else stated_dtype,
        )

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> ModelConfig:
        path = Path(model_path) / "config.json"
        return cls.from_dict(json.loads(path.read_text()))


# --- 2. Scheduler config -----------------------------------------------------

@dataclass(frozen=True)
class SchedulerConfig:
    """Admission limits for one iteration."""

    max_batched_tokens: int = 2048     # compute budget: token-positions per pass
    max_sequences: int = 16            # how many requests may be in flight
    chunk_size: int = 512              # one prefill's share of an iteration
    enable_chunked_prefill: bool = True  # off = the tests' single-pass reference
    # Prompts before decodes: what vLLM shipped before chunked prefill, and the
    # baseline the benchmarks measure against. Off by default -- a prompt large
    # enough to consume the budget leaves nothing for sequences a caller is already
    # reading. Kept reachable so the cost is measurable.
    prefill_priority: bool = False

    def __post_init__(self) -> None:
        if self.max_batched_tokens < 1:
            raise ValueError(f"max_batched_tokens must be >= 1, got {self.max_batched_tokens}")
        if self.max_sequences < 1:
            raise ValueError(f"max_sequences must be >= 1, got {self.max_sequences}")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")


# --- 3. Sampling params ------------------------------------------------------

@dataclass(frozen=True)
class SamplingParams:
    """One request's sampling configuration. temperature=0 is greedy; top_k=0 and
    top_p=1.0 both mean disabled, so the default is plain temperature-1 sampling.
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    # Completions per prompt. n > 1 prefills once and forks the block table, so the
    # branches share KV until one writes -- copy-on-write on the serving path.
    n: int = 1

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0


# --- 4. Engine config --------------------------------------------------------

@dataclass(frozen=True)
class EngineConfig:
    """Everything LLM accepts, as one frozen record.

    LLM(model, **overrides) builds one internally, so keyword arguments and an
    explicit EngineConfig are the same thing; the dataclass exists so the defaults
    live in one place and a configuration can be logged, compared, or replayed.
    """

    model: str = DEFAULT_MODEL_ID
    device: str = "cuda"
    dtype: torch.dtype | None = None

    # KV memory.
    num_blocks: int | None = None  # None sizes the pool to kv_fraction of free VRAM
    block_size: int = 16           # tokens per KV page
    kv_cache_dtype: str = "auto"   # "fp8" halves the cost of a page
    kv_fraction: float = DEFAULT_KV_FRACTION

    # Scheduling.
    max_batched_tokens: int = 2048
    max_sequences: int = 32
    chunk_size: int = 512
    enable_chunked_prefill: bool = True
    prefill_priority: bool = False

    # Features.
    enable_prefix_caching: bool = False
    num_speculative_tokens: int = 0
    draft_model: str | None = None
    num_draft_layers: int | None = None
    draft_blocks: int | None = None
    use_cuda_kernels: bool = True

    # Reproducibility. None leaves sampling on the global RNG; greedy is
    # deterministic either way.
    seed: int | None = None

    def __post_init__(self) -> None:
        resolve_kv_dtype(self.kv_cache_dtype)  # raises on an unknown name
        if self.block_size < 1 or self.block_size & (self.block_size - 1):
            raise ValueError(f"block_size must be a positive power of two, got {self.block_size}")
        if not 0.0 < self.kv_fraction <= 1.0:
            raise ValueError(f"kv_fraction must be in (0, 1], got {self.kv_fraction}")
        if self.num_speculative_tokens < 0:
            raise ValueError(
                f"num_speculative_tokens must be >= 0, got {self.num_speculative_tokens}"
            )

    @property
    def kv_dtype(self) -> torch.dtype | None:
        """The KV storage dtype, or None to store in the model's dtype."""
        return resolve_kv_dtype(self.kv_cache_dtype)

    def scheduler_config(self) -> SchedulerConfig:
        """The scheduler's slice of this configuration."""
        return SchedulerConfig(
            max_batched_tokens=self.max_batched_tokens,
            max_sequences=self.max_sequences,
            chunk_size=self.chunk_size,
            enable_chunked_prefill=self.enable_chunked_prefill,
            prefill_priority=self.prefill_priority,
        )
