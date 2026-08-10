"""Step 1.7 — read a Qwen3 checkpoint and rename its weights to ours.

The interesting part of this file is not the file I/O, it is
:func:`load_weights` asserting the name mapping is **total in both directions**:
every tensor in the checkpoint is consumed, and every tensor the model needs is
produced.

That matters more than it sounds. A weight left unmapped is not a crash — the
model runs, and produces fluent, confident, wrong text, because one projection is
quietly still at its random initialization. It is the worst class of bug in the
project: no stack trace, no obviously broken output, and it looks like a subtle
numerical problem for as long as you are willing to believe that. So the mapping
is checked rather than trusted, and the shapes are checked against the config too.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

__all__ = ["ModelConfig", "load_weights", "resolve_model_path", "map_name", "expected_names"]

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"


# --------------------------------------------------------------------- config


def _rope_theta(raw: dict) -> float:
    """Read the RoPE base, wherever this config generation happens to keep it.

    Older configs put `rope_theta` at the top level; transformers 5.x nests it
    under `rope_parameters` (and, for a while, `rope_scaling`). All three appear
    in checkpoints in the wild, and the value is load-bearing — a wrong base
    silently changes every position encoding.
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
    """The subset of ``config.json`` this engine actually uses (DESIGN.md §3)."""

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
        """``G = H_q / H_k``: how many query heads share each KV head."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def q_projection_size(self) -> int:
        """``H_q · D``, which is *not* ``E`` — it is twice it in Qwen3-0.6B."""
        return self.num_attention_heads * self.head_dim

    @property
    def kv_projection_size(self) -> int:
        """``H_k · D``."""
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
        # head_dim is explicit in Qwen3, but fall back to the usual assumption so
        # this also reads configs that omit it.
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


# -------------------------------------------------------------- name mapping

# Ours on the right. The names are shortened where HF is verbose (`wq` rather
# than `self_attn.q_proj.weight`) but the structure is deliberately unchanged, so
# a checkpoint key and our key remain recognisably the same thing.
GLOBAL_NAMES: dict[str, str] = {
    "model.embed_tokens.weight": "embedding",
    "model.norm.weight": "final_norm",
}

LAYER_NAMES: dict[str, str] = {
    "input_layernorm.weight": "attn_norm",
    "self_attn.q_proj.weight": "attn.wq",
    "self_attn.k_proj.weight": "attn.wk",
    "self_attn.v_proj.weight": "attn.wv",
    "self_attn.o_proj.weight": "attn.wo",
    "self_attn.q_norm.weight": "attn.q_norm",
    "self_attn.k_norm.weight": "attn.k_norm",
    "post_attention_layernorm.weight": "mlp_norm",
    "mlp.gate_proj.weight": "mlp.gate",
    "mlp.up_proj.weight": "mlp.up",
    "mlp.down_proj.weight": "mlp.down",
}

# Qwen3-0.6B ships `lm_head.weight` even though `tie_word_embeddings` is true,
# and it is bitwise identical to the embedding. It is therefore deliberately
# dropped rather than mapped: the model reads logits off the embedding via
# `Embedding.as_linear`, and keeping a second 155M-parameter copy would waste
# 300 MB of an 8 GB card to hold the same numbers twice.
TIED_LM_HEAD = "lm_head.weight"


def map_name(hf_name: str) -> str | None:
    """Translate one checkpoint key to ours, or None if it is deliberately dropped."""
    if hf_name == TIED_LM_HEAD:
        return None
    if hf_name in GLOBAL_NAMES:
        return GLOBAL_NAMES[hf_name]

    prefix = "model.layers."
    if hf_name.startswith(prefix):
        index, _, suffix = hf_name[len(prefix) :].partition(".")
        if suffix in LAYER_NAMES:
            return f"layers.{index}.{LAYER_NAMES[suffix]}"

    raise KeyError(f"unmapped checkpoint weight: {hf_name}")


def expected_names(config: ModelConfig) -> set[str]:
    """Every weight the model needs in order to be fully initialized."""
    names = set(GLOBAL_NAMES.values())
    for layer in range(config.num_hidden_layers):
        names.update(f"layers.{layer}.{ours}" for ours in LAYER_NAMES.values())
    return names


def expected_shape(name: str, config: ModelConfig) -> tuple[int, ...]:
    """The shape a given weight must have, derived from the config.

    Checked on load so a config that disagrees with the checkpoint fails here
    rather than as a confusing matmul error deep in the forward pass.
    """
    leaf = name.split(".")[-1]
    shapes: dict[str, tuple[int, ...]] = {
        "embedding": (config.vocab_size, config.hidden_size),
        "final_norm": (config.hidden_size,),
        "attn_norm": (config.hidden_size,),
        "mlp_norm": (config.hidden_size,),
        "wq": (config.q_projection_size, config.hidden_size),
        "wk": (config.kv_projection_size, config.hidden_size),
        "wv": (config.kv_projection_size, config.hidden_size),
        "wo": (config.hidden_size, config.q_projection_size),
        "q_norm": (config.head_dim,),
        "k_norm": (config.head_dim,),
        "gate": (config.intermediate_size, config.hidden_size),
        "up": (config.intermediate_size, config.hidden_size),
        "down": (config.hidden_size, config.intermediate_size),
    }
    return shapes[leaf]


# ------------------------------------------------------------------- file I/O


def resolve_model_path(model: str | Path = DEFAULT_MODEL_ID) -> Path:
    """A local directory, or a HuggingFace id downloaded to the hub cache."""
    path = Path(model)
    if path.is_dir():
        return path

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(model)))


def shard_files(model_path: Path) -> list[Path]:
    """The safetensors shards, in index order when the checkpoint is sharded."""
    index = model_path / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        return [model_path / name for name in sorted(set(weight_map.values()))]

    single = model_path / "model.safetensors"
    if single.is_file():
        return [single]

    raise FileNotFoundError(f"no safetensors checkpoint under {model_path}")


def iter_weights(model_path: Path, device: str = "cpu") -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(checkpoint_name, tensor)`` pairs, one shard at a time.

    safetensors memory-maps the file, so tensors are paged in as they are read
    rather than the whole 1.2 GB being materialized at once.
    """
    for shard in shard_files(model_path):
        with safe_open(shard, framework="pt", device=device) as handle:
            for name in handle.keys():
                yield name, handle.get_tensor(name)


def load_weights(
    model: str | Path = DEFAULT_MODEL_ID,
    config: ModelConfig | None = None,
    device: str = "cpu",
) -> tuple[dict[str, torch.Tensor], ModelConfig]:
    """Load a checkpoint into our naming scheme, verifying the mapping is total.

    Returns the weights keyed by our names, plus the config they were checked
    against.
    """
    model_path = resolve_model_path(model)
    if config is None:
        config = ModelConfig.from_pretrained(model_path)

    weights: dict[str, torch.Tensor] = {}
    for hf_name, tensor in iter_weights(model_path, device=device):
        ours = map_name(hf_name)  # raises on anything unrecognized
        if ours is None:
            continue

        wanted = expected_shape(ours, config)
        if tuple(tensor.shape) != wanted:
            raise ValueError(
                f"{hf_name} -> {ours}: expected shape {wanted}, got {tuple(tensor.shape)}"
            )
        if ours in weights:
            raise ValueError(f"two checkpoint weights map to {ours}")
        weights[ours] = tensor

    missing = expected_names(config) - weights.keys()
    if missing:
        raise ValueError(
            f"{len(missing)} model weights were never filled, e.g. {sorted(missing)[:5]}. "
            "An unfilled weight produces plausible garbage rather than an error, "
            "so this is fatal."
        )

    extra = weights.keys() - expected_names(config)
    if extra:
        raise ValueError(f"loaded weights the model does not use: {sorted(extra)}")

    return weights, config
