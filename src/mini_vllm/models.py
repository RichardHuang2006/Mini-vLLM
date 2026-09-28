"""Loading: mlx_lm reads the checkpoint, then its weights are moved onto our own Qwen3 classes."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx_lm

from mini_vllm.embedding import Embedding
from mini_vllm.layer_norm import RMSNorm
from mini_vllm.positional_encoding import RoPE
from mini_vllm.qwen3 import ModelConfig, Qwen3MLP, Qwen3Model, Qwen3MultiHeadAttention, Qwen3TransformerBlock

__all__ = ["DEFAULT_MODEL", "from_mlx", "load"]

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def from_mlx(mlx_model: Any, use_metal: bool = False) -> Qwen3Model:
    """Read every weight off an mlx_lm Qwen3 module tree; nothing of mlx_lm runs after this.
    use_metal routes every operator with a kernel through src/extensions/."""
    config = ModelConfig.from_mlx_args(mlx_model.args)

    def norm(dim: int, weight: mx.array) -> RMSNorm:
        return RMSNorm(dim, weight, config.rms_norm_eps, use_metal)

    # One set of rotary tables, shared by every layer.
    rope = RoPE(config.head_dim, config.max_position_embeddings, config.rope_theta, use_metal)

    layers = []
    for layer in mlx_model.model.layers:
        attn, mlp = layer.self_attn, layer.mlp
        attention = Qwen3MultiHeadAttention(
            config,
            attn.q_proj.weight,
            attn.k_proj.weight,
            attn.v_proj.weight,
            attn.o_proj.weight,
            norm(config.head_dim, attn.q_norm.weight),
            norm(config.head_dim, attn.k_norm.weight),
            rope,
            use_metal,
        )
        layers.append(
            Qwen3TransformerBlock(
                attention,
                Qwen3MLP(mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight, use_metal),
                norm(config.hidden_size, layer.input_layernorm.weight),
                norm(config.hidden_size, layer.post_attention_layernorm.weight),
            )
        )

    return Qwen3Model(
        config,
        Embedding(config.vocab_size, config.hidden_size, mlx_model.model.embed_tokens.weight),
        layers,
        norm(config.hidden_size, mlx_model.model.norm.weight),
        None if config.tie_word_embeddings else mlx_model.lm_head.weight,
    )


def load(model: str = DEFAULT_MODEL, use_metal: bool = False) -> tuple[Qwen3Model, Any]:
    """Load a BF16 Qwen3 checkpoint from the Hugging Face hub: (model, tokenizer)."""
    mlx_model, tokenizer = mlx_lm.load(model)
    return from_mlx(mlx_model, use_metal), tokenizer
