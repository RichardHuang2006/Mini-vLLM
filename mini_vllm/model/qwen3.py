"""The whole Qwen3 model, assembled from the primitives in `mini_vllm`.

No cache and no custom kernels: one forward pass over a whole sequence. This is
the readable reference implementation, and **it is never modified again**.
`model/qwen3_cached.py` forks it to add a KV cache, the CUDA kernels are diffed
against it, and paged attention is diffed against that. Every fast path in the
project is ultimately justified by agreeing with this file, so it stays pure
PyTorch and imports nothing from `mini_vllm.kernels`.

Two Qwen3-specific details that Qwen2.5 does not have, and that produce fluent
nonsense rather than errors when omitted:

* **QK-norm** — an RMSNorm over the head dimension of `q` and `k`, before RoPE.
* **GQA** — 8 key/value heads serving 16 query heads.
"""

from __future__ import annotations

import torch

from mini_vllm.attention import scaled_dot_product_attention_grouped
from mini_vllm.basics import linear, silu
from mini_vllm.embedding import Embedding
from mini_vllm.layer_norm import rms_norm
from mini_vllm.model.loader import ModelConfig, load_weights
from mini_vllm.positional_encoding import RoPE

__all__ = ["Qwen3", "Qwen3Attention", "Qwen3MLP", "Qwen3Block"]


class Qwen3Attention:
    """Grouped-query attention with QK-norm and RoPE.

    ::

        x, out:  B x L x E
        wq:      (H_q·D) x E      wk, wv: (H_k·D) x E      wo: E x (H_q·D)
    """

    def __init__(self, config: ModelConfig, weights: dict[str, torch.Tensor], rope: RoPE) -> None:
        self.config = config
        self.rope = rope
        self.wq = weights["attn.wq"]
        self.wk = weights["attn.wk"]
        self.wv = weights["attn.wv"]
        self.wo = weights["attn.wo"]
        self.q_norm = weights["attn.q_norm"]
        self.k_norm = weights["attn.k_norm"]

    def __call__(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | str | None = "causal",
    ) -> torch.Tensor:
        config = self.config
        batch, length, _ = x.shape

        # B x L x (H·D) -> B x L x H x D. Note the head count differs between q
        # and k/v: that asymmetry *is* GQA.
        q = linear(x, self.wq).reshape(batch, length, config.num_attention_heads, config.head_dim)
        k = linear(x, self.wk).reshape(batch, length, config.num_key_value_heads, config.head_dim)
        v = linear(x, self.wv).reshape(batch, length, config.num_key_value_heads, config.head_dim)

        # QK-norm: normalize each head vector over D, before the rotation. Order
        # matters — normalizing after RoPE would be a different function.
        q = rms_norm(q, self.q_norm, config.rms_norm_eps)
        k = rms_norm(k, self.k_norm, config.rms_norm_eps)

        q = self.rope(q, positions)
        k = self.rope(k, positions)

        # B x L x H x D -> B x H x L x D so attention reduces over L.
        attended = scaled_dot_product_attention_grouped(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mask=mask
        )

        merged = attended.transpose(1, 2).reshape(batch, length, config.q_projection_size)
        return linear(merged, self.wo)


class Qwen3MLP:
    """SwiGLU: ``down(silu(gate(x)) * up(x))``.

    Two parallel projections up to `intermediate`, gated against each other, then
    one back down. The elementwise product is what the fused SwiGLU kernel takes
    over, to avoid a second pass over the wide activation.
    """

    def __init__(self, weights: dict[str, torch.Tensor]) -> None:
        self.gate = weights["mlp.gate"]
        self.up = weights["mlp.up"]
        self.down = weights["mlp.down"]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return linear(silu(linear(x, self.gate)) * linear(x, self.up), self.down)


class Qwen3Block:
    """One pre-norm transformer block.

    ::

        h   = x + attention(rmsnorm(x))
        out = h + mlp(rmsnorm(h))

    Pre-norm rather than post-norm: the residual path from input to output is
    unnormalized, which is what keeps gradients (and, here, activations) stable
    through 28 layers.
    """

    def __init__(self, config: ModelConfig, weights: dict[str, torch.Tensor], rope: RoPE) -> None:
        self.config = config
        self.attention = Qwen3Attention(config, weights, rope)
        self.mlp = Qwen3MLP(weights)
        self.attn_norm = weights["attn_norm"]
        self.mlp_norm = weights["mlp_norm"]

    def __call__(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | str | None = "causal",
    ) -> torch.Tensor:
        eps = self.config.rms_norm_eps
        x = x + self.attention(rms_norm(x, self.attn_norm, eps), positions, mask)
        return x + self.mlp(rms_norm(x, self.mlp_norm, eps))


class Qwen3:
    """The full model: embedding, blocks, final norm, tied LM head.

    ::

        input_ids: B x L      (int64)
        logits:    B x L x V
    """

    def __init__(self, config: ModelConfig, weights: dict[str, torch.Tensor]) -> None:
        self.config = config
        self.weights = weights

        self.embedding = Embedding(config.vocab_size, config.hidden_size, weights["embedding"])
        self.final_norm = weights["final_norm"]

        device = weights["embedding"].device
        self.rope = RoPE(
            config.head_dim, config.max_position_embeddings, config.rope_theta, device=device
        )

        self.blocks = [
            Qwen3Block(config, self._layer_weights(weights, layer), self.rope)
            for layer in range(config.num_hidden_layers)
        ]

    @staticmethod
    def _layer_weights(
        weights: dict[str, torch.Tensor], layer: int
    ) -> dict[str, torch.Tensor]:
        prefix = f"layers.{layer}."
        return {
            name[len(prefix) :]: tensor
            for name, tensor in weights.items()
            if name.startswith(prefix)
        }

    @classmethod
    def from_pretrained(cls, model: str = "Qwen/Qwen3-0.6B", device: str = "cpu") -> Qwen3:
        weights, config = load_weights(model, device=device)
        return cls(config, weights)

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        mask: torch.Tensor | str | None = "causal",
    ) -> torch.Tensor:
        """Full forward over the whole sequence.

        ``positions`` defaults to ``arange(L)``, which is correct here precisely
        because there is no cache: the tokens fed in *are* the whole sequence. It
        stays an argument because that is not true of the cached and paged models,
        where the positions have to come from the caller.
        """
        _batch, length = input_ids.shape
        if positions is None:
            positions = torch.arange(length, device=input_ids.device)

        h = self.embedding(input_ids)
        for block in self.blocks:
            h = block(h, positions, mask)

        h = rms_norm(h, self.final_norm, self.config.rms_norm_eps)
        return self.embedding.as_linear(h)
