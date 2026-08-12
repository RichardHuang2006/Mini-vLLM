"""The same model, but it stops recomputing the past.

A **fork** of `model/qwen3.py`, not a replacement. That file stays exactly as it
was and remains the oracle: every claim made here is checked by producing the same
tokens it does. Duplicating a few dozen lines is a small price for never having to
wonder whether a "shared" refactor changed the reference.

Three things change, and they are all about position bookkeeping:

* Only the **new** tokens are fed in. Their keys and values join the cache, and
  attention runs against everything cached so far — so `q` has length `L` while
  `k` and `v` have length `S`.
* RoPE positions become ``arange(offset, offset + L)`` instead of ``arange(L)``,
  which is why the rotary embedding takes its positions explicitly rather than
  deriving them from the sequence length.
* The causal mask becomes ``(L, S)``, using the offset form of the reference mask so
  a single decode token may attend to the entire cache.

That same `offset` machinery is what chunked prefill in the serving layer needs, so
chunked prefill is a scheduler change rather than a model rewrite.

Every op routes through `mini_vllm.kernels.ops`, so the CUDA kernels can be swapped
in without editing this file.
"""

from __future__ import annotations

import torch

from mini_vllm.basics import linear
from mini_vllm.embedding import Embedding
from mini_vllm.kernels import ops
from mini_vllm.kv_cache import DenseKvCache, KvCache
from mini_vllm.model.loader import ModelConfig, load_weights
from mini_vllm.positional_encoding import RoPE

__all__ = ["Qwen3Cached", "Qwen3CachedAttention", "Qwen3CachedBlock", "Qwen3CachedMLP"]


class Qwen3CachedAttention:
    """Grouped-query attention against a growing KV cache."""

    def __init__(
        self,
        config: ModelConfig,
        weights: dict[str, torch.Tensor],
        rope: RoPE,
        use_cuda: bool = False,
    ) -> None:
        self.config = config
        self.rope = rope
        self.use_cuda = use_cuda
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
        cache: KvCache,
    ) -> torch.Tensor:
        config = self.config
        batch, length, _ = x.shape
        use_cuda = self.use_cuda

        q = linear(x, self.wq).reshape(batch, length, config.num_attention_heads, config.head_dim)
        k = linear(x, self.wk).reshape(batch, length, config.num_key_value_heads, config.head_dim)
        v = linear(x, self.wv).reshape(batch, length, config.num_key_value_heads, config.head_dim)

        q = ops.rmsnorm(q, self.q_norm, config.rms_norm_eps, use_cuda=use_cuda)
        k = ops.rmsnorm(k, self.k_norm, config.rms_norm_eps, use_cuda=use_cuda)

        # RoPE is applied *before* the cache, so cached keys carry their position
        # with them. A cache of unrotated keys would have to be re-rotated on every
        # read, and the whole saving would evaporate.
        q = ops.rope(q, positions, self.rope.cos, self.rope.sin, use_cuda=use_cuda)
        k = ops.rope(k, positions, self.rope.cos, self.rope.sin, use_cuda=use_cuda)

        # B x L x H x D -> B x H x L x D, which is also the cache's layout.
        keys, values, _offset = cache.update_and_fetch(k.transpose(1, 2), v.transpose(1, 2))

        # A single decode token may attend to everything cached, so its mask is
        # all-zeros and worth skipping entirely — that is the common case by far.
        #
        # For prefill, the shorthand rather than the tensor: the mask is a pure
        # function of `(L, S)`, both of which the callee already knows, and naming
        # it lets the flash prefill kernel apply it as an index comparison instead of
        # reading back an `L x S` tensor. The oracle builds exactly the same tensor
        # from the same shorthand, so nothing about the reference path changes.
        mask = None if length == 1 else "causal"

        attended = ops.attention(q.transpose(1, 2), keys, values, mask=mask, use_cuda=use_cuda)

        merged = attended.transpose(1, 2).reshape(batch, length, config.q_projection_size)
        return linear(merged, self.wo)


class Qwen3CachedMLP:
    """SwiGLU, with the elementwise part routed through `ops.swiglu`."""

    def __init__(self, weights: dict[str, torch.Tensor], use_cuda: bool = False) -> None:
        self.use_cuda = use_cuda
        self.gate = weights["mlp.gate"]
        self.up = weights["mlp.up"]
        self.down = weights["mlp.down"]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        gated = ops.swiglu(linear(x, self.gate), linear(x, self.up), use_cuda=self.use_cuda)
        return linear(gated, self.down)


class Qwen3CachedBlock:
    """One pre-norm block, carrying its layer's cache."""

    def __init__(
        self,
        config: ModelConfig,
        weights: dict[str, torch.Tensor],
        rope: RoPE,
        use_cuda: bool = False,
    ) -> None:
        self.config = config
        self.use_cuda = use_cuda
        self.attention = Qwen3CachedAttention(config, weights, rope, use_cuda)
        self.mlp = Qwen3CachedMLP(weights, use_cuda)
        self.attn_norm = weights["attn_norm"]
        self.mlp_norm = weights["mlp_norm"]

    def __call__(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cache: KvCache,
    ) -> torch.Tensor:
        eps = self.config.rms_norm_eps
        normed = ops.rmsnorm(x, self.attn_norm, eps, use_cuda=self.use_cuda)
        x = x + self.attention(normed, positions, cache)

        normed = ops.rmsnorm(x, self.mlp_norm, eps, use_cuda=self.use_cuda)
        return x + self.mlp(normed)


class Qwen3Cached:
    """Qwen3 with a per-layer KV cache.

    ::

        input_ids: B x L      (the *new* tokens only)
        logits:    B x L x V  (or B x 1 x V with last_only)
    """

    def __init__(
        self,
        config: ModelConfig,
        weights: dict[str, torch.Tensor],
        use_cuda: bool = False,
    ) -> None:
        self.config = config
        self.weights = weights
        self.use_cuda = use_cuda

        self.embedding = Embedding(config.vocab_size, config.hidden_size, weights["embedding"])
        self.final_norm = weights["final_norm"]

        device = weights["embedding"].device
        self.rope = RoPE(
            config.head_dim, config.max_position_embeddings, config.rope_theta, device=device
        )

        self.blocks = [
            Qwen3CachedBlock(config, self._layer_weights(weights, layer), self.rope, use_cuda)
            for layer in range(config.num_hidden_layers)
        ]

    @staticmethod
    def _layer_weights(weights: dict[str, torch.Tensor], layer: int) -> dict[str, torch.Tensor]:
        prefix = f"layers.{layer}."
        return {
            name[len(prefix) :]: tensor
            for name, tensor in weights.items()
            if name.startswith(prefix)
        }

    @classmethod
    def from_pretrained(
        cls, model: str = "Qwen/Qwen3-0.6B", device: str = "cpu", use_cuda: bool = False
    ) -> Qwen3Cached:
        weights, config = load_weights(model, device=device)
        return cls(config, weights, use_cuda=use_cuda)

    def create_kv_cache(self) -> list[KvCache]:
        """One cache per layer. Each sequence (or batch) needs its own set."""
        return [DenseKvCache() for _ in range(self.config.num_hidden_layers)]

    def __call__(
        self,
        input_ids: torch.Tensor,
        caches: list[KvCache],
        positions: torch.Tensor | None = None,
        last_only: bool = False,
    ) -> torch.Tensor:
        """Forward the new tokens, extending ``caches`` in place.

        ``positions`` defaults to ``arange(offset, offset + L)``, read from the
        caches — which is the whole trick, and why the caller does not have to
        track position separately.

        ``last_only`` skips the LM head on every position but the last. Generation
        only ever looks at the last one, and the head is a `V`-wide matmul, so on a
        128-token prefill this is about 20 GFLOP of pure waste. It defaults to
        False so the output stays directly comparable to `model/qwen3.py`.
        """
        if len(caches) != len(self.blocks):
            raise ValueError(
                f"expected {len(self.blocks)} caches, one per layer, got {len(caches)}"
            )

        _batch, length = input_ids.shape
        offset = caches[0].offset
        if positions is None:
            positions = torch.arange(offset, offset + length, device=input_ids.device)

        h = self.embedding(input_ids)
        for block, cache in zip(self.blocks, caches, strict=True):
            h = block(h, positions, cache)

        if last_only:
            h = h[:, -1:, :]

        h = ops.rmsnorm(h, self.final_norm, self.config.rms_norm_eps, use_cuda=self.use_cuda)
        return self.embedding.as_linear(h)
