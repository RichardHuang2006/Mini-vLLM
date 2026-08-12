"""Qwen3 over a ragged batch and a paged cache.

The third and last fork of the model, and the smallest departure of the three. The
version in `model/qwen3.py` recomputed everything; the one in `model/qwen3_cached.py`
kept a dense cache per sequence; this one keeps no cache of its own at all. It is
handed a `ForwardBatch` and a `BlockManager` and writes its keys and values straight
into the pool.

Two changes from `qwen3_cached.py`, and both are about shape rather than mathematics:

* **No batch axis.** Activations are `T x ...` where `T` is every scheduled token in
  the iteration, sequences concatenated. A padded `B x L` rectangle is what paging
  exists to avoid, and reintroducing one here to keep the code familiar would give
  back the memory the paged cache exists to buy.
* **Attention takes metadata instead of tensors.** `cu_seqlens_q` says which rows
  belong to which sequence, `context_lens` how far back each may look, and the block
  tables where its pages are. One call covers a 512-token prefill chunk and eleven
  decode steps together.

Everything else — QK-norm, RoPE at explicit positions, SwiGLU, the residual stream —
is character for character what the cached model does, which is what makes the
token-identity test in `test_engine.py` a meaningful check rather than a tautology.
"""

from __future__ import annotations

import torch

from mini_vllm.basics import linear
from mini_vllm.block.block_manager import BlockManager
from mini_vllm.embedding import Embedding
from mini_vllm.kernels import ops
from mini_vllm.model.loader import ModelConfig, load_weights
from mini_vllm.positional_encoding import RoPE
from mini_vllm.serve.batch import ForwardBatch

__all__ = ["Qwen3Paged"]


class Qwen3PagedAttention:
    """Grouped-query attention against the paged pool, for one layer."""

    def __init__(
        self,
        layer: int,
        config: ModelConfig,
        weights: dict[str, torch.Tensor],
        rope: RoPE,
        use_cuda: bool = True,
    ) -> None:
        self.layer = layer
        self.config = config
        self.rope = rope
        self.use_cuda = use_cuda
        self.wq = weights["attn.wq"]
        self.wk = weights["attn.wk"]
        self.wv = weights["attn.wv"]
        self.wo = weights["attn.wo"]
        self.q_norm = weights["attn.q_norm"]
        self.k_norm = weights["attn.k_norm"]

    def __call__(self, x: torch.Tensor, batch: ForwardBatch, manager: BlockManager) -> torch.Tensor:
        config = self.config
        tokens = x.shape[0]
        use_cuda = self.use_cuda

        q = linear(x, self.wq).reshape(tokens, config.num_attention_heads, config.head_dim)
        k = linear(x, self.wk).reshape(tokens, config.num_key_value_heads, config.head_dim)
        v = linear(x, self.wv).reshape(tokens, config.num_key_value_heads, config.head_dim)

        q = ops.rmsnorm(q, self.q_norm, config.rms_norm_eps, use_cuda=use_cuda)
        k = ops.rmsnorm(k, self.k_norm, config.rms_norm_eps, use_cuda=use_cuda)

        # RoPE wants a leading axis to broadcast the position across; the positions are
        # per token and come from the batch, because a chunk's tokens sit at 512..1023
        # and nothing in these shapes says so.
        q = ops.rope(q.unsqueeze(0), batch.positions, self.rope.cos, self.rope.sin, use_cuda).squeeze(0)
        k = ops.rope(k.unsqueeze(0), batch.positions, self.rope.cos, self.rope.sin, use_cuda).squeeze(0)

        # Rotated keys go into the pool, so a cached key carries its position with it
        # and never has to be re-rotated on a later read.
        manager.kv.write(self.layer, batch.slot_mapping, k, v)

        attended = ops.paged_attention(
            q.contiguous(),
            manager.kv.layer_keys(self.layer),
            manager.kv.layer_values(self.layer),
            batch.block_tables,
            batch.cu_seqlens_q,
            batch.context_lens,
            batch.seq_lens,
            batch.max_query_len,
            batch.max_context_len,
            use_cuda=use_cuda,
        )

        return linear(attended.reshape(tokens, config.q_projection_size), self.wo)


class Qwen3PagedMLP:
    """SwiGLU, unchanged but for the missing batch axis."""

    def __init__(self, weights: dict[str, torch.Tensor], use_cuda: bool = True) -> None:
        self.use_cuda = use_cuda
        self.gate = weights["mlp.gate"]
        self.up = weights["mlp.up"]
        self.down = weights["mlp.down"]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        gated = ops.swiglu(linear(x, self.gate), linear(x, self.up), use_cuda=self.use_cuda)
        return linear(gated, self.down)


class Qwen3PagedBlock:
    """One pre-norm block."""

    def __init__(
        self,
        layer: int,
        config: ModelConfig,
        weights: dict[str, torch.Tensor],
        rope: RoPE,
        use_cuda: bool = True,
    ) -> None:
        self.config = config
        self.use_cuda = use_cuda
        self.attention = Qwen3PagedAttention(layer, config, weights, rope, use_cuda)
        self.mlp = Qwen3PagedMLP(weights, use_cuda)
        self.attn_norm = weights["attn_norm"]
        self.mlp_norm = weights["mlp_norm"]

    def __call__(self, x: torch.Tensor, batch: ForwardBatch, manager: BlockManager) -> torch.Tensor:
        eps = self.config.rms_norm_eps
        normed = ops.rmsnorm(x, self.attn_norm, eps, use_cuda=self.use_cuda)
        x = x + self.attention(normed, batch, manager)

        normed = ops.rmsnorm(x, self.mlp_norm, eps, use_cuda=self.use_cuda)
        return x + self.mlp(normed)


class Qwen3Paged:
    """Qwen3 with its keys and values in a paged pool.

    ::

        batch:  a ForwardBatch of T tokens across N sequences
        logits: N x V — one row per sequence, at its last computed position

    Only `N` rows come back, not `T`. The LM head is a `V`-wide matmul and generation
    only ever samples from a sequence's last position, so computing it for the other
    `T - N` rows of a prefill chunk is pure waste — about 20 GFLOP on a 128-token
    chunk of this model.
    """

    def __init__(
        self,
        config: ModelConfig,
        weights: dict[str, torch.Tensor],
        manager: BlockManager,
        use_cuda: bool = True,
    ) -> None:
        self.config = config
        self.weights = weights
        self.manager = manager
        self.use_cuda = use_cuda

        self.embedding = Embedding(config.vocab_size, config.hidden_size, weights["embedding"])
        self.final_norm = weights["final_norm"]

        device = weights["embedding"].device
        self.rope = RoPE(
            config.head_dim, config.max_position_embeddings, config.rope_theta, device=device
        )

        self.blocks = [
            Qwen3PagedBlock(layer, config, self._layer_weights(weights, layer), self.rope, use_cuda)
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
        cls,
        model: str = "Qwen/Qwen3-0.6B",
        num_blocks: int = 2048,
        block_size: int = 16,
        device: str = "cuda",
        use_cuda: bool = True,
    ) -> Qwen3Paged:
        """Load the weights and size a pool to match the model's KV geometry."""
        weights, config = load_weights(model, device=device)
        manager = BlockManager(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=config.dtype,
            device=device,
        )
        return cls(config, weights, manager, use_cuda=use_cuda)

    @torch.no_grad()
    def __call__(self, batch: ForwardBatch) -> torch.Tensor:
        if batch.slot_mapping is None or batch.block_tables is None:
            raise ValueError(
                "this model writes into a paged pool, so it needs a batch built with a "
                "block manager: ForwardBatch.from_scheduled(..., manager=manager)"
            )

        h = self.embedding(batch.input_ids)
        for block in self.blocks:
            h = block(h, batch, self.manager)

        # One row per sequence: the last position it computed. For a decode step that
        # is its only row; for a prefill chunk, the end of the chunk.
        h = h.index_select(0, batch.last_row_indices)

        h = ops.rmsnorm(h, self.final_norm, self.config.rms_norm_eps, use_cuda=self.use_cuda)
        return self.embedding.as_linear(h)
