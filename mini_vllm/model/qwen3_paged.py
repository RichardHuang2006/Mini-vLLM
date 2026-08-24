"""Qwen3 over a ragged batch and a paged cache.

The third fork of the model. `model/qwen3.py` recomputes everything, `model/qwen3_cached.py`
keeps a dense cache per sequence, and this one keeps no cache of its own: it is handed a
`ForwardBatch` and a `BlockManager` and writes its keys and values straight into the pool.

Two changes from `qwen3_cached.py`, both about shape rather than mathematics:

* No batch axis. Activations are `T x ...` where `T` is every scheduled token in the
  iteration, sequences concatenated. A padded `B x L` rectangle would give back the memory
  the paged cache buys.
* Attention takes metadata instead of tensors: `cu_seqlens_q` for which rows belong to
  which sequence, `context_lens` for how far back each may look, and the block tables for
  where its pages are. One call covers a 512-token prefill chunk and eleven decode steps.

Everything else — QK-norm, RoPE at explicit positions, SwiGLU, the residual stream — is
what the cached model does, which is what makes the token-identity test in
`test_engine.py` a meaningful check rather than a tautology.
"""

from __future__ import annotations

import copy

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

        # RoPE needs a leading axis to broadcast the position across. Positions are per
        # token and come from the batch, since a chunk's tokens may sit at 512..1023 and
        # nothing in these shapes records that.
        q = ops.rope(q.unsqueeze(0), batch.positions, self.rope.cos, self.rope.sin, use_cuda).squeeze(0)
        k = ops.rope(k.unsqueeze(0), batch.positions, self.rope.cos, self.rope.sin, use_cuda).squeeze(0)

        # Rotated keys go into the pool, so a cached key carries its position and is
        # never re-rotated on a later read.
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
            k_scale=manager.kv.k_scale,
            v_scale=manager.kv.v_scale,
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
    samples only from a sequence's last position, so computing it for the other `T - N`
    rows of a prefill chunk is wasted work — about 20 GFLOP on a 128-token chunk of this
    model. Speculative verification is the exception and asks for all `T` with
    `all_rows=True`, needing a distribution at every proposed position.
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

    def self_draft(self, num_layers: int, manager: BlockManager) -> Qwen3Paged:
        """This model truncated to its first `num_layers`, over a separate KV pool.

        The draft model for speculative decoding on a machine that cannot afford a second
        checkpoint. Qwen3-0.6B is already the smallest of its family, and an 8 GB card has
        no room for another set of weights plus two KV caches. Running the target's early
        layers costs no additional memory — the blocks, the embedding and the head are the
        same tensors, shared rather than copied — and a prefix of a transformer is both
        cheaper and weaker, which is what a draft needs to be.

        It needs its own `manager` because a draft keeps its own keys and values: the
        pool's layer axis is `num_layers` deep here rather than the target's, and the two
        caches are written at different times over different tokens.

        Weakness is acceptable where wrongness is not. Rejection sampling is indifferent
        to draft quality — a poor draft is rejected more often — so this choice affects
        only the acceptance rate, never the output distribution.
        """
        if not 1 <= num_layers <= len(self.blocks):
            raise ValueError(
                f"a self-draft needs between 1 and {len(self.blocks)} layers, got {num_layers}"
            )
        if manager.kv.num_layers < num_layers:
            raise ValueError(
                f"the draft's pool has {manager.kv.num_layers} layers, too few for a "
                f"{num_layers}-layer draft"
            )

        draft = copy.copy(self)
        draft.blocks = self.blocks[:num_layers]
        draft.manager = manager
        return draft

    @torch.no_grad()
    def __call__(self, batch: ForwardBatch, all_rows: bool = False) -> torch.Tensor:
        """One forward pass, returning `N x V` normally and `T x V` when `all_rows`.

        `all_rows` exists for speculative verification, the one caller that needs every
        position's distribution: it forwards a sequence's pending token followed by `k`
        proposals and needs the target's distribution at all `k + 1` positions in a single
        pass. Off by default, since paying the `V`-wide matmul for a prefill chunk's
        interior rows is exactly what the `index_select` below avoids.
        """
        if batch.slot_mapping is None or batch.block_tables is None:
            raise ValueError(
                "this model writes into a paged pool, so it needs a batch built with a "
                "block manager: ForwardBatch.from_scheduled(..., manager=manager)"
            )

        h = self.embedding(batch.input_ids)
        for block in self.blocks:
            h = block(h, batch, self.manager)

        if not all_rows:
            # One row per sequence: the last position it computed. Its only row for a
            # decode step, the end of the chunk for a prefill.
            h = h.index_select(0, batch.last_row_indices)

        h = ops.rmsnorm(h, self.final_norm, self.config.rms_norm_eps, use_cuda=self.use_cuda)
        return self.embedding.as_linear(h)
