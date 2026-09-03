"""Qwen3, three ways: the dense oracle, the cached fork, and the paged fork.

The weight loader, then three variants of one model -- Qwen3 recomputes
everything, Qwen3Cached keeps a dense KV cache, and Qwen3Paged writes into
cache.py's pool over a ragged token axis. Each is diffed against the last,
greedy-token-identical, so every fast path is ultimately justified by agreeing
with HuggingFace.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors import safe_open

from mini_vllm import kernels
from mini_vllm.cache import BlockManager, DenseKvCache, KvCache
from mini_vllm.config import DEFAULT_MODEL_ID, ModelConfig
from mini_vllm.ops import (
    Embedding,
    RoPE,
    linear,
    rms_norm,
    scaled_dot_product_attention_grouped,
    silu,
)
from mini_vllm.scheduler import ForwardBatch

__all__ = [
    "ModelConfig",
    "load_weights",
    "resolve_model_path",
    "map_name",
    "expected_names",
    "expected_shape",
    "Qwen3",
    "Qwen3Cached",
    "Qwen3Paged",
]


# --- 1. The loader -----------------------------------------------------------

#
# The load-bearing part is `load_weights` asserting the name mapping is total in both
# directions: every tensor in the checkpoint is consumed, and every tensor the model
# needs is produced.
#
# An unmapped weight does not crash. The model runs and produces fluent, confident,
# wrong text because one projection is still at its random initialization — no stack
# trace, no obviously broken output, and it presents as a subtle numerical problem. The
# mapping is therefore checked rather than trusted, along with every shape against the
# config.

# Local names on the right, shortened where HF is verbose (`wq` rather than
# `self_attn.q_proj.weight`) but structurally unchanged, so a checkpoint key and a local
# key remain recognisably the same weight.
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

# Qwen3-0.6B ships `lm_head.weight` even though `tie_word_embeddings` is true, and it is
# bitwise identical to the embedding. Dropped rather than mapped: the model reads logits
# off the embedding via `Embedding.as_linear`, and a second 155M-parameter copy would
# cost 300 MB of an 8 GB card for the same numbers.
TIED_LM_HEAD = "lm_head.weight"


def map_name(hf_name: str) -> str | None:
    """Translate one checkpoint key to the local name, or None if it is dropped."""
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
    """The shape a given weight must have, derived from the config. Checked on load so
    a config that disagrees with the checkpoint fails here, not as a matmul error
    deep in the forward pass.
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
    """Yield (checkpoint_name, tensor) pairs, one shard at a time. safetensors
    memory-maps the file, so tensors page in as they are read rather than
    materializing the whole 1.2 GB at once.
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
    """Load a checkpoint into the local naming scheme, verifying the mapping is total.
    Returns the weights keyed by local names, plus the config they were checked
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


# --- 2. Shared pieces --------------------------------------------------------

def _layer_weights(weights: dict[str, torch.Tensor], layer: int) -> dict[str, torch.Tensor]:
    """One layer's weights, with the ``layers.N.`` prefix stripped."""
    prefix = f"layers.{layer}."
    return {
        name[len(prefix) :]: tensor for name, tensor in weights.items() if name.startswith(prefix)
    }


def _model_common(
    config: ModelConfig, weights: dict[str, torch.Tensor]
) -> tuple[Embedding, torch.Tensor, RoPE]:
    """The pieces every variant builds identically: embedding, final norm, RoPE tables."""
    embedding = Embedding(config.vocab_size, config.hidden_size, weights["embedding"])
    final_norm = weights["final_norm"]
    rope = RoPE(
        config.head_dim,
        config.max_position_embeddings,
        config.rope_theta,
        device=weights["embedding"].device,
    )
    return embedding, final_norm, rope


class _AttentionWeights:
    """The six attention tensors, bound identically by all three variants:
    wq (H_q*D x E), wk and wv (H_k*D x E), wo (E x H_q*D), and the QK-norm scales
    q_norm and k_norm (D each, one per head element).
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


class Qwen3MLP:
    """SwiGLU: down(silu(gate(x)) * up(x)), shared by all three variants. The
    projections are ordinary matmuls cuBLAS handles well; the elementwise product is
    what the fused kernel replaces, to avoid a second pass over the wide activation.
    """

    def __init__(self, weights: dict[str, torch.Tensor], use_cuda: bool = False) -> None:
        self.use_cuda = use_cuda
        self.gate = weights["mlp.gate"]
        self.up = weights["mlp.up"]
        self.down = weights["mlp.down"]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        gated = kernels.swiglu(linear(x, self.gate), linear(x, self.up), use_cuda=self.use_cuda)
        return linear(gated, self.down)


# --- 3. The dense oracle -----------------------------------------------------

class Qwen3Attention(_AttentionWeights):
    """Grouped-query attention with QK-norm and RoPE, x and out both B x L x E. No
    cache, pure ops.py.
    """

    def __call__(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | str | None = "causal",
    ) -> torch.Tensor:
        config = self.config
        batch, length, _ = x.shape

        # B x L x (H·D) -> B x L x H x D. The head count differs between q and k/v:
        # that asymmetry is GQA.
        q = linear(x, self.wq).reshape(batch, length, config.num_attention_heads, config.head_dim)
        k = linear(x, self.wk).reshape(batch, length, config.num_key_value_heads, config.head_dim)
        v = linear(x, self.wv).reshape(batch, length, config.num_key_value_heads, config.head_dim)

        # QK-norm: normalize each head vector over D, before the rotation. Normalizing
        # after RoPE would be a different function.
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


class Qwen3Block:
    """One pre-norm transformer block of the dense oracle:

        h   = x + attention(rmsnorm(x))
        out = h + mlp(rmsnorm(h))

    Pre-norm leaves the residual path unnormalized, which keeps activations stable
    through 28 layers.
    """

    def __init__(self, config: ModelConfig, weights: dict[str, torch.Tensor], rope: RoPE) -> None:
        self.config = config
        self.attention = Qwen3Attention(config, weights, rope)
        self.mlp = Qwen3MLP(weights, use_cuda=False)
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
    """The full dense model: embedding, blocks, final norm, tied LM head.
    input_ids [B, L] -> logits [B, L, V].

    No cache and no custom kernels -- one pass over a whole sequence, recomputing
    the entire prefix every step. Qwen3Cached is diffed against this and Qwen3Paged
    against that, so every fast path is ultimately justified by agreeing with this
    class, itself diffed against HuggingFace.
    """

    def __init__(self, config: ModelConfig, weights: dict[str, torch.Tensor]) -> None:
        self.config = config
        self.weights = weights
        self.embedding, self.final_norm, self.rope = _model_common(config, weights)
        self.blocks = [
            Qwen3Block(config, _layer_weights(weights, layer), self.rope)
            for layer in range(config.num_hidden_layers)
        ]

    @classmethod
    def from_pretrained(cls, model: str = DEFAULT_MODEL_ID, device: str = "cpu") -> Qwen3:
        weights, config = load_weights(model, device=device)
        return cls(config, weights)

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        mask: torch.Tensor | str | None = "causal",
    ) -> torch.Tensor:
        """Full forward over the whole sequence. positions defaults to arange(L), correct
        here because there is no cache, but it stays an argument because that does not
        hold for the cached and paged models.
        """
        _batch, length = input_ids.shape
        if positions is None:
            positions = torch.arange(length, device=input_ids.device)

        h = self.embedding(input_ids)
        for block in self.blocks:
            h = block(h, positions, mask)

        h = rms_norm(h, self.final_norm, self.config.rms_norm_eps)
        return self.embedding.as_linear(h)


# --- 4. The dense-cache fork -------------------------------------------------

class Qwen3CachedAttention(_AttentionWeights):
    """Grouped-query attention against a growing KV cache. Rotated keys and values go
    through cache.update_and_fetch and attention runs over everything accumulated,
    so q has length L while k and v have length S.
    """

    def __init__(
        self,
        config: ModelConfig,
        weights: dict[str, torch.Tensor],
        rope: RoPE,
        use_cuda: bool = False,
    ) -> None:
        super().__init__(config, weights, rope)
        self.use_cuda = use_cuda

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

        q = kernels.rmsnorm(q, self.q_norm, config.rms_norm_eps, use_cuda=use_cuda)
        k = kernels.rmsnorm(k, self.k_norm, config.rms_norm_eps, use_cuda=use_cuda)

        # RoPE is applied before the cache, so cached keys carry their position with
        # them. Caching unrotated keys would require re-rotating on every read, which
        # removes the saving.
        q = kernels.rope(q, positions, self.rope.cos, self.rope.sin, use_cuda=use_cuda)
        k = kernels.rope(k, positions, self.rope.cos, self.rope.sin, use_cuda=use_cuda)

        # B x L x H x D -> B x H x L x D, which is also the cache's layout.
        keys, values, _offset = cache.update_and_fetch(k.transpose(1, 2), v.transpose(1, 2))

        # A single decode token may attend to everything cached, so its mask is
        # all-zeros and skipped outright: the common case.
        #
        # Prefill passes the shorthand rather than a tensor. The mask is a pure function
        # of `(L, S)`, both known to the callee, and naming it lets the flash prefill
        # kernel apply it as an index comparison instead of reading an `L x S` tensor.
        # The oracle builds the same tensor from the same shorthand.
        mask = None if length == 1 else "causal"

        attended = kernels.attention(q.transpose(1, 2), keys, values, mask=mask, use_cuda=use_cuda)

        merged = attended.transpose(1, 2).reshape(batch, length, config.q_projection_size)
        return linear(merged, self.wo)


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
        self.mlp = Qwen3MLP(weights, use_cuda)
        self.attn_norm = weights["attn_norm"]
        self.mlp_norm = weights["mlp_norm"]

    def __call__(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cache: KvCache,
    ) -> torch.Tensor:
        eps = self.config.rms_norm_eps
        normed = kernels.rmsnorm(x, self.attn_norm, eps, use_cuda=self.use_cuda)
        x = x + self.attention(normed, positions, cache)

        normed = kernels.rmsnorm(x, self.mlp_norm, eps, use_cuda=self.use_cuda)
        return x + self.mlp(normed)


class Qwen3Cached:
    """Qwen3 with a per-layer dense KV cache: it stops recomputing the past. input_ids
    [B, L] is the new tokens only; logits [B, L, V], or [B, 1, V] with last_only.

    A fork of Qwen3 rather than a replacement -- that class stays the oracle. Three
    changes, all position bookkeeping: only new tokens are fed in, RoPE positions
    become arange(offset, offset + L), and the mask becomes (L, S) so a decode token
    may attend to the whole cache. That offset machinery is what chunked prefill
    needs, which is why chunking is a scheduler change rather than a model rewrite.
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
        self.embedding, self.final_norm, self.rope = _model_common(config, weights)
        self.blocks = [
            Qwen3CachedBlock(config, _layer_weights(weights, layer), self.rope, use_cuda)
            for layer in range(config.num_hidden_layers)
        ]

    @classmethod
    def from_pretrained(
        cls, model: str = DEFAULT_MODEL_ID, device: str = "cpu", use_cuda: bool = False
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
        """Forward the new tokens, extending caches in place. positions defaults to
        arange(offset, offset + L) read from the caches, so the caller tracks nothing.
        last_only skips the LM head on every position but the last, saving about
        20 GFLOP on a 128-token prefill; it defaults to False so the output stays
        directly comparable to Qwen3.
        """
        if len(caches) != len(self.blocks):
            raise ValueError(f"expected {len(self.blocks)} caches, one per layer, got {len(caches)}")

        _batch, length = input_ids.shape
        offset = caches[0].offset
        if positions is None:
            positions = torch.arange(offset, offset + length, device=input_ids.device)

        h = self.embedding(input_ids)
        for block, cache in zip(self.blocks, caches, strict=True):
            h = block(h, positions, cache)

        if last_only:
            h = h[:, -1:, :]

        h = kernels.rmsnorm(h, self.final_norm, self.config.rms_norm_eps, use_cuda=self.use_cuda)
        return self.embedding.as_linear(h)


# --- 5. The paged-pool fork --------------------------------------------------

class Qwen3PagedAttention(_AttentionWeights):
    """Grouped-query attention against the paged pool, for one layer. Rotated keys and
    values are scattered into the pool through the batch's slot mapping, and
    attention gathers them back through the block table inside the kernel.
    """

    def __init__(
        self,
        layer: int,
        config: ModelConfig,
        weights: dict[str, torch.Tensor],
        rope: RoPE,
        use_cuda: bool = True,
    ) -> None:
        super().__init__(config, weights, rope)
        self.layer = layer
        self.use_cuda = use_cuda

    def __call__(self, x: torch.Tensor, batch: ForwardBatch, manager: BlockManager) -> torch.Tensor:
        config = self.config
        tokens = x.shape[0]
        use_cuda = self.use_cuda

        q = linear(x, self.wq).reshape(tokens, config.num_attention_heads, config.head_dim)
        k = linear(x, self.wk).reshape(tokens, config.num_key_value_heads, config.head_dim)
        v = linear(x, self.wv).reshape(tokens, config.num_key_value_heads, config.head_dim)

        q = kernels.rmsnorm(q, self.q_norm, config.rms_norm_eps, use_cuda=use_cuda)
        k = kernels.rmsnorm(k, self.k_norm, config.rms_norm_eps, use_cuda=use_cuda)

        # RoPE needs a leading axis to broadcast the position across. Positions are per
        # token and come from the batch, since a chunk's tokens may sit at 512..1023 and
        # nothing in these shapes records that.
        q = kernels.rope(
            q.unsqueeze(0), batch.positions, self.rope.cos, self.rope.sin, use_cuda=use_cuda
        ).squeeze(0)
        k = kernels.rope(
            k.unsqueeze(0), batch.positions, self.rope.cos, self.rope.sin, use_cuda=use_cuda
        ).squeeze(0)

        # Rotated keys go into the pool, so a cached key carries its position and is
        # never re-rotated on a later read.
        manager.kv.write(self.layer, batch.slot_mapping, k, v)

        attended = kernels.paged_attention(
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


class Qwen3PagedBlock:
    """One pre-norm block over the ragged token axis."""

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
        self.mlp = Qwen3MLP(weights, use_cuda)
        self.attn_norm = weights["attn_norm"]
        self.mlp_norm = weights["mlp_norm"]

    def __call__(self, x: torch.Tensor, batch: ForwardBatch, manager: BlockManager) -> torch.Tensor:
        eps = self.config.rms_norm_eps
        normed = kernels.rmsnorm(x, self.attn_norm, eps, use_cuda=self.use_cuda)
        x = x + self.attention(normed, batch, manager)

        normed = kernels.rmsnorm(x, self.mlp_norm, eps, use_cuda=self.use_cuda)
        return x + self.mlp(normed)


class Qwen3Paged:
    """Qwen3 over a ragged batch and a paged cache: the model the engine runs. Takes a
    ForwardBatch of T tokens across N sequences, returns logits [N, V].

    The third fork keeps no cache of its own, writing keys and values straight into
    the pool. Two changes from Qwen3Cached, both shape rather than mathematics.
    There is no batch axis -- activations are T x ... over every scheduled token,
    since a padded rectangle would give back the memory paging buys. And attention
    takes metadata instead of tensors: cu_seqlens_q, context_lens, and the block
    tables, so one call covers a 512-token prefill chunk and eleven decode steps.
    Only N rows come back because generation samples from a sequence's last position
    and the head is a V-wide matmul; speculative verification asks for all T.
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
        self.embedding, self.final_norm, self.rope = _model_common(config, weights)
        self.blocks = [
            Qwen3PagedBlock(layer, config, _layer_weights(weights, layer), self.rope, use_cuda)
            for layer in range(config.num_hidden_layers)
        ]

    @classmethod
    def from_pretrained(
        cls,
        model: str = DEFAULT_MODEL_ID,
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
        """This model truncated to its first num_layers, over a separate KV pool.

        The draft for speculative decoding on a machine that cannot afford a second
        checkpoint. Running the target's early layers costs no additional memory -- the
        blocks, embedding and head are the same tensors -- and a prefix of a transformer
        is both cheaper and weaker, which is what a draft needs to be. It needs its own
        manager because a draft keeps its own KV. Weakness is acceptable where wrongness
        is not: rejection sampling is indifferent to draft quality.
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
        """One forward pass, returning [N, V] normally and [T, V] when all_rows. all_rows
        exists for speculative verification, which forwards a pending token plus k
        proposals and needs the target's distribution at all k + 1 positions in one pass.
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

        h = kernels.rmsnorm(h, self.final_norm, self.config.rms_norm_eps, use_cuda=self.use_cuda)
        return self.embedding.as_linear(h)
