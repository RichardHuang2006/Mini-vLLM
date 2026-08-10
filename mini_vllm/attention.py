"""Steps 1.2 and 1.4 — attention, from the textbook form to Qwen3's GQA.

Two implementations live here and both are kept forever:

* :func:`scaled_dot_product_attention_simple` is the textbook form, `H_q == H_k`.
* :func:`scaled_dot_product_attention_grouped` is what Qwen3 actually uses, with
  fewer key/value heads than query heads.

Neither is fast. They materialize the whole `L x S` score matrix, which is
exactly what the FlashAttention-style kernels in Steps 3.4 and 3.5 avoid. That
is the point: these are the oracles those kernels are diffed against, so they
are written to be obviously right rather than efficient.
"""

from __future__ import annotations

import math

import torch

from mini_vllm.basics import linear, softmax

__all__ = [
    "causal_mask",
    "scaled_dot_product_attention_simple",
    "scaled_dot_product_attention_grouped",
    "SimpleMultiHeadAttention",
]


def scaled_dot_product_attention_simple(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:
    """``softmax(Q Kᵀ / sqrt(D) + M) V``, over the last two dimensions.

    ::

        q:    N.. x L x D
        k, v: N.. x S x D
        mask: broadcastable to N.. x L x S, additive (0 keeps, -inf drops)
        out:  N.. x L x D

    ``scale`` defaults to ``1/sqrt(D)``. The mask is *additive* rather than
    boolean because that is what composes: adding ``-inf`` before the softmax
    drives a position to exactly zero afterwards, and several masks can simply
    be summed.
    """
    head_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    scores = (q @ k.transpose(-2, -1)) * scale

    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError(f"unknown mask shorthand {mask!r}, expected 'causal'")
        mask = causal_mask(q.shape[-2], k.shape[-2], q.dtype, q.device)
    if mask is not None:
        scores = scores + mask

    return softmax(scores, dim=-1) @ v


def causal_mask(
    query_len: int,
    source_len: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """An additive causal mask of shape ``query_len x source_len``.

    Query ``i`` may attend to key ``j`` only when ``j <= source_len - query_len + i``.

    That offset is the whole subtlety. When ``L == S`` (prefill) it reduces to
    the familiar lower triangle. When ``L < S`` (decode against a filled cache)
    the ``L`` queries are the *last* ``L`` positions of the sequence, not the
    first, so the diagonal has to be shifted right by ``S - L``. Getting this
    wrong yields a model that prefills correctly and then decodes nonsense,
    because a single decode token would be forbidden from seeing its own cache.
    """
    offset = source_len - query_len
    rows = torch.arange(query_len, device=device).unsqueeze(1)
    columns = torch.arange(source_len, device=device).unsqueeze(0)

    allowed = columns <= rows + offset
    return torch.zeros(query_len, source_len, dtype=dtype, device=device).masked_fill(
        ~allowed, float("-inf")
    )


def scaled_dot_product_attention_grouped(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:
    """Grouped-query attention: ``H_k`` key/value heads serving ``H_q`` query heads.

    ::

        q:    B x H_q x L x D
        k, v: B x H_k x S x D
        out:  B x H_q x L x D

        G = H_q / H_k        each KV head serves G query heads

    Implemented by reshaping the query into ``B x H_k x G x L x D`` and letting
    broadcasting line each KV head up against its own group, rather than by
    materializing ``G`` copies of K and V with ``repeat_interleave``. Same
    numbers, but no ``G``-fold duplication of the cache — which is the entire
    reason GQA exists, and which the paged kernels in Step 4.8 depend on.
    """
    *batch, num_query_heads, query_len, head_dim = q.shape
    num_kv_heads = k.shape[-3]

    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"H_q ({num_query_heads}) must be a multiple of H_k ({num_kv_heads})"
        )
    group_size = num_query_heads // num_kv_heads
    source_len = k.shape[-2]

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Split the query-head axis into (kv_head, group) so it broadcasts against
    # the singleton group axis inserted into K and V below.
    q = q.reshape(*batch, num_kv_heads, group_size, query_len, head_dim)
    k = k.unsqueeze(-3)
    v = v.unsqueeze(-3)

    scores = (q @ k.transpose(-2, -1)) * scale

    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError(f"unknown mask shorthand {mask!r}, expected 'causal'")
        mask = causal_mask(query_len, source_len, q.dtype, q.device)
    if mask is not None:
        if mask.ndim == 4:
            # A per-head mask is shaped for B x H_q x L x S, so its head axis
            # needs the same (kv_head, group) split as the query.
            mask = mask.reshape(*mask.shape[:-3], num_kv_heads, group_size, query_len, source_len)
        elif mask.ndim == 3:
            # (B, L, S) and (H_q, L, S) are indistinguishable from the shape
            # alone and reshape the wrong way silently, so refuse to guess.
            raise ValueError(
                "ambiguous 3-D mask: pass L x S to share across heads, "
                "or B x H_q x L x S to vary per head"
            )
        scores = scores + mask

    out = softmax(scores, dim=-1) @ v
    return out.reshape(*batch, num_query_heads, query_len, head_dim)


class SimpleMultiHeadAttention:
    """Textbook multi-head attention, ``H_q == H_k``, no RoPE and no cache.

    ::

        x, out:   B x L x E
        wq/wk/wv: (H·D) x E
        wo:       E x (H·D)

    Note that ``H·D`` need not equal ``E``: in Qwen3-0.6B the attention
    projection is twice the hidden size. The head dimension is therefore derived
    from the weight rather than from ``E / H``, which is the assumption that
    breaks on the real model.

    Superseded by the Qwen3 attention block in Step 1.8, which adds QK-norm,
    RoPE, GQA and a cache. Kept because it is the smallest thing that can be
    checked directly against ``torch.nn.MultiheadAttention``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
    ) -> None:
        if wq.shape[0] % num_heads != 0:
            raise ValueError(
                f"wq rows ({wq.shape[0]}) must be a multiple of num_heads ({num_heads})"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = wq.shape[0] // num_heads
        self.wq, self.wk, self.wv, self.wo = wq, wk, wv, wo

    def __call__(self, x: torch.Tensor, mask: torch.Tensor | str | None = None) -> torch.Tensor:
        batch, length, _ = x.shape
        heads, head_dim = self.num_heads, self.head_dim

        def project(weight: torch.Tensor) -> torch.Tensor:
            # B x L x (H·D) -> B x H x L x D, so attention reduces over L and D
            # with the head axis carried along as a batch dimension.
            return linear(x, weight).reshape(batch, length, heads, head_dim).transpose(1, 2)

        attended = scaled_dot_product_attention_simple(
            project(self.wq), project(self.wk), project(self.wv), mask=mask
        )

        merged = attended.transpose(1, 2).reshape(batch, length, heads * head_dim)
        return linear(merged, self.wo)
