"""The reference implementations: every operation the engine runs, in its slow,
obviously correct form.

What this file teaches
    The complete inventory of math inside an LLM inference engine — nine
    operations, in the order a token meets them: linear projection, activation,
    normalization, rotary position, attention (grouped, causal, and paged), and
    sampling. Everything else in this project is bookkeeping around these.

Inputs and outputs
    Plain tensors in, plain tensors out. Shapes are written with ``B`` batch,
    ``L`` query length, ``S`` source (context) length, ``H_q``/``H_k`` query/KV
    heads, ``D`` head dim, ``E`` hidden size, ``V`` vocabulary, ``T`` total
    tokens of a ragged batch, and ``N..`` for any leading batch dimensions.

Read next
    `model.py` — these ops assembled into the Qwen3 forward pass.

One invariant
    Every function here is a correctness oracle. The CUDA kernels in `csrc/`
    and the fast paths in `kernels.py` are diffed against these exact
    implementations, so they are written for clarity and kept frozen: a change
    here changes what "correct" means for the whole project.

Runnable example
    >>> import torch
    >>> from mini_vllm import ops
    >>> q = torch.randn(1, 16, 4, 64)   # B x H_q x L x D
    >>> kv = torch.randn(1, 8, 4, 64)   # B x H_k x S x D
    >>> ops.scaled_dot_product_attention_grouped(q, kv, kv, mask="causal").shape
    torch.Size([1, 16, 4, 64])
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from mini_vllm.config import SamplingParams

__all__ = [
    "linear",
    "Embedding",
    "silu",
    "softmax",
    "rms_norm",
    "rotate_half",
    "apply_rope",
    "RoPE",
    "scaled_dot_product_attention_grouped",
    "causal_mask",
    "paged_attention_gathered",
    "sampling_probabilities",
    "sample",
]


# ------------------------------------------------------------------- 1. linear


def linear(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``y = x @ w.T (+ bias)``.

    ::

        x:    N.. x I
        w:    O x I        (transposed, the HuggingFace storage convention)
        bias: O
        out:  N.. x O

    The weight is stored as ``O x I`` rather than ``I x O`` because that is the
    checkpoint convention. Matching it here means the weight loader never transposes,
    and a transposed weight surfaces as a shape error rather than wrong numbers.
    """
    out = x @ w.transpose(-2, -1)
    if bias is not None:
        out = out + bias
    return out


class Embedding:
    """A ``V x E`` table, readable as a lookup or as a linear projection.

    Qwen3-0.6B sets ``tie_word_embeddings=true``, so one matrix serves as both the
    input embedding and the output projection: hence one object with two methods
    rather than two layers. At ``V x E`` = 151936 x 1024 it is about 155M of the
    596M parameters, and untying it would add another 155M.

    ::

        weight:         V x E
        __call__(ids):  B x L      (int64)  ->  B x L x E
        as_linear(h):   B x L x E            ->  B x L x V
    """

    def __init__(self, vocab_size: int, dim: int, weight: torch.Tensor) -> None:
        if weight.shape != (vocab_size, dim):
            raise ValueError(
                f"weight must have shape ({vocab_size}, {dim}), got {tuple(weight.shape)}"
            )
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = weight

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        """Gather one row per token id."""
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"ids must be integer, got {ids.dtype}")
        return self.weight[ids]

    def as_linear(self, h: torch.Tensor) -> torch.Tensor:
        """``h @ weightᵀ``: the tied LM head.

        A vocabulary-wide matmul, and the most expensive op in a decode step: `E x V`
        work to produce logits for one token. The serving layer therefore runs it only
        on the last position of each sequence.
        """
        if h.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {h.shape[-1]}")
        return linear(h, self.weight)


# ----------------------------------------------------------- 2. silu / softmax


def silu(x: torch.Tensor) -> torch.Tensor:
    """``x * sigmoid(x)``, the activation inside Qwen3's SwiGLU MLP.

    The fused SwiGLU CUDA kernel computes ``silu(gate) * up`` in one pass; this is
    the expression it must agree with.
    """
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax along ``dim``, computed in fp32 and returned in the input dtype.

    Two properties that recur in every attention kernel:

    Subtract the row max first. ``exp`` overflows to ``inf`` around 88 in fp32 and
    attention logits routinely exceed that; subtracting the max makes the largest
    exponent exactly ``exp(0) == 1`` without changing the result, since the shared factor
    cancels between numerator and denominator. This is the basis of the online-softmax
    recurrence in the decode attention kernel, where the max arrives incrementally and
    the running total is rescaled as it changes.

    Reduce in fp32. Summing bf16 exponentials loses enough precision to move greedy
    tokens a few layers downstream.
    """
    x32 = x.float()
    x32 = x32 - x32.max(dim=dim, keepdim=True).values
    exp = torch.exp(x32)
    return (exp / exp.sum(dim=dim, keepdim=True)).to(x.dtype)


# ------------------------------------------------------------------ 3. RMSNorm


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``x * rsqrt(mean(x²) + eps) * weight``, reducing over the last dimension.

    ::

        x:      N.. x dim
        weight: dim
        out:    N.. x dim

    Qwen3 uses this in three places: before attention, before the MLP, and inside
    attention as QK-norm over the head dimension of `q` and `k`. The same function
    serves all three, differing only in the width of the reduced axis. Unlike
    LayerNorm there is no mean subtraction and no bias: the vector is rescaled but
    not recentred.

    The reduction is fp32 even when ``x`` is bf16, and that is required rather than
    conservative. A bf16 sum of 1024 squares carries roughly three decimal digits, and
    the error feeds a multiplicative rescale of the whole residual stream; across 28
    layers it moves greedy tokens.

    The cast back to the input dtype happens before the weight multiply, matching
    HuggingFace, so this is exactly rather than approximately comparable to the oracle.
    """
    input_dtype = x.dtype

    x32 = x.float()
    mean_square = x32.pow(2).mean(dim=-1, keepdim=True)
    normalized = x32 * torch.rsqrt(mean_square + eps)

    return weight * normalized.to(input_dtype)


# --------------------------------------------------------------------- 4. RoPE


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[x1, x2] -> [-x2, x1]``, splitting the head dimension in half.

    The "rotate halves" convention, pairing element ``i`` with element ``i + D/2``. The
    original RoFormer paper pairs adjacent elements instead; the two are related by a
    permutation of the head dimension, so both are self-consistent but not
    interchangeable against a given checkpoint. Qwen3's weights were trained with this
    one, and the other produces fluent nonsense rather than an error.
    """
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Rotate ``x`` using precomputed tables, gathered at ``positions``.

    ::

        x:         B x L x H x D
        positions: L  or  B x L
        cos, sin:  max_seq_len x D
        out:       same shape and dtype as x

    Split out from :class:`RoPE` so it takes the tables as plain tensors: the fused RoPE
    kernel has the same signature, which lets `kernels.rope` dispatch between the two
    without either side knowing about the other.
    """
    # positions is L or B x L, so the gather yields L x D or B x L x D; the head
    # axis is inserted so one row applies to every head of its token.
    gathered_cos = cos.to(x.device)[positions].unsqueeze(-2)
    gathered_sin = sin.to(x.device)[positions].unsqueeze(-2)

    rotated = x.float() * gathered_cos + rotate_half(x.float()) * gathered_sin
    return rotated.to(x.dtype)


class RoPE:
    """Precomputed rotary embedding tables, applied at explicit positions.

    The interface carries the design decision: :meth:`__call__` takes the absolute
    position of every token as a tensor and never assumes ``arange(L)``. That is what
    makes the serving layer possible. A decode step for a sequence at position 500 and
    a prefill chunk covering positions 0-511 share a single forward pass, so position
    is a property of each token rather than of the batch. An implicit ``arange``
    produces correct prefill and a subtly wrong continuation.

    ::

        x:         B x L x H x D   (or any N.. x H x D with positions to match)
        positions: L  or  B x L    (int64, absolute position of each token)
        out:       same shape as x

    ``cos`` and ``sin`` are exposed as attributes because the fused RoPE kernel reads
    the same tables directly and must agree with this implementation row for row.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 1_000_000.0,
        device: torch.device | str | None = None,
    ) -> None:
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Frequency i decays as theta^(-2i/D): the first pairs rotate quickly and encode
        # local offsets, the last rotate slowly and encode long-range position. Qwen3's
        # theta of 1e6 stretches the slow end far enough to cover a long context.
        exponents = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
        inverse_frequencies = 1.0 / (theta**exponents)

        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = torch.outer(positions, inverse_frequencies)  # max_seq_len x D/2

        # Duplicated rather than interleaved, matching `rotate_half`: element i and
        # element i + D/2 share a rotation angle.
        angles = torch.cat((angles, angles), dim=-1)  # max_seq_len x D

        # Tables stay fp32 even when activations are bf16: a bf16 cosine near a zero
        # crossing loses enough precision to shift tokens.
        self.cos = angles.cos()
        self.sin = angles.sin()

    def __call__(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate ``x`` at ``positions``, returning ``x``'s dtype."""
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"positions must be integer, got {positions.dtype}")

        maximum = int(positions.max()) if positions.numel() else -1
        if maximum >= self.max_seq_len:
            raise ValueError(
                f"position {maximum} is beyond the precomputed table "
                f"(max_seq_len={self.max_seq_len})"
            )

        return apply_rope(x, positions, self.cos, self.sin)


# ------------------------------------------------- 5. grouped-query attention


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
    broadcasting align each KV head with its own group, rather than materializing ``G``
    copies of K and V with ``repeat_interleave``. Same numbers, without the ``G``-fold
    duplication of the cache that GQA exists to avoid and that the paged attention
    kernels rely on.

    This materializes the whole ``L x S`` score matrix, which is what the
    FlashAttention-style kernels avoid; it exists as the oracle they are diffed
    against, so it is written for clarity rather than speed. Pass ``mask="causal"``
    for dense causal attention (see :func:`causal_mask` for the decode offset).
    """
    *batch, num_query_heads, query_len, head_dim = q.shape
    num_kv_heads = k.shape[-3]

    if num_query_heads % num_kv_heads != 0:
        raise ValueError(f"H_q ({num_query_heads}) must be a multiple of H_k ({num_kv_heads})")
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
            # (B, L, S) and (H_q, L, S) are indistinguishable from the shape alone and
            # would reshape the wrong way silently, so refuse to guess.
            raise ValueError(
                "ambiguous 3-D mask: pass L x S to share across heads, "
                "or B x H_q x L x S to vary per head"
            )
        scores = scores + mask

    out = softmax(scores, dim=-1) @ v
    return out.reshape(*batch, num_query_heads, query_len, head_dim)


# ---------------------------------------------------- 6. dense causal attention


def causal_mask(
    query_len: int,
    source_len: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """An additive causal mask of shape ``query_len x source_len``.

    Dense causal attention is :func:`scaled_dot_product_attention_grouped` plus this
    mask: query ``i`` may attend to key ``j`` only when ``j <= source_len - query_len
    + i``. The mask is additive rather than boolean because additive masks compose:
    adding ``-inf`` before the softmax drives a position to exactly zero afterwards,
    and several masks can be summed.

    The offset carries the subtlety. With ``L == S`` (prefill) it reduces to a lower
    triangle. With ``L < S`` (decode against a filled cache) the ``L`` queries are the
    last ``L`` positions of the sequence rather than the first, so the diagonal shifts
    right by ``S - L``. Omitting the shift yields a model that prefills correctly and
    then decodes nonsense, because a decode token would be forbidden from seeing its own
    cache.
    """
    offset = source_len - query_len
    rows = torch.arange(query_len, device=device).unsqueeze(1)
    columns = torch.arange(source_len, device=device).unsqueeze(0)

    allowed = columns <= rows + offset
    return torch.zeros(query_len, source_len, dtype=dtype, device=device).masked_fill(
        ~allowed, float("-inf")
    )


# --------------------------------------------- 7. reference paged attention


def paged_attention_gathered(
    q: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float | None = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> torch.Tensor:
    """Grouped-query causal attention over a paged cache, one sequence at a time.

    The oracle for the paged attention kernels. For each sequence it walks the block
    table, copies that sequence's keys and values out of the pool into a contiguous
    tensor, and calls the reference attention on it.

    That is a full copy of the cache every iteration, the traffic paging exists to
    avoid and worse than the dense cache it replaces. It is also correct by
    construction: with :func:`scaled_dot_product_attention_grouped` as the reference,
    a correct gather gives a correct answer, which is what the kernels are diffed
    against.

    ::

        q:            T x H_q x D            every scheduled token, sequences concatenated
        key_pool:     num_blocks x P x H_k x D   (one layer's pages; P = block size)
        value_pool:   num_blocks x P x H_k x D
        block_tables: int32 N x max_blocks   -1 padded
        cu_seqlens_q: int32 N + 1            where each sequence's queries start
        context_lens: int32 N                how many cached tokens each attends over
        out:          T x H_q x D

    The flattened token axis is what lets one call serve a mixed batch: a 300-token
    prefill chunk followed by a dozen single-token decodes is 312 rows here,
    distinguished only by `cu_seqlens_q`. Each sequence is masked causally with its own
    `(L, S)` offset: a decode step's single query sees the whole context, and a prefill
    chunk's queries are the last `L` positions of `S`.

    With FP8 pools the gather also dequantizes, casting each cached key and value up and
    multiplying by its scale before the math, so the oracle attends in the activation
    dtype exactly as the kernel does after dequantizing in registers. The scales default
    to 1.0, the identity for an unquantized pool.
    """
    if q.dim() != 3:
        raise ValueError(f"expected q shaped T x H_q x D, got {tuple(q.shape)}")
    if key_pool.dim() != 4 or key_pool.shape != value_pool.shape:
        raise ValueError(
            f"expected matching pools shaped num_blocks x P x H_k x D, got "
            f"{tuple(key_pool.shape)} and {tuple(value_pool.shape)}"
        )

    num_sequences = context_lens.shape[0]
    if block_tables.shape[0] != num_sequences or cu_seqlens_q.shape[0] != num_sequences + 1:
        raise ValueError(
            f"metadata disagrees on the sequence count: block_tables "
            f"{tuple(block_tables.shape)}, cu_seqlens_q {tuple(cu_seqlens_q.shape)}, "
            f"context_lens {tuple(context_lens.shape)}"
        )
    if int(cu_seqlens_q[-1]) != q.shape[0]:
        raise ValueError(
            f"cu_seqlens_q ends at {int(cu_seqlens_q[-1])} but q has {q.shape[0]} rows"
        )

    block_size, num_kv_heads, head_dim = key_pool.shape[1:]
    out = torch.empty_like(q)

    for index in range(num_sequences):
        start, end = int(cu_seqlens_q[index]), int(cu_seqlens_q[index + 1])
        query_len = end - start
        context_len = int(context_lens[index])
        if context_len < query_len:
            raise ValueError(
                f"sequence {index} attends over {context_len} tokens but computes "
                f"{query_len}; S >= L is causality, not convention"
            )

        blocks_used = -(-context_len // block_size)
        table = block_tables[index, :blocks_used]
        if int(table.min()) < 0:
            raise ValueError(
                f"sequence {index} needs {blocks_used} blocks for {context_len} tokens "
                f"but its table is padded there: {block_tables[index].tolist()}"
            )

        # The gather: `index_select` on the block axis, then flatten block and offset
        # back into one logical axis, which works because the two are adjacent. The
        # kernel does the same arithmetic per element instead of per block.
        ids = table.to(dtype=torch.int64, device=key_pool.device)
        keys = key_pool.index_select(0, ids).reshape(-1, num_kv_heads, head_dim)
        values = value_pool.index_select(0, ids).reshape(-1, num_kv_heads, head_dim)

        # Dequantize an FP8 cache before the math; a no-op cast for a matching pool.
        if keys.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            keys = keys.float().mul_(k_scale).to(q.dtype)
            values = values.float().mul_(v_scale).to(q.dtype)

        keys = keys[:context_len].permute(1, 0, 2).unsqueeze(0)  # 1 x H_k x S x D
        values = values[:context_len].permute(1, 0, 2).unsqueeze(0)
        queries = q[start:end].permute(1, 0, 2).unsqueeze(0)  # 1 x H_q x L x D

        attended = scaled_dot_product_attention_grouped(
            queries, keys, values, scale=scale, mask="causal"
        )
        out[start:end] = attended.squeeze(0).permute(1, 0, 2)

    return out


# --------------------------------------------------- 8. sampling probabilities

# Everything below is vectorized across the batch with per-row parameters. A server
# batches whatever requests arrive together and they will not agree on temperature:
# row 0 may be greedy while row 1 wants `temperature=1.2, top_p=0.9`. Looping over rows
# would put a Python loop inside the decode step, the hottest path in the engine, so
# every row is masked and scaled in parallel and greedy is handled as `temperature == 0`
# rather than as a separate code path. The cost is one sort along the vocabulary axis
# per step, which is what makes per-row top-k and top-p expressible as pure tensor ops.


def _as_columns(
    params: SamplingParams | Sequence[SamplingParams],
    batch: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Broadcast per-request parameters into three ``B x 1`` tensors."""
    rows = [params] * batch if isinstance(params, SamplingParams) else list(params)
    if len(rows) != batch:
        raise ValueError(f"got {len(rows)} sampling params for a batch of {batch}")

    def column(values, dtype) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device=device).unsqueeze(1)

    return (
        column([row.temperature for row in rows], torch.float32),
        column([row.top_k for row in rows], torch.int64),
        column([row.top_p for row in rows], torch.float32),
    )


def sampling_probabilities(
    logits: torch.Tensor,
    params: SamplingParams | Sequence[SamplingParams],
) -> torch.Tensor:
    """The distribution each row will actually be sampled from.

    ::

        logits: B x V   ->   probabilities: B x V   (fp32, rows sum to 1)

    Exposed separately from :func:`sample` because it makes the sampler testable: a
    truncation rule is easier to verify by inspecting the distribution it produces than
    by drawing from it. Greedy rows come back one-hot. Speculative decoding also builds
    on this: rejection sampling needs the target's and the draft's *distributions*, not
    their draws.

    Temperature is applied first, then top-k and top-p together on the scaled
    distribution, so the truncation sees the same probabilities the draw will.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected B x V logits, got shape {tuple(logits.shape)}")

    batch, vocab = logits.shape
    temperature, top_k, top_p = _as_columns(params, batch, logits.device)

    greedy = temperature == 0.0
    # Divide by 1.0 on greedy rows to keep this branch-free and finite; those rows
    # are overwritten with their one-hot below.
    scaled = logits.float() / torch.where(greedy, torch.ones_like(temperature), temperature)

    descending, order = scaled.sort(dim=-1, descending=True)
    probabilities = descending.softmax(dim=-1)

    # top-k: keep ranks [0, k). k == 0 disables it.
    rank = torch.arange(vocab, device=logits.device).unsqueeze(0)
    effective_k = torch.where(top_k == 0, torch.full_like(top_k, vocab), top_k)
    keep = rank < effective_k

    # top-p: keep a token while the probability mass strictly before it is below p. That
    # gives the smallest prefix whose total reaches p, including the boundary token that
    # crosses it rather than stopping short.
    mass_before = probabilities.cumsum(dim=-1) - probabilities
    keep &= mass_before < top_p

    # The most likely token is always kept, so no row is fully masked however small p is.
    keep[:, 0] = True

    truncated = probabilities * keep
    truncated = truncated / truncated.sum(dim=-1, keepdim=True)

    # Back to vocabulary order.
    result = torch.zeros_like(truncated).scatter_(dim=-1, index=order, src=truncated)

    if bool(greedy.any()):
        one_hot = torch.zeros_like(result).scatter_(
            dim=-1, index=logits.argmax(dim=-1, keepdim=True), value=1.0
        )
        result = torch.where(greedy, one_hot, result)

    return result


# ----------------------------------------------------------------- 9. sampling


def sample(
    logits: torch.Tensor,
    params: SamplingParams | Sequence[SamplingParams] = SamplingParams(),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one token per row: greedy, temperature, top-k and top-p in one call.

    ::

        logits: B x V   ->   tokens: B   (int64)

    Greedy rows (``temperature == 0``) come out of the same path — their distribution
    is one-hot, so the multinomial draw is deterministic. Pass a ``generator`` to make
    a draw reproducible without disturbing global RNG state, which is what lets a
    server replay one request.
    """
    probabilities = sampling_probabilities(logits, params)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(1)
