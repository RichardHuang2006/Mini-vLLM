"""The seam between the model and its kernels.

Every op the cached model performs goes through a function here, and each one can run
either the PyTorch reference implementation or a hand-written CUDA kernel. The model
never learns which: it passes a `use_cuda` flag down and this module dispatches.

The seam keeps each kernel an independently verifiable change rather than a rewrite:
landing the RMSNorm kernel means writing `csrc/rmsnorm.cu`, pointing `rmsnorm` at it, and
setting one entry in :data:`CUDA_KERNELS`. The model file is untouched and the PyTorch
version remains the oracle.

`use_cuda=True` means use kernels where they exist, so an op with no kernel falls back
silently, as does any op called on CPU tensors, since the same model object serves CPU
tests and GPU runs. That silence is how a kernel that never ran ends up in a benchmark,
so :func:`dispatch_report` states which path each op would take and the benchmark prints
it.
"""

from __future__ import annotations

import math

import torch

from mini_vllm import attention as reference_attention
from mini_vllm.basics import silu
from mini_vllm.kernels.extension import load_extension
from mini_vllm.layer_norm import rms_norm
from mini_vllm.paged_attention import paged_attention_gathered as reference_paged_attention
from mini_vllm.positional_encoding import apply_rope

__all__ = [
    "CUDA_KERNELS",
    "attention",
    "cuda_kernel_names",
    "dispatch_report",
    "paged_attention",
    "quantize_scatter",
    "rmsnorm",
    "rope",
    "swiglu",
]

# Which ops have a hand-written kernel, paired with the source file the dispatch report
# names beside each one.
#
# The keys are the names the extension exports rather than synonyms:
# `test_every_claimed_kernel_is_callable` looks each up in the compiled module, so a
# kernel cannot be claimed here while being missing, misspelled, or renamed.
CUDA_KERNELS: dict[str, tuple[bool, str]] = {
    "rmsnorm": (True, "csrc/rmsnorm.cu"),
    "rope": (True, "csrc/rope.cu"),
    "swiglu": (True, "csrc/swiglu.cu"),
    "decode_attention": (True, "csrc/decode_attention.cu"),
    "flash_prefill": (True, "csrc/flash_prefill.cu"),
    "paged_attention": (True, "csrc/paged_attention.cu"),
    "kv_quantize_scatter": (True, "csrc/kv_quantize.cu"),
}

# The one FP8 format the kernels accelerate. e5m2 remains a legal storage dtype for the
# pool and the PyTorch paths quantize and dequantize it correctly, but it does not justify
# a second set of template instantiations in every kernel, so a pool in it degrades to the
# oracle rather than being half-supported.
FP8_KERNEL_DTYPE = torch.float8_e4m3fn

# Kernels that are correct, tested, and not the default, with the reason measured by the
# benchmark. A kernel earns the dispatch by being faster, and routing to a slower one
# behind `use_cuda=True` is what `dispatch_report` exists to prevent.
#
# Prefill is listed because its inner loop is scalar FMA against cuBLAS on tensor cores,
# a gap tuning does not close; replacing the loop with `mma.sync` would flip this entry.
# The tests call the kernel directly through the extension, so its coverage does not
# depend on routing.
NOT_YET_FASTER: dict[str, str] = {
    "flash_prefill": "0.4x cuBLAS at L=512",
}

# The prefill kernel stages query, key and value tiles in shared memory simultaneously,
# which bounds the head dimension it can serve. Qwen3-0.6B uses 128; anything wider goes
# to the oracle.
MAX_KERNEL_HEAD_DIM = 192

# The paged kernel holds the query and the accumulator in registers instead, `D / 32`
# floats per lane of each, which is where its own ceiling comes from.
MAX_PAGED_HEAD_DIM = 256


def _use_kernel(name: str, use_cuda: bool, x: torch.Tensor) -> bool:
    """Whether ``name`` should run on the GPU for this call.

    Requires an implemented kernel that is the default and CUDA tensors: the same model
    object serves CPU tests and GPU runs.
    """
    implemented, _source = CUDA_KERNELS[name]
    return use_cuda and implemented and name not in NOT_YET_FASTER and x.is_cuda


def cuda_kernel_names() -> list[str]:
    """The ops that currently have a working kernel."""
    return [name for name, (implemented, _) in CUDA_KERNELS.items() if implemented]


def dispatch_report(use_cuda: bool) -> str:
    """One line per op saying which implementation a run would use."""
    lines = []
    for name, (implemented, source) in CUDA_KERNELS.items():
        if not implemented:
            note = f"no kernel yet — {source}"
        elif name in NOT_YET_FASTER:
            note = f"kernel from {source} is slower: {NOT_YET_FASTER[name]}"
        elif not use_cuda:
            note = f"kernel exists from {source}; use_cuda is off"
        else:
            lines.append(f"  {name:<18} cuda    (kernel from {source})")
            continue
        lines.append(f"  {name:<18} torch   ({note})")
    return "\n".join(lines)


# ------------------------------------------------------------------------ ops


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    use_cuda: bool = False,
) -> torch.Tensor:
    """RMSNorm over the last dimension. Kernel: ``csrc/rmsnorm.cu``.

    The kernel requires ``weight`` to have the same dtype as ``x``. PyTorch would promote
    instead, and returning a different dtype than the oracle is worse than declining the
    kernel. The model never mixes them.
    """
    if _use_kernel("rmsnorm", use_cuda, x) and weight.dtype == x.dtype:
        return load_extension().rmsnorm(x, weight, eps)
    return rms_norm(x, weight, eps)


def rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    use_cuda: bool = False,
) -> torch.Tensor:
    """Rotary embedding at explicit positions. Kernel: ``csrc/rope.cu``.

    The kernel requires fp32 tables, which `RoPE` builds regardless of the activation
    dtype, and a head axis to broadcast the position across. Anything else goes to the
    oracle.
    """
    if _use_kernel("rope", use_cuda, x) and cos.dtype == torch.float32 and x.dim() >= 3:
        return load_extension().rope(x, positions, cos, sin)
    return apply_rope(x, positions, cos, sin)


def swiglu(gate: torch.Tensor, up: torch.Tensor, use_cuda: bool = False) -> torch.Tensor:
    """``silu(gate) * up``, the elementwise half of the MLP. Kernel: ``csrc/swiglu.cu``.

    Takes the two projections rather than the input: the projections are ordinary matmuls
    that cuBLAS handles better than a hand-written kernel. What is worth fusing is this
    part, two full passes over a `B x L x intermediate` tensor for a few flops per
    element, which is pure memory traffic.
    """
    if _use_kernel("swiglu", use_cuda, gate) and gate.dtype == up.dtype:
        return load_extension().swiglu(gate, up)
    return silu(gate) * up


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | str | None = None,
    use_cuda: bool = False,
) -> torch.Tensor:
    """Grouped-query attention, with a separate kernel for decode and for prefill.

    ::

        q:    B x H_q x L x D
        k, v: B x H_k x S x D
        out:  B x H_q x L x D

    Decode and prefill are different problems rather than different sizes of one. Decode
    has `L = 1`: no parallelism across queries, one long pass over the cache, entirely
    memory-bound. Prefill has `L = S`: a matrix multiply worth tiling. Hence separate
    kernels, ``csrc/decode_attention.cu`` and ``csrc/flash_prefill.cu``, with the routing
    here so the model does not have to choose.
    """
    is_decode = q.shape[-2] == 1
    name = "decode_attention" if is_decode else "flash_prefill"

    if _use_kernel(name, use_cuda, q) and _kernel_can_attend(q, k, v, mask, is_decode):
        scale = 1.0 / math.sqrt(q.shape[-1])
        if is_decode:
            return load_extension().decode_attention(q, k, v, scale)
        return load_extension().flash_prefill(q, k, v, scale)

    return reference_attention.scaled_dot_product_attention_grouped(q, k, v, mask=mask)


def paged_attention(
    q: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    context_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    max_context_len: int,
    scale: float | None = None,
    use_cuda: bool = False,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> torch.Tensor:
    """Ragged attention over a paged cache. Kernel: ``csrc/paged_attention.cu``.

    ::

        q:    T x H_q x D              every scheduled token, sequences concatenated
        pools num_blocks x P x H_k x D
        out:  T x H_q x D

    The fallback differs in kind from the others here. The PyTorch path for `rmsnorm` is
    a slower way to do the same work; the PyTorch path for this copies every sequence's
    whole cache into a contiguous temporary first, which is the traffic paging exists to
    remove. It is the oracle rather than an alternative, and the engine's
    `--mode throughput` numbers are only meaningful on the kernel path.

    `max_query_len` and `max_context_len` are plain integers rather than reads off
    `seq_lens` and `context_lens`. They decide a grid shape and a split count, so reading
    them from a device tensor would synchronize on every iteration's critical path, and
    the caller already holds them as Python integers from building the batch.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    if _use_kernel("paged_attention", use_cuda, q) and _kernel_can_page(
        q, key_pool, value_pool, block_tables
    ):
        return load_extension().paged_attention(
            q,
            key_pool,
            value_pool,
            block_tables,
            cu_seqlens_q,
            context_lens,
            seq_lens,
            int(max_query_len),
            int(max_context_len),
            scale,
            k_scale,
            v_scale,
        )

    return reference_paged_attention(
        q,
        key_pool,
        value_pool,
        block_tables,
        cu_seqlens_q,
        context_lens,
        scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )


def quantize_scatter(
    key: torch.Tensor,
    value: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: float,
    v_scale: float,
    use_cuda: bool = False,
) -> None:
    """Quantize a step's KV to FP8 and scatter it into the pool. Kernel: ``csrc/kv_quantize.cu``.

    ``key_pool`` and ``value_pool`` are one layer's pages flattened to
    ``num_slots x H_k x D``, the same view :func:`paged_attention` reads back. Writes in
    place, returns nothing. The oracle is the two-pass PyTorch it replaces — quantize into
    a temporary, then ``index_copy_`` — and the two agree bit for bit: both divide by the
    scale in fp32 and round to nearest even.
    """
    if _use_kernel("kv_quantize_scatter", use_cuda, key) and key_pool.dtype is FP8_KERNEL_DTYPE:
        load_extension().kv_quantize_scatter(
            key.contiguous(),
            value.contiguous(),
            key_pool,
            value_pool,
            slot_mapping.to(torch.int64),
            float(k_scale),
            float(v_scale),
        )
        return

    index = slot_mapping.to(device=key_pool.device, dtype=torch.int64)
    quant_keys = (key.float() / k_scale).to(key_pool.dtype)
    quant_values = (value.float() / v_scale).to(value_pool.dtype)
    key_pool.view(torch.uint8).index_copy_(0, index, quant_keys.view(torch.uint8))
    value_pool.view(torch.uint8).index_copy_(0, index, quant_values.view(torch.uint8))


def _kernel_can_page(
    q: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_tables: torch.Tensor,
) -> bool:
    """Whether the paged kernel can serve this call.

    The kernel walks the pools by slot arithmetic rather than by stride, so they must be
    contiguous: a narrowed view would read the wrong pages rather than read slowly, which
    is why this is a condition and not a copy.

    FP8 pools are served as well, dequantized in registers, so the pools may differ from
    the query in dtype provided they agree with each other and use a storage type the
    kernel knows.
    """
    pools_match = key_pool.dtype == value_pool.dtype
    pool_is_fp8 = key_pool.dtype is FP8_KERNEL_DTYPE
    dtype_ok = pools_match and (key_pool.dtype == q.dtype or pool_is_fp8)
    if q.dim() != 3 or not dtype_ok:
        return False
    if q.dtype == torch.float64 or q.shape[-1] > MAX_PAGED_HEAD_DIM:
        return False
    if not (q.is_contiguous() and key_pool.is_contiguous() and value_pool.is_contiguous()):
        return False
    return block_tables.dtype == torch.int32 and block_tables.is_contiguous()


def _kernel_can_attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | str | None,
    is_decode: bool,
) -> bool:
    """Whether the attention kernels can serve this exact call.

    Both kernels encode the causal structure in arithmetic rather than reading a mask
    tensor — decode attends to the whole cache, prefill compares each index against the
    diagonal — so `mask="causal"` is the only mask they can honour. An explicit tensor
    could express anything, so it falls back to the oracle rather than being ignored.
    """
    is_causal_shorthand = isinstance(mask, str) and mask == "causal"
    if is_decode:
        # A single query token may attend to every position cached before it, so the
        # causal mask forbids nothing and `None` requests the same thing.
        if mask is not None and not is_causal_shorthand:
            return False
    elif not is_causal_shorthand:
        # Prefill is the reverse: the kernel always masks causally, so it cannot serve
        # an unmasked multi-token call either.
        return False
    if q.dim() != 4 or q.dtype != k.dtype or q.dtype != v.dtype or q.dtype == torch.float64:
        return False
    if q.shape[-1] > MAX_KERNEL_HEAD_DIM or k.shape[-2] < q.shape[-2]:
        return False
    return k.shape[-2] > 0 and q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
