"""Step 2.2 — the seam between the model and its kernels.

Every op the cached model performs goes through a function here, and each one can
run either the readable PyTorch version from Phase 1 or a hand-written CUDA kernel
from Phase 3. The model itself never learns which: it passes a `use_cuda` flag
down and this module decides.

The point is that Phase 3 becomes a sequence of small, independently verifiable
changes rather than one rewrite. Landing the RMSNorm kernel means writing
`csrc/rmsnorm.cu`, pointing `rmsnorm` at it, and flipping one entry in
:data:`CUDA_KERNELS` — the model file is not touched, and the PyTorch version
stays as the oracle to diff against.

`use_cuda=True` means "use kernels where they exist", so an op with no kernel yet
falls back silently rather than raising — and so does any op called on CPU
tensors, since the same model object serves CPU tests and GPU runs. Silence is
how you end up benchmarking a kernel that never ran, so :func:`dispatch_report`
says out loud which path each op would take, and the benchmark prints it.
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
    "rmsnorm",
    "rope",
    "swiglu",
]

# Which ops have a hand-written kernel, and which step introduces it. Flipped to
# True by the Phase 3 step named beside it, one at a time.
#
# The keys are the names the extension exports, not prettier synonyms for them:
# `test_every_claimed_kernel_is_callable` looks each one up in the compiled module,
# so a kernel cannot be claimed here and be missing, misspelled, or renamed.
CUDA_KERNELS: dict[str, tuple[bool, str]] = {
    "rmsnorm": (True, "Step 3.1"),
    "rope": (True, "Step 3.2"),
    "swiglu": (True, "Step 3.3"),
    "decode_attention": (True, "Step 3.4"),
    "flash_prefill": (True, "Step 3.5"),
    "paged_attention": (True, "Step 4.8"),
}

# Kernels that are correct, tested, and deliberately *not* the default, with the
# reason the benchmark gives. A kernel earns the dispatch by being faster; shipping
# one that is not would be a regression dressed up as progress, and hiding that
# behind `use_cuda=True` is exactly what `dispatch_report` exists to prevent.
#
# Prefill is here because its inner loop is scalar FMA against cuBLAS on tensor
# cores, and no amount of tuning closes that gap — Step 3.6 replaces the loop with
# `mma.sync` and is what flips this entry. Everything that tests the kernel calls it
# directly through the extension, so nothing about its coverage depends on routing.
NOT_YET_FASTER: dict[str, str] = {
    "flash_prefill": "0.4x cuBLAS at L=512 until Step 3.6",
}

# The prefill kernel stages a query, key and value tile in shared memory at once,
# which bounds the head dimension it can serve. Qwen3-0.6B uses 128; anything
# wider goes to the oracle rather than to a kernel that would refuse it.
MAX_KERNEL_HEAD_DIM = 192

# The paged kernel holds the query and the accumulator in registers instead, `D / 32`
# floats per lane of each, which is where its own ceiling comes from.
MAX_PAGED_HEAD_DIM = 256


def _use_kernel(name: str, use_cuda: bool, x: torch.Tensor) -> bool:
    """Whether ``name`` should run on the GPU for this call.

    A kernel needs three things to be true, and CUDA tensors is one of them: the
    same model object is used for CPU tests and GPU serving.
    """
    implemented, _step = CUDA_KERNELS[name]
    return use_cuda and implemented and name not in NOT_YET_FASTER and x.is_cuda


def cuda_kernel_names() -> list[str]:
    """The ops that currently have a working kernel."""
    return [name for name, (implemented, _) in CUDA_KERNELS.items() if implemented]


def dispatch_report(use_cuda: bool) -> str:
    """One line per op saying which implementation a run would use."""
    lines = []
    for name, (implemented, step) in CUDA_KERNELS.items():
        if not implemented:
            note = f"no kernel yet — {step}"
        elif name in NOT_YET_FASTER:
            note = f"kernel from {step} is slower: {NOT_YET_FASTER[name]}"
        elif not use_cuda:
            note = f"kernel exists from {step}; use_cuda is off"
        else:
            lines.append(f"  {name:<18} cuda    (kernel from {step})")
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
    """RMSNorm over the last dimension. Kernel: Step 3.1.

    The kernel requires ``weight`` to have the same dtype as ``x``; PyTorch would
    promote instead, and quietly returning a different dtype than the oracle
    would be worse than declining the kernel. The model never mixes them.
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
    """Rotary embedding at explicit positions. Kernel: Step 3.2.

    The kernel wants fp32 tables, which is what `RoPE` builds regardless of the
    activation dtype, and a head axis to broadcast the position across. Anything
    else is the oracle's problem.
    """
    if _use_kernel("rope", use_cuda, x) and cos.dtype == torch.float32 and x.dim() >= 3:
        return load_extension().rope(x, positions, cos, sin)
    return apply_rope(x, positions, cos, sin)


def swiglu(gate: torch.Tensor, up: torch.Tensor, use_cuda: bool = False) -> torch.Tensor:
    """``silu(gate) * up``, the elementwise half of the MLP. Kernel: Step 3.3.

    Takes the two projections rather than the input, because the projections are
    ordinary matmuls that cuBLAS already does better than we will. What is worth
    fusing is this part: two full passes over a `B x L x intermediate` tensor to
    do a few flops per element, which is pure memory traffic.
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
    """Grouped-query attention. Kernels: Step 3.4 (decode) and Step 3.5 (prefill).

    ::

        q:    B x H_q x L x D
        k, v: B x H_k x S x D
        out:  B x H_q x L x D

    Decode and prefill are split because they are different problems, not
    different sizes of one. Decode has `L = 1`: no parallelism across queries, one
    long pass over the cache, entirely memory-bound. Prefill has `L = S`: a real
    matrix multiply worth tiling. They get separate kernels for that reason, and
    the routing lives here so the model does not have to care.
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
) -> torch.Tensor:
    """Ragged attention over a paged cache. Kernel: Step 4.8.

    ::

        q:    T x H_q x D              every scheduled token, sequences concatenated
        pools num_blocks x P x H_k x D
        out:  T x H_q x D

    The fallback here is not the same kind of fallback as elsewhere. The PyTorch path
    for `rmsnorm` is a slower way to do the same work; the PyTorch path for this
    *copies every sequence's whole cache into a contiguous temporary first*, which is
    the traffic paging exists to remove. It is the oracle, not an alternative, and the
    engine's own `--mode throughput` numbers are only meaningful on the kernel path.

    `max_query_len` and `max_context_len` are passed as plain integers rather than
    read off `seq_lens` and `context_lens` on the device. They decide a grid shape and
    a split count, so reading them from a tensor would mean a device-to-host
    synchronization on the critical path of every iteration — and the caller already
    knows them, because it built the batch out of Python integers.
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
        )

    return reference_paged_attention(
        q, key_pool, value_pool, block_tables, cu_seqlens_q, context_lens, scale
    )


def _kernel_can_page(
    q: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_tables: torch.Tensor,
) -> bool:
    """Whether the paged kernel can serve this call.

    The kernel walks the pools by slot arithmetic rather than by stride, so it needs
    them contiguous — a narrowed view would have it reading the wrong pages rather
    than reading slowly, which is why this is a condition and not a copy.
    """
    if q.dim() != 3 or q.dtype != key_pool.dtype or q.dtype != value_pool.dtype:
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

    Both kernels bake the causal structure into arithmetic rather than reading a
    mask tensor — decode attends to the whole cache, prefill compares each index
    against the diagonal — so `mask="causal"` is the only mask they can honour. An
    explicit tensor could say anything, and falling back to the oracle is the
    honest response to it; silently ignoring it would not be.
    """
    is_causal_shorthand = isinstance(mask, str) and mask == "causal"
    if is_decode:
        # A single query token may attend to every position cached before it, so
        # the causal mask forbids it nothing and `None` asks for the same thing.
        if mask is not None and not is_causal_shorthand:
            return False
    elif not is_causal_shorthand:
        # Prefill is the other way round: the kernel *always* masks causally, so
        # an unmasked multi-token call is one it cannot serve either.
        return False
    if q.dim() != 4 or q.dtype != k.dtype or q.dtype != v.dtype or q.dtype == torch.float64:
        return False
    if q.shape[-1] > MAX_KERNEL_HEAD_DIM or k.shape[-2] < q.shape[-2]:
        return False
    return k.shape[-2] > 0 and q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
