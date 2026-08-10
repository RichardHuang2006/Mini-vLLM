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
from mini_vllm.positional_encoding import apply_rope

__all__ = [
    "CUDA_KERNELS",
    "attention",
    "cuda_kernel_names",
    "dispatch_report",
    "rmsnorm",
    "rope",
    "swiglu",
]

# Which ops have a hand-written kernel, and which step introduces it. Flipped to
# True by the Phase 3 step named beside it, one at a time.
CUDA_KERNELS: dict[str, tuple[bool, str]] = {
    "rmsnorm": (True, "Step 3.1"),
    "rope": (True, "Step 3.2"),
    "swiglu": (True, "Step 3.3"),
    "decode_attention": (True, "Step 3.4"),
    "prefill_attention": (False, "Step 3.5"),
}


def _use_kernel(name: str, use_cuda: bool, x: torch.Tensor) -> bool:
    """Whether ``name`` should run on the GPU for this call.

    A kernel needs three things to be true, and CUDA tensors is one of them: the
    same model object is used for CPU tests and GPU serving.
    """
    implemented, _step = CUDA_KERNELS[name]
    return use_cuda and implemented and x.is_cuda


def cuda_kernel_names() -> list[str]:
    """The ops that currently have a working kernel."""
    return [name for name, (implemented, _) in CUDA_KERNELS.items() if implemented]


def dispatch_report(use_cuda: bool) -> str:
    """One line per op saying which implementation a run would use."""
    lines = []
    for name, (implemented, step) in CUDA_KERNELS.items():
        if implemented and use_cuda:
            lines.append(f"  {name:<18} cuda    (kernel from {step})")
        elif implemented:
            lines.append(f"  {name:<18} torch   (kernel exists from {step}; use_cuda is off)")
        else:
            lines.append(f"  {name:<18} torch   (no kernel yet — {step})")
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
    name = "decode_attention" if is_decode else "prefill_attention"

    if _use_kernel(name, use_cuda, q) and _kernel_can_attend(q, k, v, mask, is_decode):
        scale = 1.0 / math.sqrt(q.shape[-1])
        if is_decode:
            return load_extension().decode_attention(q, k, v, scale)

    return reference_attention.scaled_dot_product_attention_grouped(q, k, v, mask=mask)


def _kernel_can_attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | str | None,
    is_decode: bool,
) -> bool:
    """Whether the attention kernels can serve this exact call.

    Both kernels bake in the causal structure rather than reading a mask tensor —
    decode attends to the whole cache, prefill masks arithmetically against the
    diagonal — so an explicit mask is the one thing they cannot honour. Falling
    back is right; silently ignoring it would not be.
    """
    # A single query token may attend to every cached position, so the causal
    # shorthand is not a restriction on it and the kernel's unmasked pass is
    # already correct. An explicit tensor could say anything, so it is not.
    is_causal_shorthand = isinstance(mask, str) and mask == "causal"
    if mask is not None and not (is_decode and is_causal_shorthand):
        return False
    if q.dim() != 4 or q.dtype != k.dtype or q.dtype != v.dtype or q.dtype == torch.float64:
        return False
    return k.shape[-2] > 0 and q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
