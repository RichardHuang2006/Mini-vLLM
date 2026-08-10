// Step 3.3 — SwiGLU's elementwise half: out = silu(gate) * up.
//
// The two projections stay on cuBLAS. What is worth fusing is this part: three
// full passes over a `(B·L) x intermediate` tensor (sigmoid, then two multiplies)
// to do four flops per element, which is pure memory traffic. Qwen3-0.6B's
// intermediate width is 3072 — three times the hidden size — so this is the
// widest activation in the model.
//
// The oracle is `silu(gate) * up` in mini_vllm/kernels/ops.py, built from
// `mini_vllm.basics.silu`. Matching it means rounding where PyTorch rounds: it
// evaluates three separate ops, each landing back in the input dtype, so a kernel
// that carried fp32 all the way to the store would be *more* accurate than the
// reference and the differential test could not tell that from a bug.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "kernel_utils.cuh"

namespace {

using namespace mini_vllm;

constexpr int kThreads = 256;

// Pure elementwise, so there is no row structure to respect: one thread per
// 16-byte chunk of a flat buffer, which is the shape that saturates bandwidth.
template <typename scalar_t, typename chunk_t>
__global__ void swiglu_kernel(const scalar_t* __restrict__ gate,
                              const scalar_t* __restrict__ up,
                              scalar_t* __restrict__ out,
                              const int64_t chunks) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= chunks) {
    return;
  }

  constexpr int kLanes = chunk_t::kLanes;
  const chunk_t gate_chunk = reinterpret_cast<const chunk_t*>(gate)[index];
  const chunk_t up_chunk = reinterpret_cast<const chunk_t*>(up)[index];

  chunk_t result;
#pragma unroll
  for (int j = 0; j < kLanes; ++j) {
    const float g = static_cast<float>(gate_chunk.lane[j]);

    // Three roundings, in the same three places PyTorch does them: after the
    // sigmoid, after the silu multiply, and on the store. `expf` rather than
    // `__expf`, because the fast intrinsic is a different function and would
    // disagree in the last bits for no benefit on a memory-bound kernel.
    const scalar_t sigmoid = static_cast<scalar_t>(1.0f / (1.0f + expf(-g)));
    const scalar_t activated = static_cast<scalar_t>(g * static_cast<float>(sigmoid));
    result.lane[j] =
        static_cast<scalar_t>(static_cast<float>(activated) * static_cast<float>(up_chunk.lane[j]));
  }
  reinterpret_cast<chunk_t*>(out)[index] = result;
}

template <typename scalar_t>
void launch_swiglu(const at::Tensor& gate, const at::Tensor& up, at::Tensor& out) {
  const scalar_t* gate_pointer = gate.data_ptr<scalar_t>();
  const scalar_t* up_pointer = up.data_ptr<scalar_t>();
  scalar_t* out_pointer = out.data_ptr<scalar_t>();

  constexpr int kLanes = Vector<scalar_t>::kLanes;
  const int64_t elements = gate.numel();
  const bool vectorizable = elements % kLanes == 0 && is_vector_aligned(gate_pointer) &&
                            is_vector_aligned(up_pointer) && is_vector_aligned(out_pointer);

  const int64_t chunks = vectorizable ? elements / kLanes : elements;
  const int64_t blocks = (chunks + kThreads - 1) / kThreads;
  const auto stream = at::cuda::getCurrentCUDAStream();

  if (vectorizable) {
    swiglu_kernel<scalar_t, Vector<scalar_t>>
        <<<blocks, kThreads, 0, stream>>>(gate_pointer, up_pointer, out_pointer, chunks);
  } else {
    swiglu_kernel<scalar_t, Scalar<scalar_t>>
        <<<blocks, kThreads, 0, stream>>>(gate_pointer, up_pointer, out_pointer, chunks);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor swiglu(const torch::Tensor& gate, const torch::Tensor& up) {
  TORCH_CHECK(gate.is_cuda() && up.is_cuda(), "swiglu: gate and up must be CUDA tensors");
  TORCH_CHECK(gate.sizes() == up.sizes(),
              "swiglu: gate is ",
              gate.sizes(),
              " but up is ",
              up.sizes(),
              "; they are the two projections of the same tokens and must match");
  TORCH_CHECK(gate.scalar_type() == up.scalar_type(),
              "swiglu: gate is ",
              gate.scalar_type(),
              " but up is ",
              up.scalar_type());
  TORCH_CHECK(gate.scalar_type() != at::kDouble,
              "swiglu: float64 is not supported; this kernel evaluates the sigmoid in fp32. "
              "Use the PyTorch path.");

  const at::Tensor gate_input = gate.contiguous();
  const at::Tensor up_input = up.contiguous();
  at::Tensor out = torch::empty_like(gate_input);

  if (gate_input.numel() == 0) {
    return out;
  }

  AT_DISPATCH_SWITCH(gate_input.scalar_type(),
                     "swiglu",
                     AT_DISPATCH_CASE(at::ScalarType::Float,
                                      [&] { launch_swiglu<scalar_t>(gate_input, up_input, out); })
                         AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES(
                             [&] { launch_swiglu<scalar_t>(gate_input, up_input, out); }));

  return out;
}
