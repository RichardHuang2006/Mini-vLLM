// Step 0.1 — axpby: y = a*x + b.
//
// This kernel is deliberately trivial. Its only job is to prove that the whole
// extension pipeline works before anything depends on it: nvcc finds a CUDA
// version matching torch, the sm_120 target compiles, pybind11 exposes the
// symbol, and a launch on a real tensor produces the right numbers.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

// Accumulate in float even for half/bfloat16 inputs. Every kernel in this
// project follows that rule (see DESIGN.md §5.3); starting here keeps it
// habitual rather than something remembered later under pressure.
template <typename scalar_t>
__global__ void axpby_kernel(const scalar_t* __restrict__ x,
                             scalar_t* __restrict__ y,
                             const float a,
                             const float b,
                             const int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = static_cast<scalar_t>(a * static_cast<float>(x[i]) + b);
  }
}

}  // namespace

torch::Tensor hello(const torch::Tensor& x, double a, double b) {
  TORCH_CHECK(x.is_cuda(), "hello: x must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
                  x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kDouble,
              "hello: unsupported dtype ", x.scalar_type());

  const at::Tensor xc = x.contiguous();
  at::Tensor y = torch::empty_like(xc);

  const int64_t n = xc.numel();
  if (n == 0) {
    return y;
  }

  constexpr int kThreads = 256;
  const int64_t blocks = (n + kThreads - 1) / kThreads;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, xc.scalar_type(), "hello", [&] {
        axpby_kernel<scalar_t><<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            xc.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            static_cast<float>(a),
            static_cast<float>(b),
            n);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return y;
}
