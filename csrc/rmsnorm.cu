// Step 3.1 — RMSNorm: out = x * rsqrt(mean(x^2) + eps) * weight, over the last dim.
//
// The oracle is `rms_norm` in mini_vllm/layer_norm.py, and this kernel is only
// correct insofar as it agrees with it. Two details of that function are load
// bearing and easy to get wrong here:
//
//   * the mean of squares accumulates in fp32 even when the tensor is bf16, and
//   * the normalized value is rounded back to the input dtype *before* the
//     weight multiply, which is what HuggingFace does.
//
// The shape of the kernel — one block per row, a warp-shuffle reduction for the
// statistic, 16-byte vectorized loads — is the pattern every later elementwise
// kernel in the project reuses (DESIGN.md §5.1). The op is memory-bound, so the
// number worth reporting is achieved bandwidth against the card's peak, not
// wall-clock; `python -m mini_vllm.bench --mode kernels` prints it.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxThreads = 1024;

// 16 bytes is the widest load a thread can issue (LDG.E.128), so it sets the
// vector width: 8 bf16/fp16 lanes, 4 fp32, 2 fp64.
constexpr int kBytesPerVector = 16;

template <typename scalar_t>
struct alignas(kBytesPerVector) Vector {
  static constexpr int kLanes = kBytesPerVector / sizeof(scalar_t);
  scalar_t lane[kLanes];
};

// The un-vectorized fallback, for widths that are not a multiple of the vector
// width or pointers that are not 16-byte aligned. Same code path, one lane wide.
template <typename scalar_t>
struct Scalar {
  static constexpr int kLanes = 1;
  scalar_t lane[1];
};

// Sum 32 lanes in five shuffles. The mask says all 32 lanes participate, so the
// caller must launch whole warps — `threads_for` below is what guarantees it.
// A partial warp here is not slow, it is undefined.
__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// Sum across the whole block: reduce within each warp by shuffling, then reduce
// the one partial per warp in the first warp. Only thread 0's result is valid,
// which is enough — the caller broadcasts it through shared memory.
__device__ __forceinline__ float block_reduce_sum(float value) {
  __shared__ float warp_totals[kMaxThreads / kWarpSize];

  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int warps = (blockDim.x + kWarpSize - 1) / kWarpSize;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    warp_totals[warp] = value;
  }
  __syncthreads();

  value = (threadIdx.x < static_cast<unsigned>(warps)) ? warp_totals[threadIdx.x] : 0.0f;
  return warp == 0 ? warp_reduce_sum(value) : 0.0f;
}

// One block per row. `chunk_t` is either Vector<scalar_t> or Scalar<scalar_t>,
// which is the only difference between the fast and the fallback launch.
template <typename scalar_t, typename chunk_t>
__global__ void rmsnorm_kernel(const scalar_t* __restrict__ input,
                               const scalar_t* __restrict__ weight,
                               scalar_t* __restrict__ out,
                               const int64_t dim,
                               const float eps) {
  constexpr int kLanes = chunk_t::kLanes;
  const int64_t row = blockIdx.x;
  const int64_t chunks = dim / kLanes;

  const chunk_t* row_in = reinterpret_cast<const chunk_t*>(input + row * dim);
  chunk_t* row_out = reinterpret_cast<chunk_t*>(out + row * dim);
  const chunk_t* row_weight = reinterpret_cast<const chunk_t*>(weight);

  // Normalizing needs every element twice: once for the statistic, once to
  // scale. When the row fits in one chunk per thread — true for every width
  // Qwen3 uses, `E` = 1024 and `D` = 128 — the first read is kept in registers
  // and the second pass costs no memory traffic at all. Wider rows fall back to
  // re-reading, which is an L1 hit rather than a trip to DRAM.
  const bool resident = chunks <= static_cast<int64_t>(blockDim.x);
  chunk_t held;

  float sum_squares = 0.0f;
  for (int64_t i = threadIdx.x; i < chunks; i += blockDim.x) {
    const chunk_t chunk = row_in[i];
    if (resident) {
      held = chunk;
    }
#pragma unroll
    for (int j = 0; j < kLanes; ++j) {
      const float value = static_cast<float>(chunk.lane[j]);
      sum_squares += value * value;
    }
  }

  __shared__ float inverse_rms;
  const float total = block_reduce_sum(sum_squares);
  if (threadIdx.x == 0) {
    inverse_rms = rsqrtf(total / static_cast<float>(dim) + eps);
  }
  __syncthreads();

  const float scale = inverse_rms;
  for (int64_t i = threadIdx.x; i < chunks; i += blockDim.x) {
    const chunk_t chunk = resident ? held : row_in[i];
    const chunk_t weights = row_weight[i];

    chunk_t result;
#pragma unroll
    for (int j = 0; j < kLanes; ++j) {
      // The round to scalar_t here, before the weight multiply, is not
      // incidental: it is where the oracle's `weight * normalized.to(dtype)`
      // loses its low bits, and skipping it would leave this kernel very
      // slightly *more* accurate than the reference it must match.
      const float scaled = static_cast<float>(chunk.lane[j]) * scale;
      const scalar_t normalized = static_cast<scalar_t>(scaled);
      const float weighted = static_cast<float>(normalized) * static_cast<float>(weights.lane[j]);
      result.lane[j] = static_cast<scalar_t>(weighted);
    }
    row_out[i] = result;
  }
}

// Enough threads to cover the row one chunk each, capped at the block limit.
// Rounding up to a whole warp is required, not tidiness: the reduction shuffles
// with a full 32-lane mask, so a block of, say, 100 threads would leave the last
// warp partially populated and its shuffle undefined.
int threads_for(int64_t chunks) {
  const int64_t rounded = ((chunks + kWarpSize - 1) / kWarpSize) * kWarpSize;
  return static_cast<int>(std::min<int64_t>(std::max<int64_t>(rounded, kWarpSize), kMaxThreads));
}

bool is_vector_aligned(const void* pointer) {
  return reinterpret_cast<uintptr_t>(pointer) % kBytesPerVector == 0;
}

template <typename scalar_t>
void launch_rmsnorm(const at::Tensor& input,
                    const at::Tensor& weight,
                    at::Tensor& out,
                    int64_t rows,
                    int64_t dim,
                    float eps) {
  const scalar_t* input_pointer = input.data_ptr<scalar_t>();
  const scalar_t* weight_pointer = weight.data_ptr<scalar_t>();
  scalar_t* out_pointer = out.data_ptr<scalar_t>();

  // A tensor view can start part way into its storage — `h[:, -1:, :]` is one
  // the model actually produces — so 16-byte alignment is a property of this
  // call rather than of the dtype, and has to be checked rather than assumed.
  // A misaligned 128-bit load does not degrade, it faults.
  constexpr int kLanes = Vector<scalar_t>::kLanes;
  const bool vectorizable = dim % kLanes == 0 && is_vector_aligned(input_pointer) &&
                            is_vector_aligned(weight_pointer) && is_vector_aligned(out_pointer);

  const auto stream = at::cuda::getCurrentCUDAStream();
  if (vectorizable) {
    rmsnorm_kernel<scalar_t, Vector<scalar_t>><<<rows, threads_for(dim / kLanes), 0, stream>>>(
        input_pointer, weight_pointer, out_pointer, dim, eps);
  } else {
    rmsnorm_kernel<scalar_t, Scalar<scalar_t>><<<rows, threads_for(dim), 0, stream>>>(
        input_pointer, weight_pointer, out_pointer, dim, eps);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor rmsnorm(const torch::Tensor& x, const torch::Tensor& weight, double eps) {
  TORCH_CHECK(x.is_cuda(), "rmsnorm: x must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "rmsnorm: weight must be a CUDA tensor");
  TORCH_CHECK(x.dim() >= 1, "rmsnorm: x must have at least one dimension");
  TORCH_CHECK(weight.dim() == 1, "rmsnorm: weight must be 1-D, got ", weight.dim(), " dimensions");
  TORCH_CHECK(x.size(-1) == weight.size(0),
              "rmsnorm: x's last dimension is ",
              x.size(-1),
              " but weight has ",
              weight.size(0),
              " elements");
  // fp64 is refused rather than accumulated in fp32 like the rest. Halving the
  // precision of a tensor that asked for double, silently, is a worse outcome
  // than not running: the PyTorch oracle handles it exactly and the model never
  // uses it.
  TORCH_CHECK(x.scalar_type() != at::kDouble,
              "rmsnorm: float64 is not supported; this kernel accumulates in fp32, "
              "which would silently lose precision. Use the PyTorch path.");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(),
              "rmsnorm: x is ",
              x.scalar_type(),
              " but weight is ",
              weight.scalar_type(),
              "; the PyTorch path promotes mixed dtypes, this kernel does not");

  const at::Tensor input = x.contiguous();
  const at::Tensor weights = weight.contiguous();
  at::Tensor out = torch::empty_like(input);

  const int64_t dim = input.size(-1);
  const int64_t rows = dim == 0 ? 0 : input.numel() / dim;
  if (rows == 0 || dim == 0) {
    return out;
  }

  AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "rmsnorm",
      AT_DISPATCH_CASE(at::ScalarType::Float,
                       [&] {
                         launch_rmsnorm<scalar_t>(
                             input, weights, out, rows, dim, static_cast<float>(eps));
                       })
          AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES([&] {
            launch_rmsnorm<scalar_t>(input, weights, out, rows, dim, static_cast<float>(eps));
          }));

  return out;
}
