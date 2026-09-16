// Pieces every kernel in csrc/ needs: 16-byte vector loads and warp reductions.

#pragma once

#include <cstdint>

namespace mini_vllm {

constexpr int kWarpSize = 32;
constexpr int kMaxThreads = 1024;

// 16 bytes is the widest load a thread can issue (LDG.E.128): 8 bf16 lanes, 4 fp32.
constexpr int kBytesPerVector = 16;

template <typename scalar_t>
struct alignas(kBytesPerVector) Vector {
  static constexpr int kLanes = kBytesPerVector / sizeof(scalar_t);
  scalar_t lane[kLanes];
};

// The un-vectorized fallback for unaligned pointers or ragged widths: one lane wide.
template <typename scalar_t>
struct Scalar {
  static constexpr int kLanes = 1;
  scalar_t lane[1];
};

// A view can start part way into its storage, and a misaligned 128-bit load faults.
inline bool is_vector_aligned(const void* pointer) {
  return reinterpret_cast<uintptr_t>(pointer) % kBytesPerVector == 0;
}

// Enough threads to cover `units`, in whole warps: the shuffles use a full 32-lane mask.
inline int threads_for(int64_t units) {
  const int64_t rounded = ((units + kWarpSize - 1) / kWarpSize) * kWarpSize;
  const int64_t clamped = rounded < kWarpSize ? kWarpSize : rounded;
  return static_cast<int>(clamped > kMaxThreads ? kMaxThreads : clamped);
}

// Sum 32 lanes in five shuffles; only lane 0's result is meaningful.
__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// The butterfly variant: every lane comes back with the total, not just lane 0.
__device__ __forceinline__ float warp_all_reduce_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

// The same tree, for the running maximum of the online softmax.
__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

// One float of `scratch` per warp; result in thread 0 only; __syncthreads() between calls.
__device__ __forceinline__ float block_reduce_sum(float value, float* scratch) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int warps = (blockDim.x + kWarpSize - 1) / kWarpSize;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();

  value = (threadIdx.x < static_cast<unsigned>(warps)) ? scratch[threadIdx.x] : 0.0f;
  return warp == 0 ? warp_reduce_sum(value) : 0.0f;
}

__device__ __forceinline__ float block_reduce_max(float value, float* scratch) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int warps = (blockDim.x + kWarpSize - 1) / kWarpSize;

  value = warp_reduce_max(value);
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();

  // The identity for a max is -inf: zero would win over an all-negative block.
  value = (threadIdx.x < static_cast<unsigned>(warps)) ? scratch[threadIdx.x] : -INFINITY;
  return warp == 0 ? warp_reduce_max(value) : -INFINITY;
}

}  // namespace mini_vllm
