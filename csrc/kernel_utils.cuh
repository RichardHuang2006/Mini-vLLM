// Pieces every kernel in csrc/ needs: 16-byte vector loads and warp reductions.
//
// Shared between the kernels only. Nothing here is shared with `mini_vllm/`,
// because the PyTorch implementations there are the oracles these kernels are
// diffed against, and an oracle that shared code with the thing it validates
// would be comparing an implementation against itself (PLAN.md "Repo layout").

#pragma once

#include <cstdint>

namespace mini_vllm {

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

// A tensor view can start part way into its storage — `h[:, -1:, :]` is one the
// model actually produces — so 16-byte alignment is a property of a particular
// call rather than of the dtype, and has to be checked rather than assumed. A
// misaligned 128-bit load does not degrade, it faults.
inline bool is_vector_aligned(const void* pointer) {
  return reinterpret_cast<uintptr_t>(pointer) % kBytesPerVector == 0;
}

// Enough threads to cover `units` one each, capped at the block limit. Rounding
// up to a whole warp is required, not tidiness: the reductions below shuffle
// with a full 32-lane mask, so a block of, say, 100 threads would leave the last
// warp partially populated and its shuffle undefined.
inline int threads_for(int64_t units) {
  const int64_t rounded = ((units + kWarpSize - 1) / kWarpSize) * kWarpSize;
  const int64_t clamped = rounded < kWarpSize ? kWarpSize : rounded;
  return static_cast<int>(clamped > kMaxThreads ? kMaxThreads : clamped);
}

// Sum 32 lanes in five shuffles. The mask says all 32 lanes participate, so the
// caller must launch whole warps — `threads_for` is what guarantees it.
__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// The same tree, for a maximum. Needed by the online softmax in Step 3.4, where
// the running max and the running sum are reduced side by side.
__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

// Sum across the whole block: reduce within each warp by shuffling, then reduce
// the one partial per warp in the first warp. Only thread 0's result is valid,
// which is enough — the caller broadcasts it through shared memory.
//
// `scratch` is passed in rather than declared here so a kernel that reduces more
// than once (attention reduces a max and a sum) controls its own shared memory
// and the barriers around it.
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

}  // namespace mini_vllm
