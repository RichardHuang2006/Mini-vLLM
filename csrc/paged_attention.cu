// Attention over a paged KV cache: the dense kernels with the block-table walk inside.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "kernel_utils.cuh"

namespace {

using namespace mini_vllm;

constexpr int kDecodeThreads = 128;   // four warps, so four keys are scored at once
constexpr int kDecodeTileKeys = 64;

constexpr int kPrefillWarps = 8;
constexpr int kPrefillThreads = kPrefillWarps * kWarpSize;  // one warp per query row
constexpr int kPrefillTileKeys = 16;                        // K and V tiles in shared memory

// Keys a warp scores per softmax update; 8 measured fastest of 1, 4, 8 and 16.
constexpr int kPrefillKeysPerStep = 8;

// The query and the accumulator live in registers, `head_dim / 32` floats per lane each.
constexpr int kMaxLanesPerThread = 8;
constexpr int kMaxHeadDim = kMaxLanesPerThread * kWarpSize;

// Splitting the key axis, for the one case a served batch misses: a long conversation.
constexpr int kMaxSplits = 32;
constexpr int kMinKeysPerSplit = 512;
constexpr int kBlocksPerSm = 4;

// Where logical key `position` physically lives, given one sequence's table.
__device__ __forceinline__ int64_t slot_of(const int32_t* __restrict__ table,
                                           const int64_t position,
                                           const int block_shift,
                                           const int block_mask) {
  const int64_t block = table[position >> block_shift];
  return (block << block_shift) | (position & block_mask);
}

// FP8 needs no change to the inner loops; the scales are hoisted, see README §15.

template <typename scalar_t, typename cache_t>
__global__ void paged_decode_kernel(const scalar_t* __restrict__ q,
                                    const cache_t* __restrict__ key_pool,
                                    const cache_t* __restrict__ value_pool,
                                    scalar_t* __restrict__ out,
                                    const int32_t* __restrict__ block_tables,
                                    const int32_t* __restrict__ cu_seqlens_q,
                                    const int32_t* __restrict__ context_lens,
                                    const int32_t* __restrict__ seq_lens,
                                    const int64_t num_query_heads,
                                    const int64_t num_kv_heads,
                                    const int64_t head_dim,
                                    const int64_t max_blocks,
                                    const int block_shift,
                                    const int block_mask,
                                    float* __restrict__ partial_out,
                                    float* __restrict__ partial_max,
                                    float* __restrict__ partial_sum,
                                    const int splits,
                                    const float scale,
                                    const float v_scale) {
  const int64_t sequence = blockIdx.y;
  if (seq_lens[sequence] != 1) {
    return;  // a prefill chunk; the other kernel has it
  }

  const int64_t query_head = blockIdx.x;
  const int split = blockIdx.z;
  const int64_t kv_head = query_head / (num_query_heads / num_kv_heads);
  const int64_t context = context_lens[sequence];
  const int64_t row = cu_seqlens_q[sequence];
  const int32_t* table = block_tables + sequence * max_blocks;

  // Cut on a tile boundary and per sequence, so a short one is divided by its own length.
  const int64_t tiles = (context + kDecodeTileKeys - 1) / kDecodeTileKeys;
  const int64_t tiles_per_split = (tiles + splits - 1) / splits;
  const int64_t begin = split * tiles_per_split * kDecodeTileKeys;
  const int64_t split_end = begin + tiles_per_split * kDecodeTileKeys;
  const int64_t end = split_end < context ? split_end : context;

  // query | output accumulator | tile scores | reduction scratch | broadcast slots
  extern __shared__ float shared[];
  float* query_shared = shared;
  float* accumulator = query_shared + head_dim;
  float* scores = accumulator + head_dim;
  float* scratch = scores + kDecodeTileKeys;
  float* reduced = scratch + kDecodeThreads / kWarpSize;

  // The tile's physical slots, resolved once per key for both loops below.
  __shared__ int64_t slots[kDecodeTileKeys];

  const int thread = threadIdx.x;
  const int lane = thread % kWarpSize;
  const int warp = thread / kWarpSize;
  constexpr int kWarps = kDecodeThreads / kWarpSize;

  const scalar_t* query = q + (row * num_query_heads + query_head) * head_dim;
  for (int64_t d = thread; d < head_dim; d += kDecodeThreads) {
    query_shared[d] = static_cast<float>(query[d]);
    accumulator[d] = 0.0f;
  }

  float running_max = -INFINITY;
  float running_sum = 0.0f;

  for (int64_t tile_start = begin; tile_start < end; tile_start += kDecodeTileKeys) {
    const int64_t remaining = end - tile_start;
    const int tile = static_cast<int>(remaining < kDecodeTileKeys ? remaining : kDecodeTileKeys);

    for (int j = thread; j < tile; j += kDecodeThreads) {
      slots[j] = slot_of(table, tile_start + j, block_shift, block_mask);
    }
    __syncthreads();

    // QKᵀ: one warp per key, lanes striding over the head dimension.
    for (int j = warp; j < tile; j += kWarps) {
      const cache_t* key = key_pool + (slots[j] * num_kv_heads + kv_head) * head_dim;
      float dot = 0.0f;
      for (int64_t d = lane; d < head_dim; d += kWarpSize) {
        dot += query_shared[d] * static_cast<float>(key[d]);
      }
      dot = warp_reduce_sum(dot);
      if (lane == 0) {
        scores[j] = dot * scale;
      }
    }
    __syncthreads();

    float tile_max = -INFINITY;
    for (int j = thread; j < tile; j += kDecodeThreads) {
      tile_max = fmaxf(tile_max, scores[j]);
    }
    tile_max = block_reduce_max(tile_max, scratch);
    if (thread == 0) {
      reduced[0] = tile_max;
    }
    __syncthreads();
    const float new_max = fmaxf(running_max, reduced[0]);

    float tile_sum = 0.0f;
    for (int j = thread; j < tile; j += kDecodeThreads) {
      const float weight = __expf(scores[j] - new_max);
      scores[j] = weight;
      tile_sum += weight;
    }
    tile_sum = block_reduce_sum(tile_sum, scratch);
    if (thread == 0) {
      reduced[1] = tile_sum;
    }
    __syncthreads();

    // On the first tile the correction is exp(-inf) == 0, zeroing an already-zero sum.
    const float correction = __expf(running_max - new_max);
    running_sum = running_sum * correction + reduced[1];

    // P·V: one thread per dimension, walking the tile's keys.
    for (int64_t d = thread; d < head_dim; d += kDecodeThreads) {
      float total = accumulator[d] * correction;
      for (int j = 0; j < tile; ++j) {
        total += scores[j] *
                 static_cast<float>(value_pool[(slots[j] * num_kv_heads + kv_head) * head_dim + d]);
      }
      accumulator[d] = total;
    }
    running_max = new_max;
    __syncthreads();  // the next tile overwrites `scores` and `slots`
  }

  if (partial_out == nullptr) {
    scalar_t* destination = out + (row * num_query_heads + query_head) * head_dim;
    for (int64_t d = thread; d < head_dim; d += kDecodeThreads) {
      // v_scale undoes the value quantization, and is 1.0 for a bf16 pool.
      destination[d] = static_cast<scalar_t>(accumulator[d] * v_scale / running_sum);
    }
    return;
  }

  // Split path: the merge needs every split's `l`, and an empty split reports -inf and 0.
  const int64_t slot = (sequence * num_query_heads + query_head) * splits + split;
  if (thread == 0) {
    partial_max[slot] = running_max;
    partial_sum[slot] = running_sum;
  }
  for (int64_t d = thread; d < head_dim; d += kDecodeThreads) {
    partial_out[slot * head_dim + d] = accumulator[d];
  }
}

// One block per (sequence, query head), applying the tile loop's rescaling one level up.
template <typename scalar_t>
__global__ void paged_decode_merge_kernel(const float* __restrict__ partial_out,
                                          const float* __restrict__ partial_max,
                                          const float* __restrict__ partial_sum,
                                          scalar_t* __restrict__ out,
                                          const int32_t* __restrict__ cu_seqlens_q,
                                          const int32_t* __restrict__ seq_lens,
                                          const int64_t num_query_heads,
                                          const int64_t head_dim,
                                          const int splits,
                                          const float v_scale) {
  const int64_t sequence = blockIdx.y;
  if (seq_lens[sequence] != 1) {
    // Its partials were never written; writing here would clobber the prefill's row.
    return;
  }

  const int64_t query_head = blockIdx.x;
  const int64_t row = cu_seqlens_q[sequence];
  const int64_t base = (sequence * num_query_heads + query_head) * splits;

  extern __shared__ float shared[];
  float* maxima = shared;
  float* sums = maxima + kMaxSplits;

  for (int s = threadIdx.x; s < splits; s += blockDim.x) {
    maxima[s] = partial_max[base + s];
    sums[s] = partial_sum[base + s];
  }
  __syncthreads();

  float global_max = -INFINITY;
  for (int s = 0; s < splits; ++s) {
    global_max = fmaxf(global_max, maxima[s]);
  }

  float total_sum = 0.0f;
  for (int s = 0; s < splits; ++s) {
    total_sum += __expf(maxima[s] - global_max) * sums[s];
  }

  scalar_t* destination = out + (row * num_query_heads + query_head) * head_dim;
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float total = 0.0f;
    for (int s = 0; s < splits; ++s) {
      total += __expf(maxima[s] - global_max) * partial_out[(base + s) * head_dim + d];
    }
    destination[d] = static_cast<scalar_t>(total * v_scale / total_sum);
  }
}

template <typename scalar_t, typename cache_t>
__global__ void paged_prefill_kernel(const scalar_t* __restrict__ q,
                                     const cache_t* __restrict__ key_pool,
                                     const cache_t* __restrict__ value_pool,
                                     scalar_t* __restrict__ out,
                                     const int32_t* __restrict__ block_tables,
                                     const int32_t* __restrict__ cu_seqlens_q,
                                     const int32_t* __restrict__ context_lens,
                                     const int32_t* __restrict__ seq_lens,
                                     const int64_t num_query_heads,
                                     const int64_t num_kv_heads,
                                     const int64_t head_dim,
                                     const int64_t max_blocks,
                                     const int block_shift,
                                     const int block_mask,
                                     const float scale,
                                     const float v_scale) {
  const int64_t sequence = blockIdx.z;
  const int query_len = seq_lens[sequence];
  if (query_len <= 1) {
    return;  // a decode step; the other kernel has it
  }

  const int rows_before = static_cast<int>(blockIdx.x) * kPrefillWarps;
  if (rows_before >= query_len) {
    return;  // this tile is past the end of a shorter sequence in the batch
  }

  const int64_t query_head = blockIdx.y;
  const int64_t kv_head = query_head / (num_query_heads / num_kv_heads);
  const int64_t context = context_lens[sequence];
  const int64_t row_base = cu_seqlens_q[sequence];
  const int32_t* table = block_tables + sequence * max_blocks;

  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int local_row = rows_before + warp;
  const bool active = local_row < query_len;

  // The reference's causal mask as a comparison: query i may see keys up to S - L + i.
  const int64_t offset = context - query_len;
  const int64_t visible = active ? offset + local_row + 1 : 0;

  // Block-uniform because the staging loop has barriers: the furthest row sets it.
  const int64_t last_row =
      min(static_cast<int64_t>(rows_before + kPrefillWarps),
          static_cast<int64_t>(query_len)) - 1;
  const int64_t block_visible = offset + last_row + 1;

  extern __shared__ float shared[];
  float* key_tile = shared;
  float* value_tile = key_tile + kPrefillTileKeys * head_dim;

  const int lanes = static_cast<int>((head_dim + kWarpSize - 1) / kWarpSize);
  float query_reg[kMaxLanesPerThread];
  float accumulator[kMaxLanesPerThread];
#pragma unroll
  for (int i = 0; i < kMaxLanesPerThread; ++i) {
    query_reg[i] = 0.0f;
    accumulator[i] = 0.0f;
  }
  if (active) {
    const scalar_t* query = q + ((row_base + local_row) * num_query_heads + query_head) * head_dim;
    for (int i = 0; i < lanes; ++i) {
      const int64_t d = i * kWarpSize + lane;
      if (d < head_dim) {
        query_reg[i] = static_cast<float>(query[d]);
      }
    }
  }

  float running_max = -INFINITY;
  float running_sum = 0.0f;

  for (int64_t tile_start = 0; tile_start < block_visible; tile_start += kPrefillTileKeys) {
    const int64_t remaining = block_visible - tile_start;
    const int tile = static_cast<int>(remaining < kPrefillTileKeys ? remaining : kPrefillTileKeys);

    // Stage the tile: one warp per key, so the gather coalesces and every warp reuses it.
    for (int j = warp; j < tile; j += kPrefillWarps) {
      const int64_t slot = slot_of(table, tile_start + j, block_shift, block_mask);
      const cache_t* key = key_pool + (slot * num_kv_heads + kv_head) * head_dim;
      const cache_t* value = value_pool + (slot * num_kv_heads + kv_head) * head_dim;
      for (int64_t d = lane; d < head_dim; d += kWarpSize) {
        key_tile[j * head_dim + d] = static_cast<float>(key[d]);
        value_tile[j * head_dim + d] = static_cast<float>(value[d]);
      }
    }
    __syncthreads();

    if (active) {
      // Keys ascend, so this row's tile ends at the first masked one.
      const int scoreable =
          static_cast<int>(min(static_cast<int64_t>(tile), max(visible - tile_start, int64_t{0})));

      for (int j = 0; j < scoreable; j += kPrefillKeysPerStep) {
        const int count = min(kPrefillKeysPerStep, scoreable - j);

        float dots[kPrefillKeysPerStep] = {};
        for (int i = 0; i < lanes; ++i) {
          const int64_t d = i * kWarpSize + lane;
          if (d < head_dim) {
            const float query_value = query_reg[i];
#pragma unroll
            for (int u = 0; u < kPrefillKeysPerStep; ++u) {
              if (u < count) {
                dots[u] += query_value * key_tile[(j + u) * head_dim + d];
              }
            }
          }
        }
        // Every lane needs every score, so the reductions are issued together.
#pragma unroll
        for (int u = 0; u < kPrefillKeysPerStep; ++u) {
          dots[u] = warp_all_reduce_sum(dots[u]) * scale;
        }

        float step_max = -INFINITY;
        for (int u = 0; u < count; ++u) {
          step_max = fmaxf(step_max, dots[u]);
        }
        const float new_max = fmaxf(running_max, step_max);
        const float correction = __expf(running_max - new_max);

        float weights[kPrefillKeysPerStep] = {};
        float step_sum = 0.0f;
        for (int u = 0; u < count; ++u) {
          weights[u] = __expf(dots[u] - new_max);
          step_sum += weights[u];
        }
        running_sum = running_sum * correction + step_sum;

        for (int i = 0; i < lanes; ++i) {
          const int64_t d = i * kWarpSize + lane;
          if (d < head_dim) {
            float total = accumulator[i] * correction;
            for (int u = 0; u < count; ++u) {
              total += weights[u] * value_tile[(j + u) * head_dim + d];
            }
            accumulator[i] = total;
          }
        }
        running_max = new_max;
      }
    }
    __syncthreads();  // the next tile overwrites what the loop above is reading
  }

  if (active) {
    scalar_t* destination =
        out + ((row_base + local_row) * num_query_heads + query_head) * head_dim;
    for (int i = 0; i < lanes; ++i) {
      const int64_t d = i * kWarpSize + lane;
      if (d < head_dim) {
        destination[d] = static_cast<scalar_t>(accumulator[i] * v_scale / running_sum);
      }
    }
  }
}

// Enough blocks to fill the card, never so many that a split is under kMinKeysPerSplit.
int splits_for(int64_t rows, int64_t context_len) {
  const int64_t affordable = (context_len + kMinKeysPerSplit - 1) / kMinKeysPerSplit;
  const int64_t wanted =
      (kBlocksPerSm * at::cuda::getCurrentDeviceProperties()->multiProcessorCount + rows - 1) /
      rows;
  const int64_t splits = affordable < wanted ? affordable : wanted;
  return static_cast<int>(splits < 1 ? 1 : (splits > kMaxSplits ? kMaxSplits : splits));
}

template <typename scalar_t, typename cache_t>
void launch_paged_attention(const at::Tensor& q,
                            const at::Tensor& key_pool,
                            const at::Tensor& value_pool,
                            at::Tensor& out,
                            const at::Tensor& block_tables,
                            const at::Tensor& cu_seqlens_q,
                            const at::Tensor& context_lens,
                            const at::Tensor& seq_lens,
                            const int64_t max_query_len,
                            const int64_t max_context_len,
                            const float scale,
                            const float k_scale,
                            const float v_scale) {
  // Fold the key scale into the softmax scale once, on the host.
  const float effective_scale = scale * k_scale;
  const int64_t num_sequences = seq_lens.size(0);
  const int64_t num_query_heads = q.size(1);
  const int64_t head_dim = q.size(2);
  const int64_t block_size = key_pool.size(1);
  const int64_t num_kv_heads = key_pool.size(2);
  const int64_t max_blocks = block_tables.size(1);

  int block_shift = 0;
  while ((1LL << block_shift) < block_size) {
    ++block_shift;
  }
  const int block_mask = static_cast<int>(block_size - 1);

  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t rows = num_sequences * num_query_heads;
  const int splits = splits_for(rows, max_context_len);

  at::Tensor partial_out, partial_max, partial_sum;
  if (splits > 1) {
    const auto scratch_options = q.options().dtype(at::kFloat);
    partial_out = torch::empty({rows * splits * head_dim}, scratch_options);
    partial_max = torch::empty({rows * splits}, scratch_options);
    partial_sum = torch::empty({rows * splits}, scratch_options);
  }

  const size_t decode_floats = 2 * head_dim + kDecodeTileKeys + kDecodeThreads / kWarpSize + 2;
  const dim3 decode_grid(static_cast<unsigned>(num_query_heads),
                         static_cast<unsigned>(num_sequences),
                         static_cast<unsigned>(splits));
  paged_decode_kernel<scalar_t, cache_t>
      <<<decode_grid, kDecodeThreads, decode_floats * sizeof(float), stream>>>(
          q.data_ptr<scalar_t>(),
          key_pool.data_ptr<cache_t>(),
          value_pool.data_ptr<cache_t>(),
          out.data_ptr<scalar_t>(),
          block_tables.data_ptr<int32_t>(),
          cu_seqlens_q.data_ptr<int32_t>(),
          context_lens.data_ptr<int32_t>(),
          seq_lens.data_ptr<int32_t>(),
          num_query_heads,
          num_kv_heads,
          head_dim,
          max_blocks,
          block_shift,
          block_mask,
          splits > 1 ? partial_out.data_ptr<float>() : nullptr,
          splits > 1 ? partial_max.data_ptr<float>() : nullptr,
          splits > 1 ? partial_sum.data_ptr<float>() : nullptr,
          splits,
          effective_scale,
          v_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (splits > 1) {
    const dim3 merge_grid(static_cast<unsigned>(num_query_heads),
                          static_cast<unsigned>(num_sequences));
    paged_decode_merge_kernel<scalar_t>
        <<<merge_grid, kDecodeThreads, 2 * kMaxSplits * sizeof(float), stream>>>(
            partial_out.data_ptr<float>(),
            partial_max.data_ptr<float>(),
            partial_sum.data_ptr<float>(),
            out.data_ptr<scalar_t>(),
            cu_seqlens_q.data_ptr<int32_t>(),
            seq_lens.data_ptr<int32_t>(),
            num_query_heads,
            head_dim,
            splits,
            v_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // Skipped for a pure-decode iteration, which `max_query_len` settles without a read.
  if (max_query_len > 1) {
    const int64_t query_tiles = (max_query_len + kPrefillWarps - 1) / kPrefillWarps;
    const size_t prefill_floats = 2 * kPrefillTileKeys * head_dim;
    const dim3 prefill_grid(static_cast<unsigned>(query_tiles),
                            static_cast<unsigned>(num_query_heads),
                            static_cast<unsigned>(num_sequences));
    paged_prefill_kernel<scalar_t, cache_t>
        <<<prefill_grid, kPrefillThreads, prefill_floats * sizeof(float), stream>>>(
            q.data_ptr<scalar_t>(),
            key_pool.data_ptr<cache_t>(),
            value_pool.data_ptr<cache_t>(),
            out.data_ptr<scalar_t>(),
            block_tables.data_ptr<int32_t>(),
            cu_seqlens_q.data_ptr<int32_t>(),
            context_lens.data_ptr<int32_t>(),
            seq_lens.data_ptr<int32_t>(),
            num_query_heads,
            num_kv_heads,
            head_dim,
            max_blocks,
            block_shift,
            block_mask,
            effective_scale,
            v_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

// Pick the storage type from the pool's dtype; `if constexpr` keeps nvcc's memory down.
template <typename scalar_t>
void dispatch_by_cache(const at::Tensor& q,
                       const at::Tensor& key_pool,
                       const at::Tensor& value_pool,
                       at::Tensor& out,
                       const at::Tensor& block_tables,
                       const at::Tensor& cu_seqlens_q,
                       const at::Tensor& context_lens,
                       const at::Tensor& seq_lens,
                       const int64_t max_query_len,
                       const int64_t max_context_len,
                       const float scale,
                       const float k_scale,
                       const float v_scale) {
  const auto pool_dtype = key_pool.scalar_type();
  auto run = [&](auto cache_tag) {
    using cache_t = decltype(cache_tag);
    launch_paged_attention<scalar_t, cache_t>(q, key_pool, value_pool, out, block_tables,
                                              cu_seqlens_q, context_lens, seq_lens,
                                              max_query_len, max_context_len, scale, k_scale,
                                              v_scale);
  };
  if (pool_dtype == q.scalar_type()) {
    run(scalar_t{});
    return;
  }
  if constexpr (!std::is_same_v<scalar_t, float>) {
    if (pool_dtype == at::kFloat8_e4m3fn) {
      run(at::Float8_e4m3fn{});
      return;
    }
  }
  TORCH_CHECK(false,
              "paged_attention: cannot pair a ",
              pool_dtype,
              " cache with a ",
              q.scalar_type(),
              " query");
}

}  // namespace

torch::Tensor paged_attention(const torch::Tensor& q,
                              const torch::Tensor& key_pool,
                              const torch::Tensor& value_pool,
                              const torch::Tensor& block_tables,
                              const torch::Tensor& cu_seqlens_q,
                              const torch::Tensor& context_lens,
                              const torch::Tensor& seq_lens,
                              int64_t max_query_len,
                              int64_t max_context_len,
                              double scale,
                              double k_scale,
                              double v_scale) {
  TORCH_CHECK(q.is_cuda() && key_pool.is_cuda() && value_pool.is_cuda(),
              "paged_attention: q and the pools must be CUDA tensors");
  TORCH_CHECK(q.dim() == 3,
              "paged_attention: expected q shaped T x H_q x D (the flattened token axis), got ",
              q.sizes());
  TORCH_CHECK(key_pool.dim() == 4 && key_pool.sizes() == value_pool.sizes(),
              "paged_attention: expected matching pools shaped num_blocks x P x H_k x D, got ",
              key_pool.sizes(),
              " and ",
              value_pool.sizes());
  TORCH_CHECK(q.is_contiguous() && key_pool.is_contiguous() && value_pool.is_contiguous(),
              "paged_attention: q and the pools must be contiguous; the kernel walks them "
              "by slot arithmetic rather than by stride");
  TORCH_CHECK(key_pool.scalar_type() == value_pool.scalar_type(),
              "paged_attention: the key and value pools must share a dtype, got ",
              key_pool.scalar_type(),
              " and ",
              value_pool.scalar_type());
  const bool pool_is_fp8 = key_pool.scalar_type() == at::kFloat8_e4m3fn;
  TORCH_CHECK(key_pool.scalar_type() == q.scalar_type() || pool_is_fp8,
              "paged_attention: the pools must either match q's dtype or be FP8 e4m3, got q ",
              q.scalar_type(),
              " and pools ",
              key_pool.scalar_type());
  TORCH_CHECK(!pool_is_fp8 || q.scalar_type() != at::kFloat,
              "paged_attention: an FP8 cache pairs with a reduced-precision query "
              "(bf16 or half), not fp32");
  TORCH_CHECK(q.scalar_type() != at::kDouble,
              "paged_attention: float64 is not supported; the accumulators here are fp32");
  TORCH_CHECK(q.size(2) == key_pool.size(3),
              "paged_attention: head dim mismatch, q has ",
              q.size(2),
              " and the pool has ",
              key_pool.size(3));
  TORCH_CHECK(q.size(2) <= kMaxHeadDim,
              "paged_attention: head dim ",
              q.size(2),
              " exceeds ",
              kMaxHeadDim,
              "; the query and the accumulator are held in registers");
  TORCH_CHECK(key_pool.size(2) > 0 && q.size(1) % key_pool.size(2) == 0,
              "paged_attention: H_q (",
              q.size(1),
              ") must be a multiple of H_k (",
              key_pool.size(2),
              ")");

  const int64_t block_size = key_pool.size(1);
  TORCH_CHECK(block_size > 0 && (block_size & (block_size - 1)) == 0,
              "paged_attention: the block size must be a power of two, got ",
              block_size);

  const int64_t num_sequences = seq_lens.size(0);
  TORCH_CHECK(block_tables.dim() == 2 && block_tables.size(0) == num_sequences,
              "paged_attention: expected a block table per sequence, got ",
              block_tables.sizes(),
              " for ",
              num_sequences,
              " sequences");
  TORCH_CHECK(context_lens.size(0) == num_sequences && cu_seqlens_q.size(0) == num_sequences + 1,
              "paged_attention: metadata disagrees on the sequence count");
  TORCH_CHECK(block_tables.scalar_type() == at::kInt && cu_seqlens_q.scalar_type() == at::kInt &&
                  context_lens.scalar_type() == at::kInt && seq_lens.scalar_type() == at::kInt,
              "paged_attention: the metadata tensors must be int32");
  TORCH_CHECK(block_tables.is_contiguous(),
              "paged_attention: block_tables must be contiguous");
  TORCH_CHECK(max_query_len >= 1,
              "paged_attention: max_query_len must be >= 1, got ",
              max_query_len);
  TORCH_CHECK(max_context_len >= max_query_len,
              "paged_attention: max_context_len (",
              max_context_len,
              ") must be at least max_query_len (",
              max_query_len,
              "); a pass cannot compute more tokens than it may attend over");

  at::Tensor out = torch::empty(q.sizes(), q.options());

  AT_DISPATCH_SWITCH(q.scalar_type(),
                     "paged_attention",
                     AT_DISPATCH_CASE(at::ScalarType::Float,
                                      [&] {
                                        dispatch_by_cache<scalar_t>(q,
                                                                    key_pool,
                                                                    value_pool,
                                                                    out,
                                                                    block_tables,
                                                                    cu_seqlens_q,
                                                                    context_lens,
                                                                    seq_lens,
                                                                    max_query_len,
                                                                    max_context_len,
                                                                    static_cast<float>(scale),
                                                                    static_cast<float>(k_scale),
                                                                    static_cast<float>(v_scale));
                                      })
                         AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES([&] {
                           dispatch_by_cache<scalar_t>(q,
                                                       key_pool,
                                                       value_pool,
                                                       out,
                                                       block_tables,
                                                       cu_seqlens_q,
                                                       context_lens,
                                                       seq_lens,
                                                       max_query_len,
                                                       max_context_len,
                                                       static_cast<float>(scale),
                                                       static_cast<float>(k_scale),
                                                       static_cast<float>(v_scale));
                         }));

  return out;
}
