// Decode attention: one query token against the cache, by the README §14 recurrence.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "kernel_utils.cuh"

namespace {

using namespace mini_vllm;

constexpr int kThreads = 128;  // four warps, so four keys are scored at a time
constexpr int kTileKeys = 64;  // scores per tile; only 256 B of shared memory
constexpr int kMaxHeadDim = 1024;

// How the key axis is divided when one block per (sequence, head) does not fill the card.
constexpr int kMaxSplits = 32;
constexpr int kMinKeysPerSplit = 512;
constexpr int kBlocksPerSm = 4;

// Where a tensor's (batch, head, position) axes live, so a narrowed cache is read in place.
struct Strides {
  int64_t batch;
  int64_t head;
  int64_t position;
};

template <typename scalar_t>
__global__ void decode_attention_kernel(const scalar_t* __restrict__ q,
                                        const scalar_t* __restrict__ k,
                                        const scalar_t* __restrict__ v,
                                        scalar_t* __restrict__ out,
                                        float* __restrict__ partial_out,
                                        float* __restrict__ partial_max,
                                        float* __restrict__ partial_sum,
                                        const Strides q_strides,
                                        const Strides k_strides,
                                        const Strides v_strides,
                                        const Strides out_strides,
                                        const int64_t source_len,
                                        const int64_t head_dim,
                                        const int64_t group_size,
                                        const int splits,
                                        const float scale) {
  const int64_t query_head = blockIdx.x;
  const int64_t sequence = blockIdx.y;
  const int split = blockIdx.z;
  const int64_t kv_head = query_head / group_size;

  // Cut on a tile boundary, so no split masks a partial tile mid-cache.
  const int64_t tiles = (source_len + kTileKeys - 1) / kTileKeys;
  const int64_t tiles_per_split = (tiles + splits - 1) / splits;
  const int64_t begin = split * tiles_per_split * kTileKeys;
  const int64_t split_end = begin + tiles_per_split * kTileKeys;
  const int64_t end = split_end < source_len ? split_end : source_len;

  const scalar_t* query = q + sequence * q_strides.batch + query_head * q_strides.head;
  const scalar_t* keys = k + sequence * k_strides.batch + kv_head * k_strides.head;
  const scalar_t* values = v + sequence * v_strides.batch + kv_head * v_strides.head;

  // query | output accumulator | tile scores | reduction scratch | broadcast slot
  extern __shared__ float shared[];
  float* query_shared = shared;
  float* accumulator = query_shared + head_dim;
  float* scores = accumulator + head_dim;
  float* scratch = scores + kTileKeys;
  float* reduced = scratch + kThreads / kWarpSize;

  const int thread = threadIdx.x;
  const int lane = thread % kWarpSize;
  const int warp = thread / kWarpSize;
  constexpr int kWarps = kThreads / kWarpSize;

  // The query is read once per key, so it is staged in shared memory in fp32.
  for (int d = thread; d < head_dim; d += kThreads) {
    query_shared[d] = static_cast<float>(query[d]);
    accumulator[d] = 0.0f;
  }
  __syncthreads();

  float running_max = -INFINITY;
  float running_sum = 0.0f;

  for (int64_t tile_start = begin; tile_start < end; tile_start += kTileKeys) {
    const int64_t remaining = end - tile_start;
    const int tile = static_cast<int>(remaining < kTileKeys ? remaining : kTileKeys);

    // QKᵀ: one warp per key, lanes striding over the head dimension.
    for (int j = warp; j < tile; j += kWarps) {
      const scalar_t* key = keys + (tile_start + j) * k_strides.position;
      float dot = 0.0f;
      for (int d = lane; d < head_dim; d += kWarpSize) {
        dot += query_shared[d] * static_cast<float>(key[d]);
      }
      dot = warp_reduce_sum(dot);
      if (lane == 0) {
        // Scaled after the reduction, matching the oracle's `(q @ kᵀ) * scale`.
        scores[j] = dot * scale;
      }
    }
    __syncthreads();

    // The max over this tile, and the new running max.
    float tile_max = -INFINITY;
    for (int j = thread; j < tile; j += kThreads) {
      tile_max = fmaxf(tile_max, scores[j]);
    }
    tile_max = block_reduce_max(tile_max, scratch);
    if (thread == 0) {
      reduced[0] = tile_max;
    }
    // Also separates the two reductions: warp 0 is still reading `scratch`.
    __syncthreads();
    const float new_max = fmaxf(running_max, reduced[0]);

    // Exponentiate in place, and sum this tile's weights.
    float tile_sum = 0.0f;
    for (int j = thread; j < tile; j += kThreads) {
      const float weight = __expf(scores[j] - new_max);
      scores[j] = weight;
      tile_sum += weight;
    }
    tile_sum = block_reduce_sum(tile_sum, scratch);
    if (thread == 0) {
      reduced[1] = tile_sum;
    }
    // Also publishes every `scores[j]` written above to the P·V loop below.
    __syncthreads();

    // On the first tile the correction is exp(-inf) == 0, zeroing an already-zero sum.
    const float correction = __expf(running_max - new_max);
    running_sum = running_sum * correction + reduced[1];

    // P·V: one thread per dimension, walking the tile's keys.
    for (int d = thread; d < head_dim; d += kThreads) {
      float total = accumulator[d] * correction;
      for (int j = 0; j < tile; ++j) {
        total += scores[j] * static_cast<float>(values[(tile_start + j) * v_strides.position + d]);
      }
      accumulator[d] = total;
    }
    running_max = new_max;

    // The next tile overwrites `scores`, which the loop above is still reading.
    __syncthreads();
  }

  // The division by `l` waits until the end: no per-tile sum is available.
  if (partial_out == nullptr) {
    scalar_t* destination = out + sequence * out_strides.batch + query_head * out_strides.head;
    for (int d = thread; d < head_dim; d += kThreads) {
      destination[d] = static_cast<scalar_t>(accumulator[d] / running_sum);
    }
    return;
  }

  // Split path: the merge needs every split's `l` before it can divide.
  const int64_t slot = (sequence * gridDim.x + query_head) * splits + split;
  if (thread == 0) {
    partial_max[slot] = running_max;
    partial_sum[slot] = running_sum;
  }
  for (int d = thread; d < head_dim; d += kThreads) {
    partial_out[slot * head_dim + d] = accumulator[d];
  }
}

// One block per (sequence, query head), applying the tile loop's rescaling one level up.
template <typename scalar_t>
__global__ void decode_attention_merge_kernel(const float* __restrict__ partial_out,
                                              const float* __restrict__ partial_max,
                                              const float* __restrict__ partial_sum,
                                              scalar_t* __restrict__ out,
                                              const Strides out_strides,
                                              const int64_t head_dim,
                                              const int splits) {
  const int64_t query_head = blockIdx.x;
  const int64_t sequence = blockIdx.y;
  const int64_t base = (sequence * gridDim.x + query_head) * splits;

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
    // An empty trailing split contributes exp(-inf) * 0 == 0.
    total_sum += __expf(maxima[s] - global_max) * sums[s];
  }

  scalar_t* destination = out + sequence * out_strides.batch + query_head * out_strides.head;
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float total = 0.0f;
    for (int s = 0; s < splits; ++s) {
      total += __expf(maxima[s] - global_max) * partial_out[(base + s) * head_dim + d];
    }
    destination[d] = static_cast<scalar_t>(total / total_sum);
  }
}

Strides strides_of(const at::Tensor& tensor) {
  return Strides{tensor.stride(0), tensor.stride(1), tensor.stride(2)};
}

// Enough blocks to fill the card, never so many that a split is under kMinKeysPerSplit.
int splits_for(int64_t rows, int64_t source_len) {
  const int64_t affordable = (source_len + kMinKeysPerSplit - 1) / kMinKeysPerSplit;
  const int64_t wanted =
      (kBlocksPerSm * at::cuda::getCurrentDeviceProperties()->multiProcessorCount + rows - 1) / rows;
  const int64_t splits = affordable < wanted ? affordable : wanted;
  return static_cast<int>(splits < 1 ? 1 : (splits > kMaxSplits ? kMaxSplits : splits));
}

template <typename scalar_t>
void launch_decode_attention(const at::Tensor& q,
                             const at::Tensor& k,
                             const at::Tensor& v,
                             at::Tensor& out,
                             const float scale) {
  const int64_t batch = q.size(0);
  const int64_t num_query_heads = q.size(1);
  const int64_t head_dim = q.size(3);
  const int64_t source_len = k.size(2);
  const int64_t group_size = num_query_heads / k.size(1);
  const int64_t rows = batch * num_query_heads;

  const int splits = splits_for(rows, source_len);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto scratch_options = q.options().dtype(at::kFloat);

  // Left undefined on the single-split path, where null means normalize and store.
  at::Tensor partial_out, partial_max, partial_sum;
  if (splits > 1) {
    partial_out = torch::empty({rows * splits * head_dim}, scratch_options);
    partial_max = torch::empty({rows * splits}, scratch_options);
    partial_sum = torch::empty({rows * splits}, scratch_options);
  }

  const size_t shared_floats = 2 * head_dim + kTileKeys + kThreads / kWarpSize + 2;
  const dim3 grid(static_cast<unsigned>(num_query_heads),
                  static_cast<unsigned>(batch),
                  static_cast<unsigned>(splits));

  decode_attention_kernel<scalar_t>
      <<<grid, kThreads, shared_floats * sizeof(float), stream>>>(
          q.data_ptr<scalar_t>(),
          k.data_ptr<scalar_t>(),
          v.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>(),
          splits > 1 ? partial_out.data_ptr<float>() : nullptr,
          splits > 1 ? partial_max.data_ptr<float>() : nullptr,
          splits > 1 ? partial_sum.data_ptr<float>() : nullptr,
          strides_of(q),
          strides_of(k),
          strides_of(v),
          strides_of(out),
          source_len,
          head_dim,
          group_size,
          splits,
          scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (splits > 1) {
    const dim3 merge_grid(static_cast<unsigned>(num_query_heads), static_cast<unsigned>(batch));
    decode_attention_merge_kernel<scalar_t>
        <<<merge_grid, kThreads, 2 * kMaxSplits * sizeof(float), stream>>>(
            partial_out.data_ptr<float>(),
            partial_max.data_ptr<float>(),
            partial_sum.data_ptr<float>(),
            out.data_ptr<scalar_t>(),
            strides_of(out),
            head_dim,
            splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

}  // namespace

torch::Tensor decode_attention(const torch::Tensor& q,
                               const torch::Tensor& k,
                               const torch::Tensor& v,
                               double scale) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "decode_attention: q, k and v must be CUDA tensors");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "decode_attention: expected B x H x L x D tensors, got q.dim()=",
              q.dim());
  TORCH_CHECK(q.size(2) == 1,
              "decode_attention: this kernel is the L == 1 case, got a query length of ",
              q.size(2),
              "; use flash_prefill for longer queries");
  TORCH_CHECK(k.sizes() == v.sizes(),
              "decode_attention: k is ",
              k.sizes(),
              " but v is ",
              v.sizes());
  TORCH_CHECK(q.size(0) == k.size(0),
              "decode_attention: batch mismatch, q has ",
              q.size(0),
              " and k has ",
              k.size(0));
  TORCH_CHECK(q.size(3) == k.size(3),
              "decode_attention: head dim mismatch, q has ",
              q.size(3),
              " and k has ",
              k.size(3));
  TORCH_CHECK(k.size(1) > 0 && q.size(1) % k.size(1) == 0,
              "decode_attention: H_q (",
              q.size(1),
              ") must be a multiple of H_k (",
              k.size(1),
              ")");
  TORCH_CHECK(k.size(2) > 0,
              "decode_attention: the cache is empty, so there is nothing to attend to; "
              "softmax over zero keys is undefined");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
              "decode_attention: q, k and v must share a dtype, got ",
              q.scalar_type(),
              ", ",
              k.scalar_type(),
              " and ",
              v.scalar_type());
  TORCH_CHECK(q.scalar_type() != at::kDouble,
              "decode_attention: float64 is not supported; the accumulators here are fp32. "
              "Use the PyTorch path.");
  TORCH_CHECK(q.size(3) <= kMaxHeadDim,
              "decode_attention: head dim ",
              q.size(3),
              " exceeds ",
              kMaxHeadDim,
              "; the query and the accumulator both live in shared memory");

  // Only the head dimension must be contiguous; the rest is read through its strides.
  TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1,
              "decode_attention: the head dimension must be contiguous");

  // Not `empty_like`: it can infer a channels-last layout from the input's strides.
  at::Tensor out = torch::empty(q.sizes(), q.options());

  AT_DISPATCH_SWITCH(q.scalar_type(),
                     "decode_attention",
                     AT_DISPATCH_CASE(at::ScalarType::Float,
                                      [&] {
                                        launch_decode_attention<scalar_t>(
                                            q, k, v, out, static_cast<float>(scale));
                                      })
                         AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES([&] {
                           launch_decode_attention<scalar_t>(
                               q, k, v, out, static_cast<float>(scale));
                         }));

  return out;
}
