// Flash prefill: many query tokens against many keys, causally masked.
//
// Same recurrence as the decode attention kernel, but with `L > 1` there is a
// second axis to tile, and two things change because of it.
//
// **Shared memory earns its place.** In decode each K element is touched by
// exactly one dot product, so staging it would be a pure copy. Here every K
// element is touched by all `kQueryTile` queries in the block and every V element
// by all of them too, so a tile is loaded from global once (coalesced) and read
// `kQueryTile` times from shared. That reuse is also what lets each *thread* own a
// whole `(query, key)` dot product: no cross-lane reduction, unlike the decode
// kernel where the warp had to shuffle because K came straight from global.
//
// **Half the work does not exist.** Query `i` may attend only to keys up to
// `(S - L) + i`, so key tiles strictly above the diagonal are skipped rather than
// computed and masked — for a square prefill that is half the flops. Only the tile
// *on* the diagonal needs the per-element mask, and it is the tile where an
// off-by-one lets a token see its own future and produces a model that scores
// suspiciously well and generates nonsense.
//
// Shared layouts, both chosen for bank behaviour rather than tidiness:
//
//   q_shared[i][d]      row-major   threads in a warp share `i`, so this broadcasts
//   k_shared[d][j]      transposed  threads in a warp differ in `j`, so this is
//                                   consecutive — padded by one to keep the
//                                   coalesced *store* from conflicting as well
//   v_shared[j][d]      row-major   the P·V phase has threads differ in `d`

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "kernel_utils.cuh"

namespace {

using namespace mini_vllm;

constexpr int kQueryTile = 16;
constexpr int kKeyTile = 16;
constexpr int kThreads = 256;  // exactly one thread per (query, key) pair
constexpr int kKeyStride = kKeyTile + 1;

// Set by the shared-memory budget below, which is checked on the host.
constexpr int kMaxHeadDim = 192;

struct Strides {
  int64_t batch;
  int64_t head;
  int64_t position;
};

// (q + v) * kQueryTile*D + k * D*(kKeyTile+1) + scores + three per-query vectors
size_t shared_floats_for(int64_t head_dim) {
  return static_cast<size_t>(kQueryTile * head_dim + kKeyTile * head_dim +
                             head_dim * kKeyStride + kQueryTile * kKeyTile + 3 * kQueryTile);
}

template <typename scalar_t>
__global__ void flash_prefill_kernel(const scalar_t* __restrict__ q,
                                     const scalar_t* __restrict__ k,
                                     const scalar_t* __restrict__ v,
                                     scalar_t* __restrict__ out,
                                     const Strides q_strides,
                                     const Strides k_strides,
                                     const Strides v_strides,
                                     const Strides out_strides,
                                     const int64_t query_len,
                                     const int64_t source_len,
                                     const int64_t head_dim,
                                     const int64_t group_size,
                                     const float scale) {
  const int64_t query_begin = static_cast<int64_t>(blockIdx.x) * kQueryTile;
  const int64_t query_head = blockIdx.y;
  const int64_t sequence = blockIdx.z;
  const int64_t kv_head = query_head / group_size;

  // The offset form of the causal mask in the PyTorch reference: with a filled
  // cache the `L` queries are the *last* `L` positions of the sequence, so the
  // diagonal is shifted right by S - L.
  const int64_t offset = source_len - query_len;

  const scalar_t* query = q + sequence * q_strides.batch + query_head * q_strides.head;
  const scalar_t* keys = k + sequence * k_strides.batch + kv_head * k_strides.head;
  const scalar_t* values = v + sequence * v_strides.batch + kv_head * v_strides.head;

  extern __shared__ float shared[];
  float* q_shared = shared;
  float* v_shared = q_shared + kQueryTile * head_dim;
  float* k_shared = v_shared + kKeyTile * head_dim;
  float* scores = k_shared + head_dim * kKeyStride;
  float* running_max = scores + kQueryTile * kKeyTile;
  float* running_sum = running_max + kQueryTile;
  float* correction = running_sum + kQueryTile;

  const int thread = threadIdx.x;
  // The score phase's decomposition, fixed for the whole kernel: a warp spans two
  // query rows of 16 keys, so lanes differ in `j` and share `i` in pairs.
  const int score_query = thread / kKeyTile;
  const int score_key = thread % kKeyTile;

  // The accumulator stays in registers — another kQueryTile x D floats of shared
  // memory would not fit beside the three tiles. Thread `t` owns the
  // `(query, dimension)` pairs `t, t + kThreads, ...`, one register each.
  //
  // Every loop over these slots is bounded by a *compile-time* count and exits on
  // a runtime predicate, rather than being bounded by the runtime count directly.
  // That is deliberate: `accumulator[slot]` has to be a constant index for the
  // array to live in registers at all, and a runtime trip count would silently
  // spill it to local memory — correct, and several times slower.
  constexpr int kSlots = (kQueryTile * kMaxHeadDim + kThreads - 1) / kThreads;
  float accumulator[kSlots];

  // Which `(query, dimension)` each slot is, resolved once here rather than per
  // tile. `head_dim` is a runtime value, so `index / head_dim` is a genuine integer
  // division of around twenty instructions, and it belongs nowhere near a loop
  // whose body is one multiply-add.
  int slot_query[kSlots];
  int slot_dim[kSlots];
  const int64_t owned = kQueryTile * head_dim;
  const int live_slots = static_cast<int>((owned - thread + kThreads - 1) / kThreads);
#pragma unroll
  for (int slot = 0; slot < kSlots; ++slot) {
    const int64_t index = thread + static_cast<int64_t>(slot) * kThreads;
    accumulator[slot] = 0.0f;
    slot_query[slot] = static_cast<int>(index < owned ? index / head_dim : 0);
    slot_dim[slot] = static_cast<int>(index < owned ? index % head_dim : 0);
  }

  for (int64_t index = thread; index < kQueryTile * head_dim; index += kThreads) {
    const int64_t i = index / head_dim;
    const int64_t d = index % head_dim;
    const int64_t position = query_begin + i;
    q_shared[index] =
        position < query_len ? static_cast<float>(query[position * q_strides.position + d]) : 0.0f;
  }
  if (thread < kQueryTile) {
    running_max[thread] = -INFINITY;
    running_sum[thread] = 0.0f;
  }
  __syncthreads();

  // Every key tile that any query in this tile can see. The last query in the tile
  // has the longest reach, so its bound is the block's bound — and skipping the
  // rest is the causal speedup, not an optimization on top of it.
  const int64_t last_key = offset + query_begin + kQueryTile - 1;
  const int64_t key_limit = last_key + 1 < source_len ? last_key + 1 : source_len;

  for (int64_t key_begin = 0; key_begin < key_limit; key_begin += kKeyTile) {
    const int64_t keys_here =
        (key_limit - key_begin) < kKeyTile ? (key_limit - key_begin) : kKeyTile;

    // --- stage the tile. Reads are coalesced along `d`; the K store goes
    // transposed into a padded row so neither the store nor the later read
    // collides on a bank.
    for (int64_t index = thread; index < kKeyTile * head_dim; index += kThreads) {
      const int64_t j = index / head_dim;
      const int64_t d = index % head_dim;
      const bool live = j < keys_here;
      k_shared[d * kKeyStride + j] =
          live ? static_cast<float>(keys[(key_begin + j) * k_strides.position + d]) : 0.0f;
      v_shared[j * head_dim + d] =
          live ? static_cast<float>(values[(key_begin + j) * v_strides.position + d]) : 0.0f;
    }
    __syncthreads();

    // --- QKᵀ for this tile: one thread, one dot product, all of it from shared.
    {
      const int64_t position = query_begin + score_query;
      float dot = 0.0f;
      for (int64_t d = 0; d < head_dim; ++d) {
        dot += q_shared[score_query * head_dim + d] * k_shared[d * kKeyStride + score_key];
      }
      // Masked and out-of-range entries become -inf, which the recurrence turns
      // into a zero weight without a branch. `keys_here` covers the ragged end of
      // the cache; the causal test covers the diagonal.
      const bool visible = score_key < keys_here && position < query_len &&
                           (key_begin + score_key) <= offset + position;
      scores[score_query * kKeyTile + score_key] = visible ? dot * scale : -INFINITY;
    }
    __syncthreads();

    // --- the per-query softmax update. Sixteen threads each scan their own row of
    // sixteen scores: a cross-lane reduction over so few values would cost more in
    // barriers than the serial scan costs in arithmetic.
    if (thread < kQueryTile) {
      float* row = scores + thread * kKeyTile;

      float tile_max = -INFINITY;
      for (int j = 0; j < kKeyTile; ++j) {
        tile_max = fmaxf(tile_max, row[j]);
      }

      // A query whose whole tile is masked keeps its running max and gets a
      // correction of exactly 1, so an all -inf row needs no special case.
      const float new_max = fmaxf(running_max[thread], tile_max);
      float tile_sum = 0.0f;
      for (int j = 0; j < kKeyTile; ++j) {
        const float weight = row[j] == -INFINITY ? 0.0f : __expf(row[j] - new_max);
        row[j] = weight;
        tile_sum += weight;
      }

      const float factor = running_max[thread] == -INFINITY && new_max == -INFINITY
                               ? 1.0f
                               : __expf(running_max[thread] - new_max);
      correction[thread] = factor;
      running_sum[thread] = running_sum[thread] * factor + tile_sum;
      running_max[thread] = new_max;
    }
    __syncthreads();

    // --- P·V. Threads differ in `d`, so both the shared read and the register
    // slot they own stay contiguous.
#pragma unroll
    for (int slot = 0; slot < kSlots; ++slot) {
      if (slot >= live_slots) {
        break;
      }
      const int query = slot_query[slot];
      const int dim = slot_dim[slot];
      float total = accumulator[slot] * correction[query];
      for (int j = 0; j < kKeyTile; ++j) {
        total += scores[query * kKeyTile + j] * v_shared[j * head_dim + dim];
      }
      accumulator[slot] = total;
    }
    // The next tile overwrites k_shared, v_shared and scores, all of which the
    // loops above are still reading.
    __syncthreads();
  }

  scalar_t* destination = out + sequence * out_strides.batch + query_head * out_strides.head;
#pragma unroll
  for (int slot = 0; slot < kSlots; ++slot) {
    if (slot >= live_slots) {
      break;
    }
    const int query = slot_query[slot];
    const int64_t position = query_begin + query;
    if (position < query_len) {
      destination[position * out_strides.position + slot_dim[slot]] =
          static_cast<scalar_t>(accumulator[slot] / running_sum[query]);
    }
  }
}

Strides strides_of(const at::Tensor& tensor) {
  return Strides{tensor.stride(0), tensor.stride(1), tensor.stride(2)};
}

template <typename scalar_t>
void launch_flash_prefill(const at::Tensor& q,
                          const at::Tensor& k,
                          const at::Tensor& v,
                          at::Tensor& out,
                          const float scale) {
  const int64_t query_len = q.size(2);
  const int64_t head_dim = q.size(3);
  const int64_t query_tiles = (query_len + kQueryTile - 1) / kQueryTile;

  const dim3 grid(static_cast<unsigned>(query_tiles),
                  static_cast<unsigned>(q.size(1)),
                  static_cast<unsigned>(q.size(0)));

  flash_prefill_kernel<scalar_t>
      <<<grid, kThreads, shared_floats_for(head_dim) * sizeof(float),
         at::cuda::getCurrentCUDAStream()>>>(q.data_ptr<scalar_t>(),
                                             k.data_ptr<scalar_t>(),
                                             v.data_ptr<scalar_t>(),
                                             out.data_ptr<scalar_t>(),
                                             strides_of(q),
                                             strides_of(k),
                                             strides_of(v),
                                             strides_of(out),
                                             query_len,
                                             k.size(2),
                                             head_dim,
                                             q.size(1) / k.size(1),
                                             scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor flash_prefill(const torch::Tensor& q,
                            const torch::Tensor& k,
                            const torch::Tensor& v,
                            double scale) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "flash_prefill: q, k and v must be CUDA tensors");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "flash_prefill: expected B x H x L x D tensors, got q.dim()=",
              q.dim());
  TORCH_CHECK(k.sizes() == v.sizes(),
              "flash_prefill: k is ",
              k.sizes(),
              " but v is ",
              v.sizes());
  TORCH_CHECK(q.size(0) == k.size(0),
              "flash_prefill: batch mismatch, q has ",
              q.size(0),
              " and k has ",
              k.size(0));
  TORCH_CHECK(q.size(3) == k.size(3),
              "flash_prefill: head dim mismatch, q has ",
              q.size(3),
              " and k has ",
              k.size(3));
  TORCH_CHECK(k.size(1) > 0 && q.size(1) % k.size(1) == 0,
              "flash_prefill: H_q (",
              q.size(1),
              ") must be a multiple of H_k (",
              k.size(1),
              ")");
  TORCH_CHECK(q.size(2) > 0 && k.size(2) >= q.size(2),
              "flash_prefill: ",
              q.size(2),
              " queries against ",
              k.size(2),
              " keys; this kernel masks causally, which needs S >= L > 0");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
              "flash_prefill: q, k and v must share a dtype, got ",
              q.scalar_type(),
              ", ",
              k.scalar_type(),
              " and ",
              v.scalar_type());
  TORCH_CHECK(q.scalar_type() != at::kDouble,
              "flash_prefill: float64 is not supported; the accumulators here are fp32. "
              "Use the PyTorch path.");
  TORCH_CHECK(q.size(3) <= kMaxHeadDim,
              "flash_prefill: head dim ",
              q.size(3),
              " exceeds ",
              kMaxHeadDim,
              "; the query, key and value tiles all live in shared memory");
  TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1,
              "flash_prefill: the head dimension must be contiguous");

  at::Tensor out = torch::empty(q.sizes(), q.options());

  AT_DISPATCH_SWITCH(q.scalar_type(),
                     "flash_prefill",
                     AT_DISPATCH_CASE(at::ScalarType::Float,
                                      [&] {
                                        launch_flash_prefill<scalar_t>(
                                            q, k, v, out, static_cast<float>(scale));
                                      })
                         AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES([&] {
                           launch_flash_prefill<scalar_t>(q, k, v, out, static_cast<float>(scale));
                         }));

  return out;
}
