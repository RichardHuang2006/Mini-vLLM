// Quantize a step's keys and values straight into the FP8 cache, in one pass.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 256;

template <typename scalar_t, typename cache_t>
__global__ void kv_quantize_scatter_kernel(const scalar_t* __restrict__ key,
                                           const scalar_t* __restrict__ value,
                                           cache_t* __restrict__ key_pool,
                                           cache_t* __restrict__ value_pool,
                                           const int64_t* __restrict__ slot_mapping,
                                           const int64_t num_tokens,
                                           const int64_t inner,  // H_k * D, one token's KV
                                           const float inv_k_scale,
                                           const float inv_v_scale) {
  const int64_t element = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= num_tokens * inner) {
    return;
  }
  const int64_t token = element / inner;
  const int64_t within = element - token * inner;
  const int64_t dest = slot_mapping[token] * inner + within;

  // Divide in fp32, then round to nearest even: the oracle's two steps, in its order.
  key_pool[dest] = static_cast<cache_t>(static_cast<float>(key[element]) * inv_k_scale);
  value_pool[dest] = static_cast<cache_t>(static_cast<float>(value[element]) * inv_v_scale);
}

template <typename scalar_t, typename cache_t>
void launch(const at::Tensor& key,
            const at::Tensor& value,
            at::Tensor& key_pool,
            at::Tensor& value_pool,
            const at::Tensor& slot_mapping,
            const float inv_k_scale,
            const float inv_v_scale) {
  const int64_t num_tokens = key.size(0);
  const int64_t inner = key.numel() / num_tokens;
  const int64_t total = num_tokens * inner;
  if (total == 0) {
    return;
  }

  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t blocks = (total + kThreads - 1) / kThreads;
  kv_quantize_scatter_kernel<scalar_t, cache_t><<<blocks, kThreads, 0, stream>>>(
      key.data_ptr<scalar_t>(),
      value.data_ptr<scalar_t>(),
      key_pool.data_ptr<cache_t>(),
      value_pool.data_ptr<cache_t>(),
      slot_mapping.data_ptr<int64_t>(),
      num_tokens,
      inner,
      inv_k_scale,
      inv_v_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename cache_t>
void dispatch_by_input(const at::Tensor& key,
                       const at::Tensor& value,
                       at::Tensor& key_pool,
                       at::Tensor& value_pool,
                       const at::Tensor& slot_mapping,
                       const float inv_k_scale,
                       const float inv_v_scale) {
  AT_DISPATCH_SWITCH(
      key.scalar_type(),
      "kv_quantize_scatter",
      AT_DISPATCH_CASE(at::ScalarType::Float,
                       [&] {
                         launch<scalar_t, cache_t>(key, value, key_pool, value_pool,
                                                   slot_mapping, inv_k_scale, inv_v_scale);
                       })
          AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES([&] {
            launch<scalar_t, cache_t>(key, value, key_pool, value_pool, slot_mapping,
                                      inv_k_scale, inv_v_scale);
          }));
}

}  // namespace

void kv_quantize_scatter(const torch::Tensor& key,
                         const torch::Tensor& value,
                         torch::Tensor key_pool,
                         torch::Tensor value_pool,
                         const torch::Tensor& slot_mapping,
                         double k_scale,
                         double v_scale) {
  TORCH_CHECK(key.is_cuda() && value.is_cuda() && key_pool.is_cuda() && value_pool.is_cuda(),
              "kv_quantize_scatter: every tensor must be CUDA");
  TORCH_CHECK(key.sizes() == value.sizes(),
              "kv_quantize_scatter: key and value must match, got ",
              key.sizes(),
              " and ",
              value.sizes());
  TORCH_CHECK(key.is_contiguous() && value.is_contiguous(),
              "kv_quantize_scatter: key and value must be contiguous");
  TORCH_CHECK(key_pool.is_contiguous() && value_pool.is_contiguous(),
              "kv_quantize_scatter: the pools must be contiguous");
  TORCH_CHECK(key_pool.scalar_type() == value_pool.scalar_type(),
              "kv_quantize_scatter: the pools must share a dtype");
  TORCH_CHECK(slot_mapping.scalar_type() == at::kLong && slot_mapping.dim() == 1 &&
                  slot_mapping.size(0) == key.size(0),
              "kv_quantize_scatter: slot_mapping must be an int64 vector, one per token");

  const float inv_k_scale = static_cast<float>(1.0 / k_scale);
  const float inv_v_scale = static_cast<float>(1.0 / v_scale);
  // e4m3 only, matching the attention kernel; an e5m2 pool takes the PyTorch path.
  TORCH_CHECK(key_pool.scalar_type() == at::kFloat8_e4m3fn,
              "kv_quantize_scatter: the pools must be FP8 e4m3, got ",
              key_pool.scalar_type());
  dispatch_by_input<at::Float8_e4m3fn>(key, value, key_pool, value_pool, slot_mapping,
                                       inv_k_scale, inv_v_scale);
}
