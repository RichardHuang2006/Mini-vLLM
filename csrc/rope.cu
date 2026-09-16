// RoPE at explicit positions; the oracle is `mini_vllm.ops.apply_rope`, tables and all.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/macros/Macros.h>
#include <torch/extension.h>

#include <algorithm>

namespace {

constexpr int kThreads = 256;

// One thread per rotated pair, the unit the rotation couples.
template <typename scalar_t, typename index_t>
__global__ void rope_kernel(const scalar_t* __restrict__ x,
                            const index_t* __restrict__ positions,
                            const float* __restrict__ cos_table,
                            const float* __restrict__ sin_table,
                            scalar_t* __restrict__ out,
                            const int64_t pairs,
                            const int64_t half,
                            const int64_t heads,
                            const int64_t token_count,
                            const int64_t max_seq_len) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pairs) {
    return;
  }

  const int64_t dim = 2 * half;
  const int64_t row = index / half;  // which (token, head) vector
  const int64_t lane = index % half;

  // The oracle's unsqueeze(-2) as arithmetic: one angle per token, shared by its heads.
  const int64_t token = row / heads;
  const int64_t position = static_cast<int64_t>(positions[token % token_count]);

  // Untested: a failed device assert poisons the CUDA context for the whole process.
  CUDA_KERNEL_ASSERT(position >= 0 && position < max_seq_len);

  const int64_t low = row * dim + lane;
  const int64_t high = low + half;
  const int64_t table_low = position * dim + lane;
  const int64_t table_high = table_low + half;

  const float x_low = static_cast<float>(x[low]);
  const float x_high = static_cast<float>(x[high]);

  // `rotate_half` sends [a, b] to [-b, a]: the low half subtracts, the high half adds.
  out[low] = static_cast<scalar_t>(x_low * cos_table[table_low] - x_high * sin_table[table_low]);
  out[high] = static_cast<scalar_t>(x_high * cos_table[table_high] + x_low * sin_table[table_high]);
}

template <typename scalar_t>
void launch_rope(const at::Tensor& x,
                 const at::Tensor& positions,
                 const at::Tensor& cos,
                 const at::Tensor& sin,
                 at::Tensor& out,
                 int64_t pairs,
                 int64_t half,
                 int64_t heads) {
  const int64_t blocks = (pairs + kThreads - 1) / kThreads;
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_INDEX_TYPES(positions.scalar_type(), "rope_positions", [&] {
    rope_kernel<scalar_t, index_t><<<blocks, kThreads, 0, stream>>>(x.data_ptr<scalar_t>(),
                                                                   positions.data_ptr<index_t>(),
                                                                   cos.data_ptr<float>(),
                                                                   sin.data_ptr<float>(),
                                                                   out.data_ptr<scalar_t>(),
                                                                   pairs,
                                                                   half,
                                                                   heads,
                                                                   positions.numel(),
                                                                   cos.size(0));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor rope(const torch::Tensor& x,
                   const torch::Tensor& positions,
                   const torch::Tensor& cos,
                   const torch::Tensor& sin) {
  TORCH_CHECK(x.is_cuda(), "rope: x must be a CUDA tensor");
  TORCH_CHECK(positions.is_cuda(), "rope: positions must be a CUDA tensor");
  TORCH_CHECK(cos.is_cuda() && sin.is_cuda(), "rope: the cos/sin tables must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() != at::kDouble,
              "rope: float64 is not supported; this kernel computes the rotation in fp32. "
              "Use the PyTorch path.");
  TORCH_CHECK(
      x.dim() >= 3,
      "rope: x must be N.. x H x D — a head axis is required, since one position "
      "applies to every head of a token. Got ",
      x.dim(),
      " dimensions");

  const int64_t dim = x.size(-1);
  TORCH_CHECK(dim % 2 == 0, "rope: the head dimension must be even, got ", dim);
  TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2, "rope: the cos/sin tables must be 2-D");
  TORCH_CHECK(cos.sizes() == sin.sizes(),
              "rope: cos is ",
              cos.sizes(),
              " but sin is ",
              sin.sizes());
  TORCH_CHECK(cos.size(1) == dim,
              "rope: the tables are ",
              cos.size(1),
              " wide but the head dimension is ",
              dim,
              "; mini_vllm/ops.py builds them as max_seq_len x D with the angles "
              "duplicated");
  TORCH_CHECK(cos.scalar_type() == at::kFloat && sin.scalar_type() == at::kFloat,
              "rope: the cos/sin tables must be float32 even when x is bf16 — see "
              "mini_vllm/ops.py. Got ",
              cos.scalar_type());
  TORCH_CHECK(positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong,
              "rope: positions must be int32 or int64, got ",
              positions.scalar_type());

  const at::Tensor input = x.contiguous();
  at::Tensor out = torch::empty_like(input);

  const int64_t rows = dim == 0 ? 0 : input.numel() / dim;
  if (rows == 0) {
    return out;
  }

  const at::Tensor position_ids = positions.contiguous();
  const at::Tensor cos_table = cos.contiguous();
  const at::Tensor sin_table = sin.contiguous();

  const int64_t heads = input.size(-2);
  const int64_t tokens = rows / heads;
  const int64_t token_count = position_ids.numel();
  TORCH_CHECK(token_count > 0, "rope: positions is empty");
  TORCH_CHECK(tokens % token_count == 0,
              "rope: x holds ",
              tokens,
              " tokens, which is not a multiple of the ",
              token_count,
              " positions given, so the positions cannot broadcast over it");

  const int64_t half = dim / 2;
  const int64_t pairs = rows * half;

  AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "rope",
      AT_DISPATCH_CASE(at::ScalarType::Float,
                       [&] {
                         launch_rope<scalar_t>(
                             input, position_ids, cos_table, sin_table, out, pairs, half, heads);
                       })
          AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES([&] {
            launch_rope<scalar_t>(
                input, position_ids, cos_table, sin_table, out, pairs, half, heads);
          }));

  return out;
}
