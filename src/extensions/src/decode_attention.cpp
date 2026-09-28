#include "mini_vllm_ext.h"
#include "mlx/backend/metal/device.h"

namespace mini_vllm_ext {

mx::array decode_attention(const mx::array &q, const mx::array &k, const mx::array &v, float scale, bool causal,
                           mx::StreamOrDevice s) {
    const int num_heads = q.shape()[q.ndim() - 3];
    const int num_kv_heads = k.shape()[k.ndim() - 3];
    auto flat = [&](const mx::array &a) { return mx::contiguous(a, false, s); };
    return mx::array(q.shape(), q.dtype(),
                     std::make_shared<DecodeAttention>(mx::to_stream(s), scale, causal, num_heads, num_kv_heads),
                     {flat(q), flat(k), flat(v)});
}

void DecodeAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &q = inputs[0];
    const auto &k = inputs[1];
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &d = mx::metal::device(stream().device);
    auto kernel = d.get_kernel(kernel_name("decode_attention", out.dtype()), d.get_library("mini_vllm_ext"));
    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < 3; ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 3);

    // q is [B, H_q, L, D] and k, v [B, H_k, S, D], contiguous: rows of one head each.
    const int dim = q.shape().back();
    const int length = q.shape()[q.ndim() - 2];
    const int context = k.shape()[k.ndim() - 2];
    const int query_rows = q.size() / (length * dim);
    const int causal = causal_;
    encoder.set_bytes(length, 4);
    encoder.set_bytes(context, 5);
    encoder.set_bytes(dim, 6);
    encoder.set_bytes(num_heads_, 7);
    encoder.set_bytes(num_kv_heads_, 8);
    encoder.set_bytes(scale_, 9);
    encoder.set_bytes(causal, 10);

    // One threadgroup per query token of one head; its 32 SIMD groups split the context.
    constexpr int simdgroups = 32;
    encoder.set_threadgroup_memory_length(simdgroups * (dim + 3) * sizeof(float), 0);
    encoder.dispatch_threadgroups(MTL::Size(query_rows * length, 1, 1), MTL::Size(simdgroups * 32, 1, 1));
}

}  // namespace mini_vllm_ext
