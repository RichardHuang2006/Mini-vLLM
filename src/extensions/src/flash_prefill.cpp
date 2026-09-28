#include "mini_vllm_ext.h"
#include "mlx/backend/metal/device.h"

namespace mini_vllm_ext {

mx::array flash_prefill(const mx::array &q, const mx::array &k, const mx::array &v, float scale, bool causal,
                        mx::StreamOrDevice s) {
    const int num_heads = q.shape()[q.ndim() - 3];
    const int num_kv_heads = k.shape()[k.ndim() - 3];
    auto flat = [&](const mx::array &a) { return mx::contiguous(a, false, s); };
    return mx::array(q.shape(), q.dtype(),
                     std::make_shared<FlashPrefill>(mx::to_stream(s), scale, causal, num_heads, num_kv_heads),
                     {flat(q), flat(k), flat(v)});
}

void FlashPrefill::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &q = inputs[0];
    const auto &k = inputs[1];
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &d = mx::metal::device(stream().device);
    auto kernel = d.get_kernel("flash_prefill_bf16_d128", d.get_library("mini_vllm_ext"));
    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < 3; ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 3);

    const int length = q.shape()[q.ndim() - 2];
    const int context = k.shape()[k.ndim() - 2];
    const int query_rows = q.size() / (length * 128);
    const int causal = causal_;
    encoder.set_bytes(length, 4);
    encoder.set_bytes(context, 5);
    encoder.set_bytes(scale_, 6);
    encoder.set_bytes(causal, 7);
    encoder.set_bytes(num_heads_, 8);
    encoder.set_bytes(num_kv_heads_, 9);

    // One threadgroup of 4 SIMD groups per 32-query block of one head.
    constexpr int query_block = 32;
    encoder.dispatch_threadgroups(MTL::Size((length + query_block - 1) / query_block, query_rows, 1),
                                  MTL::Size(128, 1, 1));
}

}  // namespace mini_vllm_ext
