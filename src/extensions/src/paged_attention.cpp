#include <algorithm>
#include <cmath>

#include "mini_vllm_ext.h"
#include "mlx/backend/metal/device.h"

namespace mini_vllm_ext {

mx::array paged_cache_update(const mx::array &pool, const mx::array &slot_mapping, const mx::array &values,
                             mx::StreamOrDevice s) {
    return mx::array(pool.shape(), pool.dtype(), std::make_shared<PagedCacheUpdate>(mx::to_stream(s)),
                     {pool, mx::contiguous(slot_mapping, false, s), mx::contiguous(mx::astype(values, pool.dtype(), s), false, s)});
}

void PagedCacheUpdate::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &pool = inputs[0];
    const auto &values = inputs[2];
    auto &out = outputs[0];
    // The pool is request state, so the output shares its buffer: the kernel writes only
    // the T rows being appended, where a functional update would copy the whole pool.
    out.copy_shared_buffer(pool);

    auto &d = mx::metal::device(stream().device);
    auto kernel = d.get_kernel(kernel_name("paged_cache_update", out.dtype()), d.get_library("mini_vllm_ext"));
    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[1], 0);
    encoder.set_input_array(values, 1);
    encoder.set_output_array(out, 2);
    const int row = values.size() / values.shape()[0];  // H_k * D
    encoder.set_bytes(row, 3);
    const uint size = values.size();
    const size_t threads = std::min<size_t>(size, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(size, 1, 1), MTL::Size(threads, 1, 1));
}

mx::array paged_attention(const mx::array &q, const mx::array &key_pages, const mx::array &value_pages,
                          const mx::array &block_tables, const mx::array &cu_seqlens_q,
                          const mx::array &context_lens, float scale, float k_scale, float v_scale,
                          mx::StreamOrDevice s) {
    // The pages are only read here, so unlike the cache write they can be made contiguous:
    // free for the engine's pools, and a copy for a strided view, whose rows the kernel
    // would otherwise misread.
    auto flat = [&](const mx::array &a) { return mx::contiguous(a, false, s); };
    return mx::array(q.shape(), q.dtype(),
                     std::make_shared<PagedAttention>(mx::to_stream(s), scale, k_scale, v_scale),
                     {flat(q), flat(key_pages), flat(value_pages), flat(block_tables), flat(cu_seqlens_q),
                      flat(context_lens)});
}

void PagedAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &q = inputs[0];
    const auto &key_pages = inputs[1];
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));

    // fp8 pages are uint8 bits, read through MLX's own e4m3 conversion.
    const bool fp8 = key_pages.dtype() == mx::uint8;
    const std::string name = kernel_name(fp8 ? "paged_attention_fp8" : "paged_attention", q.dtype());
    auto &d = mx::metal::device(stream().device);
    auto kernel = d.get_kernel(name, d.get_library("mini_vllm_ext"));
    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < 6; ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 6);

    const int tokens = q.shape()[0];
    const int num_heads = q.shape()[1];
    const int head_dim = q.shape()[2];
    const int block_size = key_pages.shape()[1];
    const int num_kv_heads = key_pages.shape()[2];
    const int num_sequences = inputs[5].shape()[0];
    const int max_blocks = inputs[3].shape()[1];
    const float scale_log2 = scale_ * static_cast<float>(M_LOG2E);
    encoder.set_bytes(num_heads, 7);
    encoder.set_bytes(num_kv_heads, 8);
    encoder.set_bytes(head_dim, 9);
    encoder.set_bytes(block_size, 10);
    encoder.set_bytes(num_sequences, 11);
    encoder.set_bytes(max_blocks, 12);
    encoder.set_bytes(scale_log2, 13);
    encoder.set_bytes(k_scale_, 14);
    encoder.set_bytes(v_scale_, 15);

    // One threadgroup per (token, query head): 32 SIMD groups split its context between
    // them, each keeps an online softmax, and the 32 partial results merge at the end.
    constexpr int simdgroups = 32;
    encoder.set_threadgroup_memory_length((simdgroups * 32 + 2 * simdgroups) * sizeof(float), 0);
    encoder.dispatch_threadgroups(MTL::Size(tokens * num_heads, 1, 1), MTL::Size(simdgroups * 32, 1, 1));
}

}  // namespace mini_vllm_ext
