#include <algorithm>

#include "mini_vllm_ext.h"
#include "mlx/backend/metal/device.h"

namespace mini_vllm_ext {

mx::array kv_quantize_scatter(const mx::array &pool, const mx::array &slot_mapping, const mx::array &values,
                              float scale, mx::StreamOrDevice s) {
    return mx::array(pool.shape(), pool.dtype(), std::make_shared<KvQuantizeScatter>(mx::to_stream(s), scale),
                     {pool, mx::contiguous(slot_mapping, false, s), mx::contiguous(values, false, s)});
}

void KvQuantizeScatter::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &values = inputs[2];
    auto &out = outputs[0];
    out.copy_shared_buffer(inputs[0]);  // in place, as paged_cache_update

    auto &d = mx::metal::device(stream().device);
    auto kernel = d.get_kernel(kernel_name("kv_quantize_scatter", values.dtype()), d.get_library("mini_vllm_ext"));
    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[1], 0);
    encoder.set_input_array(values, 1);
    encoder.set_output_array(out, 2);
    const int row = values.size() / values.shape()[0];
    encoder.set_bytes(row, 3);
    encoder.set_bytes(scale_, 4);
    const uint size = values.size();
    const size_t threads = std::min<size_t>(size, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(size, 1, 1), MTL::Size(threads, 1, 1));
}

}  // namespace mini_vllm_ext
