#include <algorithm>

#include "mini_vllm_ext.h"
#include "mlx/backend/metal/device.h"

namespace mini_vllm_ext {

mx::array rms_norm(const mx::array &x, const mx::array &weight, float eps, mx::StreamOrDevice s) {
    // The kernel walks rows of D contiguous values, so a strided view is copied first.
    return mx::array(x.shape(), x.dtype(), std::make_shared<RMSNorm>(mx::to_stream(s), eps),
                     {mx::contiguous(x, false, s), mx::contiguous(weight, false, s)});
}

void RMSNorm::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &x = inputs[0];
    const auto &weight = inputs[1];
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));

    const int dim = x.shape().back();
    const int rows = x.size() / dim;
    auto &d = mx::metal::device(stream().device);
    auto kernel = d.get_kernel(kernel_name("rms_norm", out.dtype()), d.get_library("mini_vllm_ext"));
    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(x, 0);
    encoder.set_input_array(weight, 1);
    encoder.set_output_array(out, 2);
    encoder.set_bytes(dim, 3);
    encoder.set_bytes(eps_, 4);

    // One threadgroup per row, one thread per column up to 1024; wider rows stride.
    const int threads = std::min(1024, (dim + 31) / 32 * 32);
    encoder.set_threadgroup_memory_length(32 * sizeof(float), 0);
    encoder.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(threads, 1, 1));
}

}  // namespace mini_vllm_ext
