#include <algorithm>

#include "mini_vllm_ext.h"
#include "mlx/backend/metal/device.h"

namespace mini_vllm_ext {

mx::array rope(const mx::array &x, const mx::array &positions, const mx::array &cos, const mx::array &sin,
               mx::StreamOrDevice s) {
    return mx::array(x.shape(), x.dtype(), std::make_shared<RoPE>(mx::to_stream(s)),
                     {mx::contiguous(x, false, s), mx::contiguous(mx::astype(positions, mx::int32, s), false, s),
                      mx::contiguous(cos, false, s), mx::contiguous(sin, false, s)});
}

void RoPE::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &x = inputs[0];
    const auto &positions = inputs[1];
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &d = mx::metal::device(stream().device);
    auto kernel = d.get_kernel(kernel_name("rope", out.dtype()), d.get_library("mini_vllm_ext"));
    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < 4; ++i) {
        encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 4);

    const int length = x.shape()[1];
    const int heads = x.shape()[2];
    const int head_dim = x.shape()[3];
    // positions is [L], shared by every batch row, or [B, L], one row each.
    const int position_stride = positions.ndim() == 1 ? 0 : length;
    encoder.set_bytes(length, 5);
    encoder.set_bytes(heads, 6);
    encoder.set_bytes(head_dim, 7);
    encoder.set_bytes(position_stride, 8);

    // One thread per rotated pair (i, i + D/2) of one head of one token.
    const uint pairs = x.size() / 2;
    const size_t threads = std::min<size_t>(pairs, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(pairs, 1, 1), MTL::Size(threads, 1, 1));
}

}  // namespace mini_vllm_ext
