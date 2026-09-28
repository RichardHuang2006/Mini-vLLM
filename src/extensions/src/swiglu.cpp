#include <algorithm>

#include "mini_vllm_ext.h"
#include "mlx/backend/metal/device.h"

namespace mini_vllm_ext {

mx::array swiglu(const mx::array &gate, const mx::array &up, mx::StreamOrDevice s) {
    return mx::array(gate.shape(), gate.dtype(), std::make_shared<SwiGLU>(mx::to_stream(s)),
                     {mx::contiguous(gate, false, s), mx::contiguous(up, false, s)});
}

void SwiGLU::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &d = mx::metal::device(stream().device);
    auto kernel = d.get_kernel(kernel_name("swiglu", out.dtype()), d.get_library("mini_vllm_ext"));
    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_output_array(out, 2);
    const uint size = out.size();
    encoder.set_bytes(size, 3);
    const size_t threads = std::min<size_t>(size, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(size, 1, 1), MTL::Size(threads, 1, 1));
}

}  // namespace mini_vllm_ext
