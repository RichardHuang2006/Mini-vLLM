#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/fp8.h"

using namespace metal;

// Quantize each value to e4m3 at a static per-tensor scale and scatter it into the uint8
// pool, in one pass. fp8_e4m3 is MLX's own conversion, the one mx.to_fp8 runs, and the
// division is the same fp32 one quantize.py does, so the bytes match it exactly.
template <typename T>
[[kernel]] void kv_quantize_scatter(
    device const int* slot_mapping [[buffer(0)]],
    device const T* values [[buffer(1)]],
    device uint8_t* pool [[buffer(2)]],
    constant const int& row [[buffer(3)]],
    constant const float& scale [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const int token = index / row;
    pool[size_t(slot_mapping[token]) * row + index % row] =
        fp8_e4m3(static_cast<float>(values[index]) / scale).bits;
}

instantiate_kernel("kv_quantize_scatter_f32", kv_quantize_scatter, float);
instantiate_kernel("kv_quantize_scatter_f16", kv_quantize_scatter, half);
instantiate_kernel("kv_quantize_scatter_bf16", kv_quantize_scatter, bfloat16_t);
