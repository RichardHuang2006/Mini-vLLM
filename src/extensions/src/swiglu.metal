#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// silu(gate) * up in fp32, rounded once: the MLP's two up-projections merged in one pass.
template <typename T>
[[kernel]] void swiglu(
    device const T* gate [[buffer(0)]],
    device const T* up [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const uint& size [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= size) return;
    const float g = static_cast<float>(gate[index]);
    out[index] = static_cast<T>(g / (1.0f + exp(-g)) * static_cast<float>(up[index]));
}

instantiate_kernel("swiglu_f32", swiglu, float);
instantiate_kernel("swiglu_f16", swiglu, half);
instantiate_kernel("swiglu_bf16", swiglu, bfloat16_t);
