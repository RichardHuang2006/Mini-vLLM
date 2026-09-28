#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// One threadgroup per row: a strided sum of squares, reduced within each SIMD group and
// then across them, and a strided write. The normalized value is rounded to T before the
// weight multiply, the order HuggingFace (and layer_norm.py) uses.
template <typename T>
[[kernel]] void rms_norm(
    device const T* x [[buffer(0)]],
    device const T* weight [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const int& dim [[buffer(3)]],
    constant const float& eps [[buffer(4)]],
    threadgroup float* partials [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    device const T* in = x + size_t(row) * dim;
    device T* result = out + size_t(row) * dim;

    float sum = 0.0f;
    for (int col = tid; col < dim; col += threads) {
        const float value = static_cast<float>(in[col]);
        sum += value * value;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        partials[simdgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Every SIMD group reduces the partials itself, so no second barrier is needed.
    const uint simdgroups = (threads + 31) / 32;
    sum = simd_sum(lane < simdgroups ? partials[lane] : 0.0f);
    const float inverse_rms = rsqrt(sum / dim + eps);

    for (int col = tid; col < dim; col += threads) {
        const T normalized = static_cast<T>(static_cast<float>(in[col]) * inverse_rms);
        result[col] = static_cast<T>(static_cast<float>(normalized) * static_cast<float>(weight[col]));
    }
}

instantiate_kernel("rms_norm_f32", rms_norm, float);
instantiate_kernel("rms_norm_f16", rms_norm, half);
instantiate_kernel("rms_norm_bf16", rms_norm, bfloat16_t);
