#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Rotates x [B, L, H, D] by the fp32 tables [max_seq_len, D] gathered at each token's own
// position, pairing element i with i + D/2 as Qwen3 was trained. The tables are the ones
// positional_encoding.py builds, so the angles are the oracle's, bit for bit.
template <typename T>
[[kernel]] void rope(
    device const T* x [[buffer(0)]],
    device const int* positions [[buffer(1)]],
    device const float* cos_table [[buffer(2)]],
    device const float* sin_table [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant const int& length [[buffer(5)]],
    constant const int& heads [[buffer(6)]],
    constant const int& head_dim [[buffer(7)]],
    constant const int& position_stride [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
    const int half_dim = head_dim / 2;
    const int pair = index % half_dim;
    const int row = index / half_dim;   // (token, head)
    const int token = row / heads;      // b * L + l
    const int b = token / length;
    const int l = token - b * length;
    const int position = positions[b * position_stride + l];

    const int first = row * head_dim + pair;
    const int second = first + half_dim;
    const float x1 = static_cast<float>(x[first]);
    const float x2 = static_cast<float>(x[second]);
    const int table = position * head_dim + pair;
    // rotate_half([x1, x2]) = [-x2, x1]; the tables repeat each angle in both halves.
    out[first] = static_cast<T>(x1 * cos_table[table] - x2 * sin_table[table]);
    out[second] = static_cast<T>(x2 * cos_table[table + half_dim] + x1 * sin_table[table + half_dim]);
}

instantiate_kernel("rope_f32", rope, float);
instantiate_kernel("rope_f16", rope, half);
instantiate_kernel("rope_bf16", rope, bfloat16_t);
