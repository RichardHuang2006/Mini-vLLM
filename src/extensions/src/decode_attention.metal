#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Dense attention for a few query tokens per head (a decode step), ported from tiny-llm's
// week2_decode_attention. q is [B * H_q, L, D], k and v [B * H_k, S, D]. One threadgroup
// per query token: each of its 32 SIMD groups walks every 32nd key with an online softmax,
// each lane owning D / 32 dimensions, and the 32 partials merge through shared memory.
template <typename T>
[[kernel]] void decode_attention(
    device const T* q [[buffer(0)]],
    device const T* k [[buffer(1)]],
    device const T* v [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant const int& length [[buffer(4)]],
    constant const int& context [[buffer(5)]],
    constant const int& dim [[buffer(6)]],
    constant const int& num_heads [[buffer(7)]],
    constant const int& num_kv_heads [[buffer(8)]],
    constant const float& scale [[buffer(9)]],
    constant const int& causal [[buffer(10)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint query_index [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int MAX_PER_LANE = 8;  // D <= 256
    constexpr int SIMDGROUPS = 32;
    const int query_row = query_index / length;
    const int query_position = query_index % length;
    const int batch = query_row / num_heads;
    const int kv_head = (query_row % num_heads) / (num_heads / num_kv_heads);
    const int kv_row = batch * num_kv_heads + kv_head;
    const int per_lane = (dim + 31) / 32;
    // Causal: query i of L new tokens sees keys up to S - L + i.
    const int visible = causal ? context - length + query_position + 1 : context;

    float query[MAX_PER_LANE] = {0.0f};
    float accumulator[MAX_PER_LANE] = {0.0f};
    for (int item = 0; item < per_lane; ++item) {
        const int d = lane + item * 32;
        if (d < dim) query[item] = static_cast<float>(q[query_index * dim + d]) * scale;
    }

    float max_score = -1e30f;
    float sum = 0.0f;
    for (int position = simdgroup; position < visible; position += SIMDGROUPS) {
        float partial = 0.0f;
        for (int item = 0; item < per_lane; ++item) {
            const int d = lane + item * 32;
            if (d < dim) partial += query[item] * static_cast<float>(k[(kv_row * context + position) * dim + d]);
        }
        const float score = simd_sum(partial);
        const float new_max = max(max_score, score);
        const float old_factor = fast::exp(max_score - new_max);
        const float factor = fast::exp(score - new_max);
        sum = sum * old_factor + factor;
        for (int item = 0; item < per_lane; ++item) {
            const int d = lane + item * 32;
            if (d < dim) {
                accumulator[item] = accumulator[item] * old_factor +
                                    factor * static_cast<float>(v[(kv_row * context + position) * dim + d]);
            }
        }
        max_score = new_max;
    }

    threadgroup float* partial_values = scratch;                     // [SIMDGROUPS][dim]
    threadgroup float* partial_maxima = partial_values + SIMDGROUPS * dim;
    threadgroup float* partial_sums = partial_maxima + SIMDGROUPS;
    threadgroup float* factors = partial_sums + SIMDGROUPS;
    if (lane == 0) {
        partial_maxima[simdgroup] = max_score;
        partial_sums[simdgroup] = sum;
    }
    for (int item = 0; item < per_lane; ++item) {
        const int d = lane + item * 32;
        if (d < dim) partial_values[simdgroup * dim + d] = accumulator[item];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        const float global_max = simd_max(partial_maxima[lane]);
        factors[lane] = fast::exp(partial_maxima[lane] - global_max);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
        float global_sum = 0.0f;
        for (int group = 0; group < SIMDGROUPS; ++group) global_sum += partial_sums[group] * factors[group];
        partial_sums[0] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < uint(dim)) {
        float value = 0.0f;
        for (int group = 0; group < SIMDGROUPS; ++group) {
            value += partial_values[group * dim + thread_index] * factors[group];
        }
        out[query_index * dim + thread_index] = static_cast<T>(value / partial_sums[0]);
    }
}

instantiate_kernel("decode_attention_f32", decode_attention, float);
instantiate_kernel("decode_attention_f16", decode_attention, half);
instantiate_kernel("decode_attention_bf16", decode_attention, bfloat16_t);
