#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/fp8.h"

using namespace metal;

// ------------------------------- the cache write -------------------------------

// pool [num_slots, H_k * D] gets row t of values [T, H_k * D] at slot_mapping[t]. The pool
// buffer is shared with the output, so nothing else in it moves.
template <typename T>
[[kernel]] void paged_cache_update(
    device const int* slot_mapping [[buffer(0)]],
    device const T* values [[buffer(1)]],
    device T* pool [[buffer(2)]],
    constant const int& row [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const int token = index / row;
    pool[size_t(slot_mapping[token]) * row + index % row] = values[index];
}

instantiate_kernel("paged_cache_update_f32", paged_cache_update, float);
instantiate_kernel("paged_cache_update_f16", paged_cache_update, half);
instantiate_kernel("paged_cache_update_bf16", paged_cache_update, bfloat16_t);

// ------------------------------- attention -------------------------------

// A page element as fp32: plain for bf16/fp16/fp32 pages, dequantized for fp8 (uint8) ones.
template <typename P>
inline float load_page(device const P* pages, size_t index, float) {
    return static_cast<float>(pages[index]);
}

template <>
inline float load_page(device const uint8_t* pages, size_t index, float scale) {
    uint8_t bits = pages[index];
    return static_cast<float>(*(thread fp8_e4m3*)(&bits)) * scale;
}

// Causal grouped attention over the ragged batch, straight from the pages.
//
// q is [T, H_q, D], the pages are [num_blocks, P, H_k, D] (token-major, as the pool stores
// them), block_tables is [N, max_blocks] padded with -1, cu_seqlens_q [N + 1] says where
// each sequence's tokens start, and context_lens [N] how many tokens each attends over.
//
// One threadgroup per (token, query head). Each of its 32 SIMD groups walks every 32nd
// visible key, keeping a running max, sum and weighted value sum (online softmax, in base
// 2); each lane owns D / 32 dimensions. The 32 partials are merged at the end.
template <typename T, typename P>
[[kernel]] void paged_attention(
    device const T* q [[buffer(0)]],
    device const P* key_pages [[buffer(1)]],
    device const P* value_pages [[buffer(2)]],
    device const int* block_tables [[buffer(3)]],
    device const int* cu_seqlens_q [[buffer(4)]],
    device const int* context_lens [[buffer(5)]],
    device T* out [[buffer(6)]],
    constant const int& num_heads [[buffer(7)]],
    constant const int& num_kv_heads [[buffer(8)]],
    constant const int& head_dim [[buffer(9)]],
    constant const int& block_size [[buffer(10)]],
    constant const int& num_sequences [[buffer(11)]],
    constant const int& max_blocks [[buffer(12)]],
    constant const float& scale_log2 [[buffer(13)]],
    constant const float& k_scale [[buffer(14)]],
    constant const float& v_scale [[buffer(15)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int SIMDGROUPS = 32;
    constexpr int MAX_PER_LANE = 4;  // D <= 128
    const int token = group / num_heads;
    const int head = group - token * num_heads;
    const int kv_head = head / (num_heads / num_kv_heads);
    const int per_lane = (head_dim + 31) / 32;

    // Which sequence this token belongs to: the last start at or before it.
    int low = 0, high = num_sequences - 1;
    while (low < high) {
        const int mid = (low + high + 1) / 2;
        if (cu_seqlens_q[mid] <= token) low = mid; else high = mid - 1;
    }
    const int sequence = low;
    const int query_len = cu_seqlens_q[sequence + 1] - cu_seqlens_q[sequence];
    const int query_position = token - cu_seqlens_q[sequence];
    const int context = context_lens[sequence];
    // Causal: query i of an L-token chunk sees keys up to S - L + i.
    const int visible = context - query_len + query_position + 1;
    device const int* table = block_tables + sequence * max_blocks;

    float query[MAX_PER_LANE] = {0.0f};
    float accumulator[MAX_PER_LANE] = {0.0f};
    for (int item = 0; item < per_lane; ++item) {
        const int dim = lane * per_lane + item;
        if (dim < head_dim) {
            query[item] = static_cast<float>(q[(size_t(token) * num_heads + head) * head_dim + dim]) * scale_log2;
        }
    }

    float max_score = -INFINITY;
    float sum = 0.0f;
    const int row = num_kv_heads * head_dim;  // one slot of the pool
    for (int position = simdgroup; position < visible; position += SIMDGROUPS) {
        const int slot = table[position / block_size] * block_size + position % block_size;
        const size_t base = size_t(slot) * row + kv_head * head_dim;

        float partial = 0.0f;
        for (int item = 0; item < per_lane; ++item) {
            const int dim = lane * per_lane + item;
            if (dim < head_dim) {
                partial += query[item] * load_page(key_pages, base + dim, k_scale);
            }
        }
        const float score = simd_sum(partial);
        const float new_max = max(max_score, score);
        const float old_factor = fast::exp2(max_score - new_max);
        const float factor = fast::exp2(score - new_max);
        sum = sum * old_factor + factor;
        for (int item = 0; item < per_lane; ++item) {
            const int dim = lane * per_lane + item;
            if (dim < head_dim) {
                accumulator[item] = accumulator[item] * old_factor +
                    factor * load_page(value_pages, base + dim, v_scale);
            }
        }
        max_score = new_max;
    }

    // Merge the 32 partials. Each SIMD group's (max, sum) goes to shared memory; every
    // lane rescales by exp2(its group's max - the global max).
    threadgroup float* outputs = scratch;                       // [32 dims-lanes][32 groups]
    threadgroup float* maxima = outputs + SIMDGROUPS * 32;
    threadgroup float* sums = maxima + SIMDGROUPS;
    if (lane == 0) {
        maxima[simdgroup] = max_score;
        sums[simdgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float group_max = maxima[lane];
    const float global_max = simd_max(group_max);
    // A SIMD group that saw no key has max -inf: its factor is 0, not NaN.
    const float factor = group_max == -INFINITY ? 0.0f : fast::exp2(group_max - global_max);
    const float global_sum = simd_sum(sums[lane] * factor);

    // Transpose through shared memory so SIMD group g reduces dimension-lane g's values.
    for (int item = 0; item < per_lane; ++item) {
        outputs[lane * SIMDGROUPS + simdgroup] = accumulator[item];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        accumulator[item] = simd_sum(outputs[simdgroup * SIMDGROUPS + lane] * factor);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (int item = 0; item < per_lane; ++item) {
            const int dim = simdgroup * per_lane + item;
            if (dim < head_dim) {
                out[(size_t(token) * num_heads + head) * head_dim + dim] = static_cast<T>(accumulator[item] / global_sum);
            }
        }
    }
}

instantiate_kernel("paged_attention_f32", paged_attention, float, float);
instantiate_kernel("paged_attention_f16", paged_attention, half, half);
instantiate_kernel("paged_attention_bf16", paged_attention, bfloat16_t, bfloat16_t);
instantiate_kernel("paged_attention_fp8_f32", paged_attention, float, uint8_t);
instantiate_kernel("paged_attention_fp8_f16", paged_attention, half, uint8_t);
instantiate_kernel("paged_attention_fp8_bf16", paged_attention, bfloat16_t, uint8_t);
