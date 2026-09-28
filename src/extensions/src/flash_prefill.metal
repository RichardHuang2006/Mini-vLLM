#include <metal_simdgroup_matrix>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "cooperative_matrix.h"

using namespace metal;

namespace {

constant constexpr int MMA = 8;
using FloatTile = simdgroup_matrix<float, MMA, MMA>;
using BfloatTile = simdgroup_matrix<bfloat, MMA, MMA>;

// Row reductions over a row of 8x8 fragments: each row's 8 values sit on 4 lanes, two each.
template <int N>
inline float row_max(thread const FloatTile* tiles) {
    float value = -INFINITY;
    for (int i = 0; i < N; ++i) {
        const auto elements = tiles[i].thread_elements();
        value = max(value, max(elements[0], elements[1]));
    }
    value = max(value, simd_shuffle_xor(value, ushort(1)));
    return max(value, simd_shuffle_xor(value, ushort(8)));
}

template <int N>
inline float row_sum(thread const FloatTile* tiles) {
    float value = 0.0f;
    for (int i = 0; i < N; ++i) {
        const auto elements = tiles[i].thread_elements();
        value += elements[0] + elements[1];
    }
    value += simd_shuffle_xor(value, ushort(1));
    return value + simd_shuffle_xor(value, ushort(8));
}

inline void multiply_accumulate(thread FloatTile& accumulator, thread BfloatTile& left, thread BfloatTile& right) {
    FloatTile result;
    simdgroup_multiply_accumulate(result, left, right, accumulator);
    accumulator = result;
}

}  // namespace

// FlashAttention-style tiled prefill, bf16 with D = 128, ported from tiny-llm's
// week2_dense_prefill_mma. q is [B * H_q, L, 128], k and v [B * H_k, S, 128]. Each
// threadgroup takes 32 queries of one head; each of its 4 SIMD groups owns 8 of them.
// Keys stream through in tiles of 16: scores by simdgroup_matrix MMA, an online softmax
// in base 2, then P @ V by MMA into 16 fp32 output fragments per SIMD group.
[[kernel, max_total_threads_per_threadgroup(128)]] void flash_prefill_bf16_d128(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant const int& length [[buffer(4)]],
    constant const int& context [[buffer(5)]],
    constant const float& scale [[buffer(6)]],
    constant const int& causal [[buffer(7)]],
    constant const int& num_heads [[buffer(8)]],
    constant const int& num_kv_heads [[buffer(9)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    ushort simd_gid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int HEAD_DIM = 128;
    constexpr int BQ = 32;
    constexpr int BK = 16;
    constexpr int THREADS = 128;
    constexpr int LDQ = HEAD_DIM + 2;  // padded rows avoid threadgroup bank conflicts
    constexpr int LDK = BK + 2;
    constexpr int LDV = HEAD_DIM + 2;
    constexpr int SCORE_TILES = BK / MMA;
    constexpr int OUTPUT_TILES = HEAD_DIM / MMA;
    constexpr float LOG2_E = 1.44269504089f;
    using QLoader = mini_vllm::CooperativeTileLoader<bfloat, BQ, HEAD_DIM, LDQ, THREADS>;
    using KLoader = mini_vllm::CooperativeTileLoader<bfloat, BK, HEAD_DIM, LDK, THREADS, true>;
    using VLoader = mini_vllm::CooperativeTileLoader<bfloat, BK, HEAD_DIM, LDV, THREADS>;

    const int query_block = group_id.x;
    const int query_row = group_id.y;
    const int batch = query_row / num_heads;
    const int kv_head = (query_row - batch * num_heads) / (num_heads / num_kv_heads);
    const int kv_row = batch * num_kv_heads + kv_head;
    const int thread_index = simd_gid * 32 + lane;
    const ushort2 coordinate = mini_vllm::matrix_coordinate(lane);
    const int query_position = query_block * BQ + simd_gid * MMA + coordinate.y;
    const bool query_valid = query_position < length;
    const int live_queries = clamp(length - query_block * BQ, 0, BQ);
    const float scale_log2 = scale * LOG2_E;

    threadgroup bfloat q_tile[BQ * LDQ];
    threadgroup bfloat kv_tile[HEAD_DIM * LDK];
    QLoader::load(q + (query_row * length + query_block * BQ) * HEAD_DIM, HEAD_DIM, q_tile, thread_index,
                  live_queries, HEAD_DIM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    FloatTile output[OUTPUT_TILES];
    for (int i = 0; i < OUTPUT_TILES; ++i) output[i] = make_filled_simdgroup_matrix<float, MMA, MMA>(0.0f);
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // Causal: this block's last query sees keys up to S - L + that query, so later tiles are skipped.
    const int total_tiles = (context + BK - 1) / BK;
    int tile_limit = total_tiles;
    if (causal) {
        const int last_key = min((query_block + 1) * BQ, length) - 1 + (context - length);
        tile_limit = clamp((last_key + BK) / BK, 0, total_tiles);
    }

    for (int tile = 0; tile < tile_limit; ++tile) {
        const int tile_start = tile * BK;
        const int live_keys = clamp(context - tile_start, 0, BK);
        KLoader::load(k + (kv_row * context + tile_start) * HEAD_DIM, HEAD_DIM, kv_tile, thread_index, live_keys,
                      HEAD_DIM);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        FloatTile scores[SCORE_TILES];
        for (int i = 0; i < SCORE_TILES; ++i) scores[i] = make_filled_simdgroup_matrix<float, MMA, MMA>(0.0f);
        for (int dim = 0; dim < HEAD_DIM; dim += MMA) {
            BfloatTile q_fragment;
            mini_vllm::load_matrix(q_fragment, q_tile + simd_gid * MMA * LDQ + dim, LDQ, lane);
            for (int key_tile = 0; key_tile < SCORE_TILES; ++key_tile) {
                BfloatTile k_fragment;
                mini_vllm::load_matrix(k_fragment, kv_tile + dim * LDK + key_tile * MMA, LDK, lane);
                multiply_accumulate(scores[key_tile], q_fragment, k_fragment);
            }
        }

        for (int key_tile = 0; key_tile < SCORE_TILES; ++key_tile) {
            thread auto& values = scores[key_tile].thread_elements();
            for (int element = 0; element < 2; ++element) {
                const int key_position = tile_start + key_tile * MMA + coordinate.x + element;
                bool valid = query_valid && key_position < context;
                if (causal) valid = valid && key_position <= query_position + (context - length);
                values[element] = valid ? values[element] * scale_log2 : -INFINITY;
            }
        }

        const float new_max = max(running_max, row_max<SCORE_TILES>(scores));
        const bool finite_row = query_valid && new_max != -INFINITY;
        const float previous_scale = running_max == -INFINITY || !finite_row ? 0.0f : fast::exp2(running_max - new_max);
        for (int i = 0; i < SCORE_TILES; ++i) {
            thread auto& values = scores[i].thread_elements();
            for (int element = 0; element < 2; ++element) {
                values[element] = values[element] == -INFINITY || !finite_row ? 0.0f : fast::exp2(values[element] - new_max);
            }
        }
        running_max = new_max;
        running_sum = previous_scale * running_sum + row_sum<SCORE_TILES>(scores);
        for (int i = 0; i < OUTPUT_TILES; ++i) {
            output[i].thread_elements()[0] *= previous_scale;
            output[i].thread_elements()[1] *= previous_scale;
        }

        BfloatTile probabilities[SCORE_TILES];
        for (int i = 0; i < SCORE_TILES; ++i) {
            probabilities[i].thread_elements()[0] = bfloat(scores[i].thread_elements()[0]);
            probabilities[i].thread_elements()[1] = bfloat(scores[i].thread_elements()[1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        VLoader::load(v + (kv_row * context + tile_start) * HEAD_DIM, HEAD_DIM, kv_tile, thread_index, live_keys,
                      HEAD_DIM);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int output_tile = 0; output_tile < OUTPUT_TILES; ++output_tile) {
            for (int key_tile = 0; key_tile < SCORE_TILES; ++key_tile) {
                BfloatTile value_fragment;
                mini_vllm::load_matrix(value_fragment, kv_tile + key_tile * MMA * LDV + output_tile * MMA, LDV, lane);
                multiply_accumulate(output[output_tile], probabilities[key_tile], value_fragment);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (query_valid) {
        for (int i = 0; i < OUTPUT_TILES; ++i) {
            const auto values = output[i].thread_elements();
            for (int element = 0; element < 2; ++element) {
                const int dim = i * MMA + coordinate.x + element;
                out[(query_row * length + query_position) * HEAD_DIM + dim] =
                    running_sum == 0.0f ? bfloat(0.0f) : bfloat(values[element] / running_sum);
            }
        }
    }
}
