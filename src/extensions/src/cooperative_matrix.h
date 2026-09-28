#pragma once

// Tile staging for simdgroup_matrix kernels, lifted from tiny-llm's cooperative_matrix.h.

#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

namespace mini_vllm {

// Copy one fixed-shape row-major device tile into padded threadgroup storage. Each thread
// owns one contiguous source chunk; partial rows and columns are written as zero, so every
// later matrix load can read the full logical tile.
template <typename T, int ROWS, int COLS, int DESTINATION_STRIDE, int THREADS, bool TRANSPOSE_DESTINATION = false>
struct CooperativeTileLoader {
    static_assert((ROWS * COLS) % THREADS == 0);
    static_assert(COLS % ((ROWS * COLS) / THREADS) == 0);

    static METAL_FUNC void load(
        device const T* source,
        int source_stride,
        threadgroup T* destination,
        uint thread_index,
        int valid_rows = ROWS,
        int valid_columns = COLS) {
        constexpr int values_per_thread = (ROWS * COLS) / THREADS;
        constexpr int threads_per_row = COLS / values_per_thread;
        const int row = int(thread_index) / threads_per_row;
        const int column = (int(thread_index) % threads_per_row) * values_per_thread;

        #pragma unroll
        for (int offset = 0; offset < values_per_thread; ++offset) {
            const int current_column = column + offset;
            const T value = row < valid_rows && current_column < valid_columns
                ? source[row * source_stride + current_column]
                : T(0);
            if constexpr (TRANSPOSE_DESTINATION) {
                destination[current_column * DESTINATION_STRIDE + row] = value;
            } else {
                destination[row * DESTINATION_STRIDE + current_column] = value;
            }
        }
    }
};

// Where lane `lane` of a SIMD group holds its two elements of an 8x8 simdgroup_matrix.
METAL_FUNC ushort2 matrix_coordinate(ushort lane) {
    return ushort2((lane & 1) * 2 + (lane & 8) / 2, (lane & 7) / 2 + (lane & 16) / 4);
}

template <typename T>
METAL_FUNC void load_matrix(
    thread simdgroup_matrix<T, 8, 8>& matrix,
    threadgroup const T* source,
    int row_stride,
    ushort lane) {
    const ushort2 coordinate = matrix_coordinate(lane);
    matrix.thread_elements()[0] = source[coordinate.y * row_stride + coordinate.x];
    matrix.thread_elements()[1] = source[coordinate.y * row_stride + coordinate.x + 1];
}

}  // namespace mini_vllm
