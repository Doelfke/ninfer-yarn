#pragma once

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/operands.h"

namespace ninfer::ops::detail {

template <typename Geometry>
__device__ __forceinline__ std::int64_t nvfp4_kv_q_index(int q_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(256) * (static_cast<std::int64_t>(q_head) +
                                             static_cast<std::int64_t>(Geometry::QHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t nvfp4_kv_partial_index(int q_head, int d, int token,
                                                               int split, int tokens) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(256) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(Geometry::QHeads) *
                    (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split));
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t nvfp4_kv_stat_index(int q_head, int token, int split,
                                                            int tokens) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(Geometry::QHeads) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split);
}

__device__ __forceinline__ int nvfp4_kv_swizzle(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

template <typename Byte>
__device__ __forceinline__ void nvfp4_kv_store_query_code(Byte* tile, int row, int d, Byte code) {
    const int col_b16 = d >> 1;
    const int byte    = d & 1;
    const int off     = (row * 128 + nvfp4_kv_swizzle(row, col_b16)) * 2 + byte;
    tile[off]         = code;
}

template <typename Geometry>
__device__ __forceinline__ void nvfp4_kv_row_to_qt(int row, int kv_head, int& q_head, int& token) {
    token             = row / Geometry::GroupSize;
    const int local_q = row - token * Geometry::GroupSize;
    q_head            = kv_head * Geometry::GroupSize + local_q;
}

template <typename Geometry>
__device__ __forceinline__ void nvfp4_kv_zero_rows(__nv_bfloat16* out, int q_head, int row_begin,
                                                   int row_end, int tid, int threads) {
    if (row_begin >= row_end) { return; }
    const int elements = (row_end - row_begin) * 256;
    for (int element = tid; element < elements; element += threads) {
        const int row                                   = row_begin + element / 256;
        const int d                                     = element - (row - row_begin) * 256;
        out[nvfp4_kv_q_index<Geometry>(q_head, d, row)] = __float2bfloat16(0.0f);
    }
}

template <int Columns>
__device__ __forceinline__ int nvfp4_kv_probability_swizzle(int row, int col) {
    if constexpr (Columns == 32) { return (((col >> 3) ^ (row & 3)) << 3) | (col & 7); }
    return nvfp4_kv_swizzle(row, col);
}

__device__ __forceinline__ int nvfp4_kv_code_swizzle(int row, int byte) {
    return (((byte >> 4) ^ (row & 7)) << 4) | (byte & 15);
}

__device__ __forceinline__ unsigned nvfp4_kv_swizzle_address(unsigned lane_base, unsigned ck,
                                                             unsigned as, unsigned row) {
    return lane_base + ((ck | as) ^ row);
}

__device__ __forceinline__ unsigned nvfp4_kv_pack_f16x2(float a, float b) {
    const __half2 packed = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&packed);
}

} // namespace ninfer::ops::detail
