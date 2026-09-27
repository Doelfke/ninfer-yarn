#pragma once

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/fp8/operands.h"

namespace ninfer::ops::detail {

template <typename Geometry>
__device__ __forceinline__ std::int64_t fp8_kv_q_index(int q_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(256) * (static_cast<std::int64_t>(q_head) +
                                             static_cast<std::int64_t>(Geometry::QHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t fp8_kv_partial_index(int q_head, int d, int token,
                                                             int split, int tokens) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(256) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(Geometry::QHeads) *
                    (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split));
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t fp8_kv_stat_index(int q_head, int token, int split,
                                                          int tokens) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(Geometry::QHeads) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split);
}

template <typename Geometry>
__device__ __forceinline__ bool fp8_kv_valid_head(int kv_head, int q_head) {
    return kv_head >= 0 && kv_head < Geometry::KVHeads && q_head >= kv_head * Geometry::GroupSize &&
           q_head < (kv_head + 1) * Geometry::GroupSize && q_head < Geometry::QHeads;
}

__device__ __forceinline__ int fp8_kv_swizzle(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

template <typename Byte>
__device__ __forceinline__ void fp8_kv_store_query_code(Byte* tile, int row, int d, Byte code) {
    const int col_b16 = d >> 1;
    const int byte    = d & 1;
    const int off     = (row * 128 + fp8_kv_swizzle(row, col_b16)) * 2 + byte;
    tile[off]         = code;
}

template <typename Geometry>
__device__ __forceinline__ void fp8_kv_row_to_qt(int row, int kv_head, int& q_head, int& token) {
    token             = row / Geometry::GroupSize;
    const int local_q = row - token * Geometry::GroupSize;
    q_head            = kv_head * Geometry::GroupSize + local_q;
}

template <typename Geometry>
__device__ __forceinline__ void fp8_kv_zero_rows(__nv_bfloat16* out, int q_head, int row_begin,
                                                 int row_end, int tid, int threads) {
    if (row_begin >= row_end) { return; }
    const int elements = (row_end - row_begin) * 256;
    for (int element = tid; element < elements; element += threads) {
        const int row                                 = row_begin + element / 256;
        const int d                                   = element - (row - row_begin) * 256;
        out[fp8_kv_q_index<Geometry>(q_head, d, row)] = __float2bfloat16(0.0f);
    }
}

template <int Columns>
__device__ __forceinline__ int fp8_kv_probability_swizzle(int row, int col) {
    if constexpr (Columns == 32) { return (((col >> 3) ^ (row & 3)) << 3) | (col & 7); }
    return fp8_kv_swizzle(row, col);
}

__device__ __forceinline__ int4 fp8_kv_dequant_f16x8(const std::uint8_t* codes, __half scale) {
    const int2 raw         = load_vec<int2>(codes);
    const std::uint16_t* c = reinterpret_cast<const std::uint16_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 value2 = kv_cache_fp8_dequant_code2_to_half2(c[i], scale);
        packed[i]            = *reinterpret_cast<const unsigned*>(&value2);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

} // namespace ninfer::ops::detail
