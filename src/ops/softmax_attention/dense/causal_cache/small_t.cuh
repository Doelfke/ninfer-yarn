#pragma once

// Quantized-cache split-KV scaffolding: layout helpers, split policies and the
// statistics merge used by the remaining legacy formats.

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCausalHeadDim = 256;

struct CausalAppendInput {
    static constexpr bool writes_cache = true;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
};

struct CausalCachedInput {
    static constexpr bool writes_cache = false;
};

template <typename Geometry>
__device__ __forceinline__ std::int64_t causal_cache_index(int physical_page, int kv_head, int d,
                                                           int page_offset) {
    return paged_kv_element_offset<kCausalHeadDim, Geometry::KVHeads>(physical_page, kv_head,
                                                                      page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t causal_q_index(int q_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kCausalHeadDim) *
                                              (static_cast<std::int64_t>(q_head) +
                                               static_cast<std::int64_t>(Geometry::QHeads) * token);
}


template <typename Geometry>
__device__ __forceinline__ std::int64_t causal_partial_acc_index(int q_head, int d, int token,
                                                                 int split, int tokens) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kCausalHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(Geometry::QHeads) *
                    (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split));
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t causal_partial_stat_index(int q_head, int token, int split,
                                                                  int tokens) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(Geometry::QHeads) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split);
}

template <typename Geometry>
__device__ __forceinline__ bool causal_valid_q_head(int kv_head, int q_head) {
    return kv_head >= 0 && kv_head < Geometry::KVHeads && q_head >= kv_head * Geometry::GroupSize &&
           q_head < (kv_head + 1) * Geometry::GroupSize && q_head < Geometry::QHeads;
}

template <typename Geometry>
__device__ __forceinline__ int causal_small_t_default_splits(int window) {
    int target_keys_per_split = 480 / Geometry::SmallTSplitScale;
    if (window <= 4096) {
        target_keys_per_split = 64 / Geometry::SmallTSplitScale;
    } else if (window <= 8198) {
        target_keys_per_split = 128 / Geometry::SmallTSplitScale;
    } else if (window <= 16390) {
        target_keys_per_split = 256 / Geometry::SmallTSplitScale;
    }
    constexpr int kMinSplits = 4 * Geometry::SmallTSplitScale;
    int splits               = div_up(window, target_keys_per_split);
    splits                   = splits > kMinSplits ? splits : kMinSplits;
    splits                   = splits < Geometry::SmallTMaximumSplits ? splits : Geometry::SmallTMaximumSplits;
    if constexpr (Geometry::SmallTSplitScale == 1) {
        // Page-safety floor (must mirror causal_small_t_split_upper_bound in small_t.cu). Keep
        // keys/split <= 3968 so each split's page span fits __shared__ physical_pages_s[64],
        // up to the 256-split ceiling of the split reducer.
        constexpr int kSplitsCeiling = 256;
        const int page_limit         = div_up(window, 3968);
        splits                       = splits > page_limit ? splits : page_limit;
        return splits < kSplitsCeiling ? splits : kSplitsCeiling;
    }
    return splits;
}

template <typename Geometry>
__device__ __forceinline__ int
causal_small_t_quantized_active_splits(int window, int launch_capacity, int tokens) {
    // causal_small_t_default_splits already applies the page-safety floor and the
    // 256-split reducer ceiling for the H24 geometry, so it must not be clamped back here.
    // The old `tokens == 1` fallback to SmallTMaximumSplits (85) under-provisioned splits
    // at large windows and read out of bounds in __shared__ physical_pages_s[64].
    (void)tokens;
    const int splits = causal_small_t_default_splits<Geometry>(window);
    return splits < launch_capacity ? splits : launch_capacity;
}

__device__ __forceinline__ int causal_small_t_tc_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

template <typename Byte>
__device__ __forceinline__ void causal_small_t_store_byte_swizzled(Byte* tile, int row, int d,
                                                                   int d_b16_stride, Byte code) {
    const int col_b16 = d >> 1;
    const int byte    = d & 1;
    const int off     = (row * d_b16_stride + causal_small_t_tc_swz(row, col_b16)) * 2 + byte;
    tile[off]         = code;
}

__device__ __forceinline__ int causal_small_t_tc_swz32(int row, int col) {
    return (((col >> 3) ^ (row & 3)) << 3) | (col & 7);
}

// Signed int8 QK MMA, k=32 contraction. A = 16x32 s8 (4 regs/thread, 4 s8 each),
// B = 8x32 s8 col-major (2 regs/thread), D = 16x8 s32 (4 regs/thread). The A/B
// register byte layout is identical to the m16n8k16 bf16 fragments loaded by
// ldmatrix_x4/x2 over a d-contiguous int8 tile reinterpreted as
// b16 (two packed int8 per 16-bit lane), so the same ldmatrix helpers and XOR
// swizzle feed this MMA. The s32 accumulator layout matches the bf16 f32
// accumulator (c0/c1 -> row groupID, c2/c3 -> row groupID+8), so score
// consumption is unchanged; only per-64-group scale rescale differs.
template <typename Geometry>
__device__ __forceinline__ void causal_small_t_tc_row_to_qt(int row, int tokens, int kv_head,
                                                            int& q_head, int& token) {
    token             = row / Geometry::GroupSize;
    const int local_q = row - token * Geometry::GroupSize;
    q_head            = kv_head * Geometry::GroupSize + local_q;
}

// Merge one query/head's split statistics once per CTA. Published scalars are separate
// from the reduction/weight storage, so later writes cannot race another warp's scalar read.
template <class Geometry>
__device__ __forceinline__ float
causal_merge_split_statistics(const float* partial_m, const float* partial_l, int q_head, int token,
                              int tokens, int splits, float* weights, float* warp_sums,
                              float* scalars) {
    static_assert(Geometry::SmallTMaximumSplits <= 256);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const auto index   = causal_partial_stat_index<Geometry>(q_head, token, tid, tokens);
    const float m      = tid < splits ? partial_m[index] : -CUDART_INF_F;
    const float warp_m = warp_max(m);
    if (lane == 0) warp_sums[warp] = warp_m;
    __syncthreads();
    if (warp == 0) {
        const float maximum = warp_max(tid < 8 ? warp_sums[tid] : -CUDART_INF_F);
        if (tid == 0) scalars[0] = maximum;
    }
    __syncthreads();
    const float maximum     = scalars[0];
    const float l           = tid < splits ? partial_l[index] : 0.0f;
    const float weight      = l > 0.0f && maximum > -CUDART_INF_F ? expf(m - maximum) : 0.0f;
    const float denominator = block_reduce_sum<256>(l * weight, warp_sums);
    if (tid == 0) scalars[1] = denominator;
    __syncthreads();
    const float total = scalars[1];
    if (tid < splits) weights[tid] = total > 0.0f ? weight : 0.0f;
    __syncthreads();
    return total;
}


} // namespace ninfer::ops
