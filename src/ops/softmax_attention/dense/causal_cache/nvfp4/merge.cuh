#pragma once

#include "ops/softmax_attention/dense/causal_cache/nvfp4/epilogue.cuh"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/split_policy.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/softmax.cuh"

namespace ninfer::ops::detail {

template <class Geometry>
__device__ __forceinline__ float
nvfp4_kv_merge_statistics(const float* partial_m, const float* partial_l, int q_head, int token,
                          int tokens, int splits, float* weights, float* warp_sums,
                          float* scalars) {
    static_assert(Nvfp4KvPartition::kMaxSplits <= 256);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const auto index   = nvfp4_kv_stat_index<Geometry>(q_head, token, tid, tokens);
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

template <class Geometry, class Schedule, bool MultiBatch, bool Masked>
__launch_bounds__(256) __global__
    void nvfp4_kv_merge_kernel(const float* partial_acc, const float* partial_m,
                               const float* partial_l, const std::int32_t* positions,
                               const std::int32_t* valid_columns, std::int32_t tokens,
                               std::int32_t batch_size, Nvfp4KvPartition partition,
                               __nv_bfloat16* out) {
    static_assert(Schedule::kDChunk == 256, "Inverse rotation merges the full D256 row");
    constexpr int DChunk  = Schedule::kDChunk;
    const int q_head      = static_cast<int>(blockIdx.x);
    const int d_start     = static_cast<int>(blockIdx.y) * DChunk;
    const int flat_column = static_cast<int>(blockIdx.z);
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid         = static_cast<int>(threadIdx.x);
    const int split_count = partition.capacity;
    if (q_head >= Geometry::QHeads || token >= tokens) return;
    if constexpr (MultiBatch) {
        if (batch >= batch_size) return;
    }
    if constexpr (MultiBatch) positions += static_cast<std::int64_t>(batch) * tokens;
    const int window  = positions[tokens - 1] + 1;
    int output_column = token;
    if constexpr (MultiBatch) output_column += batch * tokens;
    if constexpr (Masked) {
        const int absolute_column = token;
        if (absolute_column >= valid_columns[batch]) {
            if (tid < DChunk && d_start + tid < 256)
                out[nvfp4_kv_q_index<Geometry>(q_head, d_start + tid, output_column)] =
                    __float2bfloat16(0.0f);
            return;
        }
    }


    if constexpr (MultiBatch) {
        partial_acc +=
            static_cast<std::int64_t>(batch) * 256 * Geometry::QHeads * tokens * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }
    const int active_splits = partition.active(window);
    __shared__ float weights[256], warp_sums[8], scalars[2];
    const float head_l = nvfp4_kv_merge_statistics<Geometry>(
        partial_m, partial_l, q_head, token, tokens, active_splits, weights, warp_sums, scalars);
    const int d = d_start + tid;
    if (tid >= DChunk || d >= 256) return;
    float numerator = 0.0f;
    for (int split = 0; split < active_splits; ++split) {
        if (weights[split] != 0.0f)
            numerator +=
                partial_acc[nvfp4_kv_partial_index<Geometry>(q_head, d, token, split, tokens)] *
                weights[split];
    }

    __shared__ float normalized[256];
    normalized[tid] = head_l > 0.0F ? numerator / head_l : 0.0F;
    __syncthreads();
    if (tid < 32) nvfp4_kv_store_rotated_row<Geometry>(normalized, out, q_head, output_column);
}


} // namespace ninfer::ops::detail
