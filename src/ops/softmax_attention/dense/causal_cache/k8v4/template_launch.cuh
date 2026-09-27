#pragma once

#include "core/device.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/grouped_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/k8v4/tiled_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/k8v4/merge.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {

template <class G, bool Writable>
void validate_k8v4_kv_operands(const K8V4KvOperands& p, K8V4KvCacheView<Writable> cache) {
    if (p.query_heads != G::QHeads || cache.kv_heads != G::KVHeads || !p.q || !p.positions ||
        !p.out || !cache.keys || !cache.values || !cache.key_scales || !cache.value_scales ||
        !cache.tables || p.width < 1 || p.batch < 1 || p.visible_capacity < 1 ||
        static_cast<std::int64_t>(p.visible_capacity) >
            static_cast<std::int64_t>(cache.table_stride) * kPagedKVPageSize)
        throw std::invalid_argument("K8V4 attention template: invalid operands");
}

template <class G, class S, bool MultiBatch, bool Masked, bool Writable, class Input,
          bool ParallelQueries = false>
void launch_k8v4_kv_grouped_mma(const K8V4KvOperands& p, K8V4KvCacheView<Writable> cache,
                                Input input, K8V4KvPartition partition, K8V4KvPartialView partial,
                                cudaStream_t stream) {
    static_assert(Writable == Input::writes_cache);
    validate_k8v4_kv_operands<G>(p, cache);
    if ((!ParallelQueries && p.width != S::kTokenTile) || MultiBatch != (p.batch > 1) ||
        Masked != (cache.valid_columns != nullptr) || partition.capacity < 1 ||
        partition.target > K8V4KvPartition::kMaxSplits || partition.target < 1 ||
        partition.key_shift < 6 || partition.key_shift > 12 ||
        partition.capacity != partition.active(p.visible_capacity) || !partial.acc ||
        !partial.maximum || !partial.sum)
        throw std::invalid_argument("K8V4 grouped attention: invalid schedule/partials");
    if constexpr (Input::writes_cache)
        if (!input.k || !input.v) throw std::invalid_argument("K8V4 append requires K/V");
    constexpr auto kernel =
        k8v4_kv_grouped_mma_kernel<G, S, MultiBatch, Masked, Input, ParallelQueries>;
    constexpr int bytes = S::kDynamicArena ? S::kArenaBytes : 0;
    if constexpr (S::kDynamicArena) {
        static const auto status =
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        CUDA_CHECK(status);
    }
    const dim3 grid(G::KVHeads * (ParallelQueries ? div_up(p.width, S::kTokenTile) : 1),
                    partition.capacity, p.batch);
    kernel<<<grid, S::kThreads, bytes, stream>>>(
        p.q, input, p.positions, cache.keys, cache.values, cache.key_scales, cache.value_scales,
        cache.tables, cache.valid_columns, cache.table_rows, cache.table_stride, p.width,
        p.visible_capacity, partition, p.scale, partial.acc, partial.maximum, partial.sum);
    CUDA_CHECK(cudaGetLastError());
}

template <class G, class S, bool MultiBatch, bool Masked, bool Writable>
void launch_k8v4_kv_merge(const K8V4KvOperands& p, K8V4KvCacheView<Writable> cache,
                          K8V4KvPartition partition, K8V4KvPartialView partial,
                          cudaStream_t stream) {
    const dim3 grid(G::QHeads, div_up(G::kHeadDim, S::kDChunk), p.width * p.batch);
    k8v4_kv_merge_kernel<G, S, MultiBatch, Masked>
        <<<grid, S::kThreads, 0, stream>>>(partial.acc, partial.maximum, partial.sum, p.positions,
                                           cache.valid_columns, p.width, p.batch, partition, p.out);
    CUDA_CHECK(cudaGetLastError());
}

template <class G, class S>
void launch_k8v4_kv_tiled_mma(const K8V4KvOperands& p, K8V4KvReadView cache, cudaStream_t stream) {
    validate_k8v4_kv_operands<G>(p, cache);
    if (p.batch != 1)
        throw std::invalid_argument("K8V4 tiled attention requires a complete single query row");
    const auto invoke = [&]<class Metadata>(Metadata metadata) {
        constexpr auto kernel    = k8v4_kv_tiled_mma_kernel<G, S, Metadata>;
        static const auto status = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, S::kSharedBytes);
        CUDA_CHECK(status);
        const dim3 grid(div_up(p.width, S::kQueryRows), G::QHeads);
        kernel<<<grid, S::kThreads, S::kSharedBytes, stream>>>(
            p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
            p.positions, p.scale, p.out, p.width);
        CUDA_CHECK(cudaGetLastError());
    };
    if (!cache.table_rows)
        invoke(PagedKVDirectMetadata{cache.tables});
    else if (cache.valid_columns)
        invoke(PagedKVBatchMetadata<true>{cache.tables, cache.valid_columns, cache.table_rows,
                                          cache.table_stride});
    else
        invoke(PagedKVBatchMetadata<false>{cache.tables, nullptr, cache.table_rows,
                                           cache.table_stride});
}

} // namespace ninfer::ops::detail
