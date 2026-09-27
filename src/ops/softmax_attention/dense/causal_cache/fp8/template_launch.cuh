#pragma once

#include "core/device.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/grouped_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/fp8/tiled_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/fp8/merge.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {

template <class G, bool Writable>
void validate_fp8_kv_operands(const Fp8KvOperands& p, Fp8KvCacheView<Writable> cache) {
    if (p.query_heads != G::QHeads || cache.kv_heads != G::KVHeads || !p.q || !p.positions ||
        !p.out || !cache.keys || !cache.values || !cache.key_scales || !cache.value_scales ||
        !cache.tables || p.width < 1 || p.batch < 1 || p.visible_capacity < 1 ||
        static_cast<std::int64_t>(p.visible_capacity) >
            static_cast<std::int64_t>(cache.table_stride) * kPagedKVPageSize)
        throw std::invalid_argument("FP8 attention template: invalid operands");
}

template <class G, class S, bool MultiBatch, bool Masked, bool Writable, class Input,
          bool ParallelQueries = false, bool Partial = true>
void launch_fp8_kv_grouped_mma(const Fp8KvOperands& p, Fp8KvCacheView<Writable> cache, Input input,
                               int splits, Fp8KvPartialView partial, cudaStream_t stream) {
    static_assert(Writable == Input::writes_cache);
    validate_fp8_kv_operands<G>(p, cache);
    if ((!ParallelQueries && p.width != S::kTokenTile) || MultiBatch != (p.batch > 1) ||
        Masked != (cache.valid_columns != nullptr) || splits < 1 ||
        splits > Fp8KvSplitPolicy<G::QHeads>::kMaxSplits ||
        (Partial && (!partial.acc || !partial.maximum || !partial.sum)) ||
        (!Partial && splits != 1))
        throw std::invalid_argument("FP8 grouped attention: invalid schedule/partials");
    if constexpr (!ParallelQueries) {
        // Fused append stages each split's page IDs. Include key-tile and page alignment;
        // parallel read-only tiles walk the table directly and have no such span limit.
        const int span = div_up(p.visible_capacity, splits) + 2 * S::kKeyRows;
        if (span > S::kPageIds * kPagedKVPageSize)
            throw std::invalid_argument("FP8 grouped split exceeds the staged page capacity");
    }
    if constexpr (Input::writes_cache)
        if (!input.k || !input.v) throw std::invalid_argument("FP8 append requires K/V");
    constexpr auto kernel =
        fp8_kv_grouped_mma_kernel<G, S, MultiBatch, Masked, Input, ParallelQueries, Partial>;
    constexpr int bytes = S::kDynamicArena ? S::kArenaBytes : 0;
    if constexpr (S::kDynamicArena) {
        static const auto status =
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        CUDA_CHECK(status);
    }
    const dim3 grid(G::KVHeads * (ParallelQueries ? div_up(p.width, S::kTokenTile) : 1), splits,
                    p.batch);
    kernel<<<grid, S::kThreads, bytes, stream>>>(
        p.q, input, p.positions, cache.keys, cache.values, cache.key_scales, cache.value_scales,
        cache.tables, cache.valid_columns, cache.table_rows, cache.table_stride, p.width,
        p.visible_capacity, p.scale, partial.acc, partial.maximum, partial.sum, p.out);
    CUDA_CHECK(cudaGetLastError());
}

template <class G, class S, bool MultiBatch, bool Masked, bool Writable>
void launch_fp8_kv_merge(const Fp8KvOperands& p, Fp8KvCacheView<Writable> cache, int splits,
                         Fp8KvPartialView partial, cudaStream_t stream) {
    const dim3 grid(G::QHeads, div_up(G::kHeadDim, S::kDChunk), p.width * p.batch);
    fp8_kv_merge_kernel<G, S, MultiBatch, Masked>
        <<<grid, S::kThreads, 0, stream>>>(partial.acc, partial.maximum, partial.sum, p.positions,
                                           cache.valid_columns, p.width, p.batch, splits, p.out);
    CUDA_CHECK(cudaGetLastError());
}

template <class G, class S>
void launch_fp8_kv_tiled_mma(const Fp8KvOperands& p, Fp8KvReadView cache, cudaStream_t stream) {
    validate_fp8_kv_operands<G>(p, cache);
    if (p.batch != 1)
        throw std::invalid_argument("FP8 tiled attention requires a complete single query row");
    const auto invoke = [&]<class Metadata>(Metadata metadata) {
        constexpr auto kernel    = fp8_kv_tiled_mma_kernel<G, S, Metadata>;
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
