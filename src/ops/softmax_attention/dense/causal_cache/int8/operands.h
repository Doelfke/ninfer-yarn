#pragma once

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace ninfer::ops::detail {

struct Int8KvOperands {
    const __nv_bfloat16* q;
    const std::int32_t* positions;
    __nv_bfloat16* out;
    float scale;
    int width, batch, visible_capacity, query_heads;
};

template <bool Writable>
struct Int8KvCacheView {
    using Code  = std::conditional_t<Writable, std::int8_t, const std::int8_t>;
    using Scale = std::conditional_t<Writable, __half, const __half>;
    Code* keys;
    Code* values;
    Scale* key_scales;
    Scale* value_scales;
    const std::int32_t* tables;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    int table_stride, kv_heads;
};

using Int8KvReadView = Int8KvCacheView<false>;

struct Int8KvAppendInput {
    static constexpr bool writes_cache = true;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
};

struct Int8KvCachedInput {
    static constexpr bool writes_cache = false;
};

// Maximum is in natural scaled-score units; numerator and sum are unnormalized FP32.
struct Int8KvPartialView {
    float* acc;
    float* maximum;
    float* sum;
};

struct Int8KvPartialStorage {
    Tensor acc, maximum, sum;

    Int8KvPartialView view() const {
        return {static_cast<float*>(acc.data), static_cast<float*>(maximum.data),
                static_cast<float*>(sum.data)};
    }
};

inline Int8KvOperands int8_kv_operands(const Tensor& q, const Tensor& positions, Tensor& out,
                                       float scale, int visible) {
    return {static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::int32_t*>(positions.data),
            static_cast<__nv_bfloat16*>(out.data),
            scale,
            q.ne[2],
            q.ne[3],
            visible,
            q.ne[1]};
}

template <bool Writable>
Int8KvCacheView<Writable> int8_kv_cache_view(const PagedKVBatchLayerView& cache,
                                             const Tensor* valid = nullptr,
                                             const Tensor* rows  = nullptr) {
    return {static_cast<typename Int8KvCacheView<Writable>::Code*>(cache.k_pages.data),
            static_cast<typename Int8KvCacheView<Writable>::Code*>(cache.v_pages.data),
            static_cast<typename Int8KvCacheView<Writable>::Scale*>(cache.k_scale_pages.data),
            static_cast<typename Int8KvCacheView<Writable>::Scale*>(cache.v_scale_pages.data),
            static_cast<const std::int32_t*>(cache.block_tables.data),
            valid ? static_cast<const std::int32_t*>(valid->data) : nullptr,
            rows ? static_cast<const std::int32_t*>(rows->data) : nullptr,
            cache.block_tables.ne[0],
            cache.num_kv_heads};
}

template <class Allocator>
Int8KvPartialStorage int8_kv_allocate_partials(Allocator& allocator, int heads, int width,
                                               int splits, int batch) {
    return {allocator.alloc(DType::FP32, {256, heads, width, splits * batch}),
            allocator.alloc(DType::FP32, {heads, width, splits * batch}),
            allocator.alloc(DType::FP32, {heads, width, splits * batch})};
}

} // namespace ninfer::ops::detail
