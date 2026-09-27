#pragma once

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <stdexcept>

namespace ninfer::ops::detail {

struct Nvfp4KvOperands {
    const __nv_bfloat16* q;
    const std::int32_t* positions;
    __nv_bfloat16* out;
    float scale;
    int width, batch, visible_capacity, query_heads;
};

template <bool Writable>
struct Nvfp4KvCacheView {
    using Code  = std::conditional_t<Writable, std::uint8_t, const std::uint8_t>;
    using Scale = std::conditional_t<Writable, std::uint8_t, const std::uint8_t>;
    Code* keys;
    Code* values;
    Scale* key_scales;
    Scale* value_scales;
    const std::int32_t* tables;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    int table_stride, kv_heads;
};

using Nvfp4KvReadView = Nvfp4KvCacheView<false>;

struct Nvfp4KvAppendInput {
    static constexpr bool writes_cache = true;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
};

struct Nvfp4KvCachedInput {
    static constexpr bool writes_cache = false;
};

// Maximum is in natural scaled-score units; numerator and sum are unnormalized FP32.
struct Nvfp4KvPartialView {
    float* acc;
    float* maximum;
    float* sum;
};

struct Nvfp4KvPartialStorage {
    Tensor acc, maximum, sum;

    Nvfp4KvPartialView view() const {
        return {static_cast<float*>(acc.data), static_cast<float*>(maximum.data),
                static_cast<float*>(sum.data)};
    }
};

inline Nvfp4KvOperands nvfp4_kv_operands(const Tensor& q, const Tensor& positions, Tensor& out,
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
Nvfp4KvCacheView<Writable> nvfp4_kv_cache_view(const PagedKVBatchLayerView& cache,
                                               const Tensor* valid = nullptr,
                                               const Tensor* rows  = nullptr) {
    return {static_cast<typename Nvfp4KvCacheView<Writable>::Code*>(cache.k_pages.data),
            static_cast<typename Nvfp4KvCacheView<Writable>::Code*>(cache.v_pages.data),
            static_cast<typename Nvfp4KvCacheView<Writable>::Scale*>(cache.k_scale_pages.data),
            static_cast<typename Nvfp4KvCacheView<Writable>::Scale*>(cache.v_scale_pages.data),
            static_cast<const std::int32_t*>(cache.block_tables.data),
            valid ? static_cast<const std::int32_t*>(valid->data) : nullptr,
            rows ? static_cast<const std::int32_t*>(rows->data) : nullptr,
            cache.block_tables.ne[0],
            cache.num_kv_heads};
}

template <class Allocator>
Nvfp4KvPartialStorage nvfp4_kv_allocate_partials(Allocator& allocator, int heads, int width,
                                                 int splits, int batch) {
    return {allocator.alloc(DType::FP32, {256, heads, width, splits * batch}),
            allocator.alloc(DType::FP32, {heads, width, splits * batch}),
            allocator.alloc(DType::FP32, {heads, width, splits * batch})};
}

template <class G, bool Writable>
void validate_nvfp4_kv_operands(const Nvfp4KvOperands& p, Nvfp4KvCacheView<Writable> cache) {
    if (p.query_heads != G::QHeads || cache.kv_heads != G::KVHeads || !p.q || !p.positions ||
        !p.out || !cache.keys || !cache.values || !cache.key_scales || !cache.value_scales ||
        !cache.tables || p.width < 1 || p.batch < 1 || p.visible_capacity < 1 ||
        static_cast<std::int64_t>(p.visible_capacity) >
            static_cast<std::int64_t>(cache.table_stride) * kPagedKVPageSize)
        throw std::invalid_argument("NVFP4 attention template: invalid operands");
}

} // namespace ninfer::ops::detail
