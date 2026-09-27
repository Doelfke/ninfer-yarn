// Remaining quantized-cache legacy route dispatcher.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/paged_kv_storage.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/softmax_attention.h"

#include <cstdint>
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Supplies an upper bound for the device-side active-split policy over one explicit execution
// envelope. Eager calls normally pass an exact window; graph calls pass their target-private
// replay interval for the remaining legacy formats.
template <typename Geometry>
std::int32_t causal_small_t_split_upper_bound(std::int32_t window) {
    if (window <= 0) { return Geometry::SmallTMaximumSplits; }

    constexpr std::int32_t kMinSplits = 4 * Geometry::SmallTSplitScale;
    std::int32_t splits               = kMinSplits;

    const auto include_tier = [&](std::int32_t window_limit, std::int32_t target_keys_per_split) {
        const std::int32_t tier_window = (window < window_limit) ? window : window_limit;
        if (tier_window > 0) {
            const std::int32_t tier_splits = div_up(tier_window, target_keys_per_split);
            splits                         = (splits > tier_splits) ? splits : tier_splits;
        }
    };

    include_tier(4096, 64 / Geometry::SmallTSplitScale);
    if (window > 4096) { include_tier(8198, 128 / Geometry::SmallTSplitScale); }
    if (window > 8198) { include_tier(16390, 256 / Geometry::SmallTSplitScale); }
    if (window > 16390) { include_tier(window, 480 / Geometry::SmallTSplitScale); }

    splits = (splits < Geometry::SmallTMaximumSplits) ? splits : Geometry::SmallTMaximumSplits;
    if constexpr (Geometry::SmallTSplitScale == 1) {
        // Page-safety floor (must mirror causal_small_t_default_splits in small_t.cuh). Each
        // split stages up to 64 physical-page IDs into __shared__ physical_pages_s[64] and
        // indexes it by page offset, so keys/split must stay <= 3968 (62 pages, the
        // conservative bound that leaves 2 pages for tile rounding/alignment). The efficiency
        // cap above under-provisions splits at large windows; floor to page_limit so those
        // windows split enough keys, up to the 256-split ceiling of the split reducer.
        constexpr std::int32_t kSplitsCeiling = 256;
        const std::int32_t page_limit         = div_up(window, 3968);
        splits                                = (splits > page_limit) ? splits : page_limit;
        return (splits < kSplitsCeiling) ? splits : kSplitsCeiling;
    }
    return splits;
}

template <typename Geometry>
std::int32_t causal_small_t_launch_capacity(CausalAttentionExecutionEnvelope envelope) {
    std::int32_t capacity = 0;
    const auto include    = [&](std::uint32_t window) {
        if (window < envelope.min_visible_keys || window > envelope.max_visible_keys) { return; }
        const auto splits =
            causal_small_t_split_upper_bound<Geometry>(static_cast<std::int32_t>(window));
        capacity = capacity > splits ? capacity : splits;
    };
    include(envelope.min_visible_keys);
    include(envelope.max_visible_keys);
    // The policy is monotonic inside these finite segments and may drop when crossing a boundary.
    // Evaluating every segment end plus both interval ends gives the exact interval maximum.
    constexpr std::uint32_t ends[] = {4096, 8198, 16390};
    for (const std::uint32_t end : ends) { include(end); }
    return capacity;
}

} // namespace

std::int32_t causal_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                             KvCacheStorage cache_storage,
                                             CausalAttentionExecutionEnvelope envelope,
                                             std::int32_t batch_size) {
    if (tokens < 1 || tokens > (q_heads == 24 ? 8 : 6) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys) {
        throw std::invalid_argument("causal_softmax_attention split capacity: invalid profile");
    }
    (void)paged_kv_storage_layout(cache_storage, kCausalHeadDim);

    if (q_heads == CausalD256H24Kv4::QHeads) {
        const int capacity = causal_small_t_launch_capacity<CausalD256H24Kv4>(envelope);
        if (batch_size > 1) {
            // Keep complete grids within one or two 170-SM waves. Rounding from 160 CTAs
            // leaves room for the indivisible 4*B group, including B=3/5/6/7.
            const bool narrow = tokens <= 5;
            int target_ctas   = 160;
            if (cache_storage == KvCacheStorage::Nvfp4Group16) target_ctas = narrow ? 320 : 160;
            const int grid_limit = div_up(target_ctas, 4 * batch_size);
            // A split stages at most 64 physical-page IDs. Leave two 64-key pages for
            // key-tile rounding and page alignment at the 262144-key resource limit.
            const int page_limit = div_up(static_cast<int>(envelope.max_visible_keys), 3968);
            return std::min(capacity, std::max({4, grid_limit, page_limit}));
        }
        return capacity;
    }
    if (q_heads == CausalD256H16Kv2::QHeads) {
        return causal_small_t_launch_capacity<CausalD256H16Kv2>(envelope);
    }
    throw std::invalid_argument(
        "causal_softmax_attention split capacity: unsupported head geometry");
}

void causal_attention_small_t_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_small_t_k8v4_launch(q, k, v, pos, valid_columns, table_rows, scale, cache,
                                             envelope, column_begin, width, partial_acc, partial_m,
                                             partial_l, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_small_t_nvfp4_launch(q, k, v, pos, valid_columns, table_rows, scale, cache,
                                              envelope, column_begin, width, partial_acc, partial_m,
                                              partial_l, out, stream);
        return;
    }
    throw std::invalid_argument("unsupported legacy attention storage");
}

void causal_attention_cached_small_t_launch(const Tensor& q, const Tensor& pos, float scale,
                                            const PagedKVLayerView& cache,
                                            CausalAttentionExecutionEnvelope envelope,
                                            Tensor& partial_acc, Tensor& partial_m,
                                            Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_cached_small_t_k8v4_launch(q, pos, scale, cache, envelope, partial_acc,
                                                    partial_m, partial_l, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_cached_small_t_nvfp4_launch(q, pos, scale, cache, envelope, partial_acc,
                                                     partial_m, partial_l, out, stream);
        return;
    }
    throw std::invalid_argument("unsupported legacy attention storage");
}

} // namespace ninfer::ops::detail
