#include "ops/softmax_attention/dense/causal_cache/nvfp4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/operands.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {

Nvfp4KvCausalPlan make_nvfp4_kv_causal_plan(int heads, int width, int batch,
                                            CausalAttentionExecutionEnvelope envelope) {
    if ((heads != 24 && heads != 16) || width < 1 || batch < 1 || batch > 8 ||
        (batch > 1 && width > 16) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys)
        throw std::invalid_argument("NVFP4 attention: invalid plan inputs");
    constexpr int grouped_limit = Nvfp4KvCausalPlan::kTokenTile;
    const auto family           = width <= grouped_limit ? Nvfp4KvFamily::Grouped
                                  : width <= 16          ? Nvfp4KvFamily::ParallelGrouped
                                                         : Nvfp4KvFamily::Tiled;
    const int tiles =
        family == Nvfp4KvFamily::ParallelGrouped ? (width + grouped_limit - 1) / grouped_limit : 1;
    const int independent_tiles = batch * (heads == 24 ? 4 : 2) * tiles;
    const int query_tile =
        family == Nvfp4KvFamily::ParallelGrouped ? (width + 1) / 2 : std::min(width, grouped_limit);
    const int row_tiles = (query_tile * (heads == 24 ? 6 : 8) + 15) / 16;
    constexpr int sms   = 170;
    const int wave_ctas = (sms / independent_tiles) * independent_tiles;
    const int budget    = row_tiles <= 2 || wave_ctas < sms * 9 / 10 ? 2 * sms : sms;
    Nvfp4KvPartition partition{
        1, std::clamp(budget / independent_tiles, 1, Nvfp4KvPartition::kMaxSplits)};
    // Bound partial traffic by keeping enough KV work in each split.
    partition.key_shift = (row_tiles <= 2 ? 7 : 8) - (heads == 16 ? 1 : 0);
    partition.capacity  = partition.active(envelope.max_visible_keys);
    return {family, heads, width, batch, query_tile, envelope, partition};
}

std::size_t nvfp4_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                     CausalAttentionExecutionEnvelope envelope) {
    std::size_t maximum = 0;
    for (int width = min_width; width <= std::min(max_width, 16); ++width) {
        const auto plan = make_nvfp4_kv_causal_plan(heads, width, batch, envelope);
        if (plan.family == Nvfp4KvFamily::Tiled) continue;
        const int splits = plan.partition.capacity;
        WorkspaceLayoutBuilder layout;
        (void)nvfp4_kv_allocate_partials(layout, heads, width, splits, batch);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return maximum;
}

} // namespace ninfer::ops::detail
