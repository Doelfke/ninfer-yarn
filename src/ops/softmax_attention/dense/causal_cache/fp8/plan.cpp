#include "ops/softmax_attention/dense/causal_cache/fp8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/operands.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {

Fp8KvCausalPlan make_fp8_kv_causal_plan(int heads, int width, int batch,
                                        CausalAttentionExecutionEnvelope envelope) {
    if ((heads != 24 && heads != 16) || width < 1 || batch < 1 || batch > 8 ||
        (batch > 1 && width > 16) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys)
        throw std::invalid_argument("FP8 attention: invalid plan inputs");
    constexpr int grouped_limit = Fp8KvCausalPlan::kTokenTile;
    const auto family           = width <= grouped_limit ? Fp8KvFamily::Grouped
                                  : width <= 16          ? Fp8KvFamily::ParallelGrouped
                                                         : Fp8KvFamily::Tiled;
    const int tiles =
        family == Fp8KvFamily::ParallelGrouped ? (width + grouped_limit - 1) / grouped_limit : 1;
    const int independent_tiles = batch * (heads == 24 ? 4 : 2) * tiles;
    // Decode permits two resident CTAs per SM. Spec uses one; add a wave when
    // rounding to complete query tiles would leave over 10% of the 170 SMs idle.
    constexpr int sms   = 170;
    const int wave_ctas = (sms / independent_tiles) * independent_tiles;
    const int budget    = width == 1 || wave_ctas < sms * 9 / 10 ? 2 * sms : sms;
    Fp8KvPartition partition{1,
                             std::clamp(budget / independent_tiles, 1, Fp8KvPartition::kMaxSplits)};
    // Bound partial traffic by keeping enough KV work in each split.
    partition.key_shift = (width == 1 ? 7 : 8) - (heads == 16 ? 1 : 0);
    partition.capacity  = partition.active(envelope.max_visible_keys);
    return {family, heads, width, batch, envelope, partition};
}

std::size_t fp8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                   CausalAttentionExecutionEnvelope envelope) {
    std::size_t maximum = 0;
    for (int width = min_width; width <= std::min(max_width, 16); ++width) {
        const auto plan = make_fp8_kv_causal_plan(heads, width, batch, envelope);
        if (plan.family == Fp8KvFamily::Tiled) continue;
        const int splits = plan.partition.capacity;
        WorkspaceLayoutBuilder layout;
        (void)fp8_kv_allocate_partials(layout, heads, width, splits, batch);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return maximum;
}

} // namespace ninfer::ops::detail
