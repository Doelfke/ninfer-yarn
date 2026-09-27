#include "ops/softmax_attention/dense/causal_cache/fp8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/operands.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/split_policy.h"
#include <stdexcept>

namespace ninfer::ops::detail {

int Fp8KvCausalPlan::split_capacity() const {
    if (family == Fp8KvFamily::Grouped && width == 1 && envelope.max_visible_keys <= 64) return 1;
    if (family == Fp8KvFamily::ParallelGrouped) {
        const int tiles  = (width + token_tile - 1) / token_tile;
        const int ctas   = batch * (query_heads == 24 ? 4 : 2) * tiles;
        const int budget = (320 + ctas - 1) / ctas;
        const int maximum =
            query_heads == 24 ? Fp8KvSplitPolicy<24>::upper_bound(envelope.max_visible_keys, width)
                              : Fp8KvSplitPolicy<16>::upper_bound(envelope.max_visible_keys, width);
        return std::min(budget, maximum);
    }
    return query_heads == 24 ? Fp8KvSplitPolicy<24>::capacity(envelope, width, batch)
                             : Fp8KvSplitPolicy<16>::capacity(envelope, width, batch);
}

Fp8KvCausalPlan make_fp8_kv_causal_plan(int heads, int width, int batch,
                                        CausalAttentionExecutionEnvelope envelope) {
    if ((heads != 24 && heads != 16) || width < 1 || batch < 1 || batch > 8 ||
        (batch > 1 && width > 16) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys)
        throw std::invalid_argument("FP8 attention: invalid plan inputs");
    Fp8KvFamily family = Fp8KvFamily::Tiled;
    if (heads == 24 && width <= 16) {
        const int limit = width <= 4 ? 0 : width <= 8 ? 128 : 320;
        if (batch > 1 || envelope.max_visible_keys > static_cast<unsigned>(limit))
            family = width <= 8 ? Fp8KvFamily::Grouped : Fp8KvFamily::ParallelGrouped;
    } else if (width <= 6) {
        family = Fp8KvFamily::Grouped;
    } else if (batch > 1 || (heads == 16 && width <= 16 &&
                             envelope.max_visible_keys > (width <= 12 ? 512u : 1024u))) {
        family = Fp8KvFamily::ParallelGrouped;
    }
    const int tile = heads == 24 ? 8 : width <= 12 ? 4 : 6;
    return {family, heads, width, batch, tile, envelope};
}

std::size_t fp8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                   CausalAttentionExecutionEnvelope envelope) {
    std::size_t maximum = 0;
    for (int width = min_width; width <= std::min(max_width, 16); ++width) {
        const auto plan = make_fp8_kv_causal_plan(heads, width, batch, envelope);
        if (plan.family == Fp8KvFamily::Tiled) continue;
        const int splits = plan.split_capacity();
        if (splits == 1) continue;
        WorkspaceLayoutBuilder layout;
        (void)fp8_kv_allocate_partials(layout, heads, width, splits, batch);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return maximum;
}

} // namespace ninfer::ops::detail
