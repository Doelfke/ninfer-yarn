#pragma once

#include "ninfer/ops/softmax_attention.h"
#include <algorithm>

namespace ninfer::ops::detail {

template <int Heads>
struct Fp8KvSplitPolicy {
    static_assert(Heads == 24 || Heads == 16);
    static constexpr int kScale     = Heads == 24 ? 1 : 2;
    static constexpr int kMaxSplits = 85 * kScale;

    __host__ __device__ static int ceil_div(int x, int y) { return (x + y - 1) / y; }

    __host__ __device__ static int active(int visible, int capacity, int tokens) {
        int keys = 480 / kScale;
        if (visible <= 4096)
            keys = 64 / kScale;
        else if (visible <= 8198)
            keys = 128 / kScale;
        else if (visible <= 16390)
            keys = 256 / kScale;
        int count = ceil_div(visible, keys);
        count     = count < 4 * kScale ? 4 * kScale : count;
        count     = count > kMaxSplits ? kMaxSplits : count;
        if constexpr (Heads == 24)
            if (tokens == 1 && visible > 8198) count = kMaxSplits;
        return count < capacity ? count : capacity;
    }

    static int upper_bound(int visible, int tokens) {
        if constexpr (Heads == 24)
            if (tokens == 1 && visible > 8198) return kMaxSplits;
        int count          = 4 * kScale;
        const auto include = [&](int limit, int keys) {
            count = std::max(count, ceil_div(std::min(visible, limit), keys));
        };
        include(4096, 64 / kScale);
        if (visible > 4096) include(8198, 128 / kScale);
        if (visible > 8198) include(16390, 256 / kScale);
        if (visible > 16390) include(visible, 480 / kScale);
        return std::min(count, kMaxSplits);
    }

    static int capacity(CausalAttentionExecutionEnvelope envelope, int tokens, int batch) {
        int count          = 0;
        const auto include = [&](unsigned visible) {
            if (visible >= envelope.min_visible_keys && visible <= envelope.max_visible_keys)
                count = std::max(count, upper_bound(visible, tokens));
        };
        include(envelope.min_visible_keys);
        include(envelope.max_visible_keys);
        for (unsigned end : {4096u, 8198u, 16390u}) include(end);
        if constexpr (Heads == 24) {
            if (batch > 1)
                count = std::min(count, std::max({4, ceil_div(160, 4 * batch),
                                                  ceil_div(envelope.max_visible_keys, 3968)}));
        }
        return count;
    }
};

} // namespace ninfer::ops::detail
