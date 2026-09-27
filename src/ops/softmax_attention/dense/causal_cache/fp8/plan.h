#pragma once

#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/split_policy.h"

namespace ninfer::ops::detail {

enum class Fp8KvFamily { Grouped, ParallelGrouped, Tiled };

struct Fp8KvCausalPlan {
    static constexpr int kTokenTile = 8;
    Fp8KvFamily family;
    int query_heads, width, batch;
    CausalAttentionExecutionEnvelope envelope;
    Fp8KvPartition partition;
};

Fp8KvCausalPlan make_fp8_kv_causal_plan(int heads, int width, int batch,
                                        CausalAttentionExecutionEnvelope envelope);
std::size_t fp8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                   CausalAttentionExecutionEnvelope envelope);

} // namespace ninfer::ops::detail
