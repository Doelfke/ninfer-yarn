#pragma once

#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/split_policy.h"

namespace ninfer::ops::detail {

enum class K8V4KvFamily { Grouped, ParallelGrouped, Tiled };

struct K8V4KvCausalPlan {
    static constexpr int kTokenTile = 8;
    K8V4KvFamily family;
    int query_heads, width, batch, query_tile;
    CausalAttentionExecutionEnvelope envelope;
    K8V4KvPartition partition;
};

K8V4KvCausalPlan make_k8v4_kv_causal_plan(int heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope);
std::size_t k8v4_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                    CausalAttentionExecutionEnvelope envelope);

} // namespace ninfer::ops::detail
