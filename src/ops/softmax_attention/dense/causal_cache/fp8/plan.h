#pragma once

#include "ninfer/ops/softmax_attention.h"

namespace ninfer::ops::detail {

enum class Fp8KvFamily { Grouped, ParallelGrouped, Tiled };

struct Fp8KvCausalPlan {
    Fp8KvFamily family;
    int query_heads, width, batch, token_tile;
    CausalAttentionExecutionEnvelope envelope;

    int split_capacity() const;
};

Fp8KvCausalPlan make_fp8_kv_causal_plan(int heads, int width, int batch,
                                        CausalAttentionExecutionEnvelope envelope);
std::size_t fp8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                   CausalAttentionExecutionEnvelope envelope);

} // namespace ninfer::ops::detail
