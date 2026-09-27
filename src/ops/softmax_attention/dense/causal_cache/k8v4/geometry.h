#pragma once

#include "ops/softmax_attention/common/head_mapping.cuh"

namespace ninfer::ops::detail {

template <int QueryHeads, int KVHeads>
struct K8V4KvGeometry : AttentionHeadMapping<QueryHeads, KVHeads> {
    static constexpr int kHeadDim = 256;
};

using K8V4KvD256H24Kv4 = K8V4KvGeometry<24, 4>;
using K8V4KvD256H16Kv2 = K8V4KvGeometry<16, 2>;

} // namespace ninfer::ops::detail
