#pragma once

#include "ops/softmax_attention/common/head_mapping.cuh"

namespace ninfer::ops::detail {

template <int QueryHeads, int KVHeads>
struct Int8KvGeometry : AttentionHeadMapping<QueryHeads, KVHeads> {
    static constexpr int kHeadDim = 256;
};

using Int8KvD256H24Kv4 = Int8KvGeometry<24, 4>;
using Int8KvD256H16Kv2 = Int8KvGeometry<16, 2>;

} // namespace ninfer::ops::detail
