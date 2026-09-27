#pragma once

#include "ops/softmax_attention/common/head_mapping.cuh"

namespace ninfer::ops::detail {

template <int QueryHeads, int KVHeads>
struct Nvfp4KvGeometry : AttentionHeadMapping<QueryHeads, KVHeads> {
    static constexpr int kHeadDim = 256;
};

using Nvfp4KvD256H24Kv4 = Nvfp4KvGeometry<24, 4>;
using Nvfp4KvD256H16Kv2 = Nvfp4KvGeometry<16, 2>;

} // namespace ninfer::ops::detail
