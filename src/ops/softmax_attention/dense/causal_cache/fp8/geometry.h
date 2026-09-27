#pragma once

#include "ops/softmax_attention/common/head_mapping.cuh"

namespace ninfer::ops::detail {

template <int QueryHeads, int KVHeads>
struct Fp8KvGeometry : AttentionHeadMapping<QueryHeads, KVHeads> {
    static constexpr int kHeadDim = 256;
};

using Fp8KvD256H24Kv4 = Fp8KvGeometry<24, 4>;
using Fp8KvD256H16Kv2 = Fp8KvGeometry<16, 2>;

} // namespace ninfer::ops::detail
