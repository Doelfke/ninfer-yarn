#pragma once

#include "ops/softmax_attention/dense/causal_cache/fp8/tile_io.cuh"

namespace ninfer::ops::detail {

__device__ __forceinline__ void fp8_kv_store_partial_pair(float* target, float a, float b) {
    *reinterpret_cast<float2*>(target) = make_float2(a, b);
}

template <class G>
__device__ __forceinline__ void fp8_kv_store_output_pair(__nv_bfloat16* out, int head, int d,
                                                         int token, float a, float b) {
    *reinterpret_cast<unsigned*>(out + fp8_kv_q_index<G>(head, d, token)) = pack_bf16x2(a, b);
}

__device__ __forceinline__ void fp8_kv_store_output(__nv_bfloat16* out, float value) {
    *out = __float2bfloat16(value);
}

} // namespace ninfer::ops::detail
