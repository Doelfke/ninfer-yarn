#pragma once

#include "ops/softmax_attention/dense/causal_cache/nvfp4/tile_io.cuh"

namespace ninfer::ops::detail {

__device__ __forceinline__ void nvfp4_kv_store_partial_pair(float* target, float a, float b) {
    *reinterpret_cast<float2*>(target) = make_float2(a, b);
}

template <class G>
__device__ __forceinline__ void nvfp4_kv_store_output_pair(__nv_bfloat16* out, int head, int d,
                                                           int token, float a, float b) {
    *reinterpret_cast<unsigned*>(out + nvfp4_kv_q_index<G>(head, d, token)) = pack_bf16x2(a, b);
}

__device__ __forceinline__ void nvfp4_kv_store_output(__nv_bfloat16* out, float value) {
    *out = __float2bfloat16(value);
}

// A warp converts one normalized row from the represented rotated V coordinates.
template <class G>
__device__ __forceinline__ void nvfp4_kv_store_rotated_row(const float* row, __nv_bfloat16* out,
                                                           int head, int token) {
    const int lane = threadIdx.x & 31;
    float values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) values[r] = row[lane + 32 * r];
    normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (int r = 0; r < 8; ++r)
        nvfp4_kv_store_output(out + nvfp4_kv_q_index<G>(head, lane + 32 * r, token), values[r]);
}

} // namespace ninfer::ops::detail
