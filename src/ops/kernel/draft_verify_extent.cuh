#pragma once

#include <cstdint>
#include <cfloat>

namespace ninfer::ops {

__device__ __forceinline__ float clamp_argmax_max(float a, float b) {
    return a > b ? a : b;
}

__global__ void draft_confidence_clamp_kernel(const float* __restrict__ proposal_q,
                                              std::int32_t* __restrict__ extents,
                                              std::int32_t* __restrict__ valid_columns,
                                              float threshold, std::int32_t k,
                                              std::int32_t batch) {
    const std::int32_t row     = blockIdx.x;
    const std::int32_t extent  = extents[row];
    const std::int32_t clamped = extent < 0 ? 0 : extent;
    std::int32_t first_fail    = clamped;
    for (std::int32_t i = 0; i < first_fail; ++i) {
        const float* candidates = proposal_q + (static_cast<std::int64_t>(i) * batch + row) * 16;
        float conf              = -FLT_MAX;
        for (std::int32_t c = 0; c < 16; ++c) { conf = clamp_argmax_max(conf, candidates[c]); }
        if (conf < threshold) { first_fail = i; break; }
    }
    extents[row]           = first_fail;
    valid_columns[row]     = min(valid_columns[row], first_fail + 1);
}

} // namespace ninfer::ops
