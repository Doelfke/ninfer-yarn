#pragma once

#include <cstdint>

namespace ninfer::ops {

__global__ void fill_i32_positions_kernel(std::int32_t* positions, std::int32_t count,
                                          std::int32_t start) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) { positions[i] = start + i; }
}

__global__ void offset_i32_positions_kernel(const std::int32_t* source, const std::int32_t* delta,
                                            std::int32_t* destination, std::int32_t count) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) { destination[i] = source[i] + delta[0]; }
}

__global__ void scale_positions_yarn_kernel(std::int32_t* positions, std::int32_t count,
                                            std::int32_t original_context, float factor) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) {
        const std::int32_t p = positions[i];
        if (p <= original_context) {
            positions[i] = p;
        } else {
            const float scaled =
                static_cast<float>(original_context) +
                (static_cast<float>(p - original_context)) / factor + 0.5F;
            positions[i] = static_cast<std::int32_t>(scaled);
        }
    }
}

} // namespace ninfer::ops
