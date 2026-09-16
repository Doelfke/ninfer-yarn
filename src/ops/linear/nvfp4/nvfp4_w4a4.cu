#include "core/weight.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

template <class ActivationGeometry>
void launch_quantize_exact(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                           cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    constexpr int kThreads    = 256;
    const std::int32_t tasks  = tokens * ActivationGeometry::kGroupsPerRow;
    nvfp4_w4a4_quantize_kernel<ActivationGeometry>
        <<<(tasks + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), workspace.codes, workspace.scales, tokens,
            weight.input_scale_divisor);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_nvfp4_w4a4_quantize(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                                cudaStream_t stream) {
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 requires caller workspace");
    }
    switch (weight.k) {
    case Nvfp4Activation5120Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation5120Geometry>(x, weight, workspace, stream);
        return;
    case Nvfp4Activation6144Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation6144Geometry>(x, weight, workspace, stream);
        return;
    case Nvfp4Activation17408Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation17408Geometry>(x, weight, workspace, stream);
        return;
    default:
        throw std::invalid_argument("nvfp4 W4A4 quantize: unsupported K");
    }
}

} // namespace ninfer::ops::detail
