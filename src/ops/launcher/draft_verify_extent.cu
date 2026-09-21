#include "ops/launcher/draft_verify_extent.h"

#include "core/device.h"
#include "ops/kernel/draft_verify_extent.cuh"

namespace ninfer::ops::detail {

void draft_confidence_clamp_verify_extents_launch(const Tensor& proposal_q, Tensor& extents,
                                                  Tensor& valid_columns, float threshold,
                                                  std::int32_t k, std::int32_t batch,
                                                  cudaStream_t stream) {
    draft_confidence_clamp_kernel<<<batch, 1, 0, stream>>>(
        static_cast<const float*>(proposal_q.data),
        static_cast<std::int32_t*>(extents.data),
        static_cast<std::int32_t*>(valid_columns.data), threshold, k, batch);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
