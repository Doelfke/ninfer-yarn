#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void draft_confidence_clamp_verify_extents_launch(const Tensor& proposal_q, Tensor& extents,
                                                  Tensor& valid_columns, float threshold,
                                                  std::int32_t k, std::int32_t batch,
                                                  cudaStream_t stream);

} // namespace ninfer::ops::detail
