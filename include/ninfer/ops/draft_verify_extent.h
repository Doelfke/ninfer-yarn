#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: draft_confidence_clamp_verify_extents
 *
 * Caps the per-row speculative verify extent to the longest leading prefix of draft
 * positions whose drafter is confident. It is the device-side half of "early-stopping"
 * speculative verification (the accepted-prefix prefix it produces is exactly what the
 * round already verifies when a request's remaining budget is small). The target
 * verification and acceptance run unchanged over the (possibly shorter) prefix; output is
 * lossless because rejection sampling over a shorter verified prefix preserves the target
 * distribution.
 *
 * Math / indexing:
 *   For row b and position i in [0,k):
 *     conf_i  = max_{c in [0,16)} proposal_q[c, i, b];
 *     first_fail = min { i : conf_i < threshold } if any, else k;
 *     extent' = min(extents[b], first_fail);
 *   extents[b]'        = extent';
 *   valid_columns[b]'  = min(valid_columns[b], extent' + 1).
 *
 * Logical shapes:
 *   proposal_q is contiguous FP32 [16,k,batch] (candidate rank fastest). extents and
 *   valid_columns are contiguous device I32 [batch]. k is in [1,15], batch in [1,8].
 *
 * Effects:
 *   In place (no aliasing required): each extents[b] writes extent' and each valid_columns[b]
 *   writes min(valid_columns[b], extent'+1). proposal_q is read-only and unchanged.
 *
 * Numerics:
 *   conf_i is the FP32 elementwise maximum over the 16 stored FP32 values (exact). threshold is
 *   a host float in [0,1]; threshold <= 0 makes every position pass (first_fail = k), i.e. a
 *   no-op. No rounding.
 *
 * Workspace:
 *   None.
 */
void draft_confidence_clamp_verify_extents(const Tensor& proposal_q, Tensor& extents,
                                           Tensor& valid_columns, float threshold,
                                           std::int32_t k, cudaStream_t stream);

} // namespace ninfer::ops
