#include "ninfer/ops/draft_verify_extent.h"

#include "ops/launcher/draft_verify_extent.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_row_vector(const Tensor& tensor, DType dtype, std::int32_t batch, const char* name) {
    if (tensor.dtype != dtype || tensor.ne[1] != 1 || tensor.ne[2] != 1 || tensor.ne[3] != 1 ||
        tensor.ne[0] != batch || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string(name) + " must be a non-null contiguous I32 row "
                                    "vector of batch size within [1,8]");
    }
}

} // namespace

void draft_confidence_clamp_verify_extents(const Tensor& proposal_q, Tensor& extents,
                                           Tensor& valid_columns, float threshold,
                                           std::int32_t k, cudaStream_t stream) {
    require_row_vector(extents, DType::I32, extents.ne[0], "draft_confidence_clamp extents");
    require_row_vector(valid_columns, DType::I32, extents.ne[0],
                       "draft_confidence_clamp valid_columns");
    if (extents.ne[0] != valid_columns.ne[0] || extents.ne[0] < 1 || extents.ne[0] > 8) {
        throw std::invalid_argument("draft_confidence_clamp batch size must be in [1,8]");
    }
    if (!proposal_q.is_contiguous() || proposal_q.data == nullptr ||
        proposal_q.dtype != DType::FP32 || proposal_q.ne[0] != 16 || proposal_q.ne[1] != k ||
        proposal_q.ne[2] != extents.ne[0] || proposal_q.ne[3] != 1) {
        throw std::invalid_argument(
            "draft_confidence_clamp proposal_q must be a non-null contiguous FP32 [16,k,batch]");
    }
    if (k < 1 || k > 15) {
        throw std::invalid_argument("draft_confidence_clamp draft count must be in [1,15]");
    }
    detail::draft_confidence_clamp_verify_extents_launch(proposal_q, extents, valid_columns,
                                                          threshold, k, extents.ne[0], stream);
}

} // namespace ninfer::ops
