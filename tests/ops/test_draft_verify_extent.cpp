// Exact public-contract qualification for the DFlash2 draft-confidence verify-extent
// clamp. Expected extents are computed directly from the stored proposal_q values and
// the threshold; no launcher or kernel implementation is used as an oracle.
#include "ninfer/ops/draft_verify_extent.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr int kCandidates = 16;

// conf_i = max over the 16 stored candidates; extent' = min(extent, first_fail).
std::vector<std::int32_t> expected_extent(const float* proposal_q, int k, int b, float threshold,
                                          std::int32_t extent) {
    int first_fail = k;
    for (int i = 0; i < k; ++i) {
        float conf = 0.0F;
        for (int c = 0; c < kCandidates; ++c) {
            const float value = proposal_q[static_cast<std::size_t>(i) * kCandidates * b +
                                            static_cast<std::size_t>(c) * b + b];
            conf = std::max(conf, value);
        }
        if (conf < threshold) {
            first_fail = i;
            break;
        }
    }
    return {std::min(extent, first_fail)};
}

// One shape, several rows: hand-crafted per-position confidence so that the failure
// point varies across rows, plus boundary rows where every position passes.
int contract(cudaStream_t stream) {
    const int k     = 5;
    const int batch = 3;
    constexpr float threshold = 0.7F;

    std::vector<float> q(kCandidates * k * batch, 0.0F);
    // Row 0: fail at i=1 (conf_0 >= threshold, conf_1 < threshold).
    // Row 1: every position confident -> extent unchanged.
    // Row 2: fail at i=0.
    const float set[] = {
        /* row0 i0 */ 0.9F, /* row0 i1 */ 0.5F, /* row0 i2 */ 0.99F, /* row0 i3 */ 0.9F,
        /* row0 i4 */ 0.9F, /* row1 */          0.95F,                /* row2 i0 */ 0.1F,
    };
    const int row = std::array{0, 0, 0, 0, 0, 1, 2};
    const int position = std::array{0, 1, 2, 3, 4, 0, 0};
    for (int i = 0; i < 7; ++i) {
        for (int c = 0; c < kCandidates; ++c) {
            // A single confident candidate is enough; make others below threshold so the
            // max over candidates is exactly the intended confidence.
            const float value = (c == 0) ? set[i] : 0.0F;
            q[static_cast<std::size_t>(position[i]) * kCandidates * batch +
              static_cast<std::size_t>(c) * batch + row[i]] = value;
        }
    }

    std::vector<std::int32_t> extents{3, 7, 4};
    std::vector<std::int32_t> valid_columns{4, 8, 5};

    GuardedDeviceBuffer q_buffer(static_cast<std::size_t>(kCandidates) * k * batch *
                                        sizeof(float));
    q_buffer.copy_from_host(q.data(), q_buffer.bytes());
    GuardedDeviceBuffer extents_buffer(batch * sizeof(std::int32_t));
    extents_buffer.copy_from_host(extents.data(), extents_buffer.bytes());
    GuardedDeviceBuffer valid_buffer(batch * sizeof(std::int32_t));
    valid_buffer.copy_from_host(valid_columns.data(), valid_buffer.bytes());

    Tensor q_tensor(q_buffer.data(), DType::FP32, {kCandidates, k, batch});
    Tensor extents_tensor(extents_buffer.data(), DType::I32, {batch});
    Tensor valid_tensor(valid_buffer.data(), DType::I32, {batch});

    ops::draft_confidence_clamp_verify_extents(q_tensor, extents_tensor, valid_tensor, threshold,
                                               k, stream);
    cuda_synchronize(stream);

    std::vector<std::int32_t> got_extents =
        from_device<std::int32_t>(extents_buffer.data(), batch);
    std::vector<std::int32_t> got_valid = from_device<std::int32_t>(valid_buffer.data(), batch);
    std::vector<std::int32_t> exp_extents;
    std::vector<std::int32_t> exp_valid;
    for (int b = 0; b < batch; ++b) {
        std::vector<std::int32_t> single =
            expected_extent(q.data(), k, b, threshold, extents[b]);
        const std::int32_t extent_p = single[0];
        exp_extents.push_back(extent_p);
        exp_valid.push_back(std::min(valid_columns[b], extent_p + 1));
    }

    int failures = 0;
    failures += verify_exact("draft_confidence_clamp extents", got_extents, exp_extents);
    failures += verify_exact("draft_confidence_clamp valid_columns", got_valid, exp_valid);
    failures += verify_exact("draft_confidence_clamp source unchanged",
                             from_device<float>(q_buffer.data(), q.size()), q);
    failures += extents_buffer.verify_guards("draft_confidence_clamp extents") +
                valid_buffer.verify_guards("draft_confidence_clamp valid_columns") +
                q_buffer.verify_guards("draft_confidence_clamp proposal_q");
    return failures;
}

// Boundary rows: k==1, a no-op (all pass), and extent already below first_fail.
int boundaries(cudaStream_t stream) {
    int failures = 0;
    // k==1, single position below threshold -> extent clamps to 0.
    {
        const int k = 1, batch = 2;
        std::vector<float> q(kCandidates * k * batch, 0.2F);
        std::vector<std::int32_t> extents{2, 2};
        std::vector<std::int32_t> valid_columns{3, 3};
        GuardedDeviceBuffer q_buffer(q.size() * sizeof(float));
        q_buffer.copy_from_host(q.data(), q_buffer.bytes());
        GuardedDeviceBuffer extents_buffer(batch * sizeof(std::int32_t));
        extents_buffer.copy_from_host(extents.data(), extents_buffer.bytes());
        GuardedDeviceBuffer valid_buffer(batch * sizeof(std::int32_t));
        valid_buffer.copy_from_host(valid_columns.data(), valid_buffer.bytes());
        Tensor q_tensor(q_buffer.data(), DType::FP32, {kCandidates, k, batch});
        Tensor extents_tensor(extents_buffer.data(), DType::I32, {batch});
        Tensor valid_tensor(valid_buffer.data(), DType::I32, {batch});
        ops::draft_confidence_clamp_verify_extents(q_tensor, extents_tensor, valid_tensor,
                                                   0.5F, k, stream);
        cuda_synchronize(stream);
        failures += verify_exact("k==1 clamps to zero",
                                 from_device<std::int32_t>(extents_buffer.data(), batch),
                                 std::vector<std::int32_t>{0, 0});
        failures += verify_exact("k==1 valid_columns",
                                 from_device<std::int32_t>(valid_buffer.data(), batch),
                                 std::vector<std::int32_t>{1, 1});
        failures += q_buffer.verify_guards("k==1 proposal_q") +
                    extents_buffer.verify_guards("k==1 extents") +
                    valid_buffer.verify_guards("k==1 valid_columns");
    }
    // No-op: all positions confident and extent already the minimum.
    {
        const int k = 4, batch = 1;
        std::vector<float> q(kCandidates * k * batch, 0.99F);
        std::vector<std::int32_t> extents{2};
        std::vector<std::int32_t> valid_columns{3};
        GuardedDeviceBuffer q_buffer(q.size() * sizeof(float));
        q_buffer.copy_from_host(q.data(), q_buffer.bytes());
        GuardedDeviceBuffer extents_buffer(sizeof(std::int32_t));
        extents_buffer.copy_from_host(extents.data(), extents_buffer.bytes());
        GuardedDeviceBuffer valid_buffer(sizeof(std::int32_t));
        valid_buffer.copy_from_host(valid_columns.data(), valid_buffer.bytes());
        Tensor q_tensor(q_buffer.data(), DType::FP32, {kCandidates, k, batch});
        Tensor extents_tensor(extents_buffer.data(), DType::I32, {batch});
        Tensor valid_tensor(valid_buffer.data(), DType::I32, {batch});
        ops::draft_confidence_clamp_verify_extents(q_tensor, extents_tensor, valid_tensor,
                                                   0.7F, k, stream);
        cuda_synchronize(stream);
        failures += verify_exact("noop keeps extent",
                                 from_device<std::int32_t>(extents_buffer.data(), batch),
                                 std::vector<std::int32_t>{2});
        failures += verify_exact("noop keeps valid_columns",
                                 from_device<std::int32_t>(valid_buffer.data(), batch),
                                 std::vector<std::int32_t>{3});
        failures += q_buffer.verify_guards("noop proposal_q") +
                    extents_buffer.verify_guards("noop extents") +
                    valid_buffer.verify_guards("noop valid_columns");
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    int failures = 0;
    failures += contract(stream);
    failures += boundaries(stream);
    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    std::cout << (failures == 0 ? "OK" : "FAIL") << " draft_verify_extent public contract\n";
    return failures == 0 ? 0 : 1;
}
