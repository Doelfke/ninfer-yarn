#pragma once

#include "models/qwen3_5/program/internal.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/qwen3_5/execution/vision_cpu/vision_cpu_weights.h"
#include "models/qwen3_5/program/vision_control.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/vision_prefill.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

using detail::VisionWorkspacePlan;
using detail::VisionPrefillPlan;
using detail::VisionUseSpan;

struct VisionItemView {
    std::span<const std::uint16_t> patches;
    const qwen3_5::VisionItemControl* control = nullptr;
};

// `--vision-cpu` sizing: the CPU path writes only the final embedding handoff (no device encoder
// scratch), so the plan carries a handoff region with zero encode scratch. `output_hidden` is the
// text hidden size (= the host-dequantized `merger_fc2` rows).
[[nodiscard]] VisionWorkspacePlan plan_vision_workspace_offload(
    const VisionConfig& config, std::int32_t output_hidden, std::uint32_t max_merged_tokens,
    std::size_t general_capacity_bytes);

class VisionContext {
public:
    // `use_device_parameters` is false in `--vision-cpu` offload mode, where no device Vision
    // parameters exist (`parameters.vision` is empty) and the encoder runs on the host.
    VisionContext(DeviceContext& device, const execution::Parameters& parameters,
                  bool use_device_parameters = true);

    [[nodiscard]] static std::size_t workspace_bytes(const VisionConfig& config,
                                                     const VisionParameters& parameters,
                                                     std::size_t patches, std::size_t merged_tokens,
                                                     const VisionWorkspacePlan& plan);
    [[nodiscard]] static VisionWorkspacePlan plan_workspace(const VisionConfig& config,
                                                            const VisionParameters& parameters,
                                                            std::uint32_t max_merged_tokens,
                                                            std::size_t general_capacity_bytes);

    [[nodiscard]] const VisionConfig& config() const noexcept { return config_; }

    [[nodiscard]] static Tensor bind_output(DeviceSpan backing, const VisionWorkspacePlan& plan,
                                            std::size_t merged_tokens);
    void encode(const VisionItemView& item, Tensor& output, DeviceSpan backing,
                const VisionWorkspacePlan& plan) const;

private:
    DeviceContext& ctx_;
    const VisionConfig& config_;
    // Null in `--vision-cpu` offload mode (no device Vision parameters); `encode` is never called
    // on that path. Non-null and fully populated in device mode.
    const VisionParameters* parameters_ = nullptr;
};

struct VisionChunk {
    std::int32_t length                       = 0;
    const qwen3_5::VisionItemControl* control = nullptr;
    Tensor embeddings;
};

class VisionPrefillSession {
public:
    // Non-null when `--vision-cpu` is active: the ViT runs on the CPU with these host-resident
    // FP32 weights (stable for the model's lifetime) and no device Vision parameters/workspace
    // are required. Null in device mode.
    VisionPrefillSession(DeviceContext& device, const execution::Parameters& parameters,
                         DeviceSpan workspace, const VisionWorkspacePlan& workspace_plan,
                         qwen3_5::PreparedPromptData& prompt, const VisionPrefillPlan& plan,
                         std::size_t& handoff_peak_bytes,
                         const vision_cpu::CpuVisionWeights* cpu_weights = nullptr);

    [[nodiscard]] VisionChunk prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length);
    void release_encoded_media_payloads() noexcept;
    void retire_handoff() noexcept;
    [[nodiscard]] double elapsed_seconds() const;
    [[nodiscard]] bool cpu_encode() const noexcept { return cpu_weights_ != nullptr; }

    [[nodiscard]] std::size_t active_handoff_bytes() const noexcept {
        return active_handoff_bytes_;
    }

private:
    DeviceContext& device_;
    DeviceSpan workspace_;
    const VisionWorkspacePlan& workspace_plan_;
    qwen3_5::PreparedPromptData& prompt_;
    const VisionPrefillPlan& plan_;
    std::size_t& handoff_peak_bytes_;
    const vision_cpu::CpuVisionWeights* cpu_weights_ = nullptr;
    VisionContext context_;
    std::size_t next_use_ = 0;
    std::optional<std::uint32_t> active_item_;
    std::size_t active_handoff_bytes_ = 0;
    std::vector<std::uint32_t> encoded_payloads_pending_release_;
    std::vector<CudaEventTimer> timers_;
};

} // namespace ninfer::models::qwen3_5::execution
