#pragma once

// Load-time host materialization of the Qwen3.6 35B-A3B vision weights for the `--vision-cpu`
// offload. Mirrors the 27B helper; the only difference is the merger output projection extent,
// which is the variant's `TextConfig::hidden` (2048 here) rather than 5120.

#include <ninfer/targets/qwen3_6/vision_cpu_weights.h>

#include "artifact/binder.h"
#include "targets/qwen3_6_35b_a3b/impl/load/bindings.h"

#include <cstdint>

namespace ninfer::targets::qwen3_6_35b_a3b::detail {

[[nodiscard]] std::optional<qwen3_6::vision_cpu::CpuVisionWeights>
load_vision_cpu_weights(artifact::Binder& binder, const qwen3_6::VisionBackbonePlan& backbone,
                        const qwen3_6::VisionMergerInputPlan& merger_input,
                        const qwen3_6::VisionMergerNormPlan& merger_norm,
                        artifact::ObjectHandle merger_fc2, artifact::ObjectHandle merger_fc2_bias);

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail
