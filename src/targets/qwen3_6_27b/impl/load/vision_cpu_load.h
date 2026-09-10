#pragma once

// Load-time host materialization of the Qwen3.6 27B vision weights for the `--vision-cpu` offload.
//
// Called from `bind_artifact` while the artifact host payload is still reachable (before
// `binder.finish()`). It reads every vision object's raw payload, dequantizes the grouped
// quantized tensors (Q4/Q5/Q6/W8) to FP32, and converts the contiguous BF16 tensors to FP32,
// producing a single `CpuVisionWeights` whose data lives on the host and is never bound to the
// device arena. The plan (with those handles) is passed in so the shapes/formats stay in one
// authoritative source: `impl/vision/bindings.cpp`.

#include <ninfer/targets/qwen3_6/vision_cpu_weights.h>

#include "artifact/binder.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include <cstdint>

namespace ninfer::targets::qwen3_6_27b::detail {

[[nodiscard]] std::optional<qwen3_6::vision_cpu::CpuVisionWeights>
load_vision_cpu_weights(artifact::Binder& binder, const qwen3_6::VisionBackbonePlan& backbone,
                        const qwen3_6::VisionMergerInputPlan& merger_input,
                        const qwen3_6::VisionMergerNormPlan& merger_norm,
                        artifact::ObjectHandle merger_fc2, artifact::ObjectHandle merger_fc2_bias);

} // namespace ninfer::targets::qwen3_6_27b::detail
