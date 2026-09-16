#pragma once

#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/execution/vision_cpu/vision_cpu_weights.h"
#include "models/qwen3_5/weights.h"
#include "ninfer/ops/weight_input.h"

#include <memory>
#include <span>
#include <string>

namespace ninfer::models::qwen3_5 {

struct InstanceInfo {
    std::string name;
    std::string metadata_json;
    std::string provenance_json;
    artifact::ArtifactId artifact_id{};
};

class LoadPlan;

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&)                 = delete;
    Model& operator=(Model&&)      = delete;

    [[nodiscard]] const Config& config() const noexcept { return config_; }

    [[nodiscard]] const LoadOptions& options() const noexcept { return options_; }

    [[nodiscard]] const ModelWeights& weights() const noexcept { return weights_; }

    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }

    [[nodiscard]] ops::WeightInput input(WeightUseId id) const;
    [[nodiscard]] ops::WeightInput input(WeightId id) const;

    [[nodiscard]] std::span<const BoundWeight> weight_data() const noexcept { return bound_; }

    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }

    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

    // Host-resident dequantized FP32 Vision weights for the `--vision-cpu` offload. Set (and the
    // device `vision` parameters absent) only when `LoadOptions::vision_cpu_offload` is requested;
    // the encoder then runs on the host and no Vision weight is loaded into the device arena.
    [[nodiscard]] const std::optional<vision_cpu::CpuVisionWeights>& cpu_vision() const noexcept {
        return cpu_vision_;
    }

private:
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    Model(Config config, LoadOptions options, ModelWeights weights, std::vector<BoundWeight> bound,
          FrontendResources resources, InstanceInfo info, artifact::MaterializedArtifact backing,
          std::optional<vision_cpu::CpuVisionWeights> cpu_vision = std::nullopt);

    // Destroy all borrowers before backing. The caller keeps DeviceContext alive through cleanup.
    artifact::MaterializedArtifact backing_;
    Config config_;
    LoadOptions options_;
    ModelWeights weights_;
    std::vector<BoundWeight> bound_;
    FrontendResources resources_;
    InstanceInfo info_;
    std::optional<vision_cpu::CpuVisionWeights> cpu_vision_;
};

} // namespace ninfer::models::qwen3_5
