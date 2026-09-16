#include "models/qwen3_5/load.h"

#include "models/qwen3_5/execution/vision_cpu/vision_cpu.h"
#include "artifact/formats.h"
#include "artifact/reader.h"
#include "artifact/schema.h"
#include "core/weight.h"
#include "models/qwen3_5/load/bindings.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5 {

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    ModelWeights weights;
    std::vector<loading::PendingWeight> pending;
    artifact::MaterializationPlan materialization;
    FrontendResources resources;
    InstanceInfo info;
    // Set (only when `options.vision_cpu_offload`) with the host-resident FP32 dequantized Vision
    // weights; otherwise `weights.vision` is populated and this is empty.
    std::optional<vision_cpu::CpuVisionWeights> cpu_vision;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const FrontendResources& LoadPlan::resources() const { return impl_->resources; }

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    return impl_->materialization;
}

const artifact::ParameterReference& LoadPlan::parameter(WeightId id) const {
    return impl_->pending.at(id.index).reference;
}

std::span<const WeightUse> LoadPlan::uses(WeightId id) const {
    return impl_->pending.at(id.index).uses;
}

const std::optional<vision_cpu::CpuVisionWeights>& LoadPlan::cpu_vision() const {
    return impl_->cpu_vision;
}

namespace {

// `--vision-cpu` host-resident dequant. Reads the artifact's encoded Vision payloads directly
// (object-identity agnostic) and reconstructs logical FP32 weights, mirroring the per-tensor
// representations the device `bind_vision` uses: row tensors through the bit-exact row-split
// decoder (`vision_cpu::dequant_row_split_lowbit`), direct (bias/norm) tensors as BF16. Each
// Vision tensor is a single artifact object (the same representation `bind_vision` binds), so the
// loader reads that object once and never places it in the device arena.
namespace vcpu_load {

ObjectHandle object_for(artifact::Binder& b, const std::string& name) {
    const auto& reader = b.reader();
    const auto found   = reader.directory().bindings.find(name);
    if (found == reader.directory().bindings.end()) {
        throw artifact::ArtifactError("vision_cpu: missing logical parameter " + name);
    }
    const auto& binding = found->second;
    if (binding.parts.size() != 1) {
        throw artifact::ArtifactError("vision_cpu: " + name + " must be a single-part object");
    }
    return binding.parts.front().object;
}

// Read + dequant a row-split grouped-quant weight to logical FP32 [n, k]. The artifact's stored
// representation is authoritative: the loader reads the object's actual `QType` and requires it to
// be a supported row-split low-bit type (Q4/Q5/Q6/W8). This is variant-agnostic and only rejects
// genuinely unsupported representations (e.g. NVFP4/FP8), which the CPU dequant cannot decode.
std::vector<float> dequant_quant(artifact::Binder& b, const std::string& name, std::int32_t n,
                                 std::int32_t k) {
    const auto handle = object_for(b, name);
    const auto& geom  = b.reader().geometry(handle);
    const std::int32_t qtype = [&] {
        switch (geom.format) {
        case QType::Q4_G64_FP16: return 0;
        case QType::Q5_G64_FP16: return 1;
        case QType::Q6_G64_FP16: return 2;
        case QType::Q8_G32_FP16: return 3;
        default:
            throw artifact::ArtifactError(
                "vision_cpu: " + name +
                " uses an unsupported quant format for the CPU offload (expected Q4/Q5/Q6/W8 "
                "row-split)");
        }
    }();
    const auto bytes = b.reader().read_object(handle);
    return vision_cpu::dequant_row_split_lowbit(
        reinterpret_cast<const std::uint8_t*>(bytes.data()), n, k, qtype);
}

std::vector<float> dequant_bf16(artifact::Binder& b, const std::string& name) {
    const auto handle = object_for(b, name);
    if (b.reader().geometry(handle).format != QType::BF16) {
        throw artifact::ArtifactError("vision_cpu: " + name + " is not a direct BF16 tensor");
    }
    const auto bytes  = b.reader().read_object(handle);
    const auto* words = reinterpret_cast<const std::uint16_t*>(bytes.data());
    const auto count  = bytes.size() / 2;
    std::vector<float> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) { out.push_back(vision_cpu::fp::bf16_to_f32(words[i])); }
    return out;
}

// Concatenate row blocks along the output-feature axis into [rows_a + rows_b + rows_c, k]; builds
// the packed QKV [3*H, H] (and its [3*H] bias) from the separate q/k/v tensors.
std::vector<float> stack_rows(std::vector<float> a, std::vector<float> b, std::vector<float> c) {
    a.reserve(a.size() + b.size() + c.size());
    a.insert(a.end(), b.begin(), b.end());
    a.insert(a.end(), c.begin(), c.end());
    return a;
}

} // namespace vcpu_load

std::optional<vision_cpu::CpuVisionWeights>
load_vision_cpu_weights(artifact::Binder& b, const VisionConfig& v, const TextConfig& target) {
    const auto hidden      = static_cast<std::int32_t>(v.hidden_size);
    const auto intermediate = static_cast<std::int32_t>(v.intermediate_size);
    const auto depth        = static_cast<std::int32_t>(v.depth);
    const auto patch        = static_cast<std::int32_t>(v.patch_width());
    const auto position     = static_cast<std::int32_t>(v.num_position_embeddings);
    const auto merger       = static_cast<std::int32_t>(v.merger_width());
    const auto out_hidden   = static_cast<std::int32_t>(target.hidden_size);

    using vcpu_load::dequant_bf16;
    using vcpu_load::dequant_quant;
    using vcpu_load::stack_rows;

    vision_cpu::CpuVisionWeights w;
    w.patch_embed = dequant_quant(b, "vision/patch_embedding", hidden, patch);
    w.patch_bias  = dequant_bf16(b, "vision/patch_embedding_bias");
    w.position_embedding =
        dequant_bf16(b, "vision/position_embedding"); // [position_embeddings, hidden] row-major.
    (void)position;
    w.layers.resize(hidden > 0 && depth > 0 ? static_cast<std::size_t>(depth) : 0);
    for (std::int32_t layer = 0; layer < depth; ++layer) {
        const std::string p = "vision/layers/" + std::to_string(layer) + "/";
        vision_cpu::CpuVisionWeights::LayerW& lw = w.layers[static_cast<std::size_t>(layer)];
        lw.qkv      = stack_rows(
            dequant_quant(b, p + "attention/query", hidden, hidden),
            dequant_quant(b, p + "attention/key", hidden, hidden),
            dequant_quant(b, p + "attention/value", hidden, hidden));      // [3*H, H]
        lw.qkv_bias = stack_rows(
            dequant_bf16(b, p + "attention/query_bias"),
            dequant_bf16(b, p + "attention/key_bias"),
            dequant_bf16(b, p + "attention/value_bias"));                  // [3*H]
        lw.out      = dequant_quant(b, p + "attention/output", hidden, hidden);
        lw.out_bias = dequant_bf16(b, p + "attention/output_bias");
        lw.fc1      = dequant_quant(b, p + "mlp/fc1", intermediate, hidden);
        lw.fc1_bias = dequant_bf16(b, p + "mlp/fc1_bias");
        lw.fc2      = dequant_quant(b, p + "mlp/fc2", hidden, intermediate);
        lw.fc2_bias = dequant_bf16(b, p + "mlp/fc2_bias");
        lw.norm1_w  = dequant_bf16(b, p + "norm1_weight");
        lw.norm1_b  = dequant_bf16(b, p + "norm1_bias");
        lw.norm2_w  = dequant_bf16(b, p + "norm2_weight");
        lw.norm2_b  = dequant_bf16(b, p + "norm2_bias");
    }
    w.merger_norm_w    = dequant_bf16(b, "vision/merger/norm_weight");
    w.merger_norm_b    = dequant_bf16(b, "vision/merger/norm_bias");
    w.merger_fc1       = dequant_quant(b, "vision/merger/fc1", merger, merger);
    w.merger_fc1_bias  = dequant_bf16(b, "vision/merger/fc1_bias");
    w.merger_fc2       = dequant_quant(b, "vision/merger/fc2", out_hidden, merger); // [O,Hm]
    w.merger_fc2_bias  = dequant_bf16(b, "vision/merger/fc2_bias");                  // [O]
    return w;
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->options = options;
    out->config  = parse_config(reader.directory(), options);
    artifact::Binder binder(reader);
    out->resources = loading::bind_resources(binder, out->config);
    loading::Bindings bindings(binder);
    const auto& text  = out->config.text;
    out->weights.text = loading::bind_text(bindings, text, options);
    if (out->config.vision) {
        if (options.vision_cpu_offload) {
            // `--vision-cpu`: skip the device binding so no Vision weight is selected into the
            // device arena, and instead dequantize the encoded payload into host-resident FP32
            // weights; the vision encoder then runs on the CPU during prefill.
            out->cpu_vision =
                load_vision_cpu_weights(bindings.binder, *out->config.vision, text);
        } else {
            out->weights.vision = loading::bind_vision(bindings, *out->config.vision, text);
        }
    }
    if (out->config.mtp) {
        out->weights.mtp = loading::bind_mtp(bindings, text, out->weights.text);
    }
    if (out->config.draft) {
        out->weights.draft =
            loading::bind_draft(bindings, *out->config.draft, text, out->weights.text,
                                std::string(options.speculative_component()));
    }
    if (options.proposal_enabled()) {
        const auto& proposal = reader.directory().component("text").proposal;
        if (!proposal) {
            throw artifact::ArtifactError("selected proposal head is absent from artifact");
        }
        out->weights.proposal = loading::bind_proposal(bindings, *proposal, text, options,
                                                       out->resources.public_token_count);
        if (out->config.draft && out->config.draft->dflash2) {
            const auto domain =
                proposal->indexed ? proposal->rows : out->resources.public_token_count;
            if (out->config.draft->dflash2->selector_top_k > domain) {
                throw artifact::ArtifactError(
                    "proposal domain is smaller than DFlash2 selector_top_k");
            }
        }
        if (out->weights.mtp) { out->weights.mtp->output_head = out->weights.proposal->head; }
        if (out->weights.draft) { out->weights.draft->output_head = out->weights.proposal->head; }
    }
    out->weights.text.output_head_use =
        bindings.use(out->weights.text.output_head, "text/final_hidden");
    if (out->weights.mtp) {
        out->weights.mtp->output_head_use =
            bindings.use(out->weights.mtp->output_head, "mtp/final_hidden");
    }
    if (out->weights.draft) {
        out->weights.draft->output_head_use =
            bindings.use(out->weights.draft->output_head,
                         std::string(options.speculative_component()) + "/final_hidden");
    }
    out->pending         = std::move(bindings.weights);
    out->materialization = std::move(binder).finish();
    out->info.name       = reader.directory().metadata.value(
        "name", std::string(architecture_name(text.architecture)));
    out->info.metadata_json   = reader.directory().metadata.dump();
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    return LoadPlan(std::move(out));
}

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    if (!plan.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    auto data    = std::move(plan.impl_);
    auto backing = artifact::materialize(*data->materialization.source,
                                         std::move(data->materialization), device, observer);
    auto bound   = loading::resolve_weights(std::move(data->pending), backing);
    return std::unique_ptr<Model>(new Model(
        std::move(data->config), data->options, std::move(data->weights), std::move(bound),
        std::move(data->resources), std::move(data->info), std::move(backing),
        std::move(data->cpu_vision)));
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                  DeviceContext& device, const StartupObserver* observer) {
    artifact::Reader reader(path);
    return materialize_model(plan_load(reader, options), device, observer);
}

} // namespace ninfer::models::qwen3_5
