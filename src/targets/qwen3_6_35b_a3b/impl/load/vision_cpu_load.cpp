#include "targets/qwen3_6_35b_a3b/impl/load/vision_cpu_load.h"

#include <ninfer/targets/qwen3_6/vision.h>
#include "targets/qwen3_6/impl/runtime/vision_cpu/vision_cpu.h"
#include "targets/qwen3_6_35b_a3b/impl/config.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen3_6_35b_a3b::detail {
namespace {

using qc = qwen3_6::vision_cpu;

inline std::vector<float> quant_w(artifact::Binder& binder, artifact::ObjectHandle h, int n, int k,
                                  artifact::NumericFormat format) {
    const auto span = binder.payload(h);
    const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(span.data.data());
    return qc::dequant_row_split_lowbit(bytes, n, k, static_cast<std::int32_t>(format));
}

inline std::vector<float> bf16_w(artifact::Binder& binder, artifact::ObjectHandle h) {
    const auto span = binder.payload(h);
    const std::uint16_t* words = reinterpret_cast<const std::uint16_t*>(span.data.data());
    const std::size_t count = span.data.size();
    std::vector<float> out(count);
    for (std::size_t i = 0; i < count; ++i) { out[i] = qc::fp::bf16_to_f32(words[i]); }
    return out;
}

} // namespace

std::optional<qwen3_6::vision_cpu::CpuVisionWeights>
load_vision_cpu_weights(artifact::Binder& binder, const qwen3_6::VisionBackbonePlan& backbone,
                        const qwen3_6::VisionMergerInputPlan& merger_input,
                        const qwen3_6::VisionMergerNormPlan& merger_norm,
                        artifact::ObjectHandle merger_fc2, artifact::ObjectHandle merger_fc2_bias) {
    using artifact::NumericFormat;
    constexpr int H            = qwen3_6::VisionBackboneConfig::hidden;
    constexpr int PD           = qwen3_6::VisionBackboneConfig::patch_dim;
    constexpr int IH           = qwen3_6::VisionBackboneConfig::intermediate;
    constexpr int MH           = qwen3_6::VisionBackboneConfig::merger_hidden;
    constexpr int QKVO         = 3 * H;
    constexpr int OUTH         = TextConfig::hidden; // merger output projection rows (2048).
    constexpr int LAYOUT_LAYERS = qwen3_6::VisionBackboneConfig::layers;

    qwen3_6::vision_cpu::CpuVisionWeights w;
    w.patch_embed        = quant_w(binder, backbone.patch_embedding, H, PD, NumericFormat::Q6G64_F16S);
    w.patch_bias         = bf16_w(binder, backbone.patch_embedding_bias);
    w.position_embedding = bf16_w(binder, backbone.position_embedding); // [2304,1152] row-major

    w.layers.resize(LAYOUT_LAYERS);
    for (int layer = 0; layer < LAYOUT_LAYERS; ++layer) {
        const qwen3_6::VisionLayerPlan& src = backbone.layers[static_cast<std::size_t>(layer)];
        qc::CpuVisionWeights::LayerW& target = w.layers[static_cast<std::size_t>(layer)];
        target.qkv       = quant_w(binder, src.qkv, QKVO, H, NumericFormat::Q4G64_F16S);
        target.qkv_bias  = bf16_w(binder, src.qkv_bias);
        target.out       = quant_w(binder, src.output, H, H, NumericFormat::Q5G64_F16S);
        target.out_bias  = bf16_w(binder, src.output_bias);
        target.fc1       = quant_w(binder, src.fc1, IH, H, NumericFormat::Q4G64_F16S);
        target.fc1_bias  = bf16_w(binder, src.fc1_bias);
        target.fc2       = quant_w(binder, src.fc2, H, IH, NumericFormat::Q5G64_F16S);
        target.fc2_bias  = bf16_w(binder, src.fc2_bias);
        target.norm1_w   = bf16_w(binder, src.norm1_weight);
        target.norm1_b   = bf16_w(binder, src.norm1_bias);
        target.norm2_w   = bf16_w(binder, src.norm2_weight);
        target.norm2_b   = bf16_w(binder, src.norm2_bias);
    }

    w.merger_norm_w   = bf16_w(binder, merger_norm.weight);
    w.merger_norm_b   = bf16_w(binder, merger_norm.bias);
    w.merger_fc1      = quant_w(binder, merger_input.fc1, MH, MH, NumericFormat::W8G32_F16S);
    w.merger_fc1_bias = bf16_w(binder, merger_input.fc1_bias);
    w.merger_fc2      = quant_w(binder, merger_fc2, OUTH, MH, NumericFormat::W8G32_F16S); // [2048,4608]
    w.merger_fc2_bias = bf16_w(binder, merger_fc2_bias);                                 // [2048]

    return w;
}

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail
