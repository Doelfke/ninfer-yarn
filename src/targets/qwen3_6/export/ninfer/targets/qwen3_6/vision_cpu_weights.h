#pragma once

// Host-resident dequantized FP32 vision weights for the `--vision-cpu` offload.
//
// This struct is deliberately tiny and pure host C++ (no CUDA, no artifact, no geometry) so it
// can be a complete type in the export tier: `targets/qwen3_6/model_view.h` keeps a
// `std::optional<vision_cpu::CpuVisionWeights>` next to the device-resident `VisionWeights`.
// The actual dequant, the geometry, and the CPU forward pass live in the impl header
// `targets/qwen3_6/impl/runtime/vision_cpu/vision_cpu.h`, which re-exposes this type and is the
// owner of everything that actually runs here.

#include <cstddef>
#include <vector>

namespace ninfer::targets::qwen3_6::vision_cpu {

// Dequantized-to-FP32 vision weights as consumed by the CPU encode. Every matrix is logical
// [rows, cols] row-major (row == output feature, col == input feature), matching the device
// `linear` Op's `W @ X` contract. The merger output projection rows (`out_hidden`) are the
// variant's `TextConfig::hidden` (5120 for the 27B, 2048 for the 35B-A3B), so they are carried
// implicitly by the `merger_fc2`/`merger_fc2_bias` sizes rather than a constant.
struct CpuVisionWeights {
    // Backbone.
    std::vector<float> patch_embed;        // [hidden, patch_dim]
    std::vector<float> patch_bias;         // [hidden]
    std::vector<float> position_embedding; // [position_embeddings, hidden] row-major.
    struct LayerW {
        std::vector<float> qkv;              // [3*hidden, hidden]
        std::vector<float> out;              // [hidden, hidden]
        std::vector<float> fc1;              // [intermediate, hidden]
        std::vector<float> fc2;              // [hidden, intermediate]
        std::vector<float> norm1_w, norm1_b, norm2_w, norm2_b;                       // [hidden]
        std::vector<float> qkv_bias, out_bias, fc1_bias, fc2_bias;
    };
    std::vector<LayerW> layers;
    // Merger.
    std::vector<float> merger_norm_w, merger_norm_b; // [hidden]
    std::vector<float> merger_fc1;                   // [merger_hidden, merger_hidden]
    std::vector<float> merger_fc1_bias;              // [merger_hidden]
    std::vector<float> merger_fc2;                   // [out_hidden, merger_hidden]
    std::vector<float> merger_fc2_bias;              // [out_hidden]

    // Total dequantized FP32 weight bytes (for host-side / VRAM-savings reporting).
    [[nodiscard]] std::size_t bytes() const;
};

inline std::size_t CpuVisionWeights::bytes() const {
    std::size_t s = 0;
    auto add = [&](const std::vector<float>& v) { s += v.size() * sizeof(float); };
    add(patch_embed);
    add(patch_bias);
    add(position_embedding);
    for (const auto& l : layers) {
        add(l.qkv);
        add(l.out);
        add(l.fc1);
        add(l.fc2);
        add(l.norm1_w);
        add(l.norm1_b);
        add(l.norm2_w);
        add(l.norm2_b);
        add(l.qkv_bias);
        add(l.out_bias);
        add(l.fc1_bias);
        add(l.fc2_bias);
    }
    add(merger_norm_w);
    add(merger_norm_b);
    add(merger_fc1);
    add(merger_fc1_bias);
    add(merger_fc2);
    add(merger_fc2_bias);
    return s;
}

} // namespace ninfer::targets::qwen3_6::vision_cpu
