#pragma once

// Self-contained CPU implementation of the Qwen3.6 vision encoder (ViT) forward pass.
//
// This is the host-side half of the `--vision-cpu` offload. When that option is set, the
// vision weights are materialized into host DRAM and dequantized to FP32 once at load time
// (see vision_cpu_dequant), and the entire encoder runs here instead of the CUDA Op graph.
// Only the final projected embedding [out_hidden, V] is handed back to the device; the text
// prefill path is otherwise unchanged.
//
// Every op below mirrors the device kernel of the same name (see docs/maintainer/op-development.md
// and src/ops/...). The dequant is the canonical logical-weight reconstruction
// (`signed_code * fp16(group_scale)`) and is qualified bit-exactly against the test reference
// decoder tests/ops/quantized_weight.h (decode_row_split_lowbit / unpack_lowbit_code). The math
// ops are qualified against independent FP64 oracles in the test suite (linear FP64 GEMM,
// softmax_attention/oracle.h dense attention, and naive layer-norm/gelu/rope references).
//
// This header is pure host C++ (no CUDA) so it builds and is unit-tested on non-GPU machines.
//
// The `CpuVisionWeights` value type is defined in the export-tier header
// `ninfer/targets/qwen3_6/vision_cpu_weights.h` (pure host, small) so that the export-tier
// `model_view.h` can keep a complete `std::optional<CpuVisionWeights>` member. This header
// re-exposes it in the same `ninfer::targets::qwen3_6::vision_cpu` namespace and owns the math
// (dequant, geometry, and the CPU forward pass). Callers using a `namespace` alias such as
// `namespace vc = ninfer::targets::qwen3_6::vision_cpu;` reach both the struct and the functions.

#include <ninfer/targets/qwen3_6/vision_cpu_weights.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ninfer::targets::qwen3_6::vision_cpu {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Geometry (mirrors src/targets/qwen3_6/export/ninfer/targets/qwen3_6/vision.h).
// ---------------------------------------------------------------------------
namespace geom {
inline constexpr int layers                  = 27;
inline constexpr int hidden                  = 1152;
inline constexpr int intermediate            = 4304;
inline constexpr int heads                   = 16;
inline constexpr int head_dim                = 72;
inline constexpr int patch_dim               = 1536; // 3*2*16*16 raw-patch features.
inline constexpr int merge_unit              = 4;   // 2*2 merge.
inline constexpr int merger_hidden           = hidden * merge_unit; // 4608
inline constexpr int out_hidden              = 5120;               // TextConfig::hidden.
inline constexpr int position_embeddings     = 48 * 48;            // 2304
inline constexpr int rotary_dim              = head_dim;           // 72
inline constexpr float rope_theta            = 10'000.0F;
inline constexpr float norm_epsilon          = 1.0e-6F;
inline constexpr float attention_scale       = 0.11785113019775792F;
inline constexpr int qkv_out                  = 3 * hidden;         // 3456
} // namespace geom

// ---------------------------------------------------------------------------
// Floating-point conversions (copy of the exact test-reference implementations in
// tests/ops/quantized_weight.h::detail so the CPU dequant is byte-for-byte the reference).
// ---------------------------------------------------------------------------
namespace fp {

inline std::uint32_t float_bits(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

inline float bits_float(std::uint32_t u) {
    float f  = 0.0F;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// IEEE half (fp16) -> float, bit-exact with the reference `f16_to_f32`.
inline float f16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
    std::uint32_t exp        = (static_cast<std::uint32_t>(h) >> 10) & 0x1fu;
    std::uint32_t mant       = static_cast<std::uint32_t>(h) & 0x03ffu;
    if (exp == 0) {
        if (mant == 0) { return bits_float(sign); }
        int e = -14;
        while ((mant & 0x0400u) == 0) {
            mant <<= 1;
            --e;
        }
        mant &= 0x03ffu;
        return bits_float(sign | (static_cast<std::uint32_t>(e + 127) << 23) | (mant << 13));
    }
    if (exp == 31) { return bits_float(sign | 0x7f800000u | (mant << 13)); }
    exp = exp - 15 + 127;
    return bits_float(sign | (exp << 23) | (mant << 13));
}

// BF16 (16-bit, the top half of a float) -> float.
inline float bf16_to_f32(std::uint16_t h) {
    return bits_float(static_cast<std::uint32_t>(h) << 16);
}

// float -> nearest BF16 (round-to-nearest-even, ties to even), for the final embedding handoff.
// Only the low 16 bits of the float are involved in the rounding decision, so negative inputs
// never erroneously cross into the exponent overflow region.
inline std::uint16_t f32_to_bf16(float f) {
    const std::uint32_t x       = float_bits(f);
    const std::uint32_t low     = x & 0xffffu;
    const std::uint32_t lsb     = (x >> 16) & 1u;
    std::uint32_t keep          = x & 0xffff0000u;
    if (low > 0x7fffu || (low == 0x7fffu && lsb)) {
        keep += 0x10000u;
        // A carry into the exponent would only overflow if the retained bits are already in
        // the exponent-overflow region (real inf / NaN source). Clamp to signed infinity.
        if ((keep & 0x7f800000u) == 0x7f800000u) {
            keep = (x >> 31) ? 0xff800000u : 0x7f800000u;
        }
    }
    return static_cast<std::uint16_t>(keep >> 16);
}

} // namespace fp

// ---------------------------------------------------------------------------
// Row-split low-bit dequant -> logical FP32 weight matrix [n, k] (row-major).
//
// Plane layout (src/artifact/storage_layouts.cpp::row_split_geometry):
//   padded_k = align_up(k, 128); kg = padded_k / group_size;
//   nib = (bits==8 ? group : group/2); high_bpr = (bits<=4 ? 0 : group*(bits-4)/8);
//   high_off  = align_up(n*kg*nib, 256); scale_off = high_off + align_up(n*kg*high_bpr, 256).
// The dequant is bit-exact with the reference unpack_lowbit_code (see tests/ops/quantized_weight.h).
// ---------------------------------------------------------------------------
inline int quant_bits(std::int32_t qtype) {
    switch (qtype) {
    case 0: // Q4G64_F16S (QType::Q4G64_F16S)
        return 4;
    case 1: // Q5G64_F16S
        return 5;
    case 2: // Q6G64_F16S
        return 6;
    case 3: // W8G32_F16S
        return 8;
    default:
        throw std::invalid_argument("vision_cpu: unsupported row-split qtype for dequant");
    }
}

struct RowSplitDequantSpec {
    int bits;
    int group_size;
    int nibbles;
    int high_bpr;
};

inline RowSplitDequantSpec dequant_spec(std::int32_t qtype) {
    const int bits  = quant_bits(qtype);
    const int group = (bits == 8) ? 32 : 64;
    const int nib   = (bits == 8) ? group : group / 2;
    // W8 has no high plane (all 8 bits live in the code byte); Q5/Q6 carry their extra bits there.
    const int high  = (bits == 8 || bits <= 4) ? 0 : group * (bits - 4) / 8;
    return {bits, group, nib, high};
}

// Decode one code at bit-index `lane` of a group. Bit-exact with reference unpack_lowbit_code.
inline int unpack_code(const std::uint8_t* nibble, const std::uint8_t* high, const RowSplitDequantSpec& s,
                       int lane) {
    if (s.bits == 8) {
        return static_cast<int>(static_cast<std::int8_t>(nibble[lane]));
    }
    const std::uint8_t low_byte = nibble[lane >> 1];
    const std::uint32_t low     = (lane & 1) ? (low_byte >> 4) : (low_byte & 0x0fu);
    std::uint32_t hi            = 0;
    if (s.bits == 5) {
        hi = (high[lane >> 3] >> (lane & 7)) & 0x01u;
    } else if (s.bits == 6) {
        const int bitpos = lane * 2;
        hi               = (high[bitpos >> 3] >> (bitpos & 7)) & 0x03u;
    }
    const std::uint32_t u    = low | (hi << 4);
    const std::uint32_t sign = 1u << (s.bits - 1);
    const std::uint32_t span = 1u << s.bits;
    return (u & sign) ? static_cast<int>(u) - static_cast<int>(span) : static_cast<int>(u);
}

// `payload` is the whole encoded tensor (code plane | high plane | scale plane) exactly as
// materialized in the artifact. `n`/`k` are the logical [rows, cols]. Returns FP32 [n, k] row-major.
inline std::vector<float> dequant_row_split_lowbit(const std::uint8_t* payload, std::int32_t n,
                                                   std::int32_t k, std::int32_t qtype) {
    if (n <= 0 || k <= 0) {
        throw std::invalid_argument("vision_cpu: dequant shape must be positive");
    }
    const RowSplitDequantSpec s          = dequant_spec(qtype);
    const int padded_k                   = ((k + 127) / 128) * 128;
    const int kg                         = padded_k / s.group_size;
    const std::size_t nibble_bytes       = static_cast<std::size_t>(n) * kg * s.nibbles;
    const std::size_t high_bytes          = static_cast<std::size_t>(n) * kg * s.high_bpr;
    const std::size_t high_off            = ((nibble_bytes + 255) / 256) * 256;
    const std::size_t scale_off           = high_off + ((high_bytes + 255) / 256) * 256;

    auto load_u16 = [&payload](std::size_t off) {
        return static_cast<std::uint16_t>(payload[off]) |
               static_cast<std::uint16_t>(static_cast<std::uint16_t>(payload[off + 1]) << 8);
    };
    std::vector<float> deq(static_cast<std::size_t>(n) * k);
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t g = 0; g < kg; ++g) {
            const std::size_t group_index = static_cast<std::size_t>(row) * kg + g;
            const float scale             = fp::f16_to_f32(load_u16(scale_off + group_index * 2));
            const std::uint8_t* nibble    = payload + group_index * s.nibbles;
            const std::uint8_t* high =
                s.high_bpr == 0 ? nullptr : payload + high_off + group_index * s.high_bpr;
            for (std::int32_t lane = 0; lane < s.group_size; ++lane) {
                const std::int32_t kk = g * s.group_size + lane;
                if (kk >= k) { continue; } // padded tail is never used.
                const int code = unpack_code(nibble, high, s, lane);
                deq[static_cast<std::size_t>(row) * k + kk] = static_cast<float>(code) * scale;
            }
        }
    }
    return deq;
}

// ---------------------------------------------------------------------------
// Small matrix / linear-algebra helpers (FP32, row-major).
//
//   gemm(W[N,K], X[K,T]) -> Y[N,T];  Y[n][t] = sum_k W[n][k] * X[k][t]
//
// The activation `X` is the [features, tokens] block; `W` the [out, in] dequantized weight.
// ---------------------------------------------------------------------------
inline int clamp_threads(std::int64_t requested, int rows) {
    std::uint32_t hw = std::thread::hardware_concurrency();
    if (hw == 0) { hw = 1; }
    std::int64_t cap = static_cast<std::int64_t>(hw);
    std::int64_t t   = std::min(requested <= 0 ? cap : requested, cap);
    t               = std::min(t, static_cast<std::int64_t>(rows));
    return static_cast<int>(std::max<std::int64_t>(1, t));
}

inline void gemm(const float* w, int n, int k, const float* x, int t, float* y, int threads) {
    if (n <= 0 || k <= 0 || t <= 0) { throw std::invalid_argument("vision_cpu: gemm extent must be positive"); }
    const int th = clamp_threads(threads, n);
    const std::int64_t row0 = static_cast<std::int64_t>(n) / th;
    // Output is token-major / feature-fastest (y[token][n]), matching every other vision-cpu op and
    // the reference cpu_linear_gemm_fp64. The thread split is over output rows (features).
    auto worker = [&](int a, int b) {
        for (int row = static_cast<int>(a); row < static_cast<int>(b); ++row) {
            const float* wrow = w + static_cast<std::size_t>(row) * k;
            float* ycol       = y + row;                 // y[token][n], so feature row stride is n
            for (int col = 0; col < t; ++col) {
                float acc = 0.0F;
                const float* wp = wrow;
                for (int kk = 0; kk < k; ++kk) {
                    acc += wp[kk] * x[static_cast<std::size_t>(col) * k + kk];
                }
                ycol[static_cast<std::size_t>(col) * n] = acc;
            }
        }
    };
    if (th == 1) { worker(0, n); }
    else {
        std::vector<std::thread> pool;
        pool.reserve(th);
        for (int i = 0; i < th; ++i) {
            const std::int64_t a = row0 * i;
            const std::int64_t b = (i == th - 1) ? n : row0 * (i + 1);
            pool.emplace_back(worker, a, b);
        }
        for (auto& thd : pool) { thd.join(); }
    }
}

// Y[N,T] += bias[N] (broadcast across columns).
inline void add_bias(float* y, const float* bias, int n, int t) {
    for (int c = 0; c < t; ++c) {
        for (int r = 0; r < n; ++r) {
            y[static_cast<std::size_t>(c) * n + r] += bias[r];
        }
    }
}

// a[i][j] += b[i][j] over [n,t].
inline void residual_add(float* a, int n, int t, const float* b, int threads = 0) {
    const std::size_t total = static_cast<std::size_t>(n) * t;
    const int th = clamp_threads(threads, static_cast<int>(std::max<std::size_t>(1, total / 1024)));
    auto worker = [&](std::int64_t start, std::int64_t end) {
        for (std::int64_t j = start; j < end; ++j) { a[j] += b[j]; }
    };
    if (th == 1) { worker(0, static_cast<std::int64_t>(total)); }
    else {
        const std::int64_t row0 = static_cast<std::int64_t>(total) / th;
        std::vector<std::thread> pool;
        pool.reserve(th);
        for (int i = 0; i < th; ++i) {
            const std::int64_t a = row0 * i;
            const std::int64_t b = (i == th - 1) ? static_cast<std::int64_t>(total) : row0 * (i + 1);
            pool.emplace_back(worker, a, b);
        }
        for (auto& thd : pool) { thd.join(); }
    }
}

// Per-column LayerNorm over [features, tokens]: for each token col p, mean/var over r in [0,features);
// y[r][p] = (x[r][p]-mean)/sqrt(var+eps) * w[r] + b[r]. `eps` is inside the sqrt (matches the device op).
inline void layer_norm(const float* x, int features, int tokens, const float* w, const float* b, float eps,
                       float* y, int threads = 0) {
    const int th = clamp_threads(threads, tokens);
    auto worker = [&](int i0, int i1) {
        for (int p = i0; p < i1; ++p) {
            float mean = 0.0F;
            for (int r = 0; r < features; ++r) { mean += x[static_cast<std::size_t>(p) * features + r]; }
            mean /= features;
            float var = 0.0F;
            for (int r = 0; r < features; ++r) {
                const float d = x[static_cast<std::size_t>(p) * features + r] - mean;
                var += d * d;
            }
            var /= features;
            const float inv = 1.0F / std::sqrt(var + eps);
            for (int r = 0; r < features; ++r) {
                const float d = x[static_cast<std::size_t>(p) * features + r] - mean;
                y[static_cast<std::size_t>(p) * features + r] = d * inv * w[r] + b[r];
            }
        }
    };
    if (th == 1) { worker(0, tokens); }
    else {
        const int row0 = tokens / th;
        std::vector<std::thread> pool;
        pool.reserve(th);
        for (int i = 0; i < th; ++i) {
            const int a = row0 * i;
            const int b = (i == th - 1) ? tokens : row0 * (i + 1);
            pool.emplace_back(worker, a, b);
        }
        for (auto& thd : pool) { thd.join(); }
    }
}

inline float gelu_tanh(float x) {
    const float inner = x + 0.044715F * x * x * x;
    return 0.5F * x * (1.0F + std::tanh(std::sqrt(2.0F / static_cast<float>(M_PI)) * inner));
}

inline float gelu_exact(float x) {
    return 0.5F * x * (1.0F + std::erf(x / std::sqrt(2.0F)));
}

// Element-wise tanh/exact gelu over a buffer.
inline void apply_gelu(float* v, std::size_t count, bool exact, int threads = 0) {
    const int th = clamp_threads(threads, static_cast<int>(std::max<std::size_t>(1, count / 1024)));
    auto worker = [&](std::int64_t a, std::int64_t b) {
        if (exact) { for (std::int64_t i = a; i < b; ++i) { v[i] = gelu_exact(v[i]); } }
        else { for (std::int64_t i = a; i < b; ++i) { v[i] = gelu_tanh(v[i]); } }
    };
    if (th == 1) { worker(0, static_cast<std::int64_t>(count)); }
    else {
        const std::int64_t row0 = static_cast<std::int64_t>(count) / th;
        std::vector<std::thread> pool;
        pool.reserve(th);
        for (int i = 0; i < th; ++i) {
            const std::int64_t a = row0 * i;
            const std::int64_t b = (i == th - 1) ? static_cast<std::int64_t>(count) : row0 * (i + 1);
            pool.emplace_back(worker, a, b);
        }
        for (auto& thd : pool) { thd.join(); }
    }
}

// ---------------------------------------------------------------------------
// Vision 2-D RoPE (split-half NeoX, rotary_dim 72).
//
//   positions: i32, axis-major [2, tokens] -> position_ids[axis * tokens + token].
//   For pair i in [0, 36): axis = i / 18, frequency = theta^(-2*(i%18)/36).
//   ang  = positions[axis][token] * frequency; c=cos, s=sin.
//   out[d]       = x[d]       * c - x[d+36] * s
//   out[d+36]    = x[d+36]    * c + x[d]    * s
// Applied in place to q and k, which are [72, heads, tokens] (feature fastest).
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Vision 2-D RoPE (split-half NeoX, rotary_dim 72).
//
//   positions: i32, axis-major [2, tokens] -> position_ids[axis * tokens + token].
//   For pair i in [0, 36): axis = i / 18, frequency = kVisionRopeInvFrequency[i % 18]
//                          = theta^(-2*(i%18)/36).
//   ang  = positions[axis][token] * frequency; c=cos, s=sin.
//   out[d]       = x[d]       * c - x[d+36] * s
//   out[d+36]    = x[d+36]    * c + x[d]    * s
// Applied in place to q and k, which are [72, heads, tokens] (feature fastest).
// The frequency table is the exact device constant (src/ops/kernel/rope.cuh kVisionRopeInvFrequency).
// ---------------------------------------------------------------------------
namespace rope_consts {
inline constexpr float kVisionInvFreq[18] = {
    1.000000000e+00F, 5.994842503e-01F, 3.593813664e-01F, 2.154434690e-01F, 1.291549665e-01F,
    7.742636827e-02F, 4.641588834e-02F, 2.782559402e-02F, 1.668100537e-02F, 1.000000000e-02F,
    5.994842503e-03F, 3.593813664e-03F, 2.154434690e-03F, 1.291549665e-03F, 7.742636827e-04F,
    4.641588834e-04F, 2.782559402e-04F, 1.668100537e-04F};
} // namespace rope_consts

inline void vision_rope(float* q, const std::int32_t* position_ids, int tokens, int heads,
                        int threads = 0) {
    const int R = geom::rotary_dim; // 72
    const int half = R / 2;        // 36
    // Precompute cos/sin per (pair, token). pairs = 36.
    // Each element writes pair_cos/sin[i * tokens + t] for a distinct `t` -> parallel per `t`.
    std::vector<float> pair_cos(static_cast<std::size_t>(half) * tokens);
    std::vector<float> pair_sin(static_cast<std::size_t>(half) * tokens);
    {
        const int th = clamp_threads(threads, tokens);
        auto worker  = [&](int a, int b) {
            for (int t = a; t < b; ++t) {
                for (int i = 0; i < half; ++i) {
                    const int axis      = i / 18; // 0 for i<18, 1 for i>=18
                    const float frequency = rope_consts::kVisionInvFreq[i % 18];
                    const int32_t ax_pos   = position_ids[static_cast<std::size_t>(axis) * tokens + t];
                    const float ang        = static_cast<float>(ax_pos) * frequency;
                    pair_cos[static_cast<std::size_t>(i) * tokens + t] = std::cos(ang);
                    pair_sin[static_cast<std::size_t>(i) * tokens + t] = std::sin(ang);
                }
            }
        };
        if (th == 1) { worker(0, tokens); }
        else {
            const int row0 = tokens / th;
            std::vector<std::thread> pool;
            pool.reserve(th);
            for (int i = 0; i < th; ++i) {
                const int a = row0 * i;
                const int b = (i == th - 1) ? tokens : row0 * (i + 1);
                pool.emplace_back(worker, a, b);
            }
            for (auto& thd : pool) { thd.join(); }
        }
    }
    // Rotate per (head, token, pair). Writes to q[(t*heads+h)*R + i] are disjoint per t -> parallel per t.
    {
        const int th = clamp_threads(threads, tokens);
        auto worker  = [&](int a, int b) {
            for (int t = a; t < b; ++t) {
                for (int h = 0; h < heads; ++h) {
                    float* base = q + (static_cast<std::size_t>(t) * heads + h) * R;
                    for (int i = 0; i < half; ++i) {
                        const float c = pair_cos[static_cast<std::size_t>(i) * tokens + t];
                        const float s = pair_sin[static_cast<std::size_t>(i) * tokens + t];
                        const float x = base[i];
                        const float y = base[i + half];
                        base[i]       = x * c - y * s;
                        base[i + half] = y * c + x * s;
                    }
                }
            }
        };
        if (th == 1) { worker(0, tokens); }
        else {
            const int row0 = tokens / th;
            std::vector<std::thread> pool;
            pool.reserve(th);
            for (int i = 0; i < th; ++i) {
                const int a = row0 * i;
                const int b = (i == th - 1) ? tokens : row0 * (i + 1);
                pool.emplace_back(worker, a, b);
            }
            for (auto& thd : pool) { thd.join(); }
        }
    }
}

// ---------------------------------------------------------------------------
// vision_pos_embed_add (in place): x[hidden, patches] += bilinear positional embedding.
//
//   table  : position_embedding fp32 [position_embeddings(2304) * hidden] (row-major).
//   indices: i32 [4 * patches]; weights: fp32 [4 * patches] (per-patch control plane).
//   x[h][p] += sum_{c=0..3} table[ indices[4p+c] ][h] * weights[4p+c]  (skipped if index out of range).
// ---------------------------------------------------------------------------
inline void vision_pos_embed_add(float* x, const float* table, const std::int32_t* indices,
                                 const float* weights, int patches, int threads = 0) {
    const int d      = geom::hidden;
    const int table_rows = geom::position_embeddings;
    const int th = clamp_threads(threads, patches);
    auto worker = [&](int a, int b) {
        for (int p = a; p < b; ++p) {
            for (int h = 0; h < d; ++h) {
                float pos = 0.0F;
                for (int c = 0; c < 4; ++c) {
                    const std::int32_t idx = indices[static_cast<std::size_t>(p) * 4 + c];
                    if (idx >= 0 && idx < table_rows) {
                        pos += table[static_cast<std::size_t>(idx) * d + h] * weights[static_cast<std::size_t>(p) * 4 + c];
                    }
                }
                x[static_cast<std::size_t>(p) * d + h] += pos;
            }
        }
    };
    if (th == 1) { worker(0, patches); }
    else {
        const int row0 = patches / th;
        std::vector<std::thread> pool;
        pool.reserve(th);
        for (int i = 0; i < th; ++i) {
            const int a = row0 * i;
            const int b = (i == th - 1) ? patches : row0 * (i + 1);
            pool.emplace_back(worker, a, b);
        }
        for (auto& thd : pool) { thd.join(); }
    }
}

// ---------------------------------------------------------------------------
// Packed block-diagonal dense attention. The Qwen3.6 ViT runs one independent dense
// (non-causal) segment per temporal frame: tokens are grouped into `frames` frames of
// `segment_length` patches each, and a query attends only to keys within the same frame.
// q/k/v/out are [head_dim, heads, tokens] (feature fastest). scale = 1/sqrt(head_dim).
// (Mirrors ops::packed_softmax_attention with equal-length segments.)
// ---------------------------------------------------------------------------
inline void packed_dense_attention(const float* q, const float* k, const float* v, int tokens,
                                   std::int32_t segment_length, float scale, float* out,
                                   int threads = 0) {
    const int D = geom::head_dim;
    const int H = geom::heads; // query heads == kv heads (group = 1 for the ViT).
    const int HD = H * D;
    // q/k/v/out are [head_dim, heads, tokens] (feature fastest; token stride = H*D).
    // Per (query token), for each head: dense softmax over the token's frame.
    const int th   = clamp_threads(threads, tokens);
    const int row0 = tokens / th;
    auto worker    = [&](int a, int b) {
        // Each worker keeps its own per-query `scores` scratch so threads do not share state.
        std::vector<float> scores(static_cast<std::size_t>(segment_length));
        for (int tq = a; tq < b; ++tq) {
            const int frame   = tq / segment_length;
            const int seg0    = frame * segment_length;
            for (int h = 0; h < H; ++h) {
                float max_score = -std::numeric_limits<float>::infinity();
                for (int j = 0; j < segment_length; ++j) {
                    const int tj = seg0 + j;
                    float dot = 0.0F;
                    for (int d = 0; d < D; ++d) {
                        dot += q[static_cast<std::size_t>(tq) * HD + h * D + d] *
                               k[static_cast<std::size_t>(tj) * HD + h * D + d];
                    }
                    const float s = dot * scale;
                    scores[static_cast<std::size_t>(j)] = s;
                    if (s > max_score) { max_score = s; }
                }
                float denom = 0.0F;
                for (int j = 0; j < segment_length; ++j) {
                    float& s = scores[static_cast<std::size_t>(j)];
                    if (s == -std::numeric_limits<float>::infinity()) { continue; }
                    s = std::exp(s - max_score);
                    denom += s;
                }
                for (int d = 0; d < D; ++d) {
                    float num = 0.0F;
                    for (int j = 0; j < segment_length; ++j) {
                        const float w = scores[static_cast<std::size_t>(j)];
                        if (w == -std::numeric_limits<float>::infinity()) { continue; }
                        num += w * v[static_cast<std::size_t>(seg0 + j) * HD + h * D + d];
                    }
                    out[static_cast<std::size_t>(tq) * HD + h * D + d] =
                        denom > 0.0F ? num / denom : 0.0F;
                }
            }
        }
    };
    if (th == 1) { worker(0, tokens); }
    else {
        std::vector<std::thread> pool;
        pool.reserve(th);
        for (int i = 0; i < th; ++i) {
            const int a = row0 * i;
            const int b = (i == th - 1) ? tokens : row0 * (i + 1);
            pool.emplace_back(worker, a, b);
        }
        for (auto& thd : pool) { thd.join(); }
    }
}

// ---------------------------------------------------------------------------
// encode: full ViT forward. Inputs are already host-resident (patches BF16, positions i32,
// per-patch position-table control). `out_visible` is filled with FP32 [out_hidden, tokens];
// the caller rounds to BF16 for the device handoff.
//
//   patches_bf16 : int16 [tokens * patch_dim] (row-major [patches][1536]).
//   position_ids : i32 axis-major [2, patches] (see above).
//   pos_ind/pos_w: per-patch bilinear position-table control [4*patches].
//   segment_length = patches-per-frame; frames = tokens / segment_length.
//   tokens = patches (backbone); visible = tokens / merge_unit (merger handoff).
// ---------------------------------------------------------------------------
inline void encode(const CpuVisionWeights& w, const std::uint16_t* patches_bf16,
                   const std::int32_t* position_ids, const std::int32_t* pos_ind,
                   const float* pos_w, std::int32_t patches, std::int32_t segment_length, int threads,
                   std::vector<float>& out_visible) {
    const int P = patches;
    const int V = P / geom::merge_unit;
    if (P <= 0 || P % geom::merge_unit != 0) {
        throw std::invalid_argument("vision_cpu: patches must be positive and a multiple of 4");
    }
    if (segment_length <= 0 || P % segment_length != 0) {
        throw std::invalid_argument("vision_cpu: segment_length must divide patches");
    }
    const int H  = geom::hidden;
    const int Pd = geom::patch_dim;

    // Patch embedding: x = W @ x_bf16 + bias. Convert patch BF16 [P,Pd] -> fp32.
    std::vector<float> x_bf(static_cast<std::size_t>(P) * Pd);
    for (std::size_t i = 0; i < x_bf.size(); ++i) { x_bf[i] = fp::bf16_to_f32(patches_bf16[i]); }
    std::vector<float> x(static_cast<std::size_t>(H) * P);
    gemm(w.patch_embed.data(), H, Pd, x_bf.data(), P, x.data(), threads);
    add_bias(x.data(), w.patch_bias.data(), H, P);
    vision_pos_embed_add(x.data(), w.position_embedding.data(), pos_ind, pos_w, P, threads);

    const int frames = P / segment_length;
    (void)frames;

    std::vector<float> norm(static_cast<std::size_t>(H) * P);
    std::vector<float> qkv(static_cast<std::size_t>(geom::qkv_out) * P);
    std::vector<float> attended(static_cast<std::size_t>(H) * P);
    std::vector<float> up(static_cast<std::size_t>(geom::intermediate) * P);
    for (std::int32_t layer = 0; layer < geom::layers; ++layer) {
        const auto& lw = w.layers[static_cast<std::size_t>(layer)];
        // Attention.
        layer_norm(x.data(), H, P, lw.norm1_w.data(), lw.norm1_b.data(), geom::norm_epsilon, norm.data(), threads);
        gemm(lw.qkv.data(), geom::qkv_out, H, norm.data(), P, qkv.data(), threads);
        add_bias(qkv.data(), lw.qkv_bias.data(), geom::qkv_out, P);
        // Extract contiguous q/k/v buffers [head_dim, heads, P] (feature fastest) from the packed
        // [qkv_out, P] plane. Each token's q/k/v block is 3*hidden elements in the source, so the
        // per-token stride is qkv_out, not hidden; copying to [hidden, P] buffers keeps the
        // token stride = hidden (H*head_dim) that rope/attention below expect.
        const int qd                = geom::heads * geom::head_dim; // 1152.
        std::vector<float> qbuf(static_cast<std::size_t>(qd) * P);
        std::vector<float> kbuf(static_cast<std::size_t>(qd) * P);
        std::vector<float> vbuf(static_cast<std::size_t>(qd) * P);
        for (std::int32_t p = 0; p < P; ++p) {
            const float* tok = qkv.data() + static_cast<std::size_t>(p) * geom::qkv_out;
            std::memcpy(qbuf.data() + static_cast<std::size_t>(p) * qd, tok,
                        static_cast<std::size_t>(qd) * sizeof(float));
            std::memcpy(kbuf.data() + static_cast<std::size_t>(p) * qd, tok + geom::hidden,
                        static_cast<std::size_t>(qd) * sizeof(float));
            std::memcpy(vbuf.data() + static_cast<std::size_t>(p) * qd, tok + 2 * geom::hidden,
                        static_cast<std::size_t>(qd) * sizeof(float));
        }
        vision_rope(qbuf.data(), position_ids, P, geom::heads, threads);
        vision_rope(kbuf.data(), position_ids, P, geom::heads, threads);
        packed_dense_attention(qbuf.data(), kbuf.data(), vbuf.data(), P, segment_length,
                               geom::attention_scale, attended.data(), threads);
        std::vector<float> proj(static_cast<std::size_t>(H) * P);
        gemm(lw.out.data(), H, H, attended.data(), P, proj.data(), threads);
        add_bias(proj.data(), lw.out_bias.data(), H, P);
        residual_add(x.data(), H, P, proj.data(), threads);
        // MLP.
        layer_norm(x.data(), H, P, lw.norm2_w.data(), lw.norm2_b.data(), geom::norm_epsilon, norm.data(), threads);
        gemm(lw.fc1.data(), geom::intermediate, H, norm.data(), P, up.data(), threads);
        add_bias(up.data(), lw.fc1_bias.data(), geom::intermediate, P);
        apply_gelu(up.data(), static_cast<std::size_t>(geom::intermediate) * P, /*exact=*/false, threads);
        std::vector<float> down(static_cast<std::size_t>(H) * P);
        gemm(lw.fc2.data(), H, geom::intermediate, up.data(), P, down.data(), threads);
        add_bias(down.data(), lw.fc2_bias.data(), H, P);
        residual_add(x.data(), H, P, down.data(), threads);
    }

    // Merger: layer-norm, then contiguous [merger_hidden, V] view of x, fc1, gelu-exact, fc2.
    std::vector<float> normalized(static_cast<std::size_t>(H) * P);
    layer_norm(x.data(), H, P, w.merger_norm_w.data(), w.merger_norm_b.data(), geom::norm_epsilon,
               normalized.data(), threads);
    // `merged` is the same memory as `normalized` viewed as [merger_hidden, V].
    // The merger output projection is the variant's `TextConfig::hidden` (5120 for 27B, 2048 for
    // 35B-A3B); derive it from the dequantized `merger_fc2` rows so this core is variant-agnostic.
    const int out_hidden =
        geom::merger_hidden != 0 ? static_cast<int>(w.merger_fc2.size()) / geom::merger_hidden : 0;
    if (out_hidden <= 0 ||
        w.merger_fc2_bias.size() != static_cast<std::size_t>(out_hidden)) {
        throw std::invalid_argument("vision_cpu: merger_fc2/merger_fc2_bias have inconsistent extent");
    }
    std::vector<float> hidden(static_cast<std::size_t>(geom::merger_hidden) * V);
    gemm(w.merger_fc1.data(), geom::merger_hidden, geom::merger_hidden, normalized.data(), V,
         hidden.data(), threads);
    add_bias(hidden.data(), w.merger_fc1_bias.data(), geom::merger_hidden, V);
    apply_gelu(hidden.data(), static_cast<std::size_t>(geom::merger_hidden) * V, /*exact=*/true, threads);
    out_visible.assign(static_cast<std::size_t>(out_hidden) * V, 0.0F);
    gemm(w.merger_fc2.data(), out_hidden, geom::merger_hidden, hidden.data(), V,
         out_visible.data(), threads);
    add_bias(out_visible.data(), w.merger_fc2_bias.data(), out_hidden, V);
    out_visible.resize(static_cast<std::size_t>(out_hidden) * V);
}

// Round the FP32 [out_hidden, V] handoff to BF16 (the device `output` tensor is BF16).
inline std::vector<std::uint16_t> to_bf16(const std::vector<float>& v) {
    std::vector<std::uint16_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) { out[i] = fp::f32_to_bf16(v[i]); }
    return out;
}

} // namespace ninfer::targets::qwen3_6::vision_cpu
