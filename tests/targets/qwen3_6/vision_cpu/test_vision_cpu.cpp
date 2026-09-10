// Standalone qualification of the CPU vision encoder core
// (src/targets/qwen3_6/impl/runtime/vision_cpu/vision_cpu.h).
//
// Registered as a normal CMake target (see tests/CMakeLists.txt).
// Runs on the Linux GPU machine with the full build; no external dependencies.
//
// Gates:
//  – dequant_row_split_lowbit round-trip (independent packer → decoder, Q4/Q5/Q6/W8)
//  – gemm matches a naive FP64 dot
//  – layer_norm / gelu / pos_embed_add / rope / attention match naive references
//  – encode() end-to-end smoke (full 1.7 GB dequantized weights; gated by
//    VISION_CPU_FULL=1 to keep the default ctest run fast)

#include "targets/qwen3_6/impl/runtime/vision_cpu/vision_cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <numeric>
#include <random>
#include <vector>

namespace vc = ninfer::targets::qwen3_6::vision_cpu;

namespace {

int g_failures = 0;
void fail(const std::string& name) {
    std::cout << "FAIL: " << name << "\n";
    ++g_failures;
}
void check(const std::string& name, bool ok) {
    if (!ok) { fail(name); }
}

std::mt19937 g_rng(123456789U);
double uniform() {
    static std::uniform_real_distribution<double> d(-1.0, 1.0);
    return d(g_rng);
}
int uniform_int(int lo, int hi) {
    static std::uniform_int_distribution<int> d(lo, hi);
    return d(g_rng);
}

double rel_l2(const std::vector<float>& a, const std::vector<double>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d  = static_cast<double>(a[i]) - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    return den > 0.0 ? std::sqrt(num) / std::sqrt(den) : std::sqrt(num);
}

// Independent little-endian fp16 encode (RNE).
std::uint16_t f32_to_f16_bits(float f) {
    const std::uint32_t x       = [](float v) { std::uint32_t u; std::memcpy(&u, &v, 4); return u; }(f);
    const std::uint32_t sign    = (x >> 16) & 0x8000u;
    const std::uint32_t ax      = x & 0x7fffffffu;
    if (ax >= 0x7f800000u) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    int exp                = static_cast<int>((ax >> 23) & 0xffu) - 127 + 15;
    std::uint32_t mant      = ax & 0x007fffffu;
    if (exp <= 0) {
        if (exp < -10) { return static_cast<std::uint16_t>(sign); }
        mant |= 0x00800000u;
        const std::uint32_t shift   = 14 - exp;
        const std::uint32_t shifted = mant >> shift;
        const std::uint32_t rem     = mant & ((1u << shift) - 1u);
        const bool round = rem > (1u << (shift - 1)) || (rem == (1u << (shift - 1)) && (shifted & 1u));
        return static_cast<std::uint16_t>(sign | shifted + (round ? 1u : 0u));
    }
    if (exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    const std::uint32_t shifted = mant >> 13;
    const std::uint32_t rem     = mant & 0x1fffu;
    const bool round = rem > 0x1000u || (rem == 0x1000u && (shifted & 1u));
    std::uint32_t res = shifted + (round ? 1u : 0u);
    if (res == 0x0400u) { res = 0; ++exp; if (exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); } }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | res);
}

struct QSpec { int qtype, bits, group, nib, high_bpr, rmin, rmax; };

// 1) Dequant: independent packer → production decoder, per format.
void test_dequant_round_trip() {
    const std::vector<QSpec> specs = {
        {0, 4, 64, 32, 0,  -8, 7},     // Q4G64
        {1, 5, 64, 32, 8,  -16, 15},   // Q5G64
        {2, 6, 64, 32, 16, -32, 31},   // Q6G64
        {3, 8, 32, 32, 0,  -128, 127}, // W8G32
    };
    const std::int32_t n = 3, k = 70; // k not a multiple of 64 → exercises tail & padding.
    for (const auto& s : specs) {
        const std::int32_t padded_k     = ((k + 127) / 128) * 128;
        const int kg                    = padded_k / s.group;
        const std::size_t code_bytes    = static_cast<std::size_t>(n) * kg * s.nib;
        const std::size_t high_off      = ((code_bytes + 255) / 256) * 256;
        const std::size_t high_bytes    = static_cast<std::size_t>(n) * kg * s.high_bpr;
        const std::size_t scale_off     = high_off + ((high_bytes + 255) / 256) * 256;
        const std::size_t scale_bytes   = static_cast<std::size_t>(n) * kg * 2;
        for (float sv : {1.0F, 0.5F, -0.25F}) {
            const std::uint16_t sb      = f32_to_f16_bits(sv);
            const float sdecode         = vc::fp::f16_to_f32(sb);
            check("dequant scale bits exact (q=" + std::to_string(s.qtype) + ")", sdecode == sv);
            std::vector<std::uint8_t> payload(scale_off + scale_bytes, 0);
            for (int row = 0; row < n; ++row) {
                for (int g = 0; g < kg; ++g) {
                    const std::size_t gi = static_cast<std::size_t>(row) * kg + g;
                    payload[scale_off + gi * 2]     = static_cast<std::uint8_t>(sb & 0xff);
                    payload[scale_off + gi * 2 + 1] = static_cast<std::uint8_t>(sb >> 8);
                    for (int lane = 0; lane < s.group; ++lane) {
                        const int col   = g * s.group + lane;
                        const int seed  = row * 131 + col * 7 + 17;
                        const int span  = s.rmax - s.rmin;
                        const int scode = s.rmin + (seed % span);
                        if (s.bits == 8) {
                            payload[gi * s.nib + lane] =
                                static_cast<std::uint8_t>(static_cast<std::int8_t>(scode));
                            continue;
                        }
                        // Raw code packing (mirrors the reference pack_lowbit_group):
                        //   u = (uint32_t)scode & (span-1)   (two's-complement masked to `bits`).
                        // The production unpack_code inverse is: decode(u) = (u&sign)? u-span : u.
                        const std::uint32_t code_span = 1u << s.bits;
                        const std::uint32_t u         = static_cast<std::uint32_t>(scode) & (code_span - 1u);
                        const std::uint32_t low   = u & 0x0fu;
                        const std::uint32_t upper = u >> 4;
                        std::uint8_t& nb          = payload[gi * s.nib + (lane >> 1)];
                        if ((lane & 1) == 0) { nb |= static_cast<std::uint8_t>(low); }
                        else { nb |= static_cast<std::uint8_t>(low << 4); }
                        if (s.high_bpr != 0) {
                            auto* hb = &payload[high_off + gi * s.high_bpr];
                            if (s.bits == 5) {
                                hb[lane >> 3] |= static_cast<std::uint8_t>(upper & 0x1u) << (lane & 7);
                            } else if (s.bits == 6) {
                                const int bpos = lane * 2;
                                hb[bpos >> 3] |= static_cast<std::uint8_t>((upper & 0x3u) << (bpos & 7));
                            }
                        }
                    }
                }
            }
            const auto got = vc::dequant_row_split_lowbit(payload.data(), n, k, s.qtype);
            bool ok = true;
            for (int row = 0; row < n; ++row) {
                for (int col = 0; col < k; ++col) {
                    const int seed  = row * 131 + col * 7 + 17;
                    const int span  = s.rmax - s.rmin;
                    const int scode = s.rmin + (seed % span);
                    const float expected = static_cast<float>(scode) * sdecode;
                    if (got[static_cast<std::size_t>(row) * k + col] != expected) { ok = false; break; }
                }
                if (!ok) { break; }
            }
            check("dequant exact (qtype=" + std::to_string(s.qtype) + ")", ok);
        }
    }
}

// 2) GEMM vs naive FP64 dot.
void test_gemm() {
    const std::int32_t n = 64, k = 96, t = 19;
    std::vector<float> w(static_cast<std::size_t>(n) * k), x(static_cast<std::size_t>(k) * t),
                       y(static_cast<std::size_t>(n) * t);
    std::vector<double> ref(static_cast<std::size_t>(n) * t);
    for (auto& v : w) { v = static_cast<float>(uniform()); }
    for (auto& v : x) { v = static_cast<float>(uniform()); }
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < t; ++c) {
            double acc = 0.0;
            for (int kk = 0; kk < k; ++kk) {
                acc += static_cast<double>(w[static_cast<std::size_t>(r) * k + kk]) *
                       static_cast<double>(x[static_cast<std::size_t>(c) * k + kk]);
            }
            // token-major / feature-fastest (matches gemm and the reference cpu_linear_gemm_fp64).
            ref[static_cast<std::size_t>(c) * n + r] = acc;
        }
    }
    for (int threads : {1, 4}) {
        vc::gemm(w.data(), n, k, x.data(), t, y.data(), threads);
        const double err = rel_l2(y, ref);
        check("gemm matches FP64 (t=" + std::to_string(threads) + ")", err < 1e-4);
    }
}

// 3a) LayerNorm vs naive.
void test_layer_norm() {
    const int features = 64, tokens = 17;
    const float eps = 1e-6F;
    std::vector<float> x(static_cast<std::size_t>(features) * tokens), w(features), b(features),
                       ln(static_cast<std::size_t>(features) * tokens);
    for (auto& v : x) { v = static_cast<float>(uniform()); }
    for (auto& v : w) { v = static_cast<float>(uniform()); }
    for (auto& v : b) { v = static_cast<float>(uniform()); }
    vc::layer_norm(x.data(), features, tokens, w.data(), b.data(), eps, ln.data());
    double max_err = 0.0;
    for (int p = 0; p < tokens; ++p) {
        double m = 0.0;
        for (int r = 0; r < features; ++r) { m += x[static_cast<std::size_t>(p) * features + r]; }
        m /= features;
        double var = 0.0;
        for (int r = 0; r < features; ++r) {
            const double dd = x[static_cast<std::size_t>(p) * features + r] - m;
            var += dd * dd;
        }
        var /= features;
        const double inv = 1.0 / std::sqrt(var + eps);
        for (int r = 0; r < features; ++r) {
            const double expected = (x[static_cast<std::size_t>(p) * features + r] - m) * inv * w[r] + b[r];
            max_err = std::max(max_err, std::abs(static_cast<double>(ln[static_cast<std::size_t>(p) * features + r]) - expected));
        }
    }
    check("layer_norm matches naive", max_err < 1e-5);
}

// 3b) GELU (tanh & exact) vs naive.
void test_gelu() {
    std::vector<float> buf(1000);
    for (auto& v : buf) { v = static_cast<float>(uniform()); }
    std::vector<float> tanh = buf, ex = buf;
    vc::apply_gelu(tanh.data(), tanh.size(), /*exact=*/false);
    vc::apply_gelu(ex.data(), ex.size(), /*exact=*/true);
    double max_err = 0.0;
    for (std::size_t i = 0; i < buf.size(); ++i) {
        const double xv    = buf[i];
        const double inner = xv + 0.044715 * xv * xv * xv;
        const double et    = 0.5 * xv * (1.0 + std::tanh(std::sqrt(2.0 / 3.141592653589793) * inner));
        const double ee    = 0.5 * xv * (1.0 + std::erf(xv / std::sqrt(2.0)));
        max_err = std::max(max_err, std::abs(tanh[i] - et));
        max_err = std::max(max_err, std::abs(ex[i]   - ee));
    }
    check("gelu matches naive", max_err < 1e-6);
}

// 3c) pos_embed_add vs naive.
void test_pos_embed() {
    const int d = 1152, P = 5, cells = 2304;
    std::vector<float> table(static_cast<std::size_t>(cells) * d), xv(static_cast<std::size_t>(d) * P);
    for (auto& v : table) { v = static_cast<float>(uniform()); }
    std::vector<std::int32_t> idx(static_cast<std::size_t>(P) * 4);
    for (auto& v : idx) { v = uniform_int(0, cells - 1); }
    std::vector<float> pw(static_cast<std::size_t>(P) * 4);
    std::iota(pw.begin(), pw.end(), 0.0F);
    for (auto& v : xv) { v = static_cast<float>(uniform()); }
    const std::vector<float> before = xv;
    vc::vision_pos_embed_add(xv.data(), table.data(), idx.data(), pw.data(), P);
    double max_err = 0.0;
    for (int p = 0; p < P; ++p) {
        for (int h = 0; h < d; ++h) {
            float pos = 0.0F;
            for (int c = 0; c < 4; ++c) {
                const int ii = idx[static_cast<std::size_t>(p) * 4 + c];
                pos += table[static_cast<std::size_t>(ii) * d + h] * pw[static_cast<std::size_t>(p) * 4 + c];
            }
            max_err = std::max(max_err, static_cast<double>(std::abs(xv[static_cast<std::size_t>(p) * d + h] - before[static_cast<std::size_t>(p) * d + h] - pos)));
        }
    }
    check("pos_embed_add matches naive", max_err < 1e-5);
}

// 3d) Vision 2-D RoPE vs naive (exact freq table, split-half NeoX).
void test_rope() {
    const int D = 36, heads = 16, tokens = 7;
    std::vector<float> q(static_cast<std::size_t>(2 * D) * heads * tokens);
    for (auto& v : q) { v = static_cast<float>(uniform()); }
    std::vector<std::int32_t> pos(2 * tokens);
    for (auto& v : pos) { v = uniform_int(-64, 63); }
    const std::vector<float> before = q;
    vc::vision_rope(q.data(), pos.data(), tokens, heads);
    // Use the exact GPU frequency table (mirrors kVisionRopeInvFrequency[18] in rope.cuh).
    constexpr float freq_table[18] = {
        1.000000000e+00F, 5.994842503e-01F, 3.593813664e-01F, 2.154434690e-01F, 1.291549665e-01F,
        7.742636827e-02F, 4.641588834e-02F, 2.782559402e-02F, 1.668100537e-02F, 1.000000000e-02F,
        5.994842503e-03F, 3.593813664e-03F, 2.154434690e-03F, 1.291549665e-03F, 7.742636827e-04F,
        4.641588834e-04F, 2.782559402e-04F, 1.668100537e-04F};
    double max_err = 0.0;
    for (int p = 0; p < tokens; ++p) {
        for (int h = 0; h < heads; ++h) {
            for (int i = 0; i < D; ++i) {
                const int axis = i / 18;
                const float inv = freq_table[i % 18];
                const float ang = static_cast<float>(pos[static_cast<int>(axis) * tokens + p]) * inv;
                const float a   = before[static_cast<std::size_t>(p) * (heads * 2 * D) + h * (2 * D) + i];
                const float b   = before[static_cast<std::size_t>(p) * (heads * 2 * D) + h * (2 * D) + (i + D)];
                const float e0  = a * std::cos(ang) - b * std::sin(ang);
                const float e1  = b * std::cos(ang) + a * std::sin(ang);
                max_err = std::max(max_err, static_cast<double>(std::abs(q[static_cast<std::size_t>(p) * (heads * 2 * D) + h * (2 * D) + i]      - e0)));
                max_err = std::max(max_err, static_cast<double>(std::abs(q[static_cast<std::size_t>(p) * (heads * 2 * D) + h * (2 * D) + (i + D)] - e1)));
            }
        }
    }
    check("vision rope matches naive (exact freq table)", max_err < 1e-4);
}

// 3e) Dense softmax attention vs naive (full token, single segment).
void test_attention() {
    const int D = 72, H = 16, tokens = 24;
    const float scale = 0.11785113019775792F;
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> q(static_cast<std::size_t>(D) * H * tokens),
                       k(static_cast<std::size_t>(D) * H * tokens),
                       v(static_cast<std::size_t>(D) * H * tokens),
                       out(static_cast<std::size_t>(D) * H * tokens);
    for (auto& f : q) { f = dist(rng); }
    for (auto& f : k) { f = dist(rng); }
    for (auto& f : v) { f = dist(rng); }
    vc::packed_dense_attention(q.data(), k.data(), v.data(), tokens, tokens, scale, out.data());
    double max_err = 0.0;
    std::vector<double> scores(tokens);
    for (int tq = 0; tq < tokens; ++tq) {
        for (int h = 0; h < H; ++h) {
            double mx = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < tokens; ++j) {
                double dot = 0.0;
                for (int d = 0; d < D; ++d) {
                    dot += static_cast<double>(q[static_cast<std::size_t>(tq) * (H * D) + h * D + d]) *
                           static_cast<double>(k[static_cast<std::size_t>(j)   * (H * D) + h * D + d]);
                }
                scores[static_cast<std::size_t>(j)] = dot * scale;
                mx = std::max(mx, scores[static_cast<std::size_t>(j)]);
            }
            double den = 0.0;
            for (int j = 0; j < tokens; ++j) {
                scores[static_cast<std::size_t>(j)] = std::exp(scores[static_cast<std::size_t>(j)] - mx);
                den += scores[static_cast<std::size_t>(j)];
            }
            for (int d = 0; d < D; ++d) {
                double num = 0.0;
                for (int j = 0; j < tokens; ++j) {
                    num += scores[static_cast<std::size_t>(j)] *
                           static_cast<double>(v[static_cast<std::size_t>(j) * (H * D) + h * D + d]);
                }
                max_err = std::max(max_err, std::abs(static_cast<double>(out[static_cast<std::size_t>(tq) * (H * D) + h * D + d]) - num / den));
            }
        }
    }
    check("dense attention matches naive", max_err < 1e-4);
}

// 4) Full encode smoke test. Runs only when VISION_CPU_FULL=1 is set.
//    Allocates ~1.7 GB of dequantized weights. Parametrized on the merger output extent
//    (the variant's TextConfig::hidden): 5120 for the 27B and 2048 for the 35B-A3B, so the
//    encode's derivation of out_hidden from merger_fc2 is qualified for both identities.
void encode_smoke(int out_hidden) {
    constexpr std::int32_t P = 64, V = 16, seg = 64;
    vc::CpuVisionWeights w;
    auto fill = [](std::vector<float>& v, std::size_t rows, std::size_t cols) {
        v.assign(rows * cols, 0.01F);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> d(-0.01F, 0.01F);
        for (auto& f : v) { f = d(rng); }
    };
    fill(w.patch_embed, vc::geom::hidden, vc::geom::patch_dim);
    fill(w.patch_bias,  vc::geom::hidden, 1);
    fill(w.position_embedding, vc::geom::position_embeddings, vc::geom::hidden);
    w.layers.resize(vc::geom::layers);
    for (auto& l : w.layers) {
        fill(l.qkv, vc::geom::qkv_out, vc::geom::hidden);
        fill(l.out, vc::geom::hidden, vc::geom::hidden);
        fill(l.fc1, vc::geom::intermediate, vc::geom::hidden);
        fill(l.fc2, vc::geom::hidden, vc::geom::intermediate);
        fill(l.norm1_w, vc::geom::hidden, 1);
        fill(l.norm1_b, vc::geom::hidden, 1);
        fill(l.norm2_w, vc::geom::hidden, 1);
        fill(l.norm2_b, vc::geom::hidden, 1);
        fill(l.qkv_bias, vc::geom::qkv_out, 1);
        fill(l.out_bias, vc::geom::hidden, 1);
        fill(l.fc1_bias, vc::geom::intermediate, 1);
        fill(l.fc2_bias, vc::geom::hidden, 1);
    }
    fill(w.merger_norm_w, vc::geom::hidden, 1);
    fill(w.merger_norm_b, vc::geom::hidden, 1);
    fill(w.merger_fc1,       vc::geom::merger_hidden, vc::geom::merger_hidden);
    fill(w.merger_fc1_bias,  vc::geom::merger_hidden, 1);
    fill(w.merger_fc2,       out_hidden,               vc::geom::merger_hidden);
    fill(w.merger_fc2_bias,  out_hidden,               1);
    check("weight bytes positive", w.bytes() > 0);

    std::mt19937 rng(5);
    std::uniform_real_distribution<float> patch_d(-1.0F, 1.0F);
    std::vector<std::uint16_t> patches(static_cast<std::size_t>(P) * vc::geom::patch_dim);
    for (auto& b : patches) { b = vc::fp::f32_to_bf16(patch_d(rng)); } // finite BF16
    std::vector<std::int32_t> pos(2 * P);
    std::uniform_int_distribution<int> u8(0, 5);
    for (auto& p : pos) { p = u8(rng); }
    std::vector<std::int32_t> idx(static_cast<std::size_t>(P) * 4);
    std::uniform_int_distribution<int> uid(0, vc::geom::position_embeddings - 1);
    for (auto& i : idx) { i = uid(rng); }
    std::vector<float> pw(static_cast<std::size_t>(P) * 4, 0.25F);

    std::vector<float> out;
    vc::encode(w, patches.data(), pos.data(), idx.data(), pw.data(), P, seg, 4, out);
    check("encode output shape " + std::to_string(out_hidden) + " x 16",
          out.size() == static_cast<std::size_t>(out_hidden) * V);
    bool finite = true;
    for (float f : out) { if (!std::isfinite(f)) { finite = false; break; } }
    check("encode output is finite", finite);
}

void test_encode_smoke() {
    encode_smoke(vc::geom::out_hidden /* 5120, 27B */);
    encode_smoke(2048 /* 35B-A3B merger output extent */);
}

} // namespace

int main() {
    test_dequant_round_trip();
    test_gemm();
    test_layer_norm();
    test_gelu();
    test_pos_embed();
    test_rope();
    test_attention();
    if (std::getenv("VISION_CPU_FULL") != nullptr) {
        test_encode_smoke();
    } else {
        std::cout << "skip: full encode (run with VISION_CPU_FULL=1; allocates ~1.7 GB)\n";
    }
    if (g_failures == 0) {
        std::cout << "OK: vision-cpu core standalone\n";
        return 0;
    }
    std::cout << g_failures << " failure(s)\n";
    return 1;
}
