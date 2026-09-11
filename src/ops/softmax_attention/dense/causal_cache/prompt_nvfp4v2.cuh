#pragma once

// Group-16 NVFP4 causal prompt kernel with a GQA-fused, entirely on-chip decode pipeline.
//
// The per-query-head kernel (prompt_nvfp4.cuh) launches one CTA per (q_block, q_head), so the G
// group members (G = Geometry::GroupSize query heads sharing one kv_head) each re-decode the same
// NVFP4 K/V prefix and re-read the same packed bytes. This kernel launches one CTA per
// (q_block, kv_head) and folds all G members into it: producer warps decode the K/V tiles once and
// the G consumer warps (one per group member) run their head's QK / online-softmax / PV. The KV
// decode and packed-byte re-reads therefore happen once per (q_block, kv_head) instead of G times.
//
// Bit-parity: each consumer warp owns exactly the same 16-row m16n8k16 mma window the per-head
// kernel gives a single head, and the QK + PV ldmatrix/mma operand sequence is identical (Bc only
// changes the nt / k loop counts, not the fragment math). The shared per-element operations
// (hadamard, normalize, mma, exp2, __frcp_rn) produce the same values, so every head's result is
// expected bitwise identical to the current nvfp4 prompt output (locked in by the bit-parity test).

#include "ops/common/mbarrier.cuh"
#include "ops/common/math.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCausalPromptNvfp4v2BrBlock = 16; // query tokens per consumer warp (one head)
inline constexpr int kCausalPromptNvfp4v2Bc      = 32; // KV keys per producer tile
inline constexpr int kCausalPromptNvfp4v2RegsProd = 40;  // producer setmaxnreg (multiple of 8)
inline constexpr int kCausalPromptNvfp4v2RegsCons = 208; // consumer setmaxnreg (multiple of 8)

// Register handshake check at sm_120a (65,536 32-bit registers/SM, 1 CTA/SM):
//   ProducerThreads*RegsProd + G*32*RegsCons <= 65,536.
//   H24 (G=6, 12 producer warps): 384*40 + 192*208 = 55,296.
//   H16 (G=8,  8 producer warps): 256*40 + 256*208 = 63,488.
// The producer warp count is chosen per geometry so every consumer warp keeps a full q-head's
// 256-dim accumulator (acc[32][4] = 128 fp32, ~200 live across the pipeline) without the whole CTA
// overshooting the register file. ptxas -v on the target box must confirm the consumer does not
// spill; the plan's risk register (8 tokens/CTA, smaller consumer tiles) is the documented fallback.
template <typename Geometry>
struct CausalPromptNvfp4v2Schedule {
    static constexpr int G               = Geometry::GroupSize;
    static constexpr int ProducerWarps   = (G == 6) ? 12 : 8;
    static constexpr int ConsumerThreads = G * 32;
    static constexpr int ProducerThreads = ProducerWarps * 32;
    static constexpr int TotalThreads    = ProducerThreads + ConsumerThreads;

    [[nodiscard]] static constexpr int smem_bytes() noexcept {
        constexpr int q_bytes = G * kCausalPromptNvfp4v2BrBlock * kCausalPromptHeadDim *
                                static_cast<int>(sizeof(__half));
        constexpr int kv_bytes = 2 * kCausalPromptNvfp4v2Bc * kCausalPromptHeadDim *
                                 static_cast<int>(sizeof(__half));
        constexpr int barrier_bytes = 4 * static_cast<int>(sizeof(std::uint64_t));
        // Output staging reuses the Q region after the tile loop (phased one 16-row block at a
        // time), so it adds no bytes. H24: 81,952 B; H16: 98,336 B (== current nvfp4 peak, no growth).
        return q_bytes + kv_bytes + barrier_bytes;
    }
};

struct CausalPromptNvfp4v2Barriers {
    alignas(8) std::uint64_t k_full;
    alignas(8) std::uint64_t k_empty;
    alignas(8) std::uint64_t v_full;
    alignas(8) std::uint64_t v_empty;
};

__device__ __forceinline__ std::uint32_t causal_prompt_nvfp4v2_pack_f16x2(float lo, float hi) {
    const __half2 packed = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const std::uint32_t*>(&packed);
}

// KV tile decode: byte-for-byte the nvfp4 decode, except the Bc=32 tile is decoded once per
// (q_block, kv_head) and shared by all G consumer warps.
template <typename Geometry>
__device__ __forceinline__ void causal_prompt_nvfp4v2_decode_tile(__half* destination,
                                                                  const std::uint8_t* cache,
                                                                  const std::uint8_t* cache_scale,
                                                                  const std::int32_t* block_table,
                                                                  int kv_head, int tile_k0,
                                                                  int max_query_abs, int producer_tid) {
    constexpr int D             = kCausalPromptHeadDim;
    constexpr int Bc            = kCausalPromptNvfp4v2Bc;
    constexpr int GroupsPerTile = Bc * kKVCacheNvfp4Groups;
    constexpr int ProducerThreads = CausalPromptNvfp4v2Schedule<Geometry>::ProducerThreads;
    const int physical_page = block_table[tile_k0 >> kPagedKVPageShift];
    const int page_offset0  = tile_k0 & kPagedKVPageMask;

#pragma unroll 1
    for (int task = producer_tid; task < GroupsPerTile; task += ProducerThreads) {
        const int key_l    = task / kKVCacheNvfp4Groups;
        const int group    = task - key_l * kKVCacheNvfp4Groups;
        const int d        = group * kKVCacheNvfp4Group;
        const int key      = tile_k0 + key_l;
        __half* target_lo  = destination + key_l * D + causal_prompt_swz(key_l, d);
        __half* target_hi  = destination + key_l * D + causal_prompt_swz(key_l, d + 8);
        if (key <= max_query_abs) {
            const std::int64_t code_offset =
                kv_cache_nvfp4_code_index<Geometry>(physical_page, kv_head, d, page_offset0 + key_l);
            const std::int64_t scale_offset =
                kv_cache_nvfp4_scale_index<Geometry>(physical_page, kv_head, group,
                                                     page_offset0 + key_l);
            const auto represented =
                kv_cache_nvfp4_dequant_f16x16(cache + code_offset, cache_scale[scale_offset]);
            store_vec(target_lo, represented.lo);
            store_vec(target_hi, represented.hi);
        } else {
            store_vec(target_lo, make_int4(0, 0, 0, 0));
            store_vec(target_hi, make_int4(0, 0, 0, 0));
        }
    }
}

template <typename Geometry, typename Metadata>
__global__
__launch_bounds__(CausalPromptNvfp4v2Schedule<Geometry>::TotalThreads, 1)
void causal_attention_prompt_nvfp4v2_kernel(const __nv_bfloat16* __restrict__ q,
                                            const std::uint8_t* __restrict__ cache_k,
                                            const std::uint8_t* __restrict__ cache_v,
                                            const std::uint8_t* __restrict__ cache_k_scale,
                                            const std::uint8_t* __restrict__ cache_v_scale,
                                            Metadata metadata,
                                            const std::int32_t* __restrict__ positions, float scale,
                                            __nv_bfloat16* __restrict__ out, std::int32_t width) {
    using S = CausalPromptNvfp4v2Schedule<Geometry>;
    constexpr int D      = kCausalPromptHeadDim;
    constexpr int G      = S::G;
    constexpr int Br     = kCausalPromptNvfp4v2BrBlock;
    constexpr int Bc     = kCausalPromptNvfp4v2Bc;
    constexpr int QKNt   = Bc / 8;
    constexpr int QKKs   = D / 16;
    constexpr int PVNt   = D / 8;
    constexpr int PVKs   = Bc / 16;
    constexpr float Log2E       = 1.4426950408889634074F;
    constexpr unsigned FullMask = 0xffffffffU;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* q_f16  = reinterpret_cast<__half*>(smem_raw); // G*Br rows, swizzled fp16
    __half* k_f16  = q_f16 + G * Br * D;
    __half* v_f16  = k_f16 + Bc * D;
    auto* barriers = reinterpret_cast<CausalPromptNvfp4v2Barriers*>(v_f16 + Bc * D);

    const int q_block  = static_cast<int>(blockIdx.x);
    const int kv_head  = static_cast<int>(blockIdx.y);
    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int q0       = q_block * Br;
    const int tokens   = metadata.valid_tokens(width);
    if (kv_head >= Geometry::KVHeads || q0 >= width) return;

    if (q0 >= tokens) {
        // Whole query block is past the token count: zero all G heads' padded output rows.
        for (int h = 0; h < G; ++h) {
            causal_prompt_zero_output_rows<Geometry>(out, kv_head * G + h, q0, min(q0 + Br, width),
                                                     tid, S::TotalThreads);
        }
        return;
    }

    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    const int tile_rows             = min(Br, tokens - q0);
    const int max_query_abs         = base_pos + q0 + tile_rows - 1;
    const int key_blocks            = max_query_abs / Bc + 1;

    if (tid == 0) {
        cta_mbarrier_init(&barriers->k_full, 1);
        cta_mbarrier_init(&barriers->k_empty, 1);
        cta_mbarrier_init(&barriers->v_full, 1);
        cta_mbarrier_init(&barriers->v_empty, 1);
        cta_mbarrier_fence_init();
    }

    // Rotate + normalize the query tile for all G members up front, using every warp. Global row
    // r in [0, G*Br) maps to (head_local = r/Br, row_in_block = r%Br); consumer warp h then reads
    // rows h*Br .. h*Br+Br-1 via the same mma layout the nvfp4 kernel's single head uses.
    const int q_rows = G * Br;
#pragma unroll 1
    for (int row = warp; row < q_rows; row += S::TotalThreads / 32) {
        const int head_local = row / Br;
        const int row_in_blk = row - head_local * Br;
        const int my_head    = kv_head * G + head_local;
        float values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d    = lane + 32 * r;
            values[r]      = (q0 + row_in_blk) < tokens
                                 ? __bfloat162float(q[causal_prompt_q_index<Geometry>(my_head, d,
                                                                                        q0 + row_in_blk)])
                                 : 0.0F;
        }
        normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            q_f16[row * D + causal_prompt_swz(row, d)] = __float2half_rn(values[r]);
        }
    }
    __syncthreads();

    // ---- Producer warps: decode this kv_head group's K/V tiles once per CTA. ----
    if (tid < S::ProducerThreads) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" : : "n"(kCausalPromptNvfp4v2RegsProd)
                     : "memory");
        const int producer_tid = tid;
        for (int kb = 0; kb < key_blocks; ++kb) {
            const std::uint32_t empty_phase = 1U ^ static_cast<std::uint32_t>(kb & 1);
            cta_mbarrier_wait(&barriers->k_empty, empty_phase);
            causal_prompt_nvfp4v2_decode_tile<Geometry>(k_f16, cache_k, cache_k_scale, block_table,
                                                        kv_head, kb * Bc, max_query_abs, producer_tid);
            asm volatile("bar.sync 1, %0;" : : "n"(S::ProducerThreads) : "memory");
            if (producer_tid == 0) { cta_mbarrier_arrive(&barriers->k_full); }

            cta_mbarrier_wait(&barriers->v_empty, empty_phase);
            causal_prompt_nvfp4v2_decode_tile<Geometry>(v_f16, cache_v, cache_v_scale, block_table,
                                                        kv_head, kb * Bc, max_query_abs, producer_tid);
            asm volatile("bar.sync 1, %0;" : : "n"(S::ProducerThreads) : "memory");
            if (producer_tid == 0) { cta_mbarrier_arrive(&barriers->v_full); }
        }
        return;
    }

    // ---- Consumer warps: one per group member (query head) over its own 16-token window. ----
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" : : "n"(kCausalPromptNvfp4v2RegsCons)
                 : "memory");
    const int consumer_warp = (tid - S::ProducerThreads) >> 5; // 0..G-1
    const int my_head       = kv_head * G + consumer_warp;

    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;
    const int warp_row0 = consumer_warp * Br; // this head's 16-token row base

    const unsigned q_sbase     = smem_addr(q_f16);
    const unsigned k_sbase     = smem_addr(k_f16);
    const unsigned v_sbase     = smem_addr(v_f16);
    const unsigned q_lane_base = q_sbase + static_cast<unsigned>((warp_row0 + a_rowoff) * (D * 2));
    const unsigned q_as        = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r         = static_cast<unsigned>(a_rin << 4);
    const unsigned k_lane_base =
        k_sbase + static_cast<unsigned>(b_rin * (D * 2)) + (static_cast<unsigned>(lane >> 4) << 12);
    const unsigned k_as        = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r         = static_cast<unsigned>(b_rin << 4);
    const unsigned v_lane_base = v_sbase + static_cast<unsigned>(((lane >> 3) & 1) * (8 * D * 2)) +
                                 static_cast<unsigned>(b_rin * (D * 2));
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[n][i] = 0.0F;
    }
    float m0             = -CUDART_INF_F;
    float m1             = -CUDART_INF_F;
    float l0             = 0.0F;
    float l1             = 0.0F;
    const float scale_l2 = scale * Log2E;
    const bool full_block = (tokens - q0) >= Br;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0                   = kb * Bc;
        const std::uint32_t full_phase = static_cast<std::uint32_t>(kb & 1);
        cta_mbarrier_wait(&barriers->k_full, full_phase);

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt)
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0F;

        unsigned af[2][4];
        unsigned bf[2][QKNt][2];
        ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                    causal_prompt_swz_addr(q_lane_base, 0U, q_as, q_r));
#pragma unroll
        for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
            ldmatrix_x4(bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                        causal_prompt_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096), 0U,
                                               k_as, k_r));
        }
#pragma unroll
        for (int k = 0; k < QKKs; ++k) {
            const int cur = k & 1;
            const int nxt = cur ^ 1;
            if (k + 1 < QKKs) {
                const unsigned ck = static_cast<unsigned>((k + 1) << 5);
                ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                            causal_prompt_swz_addr(q_lane_base, ck, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    ldmatrix_x4(
                        bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0], bf[nxt][nt2 + 1][1],
                        causal_prompt_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096), ck,
                                               k_as, k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                mma_f16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                        af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
            }
        }

        // Release the K tile once all G consumer heads have run QK against it.
        asm volatile("bar.sync 2, %0;" : : "n"(S::ConsumerThreads) : "memory");
        if (consumer_warp == 0 && lane == 0) { cta_mbarrier_arrive(&barriers->k_empty); }

        const int token0        = q0 + gid;
        const int token1        = q0 + gid + 8;
        const int qabs0         = token0 < tokens ? base_pos + token0 : -1;
        const int qabs1         = token1 < tokens ? base_pos + token1 : -1;
        const bool full_score_tile = full_block && (k0 + Bc - 1 <= base_pos + q0);
        float bm0               = -CUDART_INF_F;
        float bm1               = -CUDART_INF_F;
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                score[nt][0]   = token0 < tokens && key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                score[nt][1]   = token0 < tokens && key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                score[nt][2]   = token1 < tokens && key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                score[nt][3]   = token1 < tokens && key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
                bm0            = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1            = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0        = fmaxf(m0, bm0);
        const float nm1        = fmaxf(m1, bm1);
        const float nm0_scaled = nm0 * scale_l2;
        const float nm1_scaled = nm1 * scale_l2;
        const float alpha0 =
            m0 == -CUDART_INF_F ? 0.0F : exp2_approx(__fmaf_rn(m0, scale_l2, -nm0_scaled));
        const float alpha1 =
            m1 == -CUDART_INF_F ? 0.0F : exp2_approx(__fmaf_rn(m1, scale_l2, -nm1_scaled));
        float bl0 = 0.0F;
        float bl1 = 0.0F;
        unsigned p_frag[PVKs][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 = score[nt][0] > -CUDART_INF_F
                                  ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                                  : 0.0F;
            const float p01 = score[nt][1] > -CUDART_INF_F
                                  ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                  : 0.0F;
            const float p10 = score[nt][2] > -CUDART_INF_F
                                  ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                                  : 0.0F;
            const float p11 = score[nt][3] > -CUDART_INF_F
                                  ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                  : 0.0F;
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            const int pk = nt >> 1;
            if ((nt & 1) == 0) {
                p_frag[pk][0] = causal_prompt_nvfp4v2_pack_f16x2(p00, p01);
                p_frag[pk][1] = causal_prompt_nvfp4v2_pack_f16x2(p10, p11);
            } else {
                p_frag[pk][2] = causal_prompt_nvfp4v2_pack_f16x2(p00, p01);
                p_frag[pk][3] = causal_prompt_nvfp4v2_pack_f16x2(p10, p11);
            }
        }

        l0 = __fmaf_rn(l0, alpha0, bl0);
        l1 = __fmaf_rn(l1, alpha1, bl1);
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        cta_mbarrier_wait(&barriers->v_full, full_phase);
        constexpr int PVHalf  = PVNt / 2;
        constexpr int PVLoads = PVKs * PVHalf;
        unsigned vf[2][4];
        ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                      causal_prompt_swz_addr(v_lane_base, 0U, v_as, v_r));
#pragma unroll
        for (int li = 0; li < PVLoads; ++li) {
            const int k      = li / PVHalf;
            const int n2     = (li % PVHalf) * 2;
            const int cur    = li & 1;
            const int nxt    = cur ^ 1;
            if (li + 1 < PVLoads) {
                const int k2        = (li + 1) / PVHalf;
                const int n2b       = ((li + 1) % PVHalf) * 2;
                const unsigned ckv  = static_cast<unsigned>(n2b << 4);
                ldmatrix_x4_t(vf[nxt][0], vf[nxt][1], vf[nxt][2], vf[nxt][3],
                              causal_prompt_swz_addr(v_lane_base + static_cast<unsigned>(k2 * 8192),
                                                     ckv, v_as, v_r));
            }
            mma_f16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[k][0], p_frag[k][1],
                    p_frag[k][2], p_frag[k][3], vf[cur][0], vf[cur][1]);
            mma_f16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3], p_frag[k][0],
                    p_frag[k][1], p_frag[k][2], p_frag[k][3], vf[cur][2], vf[cur][3]);
        }
        // Release the V tile once all G consumer heads have run PV against it.
        asm volatile("bar.sync 2, %0;" : : "n"(S::ConsumerThreads) : "memory");
        if (consumer_warp == 0 && lane == 0) { cta_mbarrier_arrive(&barriers->v_empty); }
    }

    l0   = warp_sum<4>(l0, FullMask);
    l1   = warp_sum<4>(l1, FullMask);
    const float inv_l0 = l0 > 0.0F ? __frcp_rn(l0) : 0.0F;
    const float inv_l1 = l1 > 0.0F ? __frcp_rn(l1) : 0.0F;

    // Staged write-back. The mma accumulator is scattered across the warp's lanes (row0 = gid, col
    // d0 = n*8+2*lid), so each head's 16 rows are gathered into one 16-row fp32 staging block,
    // hadamard-rotated per row, and written as bf16. Staging is fp32 and phased over the G heads
    // (one block at a time) so a single (G*Br)*256*4 block need not coexist with the tiles: it
    // reuses the Q region after the tile loop and keeps the output bitwise identical to nvfp4
    // (no extra rounding).
    float* out_staging = reinterpret_cast<float*>(smem_raw);
    __syncthreads();
#pragma unroll
    for (int phase = 0; phase < G; ++phase) {
        if (consumer_warp == phase) {
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
                const int d    = n * 8 + 2 * lid;
                const int row0 = gid;
                const int row1 = row0 + 8;
                if (row0 < Br) {
                    *reinterpret_cast<float2*>(&out_staging[row0 * D + d]) =
                        make_float2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
                }
                if (row1 < Br) {
                    *reinterpret_cast<float2*>(&out_staging[row1 * D + d]) =
                        make_float2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
                }
            }
        }
        __syncthreads();
        if (consumer_warp == phase) {
            for (int row = 0; row < Br; ++row) {
                const int token = q0 + row;
                if (token < tokens) {
                    float values[8];
#pragma unroll
                    for (int r = 0; r < 8; ++r) values[r] = out_staging[row * D + lane + 32 * r];
                    normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int d = lane + 32 * r;
                        out[causal_prompt_q_index<Geometry>(my_head, d, token)] =
                            __float2bfloat16(values[r]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // Zero rows of this head that fall beyond the token count (padded query rows).
    if (tokens < q0 + Br) {
        causal_prompt_zero_output_rows<Geometry>(out, my_head, q0 + tile_rows, min(q0 + Br, width), lane,
                                                 32);
    }
}

} // namespace ninfer::ops
