#include "core/weight.h"
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q5/q5_launch.h"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kRowsPerBlock = 8;
constexpr int kStages       = 2;

template <int ColsPerTile>
void launch_simt(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows     = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t out_ld   = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t padded_k = w.padded_shape[1];
    const auto* xp              = static_cast<const __nv_bfloat16*>(x.data);
    const bool aligned_x = (k % 8) == 0 && (reinterpret_cast<std::uintptr_t>(xp) & 0xfu) == 0;
    const std::int32_t full_slabs = aligned_x ? k / 1024 : 0;
    constexpr int kThreads        = kRowsPerBlock * 32;
    const dim3 grid(static_cast<unsigned>(div_up(rows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, ColsPerTile)), 1u);
    q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, ColsPerTile, kRowsPerBlock, kStages>
        <<<grid, kThreads, 0, stream>>>(xp, static_cast<const std::uint8_t*>(w.qdata),
                                        static_cast<const std::uint8_t*>(w.qhigh),
                                        static_cast<const std::uint8_t*>(w.scales),
                                        static_cast<__nv_bfloat16*>(out.data), nullptr, rows,
                                        out_ld, k, cols, padded_k, full_slabs);
    CUDA_CHECK(cudaGetLastError());
}

template <int ColsPerTile>
void launch_simt_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    for_each_token_slice(x.ne[1], ColsPerTile, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor out_slice     = out.slice(1, offset, count);
        launch_simt<ColsPerTile>(x_slice, w, out_slice, stream);
    });
}

} // namespace

void launch_q5_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_simt_route<4>(x, w, out, stream);
}

void launch_q5_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_simt_route<8>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
