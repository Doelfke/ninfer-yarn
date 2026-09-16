#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {

Q4Launch select_q4_n7168_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_gemv_r1_q8_direct;
    if (tokens <= 16) return tokens % 8 == 0 ? launch_q4_simt_r8_c8 : launch_q4_simt_r8_c4;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
