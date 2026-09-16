#include "ops/linear/q5/q5_shapes.h"
#include "ops/linear/q5/q5_ksplit_launch.cuh"

namespace ninfer::ops::detail {

Q5Launch select_q5_n5120_k6144(std::int32_t tokens) {
    if (tokens == 1) return launch_q5_simt_r8_c4;
    if (tokens <= 2) return launch_q5_ksplit<6144, 2, 2>;
    if (tokens <= 3) return launch_q5_ksplit<6144, 3, 2>;
    if (tokens <= 4) return launch_q5_ksplit<6144, 4, 2>;
    if (tokens <= 5) return launch_q5_ksplit<6144, 5, 2>;
    if (tokens <= 6) return launch_q5_ksplit<6144, 6, 2>;
    if (tokens <= 24) return launch_q5_simt_r8_c8;
    return launch_q5_mma_r64_c128;
}

} // namespace ninfer::ops::detail
