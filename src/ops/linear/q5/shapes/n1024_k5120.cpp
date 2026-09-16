#include "ops/linear/q5/q5_shapes.h"

namespace ninfer::ops::detail {

Q5Launch select_q5_n1024_k5120(std::int32_t tokens) {
    if (tokens <= 4) return launch_q5_simt_r8_c4;
    if (tokens <= 16) return launch_q5_simt_r8_c8;
    return launch_q5_mma_r64_c128;
}

} // namespace ninfer::ops::detail
