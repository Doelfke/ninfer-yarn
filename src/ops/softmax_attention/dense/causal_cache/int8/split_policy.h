#pragma once

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Capture reserves partials for the largest live row. Producer and merge use
// the same live count; a wider capture never changes a row's work partition.
struct Int8KvPartition {
    static constexpr int kMaxSplits = 256;
    int capacity                    = 1;
    int target                      = 1;
    int key_shift                   = 6; // log2 of the minimum KV keys per split

    __host__ __device__ int active(int visible) const {
        const int count = (visible + (1 << key_shift) - 1) >> key_shift;
        return count < target ? count : target;
    }
};

} // namespace ninfer::ops::detail
