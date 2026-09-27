// Non-RDC compilation is required for the producer/consumer register redistribution.
#include "ops/softmax_attention/dense/causal_cache/nvfp4/tiled_launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/instances.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/tiled_launch.cuh"

namespace ninfer::ops::detail {
void nvfp4_kv_tiled_attention(const Nvfp4KvOperands& p, Nvfp4KvReadView cache,
                              cudaStream_t stream) {
    if (p.query_heads == 24)
        launch_nvfp4_kv_tiled_mma<Nvfp4KvD256H24Kv4, Nvfp4KvTiledInstance>(p, cache, stream);
    else
        launch_nvfp4_kv_tiled_mma<Nvfp4KvD256H16Kv2, Nvfp4KvTiledInstance>(p, cache, stream);
}
} // namespace ninfer::ops::detail
