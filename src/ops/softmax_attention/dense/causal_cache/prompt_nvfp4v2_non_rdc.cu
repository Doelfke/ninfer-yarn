// Non-RDC ownership for the warp-specialized NVFP4 GQA-fused causal prompt kernel.
//
// One CTA per (query-token-block, kv_head) decodes the NVFP4 KV prefix once and runs all
// G = Geometry::GroupSize query heads in the consumer warps. Kept in the non-RDC library group:
// like the nvfp4 kernel it uses setmaxnreg to hand registers from producer to consumer warps, so
// relocatable device code would drop that contract.

#include "ops/softmax_attention/dense/causal_cache/prompt_nvfp4v2_non_rdc_launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_nvfp4v2.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, typename CacheView, typename Metadata>
void launch_for(const Tensor& q, const Tensor& positions, float scale, const CacheView& cache,
                Metadata metadata, Tensor& out, cudaStream_t stream) {
    using S = CausalPromptNvfp4v2Schedule<Geometry>;
    static const cudaError_t attr = cudaFuncSetAttribute(
        causal_attention_prompt_nvfp4v2_kernel<Geometry, Metadata>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, S::smem_bytes());
    CUDA_CHECK(attr);

    // Token range mirrors the nvfp4 kernel: the full prompt width q.ne[2], not the masked
    // valid-column count. Batched masked columns are handled by the consumer mma masking (the
    // keys past valid are zero-decoded), so the grid and token bound match nvfp4 exactly and the
    // numeric output is bit-comparable tile-for-tile.
    const int tokens_count = static_cast<int>(q.ne[2]);
    const dim3 grid(static_cast<unsigned>(div_up(tokens_count, kCausalPromptNvfp4v2BrBlock)),
                    static_cast<unsigned>(Geometry::KVHeads), 1U);
    causal_attention_prompt_nvfp4v2_kernel<Geometry, Metadata>
        <<<grid, S::TotalThreads, S::smem_bytes(), stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::uint8_t*>(cache.k_pages.data),
            static_cast<const std::uint8_t*>(cache.v_pages.data),
            static_cast<const std::uint8_t*>(cache.k_scale_pages.data),
            static_cast<const std::uint8_t*>(cache.v_scale_pages.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens_count);
    CUDA_CHECK(cudaGetLastError());
}

template <typename CacheView, typename Metadata>
void dispatch(const Tensor& q, const Tensor& positions, float scale, const CacheView& cache,
              Metadata metadata, Tensor& out, cudaStream_t stream) {
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        launch_for<CausalD256H24Kv4>(q, positions, scale, cache, metadata, out, stream);
        return;
    }
    launch_for<CausalD256H16Kv2>(q, positions, scale, cache, metadata, out, stream);
}

} // namespace

void causal_attention_prompt_nvfp4v2_kernel_launch(const Tensor& q, const Tensor& positions,
                                                   float scale, const PagedKVLayerView& cache,
                                                   Tensor& out, cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    dispatch(q, positions, scale, cache, metadata, out, stream);
}

void causal_attention_prompt_nvfp4v2_batch_kernel_launch(
    const Tensor& q, const Tensor& positions, const Tensor& valid_columns, const Tensor& table_rows,
    float scale, const PagedKVBatchLayerView& cache, Tensor& out, cudaStream_t stream) {
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables    = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        dispatch(q, positions, scale, cache, metadata, out, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
