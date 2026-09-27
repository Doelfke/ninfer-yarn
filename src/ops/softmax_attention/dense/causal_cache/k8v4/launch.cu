#include "ops/softmax_attention/dense/causal_cache/k8v4/launch.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/instances.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/template_launch.cuh"
#include "ops/kv_cache/append/launch.h"

namespace ninfer::ops::detail {
namespace {

template <class G, int Tokens, class Input, bool Writable>
void grouped(const K8V4KvOperands& p, K8V4KvCacheView<Writable> cache, Input input,
             K8V4KvPartition partition, K8V4KvPartialView partial, cudaStream_t stream) {
    using Instance    = K8V4KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_k8v4_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked>(
            p, cache, input, partition, partial, stream);
        launch_k8v4_kv_merge<G, typename Instance::Merge, MultiBatch, Masked>(p, cache, partition,
                                                                              partial, stream);
    };
    if (p.batch == 1) {
        if (cache.valid_columns)
            invoke.template operator()<false, true>();
        else
            invoke.template operator()<false, false>();
    } else {
        if (cache.valid_columns)
            invoke.template operator()<true, true>();
        else
            invoke.template operator()<true, false>();
    }
}

template <class G, class Input, bool Writable>
void grouped_instance(const K8V4KvOperands& p, K8V4KvCacheView<Writable> cache, Input input,
                      K8V4KvPartition partition, K8V4KvPartialView partial, cudaStream_t stream) {
    switch (p.width) {
#define NINFER_K8V4_GROUPED(T)                                                                     \
    case T:                                                                                        \
        return grouped<G, T>(p, cache, input, partition, partial, stream)
        NINFER_K8V4_GROUPED(1);
        NINFER_K8V4_GROUPED(2);
        NINFER_K8V4_GROUPED(3);
        NINFER_K8V4_GROUPED(4);
        NINFER_K8V4_GROUPED(5);
        NINFER_K8V4_GROUPED(6);
        NINFER_K8V4_GROUPED(7);
        NINFER_K8V4_GROUPED(8);
#undef NINFER_K8V4_GROUPED
    }
    throw std::logic_error("K8V4 grouped plan exceeds the selected token tile");
}

template <class Input>
void execute_grouped(const Tensor& q, const Tensor& positions, float scale,
                     PagedKVBatchLayerView cache, const Tensor* valid, const Tensor* rows,
                     Input input, const K8V4KvCausalPlan& plan, WorkspaceArena& workspace,
                     Tensor& out, cudaStream_t stream) {
    const auto view    = k8v4_kv_cache_view<Input::writes_cache>(cache, valid, rows);
    auto scope         = workspace.scope();
    const auto partial = k8v4_kv_allocate_partials(workspace, plan.query_heads, plan.width,
                                                   plan.partition.capacity, plan.batch);
    const auto p       = k8v4_kv_operands(q, positions, out, scale, plan.envelope.max_visible_keys);
    if (plan.query_heads == 24)
        grouped_instance<K8V4KvD256H24Kv4>(p, view, input, plan.partition, partial.view(), stream);
    else
        grouped_instance<K8V4KvD256H16Kv2>(p, view, input, plan.partition, partial.view(), stream);
}

template <class G, int Tokens>
void parallel_grouped(const K8V4KvOperands& p, K8V4KvReadView cache, K8V4KvPartition partition,
                      K8V4KvPartialView partial, cudaStream_t stream) {
    using Instance    = K8V4KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_k8v4_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked, false,
                                   K8V4KvCachedInput, true>(p, cache, {}, partition, partial,
                                                            stream);
        launch_k8v4_kv_merge<G, typename Instance::Merge, MultiBatch, Masked>(p, cache, partition,
                                                                              partial, stream);
    };
    if (p.batch == 1) {
        if (cache.valid_columns)
            invoke.template operator()<false, true>();
        else
            invoke.template operator()<false, false>();
    } else {
        if (cache.valid_columns)
            invoke.template operator()<true, true>();
        else
            invoke.template operator()<true, false>();
    }
}

void execute_parallel(const K8V4KvOperands& p, K8V4KvReadView cache, const K8V4KvCausalPlan& plan,
                      WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto partial = k8v4_kv_allocate_partials(workspace, plan.query_heads, plan.width,
                                                   plan.partition.capacity, plan.batch);
    const auto invoke  = [&]<class G>() {
        switch (plan.query_tile) {
#define NINFER_K8V4_PARALLEL(T)                                                                    \
    case T:                                                                                        \
        return parallel_grouped<G, T>(p, cache, plan.partition, partial.view(), stream)
            NINFER_K8V4_PARALLEL(5);
            NINFER_K8V4_PARALLEL(6);
            NINFER_K8V4_PARALLEL(7);
            NINFER_K8V4_PARALLEL(8);
#undef NINFER_K8V4_PARALLEL
        }
        throw std::logic_error("K8V4 parallel plan exceeds its query tiles");
    };
    if (plan.query_heads == 24)
        invoke.template operator()<K8V4KvD256H24Kv4>();
    else
        invoke.template operator()<K8V4KvD256H16Kv2>();
}

void tiled(const K8V4KvOperands& p, K8V4KvReadView cache, cudaStream_t stream) {
    if (p.query_heads == 24)
        launch_k8v4_kv_tiled_mma<K8V4KvD256H24Kv4, K8V4KvTiledInstance>(p, cache, stream);
    else
        launch_k8v4_kv_tiled_mma<K8V4KvD256H16Kv2, K8V4KvTiledInstance>(p, cache, stream);
}

} // namespace

void k8v4_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid, const Tensor& rows,
                              float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, cudaStream_t stream) {
    const auto plan = make_k8v4_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], envelope);
    if (plan.family != K8V4KvFamily::Grouped) {
        kv_cache_append_batch_launch(k, v, positions, valid, rows, cache, stream);
        const auto p    = k8v4_kv_operands(q, positions, out, scale, envelope.max_visible_keys);
        const auto view = k8v4_kv_cache_view<false>(cache, &valid, &rows);
        if (plan.family == K8V4KvFamily::Tiled)
            tiled(p, view, stream);
        else
            execute_parallel(p, view, plan, workspace, stream);
    } else {
        execute_grouped(q, positions, scale, cache, &valid, &rows,
                        K8V4KvAppendInput{static_cast<const __nv_bfloat16*>(k.data),
                                          static_cast<const __nv_bfloat16*>(v.data)},
                        plan, workspace, out, stream);
    }
}

void k8v4_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                              const PagedKVLayerView& cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, cudaStream_t stream) {
    const auto plan = make_k8v4_kv_causal_plan(q.ne[1], q.ne[2], 1, envelope);
    const auto view = single_row_paged_kv_batch_view(cache);
    if (plan.family == K8V4KvFamily::Tiled)
        tiled(k8v4_kv_operands(q, positions, out, scale, envelope.max_visible_keys),
              k8v4_kv_cache_view<false>(view), stream);
    else if (plan.family == K8V4KvFamily::ParallelGrouped)
        execute_parallel(k8v4_kv_operands(q, positions, out, scale, envelope.max_visible_keys),
                         k8v4_kv_cache_view<false>(view), plan, workspace, stream);
    else
        execute_grouped(q, positions, scale, view, nullptr, nullptr, K8V4KvCachedInput{}, plan,
                        workspace, out, stream);
}

} // namespace ninfer::ops::detail
