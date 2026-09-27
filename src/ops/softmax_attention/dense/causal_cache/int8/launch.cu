#include "ops/softmax_attention/dense/causal_cache/int8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/int8/instances.h"
#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/template_launch.cuh"
#include "ops/kv_cache/append/launch.h"

namespace ninfer::ops::detail {
namespace {

template <class G, int Tokens, class Input, bool Writable>
void grouped(const Int8KvOperands& p, Int8KvCacheView<Writable> cache, Input input,
             Int8KvPartition partition, Int8KvPartialView partial, cudaStream_t stream) {
    using Instance    = Int8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_int8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked>(
            p, cache, input, partition, partial, stream);
        launch_int8_kv_merge<G, typename Instance::Merge, MultiBatch, Masked>(p, cache, partition,
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
void grouped_instance(const Int8KvOperands& p, Int8KvCacheView<Writable> cache, Input input,
                      Int8KvPartition partition, Int8KvPartialView partial, cudaStream_t stream) {
    switch (p.width) {
#define NINFER_INT8_GROUPED(T)                                                                     \
    case T:                                                                                        \
        return grouped<G, T>(p, cache, input, partition, partial, stream)
        NINFER_INT8_GROUPED(1);
        NINFER_INT8_GROUPED(2);
        NINFER_INT8_GROUPED(3);
        NINFER_INT8_GROUPED(4);
        NINFER_INT8_GROUPED(5);
        NINFER_INT8_GROUPED(6);
        NINFER_INT8_GROUPED(7);
        NINFER_INT8_GROUPED(8);
#undef NINFER_INT8_GROUPED
    }
    throw std::logic_error("INT8 grouped plan exceeds the selected token tile");
}

template <class Input>
void execute_grouped(const Tensor& q, const Tensor& positions, float scale,
                     PagedKVBatchLayerView cache, const Tensor* valid, const Tensor* rows,
                     Input input, const Int8KvCausalPlan& plan, WorkspaceArena& workspace,
                     Tensor& out, cudaStream_t stream) {
    const auto view    = int8_kv_cache_view<Input::writes_cache>(cache, valid, rows);
    auto scope         = workspace.scope();
    const auto partial = int8_kv_allocate_partials(workspace, plan.query_heads, plan.width,
                                                   plan.partition.capacity, plan.batch);
    const auto p       = int8_kv_operands(q, positions, out, scale, plan.envelope.max_visible_keys);
    if (plan.query_heads == 24)
        grouped_instance<Int8KvD256H24Kv4>(p, view, input, plan.partition, partial.view(), stream);
    else
        grouped_instance<Int8KvD256H16Kv2>(p, view, input, plan.partition, partial.view(), stream);
}

template <class G, int Tokens>
void parallel_grouped(const Int8KvOperands& p, Int8KvReadView cache, Int8KvPartition partition,
                      Int8KvPartialView partial, cudaStream_t stream) {
    using Instance    = Int8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_int8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked, false,
                                   Int8KvCachedInput, true>(p, cache, {}, partition, partial,
                                                            stream);
        launch_int8_kv_merge<G, typename Instance::Merge, MultiBatch, Masked>(p, cache, partition,
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

void execute_parallel(const Int8KvOperands& p, Int8KvReadView cache, const Int8KvCausalPlan& plan,
                      WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto partial = int8_kv_allocate_partials(workspace, plan.query_heads, plan.width,
                                                   plan.partition.capacity, plan.batch);
    if (plan.query_heads == 24)
        parallel_grouped<Int8KvD256H24Kv4, Int8KvCausalPlan::kTokenTile>(p, cache, plan.partition,
                                                                         partial.view(), stream);
    else
        parallel_grouped<Int8KvD256H16Kv2, Int8KvCausalPlan::kTokenTile>(p, cache, plan.partition,
                                                                         partial.view(), stream);
}

void tiled(const Int8KvOperands& p, Int8KvReadView cache, cudaStream_t stream) {
    if (p.query_heads == 24)
        launch_int8_kv_tiled_mma<Int8KvD256H24Kv4, Int8KvTiledInstance>(p, cache, stream);
    else
        launch_int8_kv_tiled_mma<Int8KvD256H16Kv2, Int8KvTiledInstance>(p, cache, stream);
}

} // namespace

void int8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid, const Tensor& rows,
                              float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, cudaStream_t stream) {
    const auto plan = make_int8_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], envelope);
    if (plan.family != Int8KvFamily::Grouped) {
        kv_cache_append_batch_launch(k, v, positions, valid, rows, cache, stream);
        const auto p    = int8_kv_operands(q, positions, out, scale, envelope.max_visible_keys);
        const auto view = int8_kv_cache_view<false>(cache, &valid, &rows);
        if (plan.family == Int8KvFamily::Tiled)
            tiled(p, view, stream);
        else
            execute_parallel(p, view, plan, workspace, stream);
    } else {
        execute_grouped(q, positions, scale, cache, &valid, &rows,
                        Int8KvAppendInput{static_cast<const __nv_bfloat16*>(k.data),
                                          static_cast<const __nv_bfloat16*>(v.data)},
                        plan, workspace, out, stream);
    }
}

void int8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                              const PagedKVLayerView& cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, cudaStream_t stream) {
    const auto plan = make_int8_kv_causal_plan(q.ne[1], q.ne[2], 1, envelope);
    const auto view = single_row_paged_kv_batch_view(cache);
    if (plan.family == Int8KvFamily::Tiled)
        tiled(int8_kv_operands(q, positions, out, scale, envelope.max_visible_keys),
              int8_kv_cache_view<false>(view), stream);
    else if (plan.family == Int8KvFamily::ParallelGrouped)
        execute_parallel(int8_kv_operands(q, positions, out, scale, envelope.max_visible_keys),
                         int8_kv_cache_view<false>(view), plan, workspace, stream);
    else
        execute_grouped(q, positions, scale, view, nullptr, nullptr, Int8KvCachedInput{}, plan,
                        workspace, out, stream);
}

} // namespace ninfer::ops::detail
