#include "ops/softmax_attention/dense/causal_cache/fp8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/instances.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/template_launch.cuh"
#include "ops/kv_cache/append/launch.h"

namespace ninfer::ops::detail {
namespace {

template <class G, int Tokens, class Input, bool Writable>
void grouped(const Fp8KvOperands& p, Fp8KvCacheView<Writable> cache, Input input,
             Fp8KvPartition partition, Fp8KvPartialView partial, cudaStream_t stream) {
    using Instance    = Fp8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_fp8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked>(
            p, cache, input, partition, partial, stream);
        launch_fp8_kv_merge<G, typename Instance::Merge, MultiBatch, Masked>(p, cache, partition,
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
void grouped_instance(const Fp8KvOperands& p, Fp8KvCacheView<Writable> cache, Input input,
                      Fp8KvPartition partition, Fp8KvPartialView partial, cudaStream_t stream) {
    switch (p.width) {
#define NINFER_FP8_GROUPED(T)                                                                      \
    case T:                                                                                        \
        return grouped<G, T>(p, cache, input, partition, partial, stream)
        NINFER_FP8_GROUPED(1);
        NINFER_FP8_GROUPED(2);
        NINFER_FP8_GROUPED(3);
        NINFER_FP8_GROUPED(4);
        NINFER_FP8_GROUPED(5);
        NINFER_FP8_GROUPED(6);
        NINFER_FP8_GROUPED(7);
        NINFER_FP8_GROUPED(8);
#undef NINFER_FP8_GROUPED
    }
    throw std::logic_error("FP8 grouped plan exceeds the selected token tile");
}

template <class Input>
void execute_grouped(const Tensor& q, const Tensor& positions, float scale,
                     PagedKVBatchLayerView cache, const Tensor* valid, const Tensor* rows,
                     Input input, const Fp8KvCausalPlan& plan, WorkspaceArena& workspace,
                     Tensor& out, cudaStream_t stream) {
    const auto view    = fp8_kv_cache_view<Input::writes_cache>(cache, valid, rows);
    auto scope         = workspace.scope();
    const auto partial = fp8_kv_allocate_partials(workspace, plan.query_heads, plan.width,
                                                  plan.partition.capacity, plan.batch);
    const auto p       = fp8_kv_operands(q, positions, out, scale, plan.envelope.max_visible_keys);
    if (plan.query_heads == 24)
        grouped_instance<Fp8KvD256H24Kv4>(p, view, input, plan.partition, partial.view(), stream);
    else
        grouped_instance<Fp8KvD256H16Kv2>(p, view, input, plan.partition, partial.view(), stream);
}

template <class G, int Tokens>
void parallel_grouped(const Fp8KvOperands& p, Fp8KvReadView cache, Fp8KvPartition partition,
                      Fp8KvPartialView partial, cudaStream_t stream) {
    using Instance    = Fp8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_fp8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked, false,
                                  Fp8KvCachedInput, true>(p, cache, {}, partition, partial, stream);
        launch_fp8_kv_merge<G, typename Instance::Merge, MultiBatch, Masked>(p, cache, partition,
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

void execute_parallel(const Fp8KvOperands& p, Fp8KvReadView cache, const Fp8KvCausalPlan& plan,
                      WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto partial = fp8_kv_allocate_partials(workspace, plan.query_heads, plan.width,
                                                  plan.partition.capacity, plan.batch);
    if (plan.query_heads == 24)
        parallel_grouped<Fp8KvD256H24Kv4, Fp8KvCausalPlan::kTokenTile>(p, cache, plan.partition,
                                                                       partial.view(), stream);
    else
        parallel_grouped<Fp8KvD256H16Kv2, Fp8KvCausalPlan::kTokenTile>(p, cache, plan.partition,
                                                                       partial.view(), stream);
}

void tiled(const Fp8KvOperands& p, Fp8KvReadView cache, cudaStream_t stream) {
    if (p.query_heads == 24)
        launch_fp8_kv_tiled_mma<Fp8KvD256H24Kv4, Fp8KvTiledInstance>(p, cache, stream);
    else
        launch_fp8_kv_tiled_mma<Fp8KvD256H16Kv2, Fp8KvTiledInstance>(p, cache, stream);
}

} // namespace

void fp8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& positions, const Tensor& valid, const Tensor& rows,
                             float scale, PagedKVBatchLayerView cache,
                             CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                             Tensor& out, cudaStream_t stream) {
    const auto plan = make_fp8_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], envelope);
    if (plan.family != Fp8KvFamily::Grouped) {
        kv_cache_append_batch_launch(k, v, positions, valid, rows, cache, stream);
        const auto p    = fp8_kv_operands(q, positions, out, scale, envelope.max_visible_keys);
        const auto view = fp8_kv_cache_view<false>(cache, &valid, &rows);
        if (plan.family == Fp8KvFamily::Tiled)
            tiled(p, view, stream);
        else
            execute_parallel(p, view, plan, workspace, stream);
    } else {
        execute_grouped(q, positions, scale, cache, &valid, &rows,
                        Fp8KvAppendInput{static_cast<const __nv_bfloat16*>(k.data),
                                         static_cast<const __nv_bfloat16*>(v.data)},
                        plan, workspace, out, stream);
    }
}

void fp8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                             const PagedKVLayerView& cache,
                             CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                             Tensor& out, cudaStream_t stream) {
    const auto plan = make_fp8_kv_causal_plan(q.ne[1], q.ne[2], 1, envelope);
    const auto view = single_row_paged_kv_batch_view(cache);
    if (plan.family == Fp8KvFamily::Tiled)
        tiled(fp8_kv_operands(q, positions, out, scale, envelope.max_visible_keys),
              fp8_kv_cache_view<false>(view), stream);
    else if (plan.family == Fp8KvFamily::ParallelGrouped)
        execute_parallel(fp8_kv_operands(q, positions, out, scale, envelope.max_visible_keys),
                         fp8_kv_cache_view<false>(view), plan, workspace, stream);
    else
        execute_grouped(q, positions, scale, view, nullptr, nullptr, Fp8KvCachedInput{}, plan,
                        workspace, out, stream);
}

} // namespace ninfer::ops::detail
