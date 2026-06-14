"""Shared pipeline utilities for gfx1250 GEMM kernels."""


def make_tail_plan(num_buffers, pre_loaded, extra, full_prefetch=False):
    """Compute a compile-time tail execution plan for the N-stage pipeline.

    Returns a list of (load_stage, compute_stage, outstanding) tuples, one per
    tail step.  outstanding=-1 means "last step, use compute_tile (no barrier)".

    Args:
        num_buffers: total number of pipeline stages.
        pre_loaded:  stages already loaded and ready to compute.
        extra:       additional tiles that must be loaded in the tail.
        full_prefetch: when True, backfill the just-consumed buffer (load_stage
                       = compute_stage) instead of the (N-1)-ahead buffer.
    """
    steps = pre_loaded + extra
    plan = []
    prev_outstanding = 0

    def _compute_stage(idx):
        # Full-prefetch: prologue filled all N buffers (pre_loaded == N) and the
        # tail consumes tiles in order from buffer idx % N, backfilling that same
        # buffer. Legacy: pre_loaded buffers computed in place, then wrap.
        if full_prefetch:
            return idx % num_buffers
        return idx if idx < pre_loaded else (idx - pre_loaded + num_buffers - 1) % num_buffers

    for i in range(steps):
        compute_stage = _compute_stage(i)
        if i < extra:
            load_stage = compute_stage if full_prefetch else (i + num_buffers - 1) % num_buffers
        else:
            load_stage = None
        is_last = i == steps - 1
        if is_last:
            outstanding = -1
        else:
            j = i + 1
            next_compute = _compute_stage(j)
            if load_stage is not None and load_stage != next_compute:
                outstanding = 2 * (num_buffers - 2)
            elif load_stage is None:
                # No new TDM this step: decrement in-flight count by 2 (base
                # units, which map to 1 post-scaling since caller does o//2) as
                # one TDM completes during compute, but don't go below 0.
                outstanding = max(0, prev_outstanding - 2)
            else:
                outstanding = 0
        prev_outstanding = outstanding if outstanding >= 0 else 0
        plan.append((load_stage, compute_stage, outstanding))
    return plan


def tdm_epilogue_fence_threshold_bytes(*, stage_base_off, tail_plan, loop_iters, extra):
    """Return the earliest stage base that must remain untouched before epilogue.

    The TDM-store epilogue reuses the dead LDS prefix starting at byte offset 0.
    Reuse is only safe once all stages that may still be consumed after the last
    full pipeline fence are out of the reuse window.

    Args:
        stage_base_off: Physical byte base for each logical stage.
        tail_plan: Compile-time tail plan from ``make_tail_plan``.
        loop_iters: Number of fully-pipelined main-loop iterations.
        extra: Additional tail loads that happen after the main loop.
    """
    if not tail_plan:
        return 0

    if extra > 0:
        stages_after_last_full_fence = [tail_plan[-1][1]]
    elif loop_iters > 0:
        stages_after_last_full_fence = [compute_stage for _, compute_stage, _ in tail_plan]
    else:
        stages_after_last_full_fence = [tail_plan[-1][1]]

    return min(stage_base_off[stage] for stage in stages_after_last_full_fence)


__all__ = ["make_tail_plan", "tdm_epilogue_fence_threshold_bytes"]
