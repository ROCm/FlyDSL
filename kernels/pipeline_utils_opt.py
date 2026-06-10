"""Shared pipeline utilities for gfx1250 GEMM kernels (optimized tail drain).

Compared to pipeline_utils.py, this version relaxes s_wait_tensorcnt in tail
drain steps: instead of draining ALL outstanding TDM loads when no new load is
issued, it drains one load at a time (graduated drain), letting newer TDM loads
overlap with compute.
"""


def make_tail_plan(num_buffers, pre_loaded, extra):
    """Compute a compile-time tail execution plan for the N-stage pipeline.

    Returns a list of (load_stage, compute_stage, outstanding) tuples, one per
    tail step.  outstanding=-1 means "last step, use compute_tile (no barrier)".

    Args:
        num_buffers: total number of pipeline stages.
        pre_loaded:  stages already loaded and ready to compute (= num_buffers - 1).
        extra:       additional tiles that must be loaded in the tail.
    """
    steps = pre_loaded + extra
    plan = []
    base_outstanding = 2 * (num_buffers - 2)
    last_load_step_idx = extra - 1
    for i in range(steps):
        compute_stage = i if i < pre_loaded else (i - pre_loaded + num_buffers - 1) % num_buffers
        load_stage = (i + num_buffers - 1) % num_buffers if i < extra else None
        is_last = i == steps - 1
        if is_last:
            outstanding = -1
        elif load_stage is not None:
            j = i + 1
            next_compute = j if j < pre_loaded else (j - pre_loaded + num_buffers - 1) % num_buffers
            outstanding = base_outstanding if load_stage != next_compute else 0
        else:
            # Graduated drain: each non-loading step drains one more load than
            # the previous.  TDM loads complete in FIFO order, so draining only
            # the oldest load lets newer loads overlap with this step's compute.
            steps_since = i - last_load_step_idx if last_load_step_idx >= 0 else i + 1
            outstanding = max(0, base_outstanding - 2 * (steps_since - 1))
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
