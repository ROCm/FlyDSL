#!/usr/bin/env python3
"""
Minimal reproducer: variable assigned inside `if tid < threshold`
and used after the if block.

On current main, `scf_if_dispatch` has NO result/yield support,
so the if body is wrapped in a closure. The closure mutates `val`
in its local scope, but the outer scope never sees the update
(Python closure rebinding + MLIR SSA dominance).

Expected: either
  (a) a compile error ("val defined inside branch used outside"), or
  (b) correct scf.if with yield so `val` is properly live-out.

Actual (main): silently uses the PRE-if value of `val` after the if,
producing wrong results.
"""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


def test_if_liveout_silent_bug():
    """val is assigned inside dynamic if, used after — should yield but doesn't on main."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required")

    BLOCK = 64

    @flyc.kernel
    def bugKernel(
        Out: fx.Tensor,
        threshold: fx.Int32,
        block_dim: fx.Constexpr[int],
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        gid = bid * block_dim + tid

        from flydsl.expr import buffer_ops

        # val exists BEFORE the if — this is a live-out candidate
        val = fx.Float32(1.0)

        if tid < threshold:
            val = fx.Float32(2.0)
        # After the if: on main (no yield), `val` is still 1.0 for ALL threads.
        # Correct behavior: val=2.0 for tid<threshold, val=1.0 for tid>=threshold.

        rsrc = buffer_ops.create_buffer_resource(Out)
        buffer_ops.buffer_store(val.ir_value(), rsrc, gid)

    @flyc.jit
    def bugLaunch(
        Out: fx.Tensor,
        threshold: fx.Int32,
        n: fx.Int32,
        block_dim: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_x = (n + block_dim - 1) // block_dim
        bugKernel(Out, threshold, block_dim).launch(
            grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream
        )

    size = BLOCK
    threshold = BLOCK // 2  # 32

    out = torch.zeros(size, device="cuda", dtype=torch.float32)
    t_out = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=0, divisibility=4)

    bugLaunch(t_out, threshold, size, BLOCK)
    torch.cuda.synchronize()

    # Expected: [2,2,...,2, 1,1,...,1]  (first 32 = 2.0, last 32 = 1.0)
    ref = torch.ones(size, device="cuda", dtype=torch.float32)
    ref[:threshold] = 2.0

    print(f"out = {out}")
    print(f"ref = {ref}")

    try:
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        print("[PASS] val was correctly live-out from scf.if")
    except AssertionError as e:
        # On main this WILL fail: all elements are 1.0 (pre-if value)
        print(f"[EXPECTED FAIL on main] val was NOT live-out: {e}")
        raise


if __name__ == "__main__":
    test_if_liveout_silent_bug()
