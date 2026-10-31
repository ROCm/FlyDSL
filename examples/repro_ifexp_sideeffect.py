#!/usr/bin/env python3
"""
Repro: IfExp with side effects (buffer_store) in both branches.

With arith.select, BOTH stores execute unconditionally.
With scf.if, only one branch executes.

We use a side-effect trick: store to a "flag" tensor inside each branch,
then check which flags got written.
"""

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops


@flyc.kernel
def ifexp_sideeffect_kernel(
    Out: fx.Tensor,
    FlagThen: fx.Tensor,
    FlagElse: fx.Tensor,
    threshold: fx.Int32,
    block_dim: fx.Constexpr[int],
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    gid = bid * block_dim + tid

    rsrc_then = buffer_ops.create_buffer_resource(FlagThen)
    rsrc_else = buffer_ops.create_buffer_resource(FlagElse)
    rsrc_out = buffer_ops.create_buffer_resource(Out)

    # IfExp: each branch stores 1 to its flag tensor as a side effect,
    # then produces a value
    val = (
        [buffer_ops.buffer_store(arith.constant(1, type=fx.T.i32()), rsrc_then, gid), fx.Int32(10)][1]
        if tid < threshold
        else [buffer_ops.buffer_store(arith.constant(1, type=fx.T.i32()), rsrc_else, gid), fx.Int32(20)][1]
    )

    buffer_ops.buffer_store(val, rsrc_out, gid)


@flyc.jit
def ifexp_sideeffect_launch(
    Out: fx.Tensor,
    FlagThen: fx.Tensor,
    FlagElse: fx.Tensor,
    threshold: fx.Int32,
    n: fx.Int32,
    block_dim: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    grid_x = (n + block_dim - 1) // block_dim
    ifexp_sideeffect_kernel(Out, FlagThen, FlagElse, threshold, block_dim).launch(
        grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream
    )


if __name__ == "__main__":
    BLOCK = 8
    size = BLOCK
    threshold = 4  # tid < 4 -> then branch, tid >= 4 -> else branch

    out = torch.zeros(size, device="cuda", dtype=torch.int32)
    flag_then = torch.zeros(size, device="cuda", dtype=torch.int32)
    flag_else = torch.zeros(size, device="cuda", dtype=torch.int32)

    t_out = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=0, divisibility=1)
    t_then = flyc.from_dlpack(flag_then).mark_layout_dynamic(leading_dim=0, divisibility=1)
    t_else = flyc.from_dlpack(flag_else).mark_layout_dynamic(leading_dim=0, divisibility=1)

    ifexp_sideeffect_launch(t_out, t_then, t_else, threshold, size, BLOCK)
    torch.cuda.synchronize()

    print(f"out       = {out.tolist()}")
    print(f"flag_then = {flag_then.tolist()}")
    print(f"flag_else = {flag_else.tolist()}")
    print()

    # With arith.select: BOTH flag_then and flag_else will be ALL 1s
    # (both buffer_stores execute for ALL threads)
    # With scf.if: flag_then[0:4]=1, flag_then[4:8]=0
    #              flag_else[0:4]=0, flag_else[4:8]=1

    both_written = (flag_then.sum() == size and flag_else.sum() == size)
    if both_written:
        print("[BUG] Both branches executed for all threads!")
        print("      arith.select does NOT prevent side effects.")
    else:
        print("[OK] Only one branch executed per thread (scf.if behavior).")
