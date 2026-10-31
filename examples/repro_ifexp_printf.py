#!/usr/bin/env python3
"""
Repro: Does IfExp with arith.select cause both branches to trace?

Simple test: put fx.printf in a helper that also returns a value.
If arith.select is used, both printf ops will be in the IR unconditionally.
"""

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops

def then_value(tid):
    fx.printf("THEN branch: tid={}\n", tid)
    return fx.Int32(10)

def else_value(tid):
    fx.printf("ELSE branch: tid={}\n", tid)
    return fx.Int32(20)


@flyc.kernel
def ifexp_printf_kernel(
    Out: fx.Tensor,
    threshold: fx.Int32,
    block_dim: fx.Constexpr[int],
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    gid = bid * block_dim + tid

    val = then_value(tid) if tid < threshold else else_value(tid)

    rsrc = buffer_ops.create_buffer_resource(Out)
    buffer_ops.buffer_store(val, rsrc, gid)


@flyc.jit
def ifexp_printf_launch(
    Out: fx.Tensor,
    threshold: fx.Int32,
    n: fx.Int32,
    block_dim: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    grid_x = (n + block_dim - 1) // block_dim
    ifexp_printf_kernel(Out, threshold, block_dim).launch(
        grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream
    )


if __name__ == "__main__":
    BLOCK = 1
    size = BLOCK
    threshold = 2

    out = torch.zeros(size, device="cuda", dtype=torch.int32)
    t_out = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=0, divisibility=1)

    print("=== IfExp printf side-effect test ===")
    print(f"BLOCK={BLOCK}, threshold={threshold}")
    print(f"With scf.if:       tid<{threshold} prints THEN only, tid>={threshold} prints ELSE only")
    print(f"With arith.select: ALL threads print BOTH THEN and ELSE")
    print()

    ifexp_printf_launch(t_out, threshold, size, BLOCK)
    torch.cuda.synchronize()

    print(f"\nout = {out.tolist()}")
    print(f"Expected: [10, 10, 20, 20]")
