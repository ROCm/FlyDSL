#!/usr/bin/env python3
"""Repro: loop variable i initialized before for, used after loop.

Expected (Python semantics): after `for i in range(5)`, i == 4.
Current behavior: _transform_for_auto excludes i from result_names,
so i stays at its pre-loop value (999).
"""

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops


@flyc.kernel
def kernel_iv_liveout(Out: fx.Tensor, n: fx.Int32):
    i = fx.Int32(999)
    acc = fx.Int32(0)
    for i in range(n):
        acc = acc + fx.Int32(1)
    rsrc = buffer_ops.create_buffer_resource(Out)
    buffer_ops.buffer_store(i, rsrc, fx.Int32(0))
    buffer_ops.buffer_store(acc, rsrc, fx.Int32(1))


@flyc.jit
def run_iv_liveout(Out: fx.Tensor, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    kernel_iv_liveout(Out, n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


if __name__ == "__main__":
    out = torch.zeros(2, device="cuda", dtype=torch.int32)
    t_out = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=0, divisibility=1)

    run_iv_liveout(t_out, fx.Int32(5))
    torch.cuda.synchronize()

    i_val = out[0].item()
    acc_val = out[1].item()
    print(f"i = {i_val}  (expect 4)")
    print(f"acc = {acc_val}  (expect 5)")

    if i_val == 4 and acc_val == 5:
        print("[PASS]")
    else:
        print(f"[FAIL] i should be 4 (last iteration), got {i_val}")
