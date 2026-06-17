# SPDX-License-Identifier: Apache-2.0
"""Lesson 09 — Wide global loads: when they help, and when they DON'T.

A/B microkernel. The ONLY difference between the two paths is the load width:
  - NARROW: each lane does 4 separate 1-element (32-bit) loads.
  - WIDE:   each lane does 1 four-element (128-bit) load (buffer_load_dwordx4).
Same total bytes, same output. We copy a big f32 array (memory-bound) so the effect
is visible, then state the attention result (where it was NEUTRAL) — the real lesson.

Why this is its own tiny lesson and not folded into the attention kernel: the trick is
ONE parameter (`vec_width`). On a memory-bound kernel wide loads cut instruction count
and improve coalescing -> faster. But the attention kernel is NOT memory-bound (it runs
at ~12 GB/s of 5300 -> compute/LDS-bound), so there the same change was NEUTRAL. The
lesson is the METHOD: a generically-good trick is only good if the kernel is bound by the
thing it improves. Always measure achieved GB/s before reaching for wide loads.

Run BOTH:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_09_wide_loads.py
Run one:   HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_09_wide_loads.py narrow
"""

import sys

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.autotune import do_bench

N = 1 << 22  # 4M f32 = 16 MB; big enough to be HBM-bound
TPB = 256
ELEMS_PER_THREAD = 4


def make_copy(wide: bool):
    @flyc.kernel(known_block_size=[TPB, 1, 1])
    def copy_kernel(X: fx.Tensor, Y: fx.Tensor):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        base = (bid * fx.Int32(TPB) + tid) * fx.Int32(ELEMS_PER_THREAD)
        rX = fx.buffer_ops.create_buffer_resource(X)
        rY = fx.buffer_ops.create_buffer_resource(Y)
        if wide:
            # ONE 128-bit load of 4 contiguous f32.
            v = fx.buffer_ops.buffer_load(rX, base, vec_width=4, dtype=fx.Float32)
            fx.buffer_ops.buffer_store(fx.Vector(v).ir_value(), rY, base.ir_value())
        else:
            # FOUR separate 32-bit loads.
            for e in fx.range_constexpr(ELEMS_PER_THREAD):
                x = fx.buffer_ops.buffer_load(rX, base + fx.Int32(e), vec_width=1, dtype=fx.Float32)
                fx.buffer_ops.buffer_store(fx.Float32(x).ir_value(), rY, (base + fx.Int32(e)).ir_value())

    return copy_kernel


def make_run(wide: bool):
    k = make_copy(wide)

    @flyc.jit
    def run(X: fx.Tensor, Y: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        grid = N // (TPB * ELEMS_PER_THREAD)
        k(X, Y).launch(grid=(grid, 1, 1), block=(TPB, 1, 1), stream=stream)

    return run


def bench(wide):
    run = make_run(wide)
    X = torch.randn(N, dtype=torch.float32).cuda()
    Y = torch.zeros(N, dtype=torch.float32).cuda()
    run(X, Y, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    err = (Y - X).abs().max().item()
    ms = do_bench(lambda: run(X, Y, stream=torch.cuda.Stream()), warmup=20, rep=100)
    gbps = (2 * N * 4) / 1e9 / (ms / 1e3)  # read + write
    tag = "WIDE (128-bit)" if wide else "NARROW (4x32-bit)"
    print(f"{tag:18}  err={err:.1e}  {ms*1e3:7.1f} us   {gbps:6.0f} GB/s")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        bench(sys.argv[1] == "wide")
    else:
        bench(False)
        bench(True)
        print("\nMemory-bound copy: WIDE should win (fewer instrs, better coalescing).")
        print("BUT in the attention kernel, wide K/V loads were NEUTRAL — it runs at ~12 GB/s of")
        print("5300 (compute/LDS-bound), so widening the load doesn't touch the bottleneck.")
        print("Lesson: a good trick only helps if the kernel is bound by what it improves.")
