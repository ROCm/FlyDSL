# SPDX-License-Identifier: Apache-2.0
"""Lesson 09 — Wide global loads: when they help, and when they DON'T.

A/B microkernel. The ONLY difference between the two paths is the COPY-ATOM WIDTH:
  - NARROW: a 32-bit copy atom per element -> 4 separate 1-element loads/lane.
  - WIDE:   a 128-bit copy atom over a 4-element view -> 1 dwordx4 load/lane.
Same total bytes, same output. We copy a big f32 array (memory-bound) so the effect
is visible, then state the attention result (where it was NEUTRAL) — the real lesson.

### Idiomatic style (flydsl-layout-algebra skill, Recipe A "elementwise / 1-D")
"Wide loads" is the textbook layout-API case: the load width is just the copy-atom width over a
vectorized per-thread layout view. We wrap globals with `make_buffer_tensor`, two-stage
`logical_divide` down to a per-thread tile, then `copy_atom_call` through a `BufferCopy128b`
(WIDE) or `BufferCopy32b` (NARROW) atom. There is no raw `buffer_load(vec_width=...)` byte math
left — picking the atom + view width IS the wide-load knob. (Max atom width is 128 bits, so
ELEMS_PER_THREAD=4 f32 is exactly one 128-bit copy.)

Why this is its own tiny lesson and not folded into the attention kernel: the trick is
ONE choice (the copy-atom width). On a memory-bound kernel wide loads cut instruction count
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
        # IDIOMATIC (flydsl-layout-algebra skill, Recipe A "elementwise / 1-D"): "wide loads" ARE just
        # a wider copy atom over a vectorized per-thread layout view. We wrap the globals with
        # make_buffer_tensor, two-stage logical_divide down to a per-thread tile, then copy through a
        # BufferCopy{128b,32b} atom. NARROW = 32b atom over a 1-element view (4 loads/thread); WIDE =
        # 128b atom over a 4-element view (1 dwordx4 load/thread). The copy width is the whole lesson:
        # express it by picking the atom + view width, not by hand-rolling buffer_load vec_width math.
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        bX = fx.rocdl.make_buffer_tensor(X)
        bY = fx.rocdl.make_buffer_tensor(Y)

        block_elems = TPB * ELEMS_PER_THREAD
        # stage 1: split the flat array into per-block tiles, select this block.
        tX = fx.slice(fx.logical_divide(bX, fx.make_layout(block_elems, 1)), (None, bid))
        tY = fx.slice(fx.logical_divide(bY, fx.make_layout(block_elems, 1)), (None, bid))

        # stage 2: each thread owns ELEMS_PER_THREAD contiguous f32; select this thread's view.
        tX = fx.slice(fx.logical_divide(tX, fx.make_layout(ELEMS_PER_THREAD, 1)), (None, tid))
        tY = fx.slice(fx.logical_divide(tY, fx.make_layout(ELEMS_PER_THREAD, 1)), (None, tid))

        if wide:
            # WIDE: ONE 128-bit copy atom moves all ELEMS_PER_THREAD (4 f32 = 128 bits) at once.
            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
            reg = fx.make_rmem_tensor(fx.make_layout(ELEMS_PER_THREAD, 1), fx.Float32)
            fx.copy_atom_call(copy_atom, tX, reg)
            fx.copy_atom_call(copy_atom, reg, tY)
        else:
            # NARROW: a 32-bit atom per element -> ELEMS_PER_THREAD separate loads/stores. Split the
            # per-thread tile into 1-element views; element e is fx.slice(.., (None, e)).
            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            eX = fx.logical_divide(tX, fx.make_layout(1, 1))
            eY = fx.logical_divide(tY, fx.make_layout(1, 1))
            for e in fx.range_constexpr(ELEMS_PER_THREAD):
                reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
                fx.copy_atom_call(copy_atom, fx.slice(eX, (None, e)), reg)
                fx.copy_atom_call(copy_atom, reg, fx.slice(eY, (None, e)))

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
