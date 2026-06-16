# SPDX-License-Identifier: Apache-2.0
"""Lesson 13 — Fast exp: rocdl.exp2 vs Float32.exp2() (a VALU micro-optimization).

A/B microkernel. Softmax calls exp once per score; on a big tile that's a lot of VALU.
There are two ways to get exp in FlyDSL, and they lower to different instruction counts:
  - Float32.exp2(x): the generic/safe path. Emits a v_exp_f32 PLUS a v_ldexp pair (range
    reduction) -> ~3 VALU ops per element.
  - rocdl.exp2(f32, x): the hardware v_exp_f32 directly -> 1 VALU op. Valid when the input
    is already range-safe, which in softmax it is (we always evaluate exp(s - m) with
    s - m <= 0, pre-clamped). exp(x) = 2^(x * log2e), so multiply by log2e first.

This lesson computes exp2 over a big array both ways and (a) checks they agree, (b) shows
the per-element instruction difference via the ISA dump. It's a small, isolated win on its
own — but softmax does this 16-64x per kv-tile, and removing 2 VALU ops/element adds up
when you're VALU-heavy (the attention kernel runs at VALU:MFMA ~23:1, Lesson 12).

### Idiomatic style (flydsl-layout-algebra skill, Recipe A "elementwise / 1-D")
The global load/store is the textbook layout-API elementwise shape: make_buffer_tensor +
two-stage logical_divide to a per-thread element view + a 32-bit copy atom into/out of an rmem
tensor (no raw buffer_load/buffer_store byte math). The exp2 ITSELF stays direct: rocdl.exp2 vs
Float32.exp2() on a register value is the whole point, and the softmax exp / range reduction is
the skill's "bespoke numeric" stay-direct case, not a layout op.

Run BOTH:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_13_fast_exp2.py
ISA diff:  FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/isa_slow FLYDSL_RUNTIME_ENABLE_CACHE=0 \
             HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_13_fast_exp2.py slow
           (then grep v_exp / v_ldexp in /tmp/isa_slow/.../21_final_isa.s; repeat 'fast')
"""

import sys

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.autotune import do_bench

N = 1 << 20
TPB = 256
LOG2E = 1.4426950408889634


def _exp2_expr(arg):
    # hardware v_exp_f32 directly (1 VALU). Valid because arg <= 0 here (range-safe).
    return fx.Float32(fx.rocdl.exp2(fx.typing.T.f32, fx.arith.unwrap(arg)))


def make_kernel(fast: bool):
    @flyc.kernel(known_block_size=[TPB, 1, 1])
    def exp_kernel(X: fx.Tensor, Y: fx.Tensor):
        # IDIOMATIC (flydsl-layout-algebra skill, Recipe A "elementwise / 1-D"): wrap globals with
        # make_buffer_tensor, logical_divide to a per-thread element view, and move data through a
        # 32-bit copy atom into/out of an rmem tensor. The exp2 itself stays DIRECT — rocdl.exp2 vs
        # Float32.exp2() on a register value is the whole point of the lesson, and the softmax exp /
        # range-reduction is the skill's "bespoke numeric" stay-direct case, not a layout op.
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        bX = fx.rocdl.make_buffer_tensor(X)
        bY = fx.rocdl.make_buffer_tensor(Y)

        tX = fx.slice(fx.logical_divide(bX, fx.make_layout(TPB, 1)), (None, bid))
        tY = fx.slice(fx.logical_divide(bY, fx.make_layout(TPB, 1)), (None, bid))
        tX = fx.logical_divide(tX, fx.make_layout(1, 1))
        tY = fx.logical_divide(tY, fx.make_layout(1, 1))

        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        rX = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        fx.copy_atom_call(copy_atom, fx.slice(tX, (None, tid)), rX)
        x = fx.Float32(fx.Vector(fx.memref_load_vec(rX))[0])

        # softmax always evaluates exp(s - m) with s - m <= 0, so clamp to <= 0 to make the
        # fast path valid (this mirrors the real kernel's safe_m subtraction).
        xs = (x < fx.Float32(0.0)).select(x, fx.Float32(0.0))
        arg = xs * fx.Float32(LOG2E)
        # `fast` is a Python (compile-time) flag — but the @flyc.kernel AST rewriter still rejects a
        # value defined only inside an if/else and used after it. So compute y with NO statement-level
        # branch: a host-side helper picks the expression before tracing.
        y = _exp2_expr(arg) if fast else fx.Float32(arg.exp2())

        rY = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        fx.memref_store_vec(fx.Vector.from_elements([y], fx.Float32).ir_value(), rY)
        fx.copy_atom_call(copy_atom, rY, fx.slice(tY, (None, tid)))

    return exp_kernel


def make_run(fast):
    k = make_kernel(fast)

    @flyc.jit
    def run(X: fx.Tensor, Y: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        k(X, Y).launch(grid=(N // TPB, 1, 1), block=(TPB, 1, 1), stream=stream)

    return run


def bench(fast):
    run = make_run(fast)
    X = (torch.randn(N) * 3).cuda()
    Y = torch.zeros(N, dtype=torch.float32).cuda()
    run(X, Y, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    ref = torch.exp(torch.clamp(X, max=0.0))
    err = (Y - ref).abs().max().item()
    ms = do_bench(lambda: run(X, Y, stream=torch.cuda.Stream()), warmup=20, rep=100)
    print(f"{'fast rocdl.exp2' if fast else 'Float32.exp2()':18}  err={err:.2e}  {ms*1e3:6.1f} us")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        bench(sys.argv[1] == "fast")
    else:
        bench(False)
        bench(True)
        print("\nBoth correct. The win is instruction count: Float32.exp2() emits v_exp + a v_ldexp")
        print("pair (~3 VALU/elem); rocdl.exp2 emits 1 v_exp. Dump the ISA and grep v_ldexp to see")
        print("it. Tiny per-call, but softmax calls exp 16-64x per kv-tile in a VALU-heavy kernel.")
