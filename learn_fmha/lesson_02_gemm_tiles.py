# SPDX-License-Identifier: Apache-2.0
"""Lesson 02 — A real GEMM tile: loop the MFMA over K + first throughput measurement.

Goal: turn the single MFMA (Lesson 01, K=16) into a real GEMM over a longer K by
chaining MFMAs through the accumulator. Compute  C[16,16] = A[16,K] @ B[16,K]^T  for
K = 64 (= 4 MFMA K-steps). Then measure throughput with do_bench and compute TFLOPS.

If you know CK Tile: this is the inner `k_loops` of a `BlockGemm` — each iteration feeds
the next 16-wide K-slice into the same accumulator fragment. The MFMA's `c` operand is
the running accumulator; we pass last step's result back in as this step's `c`.

Key ideas:
  - The accumulator (C fragment) lives in registers across the whole K loop. Start at 0,
    chain: acc = mfma(A_k, B_k, acc) for each K-step.
  - WIDE loads: each lane needs 4 contiguous-K bf16 per K-step. 4 bf16 = 8 bytes, so we
    load with vec_width=4 (one 64-bit load) instead of 4 scalar loads. (Lesson 09 shows
    why wide loads matter — or don't — for a given kernel.)
  - range_constexpr(N) fully UNROLLS a compile-time loop (the K steps are known at compile
    time). Runtime-bounded loops use range(start,stop,step,init=[...]) + yield (Lesson 06).

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_02_gemm_tiles.py
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.autotune import do_bench

M = N = 16
KD = 64
KSTEPS = KD // 16  # 4 MFMA K-steps


@flyc.kernel(known_block_size=[64, 1, 1])
def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    lane = fx.Int32(fx.thread_idx.x)
    k_outer = lane // fx.Int32(16)  # 0..3 : which group of 4 K within a 16-K step
    mn = lane % fx.Int32(16)  # 0..15 : the M (for A) / N (for B) index

    rA = fx.buffer_ops.create_buffer_resource(A)
    rB = fx.buffer_ops.create_buffer_resource(B)
    rC = fx.buffer_ops.create_buffer_resource(C)

    acc = fx.Vector.filled(4, 0.0, fx.Float32).ir_value()
    for ks in fx.range_constexpr(KSTEPS):
        k0 = ks * 16 + 0  # this K-step's base column in the [M,K]/[N,K] matrices
        # lane (k_outer, mn) needs A[mn, k0 + k_outer*4 + e], e=0..3 -> 4 contiguous K -> 1 wide load.
        a_off = mn * fx.Int32(KD) + fx.Int32(k0) + k_outer * fx.Int32(4)
        a_vec = fx.buffer_ops.buffer_load(rA, a_off, vec_width=4, dtype=fx.BFloat16)
        b_off = mn * fx.Int32(KD) + fx.Int32(k0) + k_outer * fx.Int32(4)
        b_vec = fx.buffer_ops.buffer_load(rB, b_off, vec_width=4, dtype=fx.BFloat16)
        a_i16 = fx.Vector(a_vec).bitcast(fx.Int16)
        b_i16 = fx.Vector(b_vec).bitcast(fx.Int16)
        acc = fx.rocdl.mfma_f32_16x16x16bf16_1k_(
            fx.typing.T.vec(4, fx.typing.T.f32), a_i16.ir_value(), b_i16.ir_value(), acc, 0, 0, 0
        )

    accv = fx.Vector(acc)
    for e in fx.range_constexpr(4):
        row = k_outer * fx.Int32(4) + fx.Int32(e)  # C result row = (lane//16)*4 + e
        col = mn  # C result col = lane%16
        fx.buffer_ops.buffer_store(fx.Float32(accv[e]).ir_value(), rC, (row * fx.Int32(N) + col).ir_value())


@flyc.jit
def run_gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    gemm_kernel(A, B, C).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


if __name__ == "__main__":
    torch.manual_seed(0)
    A = torch.randn(M, KD, dtype=torch.bfloat16).cuda()
    B = torch.randn(N, KD, dtype=torch.bfloat16).cuda()
    C = torch.zeros(M, N, dtype=torch.float32).cuda()

    run_gemm(A, B, C, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    ref = A.float() @ B.float().T
    err = (C - ref).abs().max().item()
    print(f"max abs err = {err:.4f}  ->  {'PASS' if err < 5e-1 else 'FAIL'}")

    fn = lambda: run_gemm(A, B, C, stream=torch.cuda.Stream())
    ms = do_bench(fn, warmup=20, rep=100)
    flop = 2.0 * M * N * KD  # one 16x16x64 GEMM
    print(f"do_bench median = {ms*1e3:.2f} us   ({flop/1e6/ms:.2f} MFLOP/s)")
    print("NOTE: this is ENTIRELY launch overhead — a 16x16 tile is one wavefront on one")
    print("of 80 CUs, so the ~120us is kernel-launch latency, not compute. That's the")
    print("lesson: at tiny sizes you measure the launch, not the math. Throughput becomes")
    print("meaningful at real tile/grid sizes (Lesson 08+).")
