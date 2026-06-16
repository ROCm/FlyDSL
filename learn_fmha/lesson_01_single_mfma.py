# SPDX-License-Identifier: Apache-2.0
"""Lesson 01 — A single MFMA instruction (bf16, 16x16x16).

Goal: understand the ONE primitive every GPU matrix kernel is built on — the MFMA
(Matrix Fused Multiply-Add), AMD's equivalent of NVIDIA's MMA / the thing CK-Tile's
`WarpGemm` wraps. We compute a single 16x16 output tile  C = A @ B^T  with one
instruction issued by one wavefront (64 lanes), no LDS, no loops.

If you know CK Tile: this is one `WarpGemmImpl` call. The hard part in any DSL is not
"call the MMA" — it is knowing **which lane holds which matrix element** in the input
and output fragments. That lane<->element mapping is THE thing to internalize; every
later optimization (transpose, cooperative load, register-resident P) is just moving
data so it lands in the lane the MFMA expects. So this lesson's payload is the layout.

MFMA shape `mfma_f32_16x16x16bf16_1k`:  M=16, N=16, K=16, bf16 inputs, f32 accum.
  - Wavefront = 64 lanes.
  - A operand per lane = 4 bf16   (K is split: 64 lanes * 4 = 256? no — see below)
  - B operand per lane = 4 bf16
  - C/D result per lane = 4 f32

CDNA3 (gfx942) 16x16x16 fragment layout (verified empirically below, not guessed):
  - The 64 lanes are viewed as [k_outer=0..3] x [mn=0..15]:  lane = k_outer*16 + mn.
  - A operand: lane (k_outer, m) holds A[m, k_outer*4 + e] for e in 0..3  (4 contiguous K).
  - B operand: lane (k_outer, n) holds B[n, k_outer*4 + e] for e in 0..3.
        (B is read row-major as [N, K], i.e. the instruction does A @ B^T.)
  - C result: lane (i_outer, n) holds C[i_outer*4 + e, n] for e in 0..3 (4 contiguous M rows).
        i.e. result row = (lane//16)*4 + e, result col = lane%16.

We VERIFY this against torch `A @ B.T`. If the layout were wrong the assert fails — that
is the whole point: you discover the layout by checking, never by trusting a comment.

### Idiomatic note: NAME the instruction with an MMA atom
The raw intrinsic `fx.rocdl.mfma_f32_16x16x16bf16_1k_(...)` is the bare hardware opcode.
The layout-algebra style instead declares the instruction ONCE as a typed *MMA atom*
  `fx.make_mma_atom(fx.rocdl.MFMA(M, N, K, bf16))`
and calls it with `fly.mma_atom_call_ssa([acc_type], atom, a, b, c)`. The atom is the
matrix analogue of the copy atom from Lesson 00: it names WHICH instruction; the lane
fragments below say WHICH element each lane feeds it. In a full GEMM you'd let
`fx.make_tiled_mma` + `fx.gemm` build the fragments for you (Lessons 02/03/05), but here
we keep the fragment loads EXPLICIT and hand-written, because the lane<->element mapping
IS the lesson — `fx.gemm` would hide exactly what we are trying to expose.

Run:  HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_01_single_mfma.py
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly

M = N = KD = 16  # one 16x16x16 tile


@flyc.kernel(known_block_size=[64, 1, 1])
def mfma_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    lane = fx.Int32(fx.thread_idx.x)
    k_outer = lane // fx.Int32(16)  # 0..3
    mn = lane % fx.Int32(16)  # 0..15

    rA = fx.buffer_ops.create_buffer_resource(A)
    rB = fx.buffer_ops.create_buffer_resource(B)
    rC = fx.buffer_ops.create_buffer_resource(C)

    # IDIOMATIC: declare the 16x16x16 bf16 MFMA once as a typed atom (vs spelling out the
    # raw `mfma_f32_16x16x16bf16_1k_` opcode at the call site).
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(M, N, KD, fx.BFloat16))
    f32x4 = fx.typing.T.vec(4, fx.typing.T.f32)

    # --- load this lane's A fragment: A[mn, k_outer*4 + e], e=0..3 (row-major A [M,K]) ---
    # This explicit per-lane load IS the pedagogy: lane (k_outer, mn) owns 4 contiguous K.
    a_base = mn * fx.Int32(KD) + k_outer * fx.Int32(4)
    a_vec = fx.Vector.from_elements(
        [fx.buffer_ops.buffer_load(rA, a_base + fx.Int32(e), vec_width=1, dtype=fx.BFloat16) for e in range(4)],
        fx.BFloat16,
    )
    # --- load this lane's B fragment: B[mn, k_outer*4 + e] (row-major B [N,K]) ---
    b_base = mn * fx.Int32(KD) + k_outer * fx.Int32(4)
    b_vec = fx.Vector.from_elements(
        [fx.buffer_ops.buffer_load(rB, b_base + fx.Int32(e), vec_width=1, dtype=fx.BFloat16) for e in range(4)],
        fx.BFloat16,
    )

    c0 = fx.Vector.filled(4, 0.0, fx.Float32)
    # The `1k` bf16 MFMA takes its operands as vector<4xi16> (raw 16-bit lanes), so bitcast.
    a_i16 = fx.Vector(a_vec).bitcast(fx.Int16)
    b_i16 = fx.Vector(b_vec).bitcast(fx.Int16)
    acc = fly.mma_atom_call_ssa([f32x4], mma_atom, a_i16.ir_value(), b_i16.ir_value(), c0.ir_value())
    acc = fx.Vector(acc)

    # --- store: lane (i_outer=k_outer, n=mn) holds C[k_outer*4 + e, mn] ---
    for e in fx.range_constexpr(4):
        row = k_outer * fx.Int32(4) + fx.Int32(e)
        col = mn
        fx.buffer_ops.buffer_store(
            fx.Float32(acc[e]).ir_value(), rC, (row * fx.Int32(N) + col).ir_value()
        )


@flyc.jit
def run_mfma(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    mfma_kernel(A, B, C).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


if __name__ == "__main__":
    torch.manual_seed(0)
    A = torch.randn(M, KD, dtype=torch.bfloat16).cuda()
    B = torch.randn(N, KD, dtype=torch.bfloat16).cuda()
    C = torch.zeros(M, N, dtype=torch.float32).cuda()

    run_mfma(A, B, C, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    ref = (A.float() @ B.float().T)
    err = (C - ref).abs().max().item()
    print(f"max abs err = {err:.4f}  ->  {'PASS' if err < 1e-1 else 'FAIL'}")
    print("(bf16 has ~3 decimal digits, so a small err is the dtype, not a bug.)")
