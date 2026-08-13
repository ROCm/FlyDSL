# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# One 16x16x16 bf16 MFMA computing D = A @ B^T (A: 16x16 row-major MxK,
# B: 16x16 row-major NxK, D: 16x16 f32 MxN), issued by a single wavefront (64
# lanes), three ways that all produce the same result:
#
#   mfma_highlevel — MMA atom + fx.gemm (the fragments hide the VGPR layout)
#   mfma_lowlevel  — fill the operand/accumulator VGPRs by hand per the MFMA ABI
#                    and call rocdl.mfma_f32_16x16x16bf16_1k directly
#   mfma_bridge    — high-level fragments, but pull raw vectors out with .load(),
#                    call the raw rocdl op, and push the result back with .store()
#                    (shows how to move between the two representations)
#
# Companion to docs/flydsl_for_dummies/10_mfma_intrinsics.md.

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import Vector as Vec

M = N = K = 16
WAVE = 64  # one wavefront issues the whole 16x16x16 MFMA


# ---------- HIGH-LEVEL: MMA atom + fx.gemm ----------
@flyc.kernel
def mfma_highlevel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)
    # one 16x16 tile => static fragment shapes
    bA = fx.slice(fx.zipped_divide(A, (M, K)), (None, 0))
    bB = fx.slice(fx.zipped_divide(B, (N, K)), (None, 0))
    bC = fx.slice(fx.zipped_divide(C, (M, N)), (None, 0))

    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

    acopy = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
    ccopy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tcA = fx.make_tiled_copy_A(acopy, tiled_mma).get_slice(tid)
    tcB = fx.make_tiled_copy_B(acopy, tiled_mma).get_slice(tid)
    tcC = fx.make_tiled_copy_C(ccopy, tiled_mma).get_slice(tid)

    frag_A = thr_mma.make_fragment_A(bA)  # vector<4xbf16> per lane
    frag_B = thr_mma.make_fragment_B(bB)  # vector<4xbf16> per lane
    frag_C = thr_mma.make_fragment_C(bC)  # vector<4xf32>  per lane

    fx.copy(acopy, tcA.partition_S(bA), tcA.retile(frag_A))  # load A operand
    fx.copy(acopy, tcB.partition_S(bB), tcB.retile(frag_B))  # load B operand
    frag_C.fill(0)
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)  # -> rocdl.mfma...bf16_1k
    fx.copy(ccopy, tcC.retile(frag_C), tcC.partition_S(bC))  # store C


# ---------- LOW-LEVEL: fill VGPRs per the ABI, call the raw rocdl op ----------
@flyc.kernel
def mfma_lowlevel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    lane = fx.thread_idx.x  # 0..63
    # 16x16x16 wave64 operand ABI:
    #   A operand: lane holds A[m, kg*4 + i], m = lane%16, kg = lane//16, i in 0..3
    #   B operand: lane holds B[n, kg*4 + i], n = lane%16   (B stored NxK)
    #   C accum  : lane holds C[kg*4 + i, n]
    m = lane % fx.Int32(16)
    n = lane % fx.Int32(16)
    kg = lane // fx.Int32(16)
    aptr = fx.get_iter(A)
    bptr = fx.get_iter(B)
    cptr = fx.get_iter(C)

    a_el = []
    b_el = []
    for i in fx.range_constexpr(4):
        k = kg * fx.Int32(4) + fx.Int32(i)
        a_el.append(fx.ptr_load(aptr + (m * fx.Int32(K) + k)))  # A[m, k]
        b_el.append(fx.ptr_load(bptr + (n * fx.Int32(K) + k)))  # B[n, k]
    # rocdl mfma...bf16_1k takes the bf16 operands as i16 lanes
    a = Vec.from_elements(a_el, dtype=fx.BFloat16).bitcast(fx.Int16)
    b = Vec.from_elements(b_el, dtype=fx.BFloat16).bitcast(fx.Int16)
    c0 = Vec.filled(4, 0.0, fx.Float32)

    acc = rocdl.mfma_f32_16x16x16bf16_1k(
        Vec.make_type(4, fx.Float32),
        [a.ir_value(), b.ir_value(), c0.ir_value()],
    )
    acc = Vec(acc, (4,), fx.Float32)

    for i in fx.range_constexpr(4):
        mrow = kg * fx.Int32(4) + fx.Int32(i)
        fx.ptr_store(acc[i], cptr + (mrow * fx.Int32(N) + n))  # C[mrow, n]


# ---------- BRIDGE: high-level fragments, raw rocdl op in the middle ----------
@flyc.kernel
def mfma_bridge(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)
    bA = fx.slice(fx.zipped_divide(A, (M, K)), (None, 0))
    bB = fx.slice(fx.zipped_divide(B, (N, K)), (None, 0))
    bC = fx.slice(fx.zipped_divide(C, (M, N)), (None, 0))

    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
    thr_mma = tiled_mma.thr_slice(tid)
    acopy = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
    ccopy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tcA = fx.make_tiled_copy_A(acopy, tiled_mma).get_slice(tid)
    tcB = fx.make_tiled_copy_B(acopy, tiled_mma).get_slice(tid)
    tcC = fx.make_tiled_copy_C(ccopy, tiled_mma).get_slice(tid)

    frag_A = thr_mma.make_fragment_A(bA)
    frag_B = thr_mma.make_fragment_B(bB)
    frag_C = thr_mma.make_fragment_C(bC)
    fx.copy(acopy, tcA.partition_S(bA), tcA.retile(frag_A))
    fx.copy(acopy, tcB.partition_S(bB), tcB.retile(frag_B))

    # HIGH -> LOW: the fragment's registers are exactly the MFMA operand VGPRs,
    # so frag.load() hands them straight to the raw intrinsic.
    a_vec = frag_A.load()  # vector<4xbf16>
    b_vec = frag_B.load()  # vector<4xbf16>
    acc = rocdl.mfma_f32_16x16x16bf16_1k(
        Vec.make_type(4, fx.Float32),
        [
            a_vec.bitcast(fx.Int16).ir_value(),
            b_vec.bitcast(fx.Int16).ir_value(),
            Vec.filled(4, 0.0, fx.Float32).ir_value(),
        ],
    )
    # LOW -> HIGH: push the raw accumulator back into the fragment.
    frag_C.store(Vec(acc, (4,), fx.Float32))

    fx.copy(ccopy, tcC.retile(frag_C), tcC.partition_S(bC))


@flyc.jit
def run_highlevel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    mfma_highlevel(A, B, C).launch(grid=(1, 1, 1), block=(WAVE, 1, 1), stream=stream)


@flyc.jit
def run_lowlevel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    mfma_lowlevel(A, B, C).launch(grid=(1, 1, 1), block=(WAVE, 1, 1), stream=stream)


@flyc.jit
def run_bridge(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    mfma_bridge(A, B, C).launch(grid=(1, 1, 1), block=(WAVE, 1, 1), stream=stream)


if __name__ == "__main__":
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16).cuda()
    B = torch.randn(N, K, dtype=torch.bfloat16).cuda()  # stored N x K
    ref = A.float() @ B.float().T  # D = A @ B^T

    for name, fn in [("highlevel", run_highlevel), ("lowlevel", run_lowlevel), ("bridge", run_bridge)]:
        C = torch.zeros(M, N, dtype=torch.float32).cuda()
        fn(A, B, C, stream=torch.cuda.Stream())
        torch.cuda.synchronize()
        err = (C - ref).abs().max().item()
        print(f"{name:9} max abs err {err:.3e}  correct {torch.allclose(C, ref, atol=1e-1, rtol=1e-2)}")
