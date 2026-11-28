# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Run:
#   FLYDSL_COMPILE_BACKEND=cuda FLYDSL_RUNTIME_KIND=cuda \
#     python3 examples/cuda/01-MmaSync.py

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

# One mma.sync.aligned instruction tile: M=16, N=8, K=16.
INST_M = 16
INST_N = 8
INST_K = 16


@flyc.kernel
def gemm_kernel(
    A: fx.Tensor,  # (M, K) row-major
    B: fx.Tensor,  # (N, K) row-major  (so C = A @ B^T)
    C: fx.Tensor,  # (M, N) row-major
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    bA = fx.zipped_divide(A, (INST_M, INST_K))
    bB = fx.zipped_divide(B, (INST_N, INST_K))
    bC = fx.zipped_divide(C, (INST_M, INST_N))

    bA = fx.slice(bA, (None, bid))
    bB = fx.slice(bB, (None, bid))
    bC = fx.slice(bC, (None, bid))

    mma_atom = fx.make_mma_atom(fx.nvvm.MmaSync(16, 8, 16, fx.Float16))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

    copy_atom_f16 = fx.make_copy_atom(fx.UniversalCopy16b(), fx.Float16)
    copy_atom_f32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    tiled_copy_A = fx.make_tiled_copy_A(copy_atom_f16, tiled_mma)
    tiled_copy_B = fx.make_tiled_copy_B(copy_atom_f16, tiled_mma)
    tiled_copy_C = fx.make_tiled_copy_C(copy_atom_f32, tiled_mma)

    thr_copy_A = tiled_copy_A.get_slice(tid)
    thr_copy_B = tiled_copy_B.get_slice(tid)
    thr_copy_C = tiled_copy_C.get_slice(tid)

    copy_src_A = thr_copy_A.partition_S(bA)
    copy_src_B = thr_copy_B.partition_S(bB)
    copy_dst_C = thr_copy_C.partition_S(bC)

    frag_A = thr_mma.make_fragment_A(bA)
    frag_B = thr_mma.make_fragment_B(bB)
    frag_C = thr_mma.make_fragment_C(bC)

    copy_frag_A = thr_copy_A.retile(frag_A)
    copy_frag_B = thr_copy_B.retile(frag_B)
    copy_frag_C = thr_copy_C.retile(frag_C)

    fx.copy(copy_atom_f16, copy_src_A, copy_frag_A, pred=None)
    fx.copy(copy_atom_f16, copy_src_B, copy_frag_B, pred=None)

    frag_C.fill(0)
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

    fx.copy(copy_atom_f32, copy_frag_C, copy_dst_C, pred=None)


@flyc.jit
def nvvm_gemm(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    gemm_kernel(A, B, C).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


M, N, K = INST_M, INST_N, INST_K
A = torch.randn(M, K, dtype=torch.float16).cuda()
B = torch.randn(N, K, dtype=torch.float16).cuda()
C = torch.zeros(M, N, dtype=torch.float32).cuda()

nvvm_gemm(A, B, C, stream=torch.cuda.Stream())
torch.cuda.synchronize()

expected = A.float() @ B.float().T
is_correct = torch.allclose(C, expected, atol=1e-2, rtol=1e-2)
print("Result correct:", is_correct)
