# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x tiled WMMA demo (sibling of ``examples/03-tiledMma.py``).

Upstream ``03`` stays gfx9*-only (CDNA MFMA). This file teaches the matching
RDNA4 path with ``fx.rocdl.WMMA(16, 16, 16, …)`` for the three AB families the
gfx120x atom ships:

* bfloat16 → float32
* signed iu8 (Int8) → Int32
* FP8 e4m3fn (Float8E4M3FN) → float32

Allowlist: ``tests/arch_compat.EXAMPLE_ARCHITECTURES["07-tiledMma_gfx120x.py"]``
= ``("gfx120*",)``. Other arches skip this example; they keep using ``03``.
"""

import sys

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch

_arch = (get_rocm_arch() or "").lower().split(":")[0]
if not _arch.startswith("gfx120"):
    print(f"SKIP examples/07-tiledMma_gfx120x.py: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)

block_m = block_n = block_k = 16


def _run_bf16() -> bool:
    """One WMMA tile: 16x16x16 bf16×bf16 → f32."""

    @flyc.kernel
    def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor) -> None:
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        A = fx.rocdl.make_buffer_tensor(A)
        B = fx.rocdl.make_buffer_tensor(B)
        C = fx.rocdl.make_buffer_tensor(C)
        bA = fx.make_view(fx.get_iter(A), fx.make_layout((block_m, block_k), (block_k, 1)))
        bB = fx.make_view(fx.get_iter(B), fx.make_layout((block_n, block_k), (block_k, 1)))
        bC = fx.make_view(fx.get_iter(C), fx.make_layout((block_m, block_n), (block_n, 1)))
        _ = bid
        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 16, fx.BFloat16, fx.Float32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        thr_mma = tiled_mma.thr_slice(tid)
        frag_A = thr_mma.make_fragment_A(bA)
        frag_B = thr_mma.make_fragment_B(bB)
        frag_C = thr_mma.make_fragment_C(bC)
        copy_a = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.BFloat16.width), fx.BFloat16)
        copy_b = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.BFloat16.width), fx.BFloat16)
        copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.Float32.width), fx.Float32)
        thr_copy_A = fx.make_tiled_copy_A(copy_a, tiled_mma).get_slice(tid)
        thr_copy_B = fx.make_tiled_copy_B(copy_b, tiled_mma).get_slice(tid)
        thr_copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)
        fx.copy(copy_a, thr_copy_A.partition_S(bA), thr_copy_A.retile(frag_A))
        fx.copy(copy_b, thr_copy_B.partition_S(bB), thr_copy_B.retile(frag_B))
        frag_C.fill(0)
        fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
        fx.copy(copy_c, thr_copy_C.retile(frag_C), thr_copy_C.partition_S(bC))

    @flyc.jit
    def tiledMma(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)) -> None:
        gemm_kernel(A, B, C).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    M = N = K = block_m
    A = torch.randn(M, K, dtype=torch.bfloat16).cuda()
    B = torch.randn(N, K, dtype=torch.bfloat16).cuda()
    C = torch.zeros(M, N, dtype=torch.float32).cuda()
    tiledMma(A, B, C, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    expected = A.float() @ B.float().T
    ok = torch.allclose(C, expected, atol=2e-2, rtol=2e-2)
    print(f"arch={_arch} atom=WMMA bf16→f32 correct:", ok)
    if not ok:
        print("Max diff:", (C - expected).abs().max().item())
    return ok


def _run_iu8() -> bool:
    """One WMMA tile: 16x16x16 signed Int8×Int8 → Int32 (iu8 atom)."""

    @flyc.kernel
    def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor) -> None:
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        A = fx.rocdl.make_buffer_tensor(A)
        B = fx.rocdl.make_buffer_tensor(B)
        C = fx.rocdl.make_buffer_tensor(C)
        bA = fx.make_view(fx.get_iter(A), fx.make_layout((block_m, block_k), (block_k, 1)))
        bB = fx.make_view(fx.get_iter(B), fx.make_layout((block_n, block_k), (block_k, 1)))
        bC = fx.make_view(fx.get_iter(C), fx.make_layout((block_m, block_n), (block_n, 1)))
        _ = bid
        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 16, fx.Int8, fx.Int32, sign_a=True, sign_b=True, clamp=False))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        thr_mma = tiled_mma.thr_slice(tid)
        frag_A = thr_mma.make_fragment_A(bA)
        frag_B = thr_mma.make_fragment_B(bB)
        frag_C = thr_mma.make_fragment_C(bC)
        copy_a = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.Int8.width), fx.Int8)
        copy_b = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.Int8.width), fx.Int8)
        copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.Int32.width), fx.Int32)
        thr_copy_A = fx.make_tiled_copy_A(copy_a, tiled_mma).get_slice(tid)
        thr_copy_B = fx.make_tiled_copy_B(copy_b, tiled_mma).get_slice(tid)
        thr_copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)
        fx.copy(copy_a, thr_copy_A.partition_S(bA), thr_copy_A.retile(frag_A))
        fx.copy(copy_b, thr_copy_B.partition_S(bB), thr_copy_B.retile(frag_B))
        frag_C.fill(0)
        fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
        fx.copy(copy_c, thr_copy_C.retile(frag_C), thr_copy_C.partition_S(bC))

    @flyc.jit
    def tiledMma(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)) -> None:
        gemm_kernel(A, B, C).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    M = N = K = block_m
    A = torch.randint(-8, 8, (M, K), dtype=torch.int8).cuda()
    B = torch.randint(-8, 8, (N, K), dtype=torch.int8).cuda()
    C = torch.zeros(M, N, dtype=torch.int32).cuda()
    tiledMma(A, B, C, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    expected = (A.int() @ B.int().T).to(torch.int32)
    ok = torch.equal(C, expected)
    print(f"arch={_arch} atom=WMMA iu8→i32 correct:", ok)
    if not ok:
        print("Max diff:", (C - expected).abs().max().item())
    return ok


def _run_fp8() -> bool:
    """One WMMA tile: 16x16x16 Float8E4M3FN×Float8E4M3FN → f32."""
    if not hasattr(torch, "float8_e4m3fn"):
        print(f"arch={_arch} atom=WMMA fp8→f32 SKIP: torch.float8_e4m3fn missing")
        return True

    @flyc.kernel
    def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor) -> None:
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        A = fx.rocdl.make_buffer_tensor(A)
        B = fx.rocdl.make_buffer_tensor(B)
        C = fx.rocdl.make_buffer_tensor(C)
        bA = fx.make_view(fx.get_iter(A), fx.make_layout((block_m, block_k), (block_k, 1)))
        bB = fx.make_view(fx.get_iter(B), fx.make_layout((block_n, block_k), (block_k, 1)))
        bC = fx.make_view(fx.get_iter(C), fx.make_layout((block_m, block_n), (block_n, 1)))
        _ = bid
        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 16, fx.Float8E4M3FN, fx.Float32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        thr_mma = tiled_mma.thr_slice(tid)
        frag_A = thr_mma.make_fragment_A(bA)
        frag_B = thr_mma.make_fragment_B(bB)
        frag_C = thr_mma.make_fragment_C(bC)
        copy_a = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.Float8E4M3FN.width), fx.Float8E4M3FN)
        copy_b = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.Float8E4M3FN.width), fx.Float8E4M3FN)
        copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.Float32.width), fx.Float32)
        thr_copy_A = fx.make_tiled_copy_A(copy_a, tiled_mma).get_slice(tid)
        thr_copy_B = fx.make_tiled_copy_B(copy_b, tiled_mma).get_slice(tid)
        thr_copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)
        fx.copy(copy_a, thr_copy_A.partition_S(bA), thr_copy_A.retile(frag_A))
        fx.copy(copy_b, thr_copy_B.partition_S(bB), thr_copy_B.retile(frag_B))
        frag_C.fill(0)
        fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
        fx.copy(copy_c, thr_copy_C.retile(frag_C), thr_copy_C.partition_S(bC))

    @flyc.jit
    def tiledMma(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)) -> None:
        gemm_kernel(A, B, C).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    M = N = K = block_m
    A_f = torch.randn(M, K, dtype=torch.float32).cuda().clamp(-2, 2)
    B_f = torch.randn(N, K, dtype=torch.float32).cuda().clamp(-2, 2)
    A = A_f.to(torch.float8_e4m3fn)
    B = B_f.to(torch.float8_e4m3fn)
    C = torch.zeros(M, N, dtype=torch.float32).cuda()
    tiledMma(A, B, C, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    expected = A.float() @ B.float().T
    ok = torch.allclose(C, expected, atol=5e-1, rtol=5e-2)
    print(f"arch={_arch} atom=WMMA fp8_e4m3fn→f32 correct:", ok)
    if not ok:
        print("Max diff:", (C - expected).abs().max().item())
    return ok


ok_bf16 = _run_bf16()
ok_iu8 = _run_iu8()
ok_fp8 = _run_fp8()
if not (ok_bf16 and ok_iu8 and ok_fp8):
    sys.exit(1)
