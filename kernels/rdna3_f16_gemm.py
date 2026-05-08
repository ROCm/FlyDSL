#!/usr/bin/env python3
"""WMMA GEMM kernel for RDNA 3 / 3.5 (gfx1100/gfx1150, wave32).

Atom-based PoC kernel using the FlyDSL Layout API + ``MmaOpRDNA3_WMMAType``.

Differences from the existing :mod:`kernels.rdna_f16_gemm`:

- Goes through ``fly.make_mma_atom`` / ``fly.gemm`` instead of raw
  ``rocdl.wmma_*`` intrinsics. This exercises the new
  ``MmaOpRDNA3_WMMAType`` and the corresponding TiledMma layout pipeline.
- Targets ``gfx115*`` / ``gfx110*`` chips (Strix Point APUs and RDNA 3
  desktop). RDNA 4 (gfx120x) and gfx1250 use the gfx1250 atom; pick those
  via :func:`flydsl.expr.rocdl.universal.WMMA_GFX1250` if needed.

Shapes:
  - M=N=K=16 is one WMMA. The kernel stages multiple WMMAs by tiling the
    atom over an ``atom_layout`` of shape ``(reg_m, reg_n, reg_k)``.
  - Block is 32 threads = 1 wave32. The PoC uses a single warp per
    workgroup; later passes can scale to multi-warp via additional
    blocks in the X dim.

Computes ``C[M,N] = A[M,K] @ B[N,K]^T``.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


WMMA_M = 16
WMMA_N = 16
WMMA_K = 16


def create_rdna3_wmma_gemm_module(
    M: int,
    N: int,
    K: int,
    *,
    in_dtype: str = "f16",
    out_dtype: str = "f32",
    reg_m: int = 1,
    reg_n: int = 1,
    reg_k: int = 1,
):
    """Build a single-warp RDNA 3 / 3.5 WMMA GEMM.

    ``reg_m / reg_n / reg_k`` control how many 16x16x16 WMMAs each warp issues
    along M / N / K. The block tile is therefore
    ``(WMMA_M*reg_m, WMMA_N*reg_n, WMMA_K*reg_k)`` and the launch grid is
    ``(M / (WMMA_M*reg_m), N / (WMMA_N*reg_n), 1)``.
    """

    BLOCK_M = WMMA_M * reg_m
    BLOCK_N = WMMA_N * reg_n
    BLOCK_K = WMMA_K * reg_k

    assert M % BLOCK_M == 0, f"M={M} must be divisible by BLOCK_M={BLOCK_M}"
    assert N % BLOCK_N == 0, f"N={N} must be divisible by BLOCK_N={BLOCK_N}"
    assert K % BLOCK_K == 0, f"K={K} must be divisible by BLOCK_K={BLOCK_K}"

    if in_dtype == "f16":
        ab_dtype = fx.Float16
    elif in_dtype == "bf16":
        ab_dtype = fx.BFloat16
    else:
        raise ValueError(f"unsupported in_dtype={in_dtype} (must be 'f16' or 'bf16')")

    if out_dtype == "f32":
        acc_dtype = fx.Float32
    elif out_dtype == "f16" and in_dtype == "f16":
        acc_dtype = fx.Float16
    elif out_dtype == "bf16" and in_dtype == "bf16":
        acc_dtype = fx.BFloat16
    else:
        raise ValueError(
            f"unsupported (in_dtype={in_dtype}, out_dtype={out_dtype}) combination; "
            "RDNA 3 / 3.5 WMMA supports F32 acc with F16/BF16, and same-precision "
            "F16 acc with F16, BF16 acc with BF16"
        )

    THREADS_PER_BLOCK = 32  # one wave32

    @flyc.kernel
    def gemm_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        bid_m = fx.block_idx.x
        bid_n = fx.block_idx.y

        A = fx.rocdl.make_buffer_tensor(A)
        B = fx.rocdl.make_buffer_tensor(B)
        C = fx.rocdl.make_buffer_tensor(C)

        bA_all = fx.zipped_divide(A, (BLOCK_M, BLOCK_K))
        bB_all = fx.zipped_divide(B, (BLOCK_N, BLOCK_K))
        bC = fx.zipped_divide(C, (BLOCK_M, BLOCK_N))

        bC = fx.slice(bC, (None, (bid_m, bid_n)))

        # Use the explicit RDNA3 atom factory so the kernel is independent of
        # the runtime arch (the auto-dispatching `fx.rocdl.WMMA` would also
        # work on gfx115*/gfx110* but we want the test to pin the type).
        mma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA_RDNA3(WMMA_M, WMMA_N, WMMA_K, ab_dtype, acc_dtype)
        )
        # Single warp per workgroup; reg_m / reg_n / reg_k atoms within.
        tiled_mma = fx.make_tiled_mma(
            mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0))
        )
        thr_mma = tiled_mma.thr_slice(tid)

        # Use the universal CDNA3 buffer-copy atom; on gfx115*/gfx110* the
        # underlying V# flag word is patched for RDNA via
        # `_get_buffer_flags()` in `flydsl/expr/buffer_ops.py`.
        copy_bits = ab_dtype.width  # 16 bits for f16/bf16
        copy_atom = fx.make_copy_atom(
            fx.rocdl.BufferCopy(copy_bits), ab_dtype
        )
        copy_atom_c = fx.make_copy_atom(
            fx.rocdl.BufferCopy(acc_dtype.width), acc_dtype
        )
        tiled_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma)
        tiled_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma)
        tiled_copy_C = fx.make_tiled_copy_C(copy_atom_c, tiled_mma)

        thr_copy_A = tiled_copy_A.get_slice(tid)
        thr_copy_B = tiled_copy_B.get_slice(tid)
        thr_copy_C = tiled_copy_C.get_slice(tid)

        copy_dst_C = thr_copy_C.partition_S(bC)

        frag_C = thr_mma.make_fragment_C(bC)
        copy_frag_C = thr_copy_C.retile(frag_C)

        frag_C.fill(0)
        for k_tile in fx.range_constexpr(K // BLOCK_K):
            # Collapse the K-grid mode to one tile per iteration so the source
            # tile remains rank-2 ((M_in, K_in)/(N_in, K_in)) for partitioning.
            bA = fx.slice(bA_all, (None, (bid_m, k_tile)))
            bB = fx.slice(bB_all, (None, (bid_n, k_tile)))

            copy_src_A = thr_copy_A.partition_S(bA)
            copy_src_B = thr_copy_B.partition_S(bB)

            frag_A = thr_mma.make_fragment_A(bA)
            frag_B = thr_mma.make_fragment_B(bB)
            copy_frag_A = thr_copy_A.retile(frag_A)
            copy_frag_B = thr_copy_B.retile(frag_B)

            fx.copy(copy_atom, copy_src_A, copy_frag_A, pred=None)
            fx.copy(copy_atom, copy_src_B, copy_frag_B, pred=None)
            fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

        fx.copy(copy_atom_c, copy_frag_C, copy_dst_C, pred=None)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_m = M // BLOCK_M
        grid_n = N // BLOCK_N
        gemm_kernel(A, B, C).launch(
            grid=(grid_m, grid_n, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
            stream=stream,
        )

    return launch_gemm, BLOCK_M, BLOCK_N, BLOCK_K
