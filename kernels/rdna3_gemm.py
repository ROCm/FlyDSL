#!/usr/bin/env python3
"""Optimized WMMA GEMM kernel for RDNA 3 / 3.5 (gfx1100/gfx1150/gfx1151, wave32).

Builds on the atom-based path (``MmaOpRDNA3_WMMAType``) and applies the
standard atom-API optimizations:

  * Multi-wave workgroup (2x2 wave layout = 4 wave32 = 128 threads).
  * Per-warp reg_m × reg_n WMMA tile (4x4 = 16 WMMAs per warp per K-step
    by default, giving a 64x64 output tile per warp).
  * BLOCK = 128 x 128 x 32 (BLOCK_K = 2 x WMMA_K so each iteration issues
    2 K-tiles' worth of WMMAs per warp).
  * LDS staging for A and B with a 2-stage ping-pong buffer: each loop
    iteration writes the next K-tile into one half of LDS while reading
    the current K-tile from the other half, then swaps. This keeps a
    single barrier per iteration (instead of two for single-buffered).

Compared to ``kernels.rdna3_f16_gemm`` (single-wave PoC for atom
validation), this kernel is intended as a production starting point on
RDNA 3 / 3.5. Subsequent commits add software pipelining (overlapping
GMEM prefetch with WMMA compute) and bank-conflict swizzle.

Computes ``C[M, N] = A[M, K] @ B[N, K]^T`` (B is supplied transposed,
so both A and B are stored row-major contiguous along K).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx


WMMA_M = 16
WMMA_N = 16
WMMA_K = 16


def create_rdna3_gemm_module(
    M: int,
    N: int,
    K: int,
    *,
    in_dtype: str = "f16",
    out_dtype: str = "f32",
    waves_m: int = 2,
    waves_n: int = 2,
    reg_m: int = 4,
    reg_n: int = 4,
    reg_k: int = 2,
    group_m: int = 1,
):
    """Build an optimized RDNA 3 / 3.5 atom-based WMMA GEMM.

    Parameters
    ----------
    waves_m, waves_n : int
        Wave layout inside one workgroup. Default 2x2 = 4 wave32 = 128 threads.
    reg_m, reg_n : int
        Number of 16x16 WMMA atoms each warp issues along M / N per K-step.
        Default 4x4 = 16 WMMAs/warp/K-step (64x64 per-warp output tile).
    reg_k : int
        Number of WMMA_K-sized K-tiles per workgroup loop iteration. Default 2,
        so BLOCK_K = 32.
    group_m : int
        Workgroup swizzle group size along M (Triton-style super-grouping)
        for L2 locality. Default 1 (no swizzle). Empirically gfx1151 with
        4 MB L2 doesn't benefit from this at 4096^3 — left as a tunable.
    """
    BLOCK_M = WMMA_M * reg_m * waves_m
    BLOCK_N = WMMA_N * reg_n * waves_n
    BLOCK_K = WMMA_K * reg_k

    assert M % BLOCK_M == 0, f"M={M} must be divisible by BLOCK_M={BLOCK_M}"
    assert N % BLOCK_N == 0, f"N={N} must be divisible by BLOCK_N={BLOCK_N}"
    assert K % BLOCK_K == 0, f"K={K} must be divisible by BLOCK_K={BLOCK_K}"

    if in_dtype == "f16":
        ab_dtype = fx.Float16
    elif in_dtype == "bf16":
        ab_dtype = fx.BFloat16
    else:
        raise ValueError(f"unsupported in_dtype={in_dtype}")

    if out_dtype == "f32":
        acc_dtype = fx.Float32
    elif out_dtype == "f16" and in_dtype == "f16":
        acc_dtype = fx.Float16
    elif out_dtype == "bf16" and in_dtype == "bf16":
        acc_dtype = fx.BFloat16
    else:
        raise ValueError(
            f"unsupported (in_dtype={in_dtype}, out_dtype={out_dtype}) combination"
        )

    NUM_WAVES = waves_m * waves_n
    THREADS_PER_BLOCK = NUM_WAVES * 32
    GRID_M = M // BLOCK_M
    GRID_N = N // BLOCK_N
    EFFECTIVE_GROUP_M = min(group_m, GRID_M)

    # GMEM->LDS thread layout. Each thread loads a 128-bit vector
    # (val_per_thr f16 elements). thrs_col threads cover BLOCK_K, thrs_row
    # threads cover (a slice of) BLOCK_M, then the M dim repeats per thread
    # to fill BLOCK_M.
    val_per_thr_g2s = 128 // ab_dtype.width  # 8 f16 / 8 bf16 per 128b copy
    assert BLOCK_K % val_per_thr_g2s == 0, (
        f"BLOCK_K={BLOCK_K} must be divisible by val_per_thr_g2s={val_per_thr_g2s}"
    )
    thrs_col = BLOCK_K // val_per_thr_g2s
    assert THREADS_PER_BLOCK % thrs_col == 0, (
        f"THREADS_PER_BLOCK={THREADS_PER_BLOCK} not divisible by thrs_col={thrs_col}"
    )
    thrs_row = THREADS_PER_BLOCK // thrs_col
    assert BLOCK_M % thrs_row == 0 and BLOCK_M >= thrs_row, (
        f"BLOCK_M={BLOCK_M} must be a positive multiple of thrs_row={thrs_row}; "
        f"raise reg_m / reg_n / waves_* or lower BLOCK_K (this combo: "
        f"waves_m={waves_m}, waves_n={waves_n}, reg_m={reg_m}, reg_n={reg_n}, "
        f"reg_k={reg_k} -> BLOCK_M={BLOCK_M}, threads={THREADS_PER_BLOCK})"
    )
    assert BLOCK_N % thrs_row == 0 and BLOCK_N >= thrs_row, (
        f"BLOCK_N={BLOCK_N} must be a positive multiple of thrs_row={thrs_row}"
    )

    # LDS shared-memory size: 2 stages × (A tile + B tile) for ping-pong.
    LDS_STAGES = 2
    LDS_BYTES = LDS_STAGES * (BLOCK_M + BLOCK_N) * BLOCK_K * (ab_dtype.width // 8)
    K_ITERS = K // BLOCK_K
    assert K_ITERS >= 2, "Double-buffered kernel requires K >= 2 * BLOCK_K"

    @flyc.kernel
    def gemm_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        # Workgroup swizzle for L2 locality: visit EFFECTIVE_GROUP_M × GRID_N
        # tiles in M-fast order, then advance EFFECTIVE_GROUP_M M-tiles.
        # Triton-style "super-grouping". With EFFECTIVE_GROUP_M == 1 this
        # degenerates to plain N-major linear scheduling (same correctness,
        # no L2 reuse benefit).
        pid = fx.block_idx.x
        num_pid_in_group = EFFECTIVE_GROUP_M * GRID_N
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * EFFECTIVE_GROUP_M
        pid_in_group = pid % num_pid_in_group
        bid_m = first_pid_m + (pid_in_group % EFFECTIVE_GROUP_M)
        bid_n = pid_in_group // EFFECTIVE_GROUP_M

        A = fx.rocdl.make_buffer_tensor(A)
        B = fx.rocdl.make_buffer_tensor(B)
        C = fx.rocdl.make_buffer_tensor(C)

        bA_all = fx.zipped_divide(A, (BLOCK_M, BLOCK_K))
        bB_all = fx.zipped_divide(B, (BLOCK_N, BLOCK_K))
        bC = fx.zipped_divide(C, (BLOCK_M, BLOCK_N))
        bC = fx.slice(bC, (None, (bid_m, bid_n)))

        mma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA_RDNA3(WMMA_M, WMMA_N, WMMA_K, ab_dtype, acc_dtype)
        )
        # Wave layout (waves_m, waves_n, 1) — strides (1, waves_m, 0) pack
        # warp_id = warp_m + warp_n * waves_m. The number of WMMAs each warp
        # issues per K-step (reg_m, reg_n, reg_k) is *implicit* from the
        # ratio  BLOCK_M / (waves_m * WMMA_M)  etc., not passed here.
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((waves_m, waves_n, 1), (1, waves_m, 0)),
        )
        thr_mma = tiled_mma.thr_slice(tid)

        # ── LDS allocation (2-stage ping-pong) ────────────────────────
        # Layout: [sA[stage=0] | sA[stage=1] | sB[stage=0] | sB[stage=1]]
        # Each tile is row-major to match the row-major GMEM source.
        smem_base = fx.get_dyn_shared()
        smem_ab = fx.recast_iter(
            fx.PointerType.get(
                ab_dtype.ir_type, fx.AddressSpace.Shared, 128
            ),
            smem_base,
        )
        # 3D layout: outer axis is the stage (slowest), then row-major K-fast tile.
        sA = fx.make_view(
            smem_ab,
            fx.make_ordered_layout(
                (BLOCK_M, BLOCK_K, LDS_STAGES), (1, 0, 2)
            ),
        )
        smem_b_ptr = fx.add_offset(
            smem_ab, LDS_STAGES * BLOCK_M * BLOCK_K
        )
        sB = fx.make_view(
            smem_b_ptr,
            fx.make_ordered_layout(
                (BLOCK_N, BLOCK_K, LDS_STAGES), (1, 0, 2)
            ),
        )

        # ── Copy atoms ────────────────────────────────────────────────
        # GMEM->LDS uses 128-bit BufferCopy (vec_8 f16). LDS->register
        # uses the universal 128-bit copy. Register->GMEM for C uses
        # acc-dtype-wide BufferCopy.
        bcopy_128b_ab = fx.make_copy_atom(fx.rocdl.BufferCopy(128), ab_dtype)
        ucopy_128b_ab = fx.make_copy_atom(fx.UniversalCopy128b(), ab_dtype)
        copy_atom_c = fx.make_copy_atom(
            fx.rocdl.BufferCopy(acc_dtype.width), acc_dtype
        )

        # GMEM -> LDS tiled copies (cooperative across all 128 threads).
        # thread-value layout: outer (thr_col, thr_row) × inner (1 along M,
        # val_per_thr along K).  tile is (thrs_row × BLOCK_K) so along M
        # the layout repeats BLOCK_M / thrs_row times per thread.
        tiled_copy_g2s_A = fx.make_tiled_copy(
            bcopy_128b_ab,
            fx.make_layout(
                ((thrs_col, thrs_row), (1, val_per_thr_g2s)),
                ((thrs_row * val_per_thr_g2s, 1), (1, thrs_row)),
            ),
            fx.make_tile(thrs_row, BLOCK_K),
        )
        tiled_copy_g2s_B = fx.make_tiled_copy(
            bcopy_128b_ab,
            fx.make_layout(
                ((thrs_col, thrs_row), (1, val_per_thr_g2s)),
                ((thrs_row * val_per_thr_g2s, 1), (1, thrs_row)),
            ),
            fx.make_tile(thrs_row, BLOCK_K),
        )
        thr_g2s_A = tiled_copy_g2s_A.get_slice(tid)
        thr_g2s_B = tiled_copy_g2s_B.get_slice(tid)

        # LDS -> register tiled copies (per-warp; derived from tiled_mma).
        tiled_copy_s2r_A = fx.make_tiled_copy_A(ucopy_128b_ab, tiled_mma)
        tiled_copy_s2r_B = fx.make_tiled_copy_B(ucopy_128b_ab, tiled_mma)
        thr_s2r_A = tiled_copy_s2r_A.get_slice(tid)
        thr_s2r_B = tiled_copy_s2r_B.get_slice(tid)

        # Register -> GMEM for C.
        tiled_copy_C = fx.make_tiled_copy_C(copy_atom_c, tiled_mma)
        thr_copy_C = tiled_copy_C.get_slice(tid)

        # Pre-compute per-stage partitions. With LDS stages = 2 the stage
        # axis is the *outer* dim of sA/sB so slicing [..., 0] / [..., 1] is
        # constexpr and folds into a static ptr offset.
        thr_sA_g2s_stage = [
            thr_g2s_A.partition_D(sA[None, None, s]) for s in range(LDS_STAGES)
        ]
        thr_sB_g2s_stage = [
            thr_g2s_B.partition_D(sB[None, None, s]) for s in range(LDS_STAGES)
        ]
        thr_sA_s2r_stage = [
            thr_s2r_A.partition_S(sA[None, None, s]) for s in range(LDS_STAGES)
        ]
        thr_sB_s2r_stage = [
            thr_s2r_B.partition_S(sB[None, None, s]) for s in range(LDS_STAGES)
        ]

        copy_dst_C = thr_copy_C.partition_S(bC)
        frag_C = thr_mma.make_fragment_C(bC)
        copy_frag_C = thr_copy_C.retile(frag_C)
        # mma fragments are register-resident; tied to one (any) stage of LDS
        # for shape inference. They get re-filled from LDS each iter.
        frag_A = thr_mma.make_fragment_A(sA[None, None, 0])
        frag_B = thr_mma.make_fragment_B(sB[None, None, 0])
        s2r_frag_A = thr_s2r_A.retile(frag_A)
        s2r_frag_B = thr_s2r_B.retile(frag_B)

        frag_C.fill(0)

        # ── Pipeline stage helpers ────────────────────────────────────
        def issue_g2s(k_tile, stage):
            """GMEM -> LDS[stage] for A and B at K-tile index k_tile."""
            bA = fx.slice(bA_all, (None, (bid_m, k_tile)))
            bB = fx.slice(bB_all, (None, (bid_n, k_tile)))
            fx.copy(
                bcopy_128b_ab,
                thr_g2s_A.partition_S(bA),
                thr_sA_g2s_stage[stage],
                pred=None,
            )
            fx.copy(
                bcopy_128b_ab,
                thr_g2s_B.partition_S(bB),
                thr_sB_g2s_stage[stage],
                pred=None,
            )

        def compute_from_stage(stage):
            """LDS[stage] -> register -> WMMA accumulate."""
            fx.copy(ucopy_128b_ab, thr_sA_s2r_stage[stage], s2r_frag_A, pred=None)
            fx.copy(ucopy_128b_ab, thr_sB_s2r_stage[stage], s2r_frag_B, pred=None)
            fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

        # ── Pipeline ──────────────────────────────────────────────────
        # Prologue: load K-tile 0 into LDS[0].
        issue_g2s(fx.Int32(0), 0)
        fx.gpu.barrier()

        # Main loop: K_ITERS - 1 iterations. Each iter k:
        #   - issues G2S for tile k+1 into LDS[(k+1) & 1]
        #   - computes from LDS[k & 1] (already populated)
        #   - barrier so next iter can both compute from LDS[(k+1) & 1] and
        #     overwrite LDS[k & 1]
        # We unroll iterations in pairs so the stage index is constexpr.
        # Outer loop is a runtime SCF loop over (K_ITERS - 1) // 2 pairs to
        # avoid unrolling 128 K-iterations into the IR for K=4096.
        full_pairs = (K_ITERS - 1) // 2
        tail = (K_ITERS - 1) - full_pairs * 2  # 0 or 1 leftover iters

        for pair_idx in range(full_pairs):  # SCF runtime loop over pairs
            pair_i32 = fx.arith.index_cast(fx.T.i32(), pair_idx)
            base_k = pair_i32 * fx.Int32(2)

            # Iter even: compute from stage 0, prefetch into stage 1.
            issue_g2s(base_k + fx.Int32(1), 1)
            compute_from_stage(0)
            fx.gpu.barrier()

            # Iter odd: compute from stage 1, prefetch into stage 0.
            issue_g2s(base_k + fx.Int32(2), 0)
            compute_from_stage(1)
            fx.gpu.barrier()

        # Tail prefetch+compute (if K_ITERS - 1 is odd, one more iter).
        if tail == 1:
            tail_k = fx.Int32(full_pairs * 2 + 1)
            issue_g2s(tail_k, 1)
            compute_from_stage(0)
            fx.gpu.barrier()

        # Final compute for the last loaded tile (no prefetch needed).
        last_stage = (K_ITERS - 1) & 1
        compute_from_stage(last_stage)

        fx.copy(copy_atom_c, copy_frag_C, copy_dst_C, pred=None)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        gemm_kernel(A, B, C).launch(
            grid=(GRID_M * GRID_N, 1, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
            smem=LDS_BYTES,
            stream=stream,
        )

    return launch_gemm, BLOCK_M, BLOCK_N, BLOCK_K
