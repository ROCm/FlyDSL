#!/usr/bin/env python3
"""WMMA GEMM kernel for RDNA4 (gfx120x, wave32), written against the layout API.

Architecture:
- 128x128x32 tiles, 4 waves (128 threads), 2x2 wave layout
- Each wave: 4 M-repeats x 4 N-repeats (64x64 output per wave)
- 2 K-steps per tile (BLOCK_K=32, WMMA_K=16) -> 32 WMMAs per tile
- Double-buffered LDS (ping-pong): compute from stage[i%2], prefetch into the other
- A[M,K] row-major GMEM, B_T[N,K] row-major GMEM
- K-padding on the LDS tiles for bank conflict avoidance

All of the fragment index math is derived from the `gfx120x.wmma` MMA atom
(``lib/Dialect/FlyROCDL/GFX120X/MmaAtom.cpp``) via ``make_tiled_copy_{A,B,C}``,
so the v8 RDNA4 register ABI lives in one place instead of being spelled out
here as lane arithmetic.

LDS layout (per stage):
  A tile: BLOCK_M rows x (BLOCK_K + a_k_pad) cols x 2B, row-major
  B tile: BLOCK_N rows x (BLOCK_K + b_k_pad) cols x 2B, row-major
  ~20KB per stage, 40KB for both.

Pipeline: split GMEM->register load and register->LDS store, double buffered.

Computes C[M,N] = A[M,K] @ B_T[N,K]^T
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import vector
from flydsl.expr import as_ir_value, const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16
WAVE_SIZE = 32

# 128-bit GMEM loads / LDS accesses.
LOAD_VEC_BITS = 128


def create_wmma_gemm_module(
    M: int,
    N: int,
    K: int,
    in_dtype="bf16",
    out_dtype="bf16",
    *,
    rounding="rn",  # "rn" (round to nearest) or "rs" (stochastic rounding)
    reg_m=4,  # M-repeats per wave
    reg_n=4,  # N-repeats per wave
    reg_k=2,  # K-steps per tile (32/16=2)
    waves_m=2,  # waves in M dimension
    waves_n=2,  # waves in N dimension
    group_m=8,
    a_k_pad=8,  # K-padding for A in LDS (bank conflict avoidance)
    b_k_pad=8,  # K-padding for B in LDS
):
    BLOCK_M = WMMA_M * reg_m * waves_m  # 16*4*2 = 128
    BLOCK_N = WMMA_N * reg_n * waves_n  # 16*4*2 = 128
    BLOCK_K = WMMA_K * reg_k  # 16*2 = 32
    NUM_WAVES = waves_m * waves_n  # 2*2 = 4
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE  # 128

    assert reg_k >= 2 and reg_k % 2 == 0
    assert rounding in ("rn", "rs"), f"rounding must be 'rn' or 'rs', got {rounding!r}"
    if rounding == "rs":
        assert out_dtype == "bf16", "stochastic rounding currently supports bf16 output only"

    elem_cls = fx.BFloat16 if in_dtype == "bf16" else fx.Float16
    out_elem_cls = {"bf16": fx.BFloat16, "f16": fx.Float16, "f32": fx.Float32}[out_dtype]
    elem_bits = elem_cls.width

    # G2S thread geometry: each thread moves one 128-bit chunk per load, and
    # the tiled copy repeats over M to cover the whole tile.
    load_vec = LOAD_VEC_BITS // elem_bits  # 8
    thrs_k = BLOCK_K // load_vec  # 4 threads span the K extent
    thrs_m = THREADS_PER_BLOCK // thrs_k  # 32 rows per copy step
    assert thrs_k * thrs_m == THREADS_PER_BLOCK
    assert BLOCK_M % thrs_m == 0 and BLOCK_N % thrs_m == 0

    # LDS layout with K-padding for bank conflict avoidance.
    BLOCK_K_PAD_A = BLOCK_K + a_k_pad  # 40
    BLOCK_K_PAD_B = BLOCK_K + b_k_pad  # 40
    LDS_A_ELEMS = BLOCK_M * BLOCK_K_PAD_A  # 5120
    LDS_B_ELEMS = BLOCK_N * BLOCK_K_PAD_B  # 5120

    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    num_k_tiles = K // BLOCK_K
    assert num_k_tiles >= 2, "Need at least 2 K-tiles for prefetch pipeline"

    k_iters = BLOCK_K // WMMA_K  # MMA k-steps per tile
    acc_size = (BLOCK_M * BLOCK_N) // THREADS_PER_BLOCK  # accumulator VGPRs per thread

    grid_m = M // BLOCK_M
    grid_n = N // BLOCK_N

    # Two-tiles-per-iteration ping-pong: the LDS stage index must be a
    # compile-time constant, so the trip count is halved and the odd/even
    # remainder is peeled.
    tail = 1 if (num_k_tiles % 2 == 1) else 2
    loop_end = (num_k_tiles - tail) // 2
    k_tail0 = num_k_tiles - tail  # always even, so peeled stage == j % 2

    @fx.struct
    class SharedStorage:
        a0: fx.Array[elem_cls, LDS_A_ELEMS, 16]
        b0: fx.Array[elem_cls, LDS_B_ELEMS, 16]
        a1: fx.Array[elem_cls, LDS_A_ELEMS, 16]
        b1: fx.Array[elem_cls, LDS_B_ELEMS, 16]

    @flyc.kernel
    def wmma_gemm_kernel(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_bt: fx.Tensor,
        tiled_mma: fx.TiledMma,
        tiled_copy_g2s: fx.TiledCopy,
        sr_seed: fx.Int32,  # runtime seed; only read on the stochastic-rounding path
    ):
        tid = fx.thread_idx.x
        pid = gpu.block_id("x")

        # Swizzle workgroup mapping for L2 locality
        effective_group_m = min(group_m, grid_m)
        num_pid_in_group = effective_group_m * grid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * effective_group_m

        pid_in_group = pid % num_pid_in_group
        bid_m = first_pid_m + (pid_in_group % effective_group_m)
        bid_n = pid_in_group // effective_group_m

        gA = fx.rocdl.make_buffer_tensor(arg_a)
        gB = fx.rocdl.make_buffer_tensor(arg_bt)
        gC = fx.rocdl.make_buffer_tensor(arg_c)

        # Block tiles: A/B keep the K mode so the pipeline can index k-tiles.
        tA = fx.flat_divide(gA, fx.make_tile(BLOCK_M, BLOCK_K))[None, None, bid_m, None]
        tB = fx.flat_divide(gB, fx.make_tile(BLOCK_N, BLOCK_K))[None, None, bid_n, None]
        tC = fx.flat_divide(gC, fx.make_tile(BLOCK_M, BLOCK_N))[None, None, bid_m, bid_n]

        buf_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_cls)
        uni_copy = fx.make_copy_atom(fx.UniversalCopy128b(), elem_cls)

        # Per-thread slices
        thr_mma = tiled_mma.thr_slice(tid)
        thr_g2s = tiled_copy_g2s.get_slice(tid)
        thr_s2r_A = fx.make_tiled_copy_A(uni_copy, tiled_mma).get_slice(tid)
        thr_s2r_B = fx.make_tiled_copy_B(uni_copy, tiled_mma).get_slice(tid)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        sA_stages = [
            fx.make_view(arr.ptr, fx.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K_PAD_A, 1))) for arr in (lds.a0, lds.a1)
        ]
        sB_stages = [
            fx.make_view(arr.ptr, fx.make_layout((BLOCK_N, BLOCK_K), (BLOCK_K_PAD_B, 1))) for arr in (lds.b0, lds.b1)
        ]

        # Partitions
        pA_g = thr_g2s.partition_S(tA)
        pB_g = thr_g2s.partition_S(tB)
        pA_s = [thr_g2s.partition_D(s) for s in sA_stages]
        pB_s = [thr_g2s.partition_D(s) for s in sB_stages]
        pA_s2r = [thr_s2r_A.partition_S(s) for s in sA_stages]
        pB_s2r = [thr_s2r_B.partition_S(s) for s in sB_stages]

        # Fragments
        frag_copy_A = fx.make_fragment_like(pA_s[0][None, None, None])
        frag_copy_B = fx.make_fragment_like(pB_s[0][None, None, None])
        frag_A = thr_mma.make_fragment_A(sA_stages[0])
        frag_B = thr_mma.make_fragment_B(sB_stages[0])
        frag_C = thr_mma.make_fragment_C(tC)
        frag_A_retile = thr_s2r_A.retile(frag_A)
        frag_B_retile = thr_s2r_B.retile(frag_B)

        copy_out = fx.make_copy_atom(fx.rocdl.BufferCopy(out_elem_cls.width), out_elem_cls)
        thr_r2g_C = fx.make_tiled_copy_C(copy_out, tiled_mma).get_slice(tid)
        pC_g = thr_r2g_C.partition_S(tC)
        if const_expr(out_elem_cls is fx.Float32):
            frag_C_out = frag_C
        else:
            frag_C_out = fx.make_fragment_like(frag_C, out_elem_cls.ir_type)
        frag_C_retile = thr_r2g_C.retile(frag_C_out)

        # ── Pipeline stages ───────────────────────────────────────────
        def gmem_to_reg(k_tile):
            fx.copy(buf_copy, pA_g[None, None, None, k_tile], frag_copy_A)
            fx.copy(buf_copy, pB_g[None, None, None, k_tile], frag_copy_B)

        def reg_to_lds(stage):
            fx.copy(uni_copy, frag_copy_A, pA_s[stage][None, None, None])
            fx.copy(uni_copy, frag_copy_B, pB_s[stage][None, None, None])

        def mma_kloop(stage):
            # All B operands for a k-step first, then one A at a time: keeps
            # only reg_n B + 1 A fragment live on top of the accumulators.
            for ki in range_constexpr(k_iters):
                fx.copy(uni_copy, pB_s2r[stage][None, None, ki], frag_B_retile[None, None, ki])
                fx.copy(uni_copy, pA_s2r[stage][None, None, ki], frag_A_retile[None, None, ki])
                fx.gemm(tiled_mma, frag_C, frag_A[None, None, ki], frag_B[None, None, ki], frag_C)

        def compute_tile(stage, next_k_tile):
            """Compute the tile resident in `stage`, prefetching `next_k_tile`."""
            do_next = next_k_tile is not None
            if const_expr(do_next):
                gmem_to_reg(next_k_tile)
            mma_kloop(stage)
            if const_expr(do_next):
                reg_to_lds(stage ^ 1)
                # Publishes the prefetched tile and retires this tile's LDS
                # reads before the next iteration overwrites `stage`.
                gpu.barrier()

        # ── Prologue ──────────────────────────────────────────────────
        frag_C.fill(0)
        gmem_to_reg(fx.Int32(0))
        reg_to_lds(0)
        gpu.barrier()

        # ── Main tile loop: 2 tiles per iteration ─────────────────────
        if const_expr(loop_end > 0):
            for iv, state in range(0, loop_end, 1, init=[frag_C.load()]):
                frag_C.store(state[0])
                k_base = iv * 2
                compute_tile(0, fx.Int32(k_base + 1))
                compute_tile(1, fx.Int32(k_base + 2))
                results = yield [frag_C.load()]
            frag_C.store(results)

        # ── Peeled tail ───────────────────────────────────────────────
        for j in range_constexpr(tail):
            k_next = k_tail0 + j + 1
            compute_tile(j % 2, fx.Int32(k_next) if const_expr(k_next < num_k_tiles) else None)

        # ── Epilogue ──────────────────────────────────────────────────
        if const_expr(out_elem_cls is not fx.Float32):
            acc_vec = Vec(frag_C.load())
            if const_expr(rounding == "rs"):
                base_off = (pid * THREADS_PER_BLOCK + tid) * acc_size
                out_elems = []
                for p in range_constexpr(acc_size):
                    rbits = fx.rocdl.philox_4x32(fx.Uint32(base_off + p), fx.Uint32(sr_seed))[0]
                    out_elems.append(fx.rocdl.stochastic_round_bf16(acc_vec[p], rbits))
            else:
                out_elems = [acc_vec[p].to(out_elem_cls) for p in range_constexpr(acc_size)]
            out_vec = vector.from_elements(T.vec(acc_size, out_elem_cls.ir_type), [as_ir_value(e) for e in out_elems])
            frag_C_out.store(out_vec)
        fx.copy(copy_out, frag_C_retile, pC_g)

    # ── Host launcher ──────────────────────────────────────────────────────
    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_bt: fx.Tensor,
        stream: fx.Stream,
        sr_seed: fx.Int32 = 0,
    ):
        # 16x16x16 v8 WMMA atom (gfx120x.wmma), tiled over a waves_m x waves_n
        # wave grid. A 128x128 block tile then decomposes into reg_m x reg_n
        # atom repeats per wave without any explicit index math here.
        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, elem_cls, fx.Float32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((waves_m, waves_n, 1), (waves_n, 1, 0)))

        # G2S tiled copy: thread (tk, tm) moves elements [tk*load_vec, +load_vec)
        # of row tm, i.e. one 128-bit contiguous chunk.
        tiled_copy_g2s = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy128b(), elem_cls),
            fx.make_layout(
                ((thrs_k, thrs_m), (1, load_vec)),
                ((thrs_m * load_vec, 1), (1, thrs_m)),
            ),
            fx.make_tile(thrs_m, BLOCK_K),
        )

        arg_a_2d = fx.make_view(fx.get_iter(arg_a), fx.make_layout((M, K), (K, 1)))
        arg_bt_2d = fx.make_view(fx.get_iter(arg_bt), fx.make_layout((N, K), (K, 1)))
        arg_c_2d = fx.make_view(fx.get_iter(arg_c), fx.make_layout((M, N), (N, 1)))

        c1 = 1
        total_blocks = grid_m * grid_n

        wmma_gemm_kernel(arg_c_2d, arg_a_2d, arg_bt_2d, tiled_mma, tiled_copy_g2s, sr_seed).launch(
            grid=(total_blocks, c1, c1),
            block=(THREADS_PER_BLOCK, c1, c1),
            stream=stream,
        )

    return launch_gemm, BLOCK_M, BLOCK_N, BLOCK_K
