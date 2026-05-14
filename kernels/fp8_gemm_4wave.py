# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""4-wave FP8 matmul with row-wise scaling for AMD CDNA4.

Algorithm derived from HipKittens FP8_4wave
(https://github.com/HazyResearch/HipKittens/blob/7782744ba1fd259a377a99e2ea8f71384cc80e55/kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu#L1).

Global IO, scale loads, and bf16 stores go through the layout API
(``fx.rocdl.make_buffer_tensor`` + ``fx.copy`` with ``BufferCopyLDS128b``
/ ``BufferCopy{16,32,128}b``). MFMAs use ``fly.mma_atom_call_ssa`` so
the chained Vec(4, f32) accumulator stays on AGPR. The XOR swizzle and
the 8-buffer LDS pipeline ping-pong are kept as direct arithmetic to
preserve the original kernel's interleaved-cluster scheduling.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels.fp8_gemm_utils import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    StoreC,
    make_fp8_buffer_tensor,
    pack_i32x4_i32x8,
    swizzle_128,
    wait_barrier,
)


def _divmod(a, b):
    return (a // b, a % b)


def _min(a, b):
    return arith.select(a < b, a, b)


def _xcd_swizzle(num_pid_m, num_pid_n):
    NUM_XCDS = 8
    WGM = 4
    NUM_CUS = 32 * NUM_XCDS
    SWIZZLE_THRESHOLD = 4 * NUM_CUS

    wgid = fx.block_idx.x

    num_wg = num_pid_m * num_pid_n

    if num_wg <= SWIZZLE_THRESHOLD or num_wg % NUM_XCDS != 0:
        return _divmod(wgid, num_pid_n)

    intra_xcd, xcd = _divmod(wgid, NUM_XCDS)
    wgid = xcd * (num_wg // NUM_XCDS) + intra_xcd
    num_wgid_in_group = WGM * num_pid_n
    group_id, intra_group = _divmod(wgid, num_wgid_in_group)
    first_pid_m = group_id * WGM
    group_size_m = _min(num_pid_m - first_pid_m, WGM)
    pid_n, intra_group_m = _divmod(intra_group, group_size_m)
    pid_m = first_pid_m + intra_group_m
    return (pid_m, pid_n)


def compile_fp8_gemm(*, M: int, N: int, K: int, BLOCK_M: int = 256, BLOCK_N: int = 256, use_xcd_remap: bool = True):
    # MFMA atom is 16x16x128; 4 waves in a 2x2 config require BLOCK >= 64.
    BLOCK_K = 128
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    assert BLOCK_M >= 64 and BLOCK_N >= 64
    assert N % BLOCK_N == 0 and M % BLOCK_M == 0 and K % BLOCK_K == 0

    N_BLOCKS = N // BLOCK_N
    K_ITERS = K // BLOCK_K
    # Number of 16-row 16x128 tiles per wave per A/B partition.
    N_TILES_A = BLOCK_M // 4 // 16
    N_TILES_B = BLOCK_N // 4 // 16
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0

    _use_interleaved_block = BLOCK_M == 256 and BLOCK_N == 256

    A_lds_cur0_alloc = SmemAllocator(None, "gfx950", "A_lds_cur_0")
    A_lds_cur1_alloc = SmemAllocator(None, "gfx950", "A_lds_cur_1")
    A_lds_next0_alloc = SmemAllocator(None, "gfx950", "A_lds_next_0")
    A_lds_next1_alloc = SmemAllocator(None, "gfx950", "A_lds_next_1")
    B_lds_cur0_alloc = SmemAllocator(None, "gfx950", "B_lds_cur_0")
    B_lds_cur1_alloc = SmemAllocator(None, "gfx950", "B_lds_cur_1")
    B_lds_next0_alloc = SmemAllocator(None, "gfx950", "B_lds_next_0")
    B_lds_next1_alloc = SmemAllocator(None, "gfx950", "B_lds_next_1")

    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    A_lds_cur0_alloc.ptr = a_lds_size
    A_lds_cur1_alloc.ptr = a_lds_size
    A_lds_next0_alloc.ptr = a_lds_size
    A_lds_next1_alloc.ptr = a_lds_size
    B_lds_cur0_alloc.ptr = b_lds_size
    B_lds_cur1_alloc.ptr = b_lds_size
    B_lds_next0_alloc.ptr = b_lds_size
    B_lds_next1_alloc.ptr = b_lds_size

    @flyc.kernel
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type

        a_cur0 = SmemPtr(A_lds_cur0_alloc.get_base(), 0, F8_IR_t, shape=(a_lds_size,)).get()
        a_cur1 = SmemPtr(A_lds_cur1_alloc.get_base(), 0, F8_IR_t, shape=(a_lds_size,)).get()
        a_next0 = SmemPtr(A_lds_next0_alloc.get_base(), 0, F8_IR_t, shape=(a_lds_size,)).get()
        a_next1 = SmemPtr(A_lds_next1_alloc.get_base(), 0, F8_IR_t, shape=(a_lds_size,)).get()

        b_cur0 = SmemPtr(B_lds_cur0_alloc.get_base(), 0, F8_IR_t, shape=(b_lds_size,)).get()
        b_cur1 = SmemPtr(B_lds_cur1_alloc.get_base(), 0, F8_IR_t, shape=(b_lds_size,)).get()
        b_next0 = SmemPtr(B_lds_next0_alloc.get_base(), 0, F8_IR_t, shape=(b_lds_size,)).get()
        b_next1 = SmemPtr(B_lds_next1_alloc.get_base(), 0, F8_IR_t, shape=(b_lds_size,)).get()

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64

        if const_expr(use_xcd_remap):
            tile_i, tile_j = _xcd_swizzle(M // BLOCK_M, N // BLOCK_N)
        else:
            tile_i, tile_j = _divmod(fx.block_idx.x, N_BLOCKS)

        wave_i = wave_id // 2
        wave_j = wave_id % 2
        A0_gl_offset = (tile_i * BLOCK_M) * K
        A1_gl_offset = (tile_i * BLOCK_M + LDS_BLOCK_M) * K
        B0_gl_offset = (tile_j * BLOCK_N) * K
        B1_gl_offset = (tile_j * BLOCK_N + LDS_BLOCK_N) * K

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        ga_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        gb_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        def _compute_global_swizzle():
            offsets = []
            for round in range_constexpr(max(N_TILES_A, N_TILES_B)):
                row = lane_id // 8 + wave_id * 8 + round * 32
                col = (lane_id % 8) * 16
                r, c = swizzle_128(row, col)
                offsets.append(r * K + c)
            return offsets

        def _compute_lds_swizzle(s2r):
            lds_swz = []
            for row_offset in range_constexpr(s2r.n_tiles):
                row = s2r.wave_idx * (s2r.n_tiles * 16) + row_offset * 16 + lane_id % 16
                swz = []
                for i in range_constexpr(2):
                    col = (lane_id // 16) * 16 + i * 64
                    r, c = swizzle_128(row, col)
                    swz.append(r * BLOCK_K + c)
                lds_swz.append(swz)
            return lds_swz

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)

        def _interleaved_cluster(lds_dst, g2s, k_offset, s2r, lds_src, a, b, c):
            # 64x64 output via 4x4 MFMAs, with per-tile G->LDS and LDS->reg
            # loads interleaved between MFMAs to hide latency.
            rt_dst = []

            c[mfma.idx(0, 0)] = mfma.call_one(a, b, c, 0, 0)
            c[mfma.idx(0, 1)] = mfma.call_one(a, b, c, 0, 1)

            lds_swz = _compute_lds_swizzle(s2r)
            g2s.load_one(lds_dst, k_offset, 0)
            rt_dst_0 = s2r.load_one(lds_src, lds_swz[0][0])

            c[mfma.idx(0, 2)] = mfma.call_one(a, b, c, 0, 2)

            rt_dst_1 = s2r.load_one(lds_src, lds_swz[0][1])
            rt_dst.append(pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c[mfma.idx(0, 3)] = mfma.call_one(a, b, c, 0, 3)

            g2s.load_one(lds_dst, k_offset, 1)
            rt_dst_0 = s2r.load_one(lds_src, lds_swz[1][0])

            c[mfma.idx(1, 0)] = mfma.call_one(a, b, c, 1, 0)
            c[mfma.idx(1, 1)] = mfma.call_one(a, b, c, 1, 1)

            rt_dst_1 = s2r.load_one(lds_src, lds_swz[1][1])
            rt_dst.append(pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c[mfma.idx(1, 2)] = mfma.call_one(a, b, c, 1, 2)
            c[mfma.idx(1, 3)] = mfma.call_one(a, b, c, 1, 3)

            g2s.load_one(lds_dst, k_offset, 2)
            rt_dst_0 = s2r.load_one(lds_src, lds_swz[2][0])

            c[mfma.idx(2, 0)] = mfma.call_one(a, b, c, 2, 0)
            c[mfma.idx(2, 1)] = mfma.call_one(a, b, c, 2, 1)

            rt_dst_1 = s2r.load_one(lds_src, lds_swz[2][1])
            rt_dst.append(pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c[mfma.idx(2, 2)] = mfma.call_one(a, b, c, 2, 2)
            c[mfma.idx(2, 3)] = mfma.call_one(a, b, c, 2, 3)

            g2s.load_one(lds_dst, k_offset, 3)
            rt_dst_0 = s2r.load_one(lds_src, lds_swz[3][0])

            c[mfma.idx(3, 0)] = mfma.call_one(a, b, c, 3, 0)
            c[mfma.idx(3, 1)] = mfma.call_one(a, b, c, 3, 1)

            rt_dst_1 = s2r.load_one(lds_src, lds_swz[3][1])
            rt_dst.append(pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c[mfma.idx(3, 2)] = mfma.call_one(a, b, c, 3, 2)
            c[mfma.idx(3, 3)] = mfma.call_one(a, b, c, 3, 3)

            return c, rt_dst

        def _compute_cluster(lds_dst, g2s, k_offset, s2r, lds_src, a, b, c):
            g2s.load(lds_dst, k_offset)
            rt_dst = s2r.load(lds_src)
            c = mfma.call(a, b, c)
            return c, rt_dst

        def _compute_block(lds_dst, g2s, k_offset, s2r, lds_src, a, b, c):
            if const_expr(_use_interleaved_block):
                return _interleaved_cluster(lds_dst, g2s, k_offset, s2r, lds_src, a, b, c)
            else:
                return _compute_cluster(lds_dst, g2s, k_offset, s2r, lds_src, a, b, c)

        global_offsets = _compute_global_swizzle()
        a_g2s = G2SLoader(ga_div, global_offsets, N_TILES_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(gb_div, global_offsets, N_TILES_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(lane_id, wave_i, N_TILES_A)
        b_s2r = S2RLoader(lane_id, wave_j, N_TILES_B)
        store_c = StoreC(A_scale, B_scale, C, N, lane_id, mfma.idx, N_TILES_A, N_TILES_B)

        # Each wave handles 2x2 64x64 sub-tiles of the output.
        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        # Prologue: 8-buffer LDS pipeline pre-fill.
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K)
        a_g2s.load(a_next1, A1_gl_offset + 1 * BLOCK_K)

        wait_barrier((3 * N_TILES_A) + (4 * N_TILES_B))

        a0_frag = a_s2r.load(a_cur0)

        wait_barrier((3 * N_TILES_A) + (3 * N_TILES_B))

        b0_frag = b_s2r.load(b_cur0)

        for k in range_constexpr(K_ITERS - 2):
            wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))

            c00_frag, b1_frag = _compute_block(
                a_cur0,
                a_g2s,
                A0_gl_offset + (k + 2) * BLOCK_K,
                b_s2r,
                b_cur1,
                a0_frag,
                b0_frag,
                c00_frag,
            )

            c01_frag, a1_frag = _compute_block(
                b_cur0,
                b_g2s,
                B0_gl_offset + (k + 2) * BLOCK_K,
                a_s2r,
                a_cur1,
                a0_frag,
                b1_frag,
                c01_frag,
            )

            wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))

            c10_frag, a0_frag = _compute_block(
                b_cur1,
                b_g2s,
                B1_gl_offset + (k + 2) * BLOCK_K,
                a_s2r,
                a_next0,
                a1_frag,
                b0_frag,
                c10_frag,
            )

            c11_frag, b0_frag = _compute_block(
                a_cur1,
                a_g2s,
                A1_gl_offset + (k + 2) * BLOCK_K,
                b_s2r,
                b_next0,
                a1_frag,
                b1_frag,
                c11_frag,
            )

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Tail step k_iters - 2.
        wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))
        b1_frag = b_s2r.load(b_cur1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        a1_frag = a_s2r.load(a_cur1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        wait_barrier((1 * N_TILES_A) + (1 * N_TILES_B))
        a0_frag = a_s2r.load(a_next0)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        b0_frag = b_s2r.load(b_next0)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Tail step k_iters - 1.
        base_row = tile_i * BLOCK_M + wave_i * (N_TILES_A * 16)
        base_col = tile_j * BLOCK_N + wave_j * (N_TILES_B * 16)
        wait_barrier(0)
        b1_frag = b_s2r.load(b_cur1)
        a1_frag = a_s2r.load(a_cur1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        stream: fx.Stream,
    ):
        from flydsl._mlir import ir

        A_lds_cur0_alloc.finalized = False
        A_lds_cur1_alloc.finalized = False
        A_lds_next0_alloc.finalized = False
        A_lds_next1_alloc.finalized = False
        B_lds_cur0_alloc.finalized = False
        B_lds_cur1_alloc.finalized = False
        B_lds_next0_alloc.finalized = False
        B_lds_next1_alloc.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            A_lds_cur0_alloc.finalize()
            A_lds_cur1_alloc.finalize()
            A_lds_next0_alloc.finalize()
            A_lds_next1_alloc.finalize()
            B_lds_cur0_alloc.finalize()
            B_lds_cur1_alloc.finalize()
            B_lds_next0_alloc.finalize()
            B_lds_next1_alloc.finalize()
        grid_x = (M * N) // (BLOCK_M * BLOCK_N)
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            value_attrs={"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"},
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm
