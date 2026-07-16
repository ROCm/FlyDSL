# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Implicit-GEMM conv3d (FP8, CDNA4) using the fp8_gemm 4-wave pipeline.

Sibling of ``conv3d_implicit_fp8_gemm8w`` -- same conv-specific im2col A-loader
and direct-store epilogue, but the GEMM core is ported from
``kernels/gemm/fp8_gemm_4wave.py`` instead of the 8-wave kernel:

* 256 threads / 4 waves (2x2 wave grid), 256x256x128 tile.
* 8 LDS half-block ping-pong buffers, but each half-block is filled in 4 G2S
  steps (vs 2 in the 8-wave kernel) because there are half as many waves.
* Hand-scheduled *interleaved cluster*: the 4x4 = 16 ``Mfma16x16x128AGPR`` calls
  (AGPR-pinned accumulator via inline asm) are interleaved with the S2R fragment
  loads and the G2S prefetch of the k+2 tile.
* XCD block-swizzle (``_xcd_swizzle``) for L2 reuse across XCDs on large grids.

The ONLY conv-specific component is the A-operand loader (``conv_a_g2s`` /
``conv_a_g2s_one``): instead of a contiguous GEMM read it computes the im2col
activation address per LDS chunk and deposits it into the exact swizzled
half-block LDS layout the GEMM's ``S2RLoader`` reads. The B (weight) operand is a
plain KTRSC ``(k, crs)`` matrix, so the GEMM's ``G2SLoader`` is reused verbatim.

x: (N, C, D, H, W) fp8 (E4M3FN) NCDHW, weight: (K, C, T, R, S) fp8 KCTRS.
Returns (N, K, Do, Ho, Wo) bf16. No scales, no split-K.

Rigid constraints (asserted): crs % 128 == 0, k % 256 == 0, c % 16 == 0,
crs // 128 >= 2. Shapes that violate these are out of scope for this kernel; use
``conv3d_implicit_fp8`` (the general kernel) for them.
"""

import functools
import os

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec
from kernels.conv.conv3d_implicit_fp8 import (
    _make_fp8_buffer_tensor_from_addr,
    _normalize_3,
    _prep_weight_fp8,
)
from kernels.conv.conv3d_implicit_fp8_gemm8w import _transpose_activation_fp8
from kernels.gemm.fp8_gemm_4wave import _xcd_swizzle
from kernels.gemm.fp8_gemm_utils import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    compute_global_swizzle,
    divmod,
    make_fp8_buffer_tensor,
    pack_i32x4_i32x8,
    swizzle_128,
    wait_barrier,
)

# Rigid 4-wave GEMM design: 256 threads, BLOCK_M=BLOCK_N=256, BLOCK_K=128.
BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 128
WARP_SIZE = 64
N_WAVES = 4
BLOCK_THREADS = N_WAVES * WARP_SIZE  # 256

LDS_BLOCK_M = BLOCK_M // 2  # 128
LDS_BLOCK_N = BLOCK_N // 2  # 128
# 4-wave 2x2 wave grid: each wave owns 4x4 = 16 16x16 sub-tiles per half-block pair.
N_TILES_A = BLOCK_M // 4 // 16  # 4
N_TILES_B = BLOCK_N // 4 // 16  # 4
N_ACCUMS = N_TILES_A * N_TILES_B  # 16
# One G2S step per S2R tile row (4), so 4 waves * 4 steps * 1024 = 16384 = half-block.
N_LDS_STEPS_A = N_TILES_A  # 4
N_LDS_STEPS_B = N_TILES_B  # 4
N_LDS_ROUNDS = max(N_TILES_A, N_TILES_B)  # 4

FP8_BYTES = 1


@functools.lru_cache(maxsize=256)
def compile_conv3d_implicit_fp8_gemm4w(
    n, c, d, h, width, k, kt, kh, kw, st, sh, sw, pt, ph, pw, has_bias=False, use_xcd_remap=True
):
    do = (d + 2 * pt - kt) // st + 1
    ho = (h + 2 * ph - kh) // sh + 1
    wo = (width + 2 * pw - kw) // sw + 1
    dhw = do * ho * wo
    hw_o = ho * wo
    npq = n * dhw
    crs = c * kt * kh * kw
    K_ITERS = crs // BLOCK_K

    assert c % 16 == 0, f"FP8 needs C % 16 == 0, got C={c}"
    assert crs % BLOCK_K == 0, f"gemm4w needs crs % 128 == 0 (aligned K), got crs={crs}"
    assert k % BLOCK_N == 0, f"gemm4w needs k % 256 == 0 (BLOCK_N), got k={k}"
    assert K_ITERS >= 2, f"gemm4w needs crs//128 >= 2, got {K_ITERS}"

    BIG_IN = (n * c * d * h * width) > 0x7FFFFFFF
    BIG_OUT = (n * k * do * ho * wo * 2) > 0x7FFFFFFF
    temporal_only_fast = (
        kh == 1
        and kw == 1
        and st == 1
        and sh == 1
        and sw == 1
        and ph == 0
        and pw == 0
        and do == d
        and ho == h
        and wo == width
    )

    grid_m = (npq + BLOCK_M - 1) // BLOCK_M
    grid_n = k // BLOCK_N
    grid_x = grid_m * grid_n

    a_lds_size = LDS_BLOCK_M * BLOCK_K  # 16384
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    elem_ty = fx.Float8E4M3FN

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[elem_ty, a_lds_size, 16]
        A_lds_cur_1: fx.Array[elem_ty, a_lds_size, 16]
        A_lds_next_0: fx.Array[elem_ty, a_lds_size, 16]
        A_lds_next_1: fx.Array[elem_ty, a_lds_size, 16]
        B_lds_cur_0: fx.Array[elem_ty, b_lds_size, 16]
        B_lds_cur_1: fx.Array[elem_ty, b_lds_size, 16]
        B_lds_next_0: fx.Array[elem_ty, b_lds_size, 16]
        B_lds_next_1: fx.Array[elem_ty, b_lds_size, 16]

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def conv3d_fp8_gemm4w_kernel(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor):
        f8_ir_t = elem_ty.ir_type
        x_num_records = n * d * h * width * c

        y_rsrc = buffer_ops.create_buffer_resource(y, max_size=False, num_records_bytes=npq * k * 2)
        if const_expr(has_bias):
            bias_rsrc = buffer_ops.create_buffer_resource(bias, max_size=False, num_records_bytes=k * 4)

        x_buf = make_fp8_buffer_tensor(x, f8_ir_t)
        x_div = fx.logical_divide(x_buf, fx.make_layout(1, 1))
        w_buf = make_fp8_buffer_tensor(weight, f8_ir_t)
        b_div = fx.logical_divide(w_buf, fx.make_layout(1, 1))

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        tid = fx.thread_idx.x
        lane_id = tid % WARP_SIZE
        wave_id = tid // WARP_SIZE
        wave_m = wave_id // 2
        wave_n = wave_id % 2

        # _xcd_swizzle expects runtime grid dims (the GEMM passes ceildiv over runtime
        # scalar args); wrap the compile-time grid extents as Int32 so its select()
        # conditions stay IR values rather than collapsing to Python bools.
        if const_expr(use_xcd_remap):
            block_m, block_n = _xcd_swizzle(fx.Int32(grid_m), fx.Int32(grid_n))
            block_m = fx.Index(block_m)
            block_n = fx.Index(block_n)
        else:
            block_m, block_n = divmod(fx.block_idx.x, grid_n)

        m_offset = block_m * BLOCK_M

        # BIG_IN: rebase the activation buffer to this block's (nbase, base_t) origin
        # so the per-tile relative i32 element offsets do not overflow (see the general
        # conv fp8 kernel for the derivation).
        if const_expr(BIG_IN):
            nbase = m_offset // dhw
            ot_base0 = (m_offset % dhw) // hw_o
            base_t = ot_base0 - fx.Index(pt)
            base_t = arith.select(base_t < fx.Index(0), fx.Index(0), base_t)
            x_base_byte = ((nbase * fx.Index(d) + base_t) * fx.Index(h)) * fx.Index(width) * fx.Index(c)
            x_addr = fx.Int64(buffer_ops.extract_base_index(x)) + fx.Int64(x_base_byte)
            x_div = fx.logical_divide(
                _make_fp8_buffer_tensor_from_addr(x_addr, f8_ir_t, x_buf),
                fx.make_layout(1, 1),
            )

        def in_range(v, hi):
            return (v >= fx.Index(0)) & (v < fx.Index(hi))

        # ---- im2col address for a (M-row, K-col) chunk of 16 contiguous channels ----
        # k_col is a multiple of 16 and (crs%128==0, c%16==0) => the 16 elements
        # k_col..k_col+15 are 16 consecutive channels of ONE (kt,kh,kw) tap, so one
        # spatial address + contiguous channel base. Returns the OOB-sentinel element
        # offset for padded / out-of-bounds taps (hardware bounds check zeroes LDS).
        def im2col_safe_elem(m_row, k_col):
            row_valid = m_row < fx.Index(npq)
            cc = k_col % c
            if const_expr(temporal_only_fast):
                kt_i = k_col // c
                temporal_delta = kt_i - pt
                out_t = (m_row // hw_o) % d
                in_t = out_t + temporal_delta
                valid = row_valid & in_range(in_t, d)
                if const_expr(BIG_IN):
                    g_elem = ((m_row + temporal_delta * hw_o) - (nbase * dhw + base_t * hw_o)) * c + cc
                else:
                    g_elem = (m_row + temporal_delta * hw_o) * c + cc
            else:
                n_idx = m_row // dhw
                rem = m_row % dhw
                ot = rem // hw_o
                rem2 = rem % hw_o
                oh = rem2 // wo
                ow = rem2 % wo
                ckk = k_col // c
                kw_i = ckk % kw
                ckk2 = ckk // kw
                kh_i = ckk2 % kh
                kt_i = ckk2 // kh
                in_t = ot * st + kt_i - pt
                in_h = oh * sh + kh_i - ph
                in_w = ow * sw + kw_i - pw
                valid = row_valid & in_range(in_t, d) & in_range(in_h, h) & in_range(in_w, width)
                if const_expr(BIG_IN):
                    di = n_idx - nbase
                    g_elem = (((di * d + (in_t - base_t)) * h + in_h) * width + in_w) * c + cc
                else:
                    g_elem = (((n_idx * d + in_t) * h + in_h) * width + in_w) * c + cc
            g_elem_i = fx.Int32(g_elem)
            return arith.select(valid, g_elem_i, fx.Int32(x_num_records))

        # ---- conv A G2S: deposit im2col chunks into the GEMM half-block LDS layout ----
        # Physical LDS byte P = wave_id*1024 + step*(N_WAVES*1024) + lane*16 equals the
        # plain flatten row_g*128 + col_g of the logical (row_g, col_g) this lane owns;
        # the element stored there must be A[swizzle_128(row_g, col_g)] so the GEMM's
        # S2RLoader (which reads swizzle_128(row_s, col_s)) round-trips. Mirrors
        # G2SLoader._lds_dst_at exactly (with N_WAVES=4, N_LDS_STEPS_A=4).
        g2s_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        LdsPtr_t = fx.PointerType.get(f8_ir_t, 2, 512)

        # STRIP_IM2COL: replace the per-element im2col address (div/mod decomposition +
        # padding validity mask) with a plain linear (M=npq, K=crs) row-major GEMM read.
        # This is INCORRECT output but isolates the pure-GEMM cost: (full - stripped) =
        # the im2col A-gather overhead. Diagnostic only (env CONV_STRIP_IM2COL=1).
        STRIP_IM2COL = os.environ.get("CONV_STRIP_IM2COL", "0") in ("1", "true", "yes")

        def conv_a_g2s_one(lds_dst, half, k_iter, step):
            m_half_base = m_offset + fx.Index(half * LDS_BLOCK_M)
            k_base = fx.Index(k_iter * BLOCK_K)
            row_g = lane_id // 8 + wave_id * 8 + step * (N_WAVES * 8)
            col_g = (lane_id % 8) * 16
            r, cc = swizzle_128(row_g, col_g)
            m_row = m_half_base + r
            k_col = k_base + cc
            if const_expr(STRIP_IM2COL):
                lin = m_row * crs + k_col
                safe = fx.Int32(arith.select(m_row < fx.Index(npq), lin, fx.Index(x_num_records)))
            else:
                safe = im2col_safe_elem(m_row, k_col)
            step_off = wave_id * 1024 + step * (N_WAVES * 1024)
            base_i32 = fx.Int32(fx.ptrtoint(lds_dst.ptr)) + fx.Int32(step_off)
            lds_ptr = fx.inttoptr(LdsPtr_t, base_i32)
            dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
            src = fx.slice(x_div, (None, fx.Int32(safe)))
            fx.copy(g2s_atom, src, dst)

        def conv_a_g2s(lds_dst, half, k_iter):
            for step in range_constexpr(N_LDS_STEPS_A):
                conv_a_g2s_one(lds_dst, half, k_iter, step)

        # ---- B G2S: plain KTRSC (k, crs) matrix -> reuse the GEMM loader verbatim ----
        gl_off_b = compute_global_swizzle(lane_id, wave_id, crs, N_LDS_ROUNDS, preshuffled=False)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, f8_ir_t, wave_id)
        B0_gl_offset = (block_n * BLOCK_N) * crs
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * crs

        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        # NOTE: the SSA MMA atom, NOT the 4-wave GEMM's AGPR-pinned Mfma16x16x128AGPR.
        # The 4-wave layout gives each of the 256 threads the full 256-accumulator
        # (256x256 tile) fragment set; the conv im2col A-address arithmetic adds enough
        # extra VGPR pressure that the AGPR path's tied `=a,v,v,0` accumulators spill to
        # scratch (verified: scratch_store a[...] in the ISA) and corrupt the output. The
        # 8-wave kernel avoids this because 512 threads halve the per-thread accumulator
        # count. The SSA atom tolerates the spill correctly.
        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        def _compute_lds_swizzle(s2r):
            lds_swz = []
            for row_offset in range_constexpr(s2r.n_tiles):
                row = s2r.wave_idx * (s2r.n_tiles * 16) + row_offset * 16 + lane_id % 16
                swz = []
                for i in range_constexpr(2):
                    col = (lane_id // 16) * 16 + i * 64
                    r, cc = swizzle_128(row, col)
                    swz.append(r * BLOCK_K + cc)
                lds_swz.append(swz)
            return lds_swz

        # Hand-scheduled interleaved cluster: 4x4 AGPR MFMAs interleaved with the 4-step
        # G2S prefetch (via ``g2s_load_one``) and 4x2 S2R fragment loads. Mirrors
        # fp8_gemm_4wave._interleaved_cluster; only the g2s mechanism is abstracted so A
        # loads use the conv im2col loader while B loads reuse the GEMM G2SLoader.
        def _interleaved_cluster(lds_dst, g2s_load_one, s2r, lds_src, a, b, c):
            rt_dst = []

            c[mfma.idx(0, 0)] = mfma.call_one(a, b, c, 0, 0)
            c[mfma.idx(0, 1)] = mfma.call_one(a, b, c, 0, 1)

            lds_swz = _compute_lds_swizzle(s2r)
            g2s_load_one(lds_dst, 0)
            rt_dst_0 = s2r.load_one(lds_src, lds_swz[0][0])

            c[mfma.idx(0, 2)] = mfma.call_one(a, b, c, 0, 2)

            rt_dst_1 = s2r.load_one(lds_src, lds_swz[0][1])
            rt_dst.append(pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c[mfma.idx(0, 3)] = mfma.call_one(a, b, c, 0, 3)

            g2s_load_one(lds_dst, 1)
            rt_dst_0 = s2r.load_one(lds_src, lds_swz[1][0])

            c[mfma.idx(1, 0)] = mfma.call_one(a, b, c, 1, 0)
            c[mfma.idx(1, 1)] = mfma.call_one(a, b, c, 1, 1)

            rt_dst_1 = s2r.load_one(lds_src, lds_swz[1][1])
            rt_dst.append(pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c[mfma.idx(1, 2)] = mfma.call_one(a, b, c, 1, 2)
            c[mfma.idx(1, 3)] = mfma.call_one(a, b, c, 1, 3)

            g2s_load_one(lds_dst, 2)
            rt_dst_0 = s2r.load_one(lds_src, lds_swz[2][0])

            c[mfma.idx(2, 0)] = mfma.call_one(a, b, c, 2, 0)
            c[mfma.idx(2, 1)] = mfma.call_one(a, b, c, 2, 1)

            rt_dst_1 = s2r.load_one(lds_src, lds_swz[2][1])
            rt_dst.append(pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c[mfma.idx(2, 2)] = mfma.call_one(a, b, c, 2, 2)
            c[mfma.idx(2, 3)] = mfma.call_one(a, b, c, 2, 3)

            g2s_load_one(lds_dst, 3)
            rt_dst_0 = s2r.load_one(lds_src, lds_swz[3][0])

            c[mfma.idx(3, 0)] = mfma.call_one(a, b, c, 3, 0)
            c[mfma.idx(3, 1)] = mfma.call_one(a, b, c, 3, 1)

            rt_dst_1 = s2r.load_one(lds_src, lds_swz[3][1])
            rt_dst.append(pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c[mfma.idx(3, 2)] = mfma.call_one(a, b, c, 3, 2)
            c[mfma.idx(3, 3)] = mfma.call_one(a, b, c, 3, 3)

            return c, rt_dst

        def a_g2s_one(half, k_iter):
            def _load(lds_dst, step):
                conv_a_g2s_one(lds_dst, half, k_iter, step)

            return _load

        def b_g2s_one(k_offset):
            def _load(lds_dst, step):
                b_g2s.load_one(lds_dst, k_offset, step)

            return _load

        # ---- prologue: 8-buffer LDS pipeline pre-fill (k=0 and k=1) ----
        conv_a_g2s(a_cur0, 0, 0)
        b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K)
        conv_a_g2s(a_cur1, 1, 0)

        conv_a_g2s(a_next0, 0, 1)
        b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K)
        conv_a_g2s(a_next1, 1, 1)

        wait_barrier((3 * N_LDS_STEPS_A) + (4 * N_LDS_STEPS_B))

        a0_frag = a_s2r.load(a_cur0)

        wait_barrier((3 * N_LDS_STEPS_A) + (3 * N_LDS_STEPS_B))

        b0_frag = b_s2r.load(b_cur0)

        # ---- main loop (mirror kernel_gemm 4-wave, A loads swapped for conv_a_g2s) ----
        for ki in range_constexpr(K_ITERS - 2):
            wait_barrier((2 * N_LDS_STEPS_A) + (2 * N_LDS_STEPS_B))

            c00_frag, b1_frag = _interleaved_cluster(
                a_cur0, a_g2s_one(0, ki + 2), b_s2r, b_cur1, a0_frag, b0_frag, c00_frag
            )

            c01_frag, a1_frag = _interleaved_cluster(
                b_cur0, b_g2s_one(B0_gl_offset + (ki + 2) * BLOCK_K), a_s2r, a_cur1, a0_frag, b1_frag, c01_frag
            )

            wait_barrier((2 * N_LDS_STEPS_A) + (2 * N_LDS_STEPS_B))

            c10_frag, a0_frag = _interleaved_cluster(
                b_cur1, b_g2s_one(B1_gl_offset + (ki + 2) * BLOCK_K), a_s2r, a_next0, a1_frag, b0_frag, c10_frag
            )

            c11_frag, b0_frag = _interleaved_cluster(
                a_cur1, a_g2s_one(1, ki + 2), b_s2r, b_next0, a1_frag, b1_frag, c11_frag
            )

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # ---- tail step k = K_ITERS - 2 ----
        wait_barrier((2 * N_LDS_STEPS_A) + (2 * N_LDS_STEPS_B))
        b1_frag = b_s2r.load(b_cur1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        a1_frag = a_s2r.load(a_cur1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        wait_barrier((1 * N_LDS_STEPS_A) + (1 * N_LDS_STEPS_B))
        a0_frag = a_s2r.load(a_next0)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        b0_frag = b_s2r.load(b_next0)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # ---- tail step k = K_ITERS - 1 ----
        wave_m_offset = wave_m * (N_TILES_A * 16)
        wave_n_offset = wave_n * (N_TILES_B * 16)
        base_row = m_offset + fx.Index(wave_m_offset)
        base_col = block_n * BLOCK_N + fx.Index(wave_n_offset)
        wait_barrier(0)
        b1_frag = b_s2r.load(b_cur1)
        a1_frag = a_s2r.load(a_cur1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

        # ---- epilogue: direct store, map (M=npq row, N=k_out col) -> conv output ----
        if const_expr(BIG_OUT):
            y_elem_base = fx.Int64(buffer_ops.extract_base_index(y))

        def _big_store(off_elem, value):
            addr = y_elem_base + fx.Int64(off_elem) * fx.Int64(2)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=1)
            v = value.ir_value() if hasattr(value, "ir_value") else value
            llvm.StoreOp(v, ptr, alignment=2)

        # Vectorize the epilogue store when the 4 accumulator rows a lane owns are
        # contiguous in the output: for n==1, off_ncdhw = col*dhw + row and those 4
        # rows (row_base..row_base+3) are consecutive, so one 4xbf16 (dwordx2) store
        # replaces four buffer_store_short. Requires dhw % 4 == 0 and not BIG_OUT.
        _vec_store = (n == 1) and (dhw % 4 == 0) and (not BIG_OUT)

        def store_cfrag(c_frag, base_row, base_col):
            for ti in range_constexpr(N_TILES_A):
                for tj in range_constexpr(N_TILES_B):
                    col = base_col + fx.Index(tj * 16) + lane_id % 16
                    col_valid = col < fx.Index(k)
                    if const_expr(has_bias):
                        col_i = fx.Int32(arith.select(col_valid, col, fx.Index(0)))
                        bias_val = fx.Float32(buffer_ops.buffer_load(bias_rsrc, col_i, vec_width=1, dtype=fx.Float32))
                    vec_f32 = Vec(c_frag[mfma.idx(ti, tj)])
                    row_base = base_row + fx.Index(ti * 16) + (lane_id // 16) * 4

                    if const_expr(_vec_store):
                        off0 = col * dhw + row_base
                        row_ok = col_valid & (row_base + fx.Index(3) < fx.Index(npq))
                        if row_ok:
                            vals = []
                            for i in range_constexpr(4):
                                o = vec_f32[i] + bias_val if const_expr(has_bias) else vec_f32[i]
                                vals.append(o.to(fx.BFloat16))
                            v4 = fx.Vector.from_elements(vals, dtype=fx.BFloat16)
                            buffer_ops.buffer_store(v4, y_rsrc, off0)
                        continue

                    for i in range_constexpr(4):
                        row = row_base + fx.Index(i)
                        out = vec_f32[i]
                        if const_expr(has_bias):
                            out = out + bias_val
                        valid = col_valid & (row < fx.Index(npq))
                        if const_expr(n == 1):
                            off_ncdhw = col * dhw + row
                        else:
                            n_idx = row // dhw
                            sp = row % dhw
                            off_ncdhw = n_idx * (k * dhw) + col * dhw + sp
                        if const_expr(BIG_OUT):
                            if valid:
                                _big_store(off_ncdhw, out.to(fx.BFloat16))
                        else:
                            buffer_ops.buffer_store(out.to(fx.BFloat16), y_rsrc, off_ncdhw, mask=valid)

        store_cfrag(c00_frag, base_row, base_col)
        store_cfrag(c01_frag, base_row, base_col + fx.Index(LDS_BLOCK_N))
        store_cfrag(c10_frag, base_row + fx.Index(LDS_BLOCK_M), base_col)
        store_cfrag(c11_frag, base_row + fx.Index(LDS_BLOCK_M), base_col + fx.Index(LDS_BLOCK_N))

    @flyc.jit
    def launch(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        conv3d_fp8_gemm4w_kernel(
            y,
            x,
            weight,
            bias,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}",
            },
        ).launch(grid=(grid_x, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def _conv3d_impl_fp8_gemm4w(x, weight, bias=None, stride=1, padding=0, stream=None):
    n, c, d, h, width = x.shape
    k, wc, kt, kh, kw = weight.shape
    assert c == wc, f"in-channel mismatch: x has {c}, weight has {wc}"
    assert (
        x.dtype == torch.float8_e4m3fn and weight.dtype == torch.float8_e4m3fn
    ), f"expected FP8 E4M3FN x/weight, got x={x.dtype}, weight={weight.dtype}"
    st, sh, sw = _normalize_3(stride)
    pt, ph, pw = _normalize_3(padding)

    do = (d + 2 * pt - kt) // st + 1
    ho = (h + 2 * ph - kh) // sh + 1
    wo = (width + 2 * pw - kw) // sw + 1

    launch_stream = torch.cuda.current_stream() if stream is None else stream
    x_arg = _transpose_activation_fp8(x)
    w_arg = _prep_weight_fp8(weight)

    has_bias = bias is not None
    bias_arg = (
        bias.to(device=x.device, dtype=torch.float32).contiguous().view(-1)
        if has_bias
        else torch.empty(1, device=x.device, dtype=torch.float32)
    )
    if has_bias:
        assert bias_arg.numel() == k, f"bias must have {k} elements, got {bias_arg.numel()}"

    y = torch.empty((n, k, do, ho, wo), device=x.device, dtype=torch.bfloat16)
    exe = compile_conv3d_implicit_fp8_gemm4w(n, c, d, h, width, k, kt, kh, kw, st, sh, sw, pt, ph, pw, has_bias)
    exe(
        flyc.from_torch_tensor(y.view(-1)),
        flyc.from_torch_tensor(x_arg),
        flyc.from_torch_tensor(w_arg),
        flyc.from_torch_tensor(bias_arg),
        launch_stream,
    )
    return y


def _conv2d_impl_fp8_gemm4w(x, weight, bias=None, stride=1, padding=0, **kwargs):
    assert x.dim() == 4 and weight.dim() == 4, "conv2d fp8 expects (N,C,H,W) / (K,C,R,S)"
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (padding, padding) if isinstance(padding, int) else padding
    n, c, hh, ww = x.shape
    k, wc, r, s = weight.shape
    x5 = x.reshape(n, c, 1, hh, ww)
    w5 = weight.reshape(k, wc, 1, r, s)
    y5 = _conv3d_impl_fp8_gemm4w(x5, w5, bias=bias, stride=(1, sh, sw), padding=(0, ph, pw), **kwargs)
    return y5.reshape(y5.shape[0], y5.shape[1], y5.shape[3], y5.shape[4])


def _conv1d_impl_fp8_gemm4w(x, weight, bias=None, stride=1, padding=0, **kwargs):
    assert x.dim() == 3 and weight.dim() == 3, "conv1d fp8 expects (N,C,W) / (K,C,S)"
    sw = stride if isinstance(stride, int) else stride[0]
    pw = padding if isinstance(padding, int) else padding[0]
    n, c, ww = x.shape
    k, wc, s = weight.shape
    x5 = x.reshape(n, c, 1, 1, ww)
    w5 = weight.reshape(k, wc, 1, 1, s)
    y5 = _conv3d_impl_fp8_gemm4w(x5, w5, bias=bias, stride=(1, 1, sw), padding=(0, 0, pw), **kwargs)
    return y5.reshape(y5.shape[0], y5.shape[1], y5.shape[4])


def conv3d_implicit_fp8_gemm4w(x, weight, bias=None, stride=1, padding=0, **kwargs):
    """FP8 implicit-GEMM conv (fp8_gemm 4-wave pipeline); dispatches 1D/2D/3D by filter rank.

    x/weight are FP8 E4M3FN. Requires the rigid 4-wave constraints (crs%128==0,
    k%256==0, c%16==0); raises AssertionError otherwise. Returns bf16.
    """
    assert x.dim() == weight.dim(), f"x rank {x.dim()} != weight rank {weight.dim()}"
    spatial_rank = weight.dim() - 2
    if spatial_rank == 3:
        return _conv3d_impl_fp8_gemm4w(x, weight, bias=bias, stride=stride, padding=padding, **kwargs)
    if spatial_rank == 2:
        return _conv2d_impl_fp8_gemm4w(x, weight, bias=bias, stride=stride, padding=padding, **kwargs)
    if spatial_rank == 1:
        return _conv1d_impl_fp8_gemm4w(x, weight, bias=bias, stride=stride, padding=padding, **kwargs)
    raise ValueError(f"conv3d_implicit_fp8_gemm4w supports 1D/2D/3D; got filter rank {weight.dim()}")


__all__ = ["conv3d_implicit_fp8_gemm4w", "compile_conv3d_implicit_fp8_gemm4w"]
