# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Implicit-GEMM conv3d (FP8, CDNA4) using the fp8_gemm 8-wave pipeline.

This is a port of ``kernels/gemm/fp8_gemm_8wave.py``'s hand-scheduled 8-wave FP8
matmul pipeline (512 threads, 2x4 wave grid, 8 LDS half-block ping-pong buffers,
``Mfma16x16x128`` + ``s_setprio`` + direct-store epilogue) into a conv3d kernel.
The ONLY conv-specific part is the A-operand loader: instead of the GEMM's linear
(M,K) global read, it computes the im2col activation address per LDS chunk and
deposits it into the exact swizzled half-block LDS layout the GEMM's ``S2RLoader``
reads. The B (weight) operand is a plain KTRSC ``(k, crs)`` matrix, so the GEMM's
``G2SLoader`` is reused verbatim.

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
from kernels.gemm.fp8_gemm_utils import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    swizzle_128,
    wait_barrier,
)

# Rigid 8-wave GEMM design: 512 threads, BLOCK_M=BLOCK_N=256, BLOCK_K=128.
BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 128
WARP_SIZE = 64
N_WAVES = 8
BLOCK_THREADS = N_WAVES * WARP_SIZE  # 512

LDS_BLOCK_M = BLOCK_M // 2  # 128
LDS_BLOCK_N = BLOCK_N // 2  # 128
N_TILES_A = BLOCK_M // 64  # 4
N_TILES_B = BLOCK_N // 128  # 2
N_ACCUMS = N_TILES_A * N_TILES_B  # 8
N_LDS_STEPS_A = LDS_BLOCK_M // 64  # 2
N_LDS_STEPS_B = LDS_BLOCK_N // 64  # 2
N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)  # 2

# ---- Tiled fp8 NCDHW->NDHWC transpose (byte-native, no cast) ----
# Replaces the slow torch.permute pre-pass (~882 GB/s) with a coalesced tiled-LDS
# transpose (multi-TB/s). Input is already fp8 (int8-storage); pure memory reorder.
TR_TILE = 64
TR_VEC = 16  # 16 fp8 bytes per vectorized load/store
TR_THREADS = 256
TR_VPL = TR_TILE // TR_VEC  # vecs per LDS row
TR_ITERS = (TR_TILE * TR_TILE) // (TR_VEC * TR_THREADS)
TR_PAD = 16
TR_LDS_S = TR_TILE + TR_PAD


@functools.lru_cache(maxsize=64)
def compile_transpose_ncdhw_ndhwc_fp8(n, c, s):
    """Transpose flat fp8 (N, C, S) -> (N, S, C), S == D*H*W. Requires c%16==0, s%16==0.

    Byte-native (int8-storage of fp8): read coalesced along contiguous S into LDS,
    then read LDS transposed and store coalesced along C. No dtype conversion.
    """
    assert c % TR_VEC == 0, f"fp8 transpose needs C % {TR_VEC} == 0, got C={c}"
    assert s % TR_VEC == 0, f"fp8 transpose needs S % {TR_VEC} == 0, got S={s}"
    total_bytes = n * c * s
    grid_s = (s + TR_TILE - 1) // TR_TILE
    grid_c = (c + TR_TILE - 1) // TR_TILE
    u8 = fx.Uint8

    @flyc.kernel(known_block_size=[TR_THREADS, 1, 1])
    def transpose_kernel(out: fx.Tensor, inp: fx.Tensor):
        in_rsrc = buffer_ops.create_buffer_resource(inp, max_size=False, num_records_bytes=total_bytes)
        out_rsrc = buffer_ops.create_buffer_resource(out, max_size=False, num_records_bytes=total_bytes)
        lds_alloc = fx.SharedAllocator(static=False)
        lds = lds_alloc.allocate(fx.Array[u8, TR_TILE * TR_LDS_S, 16]).peek()

        tid = fx.thread_idx.x
        s0 = fx.block_idx.x * TR_TILE
        c0 = fx.block_idx.y * TR_TILE
        nb = fx.block_idx.z
        in_base = nb * c * s
        out_base = nb * s * c

        # 16 fp8 bytes = 4xi32; v16i8 buffer_load/store isn't a legal backend vector
        # width, so move 16-byte chunks as dwordx4 (i32x4) and per-byte for the LDS
        # transpose gather.
        def lds_store_i32x4(elem_offset, value_i32x4):
            base = fx.Int64(fx.ptrtoint(lds.ptr)) + fx.Int64(elem_offset)
            ptr = buffer_ops.create_llvm_ptr(base, address_space=3)
            llvm.StoreOp(value_i32x4, ptr, alignment=16)

        def lds_load_scalar(elem_offset):
            u8p = fx.recast_iter(u8, lds.ptr)
            return fx.ptr_load(u8p + fx.Int32(elem_offset), result_type=u8)

        # Read coalesced along contiguous S from NCDHW into LDS[c_local][s_local].
        for i in range_constexpr(TR_ITERS):
            lin = tid + i * TR_THREADS
            rc = lin // TR_VPL
            sv = (lin % TR_VPL) * TR_VEC
            cc = c0 + rc
            ss = s0 + sv
            valid = (cc < fx.Index(c)) & (ss < fx.Index(s))
            # buffer_load offset is in ELEMENTS of dtype (i32). The byte offset
            # (in_base + cc*s + ss) is 16-byte aligned (ss multiple of 16, s%16==0),
            # so /4 gives the int32-element offset for a dwordx4 (16B) load.
            g = fx.Int32((in_base + cc * s + ss) // 4)
            safe = arith.select(valid, g, fx.Int32(0))
            v = buffer_ops.buffer_load(in_rsrc, safe, vec_width=4, dtype=fx.Int32)  # dwordx4 = 16B
            lds_store_i32x4(rc * TR_LDS_S + sv, v.ir_value() if hasattr(v, "ir_value") else v)

        llvm.InlineAsmOp(None, [], "s_waitcnt lgkmcnt(0)\n\ts_barrier", "", has_side_effects=True)

        # Read LDS transposed (per-byte gather across strided channels), store 16
        # contiguous channels per S along C as dwordx4.
        for i in range_constexpr(TR_ITERS):
            lin = tid + i * TR_THREADS
            rs = lin // TR_VPL
            cv = (lin % TR_VPL) * TR_VEC
            ss = s0 + rs
            cc = c0 + cv
            valid = (ss < fx.Index(s)) & (cc < fx.Index(c))
            if valid:
                scalars = [lds_load_scalar((cv + j) * TR_LDS_S + rs) for j in range_constexpr(TR_VEC)]
                packed_u8 = fx.Vector.from_elements(scalars, dtype=u8)
                packed = packed_u8.bitcast(fx.Int32)  # v16u8 -> v4i32
                byte_off = out_base + ss * c + cc
                buffer_ops.buffer_store(packed, out_rsrc, byte_off, offset_is_bytes=True)

    @flyc.jit
    def launch(out: fx.Tensor, inp: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        transpose_kernel(out, inp).launch(grid=(grid_s, grid_c, n), block=(TR_THREADS, 1, 1), stream=stream)

    return launch


def _transpose_activation_fp8(x_fp8):
    """Fast tiled NCDHW->NDHWC fp8 transpose; falls back to torch for odd shapes."""
    n, c, d, h, w = x_fp8.shape
    s = d * h * w
    if not (x_fp8.is_contiguous() and c % TR_VEC == 0 and s % TR_VEC == 0):
        return x_fp8.permute(0, 2, 3, 4, 1).contiguous().view(torch.int8).view(-1)
    out = torch.empty((n * s * c,), device=x_fp8.device, dtype=torch.int8)
    exe = compile_transpose_ncdhw_ndhwc_fp8(n, c, s)
    exe(
        flyc.from_torch_tensor(out),
        flyc.from_torch_tensor(x_fp8.view(torch.int8).view(-1)),
        torch.cuda.current_stream(),
    )
    return out


FP8_BYTES = 1


@functools.lru_cache(maxsize=256)
def compile_conv3d_implicit_fp8_gemm8w(
    n, c, d, h, width, k, kt, kh, kw, st, sh, sw, pt, ph, pw, has_bias=False, wgm=1
):
    WGM = max(1, int(wgm))
    do = (d + 2 * pt - kt) // st + 1
    ho = (h + 2 * ph - kh) // sh + 1
    wo = (width + 2 * pw - kw) // sw + 1
    dhw = do * ho * wo
    hw_o = ho * wo
    npq = n * dhw
    crs = c * kt * kh * kw
    K_ITERS = crs // BLOCK_K

    assert c % 16 == 0, f"FP8 needs C % 16 == 0, got C={c}"
    assert crs % BLOCK_K == 0, f"gemm8w needs crs % 128 == 0 (aligned K), got crs={crs}"
    assert k % BLOCK_N == 0, f"gemm8w needs k % 256 == 0 (BLOCK_N), got k={k}"
    assert K_ITERS >= 2, f"gemm8w needs crs//128 >= 2, got {K_ITERS}"

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

    # Per-axis compile-time flag: is a padding bounds-check needed on the input coord?
    # A degenerate axis (kernel size 1, no pad, stride 1, out==in) can never go
    # out of bounds, so its in_range check (2 cmp + an AND -> a v_cndmask) is pure
    # waste. Skipping it in the hot im2col path trims VALU/cndmask pressure.
    need_t_check = not (kt == 1 and pt == 0 and st == 1 and do == d)
    need_h_check = not (kh == 1 and ph == 0 and sh == 1 and ho == h)
    need_w_check = not (kw == 1 and pw == 0 and sw == 1 and wo == width)

    grid_m = (npq + BLOCK_M - 1) // BLOCK_M
    grid_n = k // BLOCK_N

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
    def conv3d_fp8_gemm8w_kernel(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor):
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
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        if const_expr(WGM > 1):
            # Grouped-M workgroup L2-swizzle: visit WGM consecutive m-tiles across all
            # n-tiles before advancing, so the weight (B) tile stays hot in L2 across
            # the group. Mirrors the bf16 conv3d_implicit kernel's wgm remap; targets
            # narrow-N shapes (few n-tiles) where the default row-major grid evicts B.
            pid = fx.Index(fx.block_idx.x) + fx.Index(fx.block_idx.y) * fx.Index(grid_m)
            blocks_per_group = fx.Index(WGM * grid_n)
            group_id = pid // blocks_per_group
            first_m = group_id * fx.Index(WGM)
            group_rows = fx.Index(grid_m) - first_m
            group_rows = fx.Index(arith.select(group_rows < fx.Index(WGM), group_rows, fx.Index(WGM)))
            local = pid % blocks_per_group
            block_m = fx.Index(first_m + (local % group_rows))
            block_n = fx.Index(local // group_rows)
        else:
            block_m = fx.block_idx.x
            block_n = fx.block_idx.y

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
        #
        # PERF: the output-spatial decomposition of m_row (n_idx, ot, oh, ow) is
        # loop-INVARIANT across the K loop (m_row has no k dependence), yet it costs
        # 4 div/mods. Precompute it once per (half, step) via _spatial_of_row and reuse
        # it every k-iter; the k-loop body then only decodes the tap (kt/kh/kw from
        # k_col, cheap since c is large so k_col//c changes slowly) and combines. This
        # lifts the dominant VALU cost out of the hot loop (MfmaUtil 60% -> higher).
        def _spatial_of_row(m_row):
            row_valid = m_row < fx.Index(npq)
            if const_expr(temporal_only_fast):
                out_t = (m_row // hw_o) % d
                return (row_valid, out_t, m_row)
            n_idx = m_row // dhw
            rem = m_row % dhw
            ot = rem // hw_o
            rem2 = rem % hw_o
            oh = rem2 // wo
            ow = rem2 % wo
            return (row_valid, n_idx, ot, oh, ow)

        def im2col_safe_elem_pre(spatial, k_col):
            cc = k_col % c
            if const_expr(temporal_only_fast):
                row_valid, out_t, m_row = spatial
                kt_i = k_col // c
                temporal_delta = kt_i - pt
                in_t = out_t + temporal_delta
                valid = row_valid & in_range(in_t, d)
                if const_expr(BIG_IN):
                    g_elem = ((m_row + temporal_delta * hw_o) - (nbase * dhw + base_t * hw_o)) * c + cc
                else:
                    g_elem = (m_row + temporal_delta * hw_o) * c + cc
            else:
                row_valid, n_idx, ot, oh, ow = spatial
                ckk = k_col // c
                kw_i = ckk % kw
                ckk2 = ckk // kw
                kh_i = ckk2 % kh
                kt_i = ckk2 // kh
                in_t = ot * st + kt_i - pt
                in_h = oh * sh + kh_i - ph
                in_w = ow * sw + kw_i - pw
                valid = row_valid
                if const_expr(need_t_check):
                    valid = valid & in_range(in_t, d)
                if const_expr(need_h_check):
                    valid = valid & in_range(in_h, h)
                if const_expr(need_w_check):
                    valid = valid & in_range(in_w, width)
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
        # G2SLoader._lds_dst_at exactly.
        g2s_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        LdsPtr_t = fx.PointerType.get(f8_ir_t, 2, 512)

        # STRIP_IM2COL: replace the per-element im2col address (div/mod decomposition +
        # padding validity mask) with a plain linear (M=npq, K=crs) row-major GEMM read.
        # This is INCORRECT output but isolates the pure-GEMM cost: (full - stripped) =
        # the im2col A-gather overhead. Diagnostic only (env CONV_STRIP_IM2COL=1).
        STRIP_IM2COL = os.environ.get("CONV_STRIP_IM2COL", "0") in ("1", "true", "yes")

        # Per-(half, step) loop-invariant data: the LDS byte offset (cc, step_off) and
        # the output-spatial decomposition of m_row. Computed ONCE here (not per k-iter);
        # the SSA values dominate every conv_a_g2s call (first use is in the prologue).
        def _conv_a_g2s_pre(half):
            m_half_base = m_offset + fx.Index(half * LDS_BLOCK_M)
            steps = []
            for step in range_constexpr(N_LDS_STEPS_A):
                row_g = lane_id // 8 + wave_id * 8 + step * (N_WAVES * 8)
                col_g = (lane_id % 8) * 16
                r, cc = swizzle_128(row_g, col_g)
                m_row = m_half_base + r
                step_off = wave_id * 1024 + step * (N_WAVES * 1024)
                if const_expr(STRIP_IM2COL):
                    steps.append((step_off, cc, m_row, None))
                else:
                    steps.append((step_off, cc, m_row, _spatial_of_row(m_row)))
            return steps

        _a_g2s_pre = [_conv_a_g2s_pre(0), _conv_a_g2s_pre(1)]

        def conv_a_g2s(lds_dst, half, k_iter):
            k_base = fx.Index(k_iter * BLOCK_K)
            for step_off, cc, m_row, spatial in _a_g2s_pre[half]:
                if const_expr(STRIP_IM2COL):
                    k_col = k_base + cc
                    lin = m_row * crs + k_col
                    safe = fx.Int32(arith.select(m_row < fx.Index(npq), lin, fx.Index(x_num_records)))
                else:
                    safe = im2col_safe_elem_pre(spatial, k_base + cc)
                base_i32 = fx.Int32(fx.ptrtoint(lds_dst.ptr)) + fx.Int32(step_off)
                lds_ptr = fx.inttoptr(LdsPtr_t, base_i32)
                dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                src = fx.slice(x_div, (None, fx.Int32(safe)))
                fx.copy(g2s_atom, src, dst)

        # ---- B G2S: plain KTRSC (k, crs) matrix -> reuse the GEMM loader verbatim ----
        gl_off_b = compute_global_swizzle(lane_id, wave_id, crs, N_LDS_ROUNDS, preshuffled=False)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, f8_ir_t, wave_id)
        B0_gl_offset = (block_n * BLOCK_N) * crs
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * crs

        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        # ---- prologue: fill cur (k=0) and next (k=1) ----
        b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K)
        conv_a_g2s(a_cur0, 0, 0)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K)
        conv_a_g2s(a_cur1, 1, 0)

        if wave_m == 1:
            fx.rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K)
        conv_a_g2s(a_next0, 0, 1)
        b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        # ---- main loop (mirror kernel_gemm, A loads swapped for conv_a_g2s) ----
        for ki in range_constexpr(K_ITERS - 2):
            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            conv_a_g2s(a_next1, 1, ki + 1)
            fx.rocdl.s_barrier()

            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + (ki + 2) * BLOCK_K)
            fx.rocdl.s_barrier()

            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)

            a1_frag = a_s2r.load(a_cur1)
            conv_a_g2s(a_cur0, 0, ki + 2)
            fx.rocdl.s_barrier()

            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)

            b_g2s.load(b_cur1, B1_gl_offset + (ki + 2) * BLOCK_K)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # ---- step k = K_ITERS - 2 ----
        b0_frag = b_s2r.load(b_cur0)
        a0_frag = a_s2r.load(a_cur0)
        fx.rocdl.s_barrier()

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

        b1_frag = b_s2r.load(b_cur1)
        fx.rocdl.s_barrier()

        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)

        a1_frag = a_s2r.load(a_cur1)
        conv_a_g2s(a_next1, 1, K_ITERS - 1)
        fx.rocdl.s_barrier()

        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)

        b0_frag = b_s2r.load(b_next0)
        fx.rocdl.s_barrier()

        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # ---- step k = K_ITERS - 1 ----
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier(0)

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

        b1_frag = b_s2r.load(b_cur1)
        fx.rocdl.s_barrier()

        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)

        a1_frag = a_s2r.load(a_cur1)
        fx.rocdl.s_barrier()

        fx.rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, set_prio=False)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, set_prio=False)
        fx.rocdl.s_setprio(0)
        fx.rocdl.s_barrier()

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
        # replaces four buffer_store_short. Requires dhw % 4 == 0 (so the whole npq
        # row range this wave writes stays 4-row-group aligned -> no partial group)
        # and not BIG_OUT. ATT showed the scalar store was 35.8% of stall on shallow-K.
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

        wave_m_offset = wave_m * (N_TILES_A * 16)
        wave_n_offset = wave_n * (N_TILES_B * 16)
        base_row = m_offset + fx.Index(wave_m_offset)
        base_col = block_n * BLOCK_N + fx.Index(wave_n_offset)

        store_cfrag(c00_frag, base_row, base_col)
        store_cfrag(c01_frag, base_row, base_col + fx.Index(LDS_BLOCK_N))
        store_cfrag(c10_frag, base_row + fx.Index(LDS_BLOCK_M), base_col)
        store_cfrag(c11_frag, base_row + fx.Index(LDS_BLOCK_M), base_col + fx.Index(LDS_BLOCK_N))

    @flyc.jit
    def launch(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        conv3d_fp8_gemm8w_kernel(
            y,
            x,
            weight,
            bias,
            value_attrs={
                "rocdl.waves_per_eu": 2,
                "rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}",
            },
        ).launch(grid=(grid_m, grid_n, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def _conv3d_impl_fp8_gemm8w(x, weight, bias=None, stride=1, padding=0, stream=None, wgm=8):
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
    exe = compile_conv3d_implicit_fp8_gemm8w(n, c, d, h, width, k, kt, kh, kw, st, sh, sw, pt, ph, pw, has_bias, wgm)
    exe(
        flyc.from_torch_tensor(y.view(-1)),
        flyc.from_torch_tensor(x_arg),
        flyc.from_torch_tensor(w_arg),
        flyc.from_torch_tensor(bias_arg),
        launch_stream,
    )
    return y


def _conv2d_impl_fp8_gemm8w(x, weight, bias=None, stride=1, padding=0, **kwargs):
    assert x.dim() == 4 and weight.dim() == 4, "conv2d fp8 expects (N,C,H,W) / (K,C,R,S)"
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (padding, padding) if isinstance(padding, int) else padding
    n, c, hh, ww = x.shape
    k, wc, r, s = weight.shape
    x5 = x.reshape(n, c, 1, hh, ww)
    w5 = weight.reshape(k, wc, 1, r, s)
    y5 = _conv3d_impl_fp8_gemm8w(x5, w5, bias=bias, stride=(1, sh, sw), padding=(0, ph, pw), **kwargs)
    return y5.reshape(y5.shape[0], y5.shape[1], y5.shape[3], y5.shape[4])


def _conv1d_impl_fp8_gemm8w(x, weight, bias=None, stride=1, padding=0, **kwargs):
    assert x.dim() == 3 and weight.dim() == 3, "conv1d fp8 expects (N,C,W) / (K,C,S)"
    sw = stride if isinstance(stride, int) else stride[0]
    pw = padding if isinstance(padding, int) else padding[0]
    n, c, ww = x.shape
    k, wc, s = weight.shape
    x5 = x.reshape(n, c, 1, 1, ww)
    w5 = weight.reshape(k, wc, 1, 1, s)
    y5 = _conv3d_impl_fp8_gemm8w(x5, w5, bias=bias, stride=(1, 1, sw), padding=(0, 0, pw), **kwargs)
    return y5.reshape(y5.shape[0], y5.shape[1], y5.shape[4])


def conv3d_implicit_fp8_gemm8w(x, weight, bias=None, stride=1, padding=0, **kwargs):
    """FP8 implicit-GEMM conv (fp8_gemm 8-wave pipeline); dispatches 1D/2D/3D by filter rank.

    x/weight are FP8 E4M3FN. Requires the rigid 8-wave constraints (crs%128==0,
    k%256==0, c%16==0); raises AssertionError otherwise. Returns bf16.
    """
    assert x.dim() == weight.dim(), f"x rank {x.dim()} != weight rank {weight.dim()}"
    spatial_rank = weight.dim() - 2
    if spatial_rank == 3:
        return _conv3d_impl_fp8_gemm8w(x, weight, bias=bias, stride=stride, padding=padding, **kwargs)
    if spatial_rank == 2:
        return _conv2d_impl_fp8_gemm8w(x, weight, bias=bias, stride=stride, padding=padding, **kwargs)
    if spatial_rank == 1:
        return _conv1d_impl_fp8_gemm8w(x, weight, bias=bias, stride=stride, padding=padding, **kwargs)
    raise ValueError(f"conv3d_implicit_fp8_gemm8w supports 1D/2D/3D; got filter rank {weight.dim()}")


__all__ = ["conv3d_implicit_fp8_gemm8w", "compile_conv3d_implicit_fp8_gemm8w"]
