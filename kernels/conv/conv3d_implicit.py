# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""8-wave double-buffered implicit-GEMM conv3d (BF16).

x: (N, C, D, H, W) bf16 NCDHW, weight: (K, C, T, R, S) bf16 KCTRS.
Returns (N, K, Do, Ho, Wo) bf16. Supports stride, padding, bias, and split-K.
"""

import functools
import os
import weakref

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

# TILE_K is pinned to the MFMA k-dim (mfma_f32_16x16x32_bf16 -> 32). The tile
# size (TILE_M/TILE_N) and wave layout (WAVE_M/WAVE_N) are compile-time
# parameters of compile_conv3d_implicit_8wave (autotuned per shape).
TILE_K = 32
STAGES = 2
WARP_SIZE = 64

MFMA_M = 16
MFMA_N = 16
MFMA_A_VALUES = 8
MFMA_B_VALUES = 8
MFMA_C_VALUES = 4

LDG_VEC = 8

# gfx950 (CDNA4) LDS capacity; this kernel needs the CDNA4 bf16 MFMA anyway.
LDS_CAPACITY_BYTES = 163840
BF16_BYTES = 2

# Default tile config = the original hand-tuned 8-wave 128x128 shape.
DEFAULT_TILE = (128, 128, 2, 4)


def _autotune_enabled():
    return os.environ.get("FLYDSL_CONV3D_AUTOTUNE", "0").lower() in ("1", "true", "yes")


_WEIGHT_CACHE = {}


def _prep_weight(w, k, kt, kh, kw, c):
    key = id(w)
    ent = _WEIGHT_CACHE.get(key)
    if ent is not None and ent[0]() is w:
        return ent[1]
    wk = w.permute(0, 2, 3, 4, 1).contiguous().reshape(k, kt * kh * kw * c)
    _WEIGHT_CACHE[key] = (weakref.ref(w), wk)
    return wk


TR_TILE = 64
TR_VEC = 8
TR_THREADS = 256
_TR_VPL = TR_TILE // TR_VEC
_TR_ITERS = (TR_TILE * TR_TILE) // (TR_VEC * TR_THREADS)
_TR_PAD = 8
_TR_LDS_S = TR_TILE + _TR_PAD


@functools.lru_cache(maxsize=64)
def compile_transpose_ncdhw_ndhwc(n, c, s):
    """Transpose flat (N, C, S) -> (N, S, C) (S == T*H*W). Requires c%8==0, s%8==0."""
    grid_s = (s + TR_TILE - 1) // TR_TILE
    grid_c = (c + TR_TILE - 1) // TR_TILE
    elem_ty = fx.BFloat16
    BIG = (n * c * s) > 0x7FFFFFFF

    @flyc.kernel(known_block_size=[TR_THREADS, 1, 1])
    def transpose_kernel(out: fx.Tensor, inp: fx.Tensor):
        in_rsrc = buffer_ops.create_buffer_resource(inp)
        out_rsrc = buffer_ops.create_buffer_resource(out)
        lds_alloc = fx.SharedAllocator(static=False)
        lds = lds_alloc.allocate(fx.Array[elem_ty, TR_TILE * _TR_LDS_S, 16]).peek()

        Vec = fx.Vector

        class Vec8Ty:
            ir_type = Vec.make_type(TR_VEC, elem_ty)

        class BF16Ty:
            ir_type = elem_ty.ir_type

        tid = fx.thread_idx.x
        s0 = fx.block_idx.x * TR_TILE
        c0 = fx.block_idx.y * TR_TILE
        nb = fx.block_idx.z
        if const_expr(BIG):
            in_base_elem = fx.Index(nb) * fx.Index(c) * fx.Index(s) + fx.Index(c0) * fx.Index(s) + fx.Index(s0)
            in_addr = fx.Int64(buffer_ops.extract_base_index(inp)) + fx.Int64(in_base_elem) * fx.Int64(2)
            in_rsrc = buffer_ops.create_buffer_resource_from_addr(in_addr)
            out_base_elem = fx.Index(nb) * fx.Index(s) * fx.Index(c) + fx.Index(s0) * fx.Index(c) + fx.Index(c0)
            out_addr = fx.Int64(buffer_ops.extract_base_index(out)) + fx.Int64(out_base_elem) * fx.Int64(2)
            out_rsrc = buffer_ops.create_buffer_resource_from_addr(out_addr)
        else:
            in_base = nb * c * s
            out_base = nb * s * c

        def lds_store_vec8(elem_offset, value):
            base = fx.Int64(fx.ptrtoint(lds.ptr)) + fx.Int64(elem_offset * 2)
            ptr = buffer_ops.create_llvm_ptr(base, address_space=3)
            llvm.StoreOp(value, ptr, alignment=16)

        def lds_load_scalar(elem_offset):
            u8 = fx.recast_iter(fx.Uint8, lds.ptr)
            return fx.ptr_load(u8 + fx.Int32(elem_offset * 2), result_type=BF16Ty)

        # Read: coalesced vec8 along contiguous S -> LDS[c_local][s_local].
        for i in range_constexpr(_TR_ITERS):
            lin = tid + i * TR_THREADS
            rc = lin // _TR_VPL
            sv = (lin % _TR_VPL) * TR_VEC
            cc = c0 + rc
            ss = s0 + sv
            valid = (cc < c) & (ss < s)
            if const_expr(BIG):
                g = fx.Int32(rc * s + sv)
            else:
                g = fx.Int32(in_base + cc * s + ss)
            safe = arith.select(valid, g, fx.Int32(0))
            v = buffer_ops.buffer_load(in_rsrc, safe, vec_width=TR_VEC, dtype=elem_ty)
            lds_store_vec8(rc * _TR_LDS_S + sv, v)

        llvm.InlineAsmOp(None, [], "s_waitcnt lgkmcnt(0)\n\ts_barrier", "", has_side_effects=True)

        for i in range_constexpr(_TR_ITERS):
            lin = tid + i * TR_THREADS
            rs = lin // _TR_VPL
            cv = (lin % _TR_VPL) * TR_VEC
            ss = s0 + rs
            cc = c0 + cv
            scalars = [lds_load_scalar((cv + j) * _TR_LDS_S + rs) for j in range_constexpr(TR_VEC)]
            vv = fx.Vector.from_elements(scalars, dtype=elem_ty)
            valid = (ss < s) & (cc < c)
            if valid:
                if const_expr(BIG):
                    go = fx.Int32(rs * c + cv)
                else:
                    go = fx.Int32(out_base + ss * c + cc)
                buffer_ops.buffer_store(vv, out_rsrc, go)

    @flyc.jit
    def launch_transpose(out: fx.Tensor, inp: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        transpose_kernel(out, inp).launch(
            grid=(grid_s, grid_c, n),
            block=(TR_THREADS, 1, 1),
            stream=stream,
        )

    return launch_transpose


def _ncdhw_to_ndhwc(x, stream):
    """Fast NCDHW->NDHWC via the tiled transpose kernel; falls back to torch."""
    n, c, t, h, w = x.shape
    s = t * h * w
    if not (x.is_contiguous() and x.dtype == torch.bfloat16 and c % 8 == 0 and s % 8 == 0):
        return x.permute(0, 2, 3, 4, 1).contiguous()
    out = torch.empty((n, t, h, w, c), device=x.device, dtype=x.dtype)
    exe = compile_transpose_ncdhw_ndhwc(n, c, s)
    exe(out, x, torch.cuda.current_stream() if stream is None else stream)
    return out


@functools.lru_cache(maxsize=256)
def compile_conv3d_implicit_8wave(
    n, c, d, h, w, k, kt, kh, kw, st, sh, sw, pt, ph, pw, has_bias=False, splitk=1, tile=DEFAULT_TILE
):
    TILE_M, TILE_N, WAVE_M, WAVE_N = tile
    BLOCK_THREADS = WAVE_M * WAVE_N * WARP_SIZE
    # Per-wave MFMA grid (flat acc[mi * MI_N + ni]); WARP_M/N is the per-wave tile span.
    MI_M = TILE_M // WAVE_M // MFMA_M
    MI_N = TILE_N // WAVE_N // MFMA_N
    N_ACC = MI_M * MI_N
    WARP_M = MI_M * MFMA_M
    WARP_N = MI_N * MFMA_N
    BLOCK_VECS = LDG_VEC * BLOCK_THREADS
    LDG_A_COUNT = TILE_M * TILE_K // BLOCK_VECS
    LDG_B_COUNT = TILE_N * TILE_K // BLOCK_VECS

    assert TILE_K == 32
    assert TILE_M % (WAVE_M * MFMA_M) == 0, f"TILE_M={TILE_M} not divisible by WAVE_M*16"
    assert TILE_N % (WAVE_N * MFMA_N) == 0, f"TILE_N={TILE_N} not divisible by WAVE_N*16"
    assert (TILE_M * TILE_K) % BLOCK_VECS == 0, f"A tile {TILE_M}x{TILE_K} not a multiple of {BLOCK_VECS} vecs"
    assert (TILE_N * TILE_K) % BLOCK_VECS == 0, f"B tile {TILE_N}x{TILE_K} not a multiple of {BLOCK_VECS} vecs"
    assert LDG_A_COUNT >= 1 and LDG_B_COUNT >= 1
    assert c % LDG_VEC == 0
    assert BLOCK_THREADS <= 1024, f"BLOCK_THREADS={BLOCK_THREADS} exceeds 1024"

    do = (d + 2 * pt - kt) // st + 1
    ho = (h + 2 * ph - kh) // sh + 1
    wo = (w + 2 * pw - kw) // sw + 1
    dhw = do * ho * wo
    hw_o = ho * wo
    npq = n * dhw
    crs = c * kt * kh * kw
    k_tiles = (crs + TILE_K - 1) // TILE_K

    BIG_IN = (n * c * d * h * w) > 0x7FFFFFFF
    # Output element count; when its byte offset can exceed int32 the epilogue
    # buffer_store (i32 voffset) overflows, so store via 64-bit global pointers.
    BIG_OUT = (n * k * do * ho * wo * BF16_BYTES) > 0x7FFFFFFF

    # Async copy (global->LDS DMA via buffer_load_lds) hides the load latency the
    # sync global->VGPR->LDS path exposes. Padding/OOB taps are masked by routing
    # the byte offset past num_records so the hardware bounds check writes 0 to LDS
    # (verified). The buffer voffset is UNSIGNED 32-bit (probed), so any offset that
    # fits under 2^32 bytes is valid -- no 64-bit soffset split needed.
    #
    # Two regimes:
    #  - not BIG_IN: the raw x resource with real num_records = X_BYTES. Padding taps
    #    route to OOB_SENTINEL (> X_BYTES, < 2^32) -> hardware zero. Needs X_BYTES <
    #    OOB_SENTINEL so the sentinel stays above bounds.
    #  - BIG_IN (n==1 only): x is rebased to the block's (nbase, base_t) origin so the
    #    per-tile relative offsets are tiny (<~0.3GB, verified). The rebased resource
    #    gets a fixed num_records = BIG_ASYNC_NR (2GB): well above any legal relative
    #    tap yet below OOB_SENTINEL, so padding still zeroes. n>1 is excluded because
    #    di=(n_idx-nbase) can jump a whole batch and blow past 2^32.
    # The sync global->VGPR->LDS path is kept as the fallback for the residual cases
    # async cannot cover: n>1 BIG_IN and >~4.29GB weights. Do not delete it.
    X_BYTES = n * c * d * h * w * BF16_BYTES
    W_BYTES = k * c * kt * kh * kw * BF16_BYTES
    OOB_SENTINEL_ELEM = 0x7FFFFF80  # *2 = 0xFFFFFF00 bytes (~4.2950 GB), just under 2^32
    OOB_SENTINEL_BYTES = OOB_SENTINEL_ELEM * BF16_BYTES
    BIG_ASYNC_NR = 0x80000000  # 2 GB num_records for the rebased BIG_IN resource
    _small_ok = (X_BYTES < OOB_SENTINEL_BYTES) and (W_BYTES < OOB_SENTINEL_BYTES)
    _big_ok = BIG_IN and (n == 1) and (W_BYTES < OOB_SENTINEL_BYTES)
    USE_ASYNC = _small_ok if not BIG_IN else _big_ok
    # Async frees the A-tile VGPRs the sync path spent on the global->VGPR->LDS hop,
    # so the software pipeline can go deeper than the sync 2-stage double buffer.
    # PIPE_STAGES buffers are kept in LDS; PIPE_STAGES-1 tiles are prefetched ahead.
    # Depth 4 is the measured sweet spot on gfx950 (256x256x4x4): depths 2/3 don't
    # amortize the DMA issue overhead, depth 5 hits the 160KB LDS cap (occupancy 1).
    ASYNC_STAGES = 4
    PIPE_STAGES = ASYNC_STAGES if USE_ASYNC else STAGES

    LDS_A_SIZE = PIPE_STAGES * TILE_M * TILE_K
    LDS_B_SIZE = PIPE_STAGES * TILE_N * TILE_K
    lds_bytes = PIPE_STAGES * (TILE_M + TILE_N) * TILE_K * BF16_BYTES
    assert lds_bytes <= LDS_CAPACITY_BYTES, f"LDS {lds_bytes} exceeds {LDS_CAPACITY_BYTES}"

    n_tail = k % TILE_N != 0
    grid_n = (k + TILE_N - 1) // TILE_N

    splitk = max(1, min(splitk, k_tiles))
    tiles_per_split = k_tiles // splitk
    use_splitk = splitk > 1

    grid_m = (npq + TILE_M - 1) // TILE_M
    elem_ty = fx.BFloat16
    mfma_fn = rocdl.mfma_f32_16x16x32_bf16
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
        and wo == w
    )

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def conv3d_8wave_kernel(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor):
        if const_expr(USE_ASYNC):
            # Real num_records so OOB-routed padding taps read back as 0 from the
            # hardware bounds check (see async_a/async_b masking). The x resource is
            # (re)built below: raw+X_BYTES for the small case, rebased+BIG_ASYNC_NR
            # for BIG_IN.
            w_rsrc = buffer_ops.create_buffer_resource(weight, num_records_bytes=W_BYTES)
            if const_expr(not BIG_IN):
                x_rsrc = buffer_ops.create_buffer_resource(x, num_records_bytes=X_BYTES)
        else:
            x_rsrc = buffer_ops.create_buffer_resource(x)
            w_rsrc = buffer_ops.create_buffer_resource(weight)
        y_rsrc = buffer_ops.create_buffer_resource(y)
        if const_expr(has_bias):
            bias_rsrc = buffer_ops.create_buffer_resource(bias)

        lds_alloc = fx.SharedAllocator(static=False)
        a_lds = lds_alloc.allocate(fx.Array[elem_ty, LDS_A_SIZE, 16]).peek()
        b_lds = lds_alloc.allocate(fx.Array[elem_ty, LDS_B_SIZE, 16]).peek()

        tid = fx.thread_idx.x
        m_offset = fx.block_idx.x * TILE_M
        n_offset = fx.block_idx.y * TILE_N
        if const_expr(use_splitk):
            k_off = fx.block_idx.z * (tiles_per_split * TILE_K)
        else:
            k_off = 0

        if const_expr(BIG_IN):
            nbase = m_offset // dhw
            ot_base0 = (m_offset % dhw) // hw_o
            base_t = ot_base0 - fx.Index(pt)
            base_t = arith.select(base_t < fx.Index(0), fx.Index(0), base_t)
            x_base_elem = ((nbase * fx.Index(d) + base_t) * fx.Index(h) + fx.Index(0)) * fx.Index(w) * fx.Index(c)
            x_addr = fx.Int64(buffer_ops.extract_base_index(x)) + fx.Int64(x_base_elem) * fx.Int64(2)
            if const_expr(USE_ASYNC):
                # Bounded num_records so async OOB-routed padding taps zero: legal
                # per-tile relative offsets are <~0.3GB << BIG_ASYNC_NR (2GB) <
                # OOB_SENTINEL, so valid taps stay in-bounds and padding zeroes.
                x_rsrc = buffer_ops.create_buffer_resource_from_addr(x_addr, num_records_bytes=BIG_ASYNC_NR)
            else:
                x_rsrc = buffer_ops.create_buffer_resource_from_addr(x_addr)

        wid = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        wave_m = wid // WAVE_N
        wave_n = wid % WAVE_N

        lane_m = lane % MFMA_M
        lane_n = lane % MFMA_N
        lane_k_a = lane // MFMA_M * MFMA_A_VALUES
        lane_k_b = lane // MFMA_N * MFMA_B_VALUES
        c_m_vec = lane // MFMA_N * MFMA_C_VALUES
        c_n = lane % MFMA_N

        Vec = fx.Vector

        class Vec8Ty:
            ir_type = Vec.make_type(8, elem_ty)

        acc0 = Vec.filled(MFMA_C_VALUES, 0.0, fx.Float32)
        acc = [acc0 for _ in range_constexpr(N_ACC)]

        zero8 = Vec.filled(8, 0.0, elem_ty)

        def barrier(vmcnt=0, lgkmcnt=None):
            waits = []
            if vmcnt is not None:
                waits.append(f"vmcnt({vmcnt})")
            if lgkmcnt is not None:
                waits.append(f"lgkmcnt({lgkmcnt})")
            pre = ("s_waitcnt " + " ".join(waits) + "\n\t") if waits else ""
            llvm.InlineAsmOp(None, [], f"{pre}s_barrier", "", has_side_effects=True)

        def lds_ptr_at(lds_array, byte_offset):
            lds_base = fx.Int64(fx.ptrtoint(lds_array.ptr)) + fx.Int64(byte_offset)
            return buffer_ops.create_llvm_ptr(lds_base, address_space=3)

        def lds_store_vec8(lds_array, elem_offset, value):
            llvm.StoreOp(value, lds_ptr_at(lds_array, elem_offset * 2), alignment=16)

        def lds_load_vec8(lds_array, elem_offset):
            u8_ptr = fx.recast_iter(fx.Uint8, lds_array.ptr)
            return fx.ptr_load(u8_ptr + fx.Int32(elem_offset * 2), result_type=Vec8Ty)

        def a_lds_off(stage, row, col):
            return (fx.Index(stage) * TILE_M + row) * TILE_K + col

        def b_lds_off(stage, row, col):
            return (fx.Index(stage) * TILE_N + row) * TILE_K + col

        def in_range(v, hi):
            return (v >= 0) & (v < fx.Index(hi))

        # ---- Per-thread row decomposition (loop-invariant across K) ----
        # gather_a() is called once per K-tile in the software pipeline, but the
        # A-row a thread owns depends only on tid, not on k_base. Decompose the
        # row into (n_idx, ot/out_t, oh, ow) ONCE here and reuse every K-tile;
        # only the channel term cc and tap indices vary with k_base below.
        _row_dec = []  # per-i tuple of precomputed row terms
        for i in range_constexpr(LDG_A_COUNT):
            linear = (tid + i * BLOCK_THREADS) * LDG_VEC
            local_m = linear // TILE_K
            local_k = linear % TILE_K  # 0 (LDG_VEC==TILE_K) — kept for generality
            row = m_offset + local_m
            row_valid = row < fx.Index(npq)
            if const_expr(temporal_only_fast):
                out_t = (row // hw_o) % d
                _row_dec.append((local_k, row, row_valid, out_t))
            else:
                n_idx = row // dhw
                rem = row % dhw
                ot = rem // hw_o
                rem2 = rem % hw_o
                oh = rem2 // wo
                ow = rem2 % wo
                in_t0 = ot * st - pt
                in_h0 = oh * sh - ph
                in_w0 = ow * sw - pw
                if const_expr(BIG_IN):
                    di = n_idx - nbase
                    _row_dec.append((local_k, row_valid, di, in_t0, in_h0, in_w0))
                else:
                    _row_dec.append((local_k, row_valid, n_idx, in_t0, in_h0, in_w0))

        # When c is a multiple of TILE_K, one k-tile of TILE_K contiguous k_abs
        # stays within a single channel group, so the tap index (k_abs//c) and
        # channel base (k_base%c) are UNIFORM across threads: derive them once from
        # k_base via scalar SALU and use cc = cc_base + local_k (no wrap), replacing
        # per-thread integer div/mod by c. Falls back to per-thread when c%TILE_K!=0
        # (e.g. c=16), where a k-tile can straddle two channel groups.
        SCALAR_K = (c % TILE_K == 0)

        # ---- 3D im2col address math (shared by sync gather + async DMA) ----
        # Returns (g_off_i32_elem, valid) for A-tile load slot i at K-base k_base.
        def _a_addr(i, kbase_i, cc_base, ckk_base):
            dec = _row_dec[i]
            local_k = dec[0]
            k_abs = kbase_i + fx.Index(local_k)
            if const_expr(SCALAR_K):
                cc = cc_base + fx.Index(local_k)
            else:
                cc = k_abs % c
            k_valid = k_abs < fx.Index(crs)
            if const_expr(temporal_only_fast):
                _, row, row_valid, out_t = dec
                kt_i = ckk_base if const_expr(SCALAR_K) else k_abs // c
                temporal_delta = kt_i - pt
                in_t = out_t + temporal_delta
                valid = row_valid & k_valid & in_range(in_t, d)
                if const_expr(BIG_IN):
                    g_off = ((row + temporal_delta * hw_o) - (fx.Index(nbase) * dhw + base_t * hw_o)) * c + cc
                else:
                    g_off = (row + temporal_delta * hw_o) * c + cc
            else:
                ckk = ckk_base if const_expr(SCALAR_K) else k_abs // c
                kw_i = ckk % kw
                ckk2 = ckk // kw
                kh_i = ckk2 % kh
                kt_i = ckk2 // kh
                if const_expr(BIG_IN):
                    _, row_valid, di, in_t0, in_h0, in_w0 = dec
                    in_t = in_t0 + kt_i
                    in_h = in_h0 + kh_i
                    in_w = in_w0 + kw_i
                    valid = row_valid & k_valid & in_range(in_t, d) & in_range(in_h, h) & in_range(in_w, w)
                    g_off = (((di * d + (in_t - base_t)) * h + in_h) * w + in_w) * c + cc
                else:
                    _, row_valid, n_idx, in_t0, in_h0, in_w0 = dec
                    in_t = in_t0 + kt_i
                    in_h = in_h0 + kh_i
                    in_w = in_w0 + kw_i
                    valid = row_valid & k_valid & in_range(in_t, d) & in_range(in_h, h) & in_range(in_w, w)
                    g_off = (((n_idx * d + in_t) * h + in_h) * w + in_w) * c + cc
            return fx.Int32(g_off), valid

        def _b_addr(i, k_base):
            linear = (tid + i * BLOCK_THREADS) * LDG_VEC
            local_n = linear // TILE_K
            local_k = linear % TILE_K
            col = n_offset + fx.Index(local_n)
            g_off = fx.Int32(col * crs + (fx.Index(k_base) + fx.Index(local_k)))
            col_valid = (col < fx.Index(k)) if const_expr(n_tail) else None
            return g_off, col_valid

        # ---- 3D im2col gather (global -> registers) ----
        def gather_a(k_base):
            kbase_i = fx.Index(k_base)
            cc_base = ckk_base = None
            if const_expr(SCALAR_K):
                cc_base = kbase_i % c
                ckk_base = kbase_i // c
            raws = []
            valids = []
            for i in range_constexpr(LDG_A_COUNT):
                g_off_i, valid = _a_addr(i, kbase_i, cc_base, ckk_base)
                safe = arith.select(valid, g_off_i, fx.Int32(0))
                raw = buffer_ops.buffer_load(x_rsrc, safe, vec_width=8, dtype=elem_ty)
                raws.append(raw)
                valids.append(valid)
            return raws + valids

        def gather_b(k_base):
            raws = []
            valids = []
            for i in range_constexpr(LDG_B_COUNT):
                g_off, col_valid = _b_addr(i, k_base)
                if const_expr(n_tail):
                    safe = arith.select(col_valid, g_off, fx.Int32(0))
                    raw = buffer_ops.buffer_load(w_rsrc, safe, vec_width=8, dtype=elem_ty)
                    valids.append(col_valid)
                else:
                    raw = buffer_ops.buffer_load(w_rsrc, g_off, vec_width=8, dtype=elem_ty)
                raws.append(raw)
            return raws + valids

        def commit_a(stage, values):
            raws = list(values[:LDG_A_COUNT])
            valids = list(values[LDG_A_COUNT:])
            for i in range_constexpr(LDG_A_COUNT):
                linear = (tid + i * BLOCK_THREADS) * LDG_VEC
                local_m = linear // TILE_K
                local_k = linear % TILE_K
                raw = raws[i]
                valid = valids[i]
                val = arith.select(valid, raw, zero8)
                off = local_m * TILE_K + local_k
                lds_store_vec8(a_lds, fx.Index(stage) * TILE_M * TILE_K + off, val)

        def commit_b(stage, values):
            raws = list(values[:LDG_B_COUNT])
            if const_expr(n_tail):
                valids = list(values[LDG_B_COUNT:])
            for i in range_constexpr(LDG_B_COUNT):
                linear = (tid + i * BLOCK_THREADS) * LDG_VEC
                local_n = linear // TILE_K
                local_k = linear % TILE_K
                raw = raws[i]
                val = arith.select(valids[i], raw, zero8) if const_expr(n_tail) else raw
                off = local_n * TILE_K + local_k
                lds_store_vec8(b_lds, fx.Index(stage) * TILE_N * TILE_K + off, val)

        # ---- async copy (global -> LDS DMA), masking via OOB routing ----
        # buffer_load_lds is wave-collective: one uniform LDS base (readfirstlane)
        # and the hardware spreads lanes by lane*DMA_BYTES. conv's A/B LDS layout is
        # already thread-contiguous (thread i owns byte i*16), so the per-slot base
        # is `stage_base + i*BLOCK_THREADS*16` and lane fan-out lands each thread's
        # 16B where the sync commit_* path would. Invalid taps route the element
        # offset past num_records; the bounds check then writes 0 to LDS (verified).
        DMA_BYTES = LDG_VEC * BF16_BYTES  # 16
        OOB_ELEM = fx.Int32(OOB_SENTINEL_ELEM)  # element offset guaranteed past num_records

        def _lds_dma_ptr(lds_array, stage_tile, i):
            # buffer_load_lds is wave-collective: the LDS pointer is lane-0's base
            # and the hardware adds lane*DMA_BYTES per lane. Compute the SAME
            # per-thread element offset commit_* uses ((tid+i*BT)*LDG_VEC, contiguous
            # so consecutive lanes are DMA_BYTES apart), then readfirstlane picks
            # each wave's lane-0 base; the hardware lane spread rebuilds the layout.
            off_elems = fx.Index(stage_tile) + (fx.Index(tid) + fx.Index(i * BLOCK_THREADS)) * fx.Index(LDG_VEC)
            base_bytes = off_elems * fx.Index(BF16_BYTES)
            addr = fx.Int64(fx.ptrtoint(lds_array.ptr)) + fx.Int64(base_bytes)
            addr = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, addr.ir_value()))
            return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), addr)

        def _dma_to_lds(rsrc, lds_ptr, voff_elem):
            # voff_elem: DSL Int32 element offset; byte offset folds through *2.
            voff_b = (voff_elem * fx.Int32(BF16_BYTES)).ir_value()
            rocdl.raw_ptr_buffer_load_lds(
                rsrc,
                lds_ptr,
                arith.constant(DMA_BYTES, type=T.i32),
                voff_b,
                arith.constant(0, type=T.i32),
                arith.constant(0, type=T.i32),
                arith.constant(0, type=T.i32),
            )

        def _async_a(stage, k_base):
            kbase_i = fx.Index(k_base)
            cc_base = ckk_base = None
            if const_expr(SCALAR_K):
                cc_base = kbase_i % c
                ckk_base = kbase_i // c
            stage_tile = fx.Index(stage) * TILE_M * TILE_K
            for i in range_constexpr(LDG_A_COUNT):
                g_off_i, valid = _a_addr(i, kbase_i, cc_base, ckk_base)
                voff = fx.Int32(arith.select(valid, g_off_i, OOB_ELEM))
                _dma_to_lds(x_rsrc, _lds_dma_ptr(a_lds, stage_tile, i), voff)

        def _async_b(stage, k_base):
            stage_tile = fx.Index(stage) * TILE_N * TILE_K
            for i in range_constexpr(LDG_B_COUNT):
                g_off, col_valid = _b_addr(i, k_base)
                if const_expr(n_tail):
                    voff = fx.Int32(arith.select(col_valid, g_off, OOB_ELEM))
                else:
                    voff = g_off
                _dma_to_lds(w_rsrc, _lds_dma_ptr(b_lds, stage_tile, i), voff)

        # ---- single-vec ds_read (LDS -> register), indexed by per-wave MFMA row ----
        def read_a_vec(stage, mi):
            a_row = wave_m * WARP_M + mi * MFMA_M + lane_m
            return lds_load_vec8(a_lds, a_lds_off(stage, fx.Index(a_row), fx.Index(lane_k_a)))

        def read_b_vec(stage, ni):
            b_row = wave_n * WARP_N + ni * MFMA_N + lane_n
            return lds_load_vec8(b_lds, b_lds_off(stage, fx.Index(b_row), fx.Index(lane_k_b)))

        def mfma_one(a_frag, b_frag, c_frag):
            out = mfma_fn(
                T.vec(MFMA_C_VALUES, T.f32),
                [a_frag, b_frag, c_frag, 0, 0, 0],
            )
            rocdl.sched_mfma(1)
            return out

        def read_a_frags(stage):
            frags = [read_a_vec(stage, mi) for mi in range_constexpr(MI_M)]
            rocdl.sched_dsrd(MI_M)
            return frags

        def read_b_frags(stage):
            frags = [read_b_vec(stage, ni) for ni in range_constexpr(MI_N)]
            rocdl.sched_dsrd(MI_N)
            return frags

        def do_compute(acc_values, a_frag_values, b_frag_values):
            rocdl.s_setprio(1)
            for mi in range_constexpr(MI_M):
                for ni in range_constexpr(MI_N):
                    idx = mi * MI_N + ni
                    acc_values[idx] = mfma_one(a_frag_values[mi], b_frag_values[ni], acc_values[idx])
            rocdl.s_setprio(0)
            return acc_values

        if const_expr(USE_ASYNC):
            # Async global->LDS software pipeline (PIPE_STAGES deep): each DMA lands
            # straight in LDS (no VGPR prefetch state). Prologue issues the first
            # PIPE_STAGES-1 tiles' DMAs; each loop iter waits only the oldest
            # in-flight tile (vmcnt = remaining in-flight), reads it, launches the
            # tile PIPE_STAGES-1 ahead, and computes -- so DMA overlaps MFMA across
            # the full pipeline depth rather than a single double buffer.
            # Note: range_constexpr (full unroll) outperforms scf.for here because
            # the LLVM software pipeliner has cross-iteration visibility of all 576+
            # tile bodies, enabling global ds_read↔MFMA interleaving that beats
            # per-iteration sched hints. The spill overhead (prologue only, not hot path)
            # is smaller than the scheduling gain from full unroll.
            PREFETCH = PIPE_STAGES - 1
            for s in range_constexpr(PREFETCH):
                if const_expr(s < tiles_per_split):
                    _async_a(s, k_off + s * TILE_K)
                    _async_b(s, k_off + s * TILE_K)
            LDG_PER_TILE = LDG_A_COUNT + LDG_B_COUNT
            for kt_idx in range_constexpr(tiles_per_split):
                cur = kt_idx % PIPE_STAGES
                # vmcnt counts outstanding buffer_load_lds INSTRUCTIONS (not tiles);
                # wait until only the still-needed future tiles remain in flight.
                inflight_tiles = min(PREFETCH - 1, tiles_per_split - 1 - kt_idx)
                barrier(vmcnt=inflight_tiles * LDG_PER_TILE, lgkmcnt=0)
                a_frags = read_a_frags(cur)
                b_frags = read_b_frags(cur)
                nxt = kt_idx + PREFETCH
                if const_expr(nxt < tiles_per_split):
                    _async_a(nxt % PIPE_STAGES, k_off + nxt * TILE_K)
                    _async_b(nxt % PIPE_STAGES, k_off + nxt * TILE_K)
                    rocdl.sched_vmem(LDG_A_COUNT + LDG_B_COUNT)
                acc = do_compute(acc, a_frags, b_frags)

        # ---- sync prologue: tile 0 -> LDS, tile 1 -> VGPR prefetch ----
        elif const_expr(tiles_per_split == 1):
            stage = 0
            commit_a(stage, gather_a(k_off))
            commit_b(stage, gather_b(k_off))
            barrier(vmcnt=None, lgkmcnt=0)
            a_frags = read_a_frags(stage)
            b_frags = read_b_frags(stage)
            acc = do_compute(acc, a_frags, b_frags)
        else:
            stage = 0
            commit_a(stage, gather_a(k_off))
            commit_b(stage, gather_b(k_off))
            pf_a = gather_a(k_off + TILE_K)
            pf_b = gather_b(k_off + TILE_K)
            rocdl.sched_vmem(LDG_A_COUNT + LDG_B_COUNT)
            barrier(vmcnt=None, lgkmcnt=0)
            a_frags = read_a_frags(stage)
            b_frags = read_b_frags(stage)

            n_acc_state = N_ACC
            n_a_frag_state = MI_M
            n_b_frag_state = MI_N
            n_pf_a_state = 2 * LDG_A_COUNT

            init_state = list(acc) + list(a_frags) + list(b_frags) + list(pf_a) + list(pf_b)

            # ---- main loop: compute tile k, write prefetched k+1, load k+2 ----
            for kt_idx, state_values in range(0, tiles_per_split - 2, init=init_state):
                state_values = list(state_values)
                state_acc = list(state_values[:n_acc_state])
                pos = n_acc_state
                state_a = list(state_values[pos : pos + n_a_frag_state])
                pos += n_a_frag_state
                state_b = list(state_values[pos : pos + n_b_frag_state])
                pos += n_b_frag_state
                state_pf_a = list(state_values[pos : pos + n_pf_a_state])
                pos += n_pf_a_state
                state_pf_b = list(state_values[pos:])

                next_stage = (kt_idx + 1) % STAGES
                commit_a(next_stage, state_pf_a)
                commit_b(next_stage, state_pf_b)
                rocdl.sched_dswr(LDG_A_COUNT + LDG_B_COUNT)

                next_pf_a = gather_a(k_off + (kt_idx + 2) * TILE_K)
                next_pf_b = gather_b(k_off + (kt_idx + 2) * TILE_K)
                rocdl.sched_vmem(LDG_A_COUNT + LDG_B_COUNT)

                state_acc = do_compute(state_acc, state_a, state_b)
                barrier(vmcnt=None, lgkmcnt=0)
                next_a = read_a_frags(next_stage)
                next_b = read_b_frags(next_stage)

                results = yield (list(state_acc) + list(next_a) + list(next_b) + list(next_pf_a) + list(next_pf_b))

            results = list(results)
            acc = list(results[:n_acc_state])
            pos = n_acc_state
            a_frags = list(results[pos : pos + n_a_frag_state])
            pos += n_a_frag_state
            b_frags = list(results[pos : pos + n_b_frag_state])
            pos += n_b_frag_state
            pf_a = list(results[pos : pos + n_pf_a_state])
            pos += n_pf_a_state
            pf_b = list(results[pos:])

            final_stage = (tiles_per_split - 1) % STAGES
            commit_a(final_stage, pf_a)
            commit_b(final_stage, pf_b)
            rocdl.sched_dswr(LDG_A_COUNT + LDG_B_COUNT)

            # Compute the penultimate tile while the final tile enters LDS.
            acc = do_compute(acc, a_frags, b_frags)
            barrier(vmcnt=None, lgkmcnt=0)
            a_frags = read_a_frags(final_stage)
            b_frags = read_b_frags(final_stage)
            acc = do_compute(acc, a_frags, b_frags)

        _row_chk = npq % TILE_M != 0
        _need_chk = _row_chk or n_tail
        _vec_store = (n == 1) and (not use_splitk) and (dhw % MFMA_C_VALUES == 0) and (not BIG_OUT)

        # For >2^31-byte outputs the i32 buffer_store voffset overflows; store via
        # a 64-bit global pointer built from the full element offset instead.
        if const_expr(BIG_OUT):
            y_elem_base = fx.Int64(buffer_ops.extract_base_index(y))

        def _big_store(off_nk_i64, value):
            addr = y_elem_base + off_nk_i64 * fx.Int64(BF16_BYTES)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=1)
            llvm.StoreOp(value.ir_value() if hasattr(value, "ir_value") else value, ptr, alignment=2)

        def _valid_raw(row, col):
            if const_expr(_row_chk and n_tail):
                return arith.andi(row < fx.Index(npq), col < fx.Index(k))
            if const_expr(_row_chk):
                v = row < fx.Index(npq)
                return arith.andi(v, v)
            v = col < fx.Index(k)
            return arith.andi(v, v)

        def store_acc():
            for mi in range_constexpr(MI_M):
                row_base = m_offset + wave_m * WARP_M + mi * MFMA_M + c_m_vec
                for ni in range_constexpr(MI_N):
                    col = n_offset + fx.Index(wave_n * WARP_N + ni * MFMA_N + c_n)
                    a = Vec(acc[mi * MI_N + ni])
                    if const_expr(has_bias and not use_splitk):
                        col_i = fx.Int32(col)
                        if const_expr(n_tail):
                            col_i = arith.select(col < fx.Index(k), col_i, fx.Int32(0))
                        bias_val = fx.Float32(buffer_ops.buffer_load(bias_rsrc, col_i, vec_width=1, dtype=fx.Float32))

                    if const_expr(_vec_store):
                        row0 = fx.Index(row_base)
                        off_nk0 = col * dhw + row0

                        def _emit_vec():
                            vals = []
                            for i in range_constexpr(MFMA_C_VALUES):
                                cval = (a[i] + bias_val) if const_expr(has_bias) else a[i]
                                vals.append(cval.to(elem_ty))
                            v4 = fx.Vector.from_elements(vals, dtype=elem_ty)
                            buffer_ops.buffer_store(v4, y_rsrc, off_nk0)

                        if const_expr(_need_chk):
                            if _valid_raw(row0, col):
                                _emit_vec()
                        else:
                            _emit_vec()
                        continue

                    for i in range_constexpr(MFMA_C_VALUES):
                        row = fx.Index(row_base + i)
                        off_sk = row * k + col

                        if const_expr(n == 1):
                            off_nk = col * dhw + row
                        else:
                            n_idx = row // dhw
                            sp = row % dhw
                            off_nk = n_idx * (k * dhw) + col * dhw + sp

                        def _emit():
                            if const_expr(use_splitk):
                                off_b = fx.Int32(off_sk * 4)
                                z0 = fx.Int32(0)
                                rocdl.raw_ptr_buffer_atomic_fadd(a[i], y_rsrc, off_b, z0, z0)
                            else:
                                cval = (a[i] + bias_val).to(elem_ty) if const_expr(has_bias) else a[i].to(elem_ty)
                                if const_expr(BIG_OUT):
                                    _big_store(fx.Int64(off_nk), cval)
                                else:
                                    buffer_ops.buffer_store(cval, y_rsrc, off_nk)

                        if const_expr(_need_chk):
                            if _valid_raw(row, col):
                                _emit()
                        else:
                            _emit()

        store_acc()

    @flyc.jit
    def launch(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        conv3d_8wave_kernel(y, x, weight, bias).launch(
            grid=(grid_m, grid_n, splitk), block=(BLOCK_THREADS, 1, 1), stream=stream
        )

    return launch


def _resolve_splitk(splitk, npq, crs, k, device, tile=DEFAULT_TILE):
    # splitk=None auto-picks a value that roughly fills the CUs; an explicit value
    # is just clamped. Either way the result is snapped to a k_tiles divisor.
    k_tiles = (crs + TILE_K - 1) // TILE_K
    if splitk is None:
        tile_m, tile_n = tile[0], tile[1]
        base = ((npq + tile_m - 1) // tile_m) * ((k + tile_n - 1) // tile_n)
        if (
            npq < 4096
            or k_tiles < 16
            or k % tile_n != 0  # atomic path needs clean tiles
            or npq % tile_m != 0
            or crs % TILE_K != 0
            or npq * k * 4 > 0x7FFFFFFF  # split-K fp32 output atomic uses an i32 byte offset
        ):
            sk = 1
        else:
            try:
                num_cu = torch.cuda.get_device_properties(device).multi_processor_count
            except Exception:
                num_cu = 256
            if base >= (3 * num_cu) // 4:  # base grid already (nearly) fills the machine
                sk = 1
            else:
                sk = min(4, max(1, num_cu // base), k_tiles)  # aim to roughly fill the CUs
    else:
        sk = max(1, splitk)
    while sk > 1 and k_tiles % sk != 0:  # prefer a divisor (no overhang)
        sk -= 1
    return sk


def _conv3d_impl(x, weight, bias=None, stride=1, padding=0, splitk=None, stream=None, tile=None, autotune=None):
    # 3D implicit-GEMM implementation; the public conv3d_implicit_8wave entry
    # dispatches 1D/2D/3D by filter rank and forwards true 3D calls here.
    # x: (N,C,D,H,W) bf16, weight: (K,C,T,R,S) bf16. splitk=None -> auto-dispatch.
    # tile=(TILE_M,TILE_N,WAVE_M,WAVE_N) forces a config; autotune=True picks the
    # best tile per shape (also enabled via FLYDSL_CONV3D_AUTOTUNE=1).
    n, c, d, h, w = x.shape
    k, wc, kt, kh, kw = weight.shape
    assert c == wc
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    st, sh, sw = (stride, stride, stride) if isinstance(stride, int) else stride
    pt, ph, pw = (padding, padding, padding) if isinstance(padding, int) else padding

    # 1x1x1 fast path: y[n,k,dhw] = sum_c weight[k,c] * x[n,c,dhw] — pure channel GEMM.
    # The async im2col path pays an NCDHW→NDHWC transpose (~5-6ms for 14GB tensors)
    # that 1x1x1 never needs since x is already channel-major (dim 1). Routing through
    # torch.matmul (rocBLAS) gives 201 TF vs 78 TF for async im2col — 2.6x faster.
    # FlyDSL has no GEMM kernel for these dims (M=14M, K=48: i32 overflow + N not %128).
    if kt == 1 and kh == 1 and kw == 1 and st == 1 and sh == 1 and sw == 1 and pt == 0 and ph == 0 and pw == 0:
        wm = weight.reshape(k, c)
        if n == 1:
            y = torch.matmul(wm, x.reshape(c, d * h * w)).reshape(n, k, d, h, w)
        else:
            y = torch.matmul(wm, x.reshape(n, c, d * h * w)).reshape(n, k, d, h, w)
        if bias is not None:
            y = y + bias.to(y.dtype).view(1, k, 1, 1, 1)
        return y

    do = (d + 2 * pt - kt) // st + 1
    ho = (h + 2 * ph - kh) // sh + 1
    wo = (w + 2 * pw - kw) // sw + 1
    npq = n * do * ho * wo
    crs = c * kt * kh * kw

    launch_stream = torch.cuda.current_stream() if stream is None else stream
    has_bias = bias is not None
    bias_arg = bias.to(torch.float32).contiguous() if has_bias else torch.empty(1, device=x.device, dtype=torch.float32)

    # Transpose/weight-pack are tile-independent; do them once (also reused across
    # tuning candidates).
    x_ndhwc = _ncdhw_to_ndhwc(x, stream)
    w_packed = _prep_weight(weight, k, kt, kh, kw, c)

    shape = (n, c, d, h, w, k, kt, kh, kw, st, sh, sw, pt, ph, pw, has_bias)

    def _run(the_tile):
        sk = _resolve_splitk(splitk, npq, crs, k, x.device, the_tile)
        if sk > 1:
            y = torch.zeros((npq, k), device=x.device, dtype=torch.float32)
        else:
            y = torch.empty((n, k, do, ho, wo), device=x.device, dtype=torch.bfloat16)
        exe = compile_conv3d_implicit_8wave(
            n, c, d, h, w, k, kt, kh, kw, st, sh, sw, pt, ph, pw, has_bias, sk, the_tile
        )
        exe(y, x_ndhwc, w_packed, bias_arg, launch_stream)
        return y, sk

    if tile is not None:
        chosen = tuple(tile)
    elif autotune or (autotune is None and _autotune_enabled()):
        from kernels.conv.conv3d_autotune import BF16_CANDIDATES, autotune_conv3d

        chosen = autotune_conv3d("bf16", shape, "bf16", BF16_CANDIDATES, x.device, lambda t: _run(t)[0])
    else:
        chosen = DEFAULT_TILE

    y, sk = _run(chosen)
    if sk > 1:
        if has_bias:
            y = y + bias_arg.view(1, k)
        y = y.to(torch.bfloat16)
        return y.view(n, do, ho, wo, k).permute(0, 4, 1, 2, 3)
    return y


def _conv2d_impl(x, weight, bias=None, stride=1, padding=0, **kwargs):
    assert x.dim() == 4 and weight.dim() == 4, "conv2d expects (N,C,H,W) / (K,C,R,S)"
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (padding, padding) if isinstance(padding, int) else padding
    n, c, h, w = x.shape
    k, wc, r, s = weight.shape
    x5 = x.reshape(n, c, 1, h, w)
    w5 = weight.reshape(k, wc, 1, r, s)
    y5 = _conv3d_impl(x5, w5, bias=bias, stride=(1, sh, sw), padding=(0, ph, pw), **kwargs)
    return y5.reshape(y5.shape[0], y5.shape[1], y5.shape[3], y5.shape[4])


def _conv1d_impl(x, weight, bias=None, stride=1, padding=0, **kwargs):
    assert x.dim() == 3 and weight.dim() == 3, "conv1d expects (N,C,W) / (K,C,S)"
    sw = stride if isinstance(stride, int) else stride[0]
    pw = padding if isinstance(padding, int) else padding[0]
    n, c, w = x.shape
    k, wc, s = weight.shape
    x5 = x.reshape(n, c, 1, 1, w)
    w5 = weight.reshape(k, wc, 1, 1, s)
    y5 = _conv3d_impl(x5, w5, bias=bias, stride=(1, 1, sw), padding=(0, 0, pw), **kwargs)
    return y5.reshape(y5.shape[0], y5.shape[1], y5.shape[4])


def conv3d_implicit_8wave(x, weight, bias=None, stride=1, padding=0, **kwargs):
    """Main implicit-GEMM conv entry; dispatches 1D/2D/3D by filter rank.

    Rank is taken from the filter (weight.dim() - 2): 3 -> 3D (N,C,D,H,W)/(K,C,T,R,S),
    2 -> 2D (N,C,H,W)/(K,C,R,S), 1 -> 1D (N,C,W)/(K,C,S); x and weight must match.
    True 3D calls run the implementation directly; 2D/1D reshape to the degenerate
    5D case. stride/padding/bias and extra kwargs (splitk, tile, autotune, stream)
    forward to the chosen path.
    """
    assert x.dim() == weight.dim(), f"x rank {x.dim()} != weight rank {weight.dim()}"
    spatial_rank = weight.dim() - 2
    if spatial_rank == 3:
        return _conv3d_impl(x, weight, bias=bias, stride=stride, padding=padding, **kwargs)
    if spatial_rank == 2:
        return _conv2d_impl(x, weight, bias=bias, stride=stride, padding=padding, **kwargs)
    if spatial_rank == 1:
        return _conv1d_impl(x, weight, bias=bias, stride=stride, padding=padding, **kwargs)
    raise ValueError(f"conv3d_implicit_8wave supports 1D/2D/3D; got filter rank {weight.dim()}")
