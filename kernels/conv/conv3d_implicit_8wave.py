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
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from kernels.common.mem_ops import buffer_atomic_add

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
    LDS_A_SIZE = STAGES * TILE_M * TILE_K
    LDS_B_SIZE = STAGES * TILE_N * TILE_K

    assert TILE_K == 32
    assert TILE_M % (WAVE_M * MFMA_M) == 0, f"TILE_M={TILE_M} not divisible by WAVE_M*16"
    assert TILE_N % (WAVE_N * MFMA_N) == 0, f"TILE_N={TILE_N} not divisible by WAVE_N*16"
    assert (TILE_M * TILE_K) % BLOCK_VECS == 0, f"A tile {TILE_M}x{TILE_K} not a multiple of {BLOCK_VECS} vecs"
    assert (TILE_N * TILE_K) % BLOCK_VECS == 0, f"B tile {TILE_N}x{TILE_K} not a multiple of {BLOCK_VECS} vecs"
    assert LDG_A_COUNT >= 1 and LDG_B_COUNT >= 1
    assert c % LDG_VEC == 0
    assert BLOCK_THREADS <= 1024, f"BLOCK_THREADS={BLOCK_THREADS} exceeds 1024"
    lds_bytes = STAGES * (TILE_M + TILE_N) * TILE_K * BF16_BYTES
    assert lds_bytes <= LDS_CAPACITY_BYTES, f"LDS {lds_bytes} exceeds {LDS_CAPACITY_BYTES}"

    do = (d + 2 * pt - kt) // st + 1
    ho = (h + 2 * ph - kh) // sh + 1
    wo = (w + 2 * pw - kw) // sw + 1
    dhw = do * ho * wo
    hw_o = ho * wo
    npq = n * dhw
    crs = c * kt * kh * kw
    k_tiles = (crs + TILE_K - 1) // TILE_K

    n_tail = k % TILE_N != 0
    grid_n = (k + TILE_N - 1) // TILE_N

    splitk = max(1, min(splitk, k_tiles))
    tiles_per_split = k_tiles // splitk
    use_splitk = splitk > 1

    grid_m = (npq + TILE_M - 1) // TILE_M
    elem_ty = fx.BFloat16
    mfma_fn = rocdl.mfma_f32_16x16x32_bf16

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def conv3d_8wave_kernel(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor):
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

        # ---- 3D im2col gather (global -> registers) ----
        # Each thread loads LDG_A_COUNT vec8 chunks along the flattened (M, K) tile,
        # returning a list of (raw, valid, lds_elem_off) consumed by commit_a.
        def gather_a(k_base):
            out = []
            for i in range_constexpr(LDG_A_COUNT):
                linear = (tid + i * BLOCK_THREADS) * LDG_VEC
                local_m = linear // TILE_K
                local_k = linear % TILE_K
                row = m_offset + local_m
                row_valid = row < fx.Index(npq)
                n_idx = row // dhw
                rem = row % dhw
                ot = rem // hw_o
                rem2 = rem % hw_o
                oh = rem2 // wo
                ow = rem2 % wo
                k_abs = fx.Index(k_base) + fx.Index(local_k)
                cc = k_abs % c
                ckk = k_abs // c
                kw_i = ckk % kw
                ckk2 = ckk // kw
                kh_i = ckk2 % kh
                kt_i = ckk2 // kh
                in_t = ot * st + kt_i - pt
                in_h = oh * sh + kh_i - ph
                in_w = ow * sw + kw_i - pw
                k_valid = k_abs < fx.Index(crs)
                valid = row_valid & k_valid & in_range(in_t, d) & in_range(in_h, h) & in_range(in_w, w)
                g_off = (((n_idx * d + in_t) * h + in_h) * w + in_w) * c + cc
                g_off_i = fx.Int32(g_off)
                safe = arith.select(valid, g_off_i, fx.Int32(0))
                raw = buffer_ops.buffer_load(x_rsrc, safe, vec_width=8, dtype=elem_ty)
                out.append((raw, valid, local_m * TILE_K + local_k))
            return out

        def gather_b(k_base):
            out = []
            for i in range_constexpr(LDG_B_COUNT):
                linear = (tid + i * BLOCK_THREADS) * LDG_VEC
                local_n = linear // TILE_K
                local_k = linear % TILE_K
                col = n_offset + fx.Index(local_n)
                g_off = fx.Int32(col * crs + (fx.Index(k_base) + fx.Index(local_k)))
                if const_expr(n_tail):
                    col_valid = col < fx.Index(k)
                    safe = arith.select(col_valid, g_off, fx.Int32(0))
                    raw = buffer_ops.buffer_load(w_rsrc, safe, vec_width=8, dtype=elem_ty)
                    out.append((raw, col_valid, local_n * TILE_K + local_k))
                else:
                    raw = buffer_ops.buffer_load(w_rsrc, g_off, vec_width=8, dtype=elem_ty)
                    out.append((raw, None, local_n * TILE_K + local_k))
            return out

        def commit_a(stage, vos):
            for raw, valid, off in vos:
                val = arith.select(valid, raw, zero8)  # mask consumed here (hidden behind MFMAs)
                lds_store_vec8(a_lds, fx.Index(stage) * TILE_M * TILE_K + off, val)

        def commit_b(stage, vos):
            for raw, valid, off in vos:
                val = raw if const_expr(valid is None) else arith.select(valid, raw, zero8)
                lds_store_vec8(b_lds, fx.Index(stage) * TILE_N * TILE_K + off, val)

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

        # ---- generic double-buffered pipeline ----
        # prologue: stage 0 committed to LDS, stage 1 prefetched to VGPR.
        stage = 0
        next_stage = 1
        commit_a(stage, gather_a(k_off))
        commit_b(stage, gather_b(k_off))
        if const_expr(tiles_per_split > 1):
            pf_a = gather_a(k_off + TILE_K)
            pf_b = gather_b(k_off + TILE_K)
            rocdl.sched_vmem(LDG_A_COUNT + LDG_B_COUNT)
        barrier(vmcnt=None, lgkmcnt=0)

        a_frags = read_a_frags(stage)
        b_frags = read_b_frags(stage)

        for kt_idx in range_constexpr(tiles_per_split):
            if const_expr(kt_idx + 1 < tiles_per_split):
                commit_a(next_stage, pf_a)
                commit_b(next_stage, pf_b)
                rocdl.sched_dswr(LDG_A_COUNT + LDG_B_COUNT)
                if const_expr(kt_idx + 2 < tiles_per_split):
                    pf_a = gather_a(k_off + (kt_idx + 2) * TILE_K)
                    pf_b = gather_b(k_off + (kt_idx + 2) * TILE_K)
                    rocdl.sched_vmem(LDG_A_COUNT + LDG_B_COUNT)

            rocdl.s_setprio(1)
            for mi in range_constexpr(MI_M):
                for ni in range_constexpr(MI_N):
                    idx = mi * MI_N + ni
                    acc[idx] = mfma_one(a_frags[mi], b_frags[ni], acc[idx])
            rocdl.s_setprio(0)

            if const_expr(kt_idx + 1 < tiles_per_split):
                barrier(vmcnt=None, lgkmcnt=0)
                stage = next_stage
                next_stage = (stage + 1) % STAGES
                a_frags = read_a_frags(stage)
                b_frags = read_b_frags(stage)

        _row_chk = npq % TILE_M != 0
        _need_chk = _row_chk or n_tail

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
                                buffer_atomic_add(a[i], y_rsrc, off_b, z0, z0)
                            else:
                                cval = (a[i] + bias_val).to(elem_ty) if const_expr(has_bias) else a[i].to(elem_ty)
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


def _choose_splitk(npq, crs, k, device, tile=DEFAULT_TILE):
    tile_m, tile_n = tile[0], tile[1]
    grid_m = (npq + tile_m - 1) // tile_m
    grid_n = (k + tile_n - 1) // tile_n
    base = grid_m * grid_n
    k_tiles = (crs + TILE_K - 1) // TILE_K

    if npq < 4096 or k_tiles < 16:
        return 1
    if k % tile_n != 0 or npq % tile_m != 0 or crs % TILE_K != 0:  # atomic path needs clean tiles
        return 1
    try:
        num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        num_cu = 256
    if base >= (3 * num_cu) // 4:  # base grid already (nearly) fills the machine
        return 1
    sk = min(4, max(1, num_cu // base), k_tiles)  # aim to roughly fill the CUs
    while sk > 1 and k_tiles % sk != 0:  # prefer a divisor (no overhang)
        sk -= 1
    return sk


def _resolve_splitk(splitk, npq, crs, k, device, tile):
    sk = _choose_splitk(npq, crs, k, device, tile) if splitk is None else max(1, splitk)
    k_tiles = (crs + TILE_K - 1) // TILE_K
    while sk > 1 and k_tiles % sk != 0:
        sk -= 1
    return sk


def conv3d_implicit_8wave(
    x, weight, bias=None, stride=1, padding=0, splitk=None, stream=None, tile=None, autotune=None
):
    # x: (N,C,D,H,W) bf16, weight: (K,C,T,R,S) bf16. splitk=None -> auto-dispatch.
    # tile=(TILE_M,TILE_N,WAVE_M,WAVE_N) forces a config; autotune=True picks the
    # best tile per shape (also enabled via FLYDSL_CONV3D_AUTOTUNE=1).
    n, c, d, h, w = x.shape
    k, wc, kt, kh, kw = weight.shape
    assert c == wc
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    st, sh, sw = (stride, stride, stride) if isinstance(stride, int) else stride
    pt, ph, pw = (padding, padding, padding) if isinstance(padding, int) else padding
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
