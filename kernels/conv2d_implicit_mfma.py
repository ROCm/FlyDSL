# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""8-wave double-buffered implicit-GEMM conv2d (BF16).

x: (N, C, H, W) bf16 NCHW, weight: (K, C, R, S) bf16 KCRS.
Returns (N, K, Ho, Wo) bf16. Supports stride, padding, bias, and split-K.
"""

import functools
import weakref

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

# ---------------------------------------------------------------------------
# Tile / wave constants
# ---------------------------------------------------------------------------
TILE_M = 128
TILE_N = 128
TILE_K = 32
STAGES = 2

WAVE_M = 2
WAVE_N = 4
WARP_SIZE = 64
BLOCK_THREADS = WAVE_M * WAVE_N * WARP_SIZE  # 512

MFMA_M = 16
MFMA_N = 16
MFMA_A_VALUES = 8
MFMA_B_VALUES = 8
MFMA_C_VALUES = 4

HALF_M = TILE_M // 2
HALF_N = TILE_N // 2
QM_STEPS = HALF_M // WAVE_M // MFMA_M  # 2
QN_STEPS = HALF_N // WAVE_N // MFMA_N  # 1
N_SUB = QM_STEPS * QN_STEPS

assert QM_STEPS == 2 and QN_STEPS == 1

LDG_VEC = 8
BLOCK_VECS = LDG_VEC * BLOCK_THREADS
LDG_A_COUNT = TILE_M * TILE_K // BLOCK_VECS  # 1
LDG_B_COUNT = TILE_N * TILE_K // BLOCK_VECS  # 1
LDS_A_SIZE = STAGES * TILE_M * TILE_K
LDS_B_SIZE = STAGES * TILE_N * TILE_K

assert LDG_A_COUNT == 1 and LDG_B_COUNT == 1

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run_compiled(exe, *args):
    cf = getattr(exe, "_cf", None)
    if cf is None:
        cf = flyc.compile(exe, *args)
        exe._cf = cf
    else:
        cf(*args)


# ---------------------------------------------------------------------------
# NCHW → NHWC transpose kernel
# ---------------------------------------------------------------------------
TR_TILE = 64
TR_VEC = 8
TR_THREADS = 256
_TR_VPL = TR_TILE // TR_VEC
_TR_ITERS = (TR_TILE * TR_TILE) // (TR_VEC * TR_THREADS)
_TR_PAD = 8
_TR_LDS_S = TR_TILE + _TR_PAD


@functools.lru_cache(maxsize=64)
def _compile_transpose_nchw_nhwc(n, c, hw):
    """Transpose (N, C, HW) → (N, HW, C). Requires c%8==0, hw%8==0."""
    grid_hw = (hw + TR_TILE - 1) // TR_TILE
    grid_c = (c + TR_TILE - 1) // TR_TILE
    elem_ty = fx.BFloat16

    @flyc.kernel(known_block_size=[TR_THREADS, 1, 1])
    def transpose_kernel(out: fx.Tensor, inp: fx.Tensor):
        in_rsrc = buffer_ops.create_buffer_resource(inp, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)
        lds_alloc = fx.SharedAllocator(static=False)
        lds = lds_alloc.allocate(fx.Array[elem_ty, TR_TILE * _TR_LDS_S, 16]).peek()

        Vec = fx.Vector

        class Vec8Ty:
            ir_type = Vec.make_type(TR_VEC, elem_ty)

        class BF16Ty:
            ir_type = elem_ty.ir_type

        tid = fx.thread_idx.x
        hw0 = fx.block_idx.x * TR_TILE
        c0 = fx.block_idx.y * TR_TILE
        nb = fx.block_idx.z
        in_base = nb * c * hw
        out_base = nb * hw * c

        def lds_store_vec8(elem_offset, value):
            base = fx.Int64(fx.ptrtoint(lds.ptr)) + fx.Int64(elem_offset * 2)
            ptr = buffer_ops.create_llvm_ptr(base, address_space=3)
            llvm.StoreOp(value, ptr, alignment=16)

        def lds_load_scalar(elem_offset):
            u8 = fx.recast_iter(fx.Uint8, lds.ptr)
            return fx.ptr_load(u8 + fx.Int32(elem_offset * 2), result_type=BF16Ty)

        # Read coalesced vec8 along HW → LDS[c_local][hw_local]
        for i in range_constexpr(_TR_ITERS):
            lin = tid + i * TR_THREADS
            rc = lin // _TR_VPL
            sv = (lin % _TR_VPL) * TR_VEC
            cc = c0 + rc
            ss = hw0 + sv
            valid = (cc < c) & (ss < hw)
            g = arith.index_cast(T.i32, in_base + cc * hw + ss)
            safe = arith.select(valid, g, arith.constant(0, type=T.i32))
            v = buffer_ops.buffer_load(in_rsrc, safe, vec_width=TR_VEC, dtype=elem_ty)
            lds_store_vec8(rc * _TR_LDS_S + sv, v)

        llvm.InlineAsmOp(None, [], "s_waitcnt lgkmcnt(0)\n\ts_barrier", "", has_side_effects=True)

        # Write transposed: vec8 along C
        for i in range_constexpr(_TR_ITERS):
            lin = tid + i * TR_THREADS
            rs = lin // _TR_VPL
            cv = (lin % _TR_VPL) * TR_VEC
            ss = hw0 + rs
            cc = c0 + cv
            scalars = [lds_load_scalar((cv + j) * _TR_LDS_S + rs) for j in range_constexpr(TR_VEC)]
            vv = Vec.from_elements(scalars, dtype=elem_ty)
            valid = arith.andi(ss < hw, cc < c)
            store_if = scf.IfOp(valid, results_=[], has_else=False)
            with ir.InsertionPoint(store_if.then_block):
                go = arith.index_cast(T.i32, out_base + ss * c + cc)
                buffer_ops.buffer_store(vv, out_rsrc, go)
                scf.YieldOp([])

    @flyc.jit
    def launch_transpose(out: fx.Tensor, inp: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        transpose_kernel(out, inp).launch(
            grid=(grid_hw, grid_c, n),
            block=(TR_THREADS, 1, 1),
            stream=stream,
        )

    return launch_transpose


def _nchw_to_nhwc(x, stream):
    """Fast NCHW→NHWC via tiled transpose; falls back to torch when not aligned."""
    n, c, h, w = x.shape
    hw = h * w
    if not (x.is_contiguous() and x.dtype == torch.bfloat16 and c % 8 == 0 and hw % 8 == 0):
        return x.permute(0, 2, 3, 1).contiguous()
    out = torch.empty((n, h, w, c), device=x.device, dtype=x.dtype)
    exe = _compile_transpose_nchw_nhwc(n, c, hw)
    _run_compiled(exe, out, x, torch.cuda.current_stream() if stream is None else stream)
    return out


# ---------------------------------------------------------------------------
# Weight cache: KCRS → K, (R·S·C) contiguous row-major
# ---------------------------------------------------------------------------
_WEIGHT_CACHE: dict = {}


def _prep_weight(w, k, r, s, c):
    key = id(w)
    ent = _WEIGHT_CACHE.get(key)
    if ent is not None and ent[0]() is w:
        return ent[1]
    wk = w.permute(0, 2, 3, 1).contiguous().reshape(k, r * s * c)
    _WEIGHT_CACHE[key] = (weakref.ref(w), wk)
    return wk


# ---------------------------------------------------------------------------
# Core kernel
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def compile_conv2d_implicit_8wave(n, c, h, w, k, r, s, sh, sw, ph, pw, has_bias=False, splitk=1):
    ho = (h + 2 * ph - r) // sh + 1
    wo = (w + 2 * pw - s) // sw + 1
    hw_o = ho * wo
    npq = n * hw_o
    crs = c * r * s
    k_tiles = crs // TILE_K

    assert c % LDG_VEC == 0, f"C must be a multiple of {LDG_VEC}, got C={c}"
    assert crs % TILE_K == 0, f"C*R*S must be a multiple of {TILE_K}, got CRS={crs}"

    n_tail = k % TILE_N != 0
    grid_n = (k + TILE_N - 1) // TILE_N

    if (k % TILE_N != 0) or (npq % TILE_M != 0):
        splitk = 1
    splitk = max(1, min(splitk, k_tiles))
    while k_tiles % splitk != 0:
        splitk -= 1
    tiles_per_split = k_tiles // splitk
    use_splitk = splitk > 1

    grid_m = (npq + TILE_M - 1) // TILE_M
    elem_ty = fx.BFloat16
    mfma_fn = rocdl.mfma_f32_16x16x32_bf16

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def conv2d_8wave_kernel(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor):
        x_rsrc = buffer_ops.create_buffer_resource(x, max_size=True)
        w_rsrc = buffer_ops.create_buffer_resource(weight, max_size=True)
        y_rsrc = buffer_ops.create_buffer_resource(y, max_size=True)
        if const_expr(has_bias):
            bias_rsrc = buffer_ops.create_buffer_resource(bias, max_size=True)

        lds_alloc = fx.SharedAllocator(static=False)
        a_lds = lds_alloc.allocate(fx.Array[elem_ty, LDS_A_SIZE, 16]).peek()
        b_lds = lds_alloc.allocate(fx.Array[elem_ty, LDS_B_SIZE, 16]).peek()

        tid = fx.thread_idx.x
        pid = fx.block_idx.x
        m_offset = pid * TILE_M
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

        acc0 = arith.constant_vector(0.0, T.vec(MFMA_C_VALUES, T.f32))
        acc00 = [acc0 for _ in range_constexpr(N_SUB)]
        acc01 = [acc0 for _ in range_constexpr(N_SUB)]
        acc10 = [acc0 for _ in range_constexpr(N_SUB)]
        acc11 = [acc0 for _ in range_constexpr(N_SUB)]

        Vec = fx.Vector

        class Vec8Ty:
            ir_type = Vec.make_type(8, elem_ty)

        zero8 = arith.constant_vector(0.0, Vec8Ty.ir_type)

        def barrier(vmcnt=0, lgkmcnt=None):
            waits = []
            if vmcnt is not None:
                waits.append(f"vmcnt({vmcnt})")
            if lgkmcnt is not None:
                waits.append(f"lgkmcnt({lgkmcnt})")
            pre = ("s_waitcnt " + " ".join(waits) + "\n\t") if waits else ""
            llvm.InlineAsmOp(None, [], f"{pre}s_barrier", "", has_side_effects=True)

        def waitcnt(vmcnt=None, lgkmcnt=None):
            waits = []
            if vmcnt is not None:
                waits.append(f"vmcnt({vmcnt})")
            if lgkmcnt is not None:
                waits.append(f"lgkmcnt({lgkmcnt})")
            if waits:
                llvm.InlineAsmOp(None, [], "s_waitcnt " + " ".join(waits), "", has_side_effects=True)

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

        # ---- 2D im2col gather (global → registers) ----
        def gather_a(k_base):
            linear = tid * LDG_VEC
            local_m = linear // TILE_K
            local_k = linear % TILE_K
            row = m_offset + local_m
            row_valid = row < fx.Index(npq)
            n_idx = row // hw_o
            pq = row % hw_o
            oh = pq // wo
            ow = pq % wo
            k_abs = fx.Index(k_base) + fx.Index(local_k)
            cc = k_abs % c
            rs = k_abs // c
            ri = rs // s
            si = rs % s
            in_h = oh * sh + ri - ph
            in_w = ow * sw + si - pw
            valid = row_valid & in_range(in_h, h) & in_range(in_w, w)
            g_off = ((n_idx * h + in_h) * w + in_w) * c + cc
            g_off_i = arith.index_cast(T.i32, g_off)
            safe = arith.select(valid, g_off_i, arith.constant(0, type=T.i32))
            raw = buffer_ops.buffer_load(x_rsrc, safe, vec_width=8, dtype=elem_ty)
            return (raw, valid, local_m * TILE_K + local_k)

        def gather_b(k_base):
            linear = tid * LDG_VEC
            local_n = linear // TILE_K
            local_k = linear % TILE_K
            col = n_offset + fx.Index(local_n)
            g_off = arith.index_cast(T.i32, col * crs + (fx.Index(k_base) + fx.Index(local_k)))
            if const_expr(n_tail):
                col_valid = col < fx.Index(k)
                safe = arith.select(col_valid, g_off, arith.constant(0, type=T.i32))
                raw = buffer_ops.buffer_load(w_rsrc, safe, vec_width=8, dtype=elem_ty)
                return (raw, col_valid, local_n * TILE_K + local_k)
            raw = buffer_ops.buffer_load(w_rsrc, g_off, vec_width=8, dtype=elem_ty)
            return (raw, None, local_n * TILE_K + local_k)

        def commit_a(stage, vo):
            raw, valid, off = vo
            val = arith.select(valid, raw, zero8)
            lds_store_vec8(a_lds, fx.Index(stage) * TILE_M * TILE_K + off, val)

        def commit_b(stage, vo):
            raw, valid, off = vo
            val = raw if const_expr(valid is None) else arith.select(valid, raw, zero8)
            lds_store_vec8(b_lds, fx.Index(stage) * TILE_N * TILE_K + off, val)

        # ---- LDS → register reads ----
        def read_a_vec(stage, m_half, wm):
            a_row = m_half * HALF_M + wave_m * (HALF_M // WAVE_M) + wm * MFMA_M + lane_m
            return lds_load_vec8(a_lds, a_lds_off(stage, fx.Index(a_row), fx.Index(lane_k_a)))

        def read_b_vec(stage, n_half, wn):
            b_row = n_half * HALF_N + wave_n * (HALF_N // WAVE_N) + wn * MFMA_N + lane_n
            return lds_load_vec8(b_lds, b_lds_off(stage, fx.Index(b_row), fx.Index(lane_k_b)))

        def mfma_one(a_frag, b_frag, c_frag):
            out = mfma_fn(T.vec(MFMA_C_VALUES, T.f32), [a_frag, b_frag, c_frag, 0, 0, 0])
            rocdl.sched_mfma(1)
            return out

        # ---- Interleaved compute + LDS-prefetch phases ----
        def phase_b_prefetch(read_stage, a0_0, a0_1, b0_0, acc):
            out = [v for v in acc]
            out[0] = mfma_one(a0_0, b0_0, out[0])
            b1_0 = read_b_vec(read_stage, 1, 0)
            rocdl.sched_dsrd(1)
            out[1] = mfma_one(a0_1, b0_0, out[1])
            return out, b1_0

        def phase_a_prefetch(read_stage, a0_0, a0_1, b1_0, acc):
            out = [v for v in acc]
            out[0] = mfma_one(a0_0, b1_0, out[0])
            a1_0 = read_a_vec(read_stage, 1, 0)
            rocdl.sched_dsrd(1)
            out[1] = mfma_one(a0_1, b1_0, out[1])
            a1_1 = read_a_vec(read_stage, 1, 1)
            rocdl.sched_dsrd(1)
            return out, a1_0, a1_1

        def phase_ab_prefetch(read_stage, a1_0, a1_1, b1_0, acc):
            out = [v for v in acc]
            out[0] = mfma_one(a1_0, b1_0, out[0])
            next_b0_0 = read_b_vec(read_stage, 0, 0)
            rocdl.sched_dsrd(1)
            out[1] = mfma_one(a1_1, b1_0, out[1])
            return out, next_b0_0

        def phase_compute(a1_0, a1_1, b_0, acc):
            out = [v for v in acc]
            out[0] = mfma_one(a1_0, b_0, out[0])
            out[1] = mfma_one(a1_1, b_0, out[1])
            return out

        def compute_prefetch_phases(read_stage, a0_0, a0_1, b0_0):
            rocdl.s_setprio(1)
            c00, b1_0 = phase_b_prefetch(read_stage, a0_0, a0_1, b0_0, acc00)
            c01, a1_0, a1_1 = phase_a_prefetch(read_stage, a0_0, a0_1, b1_0, acc01)
            rocdl.s_setprio(0)
            return c00, c01, a1_0, a1_1, b1_0

        # ---- prologue ----
        stage = 0
        next_stage = 1
        commit_a(stage, gather_a(k_off))
        commit_b(stage, gather_b(k_off))
        if const_expr(tiles_per_split > 1):
            pf_a = gather_a(k_off + TILE_K)
            pf_b = gather_b(k_off + TILE_K)
            rocdl.sched_vmem(2)
        barrier(vmcnt=None, lgkmcnt=0)

        a0_0 = read_a_vec(stage, 0, 0)
        a0_1 = read_a_vec(stage, 0, 1)
        b0_0 = read_b_vec(stage, 0, 0)
        rocdl.sched_dsrd(3)

        # ---- main loop ----
        if const_expr(tiles_per_split > 2):
            for kt_idx in range_constexpr(tiles_per_split - 2):
                acc00, acc01, a1_0, a1_1, b1_0 = compute_prefetch_phases(stage, a0_0, a0_1, b0_0)

                commit_a(next_stage, pf_a)
                rocdl.sched_dswr(1)
                pf_a = gather_a(k_off + (kt_idx + 2) * TILE_K)
                rocdl.sched_vmem(1)
                rocdl.s_setprio(1)
                acc10[0] = mfma_one(a1_0, b0_0, acc10[0])

                commit_b(next_stage, pf_b)
                rocdl.sched_dswr(1)
                pf_b = gather_b(k_off + (kt_idx + 2) * TILE_K)
                rocdl.sched_vmem(1)
                acc10[1] = mfma_one(a1_1, b0_0, acc10[1])
                rocdl.s_setprio(0)

                barrier(vmcnt=None, lgkmcnt=0)

                rocdl.s_setprio(1)
                acc11, b0_0 = phase_ab_prefetch(next_stage, a1_0, a1_1, b1_0, acc11)
                rocdl.s_setprio(0)

                stage = next_stage
                next_stage = (stage + 1) % STAGES
                a0_0 = read_a_vec(stage, 0, 0)
                a0_1 = read_a_vec(stage, 0, 1)
                rocdl.sched_dsrd(2)

        # ---- peeled iteration ----
        if const_expr(tiles_per_split >= 2):
            acc00, acc01, a1_0, a1_1, b1_0 = compute_prefetch_phases(stage, a0_0, a0_1, b0_0)

            commit_a(next_stage, pf_a)
            rocdl.sched_dswr(1)
            rocdl.s_setprio(1)
            acc10[0] = mfma_one(a1_0, b0_0, acc10[0])

            commit_b(next_stage, pf_b)
            rocdl.sched_dswr(1)
            acc10[1] = mfma_one(a1_1, b0_0, acc10[1])
            rocdl.s_setprio(0)

            barrier(vmcnt=None, lgkmcnt=0)

            rocdl.s_setprio(1)
            acc11, b0_0 = phase_ab_prefetch(next_stage, a1_0, a1_1, b1_0, acc11)
            rocdl.s_setprio(0)
            stage = next_stage
            next_stage = (stage + 1) % STAGES
            a0_0 = read_a_vec(stage, 0, 0)
            a0_1 = read_a_vec(stage, 0, 1)
            rocdl.sched_dsrd(2)

        # ---- epilogue: final tile ----
        acc00, acc01, a1_0, a1_1, b1_0 = compute_prefetch_phases(stage, a0_0, a0_1, b0_0)
        waitcnt(lgkmcnt=0)
        rocdl.s_setprio(1)
        acc10 = phase_compute(a1_0, a1_1, b0_0, acc10)
        acc11 = phase_compute(a1_0, a1_1, b1_0, acc11)
        rocdl.s_setprio(0)

        # ---- store output ----
        _row_chk = npq % TILE_M != 0
        _need_chk = _row_chk or n_tail

        def _valid_raw(row, col):
            if const_expr(_row_chk and n_tail):
                return arith.andi(row < fx.Index(npq), col < fx.Index(k))
            if const_expr(_row_chk):
                rc = row < fx.Index(npq)
                return arith.andi(rc, rc)
            cc = col < fx.Index(k)
            return arith.andi(cc, cc)

        def store_quad(acc, m_half, n_half):
            for wm in range_constexpr(QM_STEPS):
                row_base = m_offset + m_half * HALF_M + wave_m * (HALF_M // WAVE_M) + wm * MFMA_M + c_m_vec
                for wn in range_constexpr(QN_STEPS):
                    col = n_offset + fx.Index(n_half * HALF_N + wave_n * (HALF_N // WAVE_N) + wn * MFMA_N + c_n)
                    a = Vec(acc[wm * QN_STEPS + wn])
                    if const_expr(has_bias and not use_splitk):
                        col_i = arith.index_cast(T.i32, col)
                        if const_expr(n_tail):
                            col_i = arith.select(col < fx.Index(k), col_i, arith.constant(0, type=T.i32))
                        bias_val = fx.Float32(buffer_ops.buffer_load(bias_rsrc, col_i, vec_width=1, dtype=fx.Float32))
                    for i in range_constexpr(MFMA_C_VALUES):
                        row = fx.Index(row_base + i)
                        off = row * k + col

                        def _emit():
                            if const_expr(use_splitk):
                                off_b = arith.index_cast(T.i32, off * 4)
                                z0 = arith.constant(0, type=T.i32)
                                rocdl.raw_ptr_buffer_atomic_fadd(a[i], y_rsrc, off_b, z0, z0)
                            else:
                                cval = (a[i] + bias_val).to(elem_ty) if const_expr(has_bias) else a[i].to(elem_ty)
                                buffer_ops.buffer_store(cval, y_rsrc, off)

                        if const_expr(_need_chk):
                            store_if = scf.IfOp(_valid_raw(row, col), results_=[], has_else=False)
                            with ir.InsertionPoint(store_if.then_block):
                                _emit()
                                scf.YieldOp([])
                        else:
                            _emit()

        store_quad(acc00, 0, 0)
        store_quad(acc01, 0, 1)
        store_quad(acc10, 1, 0)
        store_quad(acc11, 1, 1)

    @flyc.jit
    def launch(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        conv2d_8wave_kernel(y, x, weight, bias).launch(
            grid=(grid_m, grid_n, splitk), block=(BLOCK_THREADS, 1, 1), stream=stream
        )

    return launch


# ---------------------------------------------------------------------------
# Split-K heuristic
# ---------------------------------------------------------------------------


def _choose_splitk(npq, crs, k, device):
    grid_m = (npq + TILE_M - 1) // TILE_M
    grid_n = (k + TILE_N - 1) // TILE_N
    base = grid_m * grid_n
    k_tiles = crs // TILE_K
    if npq < 16384 or k_tiles < 16:
        return 1
    if k % TILE_N != 0 or npq % TILE_M != 0:
        return 1
    try:
        num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        num_cu = 256
    if base >= num_cu // 4:
        return 1
    sk = min(4, max(1, num_cu // base), k_tiles)
    while sk > 1 and k_tiles % sk != 0:
        sk -= 1
    return sk


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def conv2d_implicit_8wave(x, weight, bias=None, stride=1, padding=0, splitk=None, stream=None):
    """Implicit-GEMM conv2d (8-wave bf16).

    Args:
        x:       (N, C, H, W) bf16
        weight:  (K, C, R, S) bf16
        bias:    (K,) float32, optional
        stride:  int or (sh, sw)
        padding: int or (ph, pw)
        splitk:  None = auto, int = override
        stream:  torch CUDA stream

    Returns:
        (N, K, Ho, Wo) bf16
    """
    n, c, h, w = x.shape
    k, wc, r, s = weight.shape
    assert c == wc, f"in-channel mismatch: x has C={c}, weight has C={wc}"
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16

    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (padding, padding) if isinstance(padding, int) else padding

    ho = (h + 2 * ph - r) // sh + 1
    wo = (w + 2 * pw - s) // sw + 1
    npq = n * ho * wo
    crs = c * r * s

    assert c % LDG_VEC == 0, f"C must be a multiple of {LDG_VEC}, got C={c}"
    assert crs % TILE_K == 0, f"C*R*S must be a multiple of {TILE_K}, got CRS={crs}"

    sk = _choose_splitk(npq, crs, k, x.device) if splitk is None else max(1, splitk)
    k_tiles = crs // TILE_K
    while sk > 1 and k_tiles % sk != 0:
        sk -= 1
    use_splitk = sk > 1

    launch_stream = torch.cuda.current_stream() if stream is None else stream
    x_nhwc = _nchw_to_nhwc(x, launch_stream)
    w_packed = _prep_weight(weight, k, r, s, c)

    if use_splitk:
        y = torch.zeros((npq, k), device=x.device, dtype=torch.float32)
    else:
        y = torch.empty((npq, k), device=x.device, dtype=torch.bfloat16)

    has_bias = bias is not None
    bias_arg = bias.to(torch.float32).contiguous() if has_bias else torch.empty(1, device=x.device, dtype=torch.float32)

    exe = compile_conv2d_implicit_8wave(n, c, h, w, k, r, s, sh, sw, ph, pw, has_bias, sk)
    _run_compiled(exe, y, x_nhwc, w_packed, bias_arg, launch_stream)

    if use_splitk:
        if has_bias:
            y = y + bias_arg.view(1, k)
        y = y.to(torch.bfloat16)

    # (N*Ho*Wo, K) → (N, K, Ho, Wo)
    return y.view(n, ho, wo, k).permute(0, 3, 1, 2)
