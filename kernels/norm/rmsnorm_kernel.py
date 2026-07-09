# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""RMSNorm kernel builder using the @flyc.kernel API.

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma

Two paths:
  - Fast path (N % tile_cols == 0): buffer_load/store vectorised access.
  - Generic path (arbitrary N): scalar copy_atom_call.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.vector import ReductionOp, full
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from kernels.common.kernels_common import dtype_to_elem_type

# Backward builders live in their own module (see review on #800); re-exported
# here so existing importers (tests, callers) keep working unchanged.
from kernels.norm.rmsnorm_bwd_kernel import (  # noqa: E402,F401
    build_fused_add_rmsnorm_bwd_module,
    build_rmsnorm_bwd_module,
)
from kernels.norm.rmsnorm_common import (
    BLOCK_THREADS,
    EPS,
    VEC_WIDTH,
    WARP_SIZE,
)
from kernels.norm.rmsnorm_common import load_scalar as _load_scalar
from kernels.norm.rmsnorm_common import load_vec as _load_vec
from kernels.norm.rmsnorm_common import make_reduction_storage as _make_reduction_storage
from kernels.norm.rmsnorm_common import store_scalar as _store_scalar
from kernels.norm.rmsnorm_common import store_vec as _store_vec
from kernels.norm.rmsnorm_common import to_elem_scalar as _to_elem_scalar
from kernels.norm.rmsnorm_common import to_elem_vec as _to_elem_vec

try:
    import torch
except ImportError:
    torch = None

KERNEL_NAME = "rmsnorm"


def _store_yscale(scale_copy_atom, yscale_div, index, val):
    r = fx.make_rmem_tensor(1, fx.Float32)
    ts = full(1, fx.Float32(val), fx.Float32)
    fx.memref_store_vec(ts, r)
    fx.copy_atom_call(scale_copy_atom, r, fx.slice(yscale_div, (None, index)))


def _quant_dtype_to_elem_type(dtype_str: str):
    if dtype_str in ("i8", "int8"):
        return fx.Int8
    raise ValueError(f"unsupported quant dtype: {dtype_str!r} (expected 'i8' or 'int8')")


def _quant_dtype_max(dtype_str: str) -> float:
    if dtype_str in ("i8", "int8"):
        return 127.0
    raise ValueError(f"unsupported quant dtype: {dtype_str!r} (expected 'i8' or 'int8')")


def build_rmsnorm_module(N: int, dtype_str: str, store_rstd: bool = False, eps: float = EPS):
    if N <= 2048:
        return _build_rmsnorm_large_m_small_n_module(N, dtype_str, store_rstd, eps)

    arch = get_hip_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    tile_cols = BLOCK_THREADS * VEC_WIDTH
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16

    SharedStorage = _make_reduction_storage(RED_SLOTS)

    @flyc.kernel
    def rmsnorm_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Rstd: fx.Tensor,
        Output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = eps
        n_float = float(N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))
        s_red2 = lds.s_red2.view(fx.make_layout(RED_SLOTS, 1))

        if const_expr(store_rstd):
            Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)
            rstd_div = fx.logical_divide(Rstd_buf, fx.make_layout(1, 1))
            rstd_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add(val):
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

        def block_reduce_add2(val0, val1):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_red, wave)
                fx.memref_store(w1, s_red2, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_red, lane_safe)
                v1 = fx.memref_load(s_red2, lane_safe)
                ww0 = in_range.select(v0, 0.0)
                ww1 = in_range.select(v1, 0.0)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == 0:
                    fx.memref_store(ww0, s_red, 0)
                    fx.memref_store(ww1, s_red2, 0)
            gpu.barrier()

            return fx.memref_load(s_red, 0), fx.memref_load(s_red2, 0)

        # ==================================================================
        # Fast path: N is a multiple of tile_cols
        # ==================================================================
        if const_expr(N >= tile_cols and N % tile_cols == 0 and elem_bits <= 16):
            num_tiles = N // tile_cols
            # ── Layout API: buffer-backed tensors + tiled access ─────
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f
            thread_dummy = c_zero_f
            in_local = []

            # Pass 1: load + cache + sumsq
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                vec = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx)
                in_local.append(vec)
                x = vec.to(fx.Float32)

                x2 = x * x
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + red2

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            if const_expr(store_rstd):
                if tid == 0:
                    _store_scalar(rstd_copy_atom, fx.Float32, fx.Float32, rstd_div, bid, rrms)

            # Pass 2: normalize + gamma + store (reuse cached input)
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS

                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx).to(fx.Float32)
                x = in_local[tile_i].to(fx.Float32)

                y = (x * rrms) * g
                out_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)

                out_idx = tid + tile_i * BLOCK_THREADS
                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, out_e, out_div, out_idx)

        else:
            # ==============================================================
            # Generic path: scalar 2-pass for arbitrary N
            # ==============================================================
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )

            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                x2 = x * x
                x2_safe = is_valid.select(x2, c_zero_f)
                thread_sumsq = thread_sumsq + x2_safe

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            if const_expr(store_rstd):
                if tid == 0:
                    _store_scalar(rstd_copy_atom, fx.Float32, fx.Float32, rstd_div, bid, rrms)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    norm = x * rrms
                    y = norm * g
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    _store_scalar(copy_atom_s, elem_dtype, elem_dtype, out_div, idx, y_e)

    if store_rstd:

        @flyc.jit
        def launch_rmsnorm(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            Rstd: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = rmsnorm_kernel(Input, Gamma, Rstd, Output)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_rmsnorm

    @flyc.jit
    def launch_rmsnorm(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # store_rstd=False path: the Rstd slot is an unused placeholder here, so
        # we pass Gamma to fill the argument (it is never dereferenced in-kernel).
        launcher = rmsnorm_kernel(Input, Gamma, Gamma, Output)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm


def _build_rmsnorm_large_m_small_n_module(N: int, dtype_str: str, store_rstd: bool = False, eps: float = EPS):
    BLOCK_N = 1 << (N - 1).bit_length()
    BLOCK_M = max(min(16384 // BLOCK_N, 32), 8)
    THREADS_PER_ROW = min(WARP_SIZE, 1024 // BLOCK_M)
    BLOCK_THREADS_SPECIAL = BLOCK_M * THREADS_PER_ROW
    elem_bits = 32 if dtype_str == "f32" else 16

    @flyc.kernel(known_block_size=[BLOCK_THREADS_SPECIAL, 1, 1])
    def rmsnorm_large_m_small_n_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Rstd: fx.Tensor,
        Output: fx.Tensor,
        MIn: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        lane = tid % THREADS_PER_ROW
        row_local = tid // THREADS_PER_ROW
        row = bid * fx.Int32(BLOCK_M) + row_local

        if row < MIn:
            elem_dtype = dtype_to_elem_type(dtype_str)
            fm_fast = arith.FastMathFlags.fast
            eps_c = eps
            n_float = float(N)

            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            if const_expr(store_rstd):
                Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)
                rstd_div = fx.logical_divide(Rstd_buf, fx.make_layout(1, 1))
                rstd_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

            row_in = fx.slice(Input_buf, (row, None))
            row_out = fx.slice(Output_buf, (row, None))

            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )

            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))

            def group_reduce_add(x):
                w = x
                for _sh_exp in range_constexpr(int(math.log2(THREADS_PER_ROW))):
                    off = THREADS_PER_ROW // (2 << _sh_exp)
                    peer = w.shuffle_xor(off, fx.Int32(THREADS_PER_ROW))
                    w = w.addf(peer, fastmath=fm_fast)
                return w

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, BLOCK_N, THREADS_PER_ROW):
                idx = lane + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                x2 = x * x
                thread_sumsq = thread_sumsq + is_valid.select(x2, c_zero_f)

            sum_sq = group_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            if const_expr(store_rstd):
                if lane == 0:
                    _store_scalar(rstd_copy_atom, fx.Float32, fx.Float32, rstd_div, row, rrms)

            for base_idx_int in range_constexpr(0, BLOCK_N, THREADS_PER_ROW):
                idx = lane + base_idx_int
                if idx < N:
                    x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    y = (x * rrms) * g
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    _store_scalar(copy_atom_s, elem_dtype, elem_dtype, out_div, idx, y_e)

    if store_rstd:

        @flyc.jit
        def launch_rmsnorm_large_m_small_n(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            Rstd: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = rmsnorm_large_m_small_n_kernel(Input, Gamma, Rstd, Output, m_in)
            launcher.launch(
                grid=((m_in + fx.Int32(BLOCK_M - 1)) // fx.Int32(BLOCK_M), 1, 1),
                block=(BLOCK_THREADS_SPECIAL, 1, 1),
                stream=stream,
            )

        return launch_rmsnorm_large_m_small_n

    @flyc.jit
    def launch_rmsnorm_large_m_small_n(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # store_rstd=False path: the Rstd slot is an unused placeholder here, so
        # we pass Gamma to fill the argument (it is never dereferenced in-kernel).
        launcher = rmsnorm_large_m_small_n_kernel(Input, Gamma, Gamma, Output, m_in)
        launcher.launch(
            grid=((m_in + fx.Int32(BLOCK_M - 1)) // fx.Int32(BLOCK_M), 1, 1),
            block=(BLOCK_THREADS_SPECIAL, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm_large_m_small_n


def build_fused_add_rmsnorm_module(N: int, dtype_str: str, store_rstd: bool = False, eps: float = EPS):
    arch = get_hip_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    tile_cols = BLOCK_THREADS * VEC_WIDTH
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16

    SharedStorage = _make_reduction_storage(RED_SLOTS)

    @flyc.kernel
    def fused_add_rmsnorm_kernel(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
        Rstd: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = eps
        n_float = float(N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))
        s_red2 = lds.s_red2.view(fx.make_layout(RED_SLOTS, 1))

        if const_expr(store_rstd):
            Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)
            rstd_div = fx.logical_divide(Rstd_buf, fx.make_layout(1, 1))
            rstd_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add(val):
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

        def block_reduce_add2(val0, val1):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_red, wave)
                fx.memref_store(w1, s_red2, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_red, lane_safe)
                v1 = fx.memref_load(s_red2, lane_safe)
                ww0 = in_range.select(v0, 0.0)
                ww1 = in_range.select(v1, 0.0)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == 0:
                    fx.memref_store(ww0, s_red, 0)
                    fx.memref_store(ww1, s_red2, 0)
            gpu.barrier()

            return fx.memref_load(s_red, 0), fx.memref_load(s_red2, 0)

        # ==================================================================
        # Fast path: N is a multiple of tile_cols
        # ==================================================================
        if const_expr(N >= tile_cols and N % tile_cols == 0 and elem_bits <= 16):
            num_tiles = N // tile_cols
            # ── Layout API: buffer-backed tensors + tiled access ─────
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(VEC_WIDTH, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f
            thread_dummy = c_zero_f
            add_local = []

            # Pass 1: add + cache + sumsq (also write residual_out)
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                x = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx).to(fx.Float32)
                residual = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, residual_in_div, idx).to(fx.Float32)
                added_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, x + residual)
                add_local.append(added_e)
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)

                added2 = added * added
                red2 = added2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + red2

                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, added_e, residual_out_div, idx)

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            if const_expr(store_rstd):
                if tid == 0:
                    _store_scalar(rstd_copy_atom, fx.Float32, fx.Float32, rstd_div, bid, rrms)

            # Pass 2: normalize + gamma + store (reuse cached added values)
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx).to(fx.Float32)
                added = add_local[tile_i] if dtype_str == "f32" else add_local[tile_i].to(fx.Float32)
                y = (added * rrms) * g
                y_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)
                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, y_e, out_div, idx)

        else:
            # ==============================================================
            # Generic path: scalar 2-pass for arbitrary N
            # ==============================================================
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )

            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(1, 1))

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                residual_e = _load_scalar(copy_atom_s, elem_dtype, residual_in_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                residual = residual_e if dtype_str == "f32" else residual_e.to(fx.Float32)
                added_e = _to_elem_scalar(dtype_str, elem_dtype, x + residual)
                if idx < N:
                    _store_scalar(copy_atom_s, elem_dtype, elem_dtype, residual_out_div, idx, added_e)
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                added2 = added * added
                thread_sumsq = thread_sumsq + is_valid.select(added2, c_zero_f)

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            if const_expr(store_rstd):
                if tid == 0:
                    _store_scalar(rstd_copy_atom, fx.Float32, fx.Float32, rstd_div, bid, rrms)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    added_e = _load_scalar(copy_atom_s, elem_dtype, residual_out_div, idx)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                    y = (added * rrms) * g
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    _store_scalar(copy_atom_s, elem_dtype, elem_dtype, out_div, idx, y_e)

    if store_rstd:

        @flyc.jit
        def launch_fused_add_rmsnorm(
            Input: fx.Tensor,
            ResidualIn: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            ResidualOut: fx.Tensor,
            Rstd: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = fused_add_rmsnorm_kernel(Input, ResidualIn, Gamma, Output, ResidualOut, Rstd)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_fused_add_rmsnorm

    @flyc.jit
    def launch_fused_add_rmsnorm(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = fused_add_rmsnorm_kernel(Input, ResidualIn, Gamma, Output, ResidualOut, Gamma)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fused_add_rmsnorm


def _build_rmsnorm_quant_module(
    N: int,
    dtype_str: str,
    *,
    is_smooth: bool,
    quant_dtype_str: str = "i8",
):
    tile_cols = BLOCK_THREADS * VEC_WIDTH
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)

    SharedStorage = _make_reduction_storage(RED_SLOTS)

    @flyc.kernel
    def rmsnorm_quant_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        XScale: fx.Tensor,
        YScale: fx.Tensor,
        Output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        quant_dtype = _quant_dtype_to_elem_type(quant_dtype_str)

        fm_fast = arith.FastMathFlags.fast
        eps_c = EPS
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_neg_inf = fx.Float32(float("-inf"))
        c_dtype_max = fx.Float32(quant_dtype_max)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))
        s_red2 = lds.s_red2.view(fx.make_layout(RED_SLOTS, 1))

        YScale_buf = fx.rocdl.make_buffer_tensor(YScale)
        yscale_div = fx.logical_divide(YScale_buf, fx.make_layout(1, 1))
        scale_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def wave_reduce_max(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.maximumf(peer)
            return w

        def block_reduce_add(val):
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

        def block_reduce_add2(val0, val1):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_red, wave)
                fx.memref_store(w1, s_red2, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_red, lane_safe)
                v1 = fx.memref_load(s_red2, lane_safe)
                ww0 = in_range.select(v0, c_zero_f)
                ww1 = in_range.select(v1, c_zero_f)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == 0:
                    fx.memref_store(ww0, s_red, 0)
                    fx.memref_store(ww1, s_red2, 0)
            gpu.barrier()

            return fx.memref_load(s_red, 0), fx.memref_load(s_red2, 0)

        def block_reduce_max(val):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_max(val)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w = wave_reduce_max(val)
            if lane == 0:
                fx.memref_store(w, s_red, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_red, lane_safe)
                ww = in_range.select(v, c_neg_inf)
                ww = wave_reduce_max(ww)
                if lane == 0:
                    fx.memref_store(ww, s_red, 0)
            gpu.barrier()

            return fx.memref_load(s_red, 0)

        # ==================================================================
        # Fast path: N is a multiple of tile_cols
        # ==================================================================
        if const_expr(N >= tile_cols and N % tile_cols == 0 and elem_bits <= 16):
            num_tiles = N // tile_cols
            quant_half_width = VEC_WIDTH // 2
            abs_mask = full(VEC_WIDTH, fx.Uint32(0x7FFFFFFF), fx.Uint32)
            # ── Layout API: buffer-backed tensors + tiled access ─────
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            out_div_q = fx.logical_divide(row_out, fx.make_layout(quant_half_width, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            if const_expr(is_smooth):
                copy_atom_xs = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            copy_atom_q = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 8)

            thread_sumsq = c_zero_f
            thread_dummy = c_zero_f
            in_local = []

            # Pass 1: load + cache + sumsq
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                vec = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx)
                in_local.append(vec)
                x = vec.to(fx.Float32)
                x2 = x * x
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + red2

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            thread_row_max = c_zero_f
            y_local = []

            # Pass 2: normalize + gamma (+ optional smooth scale), cache output, and accumulate row max
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS

                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx).to(fx.Float32)
                x = in_local[tile_i].to(fx.Float32)
                y = (x * rrms) * g
                if const_expr(is_smooth):
                    s = _load_vec(copy_atom_xs, VEC_WIDTH, elem_dtype, xscale_div, idx).to(fx.Float32)
                    y = y * s

                y_local.append(y)
                y_abs = (y.bitcast(fx.Uint32) & abs_mask).bitcast(fx.Float32)
                tile_max = y_abs.reduce(ReductionOp.MAX)
                thread_row_max = thread_row_max.maximumf(tile_max)

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                _store_yscale(scale_copy_atom, yscale_div, bid, final_scale)

            inv_scale = c_one_f / final_scale

            # Pass 3: quantize + store using per-row scale
            for tile_i in range_constexpr(num_tiles):
                q = y_local[tile_i] * inv_scale
                q_i8 = q.to(quant_dtype)
                q_lo = q_i8.shuffle(q_i8, [0, 1, 2, 3])
                q_hi = q_i8.shuffle(q_i8, [4, 5, 6, 7])
                out_idx = tid * 2 + tile_i * BLOCK_THREADS * 2
                _store_vec(copy_atom_q, quant_half_width, quant_dtype, q_lo, out_div_q, out_idx)
                _store_vec(copy_atom_q, quant_half_width, quant_dtype, q_hi, out_div_q, out_idx + 1)

        else:
            # ==============================================================
            # Generic path: scalar 3-pass for arbitrary N
            # ==============================================================
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )
            copy_atom_qs = fx.make_copy_atom(fx.rocdl.BufferCopy(8), 8)
            if const_expr(is_smooth):
                copy_atom_xs = fx.make_copy_atom(
                    fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                    elem_bits,
                )

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(1, 1))

            def _abs_scalar(val):
                is_neg = val < c_zero_f
                neg_val = c_zero_f - val
                return is_neg.select(neg_val, val)

            thread_sumsq = c_zero_f

            # Pass 1: accumulate sumsq
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                x2 = x * x
                thread_sumsq = thread_sumsq + is_valid.select(x2, c_zero_f)

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            thread_row_max = c_zero_f
            # Pass 2: normalize, apply gamma (+ optional smooth scale), and accumulate row max
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                y = (x * rrms) * g
                if const_expr(is_smooth):
                    s_e = _load_scalar(copy_atom_xs, elem_dtype, xscale_div, idx_safe)
                    s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                    y = y * s
                y_abs = _abs_scalar(y)
                thread_row_max = thread_row_max.maximumf(is_valid.select(y_abs, c_zero_f))

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                _store_yscale(scale_copy_atom, yscale_div, bid, final_scale)

            inv_scale = c_one_f / final_scale

            # Pass 3: quantize + store using per-row scale
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    y = (x * rrms) * g
                    if const_expr(is_smooth):
                        s_e = _load_scalar(copy_atom_xs, elem_dtype, xscale_div, idx)
                        s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                        y = y * s
                    q = y * inv_scale
                    q_i8 = q.to(quant_dtype)
                    _store_scalar(copy_atom_qs, quant_dtype, quant_dtype, out_div, idx, q_i8)

    if is_smooth:

        @flyc.jit
        def launch_rmsnorm_smoothquant(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            XScale: fx.Tensor,
            Output: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = rmsnorm_quant_kernel(Input, Gamma, XScale, YScale, Output)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_rmsnorm_smoothquant

    else:

        @flyc.jit
        def launch_rmsnorm_dynamicquant(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = rmsnorm_quant_kernel(Input, Gamma, Gamma, YScale, Output)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_rmsnorm_dynamicquant


def build_rmsnorm_dynamicquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
):
    return _build_rmsnorm_quant_module(
        N,
        dtype_str,
        is_smooth=False,
        quant_dtype_str=quant_dtype_str,
    )


def build_rmsnorm_smoothquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
):
    return _build_rmsnorm_quant_module(
        N,
        dtype_str,
        is_smooth=True,
        quant_dtype_str=quant_dtype_str,
    )


def _build_fused_add_rmsnorm_quant_module(
    N: int,
    dtype_str: str,
    *,
    is_smooth: bool,
    quant_dtype_str: str = "i8",
):
    arch = get_hip_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    tile_cols = BLOCK_THREADS * VEC_WIDTH
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)

    SharedStorage = _make_reduction_storage(RED_SLOTS)

    @flyc.kernel
    def fused_add_rmsnorm_quant_kernel(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        XScale: fx.Tensor,
        YScale: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        quant_dtype = _quant_dtype_to_elem_type(quant_dtype_str)

        fm_fast = arith.FastMathFlags.fast
        eps_c = EPS
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_neg_inf = fx.Float32(float("-inf"))
        c_dtype_max = fx.Float32(quant_dtype_max)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))
        s_red2 = lds.s_red2.view(fx.make_layout(RED_SLOTS, 1))

        YScale_buf = fx.rocdl.make_buffer_tensor(YScale)
        yscale_div = fx.logical_divide(YScale_buf, fx.make_layout(1, 1))
        scale_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def wave_reduce_max(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.maximumf(peer)
            return w

        def block_reduce_add(val):
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

        def block_reduce_add2(val0, val1):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_red, wave)
                fx.memref_store(w1, s_red2, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_red, lane_safe)
                v1 = fx.memref_load(s_red2, lane_safe)
                ww0 = in_range.select(v0, c_zero_f)
                ww1 = in_range.select(v1, c_zero_f)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == 0:
                    fx.memref_store(ww0, s_red, 0)
                    fx.memref_store(ww1, s_red2, 0)
            gpu.barrier()

            return fx.memref_load(s_red, 0), fx.memref_load(s_red2, 0)

        def block_reduce_max(val):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_max(val)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w = wave_reduce_max(val)
            if lane == 0:
                fx.memref_store(w, s_red, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_red, lane_safe)
                ww = in_range.select(v, c_neg_inf)
                ww = wave_reduce_max(ww)
                if lane == 0:
                    fx.memref_store(ww, s_red, 0)
            gpu.barrier()

            return fx.memref_load(s_red, 0)

        # ==================================================================
        # Fast path: N is a multiple of tile_cols
        # ==================================================================
        if const_expr(N >= tile_cols and N % tile_cols == 0 and elem_bits <= 16):
            num_tiles = N // tile_cols
            quant_half_width = VEC_WIDTH // 2
            abs_mask = full(VEC_WIDTH, fx.Uint32(0x7FFFFFFF), fx.Uint32)
            # ── Layout API: buffer-backed tensors + tiled access ─────
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(VEC_WIDTH, 1))
            out_div_q = fx.logical_divide(row_out, fx.make_layout(quant_half_width, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            if const_expr(is_smooth):
                copy_atom_xs = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            copy_atom_q = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 8)

            thread_sumsq = c_zero_f
            thread_dummy = c_zero_f
            add_local = []

            # Pass 1: add + cache + sumsq (also write residual_out)
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                x = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx).to(fx.Float32)
                residual = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, residual_in_div, idx).to(fx.Float32)
                added_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, x + residual)
                add_local.append(added_e)
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                added2 = added * added
                red2 = added2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + red2
                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, added_e, residual_out_div, idx)

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            thread_row_max = c_zero_f
            y_local = []

            # Pass 2: normalize + gamma (+ optional smooth scale), cache output, and accumulate row max
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx).to(fx.Float32)
                added = add_local[tile_i] if dtype_str == "f32" else add_local[tile_i].to(fx.Float32)
                y = (added * rrms) * g
                if const_expr(is_smooth):
                    s = _load_vec(copy_atom_xs, VEC_WIDTH, elem_dtype, xscale_div, idx).to(fx.Float32)
                    y = y * s

                y_local.append(y)
                y_abs = (y.bitcast(fx.Uint32) & abs_mask).bitcast(fx.Float32)
                tile_max = y_abs.reduce(ReductionOp.MAX)
                thread_row_max = thread_row_max.maximumf(tile_max)

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                _store_yscale(scale_copy_atom, yscale_div, bid, final_scale)

            inv_scale = c_one_f / final_scale

            # Pass 3: quantize + store using per-row scale
            for tile_i in range_constexpr(num_tiles):
                q = y_local[tile_i] * inv_scale
                q_i8 = q.to(quant_dtype)
                q_lo = q_i8.shuffle(q_i8, [0, 1, 2, 3])
                q_hi = q_i8.shuffle(q_i8, [4, 5, 6, 7])
                out_idx = tid * 2 + tile_i * BLOCK_THREADS * 2
                _store_vec(copy_atom_q, quant_half_width, quant_dtype, q_lo, out_div_q, out_idx)
                _store_vec(copy_atom_q, quant_half_width, quant_dtype, q_hi, out_div_q, out_idx + 1)

        else:
            # ==============================================================
            # Generic path: scalar 3-pass for arbitrary N
            # ==============================================================
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )
            copy_atom_qs = fx.make_copy_atom(fx.rocdl.BufferCopy(8), 8)
            if const_expr(is_smooth):
                copy_atom_xs = fx.make_copy_atom(
                    fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                    elem_bits,
                )

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(1, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(1, 1))

            def _abs_scalar(val):
                is_neg = val < c_zero_f
                neg_val = c_zero_f - val
                return is_neg.select(neg_val, val)

            thread_sumsq = c_zero_f

            # Pass 1: add, write residual_out, and accumulate sumsq
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                residual_e = _load_scalar(copy_atom_s, elem_dtype, residual_in_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                residual = residual_e if dtype_str == "f32" else residual_e.to(fx.Float32)
                added_e = _to_elem_scalar(dtype_str, elem_dtype, x + residual)
                if idx < N:
                    _store_scalar(copy_atom_s, elem_dtype, elem_dtype, residual_out_div, idx, added_e)
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                added2 = added * added
                thread_sumsq = thread_sumsq + is_valid.select(added2, c_zero_f)

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            thread_row_max = c_zero_f
            # Pass 2: normalize, apply gamma (+ optional smooth scale), and accumulate row max
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx_safe)
                added_e = _load_scalar(copy_atom_s, elem_dtype, residual_out_div, idx_safe)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                y = (added * rrms) * g
                if const_expr(is_smooth):
                    s_e = _load_scalar(copy_atom_xs, elem_dtype, xscale_div, idx_safe)
                    s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                    y = y * s
                y_abs = _abs_scalar(y)
                thread_row_max = thread_row_max.maximumf(is_valid.select(y_abs, c_zero_f))

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                _store_yscale(scale_copy_atom, yscale_div, bid, final_scale)

            inv_scale = c_one_f / final_scale

            # Pass 3: quantize + store using per-row scale
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    added_e = _load_scalar(copy_atom_s, elem_dtype, residual_out_div, idx)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                    y = (added * rrms) * g
                    if const_expr(is_smooth):
                        s_e = _load_scalar(copy_atom_xs, elem_dtype, xscale_div, idx)
                        s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                        y = y * s
                    q = y * inv_scale
                    q_i8 = q.to(quant_dtype)
                    _store_scalar(copy_atom_qs, quant_dtype, quant_dtype, out_div, idx, q_i8)

    if is_smooth:

        @flyc.jit
        def launch_fused_add_rmsnorm_smoothquant(
            Input: fx.Tensor,
            ResidualIn: fx.Tensor,
            Gamma: fx.Tensor,
            XScale: fx.Tensor,
            Output: fx.Tensor,
            ResidualOut: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = fused_add_rmsnorm_quant_kernel(Input, ResidualIn, Gamma, XScale, YScale, Output, ResidualOut)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_fused_add_rmsnorm_smoothquant

    else:

        @flyc.jit
        def launch_fused_add_rmsnorm_dynamicquant(
            Input: fx.Tensor,
            ResidualIn: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            ResidualOut: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = fused_add_rmsnorm_quant_kernel(Input, ResidualIn, Gamma, Gamma, YScale, Output, ResidualOut)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_fused_add_rmsnorm_dynamicquant


def build_fused_add_rmsnorm_dynamicquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
):
    return _build_fused_add_rmsnorm_quant_module(
        N,
        dtype_str,
        is_smooth=False,
        quant_dtype_str=quant_dtype_str,
    )


def build_fused_add_rmsnorm_smoothquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
):
    return _build_fused_add_rmsnorm_quant_module(
        N,
        dtype_str,
        is_smooth=True,
        quant_dtype_str=quant_dtype_str,
    )


# =====================================================================
# Python wrappers + autograd (quack-aligned). PR 1: plain rmsnorm.
# =====================================================================
if torch is not None:
    from kernels.norm.rmsnorm_common import torch_dtype_to_str as _torch_dtype_to_str

    # Compiled-fn caches. Keys include device: a compiled function is bound to
    # the device/context it was built on, so reusing it on another GPU faults.
    # eps is a compile-time kernel constant, so it is part of the fwd key too.
    _FWD_CACHE: dict = {}
    _BWD_CACHE: dict = {}

    def _get_fwd_compiled(x, weight, out, rstd, M, N, dtype_str, store_rstd, eps, stream):
        key = (N, dtype_str, store_rstd, float(eps), x.device)
        entry = _FWD_CACHE.get(key)
        if entry is None:
            launch_fn = build_rmsnorm_module(N, dtype_str, store_rstd=store_rstd, eps=eps)
            if store_rstd:
                compiled = flyc.compile(launch_fn, x, weight, out, rstd, M, stream)
            else:
                compiled = flyc.compile(launch_fn, x, weight, out, M, stream)
            _FWD_CACHE[key] = compiled
            entry = compiled
        return entry

    def rmsnorm_fwd(x, weight, eps=EPS, store_rstd=False):
        """Forward RMSNorm. Returns (out, rstd). eps is baked into the kernel."""
        assert x.dim() == 2, "rmsnorm_fwd expects a 2D (M, N) input"
        assert x.is_contiguous() and weight.is_contiguous(), "rmsnorm_fwd expects contiguous inputs"
        # gamma is read at x's element width and the kernel launches on x's device,
        # so a mismatched weight would silently corrupt output (or read out of bounds).
        assert weight.device == x.device, "rmsnorm_fwd: weight and x must be on the same device"
        assert weight.dtype == x.dtype, "rmsnorm_fwd: weight dtype must match x dtype"
        M, N = x.shape
        out = torch.empty_like(x)
        rstd = torch.empty((M,), device=x.device, dtype=torch.float32) if store_rstd else None
        dtype_str = _torch_dtype_to_str(x.dtype)
        # Bind compile + launch to the tensors' device so the compiled kernel and
        # the stream belong to the right GPU/context (multi-GPU correctness).
        with torch.cuda.device(x.device):
            stream = torch.cuda.current_stream()
            compiled = _get_fwd_compiled(x, weight, out, rstd, M, N, dtype_str, store_rstd, eps, stream)
            if store_rstd:
                compiled(x, weight, out, rstd, M, stream)
            else:
                compiled(x, weight, out, M, stream)
        return out, rstd

    def rmsnorm_bwd(x, weight, dout, rstd, eps=EPS):
        """Backward RMSNorm. Returns (dx, dw) with dw cast to weight dtype.

        eps is not used directly here — it is already baked into `rstd` by the
        forward — but is accepted so callers can pass it symmetrically.
        """
        assert x.dim() == 2, "rmsnorm_bwd expects a 2D (M, N) input"
        assert x.is_contiguous() and dout.is_contiguous(), "rmsnorm_bwd expects contiguous inputs"
        assert weight.is_contiguous(), "rmsnorm_bwd: weight must be contiguous"
        # Same-device/same-dtype contract as the forward: gamma and dy are read at
        # x's element width and the kernel launches on x's device (multi-GPU correctness).
        assert weight.device == x.device == dout.device == rstd.device, "rmsnorm_bwd: inputs must share a device"
        assert weight.dtype == x.dtype and dout.dtype == x.dtype, "rmsnorm_bwd: weight/dout dtype must match x"
        M, N = x.shape
        dtype_str = _torch_dtype_to_str(x.dtype)
        dx = torch.empty_like(x)
        dweight = torch.zeros((N,), device=x.device, dtype=torch.float32)
        key = (N, dtype_str, x.device)
        # Bind compile + launch to the tensors' device (multi-GPU correctness).
        with torch.cuda.device(x.device):
            stream = torch.cuda.current_stream()
            compiled = _BWD_CACHE.get(key)
            if compiled is None:
                launch_fn = build_rmsnorm_bwd_module(N, dtype_str)
                # flyc.compile executes the kernel once during tracing, which would
                # accumulate into DWeight; zero it AFTER compiling.
                compiled = flyc.compile(launch_fn, x, weight, dout, rstd, dx, dweight, M, stream)
                _BWD_CACHE[key] = compiled
            dweight.zero_()
            compiled(x, weight, dout, rstd, dx, dweight, M, stream)
        return dx, dweight.to(weight.dtype)

    class RMSNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, eps):
            need_grad = x.requires_grad or weight.requires_grad
            out, rstd = rmsnorm_fwd(x, weight, eps=eps, store_rstd=need_grad)
            ctx.save_for_backward(x, weight, rstd)
            ctx.eps = eps
            return out

        @staticmethod
        def backward(ctx, dout):
            x, weight, rstd = ctx.saved_tensors
            dx, dw = rmsnorm_bwd(x, weight, dout.contiguous(), rstd, eps=ctx.eps)
            return dx, dw, None

    def rmsnorm(x, weight=None, eps=EPS):
        """Public entry: plain RMSNorm with autograd (weight required in PR 1)."""
        assert weight is not None, "PR 1 rmsnorm requires an explicit weight"
        N = weight.shape[-1]
        assert x.shape[-1] == N, f"x last dim {x.shape[-1]} != weight length {N}"
        # reshape() can return a non-contiguous view (e.g. from a strided slice);
        # the kernel indexes rows by raw stride, so force contiguity here rather
        # than relying only on the fwd assert (which vanishes under python -O).
        x_flat = x.reshape(-1, N).contiguous()
        out_flat = RMSNormFunction.apply(x_flat, weight, eps)
        return out_flat.reshape(x.shape)

    # -----------------------------------------------------------------
    # Fused-add / prenorm RMSNorm wrappers + autograd (PR 2).
    # -----------------------------------------------------------------
    _FUSED_ADD_FWD_CACHE: dict = {}
    _FUSED_ADD_BWD_CACHE: dict = {}

    def _get_fused_add_fwd_compiled(
        x, residual, weight, out, residual_out, rstd, M, N, dtype_str, store_rstd, eps, stream
    ):
        key = (N, dtype_str, store_rstd, float(eps), x.device)
        entry = _FUSED_ADD_FWD_CACHE.get(key)
        if entry is None:
            launch_fn = build_fused_add_rmsnorm_module(N, dtype_str, store_rstd=store_rstd, eps=eps)
            if store_rstd:
                compiled = flyc.compile(launch_fn, x, residual, weight, out, residual_out, rstd, M, stream)
            else:
                compiled = flyc.compile(launch_fn, x, residual, weight, out, residual_out, M, stream)
            _FUSED_ADD_FWD_CACHE[key] = compiled
            entry = compiled
        return entry

    def fused_add_rmsnorm_fwd(x, residual, weight, eps=EPS, store_rstd=False):
        """Forward fused-add RMSNorm. Returns (out, residual_out, rstd).

        residual_out = x + residual ; out = residual_out * rstd * weight.
        eps is baked into the kernel.
        """
        assert x.dim() == 2, "fused_add_rmsnorm_fwd expects a 2D (M, N) input"
        assert (
            x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()
        ), "fused_add_rmsnorm_fwd expects contiguous inputs"
        assert x.shape == residual.shape, "x and residual must have the same shape"
        # The kernel reads x/residual/weight with a single elem dtype derived from x;
        # a mismatch would silently bit-reinterpret residual/weight bytes.
        assert (
            x.dtype == residual.dtype == weight.dtype
        ), f"x/residual/weight dtypes must match, got {x.dtype}/{residual.dtype}/{weight.dtype}"
        # Only x.device gates compile/stream/cache; all operands must co-reside.
        assert (
            x.device == residual.device == weight.device
        ), f"x/residual/weight must be on the same device, got {x.device}/{residual.device}/{weight.device}"
        M, N = x.shape
        out = torch.empty_like(x)
        residual_out = torch.empty_like(x)
        rstd = torch.empty((M,), device=x.device, dtype=torch.float32) if store_rstd else None
        dtype_str = _torch_dtype_to_str(x.dtype)
        with torch.cuda.device(x.device):
            stream = torch.cuda.current_stream()
            compiled = _get_fused_add_fwd_compiled(
                x, residual, weight, out, residual_out, rstd, M, N, dtype_str, store_rstd, eps, stream
            )
            if store_rstd:
                compiled(x, residual, weight, out, residual_out, rstd, M, stream)
            else:
                compiled(x, residual, weight, out, residual_out, M, stream)
        return out, residual_out, rstd

    def fused_add_rmsnorm_bwd(added, weight, dout, rstd, dresidual_out=None, eps=EPS):
        """Backward fused-add RMSNorm. Returns (dx, dresidual, dw).

        `added` is the residual_out saved by the forward. Because added = x +
        residual_in, dx == dresidual unconditionally, so the kernel computes it
        once and this returns dx aliased as dresidual (no second buffer/store).
        dresidual_out is the downstream grad flowing into residual_out; when None
        it is treated as zero (a zero tensor is passed to the branch-free
        kernel). eps is already baked into `rstd`.
        """
        assert added.dim() == 2, "fused_add_rmsnorm_bwd expects a 2D (M, N) input"
        assert added.is_contiguous() and dout.is_contiguous(), "fused_add_rmsnorm_bwd expects contiguous inputs"
        assert (
            added.dtype == weight.dtype == dout.dtype
        ), f"added/weight/dout dtypes must match, got {added.dtype}/{weight.dtype}/{dout.dtype}"
        assert (
            added.device == weight.device == dout.device == rstd.device
        ), "fused_add_rmsnorm_bwd expects all tensors on the same device"
        if dresidual_out is not None:
            assert dresidual_out.is_contiguous(), "fused_add_rmsnorm_bwd expects contiguous dresidual_out"
            assert dresidual_out.dtype == added.dtype, "dresidual_out dtype must match added"
            assert dresidual_out.device == added.device, "dresidual_out must be on the same device as added"
        M, N = added.shape
        dtype_str = _torch_dtype_to_str(added.dtype)
        if dresidual_out is None:
            dresidual_out = torch.zeros_like(added)
        dx = torch.empty_like(added)
        dweight = torch.zeros((N,), device=added.device, dtype=torch.float32)
        key = (N, dtype_str, added.device)
        with torch.cuda.device(added.device):
            stream = torch.cuda.current_stream()
            compiled = _FUSED_ADD_BWD_CACHE.get(key)
            if compiled is None:
                launch_fn = build_fused_add_rmsnorm_bwd_module(N, dtype_str)
                # flyc.compile executes the kernel once during tracing, which would
                # accumulate into DWeight; zero it AFTER compiling.
                compiled = flyc.compile(launch_fn, added, weight, dout, dresidual_out, rstd, dx, dweight, M, stream)
                _FUSED_ADD_BWD_CACHE[key] = compiled
            dweight.zero_()
            compiled(added, weight, dout, dresidual_out, rstd, dx, dweight, M, stream)
        # dx == dresidual by construction; return dx as both (aliased).
        return dx, dx, dweight.to(weight.dtype)

    class FusedAddRMSNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, residual, weight, eps, prenorm):
            need_grad = x.requires_grad or residual.requires_grad or weight.requires_grad
            out, residual_out, rstd = fused_add_rmsnorm_fwd(x, residual, weight, eps=eps, store_rstd=need_grad)
            ctx.save_for_backward(residual_out, weight, rstd)
            ctx.eps = eps
            ctx.prenorm = prenorm
            if prenorm:
                return out, residual_out
            return out

        @staticmethod
        def backward(ctx, dout, *args):
            added, weight, rstd = ctx.saved_tensors
            dresidual_out = args[0].contiguous() if ctx.prenorm else None
            dx, dresidual, dw = fused_add_rmsnorm_bwd(
                added, weight, dout.contiguous(), rstd, dresidual_out=dresidual_out, eps=ctx.eps
            )
            return dx, dresidual, dw, None, None

    def fused_add_rmsnorm(x, residual, weight, eps=EPS, prenorm=True):
        """Public entry: fused-add (prenorm) RMSNorm with autograd.

        residual_out = x + residual ; out = rmsnorm(residual_out) * weight.
        prenorm=True (training-relevant) returns (out, residual_out).
        """
        assert weight is not None, "fused_add_rmsnorm requires an explicit weight"
        N = weight.shape[-1]
        assert x.shape[-1] == N, f"x last dim {x.shape[-1]} != weight length {N}"
        assert x.shape == residual.shape, "x and residual must have the same shape"
        # reshape() can return a non-contiguous view; force contiguity so the
        # kernel's raw-stride row indexing stays correct even under python -O.
        x_flat = x.reshape(-1, N).contiguous()
        residual_flat = residual.reshape(-1, N).contiguous()
        if prenorm:
            out_flat, residual_out_flat = FusedAddRMSNormFunction.apply(x_flat, residual_flat, weight, eps, True)
            return out_flat.reshape(x.shape), residual_out_flat.reshape(x.shape)
        out_flat = FusedAddRMSNormFunction.apply(x_flat, residual_flat, weight, eps, False)
        return out_flat.reshape(x.shape)
