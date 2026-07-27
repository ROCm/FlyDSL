# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""LayerNorm kernel builder using the @flyc.kernel API.

LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta

Two paths:
  - Fast path (N == BLOCK_THREADS * VEC_WIDTH * 4): vectorised tiled copy,
    register caching, pipelined gamma/beta loads.
  - Generic path (arbitrary N): scalar 2-pass implementation.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, full
from flydsl.runtime.device import get_rocm_arch
from kernels.common.kernels_common import atomic_add, dtype_to_elem_type, get_warp_size

try:
    import torch
except ImportError:
    torch = None

KERNEL_NAME = "layernorm"

EPS = 1e-5

BLOCK_THREADS = 256
WARP_SIZE = get_warp_size()
VEC_WIDTH = 8


# ── Shared-memory allocation for block reductions ─────────────────────
def _make_reduction_storage(red_slots: int):
    @fx.struct
    class SharedStorage:
        s_sum: fx.Array[fx.Float32, red_slots, 16]
        s_sumsq: fx.Array[fx.Float32, red_slots, 16]

    return SharedStorage


def _load_scalar(copy_atom, elem_dtype, divided_tensor, index):
    view = fx.slice(divided_tensor, (None, index))
    r = fx.make_rmem_tensor(1, elem_dtype)
    fx.copy_atom_call(copy_atom, view, r)
    return fx.memref_load_vec(r)[0]


def _store_scalar(copy_atom, elem_dtype, store_dtype, divided_tensor, index, val):
    r = fx.make_rmem_tensor(1, elem_dtype)
    ts = full(1, store_dtype(val), store_dtype)
    fx.memref_store_vec(ts, r)
    view = fx.slice(divided_tensor, (None, index))
    fx.copy_atom_call(copy_atom, r, view)


def _load_vec(copy_atom, vec_width, elem_dtype, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, idx)), r)
    return fx.memref_load_vec(r)


def _store_vec(copy_atom, vec_width, elem_dtype, val, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.memref_store_vec(val, r)
    fx.copy_atom_call(copy_atom, r, fx.slice(div_tensor, (None, idx)))


def _to_elem_scalar(dtype_str: str, elem_dtype, y):
    if const_expr(dtype_str == "f32"):
        return y
    return y.to(elem_dtype)


def _to_elem_vec(dtype_str: str, elem_dtype, use_hw_cvt_bf16: bool, y):
    if const_expr(dtype_str == "bf16"):
        if const_expr(use_hw_cvt_bf16):
            return y.to(elem_dtype)
        u = y.bitcast(fx.Uint32)
        upper = u >> 16
        lsb = upper & 1
        bias = lsb + 0x7FFF
        u_round = y.bitcast(fx.Uint32) + bias
        bf16_bits = u_round >> 16
        even = bf16_bits.shuffle(bf16_bits, [0, 2, 4, 6])
        odd = bf16_bits.shuffle(bf16_bits, [1, 3, 5, 7])
        odd_sh = odd << 16
        packed = even | odd_sh
        return packed.bitcast(elem_dtype)
    if const_expr(dtype_str == "f32"):
        return y
    return y.to(elem_dtype)


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


def build_layernorm_module(N: int, dtype_str: str, store_stats: bool = False, eps: float = EPS):
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16

    SharedStorage = _make_reduction_storage(RED_SLOTS)

    # ── GPU kernel ────────────────────────────────────────────────────────
    @flyc.kernel
    def layernorm_kernel(
        Input: fx.Pointer,
        Gamma: fx.Pointer,
        Beta: fx.Pointer,
        Output: fx.Pointer,
        Mean: fx.Pointer,
        Rstd: fx.Pointer,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = eps

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_sum = lds.s_sum.view(fx.make_layout(RED_SLOTS, 1))
        s_sumsq = lds.s_sumsq.view(fx.make_layout(RED_SLOTS, 1))

        # The wrapper guarantees contiguous [M, N] rows and contiguous [N]
        # affine parameters. Reconstruct only this block's static views so the
        # kernel ABI carries one address per argument, without shape/stride
        # metadata for layouts already fixed by the contract.
        row_layout = fx.make_layout(N, 1)
        scalar_layout = fx.make_layout(1, 1)
        bid_i64 = fx.Int64(bid)
        row_offset = bid_i64 * N
        Input_buf = fx.rocdl.make_buffer_tensor((Input + row_offset).view(row_layout))
        Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma.view(row_layout))
        Beta_buf = fx.rocdl.make_buffer_tensor(Beta.view(row_layout))
        Output_buf = fx.rocdl.make_buffer_tensor((Output + row_offset).view(row_layout))

        if const_expr(store_stats):
            mean_row = fx.rocdl.make_buffer_tensor((Mean + bid_i64).view(scalar_layout))
            rstd_row = fx.rocdl.make_buffer_tensor((Rstd + bid_i64).view(scalar_layout))
            mean_div = fx.logical_divide(mean_row, scalar_layout)
            rstd_div = fx.logical_divide(rstd_row, scalar_layout)
            stats_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        # ── helpers: wave / block reduction ───────────────────────────────
        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add2(val0, val1):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_sum, wave)
                fx.memref_store(w1, s_sumsq, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_sum, lane_safe)
                v1 = fx.memref_load(s_sumsq, lane_safe)
                ww0 = in_range.select(v0, 0.0)
                ww1 = in_range.select(v1, 0.0)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == 0:
                    fx.memref_store(ww0, s_sum, 0)
                    fx.memref_store(ww1, s_sumsq, 0)
            gpu.barrier()

            return fx.memref_load(s_sum, 0), fx.memref_load(s_sumsq, 0)

        def compute_mean_rstd(sum_val, sumsq_val):
            inv_n = 1.0 / float(N)
            mean = sum_val * inv_n
            mean_sq = sumsq_val * inv_n
            mean2 = mean * mean
            var = mean_sq - mean2
            is_neg = var < 0.0
            var = is_neg.select(0.0, var)
            var_eps = var + eps_c
            rstd = fmath.rsqrt(var_eps, fastmath=fm_fast)
            return mean, rstd

        # ==================================================================
        # Fast path: N == BLOCK_THREADS * VEC_WIDTH * 4
        # Uses buffer_load / buffer_store for high-bandwidth vectorised
        # memory access (same approach as preshuffle_gemm).
        # ==================================================================
        if const_expr(N == (BLOCK_THREADS * VEC_WIDTH * 4) and elem_bits <= 16):
            num_tiles_py = 4
            c_zero_f = fx.Float32(0.0)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            in_local = []

            # ── Layout API: buffer-backed tensors + tiled access ─────
            in_div = fx.logical_divide(Input_buf, fx.make_layout(VEC_WIDTH, 1))
            out_div = fx.logical_divide(Output_buf, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            # ── Pass 1: load input, accumulate sum / sumsq ───────────────
            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                vec = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx)
                in_local.append(vec)
                x = vec.to(fx.Float32)

                x2 = x * x
                red = x.reduce(ReductionOp.ADD, fastmath=fm_fast)
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sum = thread_sum + red
                thread_sumsq = thread_sumsq + red2

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            if const_expr(store_stats):
                if tid == 0:
                    _store_scalar(stats_copy_atom, fx.Float32, fx.Float32, mean_div, 0, mean)
                    _store_scalar(stats_copy_atom, fx.Float32, fx.Float32, rstd_div, 0, rstd)

            g_cur = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, tid).to(fx.Float32)
            b_cur = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, beta_div, tid).to(fx.Float32)

            # ── Pass 2: normalize + affine + store ───────────────────────
            for tile_i in range_constexpr(num_tiles_py):
                g_next = g_cur
                b_next = b_cur
                if const_expr(tile_i + 1 < num_tiles_py):
                    next_idx = tid + (tile_i + 1) * BLOCK_THREADS
                    g_next = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, next_idx).to(fx.Float32)
                    b_next = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, beta_div, next_idx).to(fx.Float32)
                else:
                    g_next = g_cur
                    b_next = b_cur

                x = in_local[tile_i].to(fx.Float32)
                y = (x - mean) * rstd
                y = y * g_cur + b_cur

                out_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)
                out_idx = tid + tile_i * BLOCK_THREADS
                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, out_e, out_div, out_idx)

                g_cur = g_next
                b_cur = b_next

        else:
            # ==============================================================
            # Generic path: 2-pass scalar implementation for arbitrary N
            # ==============================================================
            c_zero_f = fx.Float32(0.0)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f

            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )

            row_div = fx.logical_divide(Input_buf, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(Output_buf, fx.make_layout(1, 1))

            # ── Pass 1: sum + sumsq ──────────────────────────────────────
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                x2 = x * x
                x_safe = is_valid.select(x, c_zero_f)
                x2_safe = is_valid.select(x2, c_zero_f)
                thread_sum = thread_sum + x_safe
                thread_sumsq = thread_sumsq + x2_safe

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            if const_expr(store_stats):
                if tid == 0:
                    _store_scalar(stats_copy_atom, fx.Float32, fx.Float32, mean_div, 0, mean)
                    _store_scalar(stats_copy_atom, fx.Float32, fx.Float32, rstd_div, 0, rstd)

            # ── Pass 2: normalize + affine + store ───────────────────────
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    b_e = _load_scalar(copy_atom_s, elem_dtype, beta_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    diff = x - mean
                    norm = diff * rstd
                    scaled = norm * g
                    y = scaled + b
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    _store_scalar(copy_atom_s, elem_dtype, elem_dtype, out_div, idx, y_e)

    # ── JIT host launcher ─────────────────────────────────────────────────
    if store_stats:

        @flyc.jit
        def launch_layernorm(
            Input: fx.Pointer,
            Gamma: fx.Pointer,
            Beta: fx.Pointer,
            Output: fx.Pointer,
            Mean: fx.Pointer,
            Rstd: fx.Pointer,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = layernorm_kernel(Input, Gamma, Beta, Output, Mean, Rstd)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_layernorm

    @flyc.jit
    def launch_layernorm(
        Input: fx.Pointer,
        Gamma: fx.Pointer,
        Beta: fx.Pointer,
        Output: fx.Pointer,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = layernorm_kernel(Input, Gamma, Beta, Output, Gamma, Gamma)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_layernorm


def build_layernorm_bwd_module(N: int, dtype_str: str):
    """Fused LayerNorm backward: grid=(M,), one block per row.

    With x_hat = (x - mean)*rstd, wdy = dy*gamma:
      c1 = mean_N(wdy) ; c2 = mean_N(wdy * x_hat)
    Pass 2:
      dx = (wdy - c1 - x_hat*c2) * rstd            -> DX (elem dtype);
      dgamma_elem = dy * x_hat (fp32)              -> atomicAdd into DGamma[idx];
      dbias_elem  = dy        (fp32)               -> atomicAdd into DBias[idx].
    eps is baked into Rstd by the forward, so it is not needed here.

    Perf follow-ups (deferred; correctness-complete as-is): generic scalar path
    only — a vectorized fast path (mirroring the forward) and caching x/dy/gamma
    between pass 1 and pass 2 would cut global traffic.
    """
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    SharedStorage = _make_reduction_storage(RED_SLOTS)

    @flyc.kernel
    def layernorm_bwd_kernel(
        Input: fx.Pointer,
        Gamma: fx.Pointer,
        DY: fx.Pointer,
        Mean: fx.Pointer,
        Rstd: fx.Pointer,
        DX: fx.Pointer,
        DGamma: fx.Pointer,
        DBias: fx.Pointer,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_sum = lds.s_sum.view(fx.make_layout(RED_SLOTS, 1))
        s_sumsq = lds.s_sumsq.view(fx.make_layout(RED_SLOTS, 1))

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add2(val0, val1):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_sum, wave)
                fx.memref_store(w1, s_sumsq, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_sum, lane_safe)
                v1 = fx.memref_load(s_sumsq, lane_safe)
                ww0 = in_range.select(v0, 0.0)
                ww1 = in_range.select(v1, 0.0)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == 0:
                    fx.memref_store(ww0, s_sum, 0)
                    fx.memref_store(ww1, s_sumsq, 0)
            gpu.barrier()

            return fx.memref_load(s_sum, 0), fx.memref_load(s_sumsq, 0)

        # All layouts are fixed by the wrapper's contiguous contract. Rebuild
        # only the row/scalar views used by this block so the kernel ABI carries
        # one address per tensor and no dynamic shape/stride operands.
        row_layout = fx.make_layout(N, 1)
        scalar_layout = fx.make_layout(1, 1)
        bid_i64 = fx.Int64(bid)
        row_offset = bid_i64 * N
        row_in = fx.rocdl.make_buffer_tensor((Input + row_offset).view(row_layout))
        gamma = fx.rocdl.make_buffer_tensor(Gamma.view(row_layout))
        row_dy = fx.rocdl.make_buffer_tensor((DY + row_offset).view(row_layout))
        mean_row = fx.rocdl.make_buffer_tensor((Mean + bid_i64).view(scalar_layout))
        rstd_row = fx.rocdl.make_buffer_tensor((Rstd + bid_i64).view(scalar_layout))
        row_dx = fx.rocdl.make_buffer_tensor((DX + row_offset).view(row_layout))

        copy_atom_s = fx.make_copy_atom(
            fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
            elem_bits,
        )
        copy_atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
        dy_div = fx.logical_divide(row_dy, fx.make_layout(1, 1))
        gamma_div = fx.logical_divide(gamma, fx.make_layout(1, 1))
        dx_div = fx.logical_divide(row_dx, fx.make_layout(1, 1))
        mean_div = fx.logical_divide(mean_row, scalar_layout)
        rstd_div = fx.logical_divide(rstd_row, scalar_layout)

        mean = _load_scalar(copy_atom_f32, fx.Float32, mean_div, 0)
        rstd = _load_scalar(copy_atom_f32, fx.Float32, rstd_div, 0)
        dgamma_view = DGamma.view(fx.make_layout(N, 1))
        dbias_view = DBias.view(fx.make_layout(N, 1))

        # Pass 1: c1 = mean(wdy) ; c2 = mean(wdy * x_hat)
        thread_c1 = c_zero_f
        thread_c2 = c_zero_f
        for base in range_constexpr(0, N, BLOCK_THREADS):
            idx = tid + base
            is_valid = idx < N
            idx_safe = is_valid.select(idx, 0)
            x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
            dy_e = _load_scalar(copy_atom_s, elem_dtype, dy_div, idx_safe)
            g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx_safe)
            x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
            dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
            g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
            x_hat = (x - mean) * rstd
            wdy = dy * g
            thread_c1 = thread_c1 + is_valid.select(wdy, c_zero_f)
            thread_c2 = thread_c2 + is_valid.select(wdy * x_hat, c_zero_f)

        sum_c1, sum_c2 = block_reduce_add2(thread_c1, thread_c2)
        c1 = sum_c1 / n_float
        c2 = sum_c2 / n_float

        # Pass 2: dx = (wdy - c1 - x_hat*c2) * rstd ; dgamma += dy*x_hat ; dbias += dy
        for base in range_constexpr(0, N, BLOCK_THREADS):
            idx = tid + base
            if idx < N:
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx)
                dy_e = _load_scalar(copy_atom_s, elem_dtype, dy_div, idx)
                g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                x_hat = (x - mean) * rstd
                wdy = dy * g
                dx = (wdy - c1 - x_hat * c2) * rstd
                dx_e = dx if dtype_str == "f32" else dx.to(elem_dtype)
                _store_scalar(copy_atom_s, elem_dtype, elem_dtype, dx_div, idx, dx_e)

                dgamma = dy * x_hat
                atomic_add(dgamma_view, idx, dgamma, dtype_bytes=4)
                atomic_add(dbias_view, idx, dy, dtype_bytes=4)

    @flyc.jit
    def launch_layernorm_bwd(
        Input: fx.Pointer,
        Gamma: fx.Pointer,
        DY: fx.Pointer,
        Mean: fx.Pointer,
        Rstd: fx.Pointer,
        DX: fx.Pointer,
        DGamma: fx.Pointer,
        DBias: fx.Pointer,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = layernorm_bwd_kernel(Input, Gamma, DY, Mean, Rstd, DX, DGamma, DBias)
        launcher.launch(grid=(m_in, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_layernorm_bwd


def build_fused_add_layernorm_module(N: int, dtype_str: str):
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16

    SharedStorage = _make_reduction_storage(RED_SLOTS)

    @flyc.kernel
    def fused_add_layernorm_kernel(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = EPS

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_sum = lds.s_sum.view(fx.make_layout(RED_SLOTS, 1))
        s_sumsq = lds.s_sumsq.view(fx.make_layout(RED_SLOTS, 1))

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add2(val0, val1):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_sum, wave)
                fx.memref_store(w1, s_sumsq, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_sum, lane_safe)
                v1 = fx.memref_load(s_sumsq, lane_safe)
                ww0 = in_range.select(v0, 0.0)
                ww1 = in_range.select(v1, 0.0)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == 0:
                    fx.memref_store(ww0, s_sum, 0)
                    fx.memref_store(ww1, s_sumsq, 0)
            gpu.barrier()

            return fx.memref_load(s_sum, 0), fx.memref_load(s_sumsq, 0)

        def compute_mean_rstd(sum_val, sumsq_val):
            inv_n = 1.0 / float(N)
            mean = sum_val * inv_n
            mean_sq = sumsq_val * inv_n
            var = mean_sq - mean * mean
            var = (var < 0.0).select(0.0, var)
            return mean, fmath.rsqrt(var + eps_c, fastmath=fm_fast)

        # ==================================================================
        # Fast path: N == BLOCK_THREADS * VEC_WIDTH * 4
        # ==================================================================
        if const_expr(N == (BLOCK_THREADS * VEC_WIDTH * 4) and elem_bits <= 16):
            num_tiles_py = 4
            c_zero_f = fx.Float32(0.0)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            added_local = []

            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(VEC_WIDTH, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            # Pass 1: add residual, cache/store it, and accumulate sum/sumsq.
            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                x = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx).to(fx.Float32)
                residual = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, residual_in_div, idx).to(fx.Float32)
                added_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, x + residual)
                added_local.append(added_e)
                added = added_e.to(fx.Float32)
                added2 = added * added
                thread_sum = thread_sum + added.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + added2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, added_e, residual_out_div, idx)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            # Pass 2: normalize + affine + store, reusing cached added values.
            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                added = added_local[tile_i].to(fx.Float32)
                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx).to(fx.Float32)
                b = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, beta_div, idx).to(fx.Float32)
                y = (added - mean) * rstd
                y = y * g + b
                y_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)
                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, y_e, out_div, idx)

        else:
            # ==============================================================
            # Generic path: scalar 2-pass implementation for arbitrary N
            # ==============================================================
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
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

            in_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(1, 1))

            c_zero_f = fx.Float32(0.0)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, in_div, idx_safe)
                r_e = _load_scalar(copy_atom_s, elem_dtype, residual_in_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                residual = r_e if dtype_str == "f32" else r_e.to(fx.Float32)
                added_e = _to_elem_scalar(dtype_str, elem_dtype, x + residual)
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                added_safe = is_valid.select(added, c_zero_f)
                thread_sum = thread_sum + added_safe
                thread_sumsq = thread_sumsq + is_valid.select(added * added, c_zero_f)
                if idx < N:
                    _store_scalar(copy_atom_s, elem_dtype, elem_dtype, residual_out_div, idx, added_e)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    added_e = _load_scalar(copy_atom_s, elem_dtype, residual_out_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    b_e = _load_scalar(copy_atom_s, elem_dtype, beta_div, idx)
                    added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    y = (added - mean) * rstd
                    y = y * g + b
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    _store_scalar(copy_atom_s, elem_dtype, elem_dtype, out_div, idx, y_e)

    @flyc.jit
    def launch_fused_add_layernorm(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = fused_add_layernorm_kernel(Input, ResidualIn, Gamma, Beta, Output, ResidualOut)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fused_add_layernorm


def _build_layernorm_quant_module(
    N: int,
    dtype_str: str,
    *,
    is_smooth: bool,
    quant_dtype_str: str = "i8",
):
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)

    SharedStorage = _make_reduction_storage(RED_SLOTS)

    @flyc.kernel
    def layernorm_quant_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
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
        s_sum = lds.s_sum.view(fx.make_layout(RED_SLOTS, 1))
        s_sumsq = lds.s_sumsq.view(fx.make_layout(RED_SLOTS, 1))

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

        def block_reduce_add2(val0, val1):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_sum, wave)
                fx.memref_store(w1, s_sumsq, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_sum, lane_safe)
                v1 = fx.memref_load(s_sumsq, lane_safe)
                ww0 = in_range.select(v0, c_zero_f)
                ww1 = in_range.select(v1, c_zero_f)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)
                if lane == 0:
                    fx.memref_store(ww0, s_sum, 0)
                    fx.memref_store(ww1, s_sumsq, 0)
            gpu.barrier()

            return fx.memref_load(s_sum, 0), fx.memref_load(s_sumsq, 0)

        def block_reduce_max(val):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_max(val)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            w = wave_reduce_max(val)
            if lane == 0:
                fx.memref_store(w, s_sum, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_sum, lane_safe)
                ww = in_range.select(v, c_neg_inf)
                ww = wave_reduce_max(ww)
                if lane == 0:
                    fx.memref_store(ww, s_sum, 0)
            gpu.barrier()

            return fx.memref_load(s_sum, 0)

        # ==================================================================
        # Fast path: N == BLOCK_THREADS * VEC_WIDTH * 4
        # ==================================================================
        if const_expr(N == (BLOCK_THREADS * VEC_WIDTH * 4) and elem_bits <= 16):
            num_tiles_py = 4
            quant_half_width = VEC_WIDTH // 2
            abs_mask = full(VEC_WIDTH, fx.Uint32(0x7FFFFFFF), fx.Uint32)

            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(VEC_WIDTH, 1))
            out_div_q = fx.logical_divide(row_out, fx.make_layout(quant_half_width, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            copy_atom_q = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 8)
            if const_expr(is_smooth):
                copy_atom_xs = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            norm_input_local = []

            # Pass 1: prepare normalization input and accumulate sum/sumsq.
            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                x_e = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx)
                norm_input_local.append(x_e)
                x_norm = x_e.to(fx.Float32)
                x2 = x_norm * x_norm
                thread_sum = thread_sum + x_norm.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + x2.reduce(ReductionOp.ADD, fastmath=fm_fast)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean = sum_val / n_float
            var = sumsq_val / n_float - mean * mean
            var = (var < c_zero_f).select(c_zero_f, var)
            rstd = fmath.rsqrt(var + eps_c, fastmath=fm_fast)

            thread_row_max = c_zero_f
            y_local = []

            # Pass 2: affine (+ optional smooth scale), cache y, accumulate row max.
            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                x = norm_input_local[tile_i].to(fx.Float32)
                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx).to(fx.Float32)
                b = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, beta_div, idx).to(fx.Float32)
                y = (x - mean) * rstd
                y = y * g + b
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

            # Pass 3: quantize + store using per-row scale.
            for tile_i in range_constexpr(num_tiles_py):
                q = y_local[tile_i] * inv_scale
                q_i8 = q.to(quant_dtype)
                q_lo = q_i8.shuffle(q_i8, [0, 1, 2, 3])
                q_hi = q_i8.shuffle(q_i8, [4, 5, 6, 7])
                out_idx = tid * 2 + tile_i * BLOCK_THREADS * 2
                _store_vec(copy_atom_q, quant_half_width, quant_dtype, q_lo, out_div_q, out_idx)
                _store_vec(copy_atom_q, quant_half_width, quant_dtype, q_hi, out_div_q, out_idx + 1)

        else:
            # ==============================================================
            # Generic path: scalar 3-pass implementation for arbitrary N
            # ==============================================================
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )
            copy_atom_qs = fx.make_copy_atom(fx.rocdl.BufferCopy(8), 8)

            in_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(1, 1))

            def _abs_scalar(val):
                is_neg = val < c_zero_f
                neg_val = c_zero_f - val
                return is_neg.select(neg_val, val)

            thread_sum = c_zero_f
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, in_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                x2 = x * x
                thread_sum = thread_sum + is_valid.select(x, c_zero_f)
                thread_sumsq = thread_sumsq + is_valid.select(x2, c_zero_f)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean = sum_val / n_float
            var = sumsq_val / n_float - mean * mean
            var = (var < c_zero_f).select(c_zero_f, var)
            rstd = fmath.rsqrt(var + eps_c, fastmath=fm_fast)

            thread_row_max = c_zero_f
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, in_div, idx_safe)
                g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx_safe)
                b_e = _load_scalar(copy_atom_s, elem_dtype, beta_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                y = (x - mean) * rstd
                y = y * g + b
                if const_expr(is_smooth):
                    s_e = _load_scalar(copy_atom_s, elem_dtype, xscale_div, idx_safe)
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

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = _load_scalar(copy_atom_s, elem_dtype, in_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    b_e = _load_scalar(copy_atom_s, elem_dtype, beta_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    y = (x - mean) * rstd
                    y = y * g + b
                    if const_expr(is_smooth):
                        s_e = _load_scalar(copy_atom_s, elem_dtype, xscale_div, idx)
                        s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                        y = y * s
                    q = y * inv_scale
                    q_i8 = q.to(quant_dtype)
                    _store_scalar(copy_atom_qs, quant_dtype, quant_dtype, out_div, idx, q_i8)

    if is_smooth:

        @flyc.jit
        def launch_layernorm_smoothquant(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Beta: fx.Tensor,
            XScale: fx.Tensor,
            Output: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = layernorm_quant_kernel(Input, Gamma, Beta, XScale, YScale, Output)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_layernorm_smoothquant

    else:

        @flyc.jit
        def launch_layernorm_dynamicquant(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Beta: fx.Tensor,
            Output: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = layernorm_quant_kernel(Input, Gamma, Beta, Gamma, YScale, Output)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

    return launch_layernorm_dynamicquant


def _build_fused_add_layernorm_quant_module(
    N: int,
    dtype_str: str,
    *,
    is_smooth: bool,
    quant_dtype_str: str = "i8",
):
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)

    SharedStorage = _make_reduction_storage(RED_SLOTS)

    @flyc.kernel
    def fused_add_layernorm_quant_kernel(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
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
        s_sum = lds.s_sum.view(fx.make_layout(RED_SLOTS, 1))
        s_sumsq = lds.s_sumsq.view(fx.make_layout(RED_SLOTS, 1))

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

        def block_reduce_add2(val0, val1):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_sum, wave)
                fx.memref_store(w1, s_sumsq, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_sum, lane_safe)
                v1 = fx.memref_load(s_sumsq, lane_safe)
                ww0 = in_range.select(v0, c_zero_f)
                ww1 = in_range.select(v1, c_zero_f)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)
                if lane == 0:
                    fx.memref_store(ww0, s_sum, 0)
                    fx.memref_store(ww1, s_sumsq, 0)
            gpu.barrier()

            return fx.memref_load(s_sum, 0), fx.memref_load(s_sumsq, 0)

        def block_reduce_max(val):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_max(val)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            w = wave_reduce_max(val)
            if lane == 0:
                fx.memref_store(w, s_sum, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_sum, lane_safe)
                ww = in_range.select(v, c_neg_inf)
                ww = wave_reduce_max(ww)
                if lane == 0:
                    fx.memref_store(ww, s_sum, 0)
            gpu.barrier()

            return fx.memref_load(s_sum, 0)

        # ==================================================================
        # Fast path: N == BLOCK_THREADS * VEC_WIDTH * 4
        # ==================================================================
        if const_expr(N == (BLOCK_THREADS * VEC_WIDTH * 4) and elem_bits <= 16):
            num_tiles_py = 4
            quant_half_width = VEC_WIDTH // 2
            abs_mask = full(VEC_WIDTH, fx.Uint32(0x7FFFFFFF), fx.Uint32)

            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
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
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(VEC_WIDTH, 1))
            out_div_q = fx.logical_divide(row_out, fx.make_layout(quant_half_width, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(VEC_WIDTH, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            copy_atom_q = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 8)
            if const_expr(is_smooth):
                copy_atom_xs = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            norm_input_local = []

            # Pass 1: add residual, store residual_out, and accumulate sum/sumsq.
            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                x = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx).to(fx.Float32)
                residual = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, residual_in_div, idx).to(fx.Float32)
                added_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, x + residual)
                norm_input_local.append(added_e)
                x_norm = added_e.to(fx.Float32)
                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, added_e, residual_out_div, idx)
                x2 = x_norm * x_norm
                thread_sum = thread_sum + x_norm.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + x2.reduce(ReductionOp.ADD, fastmath=fm_fast)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean = sum_val / n_float
            var = sumsq_val / n_float - mean * mean
            var = (var < c_zero_f).select(c_zero_f, var)
            rstd = fmath.rsqrt(var + eps_c, fastmath=fm_fast)

            thread_row_max = c_zero_f
            y_local = []

            # Pass 2: affine (+ optional smooth scale), cache y, accumulate row max.
            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                x = norm_input_local[tile_i].to(fx.Float32)
                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx).to(fx.Float32)
                b = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, beta_div, idx).to(fx.Float32)
                y = (x - mean) * rstd
                y = y * g + b
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

            # Pass 3: quantize + store using per-row scale.
            for tile_i in range_constexpr(num_tiles_py):
                q = y_local[tile_i] * inv_scale
                q_i8 = q.to(quant_dtype)
                q_lo = q_i8.shuffle(q_i8, [0, 1, 2, 3])
                q_hi = q_i8.shuffle(q_i8, [4, 5, 6, 7])
                out_idx = tid * 2 + tile_i * BLOCK_THREADS * 2
                _store_vec(copy_atom_q, quant_half_width, quant_dtype, q_lo, out_div_q, out_idx)
                _store_vec(copy_atom_q, quant_half_width, quant_dtype, q_hi, out_div_q, out_idx + 1)

        else:
            # ==============================================================
            # Generic path: scalar 3-pass implementation for arbitrary N
            # ==============================================================
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )
            copy_atom_qs = fx.make_copy_atom(fx.rocdl.BufferCopy(8), 8)

            in_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(1, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(1, 1))

            def _abs_scalar(val):
                is_neg = val < c_zero_f
                neg_val = c_zero_f - val
                return is_neg.select(neg_val, val)

            thread_sum = c_zero_f
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, in_div, idx_safe)
                r_e = _load_scalar(copy_atom_s, elem_dtype, residual_in_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                residual = r_e if dtype_str == "f32" else r_e.to(fx.Float32)
                added_e = _to_elem_scalar(dtype_str, elem_dtype, x + residual)
                if idx < N:
                    _store_scalar(copy_atom_s, elem_dtype, elem_dtype, residual_out_div, idx, added_e)
                x = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                x2 = x * x
                thread_sum = thread_sum + is_valid.select(x, c_zero_f)
                thread_sumsq = thread_sumsq + is_valid.select(x2, c_zero_f)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean = sum_val / n_float
            var = sumsq_val / n_float - mean * mean
            var = (var < c_zero_f).select(c_zero_f, var)
            rstd = fmath.rsqrt(var + eps_c, fastmath=fm_fast)

            thread_row_max = c_zero_f
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, residual_out_div, idx_safe)
                g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx_safe)
                b_e = _load_scalar(copy_atom_s, elem_dtype, beta_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                y = (x - mean) * rstd
                y = y * g + b
                if const_expr(is_smooth):
                    s_e = _load_scalar(copy_atom_s, elem_dtype, xscale_div, idx_safe)
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

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = _load_scalar(copy_atom_s, elem_dtype, residual_out_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    b_e = _load_scalar(copy_atom_s, elem_dtype, beta_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    y = (x - mean) * rstd
                    y = y * g + b
                    if const_expr(is_smooth):
                        s_e = _load_scalar(copy_atom_s, elem_dtype, xscale_div, idx)
                        s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                        y = y * s
                    q = y * inv_scale
                    q_i8 = q.to(quant_dtype)
                    _store_scalar(copy_atom_qs, quant_dtype, quant_dtype, out_div, idx, q_i8)

    if is_smooth:

        @flyc.jit
        def launch_fused_add_layernorm_smoothquant(
            Input: fx.Tensor,
            ResidualIn: fx.Tensor,
            Gamma: fx.Tensor,
            Beta: fx.Tensor,
            XScale: fx.Tensor,
            Output: fx.Tensor,
            ResidualOut: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = fused_add_layernorm_quant_kernel(
                Input, ResidualIn, Gamma, Beta, XScale, YScale, Output, ResidualOut
            )
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_fused_add_layernorm_smoothquant

    else:

        @flyc.jit
        def launch_fused_add_layernorm_dynamicquant(
            Input: fx.Tensor,
            ResidualIn: fx.Tensor,
            Gamma: fx.Tensor,
            Beta: fx.Tensor,
            Output: fx.Tensor,
            ResidualOut: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = fused_add_layernorm_quant_kernel(
                Input, ResidualIn, Gamma, Beta, Gamma, YScale, Output, ResidualOut
            )
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

    return launch_fused_add_layernorm_dynamicquant


def build_layernorm_dynamicquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
):
    return _build_layernorm_quant_module(
        N,
        dtype_str,
        is_smooth=False,
        quant_dtype_str=quant_dtype_str,
    )


def build_layernorm_smoothquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
):
    return _build_layernorm_quant_module(
        N,
        dtype_str,
        is_smooth=True,
        quant_dtype_str=quant_dtype_str,
    )


def build_fused_add_layernorm_dynamicquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
):
    return _build_fused_add_layernorm_quant_module(
        N,
        dtype_str,
        is_smooth=False,
        quant_dtype_str=quant_dtype_str,
    )


def build_fused_add_layernorm_smoothquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
):
    return _build_fused_add_layernorm_quant_module(
        N,
        dtype_str,
        is_smooth=True,
        quant_dtype_str=quant_dtype_str,
    )


# =====================================================================
# Python wrappers + autograd (quack-aligned). PR 3: plain layernorm.
# =====================================================================
if torch is not None:

    def _torch_dtype_to_str(dt) -> str:
        if dt == torch.float32:
            return "f32"
        if dt == torch.float16:
            return "f16"
        if dt == torch.bfloat16:
            return "bf16"
        raise ValueError(f"unsupported torch dtype: {dt}")

    # Compiled-fn caches. Keys include device: a compiled function is bound to
    # the device/context it was built on, so reusing it on another GPU faults.
    # eps is a compile-time kernel constant, so it is part of the fwd key too.
    _FWD_CACHE: dict = {}
    _BWD_CACHE: dict = {}

    def _get_fwd_compiled(x, weight, bias, out, mean, rstd, M, N, dtype_str, store_stats, eps, stream):
        key = (N, dtype_str, store_stats, float(eps), x.device)
        entry = _FWD_CACHE.get(key)
        if entry is None:
            launch_fn = build_layernorm_module(N, dtype_str, store_stats=store_stats, eps=eps)
            elem_dtype = dtype_to_elem_type(dtype_str)
            x_ptr = flyc.from_c_void_p(elem_dtype, x.data_ptr())
            weight_ptr = flyc.from_c_void_p(elem_dtype, weight.data_ptr())
            bias_ptr = flyc.from_c_void_p(elem_dtype, bias.data_ptr())
            out_ptr = flyc.from_c_void_p(elem_dtype, out.data_ptr())
            if store_stats:
                mean_ptr = flyc.from_c_void_p(fx.Float32, mean.data_ptr())
                rstd_ptr = flyc.from_c_void_p(fx.Float32, rstd.data_ptr())
                compiled = flyc.compile(
                    launch_fn,
                    x_ptr,
                    weight_ptr,
                    bias_ptr,
                    out_ptr,
                    mean_ptr,
                    rstd_ptr,
                    M,
                    stream,
                )
            else:
                compiled = flyc.compile(launch_fn, x_ptr, weight_ptr, bias_ptr, out_ptr, M, stream)
            _FWD_CACHE[key] = compiled
            entry = compiled
        return entry

    def layernorm_fwd(x, weight, bias, eps=EPS, store_stats=False):
        """Forward LayerNorm. Returns (out, mean, rstd). eps is baked into the kernel."""
        assert x.dim() == 2, "layernorm_fwd expects a 2D (M, N) input"
        assert (
            x.is_contiguous() and weight.is_contiguous() and bias.is_contiguous()
        ), "layernorm_fwd expects contiguous inputs"
        # The kernel reads x/weight/bias with a single elem dtype derived from x;
        # a mismatch would silently bit-reinterpret weight/bias bytes.
        assert (
            x.dtype == weight.dtype == bias.dtype
        ), f"x/weight/bias dtypes must match, got {x.dtype}/{weight.dtype}/{bias.dtype}"
        assert weight.device == x.device and bias.device == x.device, "x/weight/bias must be on the same device"
        M, N = x.shape
        assert weight.dim() == 1 and bias.dim() == 1, "weight/bias must be 1D affine vectors"
        assert weight.shape[0] == N and bias.shape[0] == N, "weight/bias length must equal x last dim (N)"
        out = torch.empty_like(x)
        mean = torch.empty((M,), device=x.device, dtype=torch.float32) if store_stats else None
        rstd = torch.empty((M,), device=x.device, dtype=torch.float32) if store_stats else None
        dtype_str = _torch_dtype_to_str(x.dtype)
        # Bind compile + launch to the tensors' device so the compiled kernel and
        # the stream belong to the right GPU/context (multi-GPU correctness).
        with torch.cuda.device(x.device):
            stream = torch.cuda.current_stream()
            compiled = _get_fwd_compiled(x, weight, bias, out, mean, rstd, M, N, dtype_str, store_stats, eps, stream)
            if store_stats:
                compiled(
                    x.data_ptr(),
                    weight.data_ptr(),
                    bias.data_ptr(),
                    out.data_ptr(),
                    mean.data_ptr(),
                    rstd.data_ptr(),
                    M,
                    stream,
                )
            else:
                compiled(x.data_ptr(), weight.data_ptr(), bias.data_ptr(), out.data_ptr(), M, stream)
        return out, mean, rstd

    def layernorm_bwd(x, weight, dout, mean, rstd, eps=EPS):
        """Backward LayerNorm. Returns (dx, dweight, dbias) cast to weight dtype.

        eps is not used directly here — it is already baked into `rstd`/`mean` by
        the forward — but is accepted so callers can pass it symmetrically.
        """
        assert x.dim() == 2, "layernorm_bwd expects a 2D (M, N) input"
        assert all(t.is_contiguous() for t in (x, weight, dout, mean, rstd)), "layernorm_bwd expects contiguous inputs"
        assert (
            x.dtype == weight.dtype == dout.dtype
        ), f"x/weight/dout dtypes must match, got {x.dtype}/{weight.dtype}/{dout.dtype}"
        M, N = x.shape
        assert dout.shape == x.shape, "dout shape must equal x shape"
        assert weight.dim() == 1 and weight.shape[0] == N, "weight must be a length-N 1D affine vector"
        assert mean.numel() == M and rstd.numel() == M, "mean/rstd length must equal x rows (M)"
        assert all(
            t.device == x.device for t in (weight, dout, mean, rstd)
        ), "x/weight/dout/mean/rstd must be on the same device"
        # mean/rstd are per-row fp32 stats saved by the forward; the kernel reads
        # them with an fp32 copy atom, so a non-fp32 dtype would be misread.
        assert mean.dtype == torch.float32 and rstd.dtype == torch.float32, "mean/rstd must be fp32"
        dtype_str = _torch_dtype_to_str(x.dtype)
        elem_dtype = dtype_to_elem_type(dtype_str)
        dx = torch.empty_like(x)
        dweight = torch.zeros((N,), device=x.device, dtype=torch.float32)
        dbias = torch.zeros((N,), device=x.device, dtype=torch.float32)
        key = (N, dtype_str, x.device)
        # Bind compile + launch to the tensors' device (multi-GPU correctness).
        with torch.cuda.device(x.device):
            stream = torch.cuda.current_stream()
            compiled = _BWD_CACHE.get(key)
            if compiled is None:
                launch_fn = build_layernorm_bwd_module(N, dtype_str)
                # flyc.compile executes the kernel once during tracing, which would
                # accumulate into DGamma/DBias; zero them AFTER compiling.
                x_ptr = flyc.from_c_void_p(elem_dtype, x.data_ptr())
                weight_ptr = flyc.from_c_void_p(elem_dtype, weight.data_ptr())
                dout_ptr = flyc.from_c_void_p(elem_dtype, dout.data_ptr())
                mean_ptr = flyc.from_c_void_p(fx.Float32, mean.data_ptr())
                rstd_ptr = flyc.from_c_void_p(fx.Float32, rstd.data_ptr())
                dx_ptr = flyc.from_c_void_p(elem_dtype, dx.data_ptr())
                dweight_ptr = flyc.from_c_void_p(fx.Float32, dweight.data_ptr())
                dbias_ptr = flyc.from_c_void_p(fx.Float32, dbias.data_ptr())
                compiled = flyc.compile(
                    launch_fn,
                    x_ptr,
                    weight_ptr,
                    dout_ptr,
                    mean_ptr,
                    rstd_ptr,
                    dx_ptr,
                    dweight_ptr,
                    dbias_ptr,
                    M,
                    stream,
                )
                _BWD_CACHE[key] = compiled
            dweight.zero_()
            dbias.zero_()
            compiled(
                x.data_ptr(),
                weight.data_ptr(),
                dout.data_ptr(),
                mean.data_ptr(),
                rstd.data_ptr(),
                dx.data_ptr(),
                dweight.data_ptr(),
                dbias.data_ptr(),
                M,
                stream,
            )
        return dx, dweight.to(weight.dtype), dbias.to(weight.dtype)

    class LayerNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, eps):
            need_grad = x.requires_grad or weight.requires_grad or bias.requires_grad
            out, mean, rstd = layernorm_fwd(x, weight, bias, eps=eps, store_stats=need_grad)
            ctx.save_for_backward(x, weight, mean, rstd)
            ctx.eps = eps
            return out

        @staticmethod
        def backward(ctx, dout):
            x, weight, mean, rstd = ctx.saved_tensors
            dx, dw, db = layernorm_bwd(x, weight, dout.contiguous(), mean, rstd, eps=ctx.eps)
            return dx, dw, db, None

    def layernorm(x, weight, bias, eps=EPS):
        """Public entry: plain LayerNorm with autograd."""
        assert weight is not None and bias is not None, "layernorm requires explicit weight and bias"
        N = weight.shape[-1]
        assert x.shape[-1] == N, f"x last dim {x.shape[-1]} != weight length {N}"
        assert (
            x.dtype == weight.dtype == bias.dtype
        ), f"x/weight/bias dtypes must match, got {x.dtype}/{weight.dtype}/{bias.dtype}"
        # Raw-pointer kernels reconstruct dense rows and stride-1 affine
        # vectors, so materialize those layouts at the public boundary.
        x_flat = x.reshape(-1, N).contiguous()
        weight_flat = weight.contiguous()
        bias_flat = bias.contiguous()
        out_flat = LayerNormFunction.apply(x_flat, weight_flat, bias_flat, eps)
        return out_flat.reshape(x.shape)
