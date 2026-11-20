# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""LayerNorm kernel builder using the @flyc.kernel API.

LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta

Two paths:
  - Vector path (16-bit dtype, N % VEC_WIDTH == 0): buffer-backed tiled copy
    over N / VEC_WIDTH tiles, input cached in registers across both passes.
  - Scalar path (fp32, or N not divisible by VEC_WIDTH): 2-pass scalar
    implementation.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T, full
from flydsl.runtime.device import get_rocm_arch
from kernels.common.kernels_common import atomic_add, dtype_to_elem_type, get_warp_size
from kernels.norm.rmsnorm_common import validate_norm_operand_dtypes

try:
    import torch
except ImportError:
    torch = None

KERNEL_NAME = "layernorm"

EPS = 1e-5

BLOCK_THREADS = 256
WARP_SIZE = get_warp_size()
VEC_WIDTH = 8


def _load_vec(copy_atom, vec_width, elem_dtype, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.copy(copy_atom, fx.slice(div_tensor, (None, idx)), r)
    return fx.memref_load_vec(r)


def _store_vec(copy_atom, vec_width, elem_dtype, val, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.memref_store_vec(val, r)
    fx.copy(copy_atom, r, fx.slice(div_tensor, (None, idx)))


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

    elem_bits = 32 if dtype_str == "f32" else 16
    USE_VEC_N = elem_bits <= 16 and N % VEC_WIDTH == 0
    VEC_TILES = N // VEC_WIDTH if USE_VEC_N else 0
    NUM_VEC_ITERS = (VEC_TILES + BLOCK_THREADS - 1) // BLOCK_THREADS if USE_VEC_N else 0

    BlockReduce = fx.coop.BlockReduce[fx.Float32x2, BLOCK_THREADS]
    SharedStorage = BlockReduce.SharedStorage

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

        # ── helpers: wave / block reduction ───────────────────────────────

        def block_reduce_add2(val0, val1):
            with fx.fastmath(fm_fast):
                values = fx.Vector.from_elements([val0, val1], fx.Float32)
                total = BlockReduce(values, fx.ReductionOp.ADD, storage=lds)
            return total[0], total[1]

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
        # Vector path for every complete f16/bf16 vec8 row.
        # ==================================================================
        if const_expr(USE_VEC_N):
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
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                vec = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx_safe)
                in_local.append(vec)
                x = vec.to(fx.Float32)

                x2 = x * x
                red = x.reduce(ReductionOp.ADD, fastmath=fm_fast)
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sum = thread_sum + is_valid.select(red, c_zero_f)
                thread_sumsq = thread_sumsq + is_valid.select(red2, c_zero_f)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            if const_expr(store_stats):
                if tid == 0:
                    mean_row[0] = mean
                    rstd_row[0] = rstd

            # ── Pass 2: normalize + affine + store ───────────────────────
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                if idx < VEC_TILES:
                    g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx).to(fx.Float32)
                    b = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, beta_div, idx).to(fx.Float32)
                    x = in_local[tile_i].to(fx.Float32)
                    y = (x - mean) * rstd
                    y = y * g + b
                    out_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)
                    _store_vec(copy_atom, VEC_WIDTH, elem_dtype, out_e, out_div, idx)

        else:
            # ==============================================================
            # Generic path: 2-pass scalar implementation for arbitrary N
            # ==============================================================
            c_zero_f = fx.Float32(0.0)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f

            # ── Pass 1: sum + sumsq ──────────────────────────────────────
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = Input_buf[idx_safe]
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
                    mean_row[0] = mean
                    rstd_row[0] = rstd

            # ── Pass 2: normalize + affine + store ───────────────────────
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = Input_buf[idx]
                    g_e = Gamma_buf[idx]
                    b_e = Beta_buf[idx]
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    diff = x - mean
                    norm = diff * rstd
                    scaled = norm * g
                    y = scaled + b
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    Output_buf[idx] = y_e

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
    BlockReduce = fx.coop.BlockReduce[fx.Float32x2, BLOCK_THREADS]
    SharedStorage = BlockReduce.SharedStorage

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

        def block_reduce_add2(val0, val1):
            with fx.fastmath(fm_fast):
                values = fx.Vector.from_elements([val0, val1], fx.Float32)
                total = BlockReduce(values, fx.ReductionOp.ADD, storage=lds)
            return total[0], total[1]

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

        mean = mean_row[0]
        rstd = rstd_row[0]
        dgamma_view = DGamma.view(fx.make_layout(N, 1))
        dbias_view = DBias.view(fx.make_layout(N, 1))

        # Pass 1: c1 = mean(wdy) ; c2 = mean(wdy * x_hat)
        thread_c1 = c_zero_f
        thread_c2 = c_zero_f
        for base in range_constexpr(0, N, BLOCK_THREADS):
            idx = tid + base
            is_valid = idx < N
            idx_safe = is_valid.select(idx, 0)
            x_e = row_in[idx_safe]
            dy_e = row_dy[idx_safe]
            g_e = gamma[idx_safe]
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
                x_e = row_in[idx]
                dy_e = row_dy[idx]
                g_e = gamma[idx]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                x_hat = (x - mean) * rstd
                wdy = dy * g
                dx = (wdy - c1 - x_hat * c2) * rstd
                dx_e = dx if dtype_str == "f32" else dx.to(elem_dtype)
                row_dx[idx] = dx_e

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


def build_fused_add_layernorm_module(N: int, dtype_str: str, eps: float = EPS):
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    elem_bits = 32 if dtype_str == "f32" else 16
    USE_VEC_N = elem_bits <= 16 and N % VEC_WIDTH == 0
    VEC_TILES = N // VEC_WIDTH if USE_VEC_N else 0
    NUM_VEC_ITERS = (VEC_TILES + BLOCK_THREADS - 1) // BLOCK_THREADS if USE_VEC_N else 0

    BlockReduce = fx.coop.BlockReduce[fx.Float32x2, BLOCK_THREADS]
    SharedStorage = BlockReduce.SharedStorage

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
        eps_c = eps

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        def block_reduce_add2(val0, val1):
            with fx.fastmath(fm_fast):
                values = fx.Vector.from_elements([val0, val1], fx.Float32)
                total = BlockReduce(values, fx.ReductionOp.ADD, storage=lds)
            return total[0], total[1]

        def compute_mean_rstd(sum_val, sumsq_val):
            inv_n = 1.0 / float(N)
            mean = sum_val * inv_n
            mean_sq = sumsq_val * inv_n
            var = mean_sq - mean * mean
            var = (var < 0.0).select(0.0, var)
            return mean, fmath.rsqrt(var + eps_c, fastmath=fm_fast)

        # ==================================================================
        # Vector path for every complete f16/bf16 vec8 row.
        # ==================================================================
        if const_expr(USE_VEC_N):
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
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                x = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx_safe).to(fx.Float32)
                residual = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, residual_in_div, idx_safe).to(fx.Float32)
                added_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, x + residual)
                added_local.append(added_e)
                added = added_e.to(fx.Float32)
                added2 = added * added
                sum_val = added.reduce(ReductionOp.ADD, fastmath=fm_fast)
                sumsq_val = added2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sum = thread_sum + is_valid.select(sum_val, c_zero_f)
                thread_sumsq = thread_sumsq + is_valid.select(sumsq_val, c_zero_f)
                if idx < VEC_TILES:
                    _store_vec(copy_atom, VEC_WIDTH, elem_dtype, added_e, residual_out_div, idx)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            # Pass 2: normalize + affine + store, reusing cached added values.
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                if idx < VEC_TILES:
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

            c_zero_f = fx.Float32(0.0)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = row_in[idx_safe]
                r_e = row_residual_in[idx_safe]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                residual = r_e if dtype_str == "f32" else r_e.to(fx.Float32)
                added_e = _to_elem_scalar(dtype_str, elem_dtype, x + residual)
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                added_safe = is_valid.select(added, c_zero_f)
                thread_sum = thread_sum + added_safe
                thread_sumsq = thread_sumsq + is_valid.select(added * added, c_zero_f)
                if idx < N:
                    row_residual_out[idx] = added_e

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    added_e = row_residual_out[idx]
                    g_e = Gamma_buf[idx]
                    b_e = Beta_buf[idx]
                    added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    y = (added - mean) * rstd
                    y = y * g + b
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    row_out[idx] = y_e

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
    eps: float = EPS,
):
    elem_bits = 32 if dtype_str == "f32" else 16
    USE_VEC_N = elem_bits <= 16 and N % VEC_WIDTH == 0
    VEC_TILES = N // VEC_WIDTH if USE_VEC_N else 0
    NUM_VEC_ITERS = (VEC_TILES + BLOCK_THREADS - 1) // BLOCK_THREADS if USE_VEC_N else 0
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)

    BlockReduce = fx.coop.BlockReduce[fx.Float32x2, BLOCK_THREADS]
    SharedStorage = BlockReduce.SharedStorage

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
        eps_c = eps
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_dtype_max = fx.Float32(quant_dtype_max)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        YScale_buf = fx.rocdl.make_buffer_tensor(YScale)

        def block_reduce_add2(val0, val1):
            with fx.fastmath(fm_fast):
                values = fx.Vector.from_elements([val0, val1], fx.Float32)
                total = BlockReduce(values, fx.ReductionOp.ADD, storage=lds)
            return total[0], total[1]

        def block_reduce_max(val):
            # Finish readers of the previous sum before reusing its shared slots.
            gpu.barrier()
            values = fx.full(2, val, fx.Float32)
            return BlockReduce(values, fx.ReductionOp.MAX, storage=lds)[0]

        # ==================================================================
        # Vector path for every complete f16/bf16 vec8 row.
        # ==================================================================
        if const_expr(USE_VEC_N):
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
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx_safe)
                norm_input_local.append(x_e)
                x_norm = x_e.to(fx.Float32)
                x2 = x_norm * x_norm
                red = x_norm.reduce(ReductionOp.ADD, fastmath=fm_fast)
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sum = thread_sum + is_valid.select(red, c_zero_f)
                thread_sumsq = thread_sumsq + is_valid.select(red2, c_zero_f)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean = sum_val / n_float
            var = sumsq_val / n_float - mean * mean
            var = (var < c_zero_f).select(c_zero_f, var)
            rstd = fmath.rsqrt(var + eps_c, fastmath=fm_fast)

            thread_row_max = c_zero_f
            y_local = []

            # Pass 2: affine (+ optional smooth scale), cache y, accumulate row max.
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                x = norm_input_local[tile_i].to(fx.Float32)
                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx_safe).to(fx.Float32)
                b = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, beta_div, idx_safe).to(fx.Float32)
                y = (x - mean) * rstd
                y = y * g + b
                if const_expr(is_smooth):
                    s = _load_vec(copy_atom_xs, VEC_WIDTH, elem_dtype, xscale_div, idx_safe).to(fx.Float32)
                    y = y * s
                y_local.append(y)
                y_abs = (y.bitcast(fx.Uint32) & abs_mask).bitcast(fx.Float32)
                tile_max = y_abs.reduce(ReductionOp.MAX)
                thread_row_max = fx.max(thread_row_max, is_valid.select(tile_max, c_zero_f))

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                YScale_buf[bid] = final_scale

            inv_scale = c_one_f / final_scale

            # Pass 3: quantize + store using per-row scale.
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                if idx < VEC_TILES:
                    q = y_local[tile_i] * inv_scale
                    q_i8 = q.to(quant_dtype)
                    q_lo = q_i8.shuffle(q_i8, [0, 1, 2, 3])
                    q_hi = q_i8.shuffle(q_i8, [4, 5, 6, 7])
                    out_idx = idx * 2
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
                x_e = row_in[idx_safe]
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
                x_e = row_in[idx_safe]
                g_e = Gamma_buf[idx_safe]
                b_e = Beta_buf[idx_safe]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                y = (x - mean) * rstd
                y = y * g + b
                if const_expr(is_smooth):
                    s_e = XScale_buf[idx_safe]
                    s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                    y = y * s
                y_abs = _abs_scalar(y)
                thread_row_max = fx.max(thread_row_max, is_valid.select(y_abs, c_zero_f))

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                YScale_buf[bid] = final_scale

            inv_scale = c_one_f / final_scale

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = row_in[idx]
                    g_e = Gamma_buf[idx]
                    b_e = Beta_buf[idx]
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    y = (x - mean) * rstd
                    y = y * g + b
                    if const_expr(is_smooth):
                        s_e = XScale_buf[idx]
                        s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                        y = y * s
                    q = y * inv_scale
                    q_i8 = q.to(quant_dtype)
                    row_out[idx] = q_i8

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
            validate_norm_operand_dtypes(Input, dtype_str, Gamma=Gamma, Beta=Beta, XScale=XScale)
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
            validate_norm_operand_dtypes(Input, dtype_str, Gamma=Gamma, Beta=Beta)
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
    eps: float = EPS,
):
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    elem_bits = 32 if dtype_str == "f32" else 16
    USE_VEC_N = elem_bits <= 16 and N % VEC_WIDTH == 0
    VEC_TILES = N // VEC_WIDTH if USE_VEC_N else 0
    NUM_VEC_ITERS = (VEC_TILES + BLOCK_THREADS - 1) // BLOCK_THREADS if USE_VEC_N else 0
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)

    BlockReduce = fx.coop.BlockReduce[fx.Float32x2, BLOCK_THREADS]
    SharedStorage = BlockReduce.SharedStorage

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
        eps_c = eps
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_dtype_max = fx.Float32(quant_dtype_max)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        YScale_buf = fx.rocdl.make_buffer_tensor(YScale)

        def block_reduce_add2(val0, val1):
            with fx.fastmath(fm_fast):
                values = fx.Vector.from_elements([val0, val1], fx.Float32)
                total = BlockReduce(values, fx.ReductionOp.ADD, storage=lds)
            return total[0], total[1]

        def block_reduce_max(val):
            # Finish readers of the previous sum before reusing its shared slots.
            gpu.barrier()
            values = fx.full(2, val, fx.Float32)
            return BlockReduce(values, fx.ReductionOp.MAX, storage=lds)[0]

        # ==================================================================
        # Vector path for every complete f16/bf16 vec8 row.
        # ==================================================================
        if const_expr(USE_VEC_N):
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
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                x = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx_safe).to(fx.Float32)
                residual = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, residual_in_div, idx_safe).to(fx.Float32)
                added_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, x + residual)
                norm_input_local.append(added_e)
                x_norm = added_e.to(fx.Float32)
                if idx < VEC_TILES:
                    _store_vec(copy_atom, VEC_WIDTH, elem_dtype, added_e, residual_out_div, idx)
                x2 = x_norm * x_norm
                red = x_norm.reduce(ReductionOp.ADD, fastmath=fm_fast)
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sum = thread_sum + is_valid.select(red, c_zero_f)
                thread_sumsq = thread_sumsq + is_valid.select(red2, c_zero_f)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean = sum_val / n_float
            var = sumsq_val / n_float - mean * mean
            var = (var < c_zero_f).select(c_zero_f, var)
            rstd = fmath.rsqrt(var + eps_c, fastmath=fm_fast)

            thread_row_max = c_zero_f
            y_local = []

            # Pass 2: affine (+ optional smooth scale), cache y, accumulate row max.
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                x = norm_input_local[tile_i].to(fx.Float32)
                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx_safe).to(fx.Float32)
                b = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, beta_div, idx_safe).to(fx.Float32)
                y = (x - mean) * rstd
                y = y * g + b
                if const_expr(is_smooth):
                    s = _load_vec(copy_atom_xs, VEC_WIDTH, elem_dtype, xscale_div, idx_safe).to(fx.Float32)
                    y = y * s
                y_local.append(y)
                y_abs = (y.bitcast(fx.Uint32) & abs_mask).bitcast(fx.Float32)
                tile_max = y_abs.reduce(ReductionOp.MAX)
                thread_row_max = fx.max(thread_row_max, is_valid.select(tile_max, c_zero_f))

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                YScale_buf[bid] = final_scale

            inv_scale = c_one_f / final_scale

            # Pass 3: quantize + store using per-row scale.
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                if idx < VEC_TILES:
                    q = y_local[tile_i] * inv_scale
                    q_i8 = q.to(quant_dtype)
                    q_lo = q_i8.shuffle(q_i8, [0, 1, 2, 3])
                    q_hi = q_i8.shuffle(q_i8, [4, 5, 6, 7])
                    out_idx = idx * 2
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
                x_e = row_in[idx_safe]
                r_e = row_residual_in[idx_safe]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                residual = r_e if dtype_str == "f32" else r_e.to(fx.Float32)
                added_e = _to_elem_scalar(dtype_str, elem_dtype, x + residual)
                if idx < N:
                    row_residual_out[idx] = added_e
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
                x_e = row_residual_out[idx_safe]
                g_e = Gamma_buf[idx_safe]
                b_e = Beta_buf[idx_safe]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                y = (x - mean) * rstd
                y = y * g + b
                if const_expr(is_smooth):
                    s_e = XScale_buf[idx_safe]
                    s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                    y = y * s
                y_abs = _abs_scalar(y)
                thread_row_max = fx.max(thread_row_max, is_valid.select(y_abs, c_zero_f))

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                YScale_buf[bid] = final_scale

            inv_scale = c_one_f / final_scale

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = row_residual_out[idx]
                    g_e = Gamma_buf[idx]
                    b_e = Beta_buf[idx]
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    y = (x - mean) * rstd
                    y = y * g + b
                    if const_expr(is_smooth):
                        s_e = XScale_buf[idx]
                        s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                        y = y * s
                    q = y * inv_scale
                    q_i8 = q.to(quant_dtype)
                    row_out[idx] = q_i8

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
            validate_norm_operand_dtypes(
                Input, dtype_str, ResidualIn=ResidualIn, Gamma=Gamma, Beta=Beta, XScale=XScale, ResidualOut=ResidualOut
            )
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
            validate_norm_operand_dtypes(
                Input, dtype_str, ResidualIn=ResidualIn, Gamma=Gamma, Beta=Beta, ResidualOut=ResidualOut
            )
            launcher = fused_add_layernorm_quant_kernel(
                Input, ResidualIn, Gamma, Beta, Gamma, YScale, Output, ResidualOut
            )
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

    return launch_fused_add_layernorm_dynamicquant


def _layernorm_smoothquant_vec_config(N: int, dtype_str: str, block_threads: int | None = None):
    """Pick the 128-bit-vector geometry for the smooth-quant kernel.

    A vector carries four fp32 or eight fp16/bf16 values; a row that is not a
    whole number of vectors finishes with a scalar tail.  Small rows are
    handled by one wave (or a sub-wave) per block; wide rows get enough
    threads to keep several 128-bit requests in flight per row.  Rows shorter
    than one vector keep the scalar builder.
    """
    vec_width = 4 if dtype_str == "f32" else VEC_WIDTH
    if N < vec_width:
        return None
    vec_tiles = N // vec_width
    if block_threads is None:
        # A full wave is the smallest useful launch: sub-wave blocks add
        # scheduling overhead without reducing the work.  Up to four waves
        # keeps the block reduction cheap; wider rows use two iterations at
        # eight waves, which measured better than 1x16-wave blocks.
        if vec_tiles <= 64:
            block_threads = 64
        elif vec_tiles <= 128:
            block_threads = 128
        elif vec_tiles <= 1536:
            block_threads = 256
        else:
            block_threads = 512
    return vec_width, vec_tiles, block_threads


def _build_layernorm_smoothquant_vec_module(
    N: int,
    dtype_str: str,
    *,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
    block_threads: int | None = None,
    safe_tail_limit: int = 512,
    input_nt: bool = False,
):
    """Vectorized, register-cached smooth-quant kernel.

    The input and the unquantized row are kept in registers between the
    statistics, row-max and store passes.  That removes the two extra global
    reads of the stock scalar path and gives every 128-bit transaction a
    single use.  One block owns one row, so no inter-block synchronisation is
    needed.
    """
    cfg = _layernorm_smoothquant_vec_config(N, dtype_str, block_threads)
    if cfg is None:
        raise ValueError(f"vector smooth-quant geometry requires N of at least one vector, got N={N}")
    vec_width, vec_tiles, block_threads = cfg
    elem_bits = 32 if dtype_str == "f32" else 16
    num_vec_iters = vec_tiles // block_threads
    tail_tiles = vec_tiles - num_vec_iters * block_threads
    scalar_tail = N - vec_tiles * vec_width
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)
    inv_quant_dtype_max = 1.0 / quant_dtype_max
    quant_word = 4
    quant_words_per_vec = vec_width // quant_word

    # At most one physical wave gets a wave reduction; wider blocks use the
    # cooperative block reduction.  Keeping the small-row case barrier-free
    # matters for the decode-sized shapes.
    warp_width = get_warp_size()
    if block_threads <= warp_width:
        BlockReduce = None
        SharedStorage = None
    else:
        BlockReduce = fx.coop.BlockReduce[fx.Float32x2, block_threads]
        SharedStorage = BlockReduce.SharedStorage

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def layernorm_smoothquant_vec_kernel(
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
        eps_c = eps
        inv_n = 1.0 / float(N)
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_inv_dtype_max = fx.Float32(inv_quant_dtype_max)
        abs_mask = full(vec_width, fx.Uint32(0x7FFFFFFF), fx.Uint32)

        if const_expr(block_threads <= warp_width):

            def block_reduce_add2(val0, val1):
                values = fx.Vector.from_elements([val0, val1], fx.Float32)
                with fx.fastmath(fm_fast):
                    total = fx.coop.warp_reduce(values, fx.ReductionOp.ADD, width=block_threads, _dtype=fx.Float32x2)
                return total[0], total[1]

            def block_reduce_max(val):
                with fx.fastmath(fm_fast):
                    return fx.coop.warp_reduce(val, fx.ReductionOp.MAX, width=block_threads)

        else:
            # Distinct scratch lets the two reductions proceed without the
            # extra block-wide barrier that reusing one set of slots needs.
            shared = fx.SharedAllocator()
            lds_sum = shared.allocate(SharedStorage).peek()
            lds_max = shared.allocate(SharedStorage).peek()

            def block_reduce_add2(val0, val1):
                values = fx.Vector.from_elements([val0, val1], fx.Float32)
                with fx.fastmath(fm_fast):
                    total = BlockReduce(values, fx.ReductionOp.ADD, storage=lds_sum)
                return total[0], total[1]

            def block_reduce_max(val):
                values = fx.full(2, val, fx.Float32)
                with fx.fastmath(fm_fast):
                    return BlockReduce(values, fx.ReductionOp.MAX, storage=lds_max)[0]

        Input_buf = fx.rocdl.make_buffer_tensor(Input)
        Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
        Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
        XScale_buf = fx.rocdl.make_buffer_tensor(XScale)
        YScale_buf = fx.rocdl.make_buffer_tensor(YScale)
        Output_buf = fx.rocdl.make_buffer_tensor(Output)

        row_in = fx.slice(Input_buf, (bid, None))
        row_out = fx.slice(Output_buf, (bid, None))
        in_div = fx.logical_divide(row_in, fx.make_layout(vec_width, 1))
        out_div_q = fx.logical_divide(row_out, fx.make_layout(quant_word, 1))
        gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(vec_width, 1))
        beta_div = fx.logical_divide(Beta_buf, fx.make_layout(vec_width, 1))
        xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(vec_width, 1))

        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
        copy_atom_x = fx.make_copy_atom(fx.rocdl.BufferCopy128b(2 if input_nt else 0), elem_bits)
        copy_atom_q = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 8)

        sum_vec = fx.full(vec_width, c_zero_f, fx.Float32)
        sumsq_vec = fx.full(vec_width, c_zero_f, fx.Float32)
        thread_sum = c_zero_f
        thread_sumsq = c_zero_f
        x_local = []
        gamma_local = []
        beta_local = []
        xscale_local = []

        # Pass 1: launch every cold HBM request (row and the three shared
        # affine vectors) before waiting on the reduction, and cache them for
        # the later passes.  When the data is not in cache this turns three
        # dependent round trips into one.
        for tile_i in range_constexpr(num_vec_iters):
            idx = tid + tile_i * block_threads
            x_e = _load_vec(copy_atom_x, vec_width, elem_dtype, in_div, idx)
            g_e = _load_vec(copy_atom, vec_width, elem_dtype, gamma_div, idx)
            b_e = _load_vec(copy_atom, vec_width, elem_dtype, beta_div, idx)
            s_e = _load_vec(copy_atom, vec_width, elem_dtype, xscale_div, idx)
            x_local.append(x_e)
            gamma_local.append(g_e)
            beta_local.append(b_e)
            xscale_local.append(s_e)
            x = x_e.to(fx.Float32)
            if const_expr(dtype_str == "f32"):
                sum_vec = sum_vec + x
                sumsq_vec = sumsq_vec + x * x
            else:
                x2 = x * x
                thread_sum = thread_sum + x.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + x2.reduce(ReductionOp.ADD, fastmath=fm_fast)

        if const_expr(dtype_str == "f32"):
            thread_sum = sum_vec.reduce(ReductionOp.ADD, fastmath=fm_fast)
            thread_sumsq = sumsq_vec.reduce(ReductionOp.ADD, fastmath=fm_fast)

        # The final partial vector tile: for a short tail the uniform
        # safe-index load is cheaper than a divergent branch; a long tail
        # avoids the redundant traffic and carries zeroes instead.
        if const_expr(tail_tiles > 0):
            tail_idx = tid + num_vec_iters * block_threads
            tail_valid = tail_idx < vec_tiles
            if const_expr(tail_tiles <= safe_tail_limit):
                tail_safe = tail_valid.select(tail_idx, 0)
                x_e = _load_vec(copy_atom_x, vec_width, elem_dtype, in_div, tail_safe)
                g_e = _load_vec(copy_atom, vec_width, elem_dtype, gamma_div, tail_safe)
                b_e = _load_vec(copy_atom, vec_width, elem_dtype, beta_div, tail_safe)
                s_e = _load_vec(copy_atom, vec_width, elem_dtype, xscale_div, tail_safe)
            else:
                zero_e = fx.full(vec_width, fx.Float32(0.0).to(elem_dtype), elem_dtype)
                x_e = zero_e
                g_e = zero_e
                b_e = zero_e
                s_e = zero_e
                if tail_valid:
                    x_e = _load_vec(copy_atom, vec_width, elem_dtype, in_div, tail_idx)
                    g_e = _load_vec(copy_atom, vec_width, elem_dtype, gamma_div, tail_idx)
                    b_e = _load_vec(copy_atom, vec_width, elem_dtype, beta_div, tail_idx)
                    s_e = _load_vec(copy_atom, vec_width, elem_dtype, xscale_div, tail_idx)
            x_local.append(x_e)
            gamma_local.append(g_e)
            beta_local.append(b_e)
            xscale_local.append(s_e)
            x = x_e.to(fx.Float32)
            x2 = x * x
            tile_sum = x.reduce(ReductionOp.ADD, fastmath=fm_fast)
            tile_sumsq = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
            if const_expr(tail_tiles <= safe_tail_limit):
                tile_sum = tail_valid.select(tile_sum, c_zero_f)
                tile_sumsq = tail_valid.select(tile_sumsq, c_zero_f)
            thread_sum = thread_sum + tile_sum
            thread_sumsq = thread_sumsq + tile_sumsq

        # Non-vector-multiple rows finish with fewer than one vector of
        # scalar values.  Keeping those in the same cached pass means an
        # arbitrary N still gets the single-read fast path.
        tail_x = c_zero_f
        tail_g = c_zero_f
        tail_b = c_zero_f
        tail_s = c_zero_f
        if const_expr(scalar_tail > 0):
            if tid < scalar_tail:
                scalar_idx = vec_tiles * vec_width + tid
                tail_x = row_in[scalar_idx].to(fx.Float32)
                tail_g = Gamma_buf[scalar_idx].to(fx.Float32)
                tail_b = Beta_buf[scalar_idx].to(fx.Float32)
                tail_s = XScale_buf[scalar_idx].to(fx.Float32)
                thread_sum = thread_sum + tail_x
                thread_sumsq = thread_sumsq + tail_x * tail_x

        sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
        mean = sum_val * inv_n
        var = sumsq_val * inv_n - mean * mean
        var = (var < c_zero_f).select(c_zero_f, var)
        rstd = fmath.rsqrt(var + eps_c, fastmath=fm_fast)
        mean_rstd = mean * rstd

        thread_row_max = c_zero_f
        y_local = []

        # Pass 2: normalize + affine + smooth scale, cached for quantization.
        for tile_i in range_constexpr(num_vec_iters):
            x = x_local[tile_i].to(fx.Float32)
            g = gamma_local[tile_i].to(fx.Float32)
            b = beta_local[tile_i].to(fx.Float32)
            s = xscale_local[tile_i].to(fx.Float32)
            # Fold the centering and the RMS scale into one FMA.  The
            # algebraically identical form removes one packed FP32 operation
            # per element from the hot pass.
            xhat = x * rstd - mean_rstd
            y = xhat * g + b
            y = y * s
            y_local.append(y)
            y_abs = (y.bitcast(fx.Uint32) & abs_mask).bitcast(fx.Float32)
            thread_row_max = fx.max(
                thread_row_max,
                y_abs.reduce(ReductionOp.MAX, fastmath=fm_fast),
            )

        if const_expr(tail_tiles > 0):
            tail_slot = num_vec_iters
            x = x_local[tail_slot].to(fx.Float32)
            g = gamma_local[tail_slot].to(fx.Float32)
            b = beta_local[tail_slot].to(fx.Float32)
            s = xscale_local[tail_slot].to(fx.Float32)
            xhat = x * rstd - mean_rstd
            y = xhat * g + b
            y = y * s
            y_local.append(y)
            y_abs = (y.bitcast(fx.Uint32) & abs_mask).bitcast(fx.Float32)
            tail_max = y_abs.reduce(ReductionOp.MAX, fastmath=fm_fast)
            if const_expr(tail_tiles <= safe_tail_limit):
                tail_max = tail_valid.select(tail_max, c_zero_f)
            thread_row_max = fx.max(thread_row_max, tail_max)

        tail_y = c_zero_f
        if const_expr(scalar_tail > 0):
            # Invalid lanes carry zeroes and therefore contribute a neutral
            # zero to the reduction and the later quantized store.
            xhat = tail_x * rstd - mean_rstd
            tail_y = (xhat * tail_g + tail_b) * tail_s
            thread_row_max = fx.max(thread_row_max, fx.absf(tail_y))

        row_max = block_reduce_max(thread_row_max)
        scale = row_max * c_inv_dtype_max
        final_scale = (scale == c_zero_f).select(c_one_f, scale)
        if tid == 0:
            YScale_buf[bid] = final_scale
        inv_scale = fx.rocdl.rcp(T.f32, final_scale)

        # Pass 3: quantize from registers and emit 32-bit packed stores.
        for tile_i in range_constexpr(num_vec_iters):
            idx = tid + tile_i * block_threads
            q_i8 = (y_local[tile_i] * inv_scale).to(quant_dtype)
            for word_i in range_constexpr(quant_words_per_vec):
                lo = word_i * quant_word
                q_word = q_i8.shuffle(q_i8, [lo, lo + 1, lo + 2, lo + 3])
                _store_vec(
                    copy_atom_q,
                    quant_word,
                    quant_dtype,
                    q_word,
                    out_div_q,
                    idx * quant_words_per_vec + word_i,
                )

        if const_expr(tail_tiles > 0):
            tail_idx = tid + num_vec_iters * block_threads
            if tail_idx < vec_tiles:
                tail_slot = num_vec_iters
                q_i8 = (y_local[tail_slot] * inv_scale).to(quant_dtype)
                for word_i in range_constexpr(quant_words_per_vec):
                    lo = word_i * quant_word
                    q_word = q_i8.shuffle(q_i8, [lo, lo + 1, lo + 2, lo + 3])
                    _store_vec(
                        copy_atom_q,
                        quant_word,
                        quant_dtype,
                        q_word,
                        out_div_q,
                        tail_idx * quant_words_per_vec + word_i,
                    )

        if const_expr(scalar_tail > 0):
            if tid < scalar_tail:
                scalar_idx = vec_tiles * vec_width + tid
                row_out[scalar_idx] = (tail_y * inv_scale).to(quant_dtype)

    @flyc.jit
    def launch_layernorm_smoothquant_vec(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        XScale: fx.Tensor,
        Output: fx.Tensor,
        YScale: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        validate_norm_operand_dtypes(Input, dtype_str, Gamma=Gamma, Beta=Beta, XScale=XScale)
        launcher = layernorm_smoothquant_vec_kernel(Input, Gamma, Beta, XScale, YScale, Output)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_layernorm_smoothquant_vec


def _build_layernorm_smoothquant_small_n_module(
    N: int,
    dtype_str: str,
    *,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
    rows_per_block: int = 4,
):
    """One-vector-per-lane variant for short rows.

    ``GROUP`` lanes cooperate on one row with a logical-warp reduction.  A
    block therefore advances ``rows_per_block`` independent rows at once,
    reusing the row-invariant affine vectors from the same cache lines.  This
    is useful when a row is shorter than a full wave and one-row blocks would
    leave most lanes idle.
    """
    cfg = _layernorm_smoothquant_vec_config(N, dtype_str)
    if cfg is None:
        raise ValueError("small-N smooth quant requires a vectorized row")
    vec_width, vec_tiles, _ = cfg
    scalar_tail = N - vec_tiles * vec_width
    group = 1 << (max(vec_tiles, scalar_tail, 1) - 1).bit_length()
    if group > get_warp_size():
        raise ValueError("small-N smooth quant requires a row group no wider than a wave")
    block_threads = group * rows_per_block
    if block_threads > 1024:
        raise ValueError("rows_per_block is too large for this row width")
    elem_bits = 32 if dtype_str == "f32" else 16
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)
    inv_quant_dtype_max = 1.0 / quant_dtype_max
    quant_word = 4
    quant_words_per_vec = vec_width // quant_word

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def layernorm_smoothquant_small_n_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        XScale: fx.Tensor,
        YScale: fx.Tensor,
        Output: fx.Tensor,
        MIn: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        lane = tid % group
        row = bid * rows_per_block + tid // group
        if row < MIn:
            elem_dtype = dtype_to_elem_type(dtype_str)
            quant_dtype = _quant_dtype_to_elem_type(quant_dtype_str)
            fm_fast = arith.FastMathFlags.fast
            inv_n = 1.0 / float(N)
            c_zero_f = fx.Float32(0.0)
            c_one_f = fx.Float32(1.0)
            c_inv_dtype_max = fx.Float32(inv_quant_dtype_max)
            abs_mask = full(vec_width, fx.Uint32(0x7FFFFFFF), fx.Uint32)

            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
            XScale_buf = fx.rocdl.make_buffer_tensor(XScale)
            YScale_buf = fx.rocdl.make_buffer_tensor(YScale)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)

            row_in = fx.slice(Input_buf, (row, None))
            row_out = fx.slice(Output_buf, (row, None))
            in_div = fx.logical_divide(row_in, fx.make_layout(vec_width, 1))
            out_div_q = fx.logical_divide(row_out, fx.make_layout(quant_word, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(vec_width, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(vec_width, 1))
            xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(vec_width, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            copy_atom_q = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 8)

            zero_e = fx.full(vec_width, fx.Float32(0.0).to(elem_dtype), elem_dtype)
            x_e = zero_e
            g_e = zero_e
            b_e = zero_e
            s_e = zero_e
            if lane < vec_tiles:
                x_e = _load_vec(copy_atom, vec_width, elem_dtype, in_div, lane)
                g_e = _load_vec(copy_atom, vec_width, elem_dtype, gamma_div, lane)
                b_e = _load_vec(copy_atom, vec_width, elem_dtype, beta_div, lane)
                s_e = _load_vec(copy_atom, vec_width, elem_dtype, xscale_div, lane)
            x = x_e.to(fx.Float32)
            x2 = x * x
            thread_sum = x.reduce(ReductionOp.ADD, fastmath=fm_fast)
            thread_sumsq = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)

            tail_x = c_zero_f
            tail_g = c_zero_f
            tail_b = c_zero_f
            tail_s = c_zero_f
            if const_expr(scalar_tail > 0):
                if lane < scalar_tail:
                    scalar_idx = vec_tiles * vec_width + lane
                    tail_x = row_in[scalar_idx].to(fx.Float32)
                    tail_g = Gamma_buf[scalar_idx].to(fx.Float32)
                    tail_b = Beta_buf[scalar_idx].to(fx.Float32)
                    tail_s = XScale_buf[scalar_idx].to(fx.Float32)
                    thread_sum = thread_sum + tail_x
                    thread_sumsq = thread_sumsq + tail_x * tail_x

            pair = fx.Vector.from_elements([thread_sum, thread_sumsq], fx.Float32)
            sums = fx.coop.warp_reduce(pair, fx.ReductionOp.ADD, width=group, _dtype=fx.Float32x2)
            mean = sums[0] * inv_n
            var = sums[1] * inv_n - mean * mean
            var = (var < c_zero_f).select(c_zero_f, var)
            rstd = fmath.rsqrt(var + eps, fastmath=fm_fast)
            mean_rstd = mean * rstd

            xhat = x * rstd - mean_rstd
            y = xhat * g_e.to(fx.Float32) + b_e.to(fx.Float32)
            y = y * s_e.to(fx.Float32)
            y_abs = (y.bitcast(fx.Uint32) & abs_mask).bitcast(fx.Float32)
            thread_row_max = y_abs.reduce(ReductionOp.MAX, fastmath=fm_fast)

            tail_y = c_zero_f
            if const_expr(scalar_tail > 0):
                if lane < scalar_tail:
                    xhat_tail = tail_x * rstd - mean_rstd
                    tail_y = (xhat_tail * tail_g + tail_b) * tail_s
                    thread_row_max = fx.max(thread_row_max, fx.absf(tail_y))

            row_max = fx.coop.warp_reduce(
                thread_row_max,
                fx.ReductionOp.MAX,
                width=group,
            )
            scale = row_max * c_inv_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)
            if lane == 0:
                YScale_buf[row] = final_scale
            inv_scale = fx.rocdl.rcp(T.f32, final_scale)

            if lane < vec_tiles:
                q_i8 = (y * inv_scale).to(quant_dtype)
                for word_i in range_constexpr(quant_words_per_vec):
                    lo = word_i * quant_word
                    q_word = q_i8.shuffle(q_i8, [lo, lo + 1, lo + 2, lo + 3])
                    _store_vec(
                        copy_atom_q,
                        quant_word,
                        quant_dtype,
                        q_word,
                        out_div_q,
                        lane * quant_words_per_vec + word_i,
                    )
            if const_expr(scalar_tail > 0):
                if lane < scalar_tail:
                    scalar_idx = vec_tiles * vec_width + lane
                    row_out[scalar_idx] = (tail_y * inv_scale).to(quant_dtype)

    @flyc.jit
    def launch_layernorm_smoothquant_small_n(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        XScale: fx.Tensor,
        Output: fx.Tensor,
        YScale: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        validate_norm_operand_dtypes(Input, dtype_str, Gamma=Gamma, Beta=Beta, XScale=XScale)
        launcher = layernorm_smoothquant_small_n_kernel(Input, Gamma, Beta, XScale, YScale, Output, m_in)
        launcher.launch(
            grid=((m_in + rows_per_block - 1) // rows_per_block, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_layernorm_smoothquant_small_n


def _small_n_rows_per_block(group: int, dense: bool, scalar_tail: int = 0) -> int:
    """Rows per block for the short-row kernel.

    Sparse blocks (one row per block) keep small-M launches spread over the
    CUs; dense blocks pack several row groups into each launch for throughput
    once M is large enough to fill the grid.  A row with a scalar remainder
    carries extra per-lane work, so it uses the smaller dense block.
    """
    if not dense:
        return 1
    threads = 64 if group <= 32 and scalar_tail else 256
    return max(1, threads // group)


def _build_layernorm_smoothquant_small_n_adaptive_module(
    N: int,
    dtype_str: str,
    *,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    cfg = _layernorm_smoothquant_vec_config(N, dtype_str)
    if cfg is None:
        raise ValueError("small-N smooth quant requires a vectorized row")
    vec_width, vec_tiles, _ = cfg
    scalar_tail = N - vec_tiles * vec_width
    group = 1 << (max(vec_tiles, scalar_tail, 1) - 1).bit_length()
    if group <= 16:
        # Several row groups share a physical wave here, so the dense layout
        # is also the low-latency one.
        sparse = None
        dense = _build_layernorm_smoothquant_small_n_module(
            N,
            dtype_str,
            quant_dtype_str=quant_dtype_str,
            eps=eps,
            rows_per_block=_small_n_rows_per_block(group, True, scalar_tail),
        )
        small = None
        if not scalar_tail:
            small = _build_layernorm_smoothquant_small_n_module(
                N,
                dtype_str,
                quant_dtype_str=quant_dtype_str,
                eps=eps,
                rows_per_block=max(1, 128 // group),
            )
    else:
        sparse = _build_layernorm_smoothquant_small_n_module(
            N,
            dtype_str,
            quant_dtype_str=quant_dtype_str,
            eps=eps,
            rows_per_block=1,
        )
        dense = _build_layernorm_smoothquant_small_n_module(
            N,
            dtype_str,
            quant_dtype_str=quant_dtype_str,
            eps=eps,
            rows_per_block=_small_n_rows_per_block(group, True, scalar_tail),
        )

    @flyc.jit
    def launch_layernorm_smoothquant_small_n(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        XScale: fx.Tensor,
        Output: fx.Tensor,
        YScale: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        validate_norm_operand_dtypes(Input, dtype_str, Gamma=Gamma, Beta=Beta, XScale=XScale)

        def launch_dense():
            dense(Input, Gamma, Beta, XScale, Output, YScale, m_in, stream)

        if const_expr(group <= 16):
            if const_expr(scalar_tail == 0):

                def launch_small():
                    small(Input, Gamma, Beta, XScale, Output, YScale, m_in, stream)

                @flyc.jit
                def dispatch_exact():
                    if m_in >= fx.Int32(1024):
                        launch_small()
                    else:
                        launch_dense()

                dispatch_exact()
            else:
                launch_dense()
        else:

            def launch_sparse():
                sparse(Input, Gamma, Beta, XScale, Output, YScale, m_in, stream)

            @flyc.jit
            def dispatch():
                # Small launches benefit from spreading one-row blocks over
                # more CUs; large launches pack row groups to cut block
                # scheduling, and the middle band stays sparse because a
                # small integer number of dense blocks would underfill the
                # machine.
                if (m_in >= fx.Int32(256)) | (m_in < fx.Int32(32)):
                    launch_dense()
                else:
                    launch_sparse()

            dispatch()

    return launch_layernorm_smoothquant_small_n


def build_layernorm_dynamicquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    return _build_layernorm_quant_module(
        N,
        dtype_str,
        is_smooth=False,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
    )


def build_layernorm_smoothquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    # Short, power-of-two vector rows can be packed several to a block; wider
    # rows use the one-row register-cached vector path.  Arbitrary tails keep
    # the correctness-complete scalar implementation.
    cfg = _layernorm_smoothquant_vec_config(N, dtype_str)
    if cfg is not None:
        _, vec_tiles, _ = cfg
        vec_width = cfg[0]
        scalar_tail = N - vec_tiles * vec_width
        group = 1 << (max(vec_tiles, scalar_tail, 1) - 1).bit_length()
        if vec_tiles <= get_warp_size() and group <= get_warp_size():
            return _build_layernorm_smoothquant_small_n_adaptive_module(
                N,
                dtype_str,
                quant_dtype_str=quant_dtype_str,
                eps=eps,
            )
        default_threads = _layernorm_smoothquant_vec_config(N, dtype_str)[2]
        if vec_tiles <= 896:
            throughput_threads = 64
        elif vec_tiles <= 1792:
            throughput_threads = 256
        else:
            throughput_threads = default_threads
        default_launch = _build_layernorm_smoothquant_vec_module(
            N,
            dtype_str,
            quant_dtype_str=quant_dtype_str,
            eps=eps,
            block_threads=default_threads,
        )
        if dtype_str == "f32" and vec_tiles > 1792:
            # Streaming x avoids evicting pending output lines for these
            # very wide f32 rows; it is neutral to positive for both cold
            # and warm evaluation across row counts.
            return _build_layernorm_smoothquant_vec_module(
                N,
                dtype_str,
                quant_dtype_str=quant_dtype_str,
                eps=eps,
                block_threads=default_threads,
                input_nt=True,
            )

        if dtype_str == "f32" and 65 <= vec_tiles <= 896:
            nt_launch = _build_layernorm_smoothquant_vec_module(
                N,
                dtype_str,
                quant_dtype_str=quant_dtype_str,
                eps=eps,
                block_threads=default_threads,
                input_nt=True,
            )

            if vec_tiles >= 257:

                @flyc.jit
                def launch_layernorm_smoothquant_nt_rows(
                    Input: fx.Tensor,
                    Gamma: fx.Tensor,
                    Beta: fx.Tensor,
                    XScale: fx.Tensor,
                    Output: fx.Tensor,
                    YScale: fx.Tensor,
                    m_in: fx.Int32,
                    stream: fx.Stream = fx.Stream(None),
                ):
                    validate_norm_operand_dtypes(Input, dtype_str, Gamma=Gamma, Beta=Beta, XScale=XScale)

                    def launch_cached():
                        default_launch(Input, Gamma, Beta, XScale, Output, YScale, m_in, stream)

                    def launch_streaming():
                        nt_launch(Input, Gamma, Beta, XScale, Output, YScale, m_in, stream)

                    @flyc.jit
                    def dispatch_nt_rows():
                        if m_in >= fx.Int32(512):
                            launch_streaming()
                        else:
                            launch_cached()

                    dispatch_nt_rows()

                return launch_layernorm_smoothquant_nt_rows

            mid_launch = _build_layernorm_smoothquant_vec_module(
                N,
                dtype_str,
                quant_dtype_str=quant_dtype_str,
                eps=eps,
                block_threads=64,
            )

            @flyc.jit
            def launch_layernorm_smoothquant_nt_mid(
                Input: fx.Tensor,
                Gamma: fx.Tensor,
                Beta: fx.Tensor,
                XScale: fx.Tensor,
                Output: fx.Tensor,
                YScale: fx.Tensor,
                m_in: fx.Int32,
                stream: fx.Stream = fx.Stream(None),
            ):
                validate_norm_operand_dtypes(Input, dtype_str, Gamma=Gamma, Beta=Beta, XScale=XScale)

                def launch_cached():
                    default_launch(Input, Gamma, Beta, XScale, Output, YScale, m_in, stream)

                def launch_mid():
                    mid_launch(Input, Gamma, Beta, XScale, Output, YScale, m_in, stream)

                def launch_streaming():
                    nt_launch(Input, Gamma, Beta, XScale, Output, YScale, m_in, stream)

                @flyc.jit
                def dispatch_nt_mid():
                    if m_in >= fx.Int32(2048):
                        launch_streaming()
                    elif m_in >= fx.Int32(512):
                        launch_mid()
                    else:
                        launch_cached()

                dispatch_nt_mid()

            return launch_layernorm_smoothquant_nt_mid

        if dtype_str == "f32" and 897 <= vec_tiles <= 1792:
            nt_launch = _build_layernorm_smoothquant_vec_module(
                N,
                dtype_str,
                quant_dtype_str=quant_dtype_str,
                eps=eps,
                block_threads=256,
                input_nt=True,
            )

            @flyc.jit
            def launch_layernorm_smoothquant_nt_rows(
                Input: fx.Tensor,
                Gamma: fx.Tensor,
                Beta: fx.Tensor,
                XScale: fx.Tensor,
                Output: fx.Tensor,
                YScale: fx.Tensor,
                m_in: fx.Int32,
                stream: fx.Stream = fx.Stream(None),
            ):
                validate_norm_operand_dtypes(Input, dtype_str, Gamma=Gamma, Beta=Beta, XScale=XScale)

                def launch_cached():
                    default_launch(Input, Gamma, Beta, XScale, Output, YScale, m_in, stream)

                def launch_streaming():
                    nt_launch(Input, Gamma, Beta, XScale, Output, YScale, m_in, stream)

                @flyc.jit
                def dispatch_nt_rows():
                    if m_in >= fx.Int32(512):
                        launch_streaming()
                    else:
                        launch_cached()

                dispatch_nt_rows()

            return launch_layernorm_smoothquant_nt_rows

        if throughput_threads == default_threads:
            # No second geometry is useful for this row width.
            return default_launch
        throughput_launch = _build_layernorm_smoothquant_vec_module(
            N,
            dtype_str,
            quant_dtype_str=quant_dtype_str,
            eps=eps,
            block_threads=throughput_threads,
        )

        @flyc.jit
        def launch_layernorm_smoothquant_dispatch(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Beta: fx.Tensor,
            XScale: fx.Tensor,
            Output: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            validate_norm_operand_dtypes(Input, dtype_str, Gamma=Gamma, Beta=Beta, XScale=XScale)

            def launch_default():
                default_launch(Input, Gamma, Beta, XScale, Output, YScale, m_in, stream)

            def launch_throughput():
                throughput_launch(Input, Gamma, Beta, XScale, Output, YScale, m_in, stream)

            @flyc.jit
            def dispatch():
                # Many rows: one-wave blocks avoid the per-block reduction
                # overhead and let the scheduler keep more row work in flight.
                if m_in >= fx.Int32(512):
                    launch_throughput()
                else:
                    launch_default()

            dispatch()

        return launch_layernorm_smoothquant_dispatch
    return _build_layernorm_quant_module(
        N,
        dtype_str,
        is_smooth=True,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
    )


def build_fused_add_layernorm_dynamicquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    return _build_fused_add_layernorm_quant_module(
        N,
        dtype_str,
        is_smooth=False,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
    )


def build_fused_add_layernorm_smoothquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    return _build_fused_add_layernorm_quant_module(
        N,
        dtype_str,
        is_smooth=True,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
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
