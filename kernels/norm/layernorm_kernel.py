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

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, full
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


def _build_layernorm_bwd_row_module(N: int, dtype_str: str):
    """Fused LayerNorm backward: grid=(M,), one block per row.

    With x_hat = (x - mean)*rstd, wdy = dy*gamma:
      c1 = mean_N(wdy) ; c2 = mean_N(wdy * x_hat)
    Pass 2:
      dx = (wdy - c1 - x_hat*c2) * rstd            -> DX (elem dtype);
      dgamma_elem = dy * x_hat (fp32)              -> atomicAdd into DGamma[idx];
      dbias_elem  = dy        (fp32)               -> atomicAdd into DBias[idx].
    eps is baked into Rstd by the forward, so it is not needed here.

    Scalar and two-pass, but a single launch: ``build_layernorm_bwd_module``
    only sends it problems small enough for launch latency to dominate.
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


# The dgamma/dbias kernel gives every 32-thread row lane one row at a time: a lane
# loads one column tile of one row (a single coalesced request) and the block folds
# its row lanes in shared memory, so a column costs one atomic add per row group.
BWD_DPARAM_THREADS = 1024
# Enough (column tile, row group) blocks to keep every CU of an MI355X busy.
BWD_DPARAM_TARGET_BLOCKS = 256
# A row group is never shorter than this, so each row lane still walks a few rows.
BWD_DPARAM_MIN_ROWS = 64
# Rows each row lane has in flight per step.
BWD_DPARAM_UNROLL = 4
# Rows wider than this many tiles run half-height blocks: more, smaller programs.
BWD_DPARAM_WIDE_TILES = 512
# The widest access that still leaves a full 32-lane group of tiles on a row.
BWD_MIN_TILES = 32

# The dx kernel walks one row at a time with the whole block on the row, over a
# persistent grid of this many programs (blocks of at most 64 threads get twice as many).
BWD_DX_PROGRAMS = 1024
BWD_DX_PROGRAMS_NARROW = 2048
# A row stays in registers between the two passes while each thread holds at most
# this many tiles of it; longer rows are read twice.
BWD_DX_MAX_CACHED_ITERS = 4

# On small problems the row kernel's single launch beats the split kernels' two.
# Measured on MI355X it wins up to 64 rows at N = 4096, 64 rows more each time N
# halves, and never past 320 rows; on wider rows the split kernels always win.
BWD_ROW_KERNEL_MAX_N = 4096
BWD_ROW_KERNEL_ROWS_PER_HALVING = 64
BWD_ROW_KERNEL_MAX_ROWS = 320


def _ln_bwd_vec(N: int, dtype_str: str) -> int:
    """Widest power-of-two element count of a 128-bit access that divides N.

    Narrow rows step the width down so a full wave still covers the row: a
    latency-bound shape gains more from 32 active lanes than from 16 wide ones.
    """
    elem_bits = 32 if dtype_str == "f32" else 16
    choices = (8, 4, 2, 1) if elem_bits == 16 else (4, 2, 1)
    min_tiles = BWD_MIN_TILES
    for v in choices:
        if N % v == 0 and N // v >= min_tiles:
            return v
    return next(v for v in choices if N % v == 0)


def _ln_bwd_copy_op(vec: int, elem_bits: int):
    return {
        16: fx.rocdl.BufferCopy16b,
        32: fx.rocdl.BufferCopy32b,
        64: fx.rocdl.BufferCopy64b,
        128: fx.rocdl.BufferCopy128b,
    }[vec * elem_bits]


def _ln_bwd_dparam_storage(dp_cols: int, dp_lanes: int):
    """Shared staging for the row-lane fold of the dgamma/dbias kernel."""

    @fx.struct
    class DpStorage:
        s_fold: fx.Array[fx.Float32, dp_cols * dp_lanes, 16]

    return DpStorage


def _ln_bwd_dx_config(N: int, vec: int):
    """(threads, programs) for the persistent dx row kernel."""
    ntiles = (N + vec - 1) // vec
    threads = 32
    for cand in (32, 64, 128, 256, 512):
        if ntiles >= cand:
            threads = cand
    progs = BWD_DX_PROGRAMS_NARROW if threads <= 64 else BWD_DX_PROGRAMS
    return threads, progs


def build_layernorm_bwd_module(N: int, dtype_str: str):
    """LayerNorm backward (dx, fp32 dgamma/dbias) as two vectorized kernels.

    With ``x_hat = (x - mean) * rstd`` and ``wdy = dy * gamma`` the row sums are
    ``c1 = mean_N(wdy)`` and ``c2 = mean_N(wdy * x_hat)``, giving

        dx      = (wdy - c1 - x_hat * c2) * rstd      -> DX (elem dtype)
        dgamma += dy * x_hat                          -> DGamma (fp32)
        dbias  += dy                                  -> DBias (fp32)

    ``dx`` needs a whole row, so it runs as a persistent row kernel: a block owns
    one row at a time, keeps that row's ``x``/``dy`` in registers across the two
    passes (so a row is read once) and reduces ``c1``/``c2`` over the block.

    ``dgamma``/``dbias`` are column sums over rows, so a second kernel runs the
    columns in parallel: every warp owns one column tile of one row lane and the
    block folds its row lanes in shared memory, which turns ``M * N`` atomic adds
    into ``row_groups * N``.  The row groups follow M, so both kernels stay
    saturated at any row count.  ``eps`` is baked into ``Rstd`` by the forward.

    Small problems (see ``BWD_ROW_KERNEL_*``) run the one-launch row kernel of
    ``_build_layernorm_bwd_row_module`` instead.
    """
    vec = _ln_bwd_vec(N, dtype_str)
    elem_bits = 32 if dtype_str == "f32" else 16
    ntiles = (N + vec - 1) // vec
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")
    CopyOp = _ln_bwd_copy_op(vec, elem_bits)

    dx_threads, dx_progs = _ln_bwd_dx_config(N, vec)
    dx_iters = (ntiles + dx_threads - 1) // dx_threads
    # Keep the row in registers between the two passes unless that would need
    # more registers than the ISA has; very long rows re-read instead.
    dx_cache = dx_iters <= BWD_DX_MAX_CACHED_ITERS

    # dgamma/dbias kernel: a wave covers a 32-tile column group and the block
    # stacks row lanes on top of it.  Wide rows run faster with the shorter
    # block (more, smaller programs); narrow ones with the taller block, where a
    # column settles with a handful of atomic adds.
    if ntiles > BWD_DPARAM_WIDE_TILES:
        dp_threads, dp_target_blocks = BWD_DPARAM_THREADS // 2, 3 * BWD_DPARAM_TARGET_BLOCKS
    else:
        dp_threads, dp_target_blocks = BWD_DPARAM_THREADS, BWD_DPARAM_TARGET_BLOCKS
    dp_lanes = dp_threads // 32
    dp_groups = max(1, min(32, ntiles))
    dp_cols = dp_groups * vec
    dp_col_tiles = (N + dp_cols - 1) // dp_cols
    # Row groups only exist to keep the grid full: each one costs one atomic add
    # per column, and every row lane should still see a few rows.
    dp_row_groups = max(1, dp_target_blocks // dp_col_tiles)
    dp_unroll = BWD_DPARAM_UNROLL
    dp_stride = dp_lanes * dp_unroll

    DpStorage = _ln_bwd_dparam_storage(dp_cols, dp_lanes)
    DxBlockReduce = fx.coop.BlockReduce[fx.Float32x2, dx_threads]
    DxSharedStorage = DxBlockReduce.SharedStorage

    # ── kernel 1: dgamma / dbias (column-parallel over rows) ──────────────
    @flyc.kernel
    def layernorm_bwd_dparam_kernel(
        Input: fx.Pointer,
        DY: fx.Pointer,
        Mean: fx.Pointer,
        Rstd: fx.Pointer,
        DGamma: fx.Pointer,
        DBias: fx.Pointer,
        MIn: fx.Int32,
    ):
        cid = fx.block_idx.x
        gid = fx.block_idx.y
        tid = fx.thread_idx.x
        num_row_groups = fx.grid_dim.y

        elem_dtype = dtype_to_elem_type(dtype_str)
        c_zero_f = fx.Float32(0.0)
        copy_atom = fx.make_copy_atom(CopyOp(), elem_bits)
        lds = fx.SharedAllocator().allocate(DpStorage).peek()
        s_fold = lds.s_fold.view(fx.make_layout(dp_cols * dp_lanes, 1))

        row_layout = fx.make_layout(N, 1)
        scalar_layout = fx.make_layout(1, 1)
        dgamma_view = DGamma.view(fx.make_layout(N, 1))
        dbias_view = DBias.view(fx.make_layout(N, 1))

        lane = tid % 32
        warp = tid // 32
        lane_ok = lane < dp_groups
        col0 = cid * dp_cols + lane * vec
        col_ok = col0 < N
        col_tile = col_ok.select(cid * dp_groups + lane, 0)

        rows_per_group = (MIn + num_row_groups - 1) // num_row_groups
        row_begin = gid * rows_per_group
        row_end = row_begin + rows_per_group

        acc_g = fx.full(vec, 0.0, fx.Float32)
        acc_b = fx.full(vec, 0.0, fx.Float32)

        if row_begin < MIn:
            num_steps = (rows_per_group + dp_stride - 1) // dp_stride
            for step in range(0, num_steps):
                for u in range_constexpr(dp_unroll):
                    row = row_begin + warp + (step * dp_unroll + u) * dp_lanes
                    ok = (row < row_end) & (row < MIn) & col_ok
                    row_s = ok.select(row, 0)
                    row_off = fx.Int64(row_s) * N
                    x_div = fx.logical_divide(
                        fx.rocdl.make_buffer_tensor((Input + row_off).view(row_layout)),
                        fx.make_layout(vec, 1),
                    )
                    dy_div = fx.logical_divide(
                        fx.rocdl.make_buffer_tensor((DY + row_off).view(row_layout)),
                        fx.make_layout(vec, 1),
                    )
                    mean = fx.rocdl.make_buffer_tensor((Mean + fx.Int64(row_s)).view(scalar_layout))[0]
                    rstd = fx.rocdl.make_buffer_tensor((Rstd + fx.Int64(row_s)).view(scalar_layout))[0]
                    x_e = _load_vec(copy_atom, vec, elem_dtype, x_div, col_tile)
                    dy_e = _load_vec(copy_atom, vec, elem_dtype, dy_div, col_tile)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
                    dgv = dy * ((x - mean) * rstd)
                    gi = []
                    bi = []
                    for i in range_constexpr(vec):
                        gi.append(ok.select(dgv[i], c_zero_f))
                        bi.append(ok.select(dy[i], c_zero_f))
                    acc_g = acc_g + fx.Vector.from_elements(gi, fx.Float32)
                    acc_b = acc_b + fx.Vector.from_elements(bi, fx.Float32)

        # Stage each row lane's columns, fold the lanes in shared memory, then
        # flush one coalesced atomic add per column: uncoalesced atomics against a
        # narrow DGamma serialize, and the fold is what keeps them coalesced.
        if lane_ok:
            for i in range_constexpr(vec):
                fx.memref_store(acc_g[i], s_fold, warp * dp_cols + lane * vec + i)
        gpu.barrier()
        if tid < dp_cols:
            col = cid * dp_cols + tid
            if col < N:
                total = c_zero_f
                for lane_row in range_constexpr(dp_lanes):
                    total = total + fx.memref_load(s_fold, lane_row * dp_cols + tid)
                atomic_add(dgamma_view, col, total, dtype_bytes=4)
        gpu.barrier()
        if lane_ok:
            for i in range_constexpr(vec):
                fx.memref_store(acc_b[i], s_fold, warp * dp_cols + lane * vec + i)
        gpu.barrier()
        if tid < dp_cols:
            col = cid * dp_cols + tid
            if col < N:
                total = c_zero_f
                for lane_row in range_constexpr(dp_lanes):
                    total = total + fx.memref_load(s_fold, lane_row * dp_cols + tid)
                atomic_add(dbias_view, col, total, dtype_bytes=4)

    # ── kernel 2: dx (persistent row kernel) ──────────────────────────────
    @flyc.kernel
    def layernorm_bwd_dx_kernel(
        Input: fx.Pointer,
        Gamma: fx.Pointer,
        DY: fx.Pointer,
        Mean: fx.Pointer,
        Rstd: fx.Pointer,
        DX: fx.Pointer,
        MIn: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)
        lds = fx.SharedAllocator().allocate(DxSharedStorage).peek()
        copy_atom = fx.make_copy_atom(CopyOp(), elem_bits)

        def block_reduce_add2(val0, val1):
            with fx.fastmath(fm_fast):
                values = fx.Vector.from_elements([val0, val1], fx.Float32)
                total = DxBlockReduce(values, fx.ReductionOp.ADD, storage=lds)
            return total[0], total[1]

        row_layout = fx.make_layout(N, 1)
        scalar_layout = fx.make_layout(1, 1)
        gamma_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(Gamma.view(row_layout)), fx.make_layout(vec, 1))
        gamma_local = []
        if const_expr(dx_cache):
            for it in range_constexpr(dx_iters):
                idx = tid + it * dx_threads
                gamma_local.append(_load_vec(copy_atom, vec, elem_dtype, gamma_div, (idx < ntiles).select(idx, 0)))

        for row in range(bid, MIn, dx_progs):
            row_off = fx.Int64(row) * N
            row_in = fx.logical_divide(
                fx.rocdl.make_buffer_tensor((Input + row_off).view(row_layout)),
                fx.make_layout(vec, 1),
            )
            row_dy = fx.logical_divide(
                fx.rocdl.make_buffer_tensor((DY + row_off).view(row_layout)),
                fx.make_layout(vec, 1),
            )
            row_dx = fx.logical_divide(
                fx.rocdl.make_buffer_tensor((DX + row_off).view(row_layout)),
                fx.make_layout(vec, 1),
            )
            mean = fx.rocdl.make_buffer_tensor((Mean + fx.Int64(row)).view(scalar_layout))[0]
            rstd = fx.rocdl.make_buffer_tensor((Rstd + fx.Int64(row)).view(scalar_layout))[0]

            x_local = []
            dy_local = []
            thread_c1 = c_zero_f
            thread_c2 = c_zero_f
            for it in range_constexpr(dx_iters):
                idx = tid + it * dx_threads
                is_valid = idx < ntiles
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_vec(copy_atom, vec, elem_dtype, row_in, idx_safe)
                dy_e = _load_vec(copy_atom, vec, elem_dtype, row_dy, idx_safe)
                if const_expr(dx_cache):
                    x_local.append(x_e)
                    dy_local.append(dy_e)
                    g_e = gamma_local[it]
                else:
                    g_e = _load_vec(copy_atom, vec, elem_dtype, gamma_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                x_hat = (x - mean) * rstd
                wdy = dy * g
                with fx.fastmath(fm_fast):
                    red1 = wdy.reduce(ReductionOp.ADD)
                    red2 = (wdy * x_hat).reduce(ReductionOp.ADD)
                thread_c1 = thread_c1 + is_valid.select(red1, c_zero_f)
                thread_c2 = thread_c2 + is_valid.select(red2, c_zero_f)

            sum_c1, sum_c2 = block_reduce_add2(thread_c1, thread_c2)
            c1 = sum_c1 / n_float
            c2 = sum_c2 / n_float

            for it in range_constexpr(dx_iters):
                idx = tid + it * dx_threads
                is_valid = idx < ntiles
                idx_safe = is_valid.select(idx, 0)
                if const_expr(dx_cache):
                    x_e = x_local[it]
                    dy_e = dy_local[it]
                    g_e = gamma_local[it]
                else:
                    x_e = _load_vec(copy_atom, vec, elem_dtype, row_in, idx_safe)
                    dy_e = _load_vec(copy_atom, vec, elem_dtype, row_dy, idx_safe)
                    g_e = _load_vec(copy_atom, vec, elem_dtype, gamma_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                x_hat = (x - mean) * rstd
                wdy = dy * g
                with fx.fastmath(fm_fast):
                    dx = (wdy - c1 - x_hat * c2) * rstd
                if is_valid:
                    dx_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, dx)
                    _store_vec(copy_atom, vec, elem_dtype, dx_e, row_dx, idx)

    @flyc.jit
    def launch_split(
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
        # Row groups scale with M: enough of them to fill the grid, no more than
        # one per `BWD_DPARAM_MIN_ROWS` rows so each row lane keeps a few rows.
        row_groups = fx.Int32(dp_row_groups)
        rows_cap = fx.Int32(m_in) // BWD_DPARAM_MIN_ROWS
        rows_cap = (rows_cap > 0).select(rows_cap, fx.Int32(1))
        row_groups = (row_groups < rows_cap).select(row_groups, rows_cap)
        dparam = layernorm_bwd_dparam_kernel(Input, DY, Mean, Rstd, DGamma, DBias, m_in)
        dparam.launch(grid=(dp_col_tiles, row_groups, 1), block=(dp_threads, 1, 1), stream=stream)
        dxk = layernorm_bwd_dx_kernel(Input, Gamma, DY, Mean, Rstd, DX, m_in)
        dxk.launch(grid=(dx_progs, 1, 1), block=(dx_threads, 1, 1), stream=stream)

    if N > BWD_ROW_KERNEL_MAX_N:
        return launch_split

    launch_rows = _build_layernorm_bwd_row_module(N, dtype_str)
    halvings = math.log2(2 * BWD_ROW_KERNEL_MAX_N / N)
    ROW_KERNEL_MAX_ROWS = min(BWD_ROW_KERNEL_MAX_ROWS, int(BWD_ROW_KERNEL_ROWS_PER_HALVING * halvings))

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
        def launch_split_kernels():
            launch_split(Input, Gamma, DY, Mean, Rstd, DX, DGamma, DBias, m_in, stream)

        def launch_row_kernel():
            launch_rows(Input, Gamma, DY, Mean, Rstd, DX, DGamma, DBias, m_in, stream)

        @flyc.jit
        def dispatch():
            if m_in > fx.Int32(ROW_KERNEL_MAX_ROWS):
                launch_split_kernels()
            else:
                launch_row_kernel()

        dispatch()

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
