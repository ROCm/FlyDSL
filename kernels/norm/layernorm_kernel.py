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


def _load_vec_pred(copy_atom, vec_width, elem_dtype, div_tensor, idx, pred):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    p = fx.make_rmem_tensor(1, fx.Boolean)
    p[0] = pred
    fx.copy(copy_atom, fx.slice(div_tensor, (None, idx)), r, pred=p)
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


# LayerNorm forward launch geometry, measured on MI355X.
# Rows of at most LN_FWD_SHORT_ROW_MAX_N elements pack several rows per block: wide
# blocks while M is small, narrower ones once M passes LN_FWD_SHORT_ROW_LARGE_M.
LN_FWD_SHORT_ROW_MAX_N = 1024
LN_FWD_SHORT_ROW_BLOCK_THREADS = 512
LN_FWD_SHORT_ROW_LARGE_M_BLOCK_THREADS = 256
LN_FWD_SHORT_ROW_LARGE_M = 2048
# Short rows up to this many give each row its own small block instead.
LN_FWD_SHORT_ROW_SMALL_M_MAX = 64
# Rows that are not 16-byte aligned take the scalar path: up to LN_FWD_SCALAR_SMALL_M
# rows get wider blocks to hide latency, the widest past LN_FWD_SCALAR_WIDE_N elements.
LN_FWD_SCALAR_SMALL_M = 2048
LN_FWD_SCALAR_WIDE_N = 8192
# Vector rows of more than 256 tiles drop to 256-thread blocks past this many rows.
LN_FWD_VEC_LARGE_M = 512
# Cache policy of the input loads once the input no longer fits the last-level cache.
LN_FWD_NT_CACHE_MODIFIER = 2


def build_layernorm_module(N: int, dtype_str: str, store_stats: bool = False, eps: float = EPS):
    """Build the LayerNorm-forward launcher for a fixed N and element dtype.

    The row is kept in registers between the two passes.  A 128-bit vector
    path is used whenever the row stride keeps every row 16-byte aligned;
    otherwise a scalar (register-cached) fallback handles the row.

    Rows wider than one wave use one CTA per row with an adaptive wave count.
    Short rows use either one small CTA per row (low M) or several rows per
    CTA, each reduced by an independent sub-wave (high M).  Selecting between
    those two geometries is a host-side branch inside the compiled launcher:
    the GPU still runs exactly one kernel per call.
    """
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    elem_dtype = dtype_to_elem_type(dtype_str)
    elem_bits = 32 if dtype_str == "f32" else 16
    vec_width = 128 // elem_bits
    VEC_TILES = N // vec_width
    VEC_N = N % vec_width == 0
    USE_VEC = VEC_N and VEC_TILES > 0

    def _choose_one_row_threads(n_units: int) -> int:
        # One 128-bit transaction per lane when the row is short.  Keeping the
        # CTA to the number of active lanes avoids issuing a full wave of
        # redundant loads for a 128-element row.
        if n_units <= 16:
            return 16
        if n_units <= 32:
            return 32
        if n_units <= 64:
            return 64
        if n_units <= 128:
            return 128
        if n_units <= 256:
            return 256
        return 512

    def _choose_scalar_threads(n: int) -> int:
        if n <= 4 * WARP_SIZE:
            return WARP_SIZE
        if n <= 2048:
            return 2 * WARP_SIZE
        return 4 * WARP_SIZE

    SCALAR_SPLIT = (not USE_VEC) and N > 256
    VEC_GEOM_SPLIT = USE_VEC and VEC_TILES > 256
    if USE_VEC:
        ONE_ROW_THREADS = _choose_one_row_threads(VEC_TILES)
        if elem_bits == 32 and 256 < VEC_TILES <= 512:
            ONE_ROW_THREADS = 448
        SCALAR_SMALL_M_THREADS = ONE_ROW_THREADS
    elif SCALAR_SPLIT:
        # Many CTAs need throughput; a handful of wide rows need latency
        # hiding, so those use more waves per row.
        ONE_ROW_THREADS = 128 if N <= 1024 else 256
        SCALAR_SMALL_M_THREADS = 256 if N <= 512 else (512 if N <= LN_FWD_SCALAR_WIDE_N else 1024)
    else:
        ONE_ROW_THREADS = _choose_scalar_threads(N)
        SCALAR_SMALL_M_THREADS = ONE_ROW_THREADS
    ONE_ROW_NT_THREADS = ONE_ROW_THREADS

    def _one_row_reduce_width(block_threads):
        if USE_VEC:
            units = VEC_TILES
        else:
            units = block_threads
        width = block_threads
        while width > 1 and width >= 2 * units:
            width //= 2
        return width

    # A short-row kernel puts THREADS_PER_ROW consecutive lanes on one row.
    SMALL_TILES = (N + vec_width - 1) // vec_width
    wanted = 8
    while wanted < SMALL_TILES and wanted < 64:
        wanted *= 2
    SMALL_TPR = min(wanted, 64)
    SMALL_ROWS = LN_FWD_SHORT_ROW_BLOCK_THREADS // SMALL_TPR
    SMALL_BLOCK = SMALL_TPR * SMALL_ROWS
    ROWS_PER_BLOCK = SMALL_ROWS
    SMALL_ROWS_LARGE = LN_FWD_SHORT_ROW_LARGE_M_BLOCK_THREADS // SMALL_TPR
    SMALL_BLOCK_LARGE = SMALL_TPR * SMALL_ROWS_LARGE
    ROWS_PER_BLOCK_LARGE = SMALL_ROWS_LARGE
    SMALL_LARGE_SPLIT = SMALL_BLOCK_LARGE != SMALL_BLOCK
    SMALL_VEC = USE_VEC and N <= LN_FWD_SHORT_ROW_MAX_N
    SMALL_EXACT = SMALL_VEC and SMALL_TILES % SMALL_TPR == 0
    _small_iters = (SMALL_TILES + SMALL_TPR - 1) // SMALL_TPR
    _small_pad = _small_iters * SMALL_TPR - SMALL_TILES
    SMALL_PRED_LOAD = SMALL_VEC and not SMALL_EXACT and _small_pad * 4 > _small_iters * SMALL_TPR
    USE_SMALL = SMALL_VEC
    SMALL_M_MAX = (
        min(LN_FWD_SHORT_ROW_SMALL_M_MAX, 64 if elem_bits <= 16 else 32)
        if (SMALL_VEC and SMALL_TILES <= SMALL_TPR)
        else 0
    )

    fm_fast = arith.FastMathFlags.fast
    eps_c = eps
    inv_n = 1.0 / float(N)

    def _compute_mean_rstd(sum_val, sumsq_val):
        mean = sum_val * inv_n
        mean_sq = sumsq_val * inv_n
        var = mean_sq - mean * mean
        var = (var < 0.0).select(0.0, var)
        return mean, fmath.rsqrt(var + eps_c, fastmath=fm_fast)

    def _one_row_reduce_factory(block_threads, reduce_width):
        if block_threads <= WARP_SIZE:

            def reduce2(v0, v1):
                with fx.fastmath(fm_fast):
                    return (
                        fx.coop.warp_reduce(v0, ReductionOp.ADD, width=block_threads),
                        fx.coop.warp_reduce(v1, ReductionOp.ADD, width=block_threads),
                    )

            return reduce2, None

        BlockReduce = fx.coop.BlockReduce[fx.Float32x2, block_threads]
        SharedStorage = BlockReduce.SharedStorage
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        def reduce2(v0, v1):
            with fx.fastmath(fm_fast):
                values = fx.Vector.from_elements([v0, v1], fx.Float32)
                total = BlockReduce(values, fx.ReductionOp.ADD, storage=lds)
            return total[0], total[1]

        return reduce2, lds

    NT_MIN_ROWS = (256 << 20) // (N * 4) if dtype_str == "bf16" and N > 1024 else 0
    use_nt_dispatch = USE_VEC and NT_MIN_ROWS > 0

    # ── One row per CTA: the wide-row and latency-friendly path ───────────
    def _make_one_row_kernel(input_cache_modifier, block_threads):
        vec_exact = USE_VEC and VEC_TILES % block_threads == 0
        vec_iters = (VEC_TILES + block_threads - 1) // block_threads
        vec_pad = vec_iters * block_threads - VEC_TILES
        # Predicated loads avoid a mostly empty final tile; for a small pad the
        # predicate setup costs more than the redundant safe load.
        vec_pred_load = USE_VEC and not vec_exact and vec_pad * 4 > vec_iters * block_threads
        preload_weights = (
            USE_VEC and input_cache_modifier == 0 and block_threads <= 512 and VEC_TILES <= 2 * block_threads
        )
        scalar_cache = (not USE_VEC) and (N + block_threads - 1) // block_threads <= 16
        _one_row_kwargs = {} if block_threads <= 256 else {"known_block_size": [block_threads, 1, 1]}

        @flyc.kernel(**_one_row_kwargs)
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

            block_reduce_add2, _ = _one_row_reduce_factory(block_threads, _one_row_reduce_width(block_threads))

            c_zero_f = fx.Float32(0.0)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            in_local = []

            if const_expr(USE_VEC):
                in_div = fx.logical_divide(Input_buf, fx.make_layout(vec_width, 1))
                out_div = fx.logical_divide(Output_buf, fx.make_layout(vec_width, 1))
                gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(vec_width, 1))
                beta_div = fx.logical_divide(Beta_buf, fx.make_layout(vec_width, 1))
                copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(input_cache_modifier), elem_bits)
                weight_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
                store_modifier = 2 if block_threads <= 32 else 0
                store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(store_modifier), elem_bits)

                for tile_i in range_constexpr((VEC_TILES + block_threads - 1) // block_threads):
                    idx = tid + tile_i * block_threads
                    if const_expr(vec_exact):
                        vec = _load_vec(copy_atom, vec_width, elem_dtype, in_div, idx)
                    elif const_expr(vec_pred_load):
                        is_valid = idx < VEC_TILES
                        vec = _load_vec_pred(copy_atom, vec_width, elem_dtype, in_div, idx, is_valid)
                    else:
                        is_valid = idx < VEC_TILES
                        idx_safe = is_valid.select(idx, 0)
                        vec = _load_vec(copy_atom, vec_width, elem_dtype, in_div, idx_safe)
                    in_local.append(vec)
                    x = vec.to(fx.Float32)
                    x2 = x * x
                    red = x.reduce(ReductionOp.ADD, fastmath=fm_fast)
                    red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                    if const_expr(vec_exact):
                        thread_sum = thread_sum + red
                        thread_sumsq = thread_sumsq + red2
                    else:
                        thread_sum = thread_sum + is_valid.select(red, c_zero_f)
                        thread_sumsq = thread_sumsq + is_valid.select(red2, c_zero_f)

                gamma_pre = []
                beta_pre = []
                if const_expr(preload_weights):
                    for tile_i in range_constexpr((VEC_TILES + block_threads - 1) // block_threads):
                        idx = tid + tile_i * block_threads
                        is_valid = idx < VEC_TILES
                        idx_safe = is_valid.select(idx, 0)
                        gamma_pre.append(_load_vec(weight_atom, vec_width, elem_dtype, gamma_div, idx_safe))
                        beta_pre.append(_load_vec(weight_atom, vec_width, elem_dtype, beta_div, idx_safe))

                sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
                mean, rstd = _compute_mean_rstd(sum_val, sumsq_val)

                if const_expr(store_stats):
                    if tid == 0:
                        mean_row[0] = mean
                        rstd_row[0] = rstd

                for tile_i in range_constexpr((VEC_TILES + block_threads - 1) // block_threads):
                    idx = tid + tile_i * block_threads
                    if idx < VEC_TILES:
                        if const_expr(preload_weights):
                            g = gamma_pre[tile_i].to(fx.Float32)
                            b = beta_pre[tile_i].to(fx.Float32)
                        else:
                            g = _load_vec(weight_atom, vec_width, elem_dtype, gamma_div, idx).to(fx.Float32)
                            b = _load_vec(weight_atom, vec_width, elem_dtype, beta_div, idx).to(fx.Float32)
                        x = in_local[tile_i].to(fx.Float32)
                        y = (x - mean) * rstd
                        y = y * g + b
                        out_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)
                        _store_vec(store_atom, vec_width, elem_dtype, out_e, out_div, idx)
            else:
                copy_atom = fx.make_copy_atom(
                    fx.rocdl.BufferCopy32b() if elem_bits == 32 else fx.rocdl.BufferCopy16b(), elem_bits
                )
                for base_idx_int in range_constexpr(0, N, block_threads):
                    idx = tid + base_idx_int
                    is_valid = idx < N
                    idx_safe = is_valid.select(idx, 0)
                    x_e = Input_buf[idx_safe]
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    if const_expr(scalar_cache):
                        in_local.append(x)
                    thread_sum = thread_sum + is_valid.select(x, c_zero_f)
                    thread_sumsq = thread_sumsq + is_valid.select(x * x, c_zero_f)

                sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
                mean, rstd = _compute_mean_rstd(sum_val, sumsq_val)

                if const_expr(store_stats):
                    if tid == 0:
                        mean_row[0] = mean
                        rstd_row[0] = rstd

                for base_idx_int in range_constexpr(0, N, block_threads):
                    idx = tid + base_idx_int
                    if idx < N:
                        g_e = Gamma_buf[idx]
                        b_e = Beta_buf[idx]
                        g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                        b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                        if const_expr(scalar_cache):
                            x = in_local[(base_idx_int // block_threads)]
                        else:
                            x_e2 = Input_buf[idx]
                            x = x_e2 if dtype_str == "f32" else x_e2.to(fx.Float32)
                        y = (x - mean) * rstd
                        y = y * g + b
                        y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                        Output_buf[idx] = y_e

        return layernorm_kernel

    layernorm_kernel = _make_one_row_kernel(0, ONE_ROW_THREADS)
    layernorm_kernel_nt = (
        _make_one_row_kernel(LN_FWD_NT_CACHE_MODIFIER, ONE_ROW_NT_THREADS) if use_nt_dispatch else None
    )
    layernorm_kernel_scalar_small_m = _make_one_row_kernel(0, SCALAR_SMALL_M_THREADS) if SCALAR_SPLIT else None
    # A large grid of medium-width rows already has enough work in flight;
    # four waves per row is then cheaper than eight (or sixteen).
    layernorm_kernel_vec_large_m = _make_one_row_kernel(0, 256) if VEC_GEOM_SPLIT else None

    def _make_scalar_nocache_kernel(block_threads):
        kwargs = {} if block_threads <= 256 else {"known_block_size": [block_threads, 1, 1]}

        @flyc.kernel(**kwargs)
        def layernorm_scalar_nocache_kernel(
            Input: fx.Pointer,
            Gamma: fx.Pointer,
            Beta: fx.Pointer,
            Output: fx.Pointer,
            Mean: fx.Pointer,
            Rstd: fx.Pointer,
        ):
            bid = fx.block_idx.x
            tid = fx.thread_idx.x
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

            block_reduce_add2, _ = _one_row_reduce_factory(block_threads, _one_row_reduce_width(block_threads))
            c_zero_f = fx.Float32(0.0)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            for base_idx_int in range_constexpr(0, N, block_threads):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = Input_buf[idx_safe]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                thread_sum = thread_sum + is_valid.select(x, c_zero_f)
                thread_sumsq = thread_sumsq + is_valid.select(x * x, c_zero_f)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = _compute_mean_rstd(sum_val, sumsq_val)
            if const_expr(store_stats):
                if tid == 0:
                    mean_row[0] = mean
                    rstd_row[0] = rstd

            for base_idx_int in range_constexpr(0, N, block_threads):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = Input_buf[idx]
                    g_e = Gamma_buf[idx]
                    b_e = Beta_buf[idx]
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    y = (x - mean) * rstd
                    y = y * g + b
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    Output_buf[idx] = y_e

        return layernorm_scalar_nocache_kernel

    MAIN_SCALAR_NOCACHE = (not USE_VEC) and (N + ONE_ROW_THREADS - 1) // ONE_ROW_THREADS > 16
    SMALL_SCALAR_NOCACHE = SCALAR_SPLIT and ((N + SCALAR_SMALL_M_THREADS - 1) // SCALAR_SMALL_M_THREADS > 16)
    if MAIN_SCALAR_NOCACHE:
        layernorm_main_launch_kernel = _make_scalar_nocache_kernel(ONE_ROW_THREADS)
    else:
        layernorm_main_launch_kernel = layernorm_kernel
    if SCALAR_SPLIT:
        if SMALL_SCALAR_NOCACHE:
            layernorm_scalar_small_m_kernel = _make_scalar_nocache_kernel(SCALAR_SMALL_M_THREADS)
        else:
            layernorm_scalar_small_m_kernel = layernorm_kernel_scalar_small_m
    else:
        layernorm_scalar_small_m_kernel = None

    # ── Several short rows per CTA, one independent sub-wave per row ──────
    if USE_SMALL:

        def _make_small_kernel(block_threads):
            SMALL_ROWS = block_threads // SMALL_TPR
            SMALL_BLOCK = SMALL_TPR * SMALL_ROWS
            SMALL_KWARGS = {} if SMALL_BLOCK <= 256 else {"known_block_size": [SMALL_BLOCK, 1, 1]}

            @flyc.kernel(**SMALL_KWARGS)
            def layernorm_small_kernel(
                Input: fx.Pointer,
                Gamma: fx.Pointer,
                Beta: fx.Pointer,
                Output: fx.Pointer,
                Mean: fx.Pointer,
                Rstd: fx.Pointer,
                MIn: fx.Int32,
            ):
                bid = fx.block_idx.x
                tid = fx.thread_idx.x
                row_layout = fx.make_layout(N, 1)
                scalar_layout = fx.make_layout(1, 1)
                lane = tid % SMALL_TPR
                row_local = tid // SMALL_TPR

                c_zero_f = fx.Float32(0.0)

                # Gamma and Beta are shared by every row in the CTA.  Their loads
                # are hoisted out of the per-row guard and reused across rows.
                Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma.view(row_layout))
                Beta_buf = fx.rocdl.make_buffer_tensor(Beta.view(row_layout))

                if const_expr(SMALL_VEC):
                    # One descriptor for the whole tensor: per-lane source offsets
                    # then stay in one wave instruction instead of serialising four
                    # distinct row descriptors inside the wave.
                    total_elems = fx.Int64(MIn) * N
                    flat_layout = fx.make_layout(total_elems, 1)
                    Input_flat = fx.rocdl.make_buffer_tensor(Input.view(flat_layout))
                    Output_flat = fx.rocdl.make_buffer_tensor(Output.view(flat_layout))
                    in_flat_div = fx.logical_divide(Input_flat, fx.make_layout(vec_width, 1))
                    out_flat_div = fx.logical_divide(Output_flat, fx.make_layout(vec_width, 1))
                    gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(vec_width, 1))
                    beta_div = fx.logical_divide(Beta_buf, fx.make_layout(vec_width, 1))
                    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(0), elem_bits)
                    store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(2), elem_bits)
                    gamma_local = []
                    beta_local = []
                    for tile_i in range_constexpr((SMALL_TILES + SMALL_TPR - 1) // SMALL_TPR):
                        idx = lane + tile_i * SMALL_TPR
                        is_valid = idx < VEC_TILES
                        idx_safe = is_valid.select(idx, 0)
                        gamma_local.append(_load_vec(copy_atom, vec_width, elem_dtype, gamma_div, idx_safe))
                        beta_local.append(_load_vec(copy_atom, vec_width, elem_dtype, beta_div, idx_safe))

                row = bid * SMALL_ROWS + row_local
                if row < MIn:
                    row_i64 = fx.Int64(row)
                    if const_expr(not SMALL_VEC):
                        row_offset = row_i64 * N
                        Input_buf = fx.rocdl.make_buffer_tensor((Input + row_offset).view(row_layout))
                        Output_buf = fx.rocdl.make_buffer_tensor((Output + row_offset).view(row_layout))
                    if const_expr(store_stats):
                        mean_row = fx.rocdl.make_buffer_tensor((Mean + row_i64).view(scalar_layout))
                        rstd_row = fx.rocdl.make_buffer_tensor((Rstd + row_i64).view(scalar_layout))

                    thread_sum = c_zero_f
                    thread_sumsq = c_zero_f
                    in_local = []

                    if const_expr(SMALL_VEC):
                        row_vec_base = fx.Int32(row) * VEC_TILES
                        for tile_i in range_constexpr((SMALL_TILES + SMALL_TPR - 1) // SMALL_TPR):
                            idx = row_vec_base + lane + tile_i * SMALL_TPR
                            if const_expr(SMALL_EXACT):
                                vec = _load_vec(copy_atom, vec_width, elem_dtype, in_flat_div, idx)
                            elif const_expr(SMALL_PRED_LOAD):
                                is_valid = idx < (fx.Int32(row) + 1) * VEC_TILES
                                vec = _load_vec_pred(copy_atom, vec_width, elem_dtype, in_flat_div, idx, is_valid)
                            else:
                                is_valid = idx < (fx.Int32(row) + 1) * VEC_TILES
                                idx_safe = is_valid.select(idx, row_vec_base)
                                vec = _load_vec(copy_atom, vec_width, elem_dtype, in_flat_div, idx_safe)
                            in_local.append(vec)
                            x = vec.to(fx.Float32)
                            x2 = x * x
                            red = x.reduce(ReductionOp.ADD, fastmath=fm_fast)
                            red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                            if const_expr(SMALL_EXACT):
                                thread_sum = thread_sum + red
                                thread_sumsq = thread_sumsq + red2
                            else:
                                thread_sum = thread_sum + is_valid.select(red, c_zero_f)
                                thread_sumsq = thread_sumsq + is_valid.select(red2, c_zero_f)
                    else:
                        for base_idx_int in range_constexpr(0, N, SMALL_TPR):
                            idx = lane + base_idx_int
                            is_valid = idx < N
                            idx_safe = is_valid.select(idx, 0)
                            x_e = Input_buf[idx_safe]
                            x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                            in_local.append(x)
                            thread_sum = thread_sum + is_valid.select(x, c_zero_f)
                            thread_sumsq = thread_sumsq + is_valid.select(x * x, c_zero_f)

                    with fx.fastmath(fm_fast):
                        sum_val = fx.coop.warp_reduce(thread_sum, ReductionOp.ADD, width=SMALL_TPR)
                        sumsq_val = fx.coop.warp_reduce(thread_sumsq, ReductionOp.ADD, width=SMALL_TPR)
                    mean, rstd = _compute_mean_rstd(sum_val, sumsq_val)

                    if const_expr(store_stats):
                        if lane == 0:
                            mean_row[0] = mean
                            rstd_row[0] = rstd

                    if const_expr(SMALL_VEC):
                        row_vec_base = fx.Int32(row) * VEC_TILES
                        for tile_i in range_constexpr((SMALL_TILES + SMALL_TPR - 1) // SMALL_TPR):
                            idx = row_vec_base + lane + tile_i * SMALL_TPR
                            if idx < (fx.Int32(row) + 1) * VEC_TILES:
                                g = gamma_local[tile_i].to(fx.Float32)
                                b = beta_local[tile_i].to(fx.Float32)
                                x = in_local[tile_i].to(fx.Float32)
                                y = (x - mean) * rstd
                                y = y * g + b
                                out_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)
                                _store_vec(store_atom, vec_width, elem_dtype, out_e, out_flat_div, idx)
                    else:
                        for base_idx_int in range_constexpr(0, N, SMALL_TPR):
                            idx = lane + base_idx_int
                            if idx < N:
                                g_e = Gamma_buf[idx]
                                b_e = Beta_buf[idx]
                                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                                b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                                x = in_local[(base_idx_int // SMALL_TPR)]
                                y = (x - mean) * rstd
                                y = y * g + b
                                y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                                Output_buf[idx] = y_e

            return layernorm_small_kernel

        layernorm_small_kernel = _make_small_kernel(LN_FWD_SHORT_ROW_BLOCK_THREADS)
        layernorm_small_large_kernel = (
            _make_small_kernel(LN_FWD_SHORT_ROW_LARGE_M_BLOCK_THREADS) if SMALL_LARGE_SPLIT else None
        )

    # ── Host launchers ────────────────────────────────────────────────────
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
            def launch_one():
                launcher = layernorm_main_launch_kernel(Input, Gamma, Beta, Output, Mean, Rstd)
                launcher.launch(grid=(m_in, 1, 1), block=(ONE_ROW_THREADS, 1, 1), stream=stream)

            if const_expr(SCALAR_SPLIT):

                def launch_scalar_small_m():
                    launcher = layernorm_scalar_small_m_kernel(Input, Gamma, Beta, Output, Mean, Rstd)
                    launcher.launch(
                        grid=(m_in, 1, 1),
                        block=(SCALAR_SMALL_M_THREADS, 1, 1),
                        stream=stream,
                    )

                @flyc.jit
                def dispatch():
                    if m_in <= fx.Int32(LN_FWD_SCALAR_SMALL_M):
                        launch_scalar_small_m()
                    else:
                        launch_one()

                dispatch()
            elif const_expr(USE_SMALL):

                def launch_small():
                    launcher = layernorm_small_kernel(Input, Gamma, Beta, Output, Mean, Rstd, m_in)
                    launcher.launch(
                        grid=((m_in + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK, 1, 1),
                        block=(SMALL_BLOCK, 1, 1),
                        stream=stream,
                    )

                def launch_small_large():
                    launcher = layernorm_small_large_kernel(Input, Gamma, Beta, Output, Mean, Rstd, m_in)
                    launcher.launch(
                        grid=((m_in + ROWS_PER_BLOCK_LARGE - 1) // ROWS_PER_BLOCK_LARGE, 1, 1),
                        block=(SMALL_BLOCK_LARGE, 1, 1),
                        stream=stream,
                    )

                @flyc.jit
                def dispatch():
                    if m_in <= fx.Int32(SMALL_M_MAX):
                        launch_small()
                    elif m_in > fx.Int32(LN_FWD_SHORT_ROW_LARGE_M):
                        if const_expr(SMALL_LARGE_SPLIT):
                            launch_small_large()
                        else:
                            launch_small()
                    else:
                        launch_one()

                dispatch()
            elif const_expr(VEC_GEOM_SPLIT):

                def launch_vec_large_m():
                    launcher = layernorm_kernel_vec_large_m(Input, Gamma, Beta, Output, Mean, Rstd)
                    launcher.launch(grid=(m_in, 1, 1), block=(256, 1, 1), stream=stream)

                if const_expr(use_nt_dispatch):

                    def launch_streaming():
                        launcher = layernorm_kernel_nt(Input, Gamma, Beta, Output, Mean, Rstd)
                        launcher.launch(grid=(m_in, 1, 1), block=(ONE_ROW_NT_THREADS, 1, 1), stream=stream)

                    @flyc.jit
                    def dispatch():
                        if m_in > fx.Int32(NT_MIN_ROWS):
                            launch_streaming()
                        elif m_in > fx.Int32(LN_FWD_VEC_LARGE_M):
                            launch_vec_large_m()
                        else:
                            launch_one()

                    dispatch()
                else:

                    @flyc.jit
                    def dispatch():
                        if m_in > fx.Int32(LN_FWD_VEC_LARGE_M):
                            launch_vec_large_m()
                        else:
                            launch_one()

                    dispatch()
            elif const_expr(use_nt_dispatch):

                def launch_streaming():
                    launcher = layernorm_kernel_nt(Input, Gamma, Beta, Output, Mean, Rstd)
                    launcher.launch(grid=(m_in, 1, 1), block=(ONE_ROW_NT_THREADS, 1, 1), stream=stream)

                @flyc.jit
                def dispatch():
                    if m_in > fx.Int32(NT_MIN_ROWS):
                        launch_streaming()
                    else:
                        launch_one()

                dispatch()
            else:
                launch_one()

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
        def launch_one():
            # Mean/Rstd are never dereferenced on this path.
            launcher = layernorm_main_launch_kernel(Input, Gamma, Beta, Output, Gamma, Gamma)
            launcher.launch(grid=(m_in, 1, 1), block=(ONE_ROW_THREADS, 1, 1), stream=stream)

        if const_expr(SCALAR_SPLIT):

            def launch_scalar_small_m():
                launcher = layernorm_scalar_small_m_kernel(Input, Gamma, Beta, Output, Gamma, Gamma)
                launcher.launch(
                    grid=(m_in, 1, 1),
                    block=(SCALAR_SMALL_M_THREADS, 1, 1),
                    stream=stream,
                )

            @flyc.jit
            def dispatch():
                if m_in <= fx.Int32(LN_FWD_SCALAR_SMALL_M):
                    launch_scalar_small_m()
                else:
                    launch_one()

            dispatch()
        elif const_expr(USE_SMALL):

            def launch_small():
                launcher = layernorm_small_kernel(Input, Gamma, Beta, Output, Gamma, Gamma, m_in)
                launcher.launch(
                    grid=((m_in + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK, 1, 1),
                    block=(SMALL_BLOCK, 1, 1),
                    stream=stream,
                )

            def launch_small_large():
                launcher = layernorm_small_large_kernel(Input, Gamma, Beta, Output, Gamma, Gamma, m_in)
                launcher.launch(
                    grid=((m_in + ROWS_PER_BLOCK_LARGE - 1) // ROWS_PER_BLOCK_LARGE, 1, 1),
                    block=(SMALL_BLOCK_LARGE, 1, 1),
                    stream=stream,
                )

            @flyc.jit
            def dispatch():
                if m_in <= fx.Int32(SMALL_M_MAX):
                    launch_small()
                elif m_in > fx.Int32(LN_FWD_SHORT_ROW_LARGE_M):
                    if const_expr(SMALL_LARGE_SPLIT):
                        launch_small_large()
                    else:
                        launch_small()
                else:
                    launch_one()

            dispatch()
        elif const_expr(VEC_GEOM_SPLIT):

            def launch_vec_large_m():
                launcher = layernorm_kernel_vec_large_m(Input, Gamma, Beta, Output, Gamma, Gamma)
                launcher.launch(grid=(m_in, 1, 1), block=(256, 1, 1), stream=stream)

            if const_expr(use_nt_dispatch):

                def launch_streaming():
                    launcher = layernorm_kernel_nt(Input, Gamma, Beta, Output, Gamma, Gamma)
                    launcher.launch(grid=(m_in, 1, 1), block=(ONE_ROW_NT_THREADS, 1, 1), stream=stream)

                @flyc.jit
                def dispatch():
                    if m_in > fx.Int32(NT_MIN_ROWS):
                        launch_streaming()
                    elif m_in > fx.Int32(LN_FWD_VEC_LARGE_M):
                        launch_vec_large_m()
                    else:
                        launch_one()

                dispatch()
            else:

                @flyc.jit
                def dispatch():
                    if m_in > fx.Int32(LN_FWD_VEC_LARGE_M):
                        launch_vec_large_m()
                    else:
                        launch_one()

                dispatch()
        elif const_expr(use_nt_dispatch):

            def launch_streaming():
                launcher = layernorm_kernel_nt(Input, Gamma, Beta, Output, Gamma, Gamma)
                launcher.launch(grid=(m_in, 1, 1), block=(ONE_ROW_NT_THREADS, 1, 1), stream=stream)

            @flyc.jit
            def dispatch():
                if m_in > fx.Int32(NT_MIN_ROWS):
                    launch_streaming()
                else:
                    launch_one()

            dispatch()
        else:
            launch_one()

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
