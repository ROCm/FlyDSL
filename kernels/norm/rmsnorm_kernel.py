# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""RMSNorm kernel builder using the @flyc.kernel API.

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma

Two paths:
  - Vector path (f16/bf16, N >= 8): 128-bit vec8 copies plus a scalar tail.
  - Scalar path (f32 or N < 8): guarded elementwise copies.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp
from flydsl.runtime.device import get_rocm_arch
from kernels.common.kernels_common import dtype_to_elem_type

# Backward builders live in their own module (see review on #800); re-exported
# here so existing importers (tests, callers) keep working unchanged.
from kernels.norm.rmsnorm_bwd_kernel import (  # noqa: E402,F401
    build_fused_add_rmsnorm_bwd_module,
    build_fused_add_rmsnorm_bwd_two_stage_module,
    build_rmsnorm_bwd_module,
    build_rmsnorm_bwd_two_stage_module,
    is_rmsnorm_bwd_two_stage_vec_config,
)
from kernels.norm.rmsnorm_common import (
    BLOCK_THREADS,
    EPS,
    VEC_WIDTH,
    WARP_SIZE,
)
from kernels.norm.rmsnorm_common import load_vec as _load_vec
from kernels.norm.rmsnorm_common import load_weight_vec as _load_weight_vec
from kernels.norm.rmsnorm_common import resolve_rmsnorm_weight_dtype as _resolve_rmsnorm_weight_dtype
from kernels.norm.rmsnorm_common import store_vec as _store_vec
from kernels.norm.rmsnorm_common import to_elem_scalar as _to_elem_scalar
from kernels.norm.rmsnorm_common import to_elem_vec as _to_elem_vec
from kernels.norm.rmsnorm_common import validate_norm_operand_dtypes as _validate_norm_operand_dtypes
from kernels.norm.rmsnorm_common import weight_vec_width as _weight_vec_width

try:
    import torch
except ImportError:
    torch = None

KERNEL_NAME = "rmsnorm"

SMALL_N_THRESHOLD = 2048
FWD_MULTI_ROW_THRESHOLD = 8192
FWD_THROUGHPUT_MIN_ROWS = 8 * WARP_SIZE
FWD_THROUGHPUT_MAX_VEC_ITERS = 8
DEFAULT_BLOCK_THREADS = BLOCK_THREADS
WIDE_ROW_MIN_N = 8192
WIDE_ROW_BLOCK_THREADS = 512
# A quantized forward row lives in one block, so the block width is that row's
# only parallelism; the hardware limit is the ceiling.
MAX_QUANT_BLOCK_THREADS = 1024
# One block per row: from this many rows on, the launch already fills the device
# and a narrower block wins on occupancy instead of width.
QUANT_THROUGHPUT_MIN_ROWS = 256
# Packed narrow-row kernel: several rows share a block, so this is the only
# choice of block width left.  Two waves measured best across the small-M shapes.
NARROW_ROW_BLOCK_THREADS = 512
# Warp count at which the block reduction switches from "every thread folds the
# per-warp slots" to "warp 0 folds them across its lanes" (see WARP_FOLD).
WARP_FOLD_MIN_WARPS = 12

# Both CDNA3 and CDNA4 share a 256MB last-level cache. Non-temporal streaming
# only pays off once input plus output stop fitting in it; below that it throws
# away reuse the cache would have served. The two policies measure as a tie at
# 1.125x LLC and streaming wins outright past it, uniformly across row widths.
LLC_BYTES = 256 * 1024 * 1024
NT_MIN_WORKING_SET = LLC_BYTES * 9 // 8


def default_block_threads(N: int, arch=None) -> int:
    """Wide rows hold fewer resident vectors per thread at 512 than at 256."""
    if arch is None:
        arch = get_rocm_arch()
    if str(arch).startswith("gfx95") and N >= WIDE_ROW_MIN_N:
        return WIDE_ROW_BLOCK_THREADS
    return DEFAULT_BLOCK_THREADS


def _quant_dtype_to_elem_type(dtype_str: str):
    if dtype_str in ("i8", "int8"):
        return fx.Int8
    raise ValueError(f"unsupported quant dtype: {dtype_str!r} (expected 'i8' or 'int8')")


def _quant_dtype_max(dtype_str: str) -> float:
    if dtype_str in ("i8", "int8"):
        return 127.0
    raise ValueError(f"unsupported quant dtype: {dtype_str!r} (expected 'i8' or 'int8')")


def build_rmsnorm_module(
    N: int,
    dtype_str: str,
    store_rstd: bool = False,
    eps: float = EPS,
    BLOCK_THREADS: int | None = None,
    weight_dtype_str: str | None = None,
    _enable_throughput_dispatch: bool = True,
    _force_nt: bool | None = None,
):
    weight_dtype_str = _resolve_rmsnorm_weight_dtype(dtype_str, weight_dtype_str)
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")
    if BLOCK_THREADS is None:
        BLOCK_THREADS = default_block_threads(N, arch)

    # BLOCK_THREADS controls storage, tiling, and launch geometry.
    elem_bits = 32 if dtype_str == "f32" else 16
    weight_elem_bits = 32 if weight_dtype_str == "f32" else 16
    USE_VEC_N = elem_bits <= 16 and N >= VEC_WIDTH
    VEC_TILES = N // VEC_WIDTH if USE_VEC_N else 0
    VEC_ELEMS = VEC_TILES * VEC_WIDTH
    TAIL_ELEMS = N - VEC_ELEMS
    NUM_VEC_ITERS = (VEC_TILES + BLOCK_THREADS - 1) // BLOCK_THREADS if USE_VEC_N else 0
    _kernel_kwargs = {} if BLOCK_THREADS <= 256 else {"known_block_size": [BLOCK_THREADS, 1, 1]}

    # Stream large BF16 rows non-temporally. x stays in registers across both
    # passes, so x and y are each touched once and neither gains from being
    # cached; letting x fills allocate in the memory-side cache only evicts y's
    # pending stores. gfx942 streams unconditionally; gfx95x defers the choice
    # to a runtime row-count dispatch below.
    BF16_ROWS = dtype_str == "bf16" and weight_dtype_str == "bf16"
    USE_GFX942_BF16_FAST_PATH = str(arch).startswith("gfx942") and BF16_ROWS and N >= 4096
    # Whether streaming wins is set by the working set, not the row width, so on
    # gfx95x the runtime dispatch below covers every row the plain kernel handles.
    use_nt_dispatch = str(arch).startswith("gfx95") and BF16_ROWS and N > SMALL_N_THRESHOLD and _force_nt is None
    USE_NT = USE_GFX942_BF16_FAST_PATH or bool(_force_nt)
    PRELOAD_GAMMA = USE_GFX942_BF16_FAST_PATH and VEC_TILES >= BLOCK_THREADS and VEC_TILES % BLOCK_THREADS == 0
    INPUT_CACHE_MODIFIER = 2 if USE_NT else 0
    # Streaming the stores too costs 6-8% on gfx95x; letting them settle in LLC
    # drains them to HBM in better-scheduled bursts. gfx942 keeps its own tuning.
    OUTPUT_CACHE_MODIFIER = (
        2 if USE_GFX942_BF16_FAST_PATH and VEC_TILES > WARP_SIZE * FWD_THROUGHPUT_MAX_VEC_ITERS else 0
    )

    one_wave_vec_iters = (VEC_TILES + WARP_SIZE - 1) // WARP_SIZE if USE_VEC_N else 0
    THROUGHPUT_BLOCK_THREADS = WARP_SIZE if 0 < one_wave_vec_iters <= FWD_THROUGHPUT_MAX_VEC_ITERS else BLOCK_THREADS
    use_throughput_dispatch = (
        _enable_throughput_dispatch
        and USE_GFX942_BF16_FAST_PATH
        and N > SMALL_N_THRESHOLD
        and BLOCK_THREADS == DEFAULT_BLOCK_THREADS
        and THROUGHPUT_BLOCK_THREADS < BLOCK_THREADS
    )
    if use_throughput_dispatch:
        # Keep M runtime-only: compile both geometries into one launcher rather
        # than selecting a different cached build for each row count.
        latency_launch = build_rmsnorm_module(
            N,
            dtype_str,
            store_rstd=store_rstd,
            eps=eps,
            BLOCK_THREADS=BLOCK_THREADS,
            weight_dtype_str=weight_dtype_str,
            _enable_throughput_dispatch=False,
        )
        throughput_launch = build_rmsnorm_module(
            N,
            dtype_str,
            store_rstd=store_rstd,
            eps=eps,
            BLOCK_THREADS=THROUGHPUT_BLOCK_THREADS,
            weight_dtype_str=weight_dtype_str,
            _enable_throughput_dispatch=False,
        )

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
                def launch_throughput():
                    throughput_launch(Input, Gamma, Output, Rstd, m_in, stream)

                def launch_default():
                    latency_launch(Input, Gamma, Output, Rstd, m_in, stream)

                @flyc.jit
                def dispatch():
                    if m_in >= fx.Int32(FWD_THROUGHPUT_MIN_ROWS):
                        launch_throughput()
                    else:
                        launch_default()

                dispatch()

            return launch_rmsnorm

        @flyc.jit
        def launch_rmsnorm(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            def launch_throughput():
                throughput_launch(Input, Gamma, Output, m_in, stream)

            def launch_default():
                latency_launch(Input, Gamma, Output, m_in, stream)

            @flyc.jit
            def dispatch():
                if m_in >= fx.Int32(FWD_THROUGHPUT_MIN_ROWS):
                    launch_throughput()
                else:
                    launch_default()

            dispatch()

        return launch_rmsnorm

    if use_nt_dispatch:
        # Keep M runtime-only: compile both cache policies into one launcher
        # rather than selecting a different cached build for each row count.
        _nt_kwargs = dict(
            store_rstd=store_rstd,
            eps=eps,
            BLOCK_THREADS=BLOCK_THREADS,
            weight_dtype_str=weight_dtype_str,
            _enable_throughput_dispatch=False,
        )
        cached_launch = build_rmsnorm_module(N, dtype_str, _force_nt=False, **_nt_kwargs)
        streaming_launch = build_rmsnorm_module(N, dtype_str, _force_nt=True, **_nt_kwargs)
        NT_MIN_ROWS = NT_MIN_WORKING_SET // (N * 4)

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
                def launch_streaming():
                    streaming_launch(Input, Gamma, Output, Rstd, m_in, stream)

                def launch_cached():
                    cached_launch(Input, Gamma, Output, Rstd, m_in, stream)

                @flyc.jit
                def dispatch():
                    if m_in > fx.Int32(NT_MIN_ROWS):
                        launch_streaming()
                    else:
                        launch_cached()

                dispatch()

            return launch_rmsnorm

        @flyc.jit
        def launch_rmsnorm(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            def launch_streaming():
                streaming_launch(Input, Gamma, Output, m_in, stream)

            def launch_cached():
                cached_launch(Input, Gamma, Output, m_in, stream)

            @flyc.jit
            def dispatch():
                if m_in > fx.Int32(NT_MIN_ROWS):
                    launch_streaming()
                else:
                    launch_cached()

            dispatch()

        return launch_rmsnorm

    BlockReduce = fx.coop.BlockReduce[fx.Float32, BLOCK_THREADS]
    SharedStorage = BlockReduce.SharedStorage

    @flyc.kernel(**_kernel_kwargs)
    def rmsnorm_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Rstd: fx.Tensor,
        Output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        weight_elem_dtype = dtype_to_elem_type(weight_dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = eps
        n_float = float(N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        if const_expr(store_rstd):
            Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)

        def block_reduce_add(val):
            with fx.fastmath(fm_fast):
                return BlockReduce(val, fx.ReductionOp.ADD, storage=lds)

        # Vector path for the f16/bf16 vec8 prefix plus an optional scalar tail.
        if const_expr(USE_VEC_N):
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(_weight_vec_width(weight_dtype_str), 1))

            copy_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(INPUT_CACHE_MODIFIER),
                elem_bits,
            )
            output_copy_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(OUTPUT_CACHE_MODIFIER),
                elem_bits,
            )
            gamma_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), weight_elem_bits)

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f
            in_local = []
            gamma_local = []

            # Pass 1: load + cache + sumsq
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                vec = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx_safe)
                in_local.append(vec)
                if const_expr(PRELOAD_GAMMA):
                    gamma_local.append(_load_vec(gamma_copy_atom, VEC_WIDTH, weight_elem_dtype, gamma_div, idx_safe))
                x = vec.to(fx.Float32)

                x2 = x * x
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + is_valid.select(red2, c_zero_f)

            if const_expr(TAIL_ELEMS > 0):
                if tid < TAIL_ELEMS:
                    tail_idx = tid + VEC_ELEMS
                    x_tail_e = row_in[tail_idx]
                    x_tail = x_tail_e.to(fx.Float32)
                    thread_sumsq = thread_sumsq + x_tail * x_tail

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            if const_expr(store_rstd):
                if tid == 0:
                    Rstd_buf[bid] = rrms

            # Pass 2: normalize + gamma + store (reuse cached input)
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                if idx < VEC_TILES:
                    if const_expr(PRELOAD_GAMMA):
                        g = gamma_local[tile_i].to(fx.Float32)
                    else:
                        g = _load_weight_vec(gamma_copy_atom, weight_dtype_str, weight_elem_dtype, gamma_div, idx)
                    x = in_local[tile_i].to(fx.Float32)

                    y = (x * rrms) * g
                    out_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)
                    _store_vec(output_copy_atom, VEC_WIDTH, elem_dtype, out_e, out_div, idx)

            if const_expr(TAIL_ELEMS > 0):
                if tid < TAIL_ELEMS:
                    tail_idx = tid + VEC_ELEMS
                    x_tail_e = row_in[tail_idx]
                    g_tail_e = Gamma_buf[tail_idx]
                    x_tail = x_tail_e.to(fx.Float32)
                    g_tail = g_tail_e if weight_dtype_str == "f32" else g_tail_e.to(fx.Float32)
                    y_tail = (x_tail * rrms) * g_tail
                    y_tail_e = _to_elem_scalar(dtype_str, elem_dtype, y_tail)
                    row_out[tail_idx] = y_tail_e

        else:
            # Scalar fallback for arbitrary N.
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = row_in[idx_safe]
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
                    Rstd_buf[bid] = rrms

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = row_in[idx]
                    g_e = Gamma_buf[idx]
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if weight_dtype_str == "f32" else g_e.to(fx.Float32)
                    norm = x * rrms
                    y = norm * g
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    row_out[idx] = y_e

    if N <= SMALL_N_THRESHOLD:
        return _build_rmsnorm_small_n_module(
            N,
            dtype_str,
            store_rstd,
            eps,
            weight_dtype_str,
            one_row_kernel=rmsnorm_kernel,
            one_row_block_threads=BLOCK_THREADS,
        )

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


@flyc.jit
def rmsnorm_direct(
    Input: fx.Tensor,
    Gamma: fx.Tensor,
    Output: fx.Tensor,
    m_in: fx.Int32,
    N: fx.Constexpr[int],
    dtype_str: fx.Constexpr[str],
    BLOCK_THREADS: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
    weight_dtype_str: fx.Constexpr[str] = "",
):
    """Specialize the existing RMSNorm factory through JIT Constexpr inputs."""
    resolved_weight_dtype_str = dtype_str if weight_dtype_str == "" else weight_dtype_str
    launch = build_rmsnorm_module(
        N,
        dtype_str,
        BLOCK_THREADS=BLOCK_THREADS,
        weight_dtype_str=resolved_weight_dtype_str,
        _enable_throughput_dispatch=False,
    )
    launch(Input, Gamma, Output, m_in, stream)


def _build_rmsnorm_small_n_module(
    N: int,
    dtype_str: str,
    store_rstd: bool = False,
    eps: float = EPS,
    weight_dtype_str: str | None = None,
    *,
    one_row_kernel,
    one_row_block_threads: int,
):
    weight_dtype_str = _resolve_rmsnorm_weight_dtype(dtype_str, weight_dtype_str)
    BLOCK_N = 1 << (N - 1).bit_length()
    BLOCK_M = max(min(16384 // BLOCK_N, 32), 8)
    THREADS_PER_ROW = min(WARP_SIZE, 1024 // BLOCK_M)
    BLOCK_THREADS_SPECIAL = BLOCK_M * THREADS_PER_ROW
    elem_bits = 32 if dtype_str == "f32" else 16
    weight_elem_bits = 32 if weight_dtype_str == "f32" else 16
    USE_VEC_SMALL_N = elem_bits <= 16 and N >= VEC_WIDTH
    VEC_TILES = N // VEC_WIDTH if USE_VEC_SMALL_N else 0
    VEC_ELEMS = VEC_TILES * VEC_WIDTH
    TAIL_ELEMS = N - VEC_ELEMS
    NUM_VEC_ROW_ITERS = (VEC_TILES + THREADS_PER_ROW - 1) // THREADS_PER_ROW if USE_VEC_SMALL_N else 0
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

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
        row = bid * BLOCK_M + row_local

        if row < MIn:
            elem_dtype = dtype_to_elem_type(dtype_str)
            weight_elem_dtype = dtype_to_elem_type(weight_dtype_str)
            fm_fast = arith.FastMathFlags.fast
            eps_c = eps
            n_float = float(N)

            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            if const_expr(store_rstd):
                Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)

            row_in = fx.slice(Input_buf, (row, None))
            row_out = fx.slice(Output_buf, (row, None))

            def group_reduce_add(x):
                with fx.fastmath(fm_fast):
                    return fx.coop.warp_reduce(x, fx.ReductionOp.ADD, width=THREADS_PER_ROW)

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f

            if const_expr(USE_VEC_SMALL_N):
                in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
                gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(_weight_vec_width(weight_dtype_str), 1))
                out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))

                copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
                gamma_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), weight_elem_bits)

                in_local = []
                for tile_i in range_constexpr(NUM_VEC_ROW_ITERS):
                    idx = lane + tile_i * THREADS_PER_ROW
                    is_valid = idx < VEC_TILES
                    idx_safe = is_valid.select(idx, 0)
                    vec = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx_safe)
                    in_local.append(vec)
                    x = vec.to(fx.Float32)
                    x2 = x * x
                    red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                    thread_sumsq = thread_sumsq + is_valid.select(red2, c_zero_f)

                if const_expr(TAIL_ELEMS > 0):
                    if lane < TAIL_ELEMS:
                        tail_idx = lane + VEC_ELEMS
                        x_tail_e = row_in[tail_idx]
                        x_tail = x_tail_e.to(fx.Float32)
                        thread_sumsq = thread_sumsq + x_tail * x_tail

                sum_sq = group_reduce_add(thread_sumsq)
                mean_sq = sum_sq / n_float
                ms_eps = mean_sq + eps_c
                rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

                if const_expr(store_rstd):
                    if lane == 0:
                        Rstd_buf[row] = rrms

                for tile_i in range_constexpr(NUM_VEC_ROW_ITERS):
                    idx = lane + tile_i * THREADS_PER_ROW
                    if idx < VEC_TILES:
                        g = _load_weight_vec(
                            gamma_copy_atom,
                            weight_dtype_str,
                            weight_elem_dtype,
                            gamma_div,
                            idx,
                        )
                        x = in_local[tile_i].to(fx.Float32)
                        y = (x * rrms) * g
                        y_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)
                        _store_vec(copy_atom, VEC_WIDTH, elem_dtype, y_e, out_div, idx)

                if const_expr(TAIL_ELEMS > 0):
                    if lane < TAIL_ELEMS:
                        tail_idx = lane + VEC_ELEMS
                        x_tail_e = row_in[tail_idx]
                        g_tail_e = Gamma_buf[tail_idx]
                        x_tail = x_tail_e.to(fx.Float32)
                        g_tail = g_tail_e if weight_dtype_str == "f32" else g_tail_e.to(fx.Float32)
                        y_tail = (x_tail * rrms) * g_tail
                        y_tail_e = _to_elem_scalar(dtype_str, elem_dtype, y_tail)
                        row_out[tail_idx] = y_tail_e
            else:

                for base_idx_int in range_constexpr(0, BLOCK_N, THREADS_PER_ROW):
                    idx = lane + base_idx_int
                    is_valid = idx < N
                    idx_safe = is_valid.select(idx, 0)
                    x_e = row_in[idx_safe]
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    x2 = x * x
                    thread_sumsq = thread_sumsq + is_valid.select(x2, c_zero_f)

                sum_sq = group_reduce_add(thread_sumsq)
                mean_sq = sum_sq / n_float
                ms_eps = mean_sq + eps_c
                rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

                if const_expr(store_rstd):
                    if lane == 0:
                        Rstd_buf[row] = rrms

                for base_idx_int in range_constexpr(0, BLOCK_N, THREADS_PER_ROW):
                    idx = lane + base_idx_int
                    if idx < N:
                        x_e = row_in[idx]
                        g_e = Gamma_buf[idx]
                        x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                        g = g_e if weight_dtype_str == "f32" else g_e.to(fx.Float32)
                        y = (x * rrms) * g
                        y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                        row_out[idx] = y_e

    if store_rstd:

        @flyc.jit
        def launch_rmsnorm_small_n(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            Rstd: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            def launch_one_row():
                launcher = one_row_kernel(Input, Gamma, Rstd, Output)
                launcher.launch(
                    grid=(m_in, 1, 1),
                    block=(one_row_block_threads, 1, 1),
                    stream=stream,
                )

            def launch_multi_row():
                launcher = rmsnorm_large_m_small_n_kernel(Input, Gamma, Rstd, Output, m_in)
                launcher.launch(
                    grid=((m_in + BLOCK_M - 1) // BLOCK_M, 1, 1),
                    block=(BLOCK_THREADS_SPECIAL, 1, 1),
                    stream=stream,
                )

            @flyc.jit
            def dispatch():
                if m_in > fx.Int32(FWD_MULTI_ROW_THRESHOLD):
                    launch_multi_row()
                else:
                    launch_one_row()

            dispatch()

        return launch_rmsnorm_small_n

    @flyc.jit
    def launch_rmsnorm_small_n(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        def launch_one_row():
            # The Rstd slot is unused when store_rstd=False.
            launcher = one_row_kernel(Input, Gamma, Gamma, Output)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(one_row_block_threads, 1, 1),
                stream=stream,
            )

        def launch_multi_row():
            launcher = rmsnorm_large_m_small_n_kernel(Input, Gamma, Gamma, Output, m_in)
            launcher.launch(
                grid=((m_in + BLOCK_M - 1) // BLOCK_M, 1, 1),
                block=(BLOCK_THREADS_SPECIAL, 1, 1),
                stream=stream,
            )

        @flyc.jit
        def dispatch():
            if m_in > fx.Int32(FWD_MULTI_ROW_THRESHOLD):
                launch_multi_row()
            else:
                launch_one_row()

        dispatch()

    return launch_rmsnorm_small_n


def build_fused_add_rmsnorm_module(
    N: int,
    dtype_str: str,
    store_rstd: bool = False,
    eps: float = EPS,
    weight_dtype_str: str | None = None,
):
    weight_dtype_str = _resolve_rmsnorm_weight_dtype(dtype_str, weight_dtype_str)
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    elem_bits = 32 if dtype_str == "f32" else 16
    weight_elem_bits = 32 if weight_dtype_str == "f32" else 16
    USE_VEC_N = elem_bits <= 16 and N % VEC_WIDTH == 0
    VEC_TILES = N // VEC_WIDTH if USE_VEC_N else 0
    NUM_VEC_ITERS = (VEC_TILES + BLOCK_THREADS - 1) // BLOCK_THREADS if USE_VEC_N else 0

    BlockReduce = fx.coop.BlockReduce[fx.Float32x2, BLOCK_THREADS]
    SharedStorage = BlockReduce.SharedStorage

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
        weight_elem_dtype = dtype_to_elem_type(weight_dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = eps
        n_float = float(N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        if const_expr(store_rstd):
            Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)

        def block_reduce_add(val):
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

        def block_reduce_add2(val0, val1):
            with fx.fastmath(fm_fast):
                values = fx.Vector.from_elements([val0, val1], fx.Float32)
                total = BlockReduce(values, fx.ReductionOp.ADD, storage=lds)
            return total[0], total[1]

        # Vector path for every complete f16/bf16 vec8 row.
        if const_expr(USE_VEC_N):
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
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(_weight_vec_width(weight_dtype_str), 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            gamma_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), weight_elem_bits)

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f
            thread_dummy = c_zero_f
            add_local = []

            # Pass 1: add + cache + sumsq (also write residual_out)
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                x = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx_safe).to(fx.Float32)
                residual = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, residual_in_div, idx_safe).to(fx.Float32)
                added_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, x + residual)
                add_local.append(added_e)
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)

                added2 = added * added
                red2 = added2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + is_valid.select(red2, c_zero_f)

                if idx < VEC_TILES:
                    _store_vec(copy_atom, VEC_WIDTH, elem_dtype, added_e, residual_out_div, idx)

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            if const_expr(store_rstd):
                if tid == 0:
                    Rstd_buf[bid] = rrms

            # Pass 2: normalize + gamma + store (reuse cached added values)
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                if idx < VEC_TILES:
                    g = _load_weight_vec(gamma_copy_atom, weight_dtype_str, weight_elem_dtype, gamma_div, idx)
                    added = add_local[tile_i] if dtype_str == "f32" else add_local[tile_i].to(fx.Float32)
                    y = (added * rrms) * g
                    y_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)
                    _store_vec(copy_atom, VEC_WIDTH, elem_dtype, y_e, out_div, idx)

        else:
            # Scalar fallback for arbitrary N.
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = row_in[idx_safe]
                residual_e = row_residual_in[idx_safe]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                residual = residual_e if dtype_str == "f32" else residual_e.to(fx.Float32)
                added_e = _to_elem_scalar(dtype_str, elem_dtype, x + residual)
                if idx < N:
                    row_residual_out[idx] = added_e
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                added2 = added * added
                thread_sumsq = thread_sumsq + is_valid.select(added2, c_zero_f)

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            if const_expr(store_rstd):
                if tid == 0:
                    Rstd_buf[bid] = rrms

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    g_e = Gamma_buf[idx]
                    added_e = row_residual_out[idx]
                    g = g_e if weight_dtype_str == "f32" else g_e.to(fx.Float32)
                    added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                    y = (added * rrms) * g
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    row_out[idx] = y_e

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
    eps: float = EPS,
):
    elem_bits = 32 if dtype_str == "f32" else 16
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)
    USE_VEC_N = elem_bits <= 16 and N % VEC_WIDTH == 0
    VEC_TILES = N // VEC_WIDTH if USE_VEC_N else 0
    NUM_VEC_ITERS = (VEC_TILES + BLOCK_THREADS - 1) // BLOCK_THREADS if USE_VEC_N else 0

    BlockReduce = fx.coop.BlockReduce[fx.Float32x2, BLOCK_THREADS]
    SharedStorage = BlockReduce.SharedStorage

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
        eps_c = eps
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_dtype_max = fx.Float32(quant_dtype_max)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        YScale_buf = fx.rocdl.make_buffer_tensor(YScale)

        def block_reduce_add(val):
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

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

        # Vector path for every complete f16/bf16 vec8 row.
        if const_expr(USE_VEC_N):
            quant_half_width = VEC_WIDTH // 2
            abs_mask = fx.full(VEC_WIDTH, 0x7FFFFFFF, fx.Uint32)
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
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                vec = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx_safe)
                in_local.append(vec)
                x = vec.to(fx.Float32)
                x2 = x * x
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + is_valid.select(red2, c_zero_f)

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            thread_row_max = c_zero_f
            y_local = []

            # Pass 2: normalize + gamma (+ optional smooth scale), cache output, and accumulate row max
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)

                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx_safe).to(fx.Float32)
                x = in_local[tile_i].to(fx.Float32)
                y = (x * rrms) * g
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

            # Pass 3: quantize + store using per-row scale
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
            # Scalar fallback for arbitrary N.
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

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
                x_e = row_in[idx_safe]
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
                x_e = row_in[idx_safe]
                g_e = Gamma_buf[idx_safe]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                y = (x * rrms) * g
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

            # Pass 3: quantize + store using per-row scale
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = row_in[idx]
                    g_e = Gamma_buf[idx]
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    y = (x * rrms) * g
                    if const_expr(is_smooth):
                        s_e = XScale_buf[idx]
                        s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                        y = y * s
                    q = y * inv_scale
                    q_i8 = q.to(quant_dtype)
                    row_out[idx] = q_i8

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
            _validate_norm_operand_dtypes(Input, dtype_str, Gamma=Gamma, XScale=XScale)
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
            _validate_norm_operand_dtypes(Input, dtype_str, Gamma=Gamma)
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
    eps: float = EPS,
):
    return _build_rmsnorm_quant_module(
        N,
        dtype_str,
        is_smooth=False,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
    )


def build_rmsnorm_smoothquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    return _build_rmsnorm_quant_module(
        N,
        dtype_str,
        is_smooth=True,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
    )


def _build_fused_add_rmsnorm_quant_module(
    N: int,
    dtype_str: str,
    *,
    is_smooth: bool,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
    block_threads: int | None = None,
    input_cache_modifier: int = 0,
    output_cache_modifier: int = 0,
):
    """Fused add + RMSNorm + per-row int8 smooth quantization, one block per row.

    Single pass over the row: the residual add, its rounding to ``dtype_str``,
    the sum of squares and the row maximum of ``A * gamma * xscale`` are all
    produced from one set of loads.  Because ``rstd > 0``,

        max|y| = rstd * max|A * gamma * xscale|,

    so the quantizer's row maximum does not wait for the mean of squares: both
    statistics leave the block in a single reduction, and the epilogue applies
    the combined constant ``rstd / yscale`` to the cached ``A * gamma * xscale``
    products.  ``gamma`` and ``xscale`` are loaded before the row, so their
    latency hides behind the activation loads.
    """
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    elem_bits = 32 if dtype_str == "f32" else 16
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)
    # Elements moved by one 128-bit access.
    VEC_N = 4 if elem_bits == 32 else VEC_WIDTH
    USE_VEC_N = N >= VEC_N
    VEC_TILES = N // VEC_N if USE_VEC_N else 0
    VEC_ELEMS = VEC_TILES * VEC_N
    TAIL_ELEMS = N - VEC_ELEMS
    THREADS = block_threads if block_threads is not None else _fused_add_quant_threads(N, dtype_str, 1)
    FULL_ITERS = VEC_TILES // THREADS if USE_VEC_N else 0
    REMAINDER_TILES = VEC_TILES - FULL_ITERS * THREADS if USE_VEC_N else 0
    NUM_WARPS = max(1, THREADS // WARP_SIZE)
    QUANT_GROUPS = VEC_N // 4
    # From this many wave64 warps on, the per-thread fold of the shared partials
    # (two slots per warp) is expected to cost more than the extra broadcast
    # barrier that lets warp 0 fold them across its lanes instead.
    WARP_FOLD = NUM_WARPS >= WARP_FOLD_MIN_WARPS

    @fx.struct
    class ReduceStorage:
        slots: fx.Array[fx.Float32, 2 * NUM_WARPS, 16]

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
        eps_c = eps
        inv_n = 1.0 / float(N)
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_inv_dtype_max = fx.Float32(1.0 / quant_dtype_max)

        slots = fx.SharedAllocator().allocate(ReduceStorage).peek().slots.view(fx.make_layout(2 * NUM_WARPS, 1))

        YScale_buf = fx.rocdl.make_buffer_tensor(YScale)

        def block_reduce_pair(val_sum, val_max):
            """Row sum and row maximum, broadcast to every thread.

            One wave-level pass per statistic, then a single exchange through
            shared memory.  For a wide block (``WARP_FOLD``) warp 0 folds the
            per-warp slots across its lanes and rebroadcasts them, which trades
            one extra barrier for ``2 * NUM_WARPS`` fewer shared loads in every
            thread.
            """
            lane = tid % WARP_SIZE
            warp = tid // WARP_SIZE
            total_sum = fx.coop.warp_reduce(val_sum, fx.ReductionOp.ADD)
            total_max = fx.coop.warp_reduce(val_max, fx.ReductionOp.MAX)
            if const_expr(NUM_WARPS > 1):
                if lane == 0:
                    fx.memref_store(total_sum, slots, warp)
                    fx.memref_store(total_max, slots, NUM_WARPS + warp)
                gpu.barrier()
                if const_expr(WARP_FOLD):
                    if warp == 0:
                        in_range = lane < NUM_WARPS
                        idx = in_range.select(lane, 0)
                        fold_sum = in_range.select(fx.memref_load(slots, idx), c_zero_f)
                        fold_max = in_range.select(fx.memref_load(slots, NUM_WARPS + idx), c_zero_f)
                        fold_sum = fx.coop.warp_reduce(fold_sum, fx.ReductionOp.ADD)
                        fold_max = fx.coop.warp_reduce(fold_max, fx.ReductionOp.MAX)
                        if lane == 0:
                            fx.memref_store(fold_sum, slots, 0)
                            fx.memref_store(fold_max, slots, 1)
                    gpu.barrier()
                    total_sum = fx.memref_load(slots, 0)
                    total_max = fx.memref_load(slots, 1)
                else:
                    total_sum = fx.memref_load(slots, 0)
                    total_max = fx.memref_load(slots, NUM_WARPS)
                    for other in range_constexpr(1, NUM_WARPS):
                        total_sum = total_sum + fx.memref_load(slots, other)
                        total_max = fx.max(total_max, fx.memref_load(slots, NUM_WARPS + other))
            return total_sum, total_max

        # Buffers and row slices are the same for the vector path and the tail.
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

        thread_sumsq = c_zero_f
        thread_max = c_zero_f
        # The scalar tail of a row that is not a whole number of 128-bit chunks
        # keeps its pass-1 product live across the barrier so the post-reduction
        # store never waits on a second (cold) load, and is issued before the
        # vector loads so its cold access hides behind them.
        scaled_tail_s = c_zero_f
        if const_expr(TAIL_ELEMS > 0):
            if tid < TAIL_ELEMS:
                tail_idx = tid + VEC_ELEMS
                x_te = row_in[tail_idx]
                r_te = row_residual_in[tail_idx]
                x_t = x_te if dtype_str == "f32" else x_te.to(fx.Float32)
                r_t = r_te if dtype_str == "f32" else r_te.to(fx.Float32)
                a_te = _to_elem_scalar(dtype_str, elem_dtype, x_t + r_t)
                row_residual_out[tail_idx] = a_te
                a_t = a_te if dtype_str == "f32" else a_te.to(fx.Float32)
                thread_sumsq = thread_sumsq + a_t * a_t
                g_te = Gamma_buf[tail_idx]
                gate_t = g_te if dtype_str == "f32" else g_te.to(fx.Float32)
                if const_expr(is_smooth):
                    s_te = XScale_buf[tail_idx]
                    gate_t = gate_t * (s_te if dtype_str == "f32" else s_te.to(fx.Float32))
                scaled_t = a_t * gate_t
                scaled_tail_s = scaled_t
                thread_max = fx.max(thread_max, fx.max(scaled_t, c_zero_f - scaled_t))

        # Vector path for every complete 128-bit row chunk.
        if const_expr(USE_VEC_N):
            abs_mask = fx.full(VEC_N, 0x7FFFFFFF, fx.Uint32)
            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_N, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(VEC_N, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(VEC_N, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_N, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(VEC_N, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            load_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(input_cache_modifier), elem_bits)
            store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(output_cache_modifier), elem_bits)
            # One 64-bit store per eight quantized values; the f32 path emits four
            # values per chunk and keeps its single 32-bit store.
            copy_atom_q = fx.make_copy_atom(
                fx.rocdl.BufferCopy64b() if QUANT_GROUPS == 2 else fx.rocdl.BufferCopy32b(), 8
            )
            out_div_q = fx.logical_divide(row_out, fx.make_layout(VEC_N, 1))

            gate_local = []
            scaled_local = []
            scaled_tail = fx.full(VEC_N, 0.0, fx.Float32)

            # Gate load first: its latency overlaps the activation loads below.
            for tile_i in range_constexpr(FULL_ITERS):
                idx = tid + tile_i * THREADS
                gate_v = _load_vec(copy_atom, VEC_N, elem_dtype, gamma_div, idx).to(fx.Float32)
                if const_expr(is_smooth):
                    xs_v = _load_vec(copy_atom, VEC_N, elem_dtype, xscale_div, idx).to(fx.Float32)
                    gate_v = gate_v * xs_v
                gate_local.append(gate_v)
            if const_expr(REMAINDER_TILES > 0):
                idx_tail = FULL_ITERS * THREADS + tid
                idx_safe = (tid < REMAINDER_TILES).select(idx_tail, 0)
                gate_vr = _load_vec(copy_atom, VEC_N, elem_dtype, gamma_div, idx_safe).to(fx.Float32)
                if const_expr(is_smooth):
                    xs_vr = _load_vec(copy_atom, VEC_N, elem_dtype, xscale_div, idx_safe).to(fx.Float32)
                    gate_vr = gate_vr * xs_vr
                gate_local.append(gate_vr)

            # Pass 1: add + round + store residual + (sum of squares, row max)
            for tile_i in range_constexpr(FULL_ITERS):
                idx = tid + tile_i * THREADS
                x = _load_vec(load_atom, VEC_N, elem_dtype, in_div, idx).to(fx.Float32)
                residual = _load_vec(load_atom, VEC_N, elem_dtype, residual_in_div, idx).to(fx.Float32)
                added_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, x + residual)
                _store_vec(store_atom, VEC_N, elem_dtype, added_e, residual_out_div, idx)
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                added2 = added * added
                thread_sumsq = thread_sumsq + added2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                scaled = added * gate_local[tile_i]
                scaled_local.append(scaled)
                scaled_abs = (scaled.bitcast(fx.Uint32) & abs_mask).bitcast(fx.Float32)
                thread_max = fx.max(thread_max, scaled_abs.reduce(ReductionOp.MAX))
            if const_expr(REMAINDER_TILES > 0):
                if tid < REMAINDER_TILES:
                    idx = FULL_ITERS * THREADS + tid
                    x = _load_vec(load_atom, VEC_N, elem_dtype, in_div, idx).to(fx.Float32)
                    residual = _load_vec(load_atom, VEC_N, elem_dtype, residual_in_div, idx).to(fx.Float32)
                    added_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, x + residual)
                    _store_vec(store_atom, VEC_N, elem_dtype, added_e, residual_out_div, idx)
                    added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                    added2 = added * added
                    thread_sumsq = thread_sumsq + added2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                    scaled_tail = added * gate_local[FULL_ITERS]
                    scaled_abs = (scaled_tail.bitcast(fx.Uint32) & abs_mask).bitcast(fx.Float32)
                    thread_max = fx.max(thread_max, scaled_abs.reduce(ReductionOp.MAX))

        sum_sq, scaled_max = block_reduce_pair(thread_sumsq, thread_max)
        with fx.fastmath(fm_fast):
            mean_sq = sum_sq * inv_n
            rrms = fmath.rsqrt(mean_sq + eps_c, fastmath=fm_fast)
            row_max = scaled_max * rrms
            scale = row_max * c_inv_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)
            # out = trunc(y / yscale) = trunc(A * gamma * xscale * (rstd / yscale))
            quant_scale = rrms / final_scale

        if tid == 0:
            YScale_buf[bid] = final_scale

        if const_expr(USE_VEC_N):

            def store_quant(scaled, idx):
                q_i8 = (scaled * quant_scale).to(quant_dtype)
                _store_vec(copy_atom_q, VEC_N, quant_dtype, q_i8, out_div_q, idx)

            # Pass 2: quantize + store, reusing the cached products.
            for tile_i in range_constexpr(FULL_ITERS):
                store_quant(scaled_local[tile_i], tid + tile_i * THREADS)
            if const_expr(REMAINDER_TILES > 0):
                if tid < REMAINDER_TILES:
                    store_quant(scaled_tail, FULL_ITERS * THREADS + tid)

        if const_expr(TAIL_ELEMS > 0):
            if tid < TAIL_ELEMS:
                tail_idx = tid + VEC_ELEMS
                row_out[tail_idx] = (scaled_tail_s * quant_scale).to(quant_dtype)

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
            _validate_norm_operand_dtypes(
                Input, dtype_str, ResidualIn=ResidualIn, Gamma=Gamma, XScale=XScale, ResidualOut=ResidualOut
            )
            launcher = fused_add_rmsnorm_quant_kernel(Input, ResidualIn, Gamma, XScale, YScale, Output, ResidualOut)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(THREADS, 1, 1),
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
            _validate_norm_operand_dtypes(Input, dtype_str, ResidualIn=ResidualIn, Gamma=Gamma, ResidualOut=ResidualOut)
            launcher = fused_add_rmsnorm_quant_kernel(Input, ResidualIn, Gamma, Gamma, YScale, Output, ResidualOut)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(THREADS, 1, 1),
                stream=stream,
            )

        return launch_fused_add_rmsnorm_dynamicquant


def _narrow_row_tiles(N: int, dtype_str: str) -> int:
    """128-bit chunks in one row, or 0 when the packed narrow-row kernel cannot take it.

    Only 16-bit activations pack: an f32 row carries twice the bytes per lane, so
    the saving in block starts no longer covers the narrower fan-out and those
    shapes measured neutral to slightly worse packed, and they keep one block per
    row.
    """
    if dtype_str == "f32":
        return 0
    if N % VEC_WIDTH:
        return 0
    tiles = N // VEC_WIDTH
    return tiles if 0 < tiles <= WARP_SIZE else 0


def _build_fused_add_rmsnorm_narrow_quant_module(
    N: int,
    dtype_str: str,
    *,
    is_smooth: bool,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
    block_threads: int | None = None,
):
    """Fused add + RMSNorm + int8 quant for rows that fit inside one wave.

    A row this narrow cannot keep a whole block busy, so the one-block-per-row
    kernel spends its time starting blocks: on this part the fixed per-block
    dispatch cost is the largest term left for those shapes.  Here a row is
    laced onto ``LANES`` lanes of a single wave and a block carries
    ``BLOCK_M = block_threads / LANES`` rows, so the grid shrinks by that
    factor and every row aggregate is a plain ``width=LANES`` warp reduction —
    no shared memory and no barrier at all.  ``M`` stays dynamic through the
    ``row < MIn`` guard, and the grid follows it.
    """
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    elem_bits = 32 if dtype_str == "f32" else 16
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)
    VEC_N = 4 if elem_bits == 32 else VEC_WIDTH
    TILES = N // VEC_N
    LANES = 1 << (TILES - 1).bit_length()
    # A row owns ``LANES`` lanes, so the block width is a multiple of that.  Two
    # waves per block measured best: wide enough to keep several rows' chunk
    # loads in flight at once, narrow enough that a small M still spreads over
    # several CUs instead of serialising on one.
    THREADS = block_threads if block_threads is not None else NARROW_ROW_BLOCK_THREADS
    THREADS = max(LANES, min(THREADS, MAX_QUANT_BLOCK_THREADS) // WARP_SIZE * WARP_SIZE)
    BLOCK_M = THREADS // LANES
    QUANT_GROUPS = VEC_N // 4
    # TILES == LANES for every power-of-two tile count; otherwise the surplus
    # lanes still take part in the reduction and are masked out of it.
    PARTIAL_ROW = TILES != LANES

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def fused_add_rmsnorm_narrow_quant_kernel(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        XScale: fx.Tensor,
        YScale: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
        MIn: fx.Int32,
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
        c_inv_dtype_max = fx.Float32(1.0 / quant_dtype_max)

        lane = tid % LANES
        row_local = tid // LANES
        row = bid * BLOCK_M + row_local
        row_valid = row < MIn
        # A block may run past the last row, so clamp the slice and drop the
        # surplus lanes from both the reduction and the stores.
        row_safe = row_valid.select(row, 0)
        lane_safe = lane
        if const_expr(PARTIAL_ROW):
            lane_valid = lane < TILES
            lane_safe = lane_valid.select(lane, 0)
            active = row_valid & lane_valid
        else:
            active = row_valid

        Input_buf = fx.rocdl.make_buffer_tensor(Input)
        ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
        Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
        Output_buf = fx.rocdl.make_buffer_tensor(Output)
        ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)
        YScale_buf = fx.rocdl.make_buffer_tensor(YScale)
        if const_expr(is_smooth):
            XScale_buf = fx.rocdl.make_buffer_tensor(XScale)
        else:
            # dynamicquant reuses the XScale slot for Gamma and never reads it.
            XScale_buf = Gamma_buf

        row_in = fx.slice(Input_buf, (row_safe, None))
        row_residual_in = fx.slice(ResidualIn_buf, (row_safe, None))
        row_out = fx.slice(Output_buf, (row_safe, None))
        row_residual_out = fx.slice(ResidualOut_buf, (row_safe, None))

        in_div = fx.logical_divide(row_in, fx.make_layout(VEC_N, 1))
        residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(VEC_N, 1))
        residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(VEC_N, 1))
        out_div_q = fx.logical_divide(row_out, fx.make_layout(VEC_N, 1))
        gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_N, 1))
        if const_expr(is_smooth):
            xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(VEC_N, 1))

        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
        copy_atom_q = fx.make_copy_atom(fx.rocdl.BufferCopy64b() if QUANT_GROUPS == 2 else fx.rocdl.BufferCopy32b(), 8)
        abs_mask = fx.full(VEC_N, 0x7FFFFFFF, fx.Uint32)

        # The gate is shared by every row of the block, so its loads overlap the
        # activation loads below and then hit in L1 for the remaining rows.
        gate_v = _load_vec(copy_atom, VEC_N, elem_dtype, gamma_div, lane_safe).to(fx.Float32)
        if const_expr(is_smooth):
            gate_v = gate_v * _load_vec(copy_atom, VEC_N, elem_dtype, xscale_div, lane_safe).to(fx.Float32)

        x = _load_vec(copy_atom, VEC_N, elem_dtype, in_div, lane_safe).to(fx.Float32)
        residual = _load_vec(copy_atom, VEC_N, elem_dtype, residual_in_div, lane_safe).to(fx.Float32)
        added_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, x + residual)
        if active:
            _store_vec(copy_atom, VEC_N, elem_dtype, added_e, residual_out_div, lane_safe)
        added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
        scaled = added * gate_v
        sumsq = (added * added).reduce(ReductionOp.ADD, fastmath=fm_fast)
        scaled_abs = (scaled.bitcast(fx.Uint32) & abs_mask).bitcast(fx.Float32)
        row_max_local = scaled_abs.reduce(ReductionOp.MAX)
        if const_expr(PARTIAL_ROW):
            sumsq = active.select(sumsq, c_zero_f)
            row_max_local = active.select(row_max_local, c_zero_f)

        sum_sq = fx.coop.warp_reduce(sumsq, fx.ReductionOp.ADD, width=LANES)
        scaled_max = fx.coop.warp_reduce(row_max_local, fx.ReductionOp.MAX, width=LANES)

        with fx.fastmath(fm_fast):
            mean_sq = sum_sq * inv_n
            rrms = fmath.rsqrt(mean_sq + eps_c, fastmath=fm_fast)
            row_max = scaled_max * rrms
            scale = row_max * c_inv_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)
            quant_scale = rrms / final_scale

        if lane == 0:
            if row_valid:
                YScale_buf[row] = final_scale

        q_i8 = (scaled * quant_scale).to(quant_dtype)
        if active:
            _store_vec(copy_atom_q, VEC_N, quant_dtype, q_i8, out_div_q, lane_safe)

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
            _validate_norm_operand_dtypes(
                Input, dtype_str, ResidualIn=ResidualIn, Gamma=Gamma, XScale=XScale, ResidualOut=ResidualOut
            )
            launcher = fused_add_rmsnorm_narrow_quant_kernel(
                Input, ResidualIn, Gamma, XScale, YScale, Output, ResidualOut, m_in
            )
            launcher.launch(
                grid=((m_in + BLOCK_M - 1) // BLOCK_M, 1, 1),
                block=(THREADS, 1, 1),
                stream=stream,
            )

        return launch_fused_add_rmsnorm_smoothquant

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
        _validate_norm_operand_dtypes(Input, dtype_str, ResidualIn=ResidualIn, Gamma=Gamma, ResidualOut=ResidualOut)
        launcher = fused_add_rmsnorm_narrow_quant_kernel(
            Input, ResidualIn, Gamma, Gamma, YScale, Output, ResidualOut, m_in
        )
        launcher.launch(
            grid=((m_in + BLOCK_M - 1) // BLOCK_M, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    return launch_fused_add_rmsnorm_dynamicquant


def _quant_row_bytes(N: int, dtype_str: str) -> int:
    """Bytes a row costs when the kernel reads both inputs and writes both outputs."""
    elem_bytes = 4 if dtype_str == "f32" else 2
    return N * (3 * elem_bytes + 1)


def _fused_add_quant_threads(N: int, dtype_str: str, chunks_per_thread: int) -> int:
    """Block width for one row: ``chunks_per_thread`` 128-bit chunks per thread.

    A quantized forward row lives in one block, so the block width is that row's
    only parallelism.  Few rows means a wave-limited device: the widest block
    shortens both memory phases and the barrier between them.  Many rows already
    fill it, and a narrower block then wins on occupancy, so the caller picks
    the geometry from M at run time.
    """
    vec_elems = 4 if dtype_str == "f32" else VEC_WIDTH
    tiles = max(1, N // vec_elems)
    chunks = max(1, (tiles + chunks_per_thread - 1) // chunks_per_thread)
    threads = (chunks + WARP_SIZE - 1) // WARP_SIZE * WARP_SIZE
    return max(WARP_SIZE, min(threads, MAX_QUANT_BLOCK_THREADS))


def _build_fused_add_quant_dispatch(
    N: int,
    dtype_str: str,
    *,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    """Pick the row kernel per call from M: geometry, and cache policy.

    A row that fits inside one wave takes the packed narrow-row kernel, which
    carries several rows per block and needs no block-wide reduction.  Wider
    rows keep one block per row and pick a geometry from M at run time: rows
    that fit the device across M are latency bound and want the widest block,
    while a run with many rows per CU is throughput bound and prefers two chunks
    per thread.  Two cache policies: a row set that no longer fits the
    last-level cache is streamed non-temporally, because its activation lines are
    touched once and allocating them only evicts the stores still in flight; a
    smaller one keeps the cached policy, where streaming would throw away reuse.
    M stays a runtime argument, so one launcher serves every row count.
    """
    if _narrow_row_tiles(N, dtype_str):
        return _build_fused_add_rmsnorm_narrow_quant_module(
            N,
            dtype_str,
            is_smooth=True,
            quant_dtype_str=quant_dtype_str,
            eps=eps,
        )

    latency_threads = _fused_add_quant_threads(N, dtype_str, 1)
    throughput_threads = _fused_add_quant_threads(N, dtype_str, 2)
    nt_min_rows = max(1, NT_MIN_WORKING_SET // _quant_row_bytes(N, dtype_str))
    never = 1 << 62

    launch_latency = _build_fused_add_rmsnorm_quant_module(
        N,
        dtype_str,
        is_smooth=True,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
        block_threads=latency_threads,
    )
    if throughput_threads != latency_threads:
        launch_throughput = _build_fused_add_rmsnorm_quant_module(
            N,
            dtype_str,
            is_smooth=True,
            quant_dtype_str=quant_dtype_str,
            eps=eps,
            block_threads=throughput_threads,
        )
        throughput_rows = QUANT_THROUGHPUT_MIN_ROWS
    else:
        launch_throughput = launch_latency
        throughput_rows = never

    if N > SMALL_N_THRESHOLD:
        launch_streaming = _build_fused_add_rmsnorm_quant_module(
            N,
            dtype_str,
            is_smooth=True,
            quant_dtype_str=quant_dtype_str,
            eps=eps,
            block_threads=throughput_threads,
            input_cache_modifier=2,
        )
    else:
        launch_streaming = launch_throughput
        nt_min_rows = never

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
        def run_latency():
            launch_latency(Input, ResidualIn, Gamma, XScale, Output, ResidualOut, YScale, m_in, stream)

        def run_throughput():
            launch_throughput(Input, ResidualIn, Gamma, XScale, Output, ResidualOut, YScale, m_in, stream)

        def run_streaming():
            launch_streaming(Input, ResidualIn, Gamma, XScale, Output, ResidualOut, YScale, m_in, stream)

        @flyc.jit
        def dispatch():
            if m_in > fx.Int32(nt_min_rows):
                run_streaming()
            else:
                if m_in >= fx.Int32(throughput_rows):
                    run_throughput()
                else:
                    run_latency()

        dispatch()

    return launch_fused_add_rmsnorm_smoothquant


def build_fused_add_rmsnorm_dynamicquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    """Fused add + RMSNorm + per-row int8 dynamic quantization (no smoothing scale)."""
    builder = (
        _build_fused_add_rmsnorm_narrow_quant_module
        if _narrow_row_tiles(N, dtype_str)
        else _build_fused_add_rmsnorm_quant_module
    )
    return builder(
        N,
        dtype_str,
        is_smooth=False,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
    )


def build_fused_add_rmsnorm_smoothquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    return _build_fused_add_quant_dispatch(
        N,
        dtype_str,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
    )


# Python wrappers and autograd.
if torch is not None:
    from kernels.common.tensor_shim import _run_compiled
    from kernels.norm.rmsnorm_common import torch_dtype_to_str as _torch_dtype_to_str

    # Each cache retains one JitFunction launcher per specialization and device;
    # _run_compiled owns the launcher's CompiledFunction in ``._cf``. eps is a
    # compile-time forward constant, so it remains part of those cache keys.
    _FWD_CACHE: dict = {}
    _BWD_CACHE: dict = {}

    # A cached FlyDSL callable accepts the raw cudaStream_t directly.  Avoid
    # constructing a Python Stream wrapper or entering a no-op device guard on
    # the hot same-device path, while retaining the guard when tensors belong
    # to a non-current device.
    _get_current_raw_stream = getattr(
        torch._C,
        "_cuda_getCurrentRawStream",
        lambda device_index: torch.cuda.current_stream(device_index).cuda_stream,
    )

    def _run_compiled_on_device(launcher, device, args):
        device_index = device.index
        if torch.cuda.current_device() == device_index:
            _run_compiled(launcher, *args, _get_current_raw_stream(device_index))
        else:
            with torch.cuda.device(device_index):
                _run_compiled(launcher, *args, _get_current_raw_stream(device_index))

    # The staged path needs enough rows to amortize its second launch.  Its
    # 512-thread vec8 kernel needs fewer persistent programs than the scalar
    # fallback, which trades more programs for latency hiding.  Keep this
    # selector as the single source of truth shared by plain and fused wrappers.
    _BWD_TWO_STAGE_MIN_ROWS = 512
    _BWD_TWO_STAGE_MAX_N = 8192
    _BWD_CU_COUNT_CACHE: dict = {}

    def _get_rmsnorm_bwd_cu_count(device):
        num_cus = _BWD_CU_COUNT_CACHE.get(device)
        if num_cus is None:
            num_cus = torch.cuda.get_device_properties(device).multi_processor_count
            _BWD_CU_COUNT_CACHE[device] = num_cus
        return num_cus

    def _select_rmsnorm_bwd_config(M, N, dtype_str, device):
        if M >= _BWD_TWO_STAGE_MIN_ROWS and N <= _BWD_TWO_STAGE_MAX_N:
            num_cus = _get_rmsnorm_bwd_cu_count(device)
            if is_rmsnorm_bwd_two_stage_vec_config(N, dtype_str):
                num_programs = num_cus if M < 2048 else (3 * num_cus) // 2
            else:
                num_programs = num_cus if M < 1024 else 2 * num_cus
            return ("two_stage", min(M, num_programs))
        return ("atomic", None)

    def _get_fwd_launcher(
        N,
        dtype_str,
        weight_dtype_str,
        store_rstd,
        eps,
        device,
    ):
        key = (N, dtype_str, weight_dtype_str, store_rstd, float(eps), device)
        launcher = _FWD_CACHE.get(key)
        if launcher is None:
            with torch.cuda.device(device):
                launcher = build_rmsnorm_module(
                    N,
                    dtype_str,
                    store_rstd=store_rstd,
                    eps=eps,
                    weight_dtype_str=weight_dtype_str,
                )
            _FWD_CACHE[key] = launcher
        return launcher

    def rmsnorm_fwd(x, weight, eps=EPS, store_rstd=False):
        """Forward RMSNorm. Returns (out, rstd). eps is baked into the kernel."""
        assert x.dim() == 2, "rmsnorm_fwd expects a 2D (M, N) input"
        assert x.is_contiguous() and weight.is_contiguous(), "rmsnorm_fwd expects contiguous inputs"
        assert weight.device == x.device, "rmsnorm_fwd: weight and x must be on the same device"
        device = x.device
        M, N = x.shape
        out = torch.empty_like(x)
        rstd = torch.empty((M,), device=device, dtype=torch.float32) if store_rstd else None
        dtype_str = _torch_dtype_to_str(x.dtype)
        weight_dtype_str = _resolve_rmsnorm_weight_dtype(dtype_str, _torch_dtype_to_str(weight.dtype))
        launcher = _get_fwd_launcher(
            N,
            dtype_str,
            weight_dtype_str,
            store_rstd,
            eps,
            device,
        )
        args = (x, weight, out, rstd, M) if store_rstd else (x, weight, out, M)
        _run_compiled_on_device(launcher, device, args)
        return out, rstd

    def rmsnorm_bwd(x, weight, dout, rstd, eps=EPS):
        """Backward RMSNorm. Returns (dx, dw) with dw in weight dtype.

        eps is not used directly here — it is already baked into `rstd` by the
        forward — but is accepted so callers can pass it symmetrically.
        """
        device = x.device
        dtype = x.dtype
        assert x.dim() == 2, "rmsnorm_bwd expects a 2D (M, N) input"
        assert x.is_contiguous() and dout.is_contiguous(), "rmsnorm_bwd expects contiguous inputs"
        assert weight.is_contiguous(), "rmsnorm_bwd: weight must be contiguous"
        assert weight.device == device == dout.device == rstd.device, "rmsnorm_bwd: inputs must share a device"
        assert dout.dtype == dtype, "rmsnorm_bwd: dout dtype must match x"
        M, N = x.shape
        dtype_str = _torch_dtype_to_str(dtype)
        weight_dtype_str = _resolve_rmsnorm_weight_dtype(dtype_str, _torch_dtype_to_str(weight.dtype))
        dx = torch.empty_like(x)
        path, num_programs = _select_rmsnorm_bwd_config(M, N, dtype_str, device)
        if path == "two_stage":
            dweight = torch.empty_like(weight)
            partial = torch.empty((num_programs * N,), device=device, dtype=torch.float32)
            key = (path, N, dtype_str, weight_dtype_str, num_programs, device)
            launcher = _BWD_CACHE.get(key)
            if launcher is None:
                # The builder has an arch-dependent vec conversion choice, and
                # compilation/code-object loading are device-context bound.
                with torch.cuda.device(device):
                    device_index = device.index
                    launcher = build_rmsnorm_bwd_two_stage_module(
                        N,
                        dtype_str,
                        num_programs,
                        weight_dtype_str=weight_dtype_str,
                    )
                    _run_compiled(
                        launcher,
                        x,
                        weight,
                        dout,
                        rstd,
                        dx,
                        dweight,
                        partial,
                        M,
                        _get_current_raw_stream(device_index),
                    )
                _BWD_CACHE[key] = launcher
            else:
                _run_compiled_on_device(launcher, device, (x, weight, dout, rstd, dx, dweight, partial, M))
            return dx, dweight

        dweight = torch.empty((N,), device=device, dtype=torch.float32)
        key = (path, N, dtype_str, weight_dtype_str, num_programs, device)
        launcher = _BWD_CACHE.get(key)
        dweight.zero_()
        if launcher is None:
            with torch.cuda.device(device):
                device_index = device.index
                launcher = build_rmsnorm_bwd_module(N, dtype_str, weight_dtype_str=weight_dtype_str)
                _run_compiled(
                    launcher,
                    x,
                    weight,
                    dout,
                    rstd,
                    dx,
                    dweight,
                    M,
                    _get_current_raw_stream(device_index),
                )
            _BWD_CACHE[key] = launcher
        else:
            _run_compiled_on_device(launcher, device, (x, weight, dout, rstd, dx, dweight, M))
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

    # Fused-add / prenorm wrappers and autograd.
    _FUSED_ADD_FWD_CACHE: dict = {}
    _FUSED_ADD_BWD_CACHE: dict = {}

    def _get_fused_add_fwd_launcher(
        N,
        dtype_str,
        weight_dtype_str,
        store_rstd,
        eps,
        device,
    ):
        key = (N, dtype_str, weight_dtype_str, store_rstd, float(eps), device)
        launcher = _FUSED_ADD_FWD_CACHE.get(key)
        if launcher is None:
            with torch.cuda.device(device):
                launcher = build_fused_add_rmsnorm_module(
                    N,
                    dtype_str,
                    store_rstd=store_rstd,
                    eps=eps,
                    weight_dtype_str=weight_dtype_str,
                )
            _FUSED_ADD_FWD_CACHE[key] = launcher
        return launcher

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
        assert x.dtype == residual.dtype, f"x/residual dtypes must match, got {x.dtype}/{residual.dtype}"
        # Only x.device gates compile/stream/cache; all operands must co-reside.
        assert (
            x.device == residual.device == weight.device
        ), f"x/residual/weight must be on the same device, got {x.device}/{residual.device}/{weight.device}"
        M, N = x.shape
        out = torch.empty_like(x)
        residual_out = torch.empty_like(x)
        device = x.device
        rstd = torch.empty((M,), device=device, dtype=torch.float32) if store_rstd else None
        dtype_str = _torch_dtype_to_str(x.dtype)
        weight_dtype_str = _resolve_rmsnorm_weight_dtype(dtype_str, _torch_dtype_to_str(weight.dtype))
        launcher = _get_fused_add_fwd_launcher(
            N,
            dtype_str,
            weight_dtype_str,
            store_rstd,
            eps,
            device,
        )
        args = (
            (x, residual, weight, out, residual_out, rstd, M)
            if store_rstd
            else (x, residual, weight, out, residual_out, M)
        )
        _run_compiled_on_device(launcher, device, args)
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
        device = added.device
        dtype = added.dtype
        assert added.dim() == 2, "fused_add_rmsnorm_bwd expects a 2D (M, N) input"
        assert added.is_contiguous() and dout.is_contiguous(), "fused_add_rmsnorm_bwd expects contiguous inputs"
        assert dtype == dout.dtype, f"added/dout dtypes must match, got {dtype}/{dout.dtype}"
        assert (
            device == weight.device == dout.device == rstd.device
        ), "fused_add_rmsnorm_bwd expects all tensors on the same device"
        if dresidual_out is not None:
            assert dresidual_out.is_contiguous(), "fused_add_rmsnorm_bwd expects contiguous dresidual_out"
            assert dresidual_out.dtype == dtype, "dresidual_out dtype must match added"
            assert dresidual_out.device == device, "dresidual_out must be on the same device as added"
        M, N = added.shape
        dtype_str = _torch_dtype_to_str(dtype)
        weight_dtype_str = _resolve_rmsnorm_weight_dtype(dtype_str, _torch_dtype_to_str(weight.dtype))
        if dresidual_out is None:
            dresidual_out = torch.zeros_like(added)
        dx = torch.empty_like(added)
        path, num_programs = _select_rmsnorm_bwd_config(M, N, dtype_str, device)
        if path == "two_stage":
            dweight = torch.empty_like(weight)
            partial = torch.empty((num_programs * N,), device=device, dtype=torch.float32)
            key = (path, N, dtype_str, weight_dtype_str, num_programs, device)
            launcher = _FUSED_ADD_BWD_CACHE.get(key)
            if launcher is None:
                with torch.cuda.device(device):
                    device_index = device.index
                    launcher = build_fused_add_rmsnorm_bwd_two_stage_module(
                        N,
                        dtype_str,
                        num_programs,
                        weight_dtype_str=weight_dtype_str,
                    )
                    _run_compiled(
                        launcher,
                        added,
                        weight,
                        dout,
                        dresidual_out,
                        rstd,
                        dx,
                        dweight,
                        partial,
                        M,
                        _get_current_raw_stream(device_index),
                    )
                _FUSED_ADD_BWD_CACHE[key] = launcher
            else:
                _run_compiled_on_device(
                    launcher,
                    device,
                    (added, weight, dout, dresidual_out, rstd, dx, dweight, partial, M),
                )
            return dx, dx, dweight

        dweight = torch.empty((N,), device=device, dtype=torch.float32)
        key = (path, N, dtype_str, weight_dtype_str, num_programs, device)
        launcher = _FUSED_ADD_BWD_CACHE.get(key)
        dweight.zero_()
        if launcher is None:
            with torch.cuda.device(device):
                device_index = device.index
                launcher = build_fused_add_rmsnorm_bwd_module(
                    N,
                    dtype_str,
                    weight_dtype_str=weight_dtype_str,
                )
                _run_compiled(
                    launcher,
                    added,
                    weight,
                    dout,
                    dresidual_out,
                    rstd,
                    dx,
                    dweight,
                    M,
                    _get_current_raw_stream(device_index),
                )
            _FUSED_ADD_BWD_CACHE[key] = launcher
        else:
            _run_compiled_on_device(
                launcher,
                device,
                (added, weight, dout, dresidual_out, rstd, dx, dweight, M),
            )
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
