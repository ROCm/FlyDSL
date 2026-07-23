# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""RMSNorm backward kernel builders (plain + fused-add / prenorm).

Split out of ``rmsnorm_kernel.py`` so the training-only backward path lives in
its own module (per review on #800). Device-side helpers and constants are
shared via ``rmsnorm_common.py``; the forward builders and the autograd glue
that ties forward+backward together stay in ``rmsnorm_kernel.py``.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import ReductionOp
from flydsl.runtime.device import get_rocm_arch
from kernels.common.kernels_common import atomic_add, dtype_to_elem_type
from kernels.norm.rmsnorm_common import (
    BLOCK_THREADS,
    VEC_WIDTH,
    WARP_SIZE,
    load_scalar,
    load_vec,
    make_single_reduction_storage,
    store_scalar,
    store_vec,
    to_elem_vec,
)

# The staged path is selected by the Python wrapper only when M is large
# enough to keep one persistent program per CU busy.  NUM_PROGRAMS is a build
# parameter (a small multiple of the device CU count); M remains a runtime
# argument so a compiled callable can be reused across row counts.
DWEIGHT_REDUCE_COLS = 64
DWEIGHT_REDUCE_ROW_LANES = 4
DWEIGHT_REDUCE_THREADS = DWEIGHT_REDUCE_COLS * DWEIGHT_REDUCE_ROW_LANES
TWO_STAGE_PARTIAL_THREADS = 512


def is_rmsnorm_bwd_two_stage_vec_config(N: int, dtype_str: str) -> bool:
    """Whether the staged main kernel can use full-width vec8 column I/O."""
    return dtype_str in ("f16", "bf16") and N >= TWO_STAGE_PARTIAL_THREADS * VEC_WIDTH and N % VEC_WIDTH == 0


def build_rmsnorm_bwd_module(N: int, dtype_str: str):
    """Fused RMSNorm backward: grid=(M,), one block per row.

    Pass 1: c1 = mean_N(x_hat * wdy), x_hat = x*rstd, wdy = dy*gamma.
    Pass 2: dx = (wdy - x_hat*c1) * rstd  -> DX (elem dtype);
            dw_elem = dy * x_hat (fp32)   -> atomicAdd into DWeight[idx] (fp32).
    eps is baked into Rstd by the forward, so it is not needed here.

    Perf follow-ups (deferred; correctness-complete as-is): this is the generic
    scalar path only — a vectorized fast path (mirroring the forward) and caching
    x/dy/gamma between pass 1 and pass 2 (the forward caches `in_local`) would cut
    global traffic. Left out of PR 1 to keep the first backward reviewable.
    """
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    SharedStorage = make_single_reduction_storage(RED_SLOTS)

    @flyc.kernel
    def rmsnorm_bwd_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        DY: fx.Tensor,
        Rstd: fx.Tensor,
        DX: fx.Tensor,
        DWeight: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add(val):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val)
            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            w = wave_reduce_add(val)
            if lane == 0:
                fx.memref_store(w, s_red, wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_red, lane_safe)
                ww = in_range.select(v, c_zero_f)
                ww = wave_reduce_add(ww)
                if lane == 0:
                    fx.memref_store(ww, s_red, 0)
            gpu.barrier()
            return fx.memref_load(s_red, 0)

        Input_buf = fx.rocdl.make_buffer_tensor(Input)
        Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
        DY_buf = fx.rocdl.make_buffer_tensor(DY)
        Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)
        DX_buf = fx.rocdl.make_buffer_tensor(DX)

        row_in = fx.slice(Input_buf, (bid, None))
        row_dy = fx.slice(DY_buf, (bid, None))
        row_dx = fx.slice(DX_buf, (bid, None))

        copy_atom_s = fx.make_copy_atom(
            fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
            elem_bits,
        )
        copy_atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
        dy_div = fx.logical_divide(row_dy, fx.make_layout(1, 1))
        gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
        dx_div = fx.logical_divide(row_dx, fx.make_layout(1, 1))
        rstd_div = fx.logical_divide(Rstd_buf, fx.make_layout(1, 1))

        rstd = load_scalar(copy_atom_f32, fx.Float32, rstd_div, bid)

        # Pass 1: c1 = mean( x_hat * wdy ) = mean( (x*rstd) * (dy*gamma) )
        thread_acc = c_zero_f
        for base in range_constexpr(0, N, BLOCK_THREADS):
            idx = tid + base
            is_valid = idx < N
            idx_safe = is_valid.select(idx, 0)
            x_e = load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
            dy_e = load_scalar(copy_atom_s, elem_dtype, dy_div, idx_safe)
            g_e = load_scalar(copy_atom_s, elem_dtype, gamma_div, idx_safe)
            x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
            dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
            g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
            x_hat = x * rstd
            wdy = dy * g
            prod = x_hat * wdy
            thread_acc = thread_acc + is_valid.select(prod, c_zero_f)

        sum_prod = block_reduce_add(thread_acc)
        c1 = sum_prod / n_float

        # Pass 2: dx = (wdy - x_hat*c1) * rstd ; dw = dy * x_hat (atomicAdd fp32)
        for base in range_constexpr(0, N, BLOCK_THREADS):
            idx = tid + base
            if idx < N:
                x_e = load_scalar(copy_atom_s, elem_dtype, row_div, idx)
                dy_e = load_scalar(copy_atom_s, elem_dtype, dy_div, idx)
                g_e = load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                x_hat = x * rstd
                wdy = dy * g
                dx = (wdy - x_hat * c1) * rstd
                dx_e = dx if dtype_str == "f32" else dx.to(elem_dtype)
                store_scalar(copy_atom_s, elem_dtype, elem_dtype, dx_div, idx, dx_e)

                dw = dy * x_hat
                atomic_add(DWeight, idx, dw, dtype_bytes=4)

    @flyc.jit
    def launch_rmsnorm_bwd(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        DY: fx.Tensor,
        Rstd: fx.Tensor,
        DX: fx.Tensor,
        DWeight: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = rmsnorm_bwd_kernel(Input, Gamma, DY, Rstd, DX, DWeight)
        launcher.launch(grid=(m_in, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_rmsnorm_bwd


def build_fused_add_rmsnorm_bwd_module(N: int, dtype_str: str):
    """Fused-add / prenorm RMSNorm backward: grid=(M,), one block per row.

    Inputs: Added (= residual_out from fwd), Gamma, DY (= dout), DResidualOut
    (grad flowing into residual_out from downstream), Rstd.

    With a_hat = Added*rstd, wdy = dy*gamma, c1 = mean_N(a_hat*wdy):
      d_added = (wdy - a_hat*c1) * rstd          (gradient through the norm)
      total   = d_added + dresidual_out_elem     (downstream residual grad)
    Since added = x + residual_in, dx == dresidual == total unconditionally, so
    the kernel writes `total` to DX only; the python wrapper returns dx as both
    grads (aliased). dweight = sum_rows(dy * a_hat) (fp32 atomicAdd).

    eps is baked into Rstd by the forward, so it is not needed here.

    DResidualOut is ALWAYS a real tensor: the python wrapper passes a zero
    tensor when the caller has no downstream residual grad (pure-norm case).
    This keeps the kernel branch-free wrt None.

    Perf follow-ups (deferred; correctness-complete): generic scalar path only —
    a vectorized fast path + caching between passes would cut global traffic.
    """
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    SharedStorage = make_single_reduction_storage(RED_SLOTS)

    @flyc.kernel
    def fused_add_rmsnorm_bwd_kernel(
        Added: fx.Tensor,
        Gamma: fx.Tensor,
        DY: fx.Tensor,
        DResidualOut: fx.Tensor,
        Rstd: fx.Tensor,
        DX: fx.Tensor,
        DWeight: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add(val):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val)
            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            w = wave_reduce_add(val)
            if lane == 0:
                fx.memref_store(w, s_red, wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_red, lane_safe)
                ww = in_range.select(v, c_zero_f)
                ww = wave_reduce_add(ww)
                if lane == 0:
                    fx.memref_store(ww, s_red, 0)
            gpu.barrier()
            return fx.memref_load(s_red, 0)

        Added_buf = fx.rocdl.make_buffer_tensor(Added)
        Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
        DY_buf = fx.rocdl.make_buffer_tensor(DY)
        DResidualOut_buf = fx.rocdl.make_buffer_tensor(DResidualOut)
        Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)
        DX_buf = fx.rocdl.make_buffer_tensor(DX)

        row_added = fx.slice(Added_buf, (bid, None))
        row_dy = fx.slice(DY_buf, (bid, None))
        row_dres_out = fx.slice(DResidualOut_buf, (bid, None))
        row_dx = fx.slice(DX_buf, (bid, None))

        copy_atom_s = fx.make_copy_atom(
            fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
            elem_bits,
        )
        copy_atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        added_div = fx.logical_divide(row_added, fx.make_layout(1, 1))
        dy_div = fx.logical_divide(row_dy, fx.make_layout(1, 1))
        dres_out_div = fx.logical_divide(row_dres_out, fx.make_layout(1, 1))
        gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
        dx_div = fx.logical_divide(row_dx, fx.make_layout(1, 1))
        rstd_div = fx.logical_divide(Rstd_buf, fx.make_layout(1, 1))

        rstd = load_scalar(copy_atom_f32, fx.Float32, rstd_div, bid)

        # Pass 1: c1 = mean( a_hat * wdy ) = mean( (added*rstd) * (dy*gamma) )
        thread_acc = c_zero_f
        for base in range_constexpr(0, N, BLOCK_THREADS):
            idx = tid + base
            is_valid = idx < N
            idx_safe = is_valid.select(idx, 0)
            a_e = load_scalar(copy_atom_s, elem_dtype, added_div, idx_safe)
            dy_e = load_scalar(copy_atom_s, elem_dtype, dy_div, idx_safe)
            g_e = load_scalar(copy_atom_s, elem_dtype, gamma_div, idx_safe)
            a = a_e if dtype_str == "f32" else a_e.to(fx.Float32)
            dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
            g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
            a_hat = a * rstd
            wdy = dy * g
            prod = a_hat * wdy
            thread_acc = thread_acc + is_valid.select(prod, c_zero_f)

        sum_prod = block_reduce_add(thread_acc)
        c1 = sum_prod / n_float

        # Pass 2: d_added = (wdy - a_hat*c1) * rstd ; total = d_added + dres_out
        #         store total -> DX only (the wrapper aliases dx as dresidual, since
        #         added = x + residual_in => dx == dresidual) ; dw = dy*a_hat (atomicAdd fp32)
        for base in range_constexpr(0, N, BLOCK_THREADS):
            idx = tid + base
            if idx < N:
                a_e = load_scalar(copy_atom_s, elem_dtype, added_div, idx)
                dy_e = load_scalar(copy_atom_s, elem_dtype, dy_div, idx)
                g_e = load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                dres_out_e = load_scalar(copy_atom_s, elem_dtype, dres_out_div, idx)
                a = a_e if dtype_str == "f32" else a_e.to(fx.Float32)
                dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                dres_out = dres_out_e if dtype_str == "f32" else dres_out_e.to(fx.Float32)
                a_hat = a * rstd
                wdy = dy * g
                d_added = (wdy - a_hat * c1) * rstd
                total = d_added + dres_out
                total_e = total if dtype_str == "f32" else total.to(elem_dtype)
                store_scalar(copy_atom_s, elem_dtype, elem_dtype, dx_div, idx, total_e)

                dw = dy * a_hat
                atomic_add(DWeight, idx, dw, dtype_bytes=4)

    @flyc.jit
    def launch_fused_add_rmsnorm_bwd(
        Added: fx.Tensor,
        Gamma: fx.Tensor,
        DY: fx.Tensor,
        DResidualOut: fx.Tensor,
        Rstd: fx.Tensor,
        DX: fx.Tensor,
        DWeight: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = fused_add_rmsnorm_bwd_kernel(Added, Gamma, DY, DResidualOut, Rstd, DX, DWeight)
        launcher.launch(grid=(m_in, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_fused_add_rmsnorm_bwd


def _build_rmsnorm_bwd_two_stage_module(
    N: int,
    dtype_str: str,
    num_programs: int,
    *,
    fused_add: bool,
):
    """Build the large-M persistent backward + dweight finalizer.

    The 512-thread main kernel walks rows in a grid-stride loop, writes every
    assigned ``dx`` row, and accumulates one FP32 dweight partial in registers.
    Full-width 16-bit rows use vec8 I/O and retain source, dy, and gamma across
    the row reduction; other shapes keep the scalar fallback.  Each program writes
    one ``N``-wide partial row, then a second kernel reduces all partial rows and
    stores dweight directly in the parameter dtype.

    This replaces M competing atomic adds per dweight element with a bounded
    ``num_programs x N`` workspace and a deterministic final reduction.  The
    caller owns the workspace for the duration of one invocation; the builder
    never caches writable storage.
    """
    if num_programs <= 0:
        raise ValueError(f"num_programs must be positive, got {num_programs}")

    RED_SLOTS = max(1, (TWO_STAGE_PARTIAL_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    USE_VEC = is_rmsnorm_bwd_two_stage_vec_config(N, dtype_str)
    IO_WIDTH = VEC_WIDTH if USE_VEC else 1
    NUM_IO_TILES = (N + IO_WIDTH - 1) // IO_WIDTH
    NUM_IO_ITERS = (NUM_IO_TILES + TWO_STAGE_PARTIAL_THREADS - 1) // TWO_STAGE_PARTIAL_THREADS
    PARTIAL_ACC_SIZE = NUM_IO_ITERS * IO_WIDTH
    arch = get_rocm_arch() if USE_VEC else ""
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")
    SharedStorage = make_single_reduction_storage(RED_SLOTS)
    DWeightReduceStorage = make_single_reduction_storage(DWEIGHT_REDUCE_THREADS)

    @flyc.kernel(known_block_size=[TWO_STAGE_PARTIAL_THREADS, 1, 1])
    def rmsnorm_bwd_partial_kernel(
        Source: fx.Tensor,
        Gamma: fx.Tensor,
        DY: fx.Tensor,
        DResidualOut: fx.Tensor,
        Rstd: fx.Tensor,
        DX: fx.Tensor,
        DWeightPartial: fx.Tensor,
        MIn: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add(val):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val)
            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            w = wave_reduce_add(val)
            if lane == 0:
                fx.memref_store(w, s_red, wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_red, lane_safe)
                ww = in_range.select(v, c_zero_f)
                ww = wave_reduce_add(ww)
                if lane == 0:
                    fx.memref_store(ww, s_red, 0)
            gpu.barrier()
            return fx.memref_load(s_red, 0)

        Source_buf = fx.rocdl.make_buffer_tensor(Source)
        Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
        DY_buf = fx.rocdl.make_buffer_tensor(DY)
        DResidualOut_buf = fx.rocdl.make_buffer_tensor(DResidualOut)
        Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)
        DX_buf = fx.rocdl.make_buffer_tensor(DX)
        DWeightPartial_buf = fx.rocdl.make_buffer_tensor(DWeightPartial)

        rstd_div = fx.logical_divide(Rstd_buf, fx.make_layout(1, 1))
        partial_div = fx.logical_divide(DWeightPartial_buf, fx.make_layout(1, 1))

        copy_atom_io = fx.make_copy_atom(
            (
                fx.rocdl.BufferCopy128b()
                if USE_VEC
                else (fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b())
            ),
            elem_bits,
        )
        copy_atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(IO_WIDTH, 1))
        gamma_local = []
        if const_expr(USE_VEC):
            for tile_i in range_constexpr(NUM_IO_ITERS):
                io_idx = tid + tile_i * TWO_STAGE_PARTIAL_THREADS
                is_valid = io_idx < NUM_IO_TILES
                io_idx_safe = is_valid.select(io_idx, 0)
                gamma_local.append(load_vec(copy_atom_io, IO_WIDTH, elem_dtype, gamma_div, io_idx_safe))

        dweight_partial = fx.Vector.filled(PARTIAL_ACC_SIZE, 0.0, fx.Float32)
        for row in range(fx.Int32(bid), MIn, num_programs):
            row_source = fx.slice(Source_buf, (row, None))
            row_dy = fx.slice(DY_buf, (row, None))
            row_dx = fx.slice(DX_buf, (row, None))
            if const_expr(fused_add):
                row_dres_out = fx.slice(DResidualOut_buf, (row, None))

            source_div = fx.logical_divide(row_source, fx.make_layout(IO_WIDTH, 1))
            dy_div = fx.logical_divide(row_dy, fx.make_layout(IO_WIDTH, 1))
            dx_div = fx.logical_divide(row_dx, fx.make_layout(IO_WIDTH, 1))
            if const_expr(fused_add):
                dres_out_div = fx.logical_divide(row_dres_out, fx.make_layout(IO_WIDTH, 1))

            rstd = load_scalar(copy_atom_f32, fx.Float32, rstd_div, row)
            thread_acc = c_zero_f
            source_local = []
            dy_local = []
            for tile_i in range_constexpr(NUM_IO_ITERS):
                io_idx = tid + tile_i * TWO_STAGE_PARTIAL_THREADS
                is_valid = io_idx < NUM_IO_TILES
                io_idx_safe = is_valid.select(io_idx, 0)
                if const_expr(USE_VEC):
                    source_e = load_vec(copy_atom_io, IO_WIDTH, elem_dtype, source_div, io_idx_safe)
                    dy_e = load_vec(copy_atom_io, IO_WIDTH, elem_dtype, dy_div, io_idx_safe)
                    source_local.append(source_e)
                    dy_local.append(dy_e)
                    gamma_e = gamma_local[tile_i]
                else:
                    source_e = load_scalar(copy_atom_io, elem_dtype, source_div, io_idx_safe)
                    dy_e = load_scalar(copy_atom_io, elem_dtype, dy_div, io_idx_safe)
                    gamma_e = load_scalar(copy_atom_io, elem_dtype, gamma_div, io_idx_safe)

                source = source_e if dtype_str == "f32" else source_e.to(fx.Float32)
                dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
                gamma = gamma_e if dtype_str == "f32" else gamma_e.to(fx.Float32)
                source_hat = source * rstd
                wdy = dy * gamma
                prod = source_hat * wdy
                if const_expr(USE_VEC):
                    prod = prod.reduce(ReductionOp.ADD, fastmath=fm_fast)

                thread_acc = thread_acc + is_valid.select(prod, c_zero_f)

            sum_prod = block_reduce_add(thread_acc)
            c1 = sum_prod / n_float
            row_dweight = []
            for tile_i in range_constexpr(NUM_IO_ITERS):
                io_idx = tid + tile_i * TWO_STAGE_PARTIAL_THREADS
                is_valid = io_idx < NUM_IO_TILES
                io_idx_safe = is_valid.select(io_idx, 0)
                if const_expr(USE_VEC):
                    source_e = source_local[tile_i]
                    dy_e = dy_local[tile_i]
                    gamma_e = gamma_local[tile_i]
                else:
                    source_e = load_scalar(copy_atom_io, elem_dtype, source_div, io_idx_safe)
                    dy_e = load_scalar(copy_atom_io, elem_dtype, dy_div, io_idx_safe)
                    gamma_e = load_scalar(copy_atom_io, elem_dtype, gamma_div, io_idx_safe)

                source = source_e if dtype_str == "f32" else source_e.to(fx.Float32)
                dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
                gamma = gamma_e if dtype_str == "f32" else gamma_e.to(fx.Float32)
                source_hat = source * rstd
                wdy = dy * gamma

                dx = (wdy - source_hat * c1) * rstd
                if const_expr(fused_add):
                    if const_expr(USE_VEC):
                        dres = load_vec(copy_atom_io, IO_WIDTH, elem_dtype, dres_out_div, io_idx_safe).to(fx.Float32)
                    else:
                        dres_e = load_scalar(copy_atom_io, elem_dtype, dres_out_div, io_idx_safe)
                        dres = dres_e if dtype_str == "f32" else dres_e.to(fx.Float32)
                    dx = dx + dres

                if io_idx < NUM_IO_TILES:
                    if const_expr(USE_VEC):
                        dx_e = to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, dx)
                        store_vec(copy_atom_io, IO_WIDTH, elem_dtype, dx_e, dx_div, io_idx)
                    else:
                        dx_e = dx if dtype_str == "f32" else dx.to(elem_dtype)
                        store_scalar(copy_atom_io, elem_dtype, elem_dtype, dx_div, io_idx, dx_e)

                dw = dy * source_hat
                if const_expr(USE_VEC):
                    for lane in range_constexpr(IO_WIDTH):
                        row_dweight.append(is_valid.select(dw[lane], c_zero_f))
                else:
                    row_dweight.append(is_valid.select(dw, c_zero_f))

            dweight_partial = dweight_partial + fx.Vector.from_elements(row_dweight, fx.Float32)
            gpu.barrier()

        for tile_i in range_constexpr(NUM_IO_ITERS):
            io_idx = tid + tile_i * TWO_STAGE_PARTIAL_THREADS
            if io_idx < NUM_IO_TILES:
                for lane in range_constexpr(IO_WIDTH):
                    idx = io_idx * IO_WIDTH + lane
                    partial_idx = bid * N + idx
                    partial_value = dweight_partial[tile_i * IO_WIDTH + lane]
                    store_scalar(
                        copy_atom_f32,
                        fx.Float32,
                        fx.Float32,
                        partial_div,
                        partial_idx,
                        partial_value,
                    )

    @flyc.kernel
    def rmsnorm_bwd_dweight_reduce_kernel(
        DWeightPartial: fx.Tensor,
        DWeight: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        col_lane = tid % DWEIGHT_REDUCE_COLS
        partial_lane = tid // DWEIGHT_REDUCE_COLS
        col = bid * DWEIGHT_REDUCE_COLS + col_lane
        is_valid = col < N
        col_safe = is_valid.select(col, 0)

        elem_dtype = dtype_to_elem_type(dtype_str)
        DWeightPartial_buf = fx.rocdl.make_buffer_tensor(DWeightPartial)
        DWeight_buf = fx.rocdl.make_buffer_tensor(DWeight)
        partial_div = fx.logical_divide(DWeightPartial_buf, fx.make_layout(1, 1))
        dweight_div = fx.logical_divide(DWeight_buf, fx.make_layout(1, 1))
        copy_atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
        copy_atom_s = fx.make_copy_atom(
            fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
            elem_bits,
        )

        lds = fx.SharedAllocator().allocate(DWeightReduceStorage).peek()
        s_partial = lds.s_red.view(fx.make_layout(DWEIGHT_REDUCE_THREADS, 1))

        acc = fx.Float32(0.0)
        for partial_base in range(0, num_programs, DWEIGHT_REDUCE_ROW_LANES):
            partial_row = partial_base + partial_lane
            partial_valid = partial_row < num_programs
            partial_row_safe = partial_valid.select(partial_row, 0)
            partial_idx = partial_row_safe * N + col_safe
            value = load_scalar(copy_atom_f32, fx.Float32, partial_div, partial_idx)
            acc = acc + partial_valid.select(value, fx.Float32(0.0))
        fx.memref_store(acc, s_partial, tid)
        gpu.barrier()

        if partial_lane == 0:
            if col < N:
                total = fx.Float32(0.0)
                for lane in range_constexpr(DWEIGHT_REDUCE_ROW_LANES):
                    total = total + fx.memref_load(s_partial, lane * DWEIGHT_REDUCE_COLS + col_lane)
                out = total if dtype_str == "f32" else total.to(elem_dtype)
                store_scalar(copy_atom_s, elem_dtype, elem_dtype, dweight_div, col, out)

    reduce_grid = (N + DWEIGHT_REDUCE_COLS - 1) // DWEIGHT_REDUCE_COLS

    if fused_add:

        @flyc.jit
        def launch_fused_add_rmsnorm_bwd_two_stage(
            Added: fx.Tensor,
            Gamma: fx.Tensor,
            DY: fx.Tensor,
            DResidualOut: fx.Tensor,
            Rstd: fx.Tensor,
            DX: fx.Tensor,
            DWeight: fx.Tensor,
            DWeightPartial: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            partial_launcher = rmsnorm_bwd_partial_kernel(
                Added,
                Gamma,
                DY,
                DResidualOut,
                Rstd,
                DX,
                DWeightPartial,
                m_in,
            )
            partial_launcher.launch(
                grid=(num_programs, 1, 1),
                block=(TWO_STAGE_PARTIAL_THREADS, 1, 1),
                stream=stream,
            )
            reduce_launcher = rmsnorm_bwd_dweight_reduce_kernel(DWeightPartial, DWeight)
            reduce_launcher.launch(
                grid=(reduce_grid, 1, 1),
                block=(DWEIGHT_REDUCE_THREADS, 1, 1),
                stream=stream,
            )

        return launch_fused_add_rmsnorm_bwd_two_stage

    @flyc.jit
    def launch_rmsnorm_bwd_two_stage(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        DY: fx.Tensor,
        Rstd: fx.Tensor,
        DX: fx.Tensor,
        DWeight: fx.Tensor,
        DWeightPartial: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # DResidualOut is compile-time dead in the plain variant; DY fills the
        # otherwise-unused tensor slot so both variants share one main kernel.
        partial_launcher = rmsnorm_bwd_partial_kernel(
            Input,
            Gamma,
            DY,
            DY,
            Rstd,
            DX,
            DWeightPartial,
            m_in,
        )
        partial_launcher.launch(
            grid=(num_programs, 1, 1),
            block=(TWO_STAGE_PARTIAL_THREADS, 1, 1),
            stream=stream,
        )
        reduce_launcher = rmsnorm_bwd_dweight_reduce_kernel(DWeightPartial, DWeight)
        reduce_launcher.launch(
            grid=(reduce_grid, 1, 1),
            block=(DWEIGHT_REDUCE_THREADS, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm_bwd_two_stage


def build_rmsnorm_bwd_two_stage_module(N: int, dtype_str: str, num_programs: int):
    """Large-M plain RMSNorm backward without global dweight atomics."""
    return _build_rmsnorm_bwd_two_stage_module(N, dtype_str, num_programs, fused_add=False)


def build_fused_add_rmsnorm_bwd_two_stage_module(N: int, dtype_str: str, num_programs: int):
    """Large-M fused-add RMSNorm backward without global dweight atomics."""
    return _build_rmsnorm_bwd_two_stage_module(N, dtype_str, num_programs, fused_add=True)
