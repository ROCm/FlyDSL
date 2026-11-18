# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Softmax kernel builder using the @flyc.kernel API.

softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))

Uses exp2(x * log2e) for fast exponentiation.
The kernel register-buffers the row across max, exp+sum, and normalize passes.

Two paths:
  - Fast path (N % tile_cols == 0): buffer_load/store vectorised access.
  - Generic path (arbitrary N): scalar fx.copy with masking.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, full
from kernels.common.kernels_common import LOG2E, dtype_to_elem_type, get_warp_size
from kernels.norm.softmax_common import _load_vec, _store_vec

KERNEL_NAME = "softmax_kernel"

BLOCK_THREADS = 256
WARP_SIZE = get_warp_size()

# Bumped whenever a kernel or search-space change can change the tuned winner. The
# scratch cache does not fingerprint kernel source, so this is what forces a retune.
TUNING_SCHEMA = 6


def build_softmax_module(
    M: int,
    N: int,
    dtype_str: str = "f32",
    BLOCK_THREADS: int = BLOCK_THREADS,
    THREADS_PER_ROW: int | None = None,
    ROWS_PER_BLOCK: int = 1,
):
    """Build a Softmax launcher. ``M`` is vestigial: the row count is a runtime grid
    value, so it must not specialize the kernel."""
    THREADS_PER_ROW = BLOCK_THREADS if THREADS_PER_ROW is None else THREADS_PER_ROW
    if BLOCK_THREADS != THREADS_PER_ROW * ROWS_PER_BLOCK:
        raise ValueError(
            "BLOCK_THREADS must equal THREADS_PER_ROW * ROWS_PER_BLOCK, got "
            f"{BLOCK_THREADS} != {THREADS_PER_ROW} * {ROWS_PER_BLOCK}"
        )
    if THREADS_PER_ROW <= 0 or THREADS_PER_ROW & (THREADS_PER_ROW - 1):
        raise ValueError(f"THREADS_PER_ROW must be a positive power of two, got {THREADS_PER_ROW}")
    if THREADS_PER_ROW > WARP_SIZE and ROWS_PER_BLOCK != 1:
        raise ValueError("multi-row blocks require THREADS_PER_ROW <= WARP_SIZE")
    elem_bits = 32 if dtype_str == "f32" else 16
    # BufferCopy128b moves one 128-bit transaction per lane, so the register
    # vector width must satisfy vec_width * elem_bits == 128 (8 for 16-bit, 4 for f32).
    vec_width = 128 // elem_bits
    tile_cols = THREADS_PER_ROW * vec_width
    BlockReduce = fx.coop.BlockReduce[fx.Float32, BLOCK_THREADS]
    SharedStorage = BlockReduce.SharedStorage

    # No explicit known_block_size: launch_softmax passes a static block dim, so the
    # compiler infers it and max_flat_workgroup_size tracks BLOCK_THREADS above 256.
    # tests/kernels/test_softmax_autotune.py pins that.
    @flyc.kernel
    def softmax_kernel(
        A: fx.Tensor,
        _Pad0: fx.Tensor,
        _Pad1: fx.Tensor,
        C: fx.Tensor,
        MIn: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        lane = tid
        row_safe = bid
        row_valid = bid < MIn
        if const_expr(ROWS_PER_BLOCK > 1):
            lane = tid % THREADS_PER_ROW
            row_local = tid // THREADS_PER_ROW
            row = bid * ROWS_PER_BLOCK + row_local
            row_valid = row < MIn
            row_safe = row_valid.select(row, 0)

        elem_dtype = dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        c_zero_f = fx.Float32(0.0)
        c_neg_inf = fx.Float32(float("-inf"))

        def block_reduce(val, op):
            if const_expr(THREADS_PER_ROW <= WARP_SIZE):
                with fx.fastmath(fm_fast if op == ReductionOp.ADD else arith.FastMathFlags.none):
                    return fx.coop.warp_reduce(val, op, width=THREADS_PER_ROW)
            # Order the preceding reduction's readers before scratch reuse.
            gpu.barrier()
            with fx.fastmath(fm_fast if op == ReductionOp.ADD else arith.FastMathFlags.none):
                return BlockReduce(val, op, storage=lds)

        # ==================================================================
        # Fast path: N is a multiple of tile_cols
        # ==================================================================
        if const_expr(N >= tile_cols and N % tile_cols == 0):
            num_tiles = N // tile_cols
            # ── Layout API: buffer-backed tensors + tiled access ─────
            A_buf = fx.rocdl.make_buffer_tensor(A)
            C_buf = fx.rocdl.make_buffer_tensor(C)

            row_a = fx.slice(A_buf, (row_safe, None))
            row_c = fx.slice(C_buf, (row_safe, None))

            # 1. Load + compute local max
            row_buffer = []
            thread_max = c_neg_inf

            for tile_i in range_constexpr(num_tiles):
                idx = lane + tile_i * THREADS_PER_ROW
                vec = _load_vec(row_a, idx, vec_width, dtype_str)
                x = vec.to(fx.Float32)
                row_buffer.append(x)
                red_max = x.reduce(ReductionOp.MAX)
                thread_max = fx.max(thread_max, red_max)

            global_max = block_reduce(thread_max, ReductionOp.MAX)

            # 2. Exp + local sum
            thread_sum = c_zero_f

            for i in range_constexpr(num_tiles):
                x = row_buffer[i]
                scaled = (x - global_max) * LOG2E
                exp_val = fmath.exp2(scaled, fastmath=fm_fast)
                row_buffer[i] = exp_val
                red_sum = exp_val.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sum = thread_sum + red_sum

            global_sum = block_reduce(thread_sum, ReductionOp.ADD)

            # 3. Normalize + store
            inv_sum = 1.0 / global_sum

            for tile_i in range_constexpr(num_tiles):
                norm_vec = row_buffer[tile_i] * inv_sum
                out_e = norm_vec if dtype_str == "f32" else norm_vec.to(elem_dtype)

                out_idx = lane + tile_i * THREADS_PER_ROW
                if const_expr(ROWS_PER_BLOCK == 1):
                    _store_vec(row_c, out_idx, out_e, vec_width, dtype_str)
                else:
                    if row_valid:
                        _store_vec(row_c, out_idx, out_e, vec_width, dtype_str)

        else:
            # ==============================================================
            # Generic path: scalar for arbitrary N
            # ==============================================================
            A_buf = fx.rocdl.make_buffer_tensor(A)
            C_buf = fx.rocdl.make_buffer_tensor(C)

            row_a = fx.slice(A_buf, (row_safe, None))
            row_c = fx.slice(C_buf, (row_safe, None))

            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )
            store_atom_s = fx.make_copy_atom(
                (fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b()),
                elem_bits,
            )

            a_div = fx.logical_divide(row_a, fx.make_layout(1, 1))
            c_div = fx.logical_divide(row_c, fx.make_layout(1, 1))

            def _load_scalar(divided, index):
                view = fx.slice(divided, (None, index))
                r = fx.make_rmem_tensor(1, elem_dtype)
                fx.copy(copy_atom_s, view, r)
                return fx.memref_load_vec(r)[0]

            def _store_scalar(divided, index, val):
                r = fx.make_rmem_tensor(1, elem_dtype)
                ts = full(1, elem_dtype(val), elem_dtype)
                fx.memref_store_vec(ts, r)
                view = fx.slice(divided, (None, index))
                if const_expr(ROWS_PER_BLOCK == 1):
                    fx.copy(store_atom_s, r, view)
                else:
                    if row_valid:
                        fx.copy(store_atom_s, r, view)

            # 1. Load + max
            row_buffer = []
            thread_max = c_neg_inf

            for base in range_constexpr(0, N, THREADS_PER_ROW):
                idx = lane + base
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                val_e = _load_scalar(a_div, idx_safe)
                val = val_e if dtype_str == "f32" else val_e.to(fx.Float32)
                safe_val = is_valid.select(val, c_neg_inf)
                row_buffer.append((safe_val, is_valid))
                thread_max = fx.max(thread_max, safe_val)

            global_max = block_reduce(thread_max, ReductionOp.MAX)

            # 2. Exp + sum
            thread_sum = c_zero_f
            new_buffer = []
            for safe_val, is_valid in row_buffer:
                sub = safe_val - global_max
                scaled = sub * LOG2E
                exp_val = scaled.exp2(fastmath=fm_fast)
                safe_exp = is_valid.select(exp_val, c_zero_f)
                thread_sum = thread_sum + safe_exp
                new_buffer.append((exp_val, is_valid))

            global_sum = block_reduce(thread_sum, ReductionOp.ADD)
            inv_sum = 1.0 / global_sum

            # 3. Normalize + store
            buf_idx = 0
            for base in range_constexpr(0, N, THREADS_PER_ROW):
                idx = lane + base
                exp_val, is_valid = new_buffer[buf_idx]
                buf_idx += 1
                if idx < N:
                    norm_val = fx.Float32(exp_val) * inv_sum
                    out_e = norm_val
                    if const_expr(dtype_str == "f32"):
                        out_e = norm_val
                    else:
                        out_e = norm_val.to(elem_dtype)
                    _store_scalar(c_div, idx, out_e)

    @flyc.jit
    def launch_softmax(
        A: fx.Tensor,
        C: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = softmax_kernel(A, C, C, C, m_in)
        launcher.launch(
            grid=((m_in + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_softmax


@flyc.jit
def softmax_direct(
    A: fx.Tensor,
    C: fx.Tensor,
    m_in: fx.Int32,
    N: fx.Constexpr[int],
    dtype_str: fx.Constexpr[str],
    BLOCK_THREADS: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
    THREADS_PER_ROW: fx.Constexpr[int] = 0,
    ROWS_PER_BLOCK: fx.Constexpr[int] = 1,
):
    """Specialize the existing Softmax factory through JIT Constexpr inputs.

    ``tuning_schema`` is not read by the kernel. It is a declared autotune key axis, so
    bumping it partitions the winner cache and forces a fresh search.
    """
    resolved_threads_per_row = BLOCK_THREADS if THREADS_PER_ROW == 0 else THREADS_PER_ROW
    launch = build_softmax_module(
        0,
        N,
        dtype_str,
        BLOCK_THREADS=BLOCK_THREADS,
        THREADS_PER_ROW=resolved_threads_per_row,
        ROWS_PER_BLOCK=ROWS_PER_BLOCK,
    )
    launch(A, C, m_in, stream)
