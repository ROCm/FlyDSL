# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Softmax backward kernel builder using the @flyc.kernel API.

dx_i = y_i * (dy_i - sum_j(dy_j * y_j))

One block per row, fp32 dot accumulation. Two passes: the first accumulates the
row dot product, the second applies the elementwise correction.

Unlike the forward kernel, backward needs *two* operands live across the block
reduction. They are register-buffered in their native storage dtype and widened
to fp32 only at use, so a 16-bit row costs the same registers as the forward's
single fp32 buffer. Rows wider than ``MAX_RESIDENT_COLS`` would spill, so those
re-load in the second pass instead.

Two paths:
  - Fast path (N % tile_cols == 0): 128-bit vectorised access.
  - Generic path (arbitrary N): scalar access with masking.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import ReductionOp, full
from kernels.common.kernels_common import dtype_to_elem_type, get_warp_size

KERNEL_NAME = "softmax_bwd_kernel"

BLOCK_THREADS = 256
WARP_SIZE = get_warp_size()

# Both DY and Y stay live across the block reduction, so a resident row costs
# twice the forward kernel's register budget. The binding limit measured on
# gfx950 is elements held per thread per tensor (N / BLOCK_THREADS), not tiles
# or raw VGPRs: 64 elements is fine for every dtype, 128 spills to scratch and
# costs roughly half the achieved bandwidth. Beyond the cap the second pass
# re-loads from memory instead, which the LLC largely absorbs.
#
# Measured on an idle MI355X (gfx950); 3*M*N*elem_bytes traffic model. At 64
# elements/thread both dtypes land at ~160 VGPRs with zero spill; 128 would need
# roughly twice that against a 256 limit.
#   bf16 N=8192  (32 elem/thread):  78 VGPR, 0 spill
#   bf16 N=16384 (64 elem/thread): 154 VGPR, 0 spill
#   f32  N=16384 (64 elem/thread): 164 VGPR, 0 spill
# Tier choice at each width, best option last:
#   bf16 N=16384: Y only 5.20 TB/s  -> both     5.79 TB/s
#   bf16 N=32768: both   2.79 TB/s  -> neither 3.65 -> Y only 4.76 TB/s
#   f32  N=32768: both   4.19 TB/s  -> neither 3.41 -> Y only 4.72 TB/s
#   bf16 N=65536: Y only 2.39 TB/s  -> neither 3.08 TB/s
# So both-resident wins up to the cap, Y-only wins to twice the cap, and past
# that the spill cost exceeds the extra read.
MAX_RESIDENT_ELEMS_PER_THREAD = 64
MAX_RESIDENT_COLS = BLOCK_THREADS * MAX_RESIDENT_ELEMS_PER_THREAD


def softmax_bwd_buffered_operands(N: int, dtype_str: str) -> int:
    """How many of (DY, Y) stay register-resident across the block reduction.

    2 -> ideal 3-unit traffic; 1 -> DY re-read (4 units); 0 -> both re-read
    (5 units), which is also what the generic scalar path does.

    Exposed so the dispatch thresholds can be asserted without a GPU, mirroring
    ``is_rmsnorm_bwd_two_stage_vec_config`` in ``rmsnorm_bwd_kernel.py``.
    """
    vec_width = 128 // (32 if dtype_str == "f32" else 16)
    tile_cols = BLOCK_THREADS * vec_width
    if N < tile_cols or N % tile_cols != 0:
        return 0
    if N <= MAX_RESIDENT_COLS:
        return 2
    if N <= 2 * MAX_RESIDENT_COLS:
        return 1
    return 0


def build_softmax_bwd_module(N: int, dtype_str: str = "f32"):
    elem_bits = 32 if dtype_str == "f32" else 16
    # BufferCopy128b moves one 128-bit transaction per lane, so the register
    # vector width must satisfy vec_width * elem_bits == 128 (8 for 16-bit, 4 for f32).
    vec_width = 128 // elem_bits
    tile_cols = BLOCK_THREADS * vec_width
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)

    @fx.struct
    class SharedStorage:
        s_red: fx.Array[fx.Float32, RED_SLOTS, 16]

    @flyc.kernel
    def softmax_bwd_kernel(
        DY: fx.Tensor,
        Y: fx.Tensor,
        DX: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))

        c_zero_f = fx.Float32(0.0)

        # ── wave / block reduction (sum only) ─────────────────────────────
        def wave_reduce_add(x):
            w = x
            with fx.fastmath(fm_fast):
                for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                    off = WARP_SIZE // (2 << _sh_exp)
                    peer = gpu.shuffle_xor(w, off, WARP_SIZE)
                    w = w + peer
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

        DY_buf = fx.rocdl.make_buffer_tensor(DY)
        Y_buf = fx.rocdl.make_buffer_tensor(Y)
        DX_buf = fx.rocdl.make_buffer_tensor(DX)

        row_dy = fx.slice(DY_buf, (bid, None))
        row_y = fx.slice(Y_buf, (bid, None))
        row_dx = fx.slice(DX_buf, (bid, None))

        # ==================================================================
        # Fast path: N is a multiple of tile_cols
        # ==================================================================
        if const_expr(N >= tile_cols and N % tile_cols == 0):
            num_tiles = N // tile_cols
            # Three tiers by row width. Ideal traffic is 3 units (read Y, read
            # DY, write DX); each operand dropped from registers adds one more.
            #   both buffered  -> 3 units
            #   Y only         -> 4 units (DY re-read)
            #   neither        -> 5 units
            # Y alone at twice the cap costs the same registers as both operands
            # at the cap, so the middle tier is free in occupancy terms.
            buffer_dy = N <= MAX_RESIDENT_COLS
            buffer_y = N <= 2 * MAX_RESIDENT_COLS

            dy_div = fx.logical_divide(row_dy, fx.make_layout(vec_width, 1))
            y_div = fx.logical_divide(row_y, fx.make_layout(vec_width, 1))
            dx_div = fx.logical_divide(row_dx, fx.make_layout(vec_width, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            def _load_vec(div_tensor, idx):
                r = fx.make_rmem_tensor(vec_width, elem_dtype)
                fx.copy(copy_atom, fx.slice(div_tensor, (None, idx)), r)
                return fx.memref_load_vec(r)

            def _store_vec(val, div_tensor, idx):
                r = fx.make_rmem_tensor(vec_width, elem_dtype)
                fx.memref_store_vec(val, r)
                fx.copy(copy_atom, r, fx.slice(div_tensor, (None, idx)))

            # 1. Load both operands, accumulate the row dot product in fp32.
            #    Buffered in native dtype; widened only for the arithmetic.
            dy_buffer = []
            y_buffer = []
            thread_dot = c_zero_f

            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                dy_vec = _load_vec(dy_div, idx)
                y_vec = _load_vec(y_div, idx)
                if const_expr(buffer_dy):
                    dy_buffer.append(dy_vec)
                if const_expr(buffer_y):
                    y_buffer.append(y_vec)
                prod = dy_vec.to(fx.Float32) * y_vec.to(fx.Float32)
                thread_dot = thread_dot + prod.reduce(ReductionOp.ADD, fastmath=fm_fast)

            dot = block_reduce_add(thread_dot)

            # 2. Apply dx = y * (dy - dot)
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                if const_expr(buffer_dy):
                    dy_f = dy_buffer[tile_i].to(fx.Float32)
                else:
                    dy_f = _load_vec(dy_div, idx).to(fx.Float32)
                if const_expr(buffer_y):
                    y_f = y_buffer[tile_i].to(fx.Float32)
                else:
                    y_f = _load_vec(y_div, idx).to(fx.Float32)

                dx_vec = y_f * (dy_f - dot)
                out_e = dx_vec if dtype_str == "f32" else dx_vec.to(elem_dtype)
                _store_vec(out_e, dx_div, idx)

        else:
            # ==============================================================
            # Generic path: scalar for arbitrary N
            # ==============================================================
            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )

            dy_div = fx.logical_divide(row_dy, fx.make_layout(1, 1))
            y_div = fx.logical_divide(row_y, fx.make_layout(1, 1))
            dx_div = fx.logical_divide(row_dx, fx.make_layout(1, 1))

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
                fx.copy(copy_atom_s, r, view)

            # 1. Load + dot. Out-of-range lanes contribute the additive
            #    identity so the reduction stays correct on the tail.
            row_buffer = []
            thread_dot = c_zero_f

            for base in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)

                dy_e = _load_scalar(dy_div, idx_safe)
                y_e = _load_scalar(y_div, idx_safe)
                dy_f = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
                y_f = y_e if dtype_str == "f32" else y_e.to(fx.Float32)

                row_buffer.append((dy_f, y_f))
                thread_dot = thread_dot + is_valid.select(dy_f * y_f, c_zero_f)

            dot = block_reduce_add(thread_dot)

            # 2. Apply dx = y * (dy - dot)
            buf_idx = 0
            for base in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base
                dy_f, y_f = row_buffer[buf_idx]
                buf_idx += 1
                if idx < N:
                    dx_val = y_f * (dy_f - dot)
                    out_e = dx_val if dtype_str == "f32" else dx_val.to(elem_dtype)
                    _store_scalar(dx_div, idx, out_e)

    @flyc.jit
    def launch_softmax_bwd(
        DY: fx.Tensor,
        Y: fx.Tensor,
        DX: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = softmax_bwd_kernel(DY, Y, DX)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_softmax_bwd
