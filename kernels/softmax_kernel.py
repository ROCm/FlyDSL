# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Softmax kernel builder using the @flyc.kernel API.

softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))

Uses exp2(x * log2e) for fast exponentiation.
Register-buffers the entire row across three passes: max, exp+sum, normalize.

Two paths:
  - Fast path (N % tile_cols == 0): buffer_load/store vectorised access.
  - Generic path (arbitrary N): scalar copy_atom_call with masking.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T, Int32
from flydsl.expr.vector import ReductionOp, full
from flydsl.expr.numeric import Numeric, Float32

from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from flydsl._mlir import ir


KERNEL_NAME = "softmax_kernel"

import math
from kernels.kernels_common import dtype_to_elem_type, get_warp_size

BLOCK_THREADS = 256
WARP_SIZE = get_warp_size()
VEC_WIDTH = 8


def build_softmax_module(M: int, N: int, dtype_str: str = "f32"):
    arch = get_hip_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    tile_cols = BLOCK_THREADS * VEC_WIDTH
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16

    allocator = SmemAllocator(None, arch=arch)
    f32_bytes = 4
    red_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = red_offset + RED_SLOTS * f32_bytes

    @flyc.kernel
    def softmax_kernel(
        A: fx.Tensor,
        _Pad0: fx.Tensor,
        _Pad1: fx.Tensor,
        C: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32

        fm_fast = arith.FastMathFlags.fast

        base_ptr = allocator.get_base()
        s_red = SmemPtr(base_ptr, red_offset, T.f32, shape=(RED_SLOTS,))
        s_red.get()

        c_zero_f = arith.constant(0.0, type=compute_type)
        c_neg_inf = arith.constant(float("-inf"), type=compute_type)
        c_log2e = arith.constant(1.4426950408889634, type=compute_type)

        # ── wave / block reduction (supports max and sum) ─────────────────
        def wave_reduce(x, mode):
            width_i32 = fx.Int32(WARP_SIZE)
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = fx.Int32(WARP_SIZE // (2 << _sh_exp))
                peer = w.shuffle_xor(off, width_i32)
                if const_expr(mode == "max"):
                    w = w.maximumf(peer)
                else:
                    w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce(val, mode, s_red_buffer):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce(val, mode)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            neutral = c_neg_inf if mode == "max" else c_zero_f

            w = wave_reduce(val, mode)

            if lane == fx.Int32(0):
                wave_idx = ArithValue(wave).index_cast(T.index)
                SmemPtr.store(s_red_buffer, w, [wave_idx])
            gpu.barrier()

            if wave == fx.Int32(0):
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, fx.Int32(0))
                lane_safe_idx = ArithValue(lane_safe).index_cast(T.index)
                v = SmemPtr.load(s_red_buffer, [lane_safe_idx])
                z = neutral
                ww = in_range.select(v, z)
                ww = wave_reduce(ww, mode)

                if lane == fx.Int32(0):
                    c0_idx = fx.Index(0)
                    SmemPtr.store(s_red_buffer, ww, [c0_idx])
            gpu.barrier()

            c0_idx = fx.Index(0)
            return SmemPtr.load(s_red_buffer, [c0_idx])

        # ==================================================================
        # Fast path: N is a multiple of tile_cols
        # ==================================================================
        if const_expr(False and N >= tile_cols and N % tile_cols == 0):
            from flydsl.expr import math as fmath

            num_tiles = N // tile_cols
            elem_dtype = Numeric.from_ir_type(elem_type)

            # ── Layout API: buffer-backed tensors + tiled access ─────
            A_buf = fx.rocdl.make_buffer_tensor(A)
            C_buf = fx.rocdl.make_buffer_tensor(C)

            row_a = fx.slice(A_buf, (bid, None))
            row_c = fx.slice(C_buf, (bid, None))

            a_div = fx.logical_divide(row_a, fx.make_layout(VEC_WIDTH, 1))
            c_div = fx.logical_divide(row_c, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            vec_reg_ty = fx.MemRefType.get(
                elem_type, fx.LayoutType.get(VEC_WIDTH, 1), fx.AddressSpace.Register
            )
            vec_reg_lay = fx.make_layout(VEC_WIDTH, 1)

            def _load_vec(div_tensor, idx):
                r = fx.memref_alloca(vec_reg_ty, vec_reg_lay)
                fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, idx)), r)
                return fx.memref_load_vec(r)

            def _store_vec(val, div_tensor, idx):
                r = fx.memref_alloca(vec_reg_ty, vec_reg_lay)
                fx.memref_store_vec(val, r)
                fx.copy_atom_call(copy_atom, r, fx.slice(div_tensor, (None, idx)))

            # 1. Load + compute local max
            row_buffer = []
            thread_max = c_neg_inf

            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                vec = _load_vec(a_div, idx)
                x = vec.to(Float32)
                row_buffer.append(x)
                red_max = x.reduce(ReductionOp.MAX)
                thread_max = thread_max.maximumf(red_max)

            global_max = block_reduce(thread_max, "max", s_red)

            # 2. Exp + local sum
            thread_sum = c_zero_f

            for i in range_constexpr(num_tiles):
                x = row_buffer[i]
                scaled = (x - global_max) * c_log2e
                exp_val = fmath.exp2(scaled, fastmath=True)
                row_buffer[i] = exp_val
                red_sum = exp_val.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sum = thread_sum + red_sum

            global_sum = block_reduce(thread_sum, "sum", s_red)

            # 3. Normalize + store
            c_one = arith.constant(1.0, type=compute_type)
            inv_sum = c_one / ArithValue(global_sum)

            for tile_i in range_constexpr(num_tiles):
                norm_vec = row_buffer[tile_i] * inv_sum
                out_e = norm_vec if dtype_str == "f32" else norm_vec.to(elem_dtype)

                out_idx = tid + tile_i * BLOCK_THREADS
                _store_vec(out_e, c_div, out_idx)

        else:
            # ==============================================================
            # Generic path: scalar for arbitrary N
            # ==============================================================
            elem_dtype = Numeric.from_ir_type(elem_type)

            A_buf = fx.rocdl.make_buffer_tensor(A)
            C_buf = fx.rocdl.make_buffer_tensor(C)

            row_a = fx.slice(A_buf, (bid, None))
            row_c = fx.slice(C_buf, (bid, None))

            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )
            scalar_reg_ty = fx.MemRefType.get(elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
            scalar_reg_lay = fx.make_layout(1, 1)

            a_div = fx.logical_divide(row_a, fx.make_layout(1, 1))
            c_div = fx.logical_divide(row_c, fx.make_layout(1, 1))

            def _load_scalar(divided, index):
                view = fx.slice(divided, (None, index))
                r = fx.memref_alloca(scalar_reg_ty, scalar_reg_lay)
                fx.copy_atom_call(copy_atom_s, view, r)
                return fx.memref_load_vec(r)[0].ir_value()

            def _store_scalar(divided, index, val):
                r = fx.memref_alloca(scalar_reg_ty, scalar_reg_lay)
                ts = full(1, elem_dtype(val), elem_dtype)
                fx.memref_store_vec(ts, r)
                view = fx.slice(divided, (None, index))
                fx.copy_atom_call(copy_atom_s, r, view)

            # 1. Load + max
            row_buffer = []
            thread_max = c_neg_inf

            for base in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base
                c_N = Int32(N)
                is_valid = idx < c_N
                idx_safe = is_valid.select(idx, Int32(0))
                val_e = _load_scalar(a_div, idx_safe)
                val = val_e if dtype_str == "f32" else val_e.extf(compute_type)
                safe_val = is_valid.select(val, c_neg_inf)
                row_buffer.append((safe_val, is_valid))
                thread_max = thread_max.maximumf(safe_val)

            global_max = block_reduce(thread_max, "max", s_red)

            # 2. Exp + sum
            thread_sum = c_zero_f
            new_buffer = []
            for safe_val, is_valid in row_buffer:
                sub = safe_val - ArithValue(global_max)
                scaled = sub * c_log2e
                exp_val = scaled.exp2(fastmath=fm_fast)
                safe_exp = is_valid.select(exp_val, c_zero_f)
                thread_sum = thread_sum + safe_exp
                new_buffer.append((exp_val, is_valid))

            global_sum = block_reduce(thread_sum, "sum", s_red)
            c_one = arith.constant(1.0, type=compute_type)
            inv_sum = c_one / ArithValue(global_sum)

            # 3. Normalize + store
            buf_idx = 0
            for base in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base
                exp_val, is_valid = new_buffer[buf_idx]
                buf_idx += 1
                if arith.cmpi(arith.CmpIPredicate.ult, idx, Int32(N)):
                    norm_val = ArithValue(exp_val) * inv_sum
                    out_e = norm_val
                    if const_expr(dtype_str == "f32"):
                        out_e = norm_val
                    else:
                        out_e = norm_val.truncf(elem_type)
                    _store_scalar(c_div, idx, out_e)

    @flyc.jit
    def launch_softmax(
        A: fx.Tensor,
        C: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        idx_m = ArithValue(m_in).index_cast(T.index)
        launcher = softmax_kernel(A, C, C, C)
        launcher.launch(
            grid=(idx_m, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_softmax
