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

from flydsl.expr import arith, vector, gpu, range_constexpr
from flydsl.expr.typing import T, Int32

from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from flydsl._mlir import ir
from flydsl.expr import buffer_ops


KERNEL_NAME = "softmax_kernel"

BLOCK_THREADS = 256
WARP_SIZE = 64
VEC_WIDTH = 8


def dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f32":
        return T.f32
    if dtype_str == "f16":
        return T.f16
    if dtype_str == "bf16":
        return T.bf16
    raise ValueError(f"unsupported dtype: {dtype_str}")


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
            for sh in [32, 16, 8, 4, 2, 1]:
                off = fx.Int32(sh)
                peer = w.shuffle_xor(off, width_i32)
                if mode == "max":
                    w = w.maximumf(peer)
                else:
                    w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce(val, mode):
            if RED_SLOTS == 1:
                return wave_reduce(val, mode)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            neutral = c_neg_inf if mode == "max" else c_zero_f

            w = wave_reduce(val, mode)

            if lane == fx.Int32(0):
                wave_idx = arith.index_cast(T.index, wave)
                s_red.store(w, [wave_idx])
            gpu.barrier()

            if wave == fx.Int32(0):
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, fx.Int32(0))
                lane_safe_idx = arith.index_cast(T.index, lane_safe)
                v = s_red.load([lane_safe_idx])
                z = neutral
                ww = in_range.select(v, z)
                ww = wave_reduce(ww, mode)

                if lane == fx.Int32(0):
                    c0_idx = fx.Index(0)
                    s_red.store(ww, [c0_idx])
            gpu.barrier()

            c0_idx = fx.Index(0)
            return s_red.load([c0_idx])

        # ==================================================================
        # Fast path: N is a multiple of tile_cols
        # ==================================================================
        if False and N >= tile_cols and N % tile_cols == 0:
            from flydsl.expr.arith import ArithValue

            num_tiles = N // tile_cols
            elem_bytes = 4 if dtype_str == "f32" else 2
            vec_dwords = (VEC_WIDTH * elem_bytes) // 4

            vec_type_c = T.vec(VEC_WIDTH, compute_type)
            vec_type_e = T.vec(VEC_WIDTH, elem_type)

            a_rsrc = buffer_ops.create_buffer_resource(A, max_size=True)
            c_rsrc = buffer_ops.create_buffer_resource(C, max_size=True)

            row_soffset = ArithValue(bid) * (N * elem_bytes)
            thr_col_bytes = ArithValue(tid) * (VEC_WIDTH * elem_bytes)

            def _load_vec(rsrc, col_byte_off, soff=None):
                dw = col_byte_off >> fx.Int32(2)
                raw = buffer_ops.buffer_load(rsrc, dw, vec_width=vec_dwords, dtype=T.i32, soffset_bytes=soff)
                if vec_dwords == VEC_WIDTH:
                    return raw.bitcast(vec_type_e)
                return vector.bitcast(vec_type_e, raw)

            def _store_vec(data, rsrc, col_byte_off, soff=None):
                dw = col_byte_off >> fx.Int32(2)
                buffer_ops.buffer_store(data, rsrc, dw, soffset_bytes=soff)

            # 1. Load + compute local max
            row_buffer = []
            thread_max = c_neg_inf

            for tile_i in range_constexpr(num_tiles):
                col_bytes = thr_col_bytes + (tile_i * tile_cols * elem_bytes)
                vec_e = _load_vec(a_rsrc, col_bytes, soff=row_soffset)
                x = vec_e if dtype_str == "f32" else vec_e.extf(vec_type_c)
                row_buffer.append(x)
                red_max = vector.reduction(compute_type, vector.CombiningKind.MAXNUMF, x)
                thread_max = thread_max.maximumf(red_max)

            global_max = block_reduce(thread_max, "max")

            # 2. Exp + local sum
            g_max_splat = vector.broadcast(vec_type_c, global_max)
            log2e_splat = vector.broadcast(vec_type_c, c_log2e)
            thread_sum = c_zero_f

            for i in range_constexpr(num_tiles):
                x = row_buffer[i]
                sub = ArithValue(x) - ArithValue(g_max_splat)
                scaled = sub * ArithValue(log2e_splat)
                exp_val = scaled.exp2(fastmath=fm_fast)
                row_buffer[i] = exp_val
                red_sum = vector.reduction(compute_type, vector.CombiningKind.ADD, exp_val, fastmath=fm_fast)
                thread_sum = thread_sum + red_sum

            global_sum = block_reduce(thread_sum, "sum")

            # 3. Normalize + store
            c_one = arith.constant(1.0, type=compute_type)
            inv_sum = c_one / ArithValue(global_sum)
            inv_sum_splat = vector.broadcast(vec_type_c, inv_sum)

            for tile_i in range_constexpr(num_tiles):
                exp_vec = row_buffer[tile_i]
                norm_vec = ArithValue(exp_vec) * ArithValue(inv_sum_splat)

                if dtype_str == "f32":
                    out_e = norm_vec
                else:
                    out_e = norm_vec.truncf(vec_type_e)

                out_col = thr_col_bytes + (tile_i * tile_cols * elem_bytes)
                i32_vec_ty = T.vec(vec_dwords, T.i32)
                out_i32 = vector.bitcast(i32_vec_ty, out_e) if vec_dwords != VEC_WIDTH else out_e.bitcast(i32_vec_ty)
                _store_vec(out_i32, c_rsrc, out_col, soff=row_soffset)

        else:
            # ==============================================================
            # Generic path: scalar for arbitrary N
            # ==============================================================
            from flydsl.expr.arith import ArithValue

            row_a = fx.slice(A, (bid, None))
            row_c = fx.slice(C, (bid, None))

            copy_atom_s = fx.make_copy_atom(fx.UniversalCopy(elem_bits), elem_bits)
            scalar_reg_ty = fx.MemRefType.get(elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
            scalar_reg_lay = fx.make_layout(1, 1)

            a_div = fx.logical_divide(row_a, fx.make_layout(1, 1))
            c_div = fx.logical_divide(row_c, fx.make_layout(1, 1))

            def _load_scalar(divided, index):
                view = fx.slice(divided, (None, index))
                r = fx.memref_alloca(scalar_reg_ty, scalar_reg_lay)
                fx.copy_atom_call(copy_atom_s, view, r)
                v = fx.memref_load_vec(r)
                return vector.extract(v, static_position=[0])

            def _store_scalar(divided, index, val):
                r = fx.memref_alloca(scalar_reg_ty, scalar_reg_lay)
                vec_ty = T.vec(1, elem_type)
                v = vector.from_elements(vec_ty, [val])
                fx.memref_store_vec(v, r)
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

            global_max = block_reduce(thread_max, "max")

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

            global_sum = block_reduce(thread_sum, "sum")
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
                    if dtype_str == "f32":
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

        idx_m = arith.index_cast(T.index, m_in)
        launcher = softmax_kernel(A, C, C, C)
        launcher.launch(
            grid=(idx_m, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_softmax
