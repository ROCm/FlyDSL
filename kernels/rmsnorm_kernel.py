# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""RMSNorm kernel builder using the @flyc.kernel API.

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma

Two paths:
  - Fast path (N % tile_cols == 0): buffer_load/store vectorised access.
  - Generic path (arbitrary N): scalar copy_atom_call.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import arith, vector, gpu, range_constexpr, buffer_ops
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T, Int32
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from flydsl._mlir import ir


KERNEL_NAME = "rmsnorm"

EPS = 1e-5

import math
from kernels.kernels_common import dtype_to_elem_type, get_warp_size

BLOCK_THREADS = 256
WARP_SIZE = get_warp_size()
VEC_WIDTH = 8


def _quant_dtype_to_elem_type(dtype_str: str):
    if dtype_str in ("i8", "int8"):
        return T.i8
    raise ValueError(f"unsupported quant dtype: {dtype_str!r} (expected 'i8' or 'int8')")


def _quant_dtype_max(dtype_str: str) -> float:
    if dtype_str in ("i8", "int8"):
        return 127.0
    raise ValueError(f"unsupported quant dtype: {dtype_str!r} (expected 'i8' or 'int8')")


def build_rmsnorm_module(M: int, N: int, dtype_str: str):
    arch = get_hip_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    tile_cols = BLOCK_THREADS * VEC_WIDTH
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16

    allocator = SmemAllocator(None, arch=arch)
    f32_bytes = 4
    red_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = red_offset + RED_SLOTS * f32_bytes
    red2_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = red2_offset + RED_SLOTS * f32_bytes

    @flyc.kernel
    def rmsnorm_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        _Unused: fx.Tensor,
        Output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32

        fm_fast = arith.FastMathFlags.fast
        eps_c = arith.constant(EPS, type=compute_type)
        n_float = arith.constant(float(N), type=compute_type)

        base_ptr = allocator.get_base()
        s_red = SmemPtr(base_ptr, red_offset, T.f32, shape=(RED_SLOTS,))
        s_red2 = SmemPtr(base_ptr, red2_offset, T.f32, shape=(RED_SLOTS,))
        s_red.get()
        s_red2.get()

        def wave_reduce_add(x):
            width_i32 = fx.Int32(WARP_SIZE)
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = fx.Int32(WARP_SIZE // (2 << _sh_exp))
                peer = w.shuffle_xor(off, width_i32)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add(val):
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

        def block_reduce_add2(val0, val1):
            if RED_SLOTS == 1:
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == fx.Int32(0):
                wave_idx = ArithValue(wave).index_cast(T.index)
                s_red.store(w0, [wave_idx])
                s_red2.store(w1, [wave_idx])
            gpu.barrier()

            if wave == fx.Int32(0):
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, fx.Int32(0))
                lane_safe_idx = ArithValue(lane_safe).index_cast(T.index)
                v0 = s_red.load([lane_safe_idx])
                v1 = s_red2.load([lane_safe_idx])
                z = fx.Float32(0.0)
                ww0 = in_range.select(v0, z)
                ww1 = in_range.select(v1, z)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == fx.Int32(0):
                    c0_idx = fx.Index(0)
                    s_red.store(ww0, [c0_idx])
                    s_red2.store(ww1, [c0_idx])
            gpu.barrier()

            c0_idx = fx.Index(0)
            return s_red.load([c0_idx]), s_red2.load([c0_idx])

        # ==================================================================
        # Fast path: N is a multiple of tile_cols
        # ==================================================================
        if N >= tile_cols and N % tile_cols == 0 and elem_bits <= 16:
            from flydsl.expr.arith import ArithValue

            num_tiles = N // tile_cols

            vec_type_c = T.vec(VEC_WIDTH, compute_type)
            vec_type_e = T.vec(VEC_WIDTH, elem_type)

            # ── Layout API: buffer-backed tensors + tiled access ─────
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            vec_reg_ty = fx.MemRefType.get(
                elem_type, fx.LayoutType.get(VEC_WIDTH, 1), fx.AddressSpace.Register
            )
            vec_reg_lay = fx.make_layout(VEC_WIDTH, 1)

            def _load_vec(div_tensor, idx):
                r = fx.memref_alloca(vec_reg_ty, vec_reg_lay)
                fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, idx)), r)
                return ArithValue(fx.memref_load_vec(r))

            def _store_vec(val, div_tensor, idx):
                r = fx.memref_alloca(vec_reg_ty, vec_reg_lay)
                fx.memref_store_vec(val, r)
                fx.copy_atom_call(copy_atom, r, fx.slice(div_tensor, (None, idx)))

            c_zero_f = arith.constant(0.0, type=compute_type)
            thread_sumsq = c_zero_f
            thread_dummy = c_zero_f
            cache_as_elem = (dtype_str != "f32")
            in_local = []

            # Pass 1: load + cache + sumsq
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                vec_e = _load_vec(in_div, idx)

                if cache_as_elem:
                    in_local.append(vec_e)
                    x = vec_e.extf(vec_type_c)
                else:
                    x = vec_e
                    in_local.append(x)

                x_av = ArithValue(x)
                x2 = x_av * x_av
                red2 = vector.reduction(compute_type, vector.CombiningKind.ADD, x2, fastmath=fm_fast)
                thread_sumsq = ArithValue(thread_sumsq) + red2

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
            mean_sq = ArithValue(sum_sq) / n_float
            ms_eps = ArithValue(mean_sq) + eps_c
            rrms = ms_eps.rsqrt(fastmath=fm_fast)
            rrms_splat = vector.broadcast(vec_type_c, rrms)
            rrms_splat_av = ArithValue(rrms_splat)

            # Pass 2: normalize + gamma + store (reuse cached input)
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS

                g_e = _load_vec(gamma_div, idx)
                g = g_e if dtype_str == "f32" else g_e.extf(vec_type_c)

                x = in_local[tile_i]
                if cache_as_elem:
                    x = x.extf(vec_type_c)

                x_av = ArithValue(x)
                g_av = ArithValue(g)
                y = (x_av * rrms_splat_av) * g_av
                y_val = y

                if dtype_str == "bf16":
                    if USE_HW_CVT_PK_BF16_F32:
                        out_e = y_val.truncf(vec_type_e)
                    else:
                        vec_i32_ty = T.vec(VEC_WIDTH, T.i32)
                        vec4_i32_ty = T.vec(VEC_WIDTH // 2, T.i32)
                        vec_bf16_ty = T.vec(VEC_WIDTH, elem_type)
                        c16_i32 = arith.constant(16, type=T.i32)
                        c16_v = vector.broadcast(vec_i32_ty, c16_i32)
                        u = y_val.bitcast(vec_i32_ty)
                        upper = u.shrui(c16_v)
                        c1_v = vector.broadcast(vec_i32_ty, arith.constant(1, type=T.i32))
                        lsb = upper & c1_v
                        c7fff_v = vector.broadcast(vec_i32_ty, arith.constant(0x7FFF, type=T.i32))
                        bias = ArithValue(c7fff_v) + ArithValue(lsb)
                        u_round = ArithValue(u) + bias
                        bf16_bits = u_round.shrui(c16_v)
                        even = vector.shuffle(bf16_bits, bf16_bits, [0, 2, 4, 6])
                        odd = vector.shuffle(bf16_bits, bf16_bits, [1, 3, 5, 7])
                        odd_sh = odd << vector.broadcast(vec4_i32_ty, c16_i32)
                        packed = even | odd_sh
                        out_e = vector.bitcast(vec_bf16_ty, packed)
                elif dtype_str == "f32":
                    out_e = y_val
                else:
                    out_e = y_val.truncf(vec_type_e)

                out_idx = tid + tile_i * BLOCK_THREADS
                _store_vec(out_e, out_div, out_idx)

        else:
            # ==============================================================
            # Generic path: scalar 2-pass for arbitrary N
            # ==============================================================
            from flydsl.expr.arith import ArithValue

            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )
            scalar_reg_ty = fx.MemRefType.get(elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
            scalar_reg_lay = fx.make_layout(1, 1)

            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))

            def _load_scalar(divided_tensor, index):
                view = fx.slice(divided_tensor, (None, index))
                r = fx.memref_alloca(scalar_reg_ty, scalar_reg_lay)
                fx.copy_atom_call(copy_atom_s, view, r)
                v = fx.memref_load_vec(r)
                return vector.extract(v, static_position=[0])

            def _store_scalar(divided_tensor, index, val):
                r = fx.memref_alloca(scalar_reg_ty, scalar_reg_lay)
                vec_ty = T.vec(1, elem_type)
                v = vector.from_elements(vec_ty, [val])
                fx.memref_store_vec(v, r)
                view = fx.slice(divided_tensor, (None, index))
                fx.copy_atom_call(copy_atom_s, r, view)

            c_zero_f = arith.constant(0.0, type=compute_type)
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                c_N_i32 = Int32(N)
                is_valid = idx < c_N_i32
                c0_i = Int32(0)
                idx_safe = is_valid.select(idx, c0_i)
                x_e = _load_scalar(row_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.extf(compute_type)
                x_av = ArithValue(x)
                x2 = x_av * x_av
                x2_safe = is_valid.select(x2, c_zero_f)
                thread_sumsq = ArithValue(thread_sumsq) + x2_safe

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = ArithValue(sum_sq) / n_float
            ms_eps = ArithValue(mean_sq) + eps_c
            rrms = ms_eps.rsqrt(fastmath=fm_fast)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                c_N_i32 = Int32(N)
                if arith.cmpi(arith.CmpIPredicate.ult, idx, c_N_i32):
                    x_e = _load_scalar(row_div, idx)
                    g_e = _load_scalar(gamma_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.extf(compute_type)
                    g = g_e if dtype_str == "f32" else g_e.extf(compute_type)
                    norm = ArithValue(x) * ArithValue(rrms)
                    y = norm * ArithValue(g)
                    if dtype_str == "f32":
                        y_e = y
                    elif dtype_str == "bf16":
                        y_e = y.truncf(elem_type)
                    else:
                        y_e = y.truncf(elem_type)
                    _store_scalar(out_div, idx, y_e)

    @flyc.jit
    def launch_rmsnorm(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        idx_m = ArithValue(m_in).index_cast(T.index)
        launcher = rmsnorm_kernel(Input, Gamma, Gamma, Output)
        launcher.launch(
            grid=(idx_m, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm


def _build_rmsnorm_quant_module(
    M: int,
    N: int,
    dtype_str: str,
    *,
    is_smooth: bool,
    quant_dtype_str: str = "i8",
):
    arch = get_hip_arch()

    tile_cols = BLOCK_THREADS * VEC_WIDTH
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    quant_elem_type = _quant_dtype_to_elem_type(quant_dtype_str)
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)
    quant_elem_bytes = 1

    allocator = SmemAllocator(None, arch=arch)
    f32_bytes = 4
    red_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = red_offset + RED_SLOTS * f32_bytes
    red2_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = red2_offset + RED_SLOTS * f32_bytes

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

        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32

        fm_fast = arith.FastMathFlags.fast
        eps_c = arith.constant(EPS, type=compute_type)
        n_float = arith.constant(float(N), type=compute_type)
        c_zero_f = arith.constant(0.0, type=compute_type)
        c_one_f = arith.constant(1.0, type=compute_type)
        c_neg_inf = arith.constant(float("-inf"), type=compute_type)
        c_dtype_max = arith.constant(quant_dtype_max, type=compute_type)

        base_ptr = allocator.get_base()
        s_red = SmemPtr(base_ptr, red_offset, T.f32, shape=(RED_SLOTS,))
        s_red2 = SmemPtr(base_ptr, red2_offset, T.f32, shape=(RED_SLOTS,))
        s_red.get()
        s_red2.get()

        def wave_reduce_add(x):
            width_i32 = fx.Int32(WARP_SIZE)
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = fx.Int32(WARP_SIZE // (2 << _sh_exp))
                peer = w.shuffle_xor(off, width_i32)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def wave_reduce_max(x):
            width_i32 = fx.Int32(WARP_SIZE)
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = fx.Int32(WARP_SIZE // (2 << _sh_exp))
                peer = w.shuffle_xor(off, width_i32)
                w = w.maximumf(peer)
            return w

        def block_reduce_add(val):
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

        def block_reduce_add2(val0, val1):
            if RED_SLOTS == 1:
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == fx.Int32(0):
                wave_idx = ArithValue(wave).index_cast(T.index)
                s_red.store(w0, [wave_idx])
                s_red2.store(w1, [wave_idx])
            gpu.barrier()

            if wave == fx.Int32(0):
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, fx.Int32(0))
                lane_safe_idx = ArithValue(lane_safe).index_cast(T.index)
                v0 = s_red.load([lane_safe_idx])
                v1 = s_red2.load([lane_safe_idx])
                ww0 = in_range.select(v0, c_zero_f)
                ww1 = in_range.select(v1, c_zero_f)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == fx.Int32(0):
                    c0_idx = fx.Index(0)
                    s_red.store(ww0, [c0_idx])
                    s_red2.store(ww1, [c0_idx])
            gpu.barrier()

            c0_idx = fx.Index(0)
            return s_red.load([c0_idx]), s_red2.load([c0_idx])

        def block_reduce_max(val):
            if RED_SLOTS == 1:
                return wave_reduce_max(val)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w = wave_reduce_max(val)
            if lane == fx.Int32(0):
                wave_idx = ArithValue(wave).index_cast(T.index)
                s_red.store(w, [wave_idx])
            gpu.barrier()

            if wave == fx.Int32(0):
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, fx.Int32(0))
                lane_safe_idx = ArithValue(lane_safe).index_cast(T.index)
                v = s_red.load([lane_safe_idx])
                ww = in_range.select(v, c_neg_inf)
                ww = wave_reduce_max(ww)
                if lane == fx.Int32(0):
                    c0_idx = fx.Index(0)
                    s_red.store(ww, [c0_idx])
            gpu.barrier()

            c0_idx = fx.Index(0)
            return s_red.load([c0_idx])

        if N >= tile_cols and N % tile_cols == 0 and elem_bits <= 16:
            from flydsl.expr.arith import ArithValue

            num_tiles = N // tile_cols

            vec_type_c = T.vec(VEC_WIDTH, compute_type)
            vec_type_e = T.vec(VEC_WIDTH, elem_type)
            vec_type_q = T.vec(VEC_WIDTH, quant_elem_type)
            vec_q_pack_type = T.vec((VEC_WIDTH * quant_elem_bytes) // 4, T.i32)
            vec_i32_ty = T.vec(VEC_WIDTH, T.i32)
            abs_mask = arith.constant_vector(0x7FFFFFFF, vec_i32_ty)

            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            if is_smooth:
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            yscale_rsrc = buffer_ops.create_buffer_resource(YScale, max_size=True)
            out_rsrc = buffer_ops.create_buffer_resource(Output, max_size=True)
            row_soffset_out = ArithValue(bid) * (N * quant_elem_bytes)
            thr_col_bytes_out = ArithValue(tid) * (VEC_WIDTH * quant_elem_bytes)

            row_in = fx.slice(Input_buf, (bid, None))
            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            if is_smooth:
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            vec_reg_ty = fx.MemRefType.get(
                elem_type, fx.LayoutType.get(VEC_WIDTH, 1), fx.AddressSpace.Register
            )
            vec_reg_lay = fx.make_layout(VEC_WIDTH, 1)

            def _load_vec(div_tensor, idx):
                r = fx.memref_alloca(vec_reg_ty, vec_reg_lay)
                fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, idx)), r)
                return ArithValue(fx.memref_load_vec(r))

            thread_sumsq = c_zero_f
            thread_dummy = c_zero_f
            cache_as_elem = (dtype_str != "f32")
            in_local = []

            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                vec_e = _load_vec(in_div, idx)

                if cache_as_elem:
                    in_local.append(vec_e)
                    x = vec_e.extf(vec_type_c)
                else:
                    x = vec_e
                    in_local.append(x)

                x_av = ArithValue(x)
                x2 = x_av * x_av
                red2 = vector.reduction(compute_type, vector.CombiningKind.ADD, x2, fastmath=fm_fast)
                thread_sumsq = ArithValue(thread_sumsq) + red2

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
            mean_sq = ArithValue(sum_sq) / n_float
            ms_eps = ArithValue(mean_sq) + eps_c
            rrms = ms_eps.rsqrt(fastmath=fm_fast)
            rrms_splat = vector.broadcast(vec_type_c, rrms)
            rrms_splat_av = ArithValue(rrms_splat)

            thread_row_max = c_zero_f
            y_local = []

            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS

                g_e = _load_vec(gamma_div, idx)
                g = g_e if dtype_str == "f32" else g_e.extf(vec_type_c)

                x = in_local[tile_i]
                if cache_as_elem:
                    x = x.extf(vec_type_c)

                y = (ArithValue(x) * rrms_splat_av) * ArithValue(g)
                if is_smooth:
                    s_e = _load_vec(xscale_div, idx)
                    s = s_e if dtype_str == "f32" else s_e.extf(vec_type_c)
                    y = ArithValue(y) * ArithValue(s)

                y_local.append(y)
                y_bits = ArithValue(y).bitcast(vec_i32_ty)
                y_abs_bits = y_bits & abs_mask
                y_abs = vector.bitcast(vec_type_c, y_abs_bits)
                tile_max = vector.reduction(compute_type, vector.CombiningKind.MAXNUMF, y_abs)
                thread_row_max = thread_row_max.maximumf(tile_max)

            row_max = block_reduce_max(thread_row_max)
            scale = ArithValue(row_max) / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == fx.Int32(0):
                buffer_ops.buffer_store(final_scale, yscale_rsrc, bid)

            inv_scale = ArithValue(c_one_f) / ArithValue(final_scale)
            inv_scale_splat = vector.broadcast(vec_type_c, inv_scale)

            for tile_i in range_constexpr(num_tiles):
                q = ArithValue(y_local[tile_i]) * ArithValue(inv_scale_splat)
                q_i8 = arith.FPToSIOp(vec_type_q, arith.unwrap(q)).result
                q_packed = vector.bitcast(vec_q_pack_type, q_i8)
                col_bytes_out = ArithValue(thr_col_bytes_out) + (tile_i * tile_cols * quant_elem_bytes)
                dw_out = col_bytes_out.shrui(arith.constant(2, type=T.i32))
                buffer_ops.buffer_store(q_packed, out_rsrc, dw_out, soffset_bytes=row_soffset_out)

        else:
            from flydsl.expr.arith import ArithValue

            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            if is_smooth:
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            yscale_rsrc = buffer_ops.create_buffer_resource(YScale, max_size=True)
            out_rsrc = buffer_ops.create_buffer_resource(Output, max_size=True)
            row_soffset_out = ArithValue(bid) * (N * quant_elem_bytes)

            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )
            scalar_reg_ty = fx.MemRefType.get(elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
            scalar_reg_lay = fx.make_layout(1, 1)

            row_in = fx.slice(Input_buf, (bid, None))
            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            if is_smooth:
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(1, 1))

            def _load_scalar(divided_tensor, index):
                view = fx.slice(divided_tensor, (None, index))
                r = fx.memref_alloca(scalar_reg_ty, scalar_reg_lay)
                fx.copy_atom_call(copy_atom_s, view, r)
                v = fx.memref_load_vec(r)
                return vector.extract(v, static_position=[0])

            def _abs_scalar(val):
                is_neg = val < c_zero_f
                neg_val = c_zero_f - ArithValue(val)
                return is_neg.select(neg_val, val)

            thread_sumsq = c_zero_f
            c_N_i32 = Int32(N)
            c0_i = Int32(0)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < c_N_i32
                idx_safe = is_valid.select(idx, c0_i)
                x_e = _load_scalar(row_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.extf(compute_type)
                x2 = ArithValue(x) * ArithValue(x)
                thread_sumsq = ArithValue(thread_sumsq) + is_valid.select(x2, c_zero_f)

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = ArithValue(sum_sq) / n_float
            ms_eps = ArithValue(mean_sq) + eps_c
            rrms = ms_eps.rsqrt(fastmath=fm_fast)

            thread_row_max = c_zero_f
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < c_N_i32
                idx_safe = is_valid.select(idx, c0_i)
                x_e = _load_scalar(row_div, idx_safe)
                g_e = _load_scalar(gamma_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.extf(compute_type)
                g = g_e if dtype_str == "f32" else g_e.extf(compute_type)
                y = (ArithValue(x) * ArithValue(rrms)) * ArithValue(g)
                if is_smooth:
                    s_e = _load_scalar(xscale_div, idx_safe)
                    s = s_e if dtype_str == "f32" else s_e.extf(compute_type)
                    y = ArithValue(y) * ArithValue(s)
                y_abs = _abs_scalar(y)
                thread_row_max = thread_row_max.maximumf(is_valid.select(y_abs, c_zero_f))

            row_max = block_reduce_max(thread_row_max)
            scale = ArithValue(row_max) / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == fx.Int32(0):
                buffer_ops.buffer_store(final_scale, yscale_rsrc, bid)

            inv_scale = ArithValue(c_one_f) / ArithValue(final_scale)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if arith.cmpi(arith.CmpIPredicate.ult, idx, c_N_i32):
                    x_e = _load_scalar(row_div, idx)
                    g_e = _load_scalar(gamma_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.extf(compute_type)
                    g = g_e if dtype_str == "f32" else g_e.extf(compute_type)
                    y = (ArithValue(x) * ArithValue(rrms)) * ArithValue(g)
                    if is_smooth:
                        s_e = _load_scalar(xscale_div, idx)
                        s = s_e if dtype_str == "f32" else s_e.extf(compute_type)
                        y = ArithValue(y) * ArithValue(s)
                    q = ArithValue(y) * ArithValue(inv_scale)
                    q_i8 = arith.FPToSIOp(quant_elem_type, arith.unwrap(q)).result
                    buffer_ops.buffer_store(q_i8, out_rsrc, idx, soffset_bytes=row_soffset_out)

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
            allocator.finalized = False
            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                allocator.finalize()

            idx_m = ArithValue(m_in).index_cast(T.index)
            launcher = rmsnorm_quant_kernel(Input, Gamma, XScale, YScale, Output)
            launcher.launch(
                grid=(idx_m, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_rmsnorm_smoothquant

    @flyc.jit
    def launch_rmsnorm_dynamicquant(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        YScale: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        idx_m = ArithValue(m_in).index_cast(T.index)
        launcher = rmsnorm_quant_kernel(Input, Gamma, Gamma, YScale, Output)
        launcher.launch(
            grid=(idx_m, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm_dynamicquant


def build_rmsnorm_dynamicquant_module(
    M: int,
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
):
    return _build_rmsnorm_quant_module(
        M,
        N,
        dtype_str,
        is_smooth=False,
        quant_dtype_str=quant_dtype_str,
    )


def build_rmsnorm_smoothquant_module(
    M: int,
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
):
    return _build_rmsnorm_quant_module(
        M,
        N,
        dtype_str,
        is_smooth=True,
        quant_dtype_str=quant_dtype_str,
    )
