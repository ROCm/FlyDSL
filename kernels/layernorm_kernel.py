# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""LayerNorm kernel builder using the @flyc.kernel API.

LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta

Two paths:
  - Fast path (N == BLOCK_THREADS * VEC_WIDTH * 4): vectorised tiled copy,
    register caching, pipelined gamma/beta loads.
  - Generic path (arbitrary N): scalar 2-pass implementation.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T, Int32
from flydsl.expr.vector import ReductionOp, full
from flydsl.expr.numeric import Numeric, Float32, Uint32

from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from flydsl._mlir import ir


KERNEL_NAME = "layernorm"

EPS = 1e-5

import math
from kernels.kernels_common import dtype_to_elem_type, get_warp_size

BLOCK_THREADS = 256
WARP_SIZE = get_warp_size()
VEC_WIDTH = 8
USE_NONTEMPORAL = True
VEC_ALIGN = 16


def build_layernorm_module(M: int, N: int, dtype_str: str):
    arch = get_hip_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    tile_cols_py = BLOCK_THREADS * VEC_WIDTH

    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)

    elem_bits = 32 if dtype_str == "f32" else 16

    # ── Shared-memory allocation for block reductions ─────────────────────
    allocator = SmemAllocator(None, arch=arch)
    f32_bytes = 4
    sum_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = sum_offset + RED_SLOTS * f32_bytes
    sumsq_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = sumsq_offset + RED_SLOTS * f32_bytes

    # ── GPU kernel ────────────────────────────────────────────────────────
    @flyc.kernel
    def layernorm_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        Output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32

        fm_fast = arith.FastMathFlags.fast
        eps_c = arith.constant(EPS, type=compute_type)

        base_ptr = allocator.get_base()
        s_sum = SmemPtr(base_ptr, sum_offset, T.f32, shape=(RED_SLOTS,))
        s_sumsq = SmemPtr(base_ptr, sumsq_offset, T.f32, shape=(RED_SLOTS,))
        s_sum.get()
        s_sumsq.get()

        # ── helpers: wave / block reduction ───────────────────────────────
        def wave_reduce_add(x):
            width_i32 = fx.Int32(WARP_SIZE)
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = fx.Int32(WARP_SIZE // (2 << _sh_exp))
                peer = w.shuffle_xor(off, width_i32)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add2(val0, val1):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == fx.Int32(0):
                wave_idx = ArithValue(wave).index_cast(T.index)
                SmemPtr.store(s_sum, w0, [wave_idx])
                SmemPtr.store(s_sumsq, w1, [wave_idx])
            gpu.barrier()

            if wave == fx.Int32(0):
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, fx.Int32(0))
                lane_safe_idx = ArithValue(lane_safe).index_cast(T.index)
                v0 = SmemPtr.load(s_sum, [lane_safe_idx])
                v1 = SmemPtr.load(s_sumsq, [lane_safe_idx])
                z = fx.Float32(0.0)
                ww0 = in_range.select(v0, z)
                ww1 = in_range.select(v1, z)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == fx.Int32(0):
                    c0_idx = fx.Index(0)
                    SmemPtr.store(s_sum, ww0, [c0_idx])
                    SmemPtr.store(s_sumsq, ww1, [c0_idx])
            gpu.barrier()

            c0_idx = fx.Index(0)
            return SmemPtr.load(s_sum, [c0_idx]), SmemPtr.load(s_sumsq, [c0_idx])

        def compute_mean_rstd(sum_val, sumsq_val):
            inv_n = arith.constant(1.0 / float(N), type=compute_type)
            s = ArithValue(sum_val)
            ss = ArithValue(sumsq_val)
            mean = s * inv_n
            mean_sq = ss * inv_n
            mean2 = mean * mean
            var = mean_sq - mean2
            c0_f = arith.constant(0.0, type=compute_type)
            is_neg = var < c0_f
            var = is_neg.select(c0_f, var)
            var_eps = ArithValue(var) + eps_c
            rstd = var_eps.rsqrt(fastmath=fm_fast)
            return mean, rstd

        # ==================================================================
        # Fast path: N == BLOCK_THREADS * VEC_WIDTH * 4
        # Uses buffer_load / buffer_store for high-bandwidth vectorised
        # memory access (same approach as preshuffle_gemm).
        # ==================================================================
        if const_expr(N == (BLOCK_THREADS * VEC_WIDTH * 4) and elem_bits <= 16):
            num_tiles_py = 4
            elem_dtype = Numeric.from_ir_type(elem_type)

            c_zero_f = arith.constant(0.0, type=compute_type)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            in_local = []

            # ── Layout API: buffer-backed tensors + tiled access ─────
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(VEC_WIDTH, 1))

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

            # ── Pass 1: load input, accumulate sum / sumsq ───────────────
            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                vec = _load_vec(in_div, idx)
                in_local.append(vec)
                x = vec.to(Float32)

                x2 = x * x
                red = x.reduce(ReductionOp.ADD, fastmath=fm_fast)
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sum = ArithValue(thread_sum) + red
                thread_sumsq = ArithValue(thread_sumsq) + red2

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            g_cur = _load_vec(gamma_div, tid).to(Float32)
            b_cur = _load_vec(beta_div, tid).to(Float32)

            # ── Pass 2: normalize + affine + store ───────────────────────
            for tile_i in range_constexpr(num_tiles_py):
                g_next = g_cur
                b_next = b_cur
                if const_expr(tile_i + 1 < num_tiles_py):
                    next_idx = tid + (tile_i + 1) * BLOCK_THREADS
                    g_next = _load_vec(gamma_div, next_idx).to(Float32)
                    b_next = _load_vec(beta_div, next_idx).to(Float32)
                else:
                    g_next = g_cur
                    b_next = b_cur

                x = in_local[tile_i].to(Float32)
                y = (x - mean) * rstd
                y = y * g_cur + b_cur

                out_e = y.to(elem_dtype)
                if const_expr(dtype_str == "bf16"):
                    if const_expr(USE_HW_CVT_PK_BF16_F32):
                        out_e = y.to(elem_dtype)
                    else:
                        u = y.bitcast(Uint32)
                        upper = u >> 16
                        lsb = upper & 1
                        bias = lsb + 0x7FFF
                        u_round = y.bitcast(Uint32) + bias
                        bf16_bits = u_round >> 16
                        even = bf16_bits.shuffle(bf16_bits, [0, 2, 4, 6])
                        odd = bf16_bits.shuffle(bf16_bits, [1, 3, 5, 7])
                        odd_sh = odd << 16
                        packed = even | odd_sh
                        out_e = packed.bitcast(elem_dtype)
                elif const_expr(dtype_str == "f32"):
                    out_e = y
                else:
                    out_e = y.to(elem_dtype)

                out_idx = tid + tile_i * BLOCK_THREADS
                _store_vec(out_e, out_div, out_idx)

                g_cur = g_next
                b_cur = b_next

        else:
            # ==============================================================
            # Generic path: 2-pass scalar implementation for arbitrary N
            # ==============================================================
            elem_dtype = Numeric.from_ir_type(elem_type)

            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            c_zero_f = arith.constant(0.0, type=compute_type)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f

            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )
            scalar_reg_ty = fx.MemRefType.get(
                elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register
            )
            scalar_reg_lay = fx.make_layout(1, 1)

            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))

            def _load_scalar(divided_tensor, index):
                view = fx.slice(divided_tensor, (None, index))
                r = fx.memref_alloca(scalar_reg_ty, scalar_reg_lay)
                fx.copy_atom_call(copy_atom_s, view, r)
                return fx.memref_load_vec(r)[0].ir_value()

            def _store_scalar(divided_tensor, index, val):
                r = fx.memref_alloca(scalar_reg_ty, scalar_reg_lay)
                ts = full(1, elem_dtype(val), elem_dtype)
                fx.memref_store_vec(ts, r)
                view = fx.slice(divided_tensor, (None, index))
                fx.copy_atom_call(copy_atom_s, r, view)

            # ── Pass 1: sum + sumsq ──────────────────────────────────────
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                c_N_i32 = Int32(N)
                is_valid = idx < c_N_i32
                c0_i = Int32(0)
                idx_safe = is_valid.select(idx, c0_i)
                x_e = _load_scalar(row_div, idx_safe)
                x = (
                    x_e
                    if dtype_str == "f32"
                    else x_e.extf(compute_type)
                )
                x_av = ArithValue(x)
                x2 = x_av * x_av
                x_safe = is_valid.select(x, c_zero_f)
                x2_safe = is_valid.select(x2, c_zero_f)
                thread_sum = ArithValue(thread_sum) + x_safe
                thread_sumsq = ArithValue(thread_sumsq) + x2_safe

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            # ── Pass 2: normalize + affine + store ───────────────────────
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                c_N_i32 = Int32(N)
                if arith.cmpi(arith.CmpIPredicate.ult, idx, c_N_i32):
                    x_e = _load_scalar(row_div, idx)
                    g_e = _load_scalar(gamma_div, idx)
                    b_e = _load_scalar(beta_div, idx)
                    x = (
                        x_e
                        if dtype_str == "f32"
                        else x_e.extf(compute_type)
                    )
                    g = (
                        g_e
                        if dtype_str == "f32"
                        else g_e.extf(compute_type)
                    )
                    b = (
                        b_e
                        if dtype_str == "f32"
                        else b_e.extf(compute_type)
                    )
                    diff = ArithValue(x) - mean
                    norm = diff * rstd
                    scaled = norm * g
                    y = scaled + b
                    y_e = y
                    if const_expr(dtype_str == "bf16"):
                        y_e = y.truncf(elem_type)
                    elif const_expr(dtype_str == "f32"):
                        y_e = y
                    else:
                        y_e = y.truncf(elem_type)
                    _store_scalar(out_div, idx, y_e)

    # ── JIT host launcher ─────────────────────────────────────────────────
    @flyc.jit
    def launch_layernorm(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        Output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        idx_m = ArithValue(m_in).index_cast(T.index)
        launcher = layernorm_kernel(Input, Gamma, Beta, Output)
        launcher.launch(
            grid=(idx_m, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_layernorm
