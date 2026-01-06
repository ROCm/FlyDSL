"""LayerNorm kernel builder used by tests.

This file intentionally keeps the kernel builder logic identical to the version
embedded in `tests/kernels/test_layernorm.py` (before factoring) to preserve
codegen and performance. Only test-only helpers/imports are removed.
"""

from flydsl.compiler.context import RAIIMLIRContextModule
from flydsl.dialects.ext import flir
from flydsl.dialects.ext.python_control_flow import range_constexpr
from . import reduce as reduce_utils
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator
from _mlir import ir
import _mlir.extras.types as T


KERNEL_NAME = "layernorm"


def unwrap(v):
    if hasattr(v, "value"):
        return v.value
    if hasattr(v, "_value"):
        return v._value
    if hasattr(v, "result"):
        return v.result
    return v


EPS = 1e-5


def dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f32":
        return T.f32()
    if dtype_str == "f16":
        return T.f16()
    if dtype_str == "bf16":
        return T.bf16()
    raise ValueError(f"unsupported dtype: {dtype_str}")


# Expose modules through Flir interface (keep behavior/perf, avoid mlir.* imports).
gpu = flir.gpu_ext
scf = flir.scf_ext
# Keep arith as the raw dialect module here (this file uses arith.constant(Type, value) form).
arith = flir.arith
mlir_arith = flir.arith
memref = flir.memref
vector = flir.vector
math = flir.math


BLOCK_THREADS = 256
WARP_SIZE = 64
VEC_WIDTH = 8
USE_NONTEMPORAL = True
VEC_ALIGN = 16


def build_layernorm_module(M: int, N: int, dtype_str: str):
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    gpu.set_container_module(ctx.module)

    arch = get_hip_arch()
    # gfx950 supports efficient BF16 pack via v_cvt_pk_bf16_f32.
    # gfx942 does *not* support it and tends to lower f32->bf16 to heavier sequences,
    # so we keep the manual pack there for performance.
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or arch.startswith("gfx95")
    allocator = SmemAllocator(ctx, arch=arch)

    elem_type = dtype_to_elem_type(dtype_str)
    compute_type = T.f32()  # compute in fp32 for stability (and to keep bf16 safe on backend)

    tile_cols_py = BLOCK_THREADS * VEC_WIDTH

    # Allocate Shared Memory for block reductions (one slot per wave).
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    smem_red_sum = allocator.allocate_array(T.f32(), RED_SLOTS)
    smem_red_sumsq = allocator.allocate_array(T.f32(), RED_SLOTS)

    @gpu.module("layernorm_module", [f'#rocdl.target<chip = "{arch}", abi = "500">'])
    def gpu_mod():
        allocator.finalize()

        @gpu.func(emit=True)
        def layernorm_kernel(
            Input: T.memref(M, N, elem_type),
            Gamma: T.memref(N, elem_type),
            Beta: T.memref(N, elem_type),
            Output: T.memref(M, N, elem_type)
        ):
            row = flir.block_idx("x")
            tid = flir.thread_idx("x")

            zero_idx = flir.const_index(0)
            n_float = arith.constant(compute_type, float(N))
            eps = arith.constant(compute_type, EPS)
            fm_fast = mlir_arith.FastMathFlags.fast

            base_ptr = allocator.get_base()
            s_sum = smem_red_sum(base_ptr).get()
            s_sumsq = smem_red_sumsq(base_ptr).get()
            # Rocir-style tensor views + tiled copies (like elementwise_add_kernel).
            c0_idx = flir.const_index(0)
            tile_cols = BLOCK_THREADS * VEC_WIDTH  # python int
            tensor_In = flir.make_tensor(Input, shape=(M, N), strides=(N, 1))
            tensor_Out = flir.make_tensor(Output, shape=(M, N), strides=(N, 1))
            tensor_Gamma = flir.make_tensor(Gamma, shape=(N,), strides=(1,))
            tensor_Beta = flir.make_tensor(Beta, shape=(N,), strides=(1,))
            gIn = flir.zipped_divide(tensor_In, (1, tile_cols))
            gOut = flir.zipped_divide(tensor_Out, (1, tile_cols))

            thr_layout = flir.make_ordered_layout((1, BLOCK_THREADS), order=(1, 0))
            val_layout = flir.make_ordered_layout((1, VEC_WIDTH), order=(1, 0))
            copy_atom_e = flir.make_copy_atom(elem_type, vector_size=VEC_WIDTH)
            tiled_copy_e = flir.make_tiled_copy_tv(
                copy_atom_e, thr_layout, val_layout,
                thr_shape=(1, BLOCK_THREADS), val_shape=(1, VEC_WIDTH)
            )
            thr_copy_e = tiled_copy_e.get_slice(unwrap(tid))
            block_reduce_add = reduce_utils.make_block_reduce_add(
                tid=tid,
                fm_fast=fm_fast,
                WARP_SIZE=WARP_SIZE,
                RED_SLOTS=RED_SLOTS,
                gpu=gpu,
                arith=arith,
                arith_ops=mlir_arith,
                flir=flir,
                T=T,
                ir=ir,
                zero_idx=zero_idx,
            )
            block_reduce_add2 = reduce_utils.make_block_reduce_add2(
                tid=tid,
                fm_fast=fm_fast,
                WARP_SIZE=WARP_SIZE,
                RED_SLOTS=RED_SLOTS,
                gpu=gpu,
                arith=arith,
                arith_ops=mlir_arith,
                flir=flir,
                T=T,
                ir=ir,
                zero_idx=zero_idx,
            )

            def bf16_pack_vec8_rne(vec_f32):
                # Manual bf16 cast with RNE rounding (NaN-safe, no scaling/clip).
                # We keep this path for gfx942 to avoid the particularly heavy default lowering,
                # while preserving NaN/Inf semantics (this is still just an f32->bf16 cast).
                vec_i32_ty = ir.VectorType.get([VEC_WIDTH], T.i32())
                vec4_i32_ty = ir.VectorType.get([VEC_WIDTH // 2], T.i32())
                vec_bf16_ty = ir.VectorType.get([VEC_WIDTH], elem_type)

                c16_i32 = arith.constant(T.i32(), 16).value
                c7fff_i32 = arith.constant(T.i32(), 0x7FFF).value
                c1_i32 = arith.constant(T.i32(), 1).value
                c_abs_mask_i32 = arith.constant(T.i32(), 0x7FFFFFFF).value
                c_inf_i32 = arith.constant(T.i32(), 0x7F800000).value
                c_qnan_bf16_i32 = arith.constant(T.i32(), 0x0040).value

                c16_i32_v = vector.splat(vec_i32_ty, unwrap(c16_i32))
                c7fff_i32_v = vector.splat(vec_i32_ty, unwrap(c7fff_i32))
                c1_i32_v = vector.splat(vec_i32_ty, unwrap(c1_i32))
                c_abs_mask_i32_v = vector.splat(vec_i32_ty, unwrap(c_abs_mask_i32))
                c_inf_i32_v = vector.splat(vec_i32_ty, unwrap(c_inf_i32))
                c_qnan_bf16_i32_v = vector.splat(vec_i32_ty, unwrap(c_qnan_bf16_i32))

                u = mlir_arith.bitcast(vec_i32_ty, unwrap(vec_f32))

                # Detect NaN: abs(u) > +Inf (works for both qNaN/sNaN, avoids mantissa checks).
                abs_u = mlir_arith.AndIOp(unwrap(u), unwrap(c_abs_mask_i32_v)).result
                is_nan = mlir_arith.CmpIOp(mlir_arith.CmpIPredicate.ugt, unwrap(abs_u), unwrap(c_inf_i32_v)).result

                hi = mlir_arith.ShRUIOp(unwrap(u), unwrap(c16_i32_v)).result
                lsb = mlir_arith.AndIOp(unwrap(hi), unwrap(c1_i32_v)).result
                bias = mlir_arith.AddIOp(unwrap(c7fff_i32_v), unwrap(lsb)).result
                u_round = mlir_arith.AddIOp(unwrap(u), unwrap(bias)).result
                bf16_bits_rne = mlir_arith.ShRUIOp(unwrap(u_round), unwrap(c16_i32_v)).result

                # For NaN, ensure we produce a quiet NaN in bf16 (mantissa != 0).
                bf16_bits_nan = mlir_arith.OrIOp(unwrap(hi), unwrap(c_qnan_bf16_i32_v)).result
                bf16_bits = flir.arith.SelectOp(unwrap(is_nan), unwrap(bf16_bits_nan), unwrap(bf16_bits_rne)).result

                even = vector.shuffle(bf16_bits, bf16_bits, mask=[0, 2, 4, 6])
                odd = vector.shuffle(bf16_bits, bf16_bits, mask=[1, 3, 5, 7])
                odd_sh = mlir_arith.ShLIOp(unwrap(odd), unwrap(vector.splat(vec4_i32_ty, unwrap(c16_i32)))).result
                packed = mlir_arith.OrIOp(unwrap(even), unwrap(odd_sh)).result
                return vector.bitcast(vec_bf16_ty, unwrap(packed))

            # Fast-path: keep the original register-row variant for the tuned (N==8192) case.
            if N == (BLOCK_THREADS * VEC_WIDTH * 4):
                num_tiles_py = 4
                # Read Input once into registers (each thread holds 32 fp32 values = 4 vectors),
                # then reuse those registers for reduction + normalize + writeback.
                c_zero = arith.constant(compute_type, 0.0).value
                thread_sum = unwrap(c_zero)
                thread_sumsq = unwrap(c_zero)
                # Reduce VGPR pressure by caching bf16/f16 payload vectors when possible.
                cache_as_elem = (dtype_str != "f32")
                in_local = []  # bf16/f16: list[vector<VEC_WIDTH x elem_type>]; f32: list[vector<VEC_WIDTH x f32>]

                vec_type_c = ir.VectorType.get([VEC_WIDTH], compute_type)
                vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
                for tile_i in range(num_tiles_py):
                    blkIn = gIn[(unwrap(row), tile_i)]
                    thrIn = thr_copy_e.partition_S(blkIn)
                    frgIn = flir.make_fragment_like(thrIn, elem_type)
                    vec_e = flir.copy(
                        tiled_copy_e,
                        thrIn,
                        frgIn,
                        nontemporal=USE_NONTEMPORAL,
                        alignment=VEC_ALIGN,
                        return_vector=True,
                    )

                    if cache_as_elem:
                        in_local.append(vec_e)
                        x = mlir_arith.extf(vec_type_c, unwrap(vec_e))
                    else:
                        x = vec_e
                        in_local.append(x)

                    x2 = mlir_arith.MulFOp(unwrap(x), unwrap(x), fastmath=fm_fast).result
                    red = vector.reduction(compute_type, "add", unwrap(x), fastmath=fm_fast)
                    red2 = vector.reduction(compute_type, "add", unwrap(x2), fastmath=fm_fast)
                    thread_sum = mlir_arith.AddFOp(unwrap(thread_sum), unwrap(red), fastmath=fm_fast).result
                    thread_sumsq = mlir_arith.AddFOp(unwrap(thread_sumsq), unwrap(red2), fastmath=fm_fast).result

                sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq, s_sum, s_sumsq)

                inv_n = arith.constant(compute_type, 1.0 / float(N))
                mean = mlir_arith.MulFOp(unwrap(sum_val), unwrap(inv_n), fastmath=fm_fast).result
                mean_sq = mlir_arith.MulFOp(unwrap(sumsq_val), unwrap(inv_n), fastmath=fm_fast).result
                mean2 = mlir_arith.MulFOp(unwrap(mean), unwrap(mean), fastmath=fm_fast).result
                var = mlir_arith.SubFOp(unwrap(mean_sq), unwrap(mean2), fastmath=fm_fast).result
                # Numerical safety: with fast-math and cancellation, `var` can become slightly negative
                # and lead to NaNs in rsqrt for small-N cases. Clamp to >= 0 before adding eps.
                c0_f = arith.constant(compute_type, 0.0)
                is_neg = mlir_arith.CmpFOp(mlir_arith.CmpFPredicate.OLT, unwrap(var), unwrap(c0_f.value)).result
                var = flir.arith.SelectOp(unwrap(is_neg), unwrap(c0_f.value), unwrap(var)).result

                var_eps = mlir_arith.AddFOp(unwrap(var), unwrap(eps.value), fastmath=fm_fast).result
                rstd = math.rsqrt(unwrap(var_eps), fastmath=fm_fast)

                vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
                vec_type_c = ir.VectorType.get([VEC_WIDTH], compute_type)
                mean_splat = vector.splat(vec_type_c, unwrap(mean))
                rstd_splat = vector.splat(vec_type_c, unwrap(rstd))

                # Pipeline Gamma/Beta loads.
                thread_offset_base = mlir_arith.MulIOp(unwrap(tid), flir.const_index(VEC_WIDTH)).result
                c_base0 = flir.const_index(0)
                curr_idx0 = mlir_arith.AddIOp(unwrap(c_base0), unwrap(thread_offset_base)).result
                g_e_cur = vector.load(vec_type_e, Gamma, [unwrap(curr_idx0)], alignment=VEC_ALIGN)
                b_e_cur = vector.load(vec_type_e, Beta, [unwrap(curr_idx0)], alignment=VEC_ALIGN)
                g_cur = g_e_cur if dtype_str == "f32" else mlir_arith.extf(vec_type_c, unwrap(g_e_cur))
                b_cur = b_e_cur if dtype_str == "f32" else mlir_arith.extf(vec_type_c, unwrap(b_e_cur))

                for tile_i in range(num_tiles_py):
                    base_idx_int = tile_i * tile_cols
                    c_base = flir.const_index(base_idx_int)
                    curr_idx = mlir_arith.AddIOp(unwrap(c_base), unwrap(thread_offset_base)).result

                    x = in_local[tile_i]
                    if cache_as_elem:
                        x = mlir_arith.extf(vec_type_c, unwrap(x))
                    if tile_i + 1 < num_tiles_py:
                        next_base_idx_int = (tile_i + 1) * tile_cols
                        c_base_next = flir.const_index(next_base_idx_int)
                        next_idx = mlir_arith.AddIOp(unwrap(c_base_next), unwrap(thread_offset_base)).result
                        g_e_next = vector.load(vec_type_e, Gamma, [unwrap(next_idx)], alignment=VEC_ALIGN)
                        b_e_next = vector.load(vec_type_e, Beta, [unwrap(next_idx)], alignment=VEC_ALIGN)
                        g_next = g_e_next if dtype_str == "f32" else mlir_arith.extf(vec_type_c, unwrap(g_e_next))
                        b_next = b_e_next if dtype_str == "f32" else mlir_arith.extf(vec_type_c, unwrap(b_e_next))
                    else:
                        g_next = g_cur
                        b_next = b_cur

                    diff = mlir_arith.SubFOp(unwrap(x), unwrap(mean_splat), fastmath=fm_fast).result
                    norm = mlir_arith.MulFOp(unwrap(diff), unwrap(rstd_splat), fastmath=fm_fast).result
                    scaled = mlir_arith.MulFOp(unwrap(norm), unwrap(g_cur), fastmath=fm_fast).result
                    y = mlir_arith.AddFOp(unwrap(scaled), unwrap(b_cur), fastmath=fm_fast).result

                    if dtype_str == "bf16":
                        if USE_HW_CVT_PK_BF16_F32:
                            out_e = mlir_arith.truncf(vec_type_e, unwrap(y))
                        else:
                            out_e = bf16_pack_vec8_rne(y)
                    else:
                        out_e = y if dtype_str == "f32" else mlir_arith.truncf(vec_type_e, unwrap(y))

                    blkOut = gOut[(unwrap(row), tile_i)]
                    thrOut = thr_copy_e.partition_S(blkOut)
                    frgOut = flir.make_fragment_like(thrOut, elem_type)
                    vector.store(unwrap(out_e), frgOut.memref, [c0_idx, c0_idx], alignment=VEC_ALIGN)
                    flir.copy(
                        tiled_copy_e,
                        frgOut,
                        thrOut,
                        nontemporal=USE_NONTEMPORAL,
                        alignment=VEC_ALIGN,
                    )

                    g_cur = g_next
                    b_cur = b_next
            else:
                # Generic path: 2-pass global implementation supporting arbitrary N (incl. tail).
                # For these small/unaligned-N test cases, correctness & robustness matter more than peak perf.
                c_N = flir.const_index(N)
                c_zero = unwrap(arith.constant(compute_type, 0.0))
                thread_sum = unwrap(c_zero)
                thread_sumsq = unwrap(c_zero)

                # Pass1: sum + sumsq
                for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                    c_base = flir.const_index(base_idx_int)
                    idx = mlir_arith.AddIOp(unwrap(c_base), unwrap(tid)).result
                    is_valid = mlir_arith.CmpIOp(mlir_arith.CmpIPredicate.ult, unwrap(idx), unwrap(c_N)).result
                    thread_sum_next = thread_sum
                    thread_sumsq_next = thread_sumsq
                    if is_valid:
                        x_e = memref.load(Input, [unwrap(row), unwrap(idx)])
                        x = unwrap(x_e) if dtype_str == "f32" else mlir_arith.extf(compute_type, unwrap(x_e))
                        x2 = mlir_arith.MulFOp(unwrap(x), unwrap(x), fastmath=fm_fast).result
                        thread_sum_next = mlir_arith.AddFOp(unwrap(thread_sum), unwrap(x), fastmath=fm_fast).result
                        thread_sumsq_next = mlir_arith.AddFOp(unwrap(thread_sumsq), unwrap(x2), fastmath=fm_fast).result
                    thread_sum, thread_sumsq = thread_sum_next, thread_sumsq_next

                sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq, s_sum, s_sumsq)

                inv_n = arith.constant(compute_type, 1.0 / float(N))
                mean = mlir_arith.MulFOp(unwrap(sum_val), unwrap(inv_n), fastmath=fm_fast).result
                mean_sq = mlir_arith.MulFOp(unwrap(sumsq_val), unwrap(inv_n), fastmath=fm_fast).result
                mean2 = mlir_arith.MulFOp(unwrap(mean), unwrap(mean), fastmath=fm_fast).result
                var = mlir_arith.SubFOp(unwrap(mean_sq), unwrap(mean2), fastmath=fm_fast).result
                # Numerical safety: clamp variance to >=0 to avoid NaNs in rsqrt on small-N cases.
                c0_f = arith.constant(compute_type, 0.0)
                is_neg = mlir_arith.CmpFOp(mlir_arith.CmpFPredicate.OLT, unwrap(var), unwrap(c0_f.value)).result
                var = flir.arith.SelectOp(unwrap(is_neg), unwrap(c0_f.value), unwrap(var)).result
                var_eps = mlir_arith.AddFOp(unwrap(var), unwrap(eps.value), fastmath=fm_fast).result
                rstd = math.rsqrt(unwrap(var_eps), fastmath=fm_fast)

                # Pass2: normalize + affine + store
                for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                    c_base = flir.const_index(base_idx_int)
                    idx = mlir_arith.AddIOp(unwrap(c_base), unwrap(tid)).result
                    is_valid = mlir_arith.CmpIOp(mlir_arith.CmpIPredicate.ult, unwrap(idx), unwrap(c_N)).result
                    if is_valid:
                        x_e = memref.load(Input, [unwrap(row), unwrap(idx)])
                        g_e = memref.load(Gamma, [unwrap(idx)])
                        b_e = memref.load(Beta, [unwrap(idx)])
                        x = unwrap(x_e) if dtype_str == "f32" else mlir_arith.extf(compute_type, unwrap(x_e))
                        g = unwrap(g_e) if dtype_str == "f32" else mlir_arith.extf(compute_type, unwrap(g_e))
                        b = unwrap(b_e) if dtype_str == "f32" else mlir_arith.extf(compute_type, unwrap(b_e))
                        diff = mlir_arith.SubFOp(unwrap(x), unwrap(mean), fastmath=fm_fast).result
                        norm = mlir_arith.MulFOp(unwrap(diff), unwrap(rstd), fastmath=fm_fast).result
                        scaled = mlir_arith.MulFOp(unwrap(norm), unwrap(g), fastmath=fm_fast).result
                        y = mlir_arith.AddFOp(unwrap(scaled), unwrap(b), fastmath=fm_fast).result
                        if dtype_str == "bf16":
                            y_e = mlir_arith.truncf(elem_type, unwrap(y))
                        else:
                            y_e = y if dtype_str == "f32" else mlir_arith.truncf(elem_type, unwrap(y))
                        memref.store(unwrap(y_e), Output, [unwrap(row), unwrap(idx)])

    return ctx


