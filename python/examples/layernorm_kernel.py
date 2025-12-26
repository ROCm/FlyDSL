"""LayerNorm kernel builder used by tests.

This file intentionally keeps the kernel builder logic identical to the version
embedded in `tests/python/gpu/test_layernorm.py` (before factoring) to preserve
codegen and performance. Only test-only helpers/imports are removed.
"""

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.dialects.ext import rocir
from . import reduce as reduce_utils
from rocdsl.runtime.hip_util import get_hip_arch
from rocdsl.utils import SmemAllocator
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


# Expose modules through Rocir interface (keep behavior/perf, avoid mlir.* imports).
gpu = rocir.gpu_ext
scf = rocir.scf_ext
# Keep arith as the raw dialect module here (this file uses arith.constant(Type, value) form).
arith = rocir.arith
mlir_arith = rocir.arith
memref = rocir.memref
vector = rocir.vector
math = rocir.math


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

    # Register-row variant: each thread keeps its full slice in registers.
    # With current tuning, each thread owns exactly 32 elements:
    #   N == BLOCK_THREADS * VEC_WIDTH * 4  == 256 * 8 * 4 == 8192
    tile_cols_py = BLOCK_THREADS * VEC_WIDTH
    if N % tile_cols_py != 0:
        raise ValueError(f"N must be divisible by BLOCK_THREADS*VEC_WIDTH ({tile_cols_py}), got N={N}")
    num_tiles_py = N // tile_cols_py
    if num_tiles_py != 4:
        raise ValueError(
            f"Expected per-thread 32 floats (tiles==4). Got N={N}, BLOCK_THREADS={BLOCK_THREADS}, "
            f"VEC_WIDTH={VEC_WIDTH} => tiles={num_tiles_py}"
        )

    # Allocate Shared Memory for block reductions (one slot per wave)
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
            row = rocir.block_idx("x")
            tid = rocir.thread_idx("x")

            zero_idx = rocir.const_index(0)
            n_float = arith.constant(compute_type, float(N))
            eps = arith.constant(compute_type, EPS)
            fm_fast = mlir_arith.FastMathFlags.fast

            base_ptr = allocator.get_base()
            s_sum = smem_red_sum(base_ptr).get()
            s_sumsq = smem_red_sumsq(base_ptr).get()
            # Rocir-style tensor views + tiled copies (like elementwise_add_kernel).
            c0_idx = rocir.const_index(0)
            tile_cols = BLOCK_THREADS * VEC_WIDTH  # python int
            tensor_In = rocir.make_tensor(Input, shape=(M, N), strides=(N, 1))
            tensor_Out = rocir.make_tensor(Output, shape=(M, N), strides=(N, 1))
            tensor_Gamma = rocir.make_tensor(Gamma, shape=(N,), strides=(1,))
            tensor_Beta = rocir.make_tensor(Beta, shape=(N,), strides=(1,))
            gIn = rocir.zipped_divide(tensor_In, (1, tile_cols))
            gOut = rocir.zipped_divide(tensor_Out, (1, tile_cols))

            thr_layout = rocir.make_ordered_layout((1, BLOCK_THREADS), order=(1, 0))
            val_layout = rocir.make_ordered_layout((1, VEC_WIDTH), order=(1, 0))
            copy_atom_e = rocir.make_copy_atom(elem_type, vector_size=VEC_WIDTH)
            tiled_copy_e = rocir.make_tiled_copy_tv(
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
                rocir=rocir,
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
                rocir=rocir,
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
                bf16_bits = rocir.arith.SelectOp(unwrap(is_nan), unwrap(bf16_bits_nan), unwrap(bf16_bits_rne)).result

                even = vector.shuffle(bf16_bits, bf16_bits, mask=[0, 2, 4, 6])
                odd = vector.shuffle(bf16_bits, bf16_bits, mask=[1, 3, 5, 7])
                odd_sh = mlir_arith.ShLIOp(unwrap(odd), unwrap(vector.splat(vec4_i32_ty, unwrap(c16_i32)))).result
                packed = mlir_arith.OrIOp(unwrap(even), unwrap(odd_sh)).result
                return vector.bitcast(vec_bf16_ty, unwrap(packed))

            # Register row cache:
            # Read Input once into registers (each thread holds 32 fp32 values = 4 vectors),
            # then reuse those registers for reduction + normalize + writeback.
            c_zero = arith.constant(compute_type, 0.0).value
            thread_sum = unwrap(c_zero)
            thread_sumsq = unwrap(c_zero)
            # Direction D: reduce VGPR pressure by caching bf16/f16 payload vectors when possible
            # (instead of caching already-extended f32 vectors).
            cache_as_elem = (dtype_str != "f32")
            in_local = []  # bf16/f16: list[vector<VEC_WIDTH x elem_type>]; f32: list[vector<VEC_WIDTH x f32>]

            vec_type_c = ir.VectorType.get([VEC_WIDTH], compute_type)
            vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
            for tile_i in range(num_tiles_py):
                blkIn = gIn[(unwrap(row), tile_i)]
                thrIn = thr_copy_e.partition_S(blkIn)
                frgIn = rocir.make_fragment_like(thrIn, elem_type)
                vec_e = rocir.copy(
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

            var_eps = mlir_arith.AddFOp(unwrap(var), unwrap(eps.value), fastmath=fm_fast).result
            # Keep fast-math behavior consistent across the full LN pipeline (incl. rsqrt).
            rstd = math.rsqrt(unwrap(var_eps), fastmath=fm_fast)

            # Normalize + affine + store (vectorized), reusing in_local registers.
            vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
            vec_type_c = ir.VectorType.get([VEC_WIDTH], compute_type)
            mean_splat = vector.splat(vec_type_c, unwrap(mean))
            rstd_splat = vector.splat(vec_type_c, unwrap(rstd))

            # Direction C: software-pipeline Gamma/Beta loads to hide memory latency.
            # Load tile0 gamma/beta first, then in each iteration prefetch tile_{i+1}
            # before computing tile_i.
            thread_offset_base = mlir_arith.MulIOp(unwrap(tid), rocir.const_index(VEC_WIDTH)).result
            c_base0 = rocir.const_index(0)
            curr_idx0 = mlir_arith.AddIOp(unwrap(c_base0), unwrap(thread_offset_base)).result
            g_e_cur = vector.load(vec_type_e, Gamma, [unwrap(curr_idx0)], alignment=VEC_ALIGN)
            b_e_cur = vector.load(vec_type_e, Beta, [unwrap(curr_idx0)], alignment=VEC_ALIGN)
            g_cur = g_e_cur if dtype_str == "f32" else mlir_arith.extf(vec_type_c, unwrap(g_e_cur))
            b_cur = b_e_cur if dtype_str == "f32" else mlir_arith.extf(vec_type_c, unwrap(b_e_cur))

            for tile_i in range(num_tiles_py):
                base_idx_int = tile_i * tile_cols
                c_base = rocir.const_index(base_idx_int)
                curr_idx = mlir_arith.AddIOp(unwrap(c_base), unwrap(thread_offset_base)).result

                x = in_local[tile_i]
                if cache_as_elem:
                    x = mlir_arith.extf(vec_type_c, unwrap(x))
                # Prefetch next tile's Gamma/Beta early (loads are cache-friendly).
                if tile_i + 1 < num_tiles_py:
                    next_base_idx_int = (tile_i + 1) * tile_cols
                    c_base_next = rocir.const_index(next_base_idx_int)
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
                    # Arch-dependent BF16 pack:
                    # - gfx942: use manual RNE pack to avoid heavy lowering.
                    # - gfx950+: rely on hardware v_cvt_pk_bf16_f32 via truncf lowering.
                    if USE_HW_CVT_PK_BF16_F32:
                        out_e = mlir_arith.truncf(vec_type_e, unwrap(y))
                    else:
                        out_e = bf16_pack_vec8_rne(y)
                else:
                    out_e = y if dtype_str == "f32" else mlir_arith.truncf(vec_type_e, unwrap(y))

                blkOut = gOut[(unwrap(row), tile_i)]
                thrOut = thr_copy_e.partition_S(blkOut)
                frgOut = rocir.make_fragment_like(thrOut, elem_type)
                vector.store(unwrap(out_e), frgOut.memref, [c0_idx, c0_idx], alignment=VEC_ALIGN)
                rocir.copy(
                    tiled_copy_e,
                    frgOut,
                    thrOut,
                    nontemporal=USE_NONTEMPORAL,
                    alignment=VEC_ALIGN,
                )

                # Advance the pipelined gamma/beta.
                g_cur = g_next
                b_cur = b_next

    return ctx


