"""LayerNorm kernel builder used by tests.

This file intentionally keeps the kernel builder logic identical to the version
embedded in `tests/kernels/test_layernorm.py` (before factoring) to preserve
codegen and performance. Only test-only helpers/imports are removed.
"""

from flydsl.dialects.ext import flir
from flydsl.dialects.ext.python_control_flow import range_constexpr
from . import reduce as reduce_utils
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator
from _mlir import ir
import _mlir.extras.types as T


KERNEL_NAME = "layernorm"

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
    arch = get_hip_arch()
    # gfx950 supports efficient BF16 pack via v_cvt_pk_bf16_f32.
    # gfx942 does *not* support it and tends to lower f32->bf16 to heavier sequences,
    # so we keep the manual pack there for performance.
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or arch.startswith("gfx95")
    allocator = SmemAllocator(None, arch=arch)

    tile_cols_py = BLOCK_THREADS * VEC_WIDTH

    # Allocate Shared Memory for block reductions (one slot per wave).
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    _state = {}

    class _LayerNorm(flir.MlirModule):
        GPU_MODULE_NAME = "layernorm_module"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{arch}", abi = "500">']

        def init_gpu_module(self):
            elem_type = dtype_to_elem_type(dtype_str)
            compute_type = T.f32()  # compute in fp32 for stability (and to keep bf16 safe on backend)
            _state["elem_type"] = elem_type
            _state["compute_type"] = compute_type
            _state["smem_red_sum"] = allocator.allocate_array(T.f32(), RED_SLOTS)
            _state["smem_red_sumsq"] = allocator.allocate_array(T.f32(), RED_SLOTS)
            allocator.finalize()

        @flir.kernel
        def layernorm_kernel(
            self: flir.T.i64,
            Input: lambda: T.memref(M, N, _state["elem_type"]),
            Gamma: lambda: T.memref(N, _state["elem_type"]),
            Beta: lambda: T.memref(N, _state["elem_type"]),
            Output: lambda: T.memref(M, N, _state["elem_type"]),
        ):
            # Normalize to MLIR index Values early so downstream ops always see `Value`.
            row = flir.const_index(flir.block_idx("x"))
            tid = flir.const_index(flir.thread_idx("x"))

            elem_type = _state["elem_type"]
            compute_type = _state["compute_type"]

            zero_idx = flir.const_index(0)
            n_float = arith.constant(compute_type, float(N))
            eps = arith.constant(compute_type, EPS)
            fm_fast = mlir_arith.FastMathFlags.fast

            base_ptr = allocator.get_base()
            s_sum = _state["smem_red_sum"](base_ptr).get()
            s_sumsq = _state["smem_red_sumsq"](base_ptr).get()
            # FLIR-style tensor views + tiled copies (like elementwise_add_kernel).
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
            thr_copy_e = tiled_copy_e.get_slice((tid))
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

            # Fast-path: keep the original register-row variant for the tuned (N==8192) case.
            if N == (BLOCK_THREADS * VEC_WIDTH * 4):
                num_tiles_py = 4
                # Read Input once into registers (each thread holds 32 fp32 values = 4 vectors),
                # then reuse those registers for reduction + normalize + writeback.
                c_zero = arith.constant(compute_type, 0.0).value
                thread_sum = (c_zero)
                thread_sumsq = (c_zero)
                # Reduce VGPR pressure by caching bf16/f16 payload vectors when possible.
                cache_as_elem = (dtype_str != "f32")
                in_local = []  # bf16/f16: list[vector<VEC_WIDTH x elem_type>]; f32: list[vector<VEC_WIDTH x f32>]

                vec_type_c = ir.VectorType.get([VEC_WIDTH], compute_type)
                vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
                for tile_i in range_constexpr(num_tiles_py):
                    blkIn = gIn[((row), tile_i)]
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
                        x = mlir_arith.extf(vec_type_c, (vec_e))
                    else:
                        x = vec_e
                        in_local.append(x)

                    x2 = mlir_arith.MulFOp((x), (x), fastmath=fm_fast).result
                    red = vector.reduction(compute_type, "add", (x), fastmath=fm_fast)
                    red2 = vector.reduction(compute_type, "add", (x2), fastmath=fm_fast)
                    thread_sum = mlir_arith.AddFOp((thread_sum), (red), fastmath=fm_fast).result
                    thread_sumsq = mlir_arith.AddFOp((thread_sumsq), (red2), fastmath=fm_fast).result

                sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq, s_sum, s_sumsq)

                inv_n = arith.constant(compute_type, 1.0 / float(N))
                mean = mlir_arith.MulFOp((sum_val), (inv_n), fastmath=fm_fast).result
                mean_sq = mlir_arith.MulFOp((sumsq_val), (inv_n), fastmath=fm_fast).result
                mean2 = mlir_arith.MulFOp((mean), (mean), fastmath=fm_fast).result
                var = mlir_arith.SubFOp((mean_sq), (mean2), fastmath=fm_fast).result
                # Numerical safety: with fast-math and cancellation, `var` can become slightly negative
                # and lead to NaNs in rsqrt for small-N cases. Clamp to >= 0 before adding eps.
                c0_f = arith.constant(compute_type, 0.0)
                is_neg = mlir_arith.CmpFOp(mlir_arith.CmpFPredicate.OLT, (var), (c0_f.value)).result
                var = flir.arith.SelectOp((is_neg), (c0_f.value), (var)).result

                var_eps = mlir_arith.AddFOp((var), (eps.value), fastmath=fm_fast).result
                rstd = math.rsqrt((var_eps), fastmath=fm_fast)

                vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
                vec_type_c = ir.VectorType.get([VEC_WIDTH], compute_type)
                mean_splat = vector.splat(vec_type_c, (mean))
                rstd_splat = vector.splat(vec_type_c, (rstd))

                # Pipeline Gamma/Beta loads.
                thread_offset_base = mlir_arith.MulIOp((tid), flir.const_index(VEC_WIDTH)).result
                c_base0 = flir.const_index(0)
                curr_idx0 = mlir_arith.AddIOp((c_base0), (thread_offset_base)).result
                g_e_cur = vector.load(vec_type_e, Gamma, [(curr_idx0)], alignment=VEC_ALIGN)
                b_e_cur = vector.load(vec_type_e, Beta, [(curr_idx0)], alignment=VEC_ALIGN)
                g_cur = g_e_cur if dtype_str == "f32" else mlir_arith.extf(vec_type_c, (g_e_cur))
                b_cur = b_e_cur if dtype_str == "f32" else mlir_arith.extf(vec_type_c, (b_e_cur))

                for tile_i in range_constexpr(num_tiles_py):
                    base_idx_int = tile_i * tile_cols
                    c_base = flir.const_index(base_idx_int)
                    curr_idx = mlir_arith.AddIOp((c_base), (thread_offset_base)).result

                    x = in_local[tile_i]
                    if cache_as_elem:
                        x = mlir_arith.extf(vec_type_c, (x))
                    if tile_i + 1 < num_tiles_py:
                        next_base_idx_int = (tile_i + 1) * tile_cols
                        c_base_next = flir.const_index(next_base_idx_int)
                        next_idx = mlir_arith.AddIOp((c_base_next), (thread_offset_base)).result
                        g_e_next = vector.load(vec_type_e, Gamma, [(next_idx)], alignment=VEC_ALIGN)
                        b_e_next = vector.load(vec_type_e, Beta, [(next_idx)], alignment=VEC_ALIGN)
                        g_next = g_e_next if dtype_str == "f32" else mlir_arith.extf(vec_type_c, (g_e_next))
                        b_next = b_e_next if dtype_str == "f32" else mlir_arith.extf(vec_type_c, (b_e_next))
                    else:
                        g_next = g_cur
                        b_next = b_cur

                    diff = mlir_arith.SubFOp((x), (mean_splat), fastmath=fm_fast).result
                    norm = mlir_arith.MulFOp((diff), (rstd_splat), fastmath=fm_fast).result
                    scaled = mlir_arith.MulFOp((norm), (g_cur), fastmath=fm_fast).result
                    y = mlir_arith.AddFOp((scaled), (b_cur), fastmath=fm_fast).result

                    if dtype_str == "bf16":
                        if USE_HW_CVT_PK_BF16_F32:
                            out_e = mlir_arith.truncf(vec_type_e, (y))
                        else:
                            # Round-to-zero bf16 pack: truncate high 16 bits of f32 (no rounding).
                            # This intentionally trades numerical accuracy for simplicity/speed and
                            # avoids heavyweight bf16 lowering on some toolchains.
                            vec_i32_ty = ir.VectorType.get([VEC_WIDTH], T.i32())
                            vec4_i32_ty = ir.VectorType.get([VEC_WIDTH // 2], T.i32())
                            vec_bf16_ty = ir.VectorType.get([VEC_WIDTH], elem_type)
                            c16_i32 = arith.constant(T.i32(), 16).value
                            c16_i32_v = vector.splat(vec_i32_ty, (c16_i32))

                            u = mlir_arith.bitcast(vec_i32_ty, (y))
                            bf16_bits = mlir_arith.ShRUIOp((u), (c16_i32_v)).result

                            even = vector.shuffle(bf16_bits, bf16_bits, mask=[0, 2, 4, 6])
                            odd = vector.shuffle(bf16_bits, bf16_bits, mask=[1, 3, 5, 7])
                            odd_sh = mlir_arith.ShLIOp((odd), (vector.splat(vec4_i32_ty, (c16_i32)))).result
                            packed = mlir_arith.OrIOp((even), (odd_sh)).result
                            out_e = vector.bitcast(vec_bf16_ty, (packed))
                    else:
                        out_e = y if dtype_str == "f32" else mlir_arith.truncf(vec_type_e, (y))

                    blkOut = gOut[((row), tile_i)]
                    thrOut = thr_copy_e.partition_S(blkOut)
                    frgOut = flir.make_fragment_like(thrOut, elem_type)
                    vector.store((out_e), frgOut.memref, [c0_idx, c0_idx], alignment=VEC_ALIGN)
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
                c_zero = (arith.constant(compute_type, 0.0))
                thread_sum = (c_zero)
                thread_sumsq = (c_zero)

                # Pass1: sum + sumsq
                for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                    c_base = flir.const_index(base_idx_int)
                    idx = mlir_arith.AddIOp((c_base), (tid)).result
                    is_valid = mlir_arith.CmpIOp(mlir_arith.CmpIPredicate.ult, (idx), (c_N)).result
                    thread_sum_next = thread_sum
                    thread_sumsq_next = thread_sumsq
                    if is_valid:
                        x_e = memref.load(Input, [(row), (idx)])
                        x = (x_e) if dtype_str == "f32" else mlir_arith.extf(compute_type, (x_e))
                        x2 = mlir_arith.MulFOp((x), (x), fastmath=fm_fast).result
                        thread_sum_next = mlir_arith.AddFOp((thread_sum), (x), fastmath=fm_fast).result
                        thread_sumsq_next = mlir_arith.AddFOp((thread_sumsq), (x2), fastmath=fm_fast).result
                    thread_sum, thread_sumsq = thread_sum_next, thread_sumsq_next

                sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq, s_sum, s_sumsq)

                inv_n = arith.constant(compute_type, 1.0 / float(N))
                mean = mlir_arith.MulFOp((sum_val), (inv_n), fastmath=fm_fast).result
                mean_sq = mlir_arith.MulFOp((sumsq_val), (inv_n), fastmath=fm_fast).result
                mean2 = mlir_arith.MulFOp((mean), (mean), fastmath=fm_fast).result
                var = mlir_arith.SubFOp((mean_sq), (mean2), fastmath=fm_fast).result
                # Numerical safety: clamp variance to >=0 to avoid NaNs in rsqrt on small-N cases.
                c0_f = arith.constant(compute_type, 0.0)
                is_neg = mlir_arith.CmpFOp(mlir_arith.CmpFPredicate.OLT, (var), (c0_f.value)).result
                var = flir.arith.SelectOp((is_neg), (c0_f.value), (var)).result
                var_eps = mlir_arith.AddFOp((var), (eps.value), fastmath=fm_fast).result
                rstd = math.rsqrt((var_eps), fastmath=fm_fast)

                # Pass2: normalize + affine + store
                for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                    c_base = flir.const_index(base_idx_int)
                    idx = mlir_arith.AddIOp((c_base), (tid)).result
                    is_valid = mlir_arith.CmpIOp(mlir_arith.CmpIPredicate.ult, (idx), (c_N)).result
                    if is_valid:
                        x_e = memref.load(Input, [(row), (idx)])
                        g_e = memref.load(Gamma, [(idx)])
                        b_e = memref.load(Beta, [(idx)])
                        x = (x_e) if dtype_str == "f32" else mlir_arith.extf(compute_type, (x_e))
                        g = (g_e) if dtype_str == "f32" else mlir_arith.extf(compute_type, (g_e))
                        b = (b_e) if dtype_str == "f32" else mlir_arith.extf(compute_type, (b_e))
                        diff = mlir_arith.SubFOp((x), (mean), fastmath=fm_fast).result
                        norm = mlir_arith.MulFOp((diff), (rstd), fastmath=fm_fast).result
                        scaled = mlir_arith.MulFOp((norm), (g), fastmath=fm_fast).result
                        y = mlir_arith.AddFOp((scaled), (b), fastmath=fm_fast).result
                        if dtype_str == "bf16":
                            y_e = mlir_arith.truncf(elem_type, (y))
                        else:
                            y_e = y if dtype_str == "f32" else mlir_arith.truncf(elem_type, (y))
                        memref.store((y_e), Output, [(row), (idx)])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            Input: lambda: T.memref(M, N, _state["elem_type"]),
            Gamma: lambda: T.memref(N, _state["elem_type"]),
            Beta: lambda: T.memref(N, _state["elem_type"]),
            Output: lambda: T.memref(M, N, _state["elem_type"]),
        ):
            c1 = (flir.arith_ext.index(1))
            gx = (flir.arith_ext.index(M))
            bx = (flir.arith_ext.index(BLOCK_THREADS))
            flir.gpu_ext.LaunchFuncOp(
                ["layernorm_module", "layernorm_kernel"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[Input, Gamma, Beta, Output],
            )

    return _LayerNorm()


