"""LayerNorm kernel builder used by tests.

This file intentionally keeps the kernel builder logic identical to the version
embedded in `tests/python/gpu/test_layernorm.py` (before factoring) to preserve
codegen and performance. Only test-only helpers/imports are removed.
"""

from pyflir.dialects.ext import flir
from pyflir.dialects.ext import scf
from pyflir.dialects.ext.python_control_flow import range_constexpr
from . import reduce as reduce_utils
from pyflir.runtime.device import get_rocm_arch
from pyflir.utils import SmemAllocator
from _mlir import ir
import _mlir.extras.types as T


KERNEL_NAME = "layernorm"


def unwrap(v):
    if hasattr(v, "value"):
        return v.value
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
    arch = get_rocm_arch()
    allocator = SmemAllocator(None, arch=arch)
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    _state = {}

    class _LayerNorm(flir.MlirModule):
        GPU_MODULE_NAME = "layernorm_module"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{arch}">']

        def init_gpu_module(self):
            elem_type = dtype_to_elem_type(dtype_str)
            compute_type = T.f32()  # compute in fp32 for stability (and to keep bf16 safe on backend)
            _state["elem_type"] = elem_type
            _state["compute_type"] = compute_type
            _state["smem_red_sum"] = allocator.allocate_array(T.f32(), RED_SLOTS)
            _state["smem_red_sumsq"] = allocator.allocate_array(T.f32(), RED_SLOTS)
            # Represent LDS row cache as a 2D (1, N) memref so tensor indexing uses 2 indices.
            _state["smem_row"] = allocator.allocate_tensor((1, N), elem_type)
            allocator.finalize()

        @flir.kernel
        def layernorm_kernel(
            self: flir.T.i64,
            Input: lambda: T.memref(M, N, _state["elem_type"]),
            Gamma: lambda: T.memref(N, _state["elem_type"]),
            Beta: lambda: T.memref(N, _state["elem_type"]),
            Output: lambda: T.memref(M, N, _state["elem_type"]),
        ):
            row = flir.block_idx("x")
            tid = flir.thread_idx("x")
            elem_type = _state["elem_type"]
            compute_type = _state["compute_type"]

            zero_idx = flir.const_index(0)
            n_float = arith.constant(compute_type, float(N))
            eps = arith.constant(compute_type, EPS)
            fm_fast = mlir_arith.FastMathFlags.fast

            base_ptr = allocator.get_base()
            s_sum = _state["smem_red_sum"](base_ptr).get()
            s_sumsq = _state["smem_red_sumsq"](base_ptr).get()
            s_row = _state["smem_row"](base_ptr).get()
            # Flir-style tensor views + tiled copies (like elementwise_add_kernel).
            c0_idx = flir.const_index(0)
            tile_cols = BLOCK_THREADS * VEC_WIDTH  # python int
            tensor_In = flir.make_tensor(Input, shape=(M, N), strides=(N, 1))
            tensor_Out = flir.make_tensor(Output, shape=(M, N), strides=(N, 1))
            tensor_Gamma = flir.make_tensor(Gamma, shape=(N,), strides=(1,))
            tensor_Beta = flir.make_tensor(Beta, shape=(N,), strides=(1,))
            # Represent LDS row cache as a 2D tensor view (1, N) over a 1D memref.
            tensor_S = flir.make_tensor(s_row, shape=(1, N), strides=(N, 1))
            gIn = flir.zipped_divide(tensor_In, (1, tile_cols))
            gOut = flir.zipped_divide(tensor_Out, (1, tile_cols))
            gS = flir.zipped_divide(tensor_S, (1, tile_cols))

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

            def bf16_pack_vec8_rne_no_nan(vec_f32):
                # Manual bf16 pack: RNE rounding, intentionally *no* NaN/range fixups.
                # This avoids the heavy cmp/cndmask sequences in default lowering on gfx942.
                vec_i32_ty = ir.VectorType.get([VEC_WIDTH], T.i32())
                vec4_i32_ty = ir.VectorType.get([VEC_WIDTH // 2], T.i32())
                vec_bf16_ty = ir.VectorType.get([VEC_WIDTH], elem_type)

                c16_i32 = arith.constant(T.i32(), 16).value
                c7fff_i32 = arith.constant(T.i32(), 0x7FFF).value
                c1_i32 = arith.constant(T.i32(), 1).value

                c16_i32_v = vector.splat(vec_i32_ty, unwrap(c16_i32))
                c7fff_i32_v = vector.splat(vec_i32_ty, unwrap(c7fff_i32))
                c1_i32_v = vector.splat(vec_i32_ty, unwrap(c1_i32))

                u = mlir_arith.bitcast(vec_i32_ty, unwrap(vec_f32))
                hi = mlir_arith.ShRUIOp(unwrap(u), unwrap(c16_i32_v)).result
                lsb = mlir_arith.AndIOp(unwrap(hi), unwrap(c1_i32_v)).result
                bias = mlir_arith.AddIOp(unwrap(c7fff_i32_v), unwrap(lsb)).result
                u_round = mlir_arith.AddIOp(unwrap(u), unwrap(bias)).result
                bf16_bits = mlir_arith.ShRUIOp(unwrap(u_round), unwrap(c16_i32_v)).result

                even = vector.shuffle(bf16_bits, bf16_bits, mask=[0, 2, 4, 6])
                odd = vector.shuffle(bf16_bits, bf16_bits, mask=[1, 3, 5, 7])
                odd_sh = mlir_arith.ShLIOp(unwrap(odd), unwrap(vector.splat(vec4_i32_ty, unwrap(c16_i32)))).result
                packed = mlir_arith.OrIOp(unwrap(even), unwrap(odd_sh)).result
                return vector.bitcast(vec_bf16_ty, unwrap(packed))

            # Pass0: global -> LDS row cache (1-pass global read)
            c_zero = arith.constant(compute_type, 0.0).value
            FULL_TILES = (N % (BLOCK_THREADS * VEC_WIDTH) == 0)

            # If N is fully tiled, accumulate sum/sumsq directly from the global-load values
            # (so bf16/f16 unpack happens only once). Otherwise, fall back to Pass1 LDS read.
            thread_sum = unwrap(c_zero)
            thread_sumsq = unwrap(c_zero)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS * VEC_WIDTH):
                c_base = flir.const_index(base_idx_int)
                thread_offset_base = mlir_arith.MulIOp(unwrap(tid), flir.const_index(VEC_WIDTH)).result
                curr_idx = mlir_arith.AddIOp(unwrap(c_base), unwrap(thread_offset_base)).result

                tile_safe = (base_idx_int + BLOCK_THREADS * VEC_WIDTH) <= N
                if tile_safe:
                    tile_i = base_idx_int // tile_cols  # python int
                    blkIn = gIn[(unwrap(row), tile_i)]
                    blkS = gS[(0, tile_i)]
                    thrIn = thr_copy_e.partition_S(blkIn)
                    thrS = thr_copy_e.partition_S(blkS)
                    vec_e = flir.copy(
                        tiled_copy_e,
                        thrIn,
                        thrS,
                        nontemporal=USE_NONTEMPORAL,
                        alignment=VEC_ALIGN,
                        return_vector=True,
                    )

                    if FULL_TILES:
                        vec_type_c = ir.VectorType.get([VEC_WIDTH], compute_type)
                        vec = vec_e if dtype_str == "f32" else mlir_arith.extf(vec_type_c, unwrap(vec_e))
                        vec2 = mlir_arith.MulFOp(unwrap(vec), unwrap(vec), fastmath=fm_fast).result
                        red = vector.reduction(compute_type, "add", unwrap(vec), fastmath=fm_fast)
                        red2 = vector.reduction(compute_type, "add", unwrap(vec2), fastmath=fm_fast)
                        thread_sum = mlir_arith.AddFOp(unwrap(thread_sum), unwrap(red), fastmath=fm_fast).result
                        thread_sumsq = mlir_arith.AddFOp(unwrap(thread_sumsq), unwrap(red2), fastmath=fm_fast).result
                else:
                    c_N = flir.const_index(N)
                    for k in range_constexpr(VEC_WIDTH):
                        c_k = flir.const_index(k)
                        idx_k = mlir_arith.AddIOp(unwrap(curr_idx), unwrap(c_k)).result
                        is_valid = mlir_arith.CmpIOp(
                            mlir_arith.CmpIPredicate.ult, unwrap(idx_k), unwrap(c_N)
                        ).result
                        if is_valid:
                            v_e = tensor_In[(unwrap(row), unwrap(idx_k))]
                            tensor_S[(unwrap(c0_idx), unwrap(idx_k))] = unwrap(v_e)

            gpu.barrier()

            # Pass1: sum / sumsq (from LDS row cache)
            # If fully tiled, we've already accumulated from Pass0 global loads.
            if not FULL_TILES:
                thread_sum = unwrap(c_zero)
                thread_sumsq = unwrap(c_zero)

                for base_idx_int in range_constexpr(0, N, BLOCK_THREADS * VEC_WIDTH):
                    c_base = flir.const_index(base_idx_int)
                    thread_offset_base = mlir_arith.MulIOp(unwrap(tid), flir.const_index(VEC_WIDTH)).result
                    curr_idx = mlir_arith.AddIOp(unwrap(c_base), unwrap(thread_offset_base)).result

                    tile_safe = (base_idx_int + BLOCK_THREADS * VEC_WIDTH) <= N
                    if tile_safe:
                        vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
                        vec_e = vector.load(
                            vec_type_e, s_row, [unwrap(c0_idx), unwrap(curr_idx)], alignment=VEC_ALIGN
                        )
                        vec_type_c = ir.VectorType.get([VEC_WIDTH], compute_type)
                        vec = vec_e if dtype_str == "f32" else mlir_arith.extf(vec_type_c, unwrap(vec_e))
                        vec2 = mlir_arith.MulFOp(unwrap(vec), unwrap(vec), fastmath=fm_fast).result
                        red = vector.reduction(compute_type, "add", unwrap(vec), fastmath=fm_fast)
                        red2 = vector.reduction(compute_type, "add", unwrap(vec2), fastmath=fm_fast)
                        thread_sum = mlir_arith.AddFOp(unwrap(thread_sum), unwrap(red), fastmath=fm_fast).result
                        thread_sumsq = mlir_arith.AddFOp(unwrap(thread_sumsq), unwrap(red2), fastmath=fm_fast).result
                    else:
                        c_N = flir.const_index(N)
                        for k in range_constexpr(VEC_WIDTH):
                            c_k = flir.const_index(k)
                            idx_k = mlir_arith.AddIOp(unwrap(curr_idx), unwrap(c_k)).result
                            is_valid = mlir_arith.CmpIOp(
                                mlir_arith.CmpIPredicate.ult, unwrap(idx_k), unwrap(c_N)
                            ).result
                            if is_valid:
                                v_e = tensor_S[(unwrap(c0_idx), unwrap(idx_k))]
                            else:
                                v_e = arith.constant(elem_type, 0.0).value
                            v = unwrap(v_e) if dtype_str == "f32" else mlir_arith.extf(compute_type, unwrap(v_e))
                            v2 = mlir_arith.MulFOp(unwrap(v), unwrap(v), fastmath=fm_fast).result
                            thread_sum = mlir_arith.AddFOp(unwrap(thread_sum), unwrap(v), fastmath=fm_fast).result
                            thread_sumsq = mlir_arith.AddFOp(unwrap(thread_sumsq), unwrap(v2), fastmath=fm_fast).result

            sum_val = block_reduce_add(thread_sum, s_sum)
            sumsq_val = block_reduce_add(thread_sumsq, s_sumsq)

            mean = mlir_arith.DivFOp(unwrap(sum_val), unwrap(n_float.value), fastmath=fm_fast).result
            mean_sq = mlir_arith.DivFOp(unwrap(sumsq_val), unwrap(n_float.value), fastmath=fm_fast).result
            mean2 = mlir_arith.MulFOp(unwrap(mean), unwrap(mean), fastmath=fm_fast).result
            var = mlir_arith.SubFOp(unwrap(mean_sq), unwrap(mean2), fastmath=fm_fast).result

            var_eps = mlir_arith.AddFOp(unwrap(var), unwrap(eps.value), fastmath=fm_fast).result
            rstd = math.rsqrt(unwrap(var_eps))

            # Pass2: normalize + affine + store (vectorized)
            vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
            vec_type_c = ir.VectorType.get([VEC_WIDTH], compute_type)
            mean_splat = vector.splat(vec_type_c, unwrap(mean))
            rstd_splat = vector.splat(vec_type_c, unwrap(rstd))

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS * VEC_WIDTH):
                c_base = flir.const_index(base_idx_int)
                thread_offset_base = mlir_arith.MulIOp(unwrap(tid), flir.const_index(VEC_WIDTH)).result
                curr_idx = mlir_arith.AddIOp(unwrap(c_base), unwrap(thread_offset_base)).result

                tile_safe = (base_idx_int + BLOCK_THREADS * VEC_WIDTH) <= N
                if tile_safe:
                    x_e = vector.load(
                        vec_type_e, s_row, [unwrap(c0_idx), unwrap(curr_idx)], alignment=VEC_ALIGN
                    )
                    # Gamma/Beta are reused across many blocks: do NOT use nontemporal here.
                    # Let caches work for them.
                    g_e = vector.load(vec_type_e, Gamma, [unwrap(curr_idx)], alignment=VEC_ALIGN)
                    b_e = vector.load(vec_type_e, Beta, [unwrap(curr_idx)], alignment=VEC_ALIGN)

                    x = x_e if dtype_str == "f32" else mlir_arith.extf(vec_type_c, unwrap(x_e))
                    g = g_e if dtype_str == "f32" else mlir_arith.extf(vec_type_c, unwrap(g_e))
                    b = b_e if dtype_str == "f32" else mlir_arith.extf(vec_type_c, unwrap(b_e))

                    diff = mlir_arith.SubFOp(unwrap(x), unwrap(mean_splat), fastmath=fm_fast).result
                    norm = mlir_arith.MulFOp(unwrap(diff), unwrap(rstd_splat), fastmath=fm_fast).result
                    scaled = mlir_arith.MulFOp(unwrap(norm), unwrap(g), fastmath=fm_fast).result
                    y = mlir_arith.AddFOp(unwrap(scaled), unwrap(b), fastmath=fm_fast).result

                    if dtype_str == "bf16":
                        out_bf16 = bf16_pack_vec8_rne_no_nan(y)
                        tile_i = base_idx_int // tile_cols  # python int
                        blkOut = gOut[(unwrap(row), tile_i)]
                        thrOut = thr_copy_e.partition_S(blkOut)
                        frgOut = flir.make_fragment_like(thrOut, elem_type)
                        vector.store(out_bf16, frgOut.memref, [c0_idx, c0_idx], alignment=VEC_ALIGN)
                        flir.copy(
                            tiled_copy_e,
                            frgOut,
                            thrOut,
                            nontemporal=USE_NONTEMPORAL,
                            alignment=VEC_ALIGN,
                        )
                    else:
                        y_e = y if dtype_str == "f32" else mlir_arith.truncf(vec_type_e, unwrap(y))
                        tile_i = base_idx_int // tile_cols  # python int
                        blkOut = gOut[(unwrap(row), tile_i)]
                        thrOut = thr_copy_e.partition_S(blkOut)
                        frgOut = flir.make_fragment_like(thrOut, elem_type)
                        vector.store(unwrap(y_e), frgOut.memref, [c0_idx, c0_idx], alignment=VEC_ALIGN)
                        flir.copy(
                            tiled_copy_e,
                            frgOut,
                            thrOut,
                            nontemporal=USE_NONTEMPORAL,
                            alignment=VEC_ALIGN,
                        )
                else:
                    c_N = flir.const_index(N)
                    # scalar tail
                    for k in range_constexpr(VEC_WIDTH):
                        c_k = flir.const_index(k)
                        idx_k = mlir_arith.AddIOp(unwrap(curr_idx), unwrap(c_k)).result
                        is_valid = mlir_arith.CmpIOp(
                            mlir_arith.CmpIPredicate.ult, unwrap(idx_k), unwrap(c_N)
                        ).result
                        if is_valid:
                            x_e = tensor_S[(unwrap(c0_idx), unwrap(idx_k))]
                            g_e = tensor_Gamma[unwrap(idx_k)]
                            b_e = tensor_Beta[unwrap(idx_k)]
                            x = unwrap(x_e) if dtype_str == "f32" else mlir_arith.extf(compute_type, unwrap(x_e))
                            g = unwrap(g_e) if dtype_str == "f32" else mlir_arith.extf(compute_type, unwrap(g_e))
                            b = unwrap(b_e) if dtype_str == "f32" else mlir_arith.extf(compute_type, unwrap(b_e))
                            diff = mlir_arith.SubFOp(unwrap(x), unwrap(mean), fastmath=fm_fast).result
                            norm = mlir_arith.MulFOp(unwrap(diff), unwrap(rstd), fastmath=fm_fast).result
                            y = mlir_arith.AddFOp(
                                unwrap(mlir_arith.MulFOp(unwrap(norm), unwrap(g), fastmath=fm_fast).result),
                                unwrap(b),
                                fastmath=fm_fast,
                            ).result
                            y_e = y if dtype_str == "f32" else mlir_arith.truncf(elem_type, unwrap(y))
                            tensor_Out[(unwrap(row), unwrap(idx_k))] = unwrap(y_e)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            Input: lambda: T.memref(M, N, _state["elem_type"]),
            Gamma: lambda: T.memref(N, _state["elem_type"]),
            Beta: lambda: T.memref(N, _state["elem_type"]),
            Output: lambda: T.memref(M, N, _state["elem_type"]),
        ):
            c1 = flir.arith_ext.index(1).value
            gx = flir.arith_ext.index(M).value
            bx = flir.arith_ext.index(BLOCK_THREADS).value
            flir.gpu_ext.LaunchFuncOp(
                ["layernorm_module", "layernorm_kernel"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[Input, Gamma, Beta, Output],
            )

    return _LayerNorm()


