"""RMSNorm kernel builder used by tests.

This file intentionally keeps the kernel builder logic identical to the version
embedded in `tests/python/gpu/test_rmsnorm.py` (before factoring) to preserve
codegen and performance. Only test-only helpers/imports are removed.
"""

from rocdsl.dialects.ext import rocir
from rocdsl.dialects.ext.python_control_flow import range_constexpr
from . import reduce as reduce_utils
from rocdsl.runtime.hip_util import get_hip_arch
from rocdsl.utils import SmemAllocator
from _mlir import ir
import _mlir.extras.types as T


KERNEL_NAME = "rmsnorm"


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


def build_rmsnorm_module(M: int, N: int, dtype_str: str):
    arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=arch)
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    _state = {}

    class _RMSNorm(rocir.MlirModule):
        GPU_MODULE_NAME = "rmsnorm_module"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{arch}", abi = "500">']

        def init_gpu_module(self):
            elem_type = dtype_to_elem_type(dtype_str)
            compute_type = T.f32()
            _state["elem_type"] = elem_type
            _state["compute_type"] = compute_type
            _state["smem_red"] = allocator.allocate_array(T.f32(), RED_SLOTS)
            # Represent LDS row cache as a 2D (1, N) memref so tensor indexing uses 2 indices.
            _state["smem_row"] = allocator.allocate_tensor((1, N), elem_type)
            allocator.finalize()

        @rocir.kernel
        def rmsnorm_kernel(
            self: rocir.T.i64,
            Input: lambda: T.memref(M, N, _state["elem_type"]),
            Gamma: lambda: T.memref(N, _state["elem_type"]),
            Output: lambda: T.memref(M, N, _state["elem_type"]),
        ):
            row = rocir.block_idx("x")
            tid = rocir.thread_idx("x")
            elem_type = _state["elem_type"]
            compute_type = _state["compute_type"]

            zero_idx = rocir.const_index(0)
            n_float = arith.constant(compute_type, float(N))
            eps = arith.constant(compute_type, EPS)
            fm_fast = rocir.arith.FastMathFlags.fast

            base_ptr = allocator.get_base()
            s_red = _state["smem_red"](base_ptr).get()
            s_row = _state["smem_row"](base_ptr).get()
            # Rocir-style tensor views + tiled copies (like elementwise_add_kernel).
            c0_idx = rocir.const_index(0)
            tile_cols = BLOCK_THREADS * VEC_WIDTH  # python int
            tensor_In = rocir.make_tensor(Input, shape=(M, N), strides=(N, 1))
            tensor_Out = rocir.make_tensor(Output, shape=(M, N), strides=(N, 1))
            tensor_Gamma = rocir.make_tensor(Gamma, shape=(N,), strides=(1,))
            tensor_S = rocir.make_tensor(s_row, shape=(1, N), strides=(N, 1))
            gIn = rocir.zipped_divide(tensor_In, (1, tile_cols))
            gOut = rocir.zipped_divide(tensor_Out, (1, tile_cols))
            gS = rocir.zipped_divide(tensor_S, (1, tile_cols))

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
                arith_ops=rocir.arith,
                rocir=rocir,
                T=T,
                ir=ir,
                zero_idx=zero_idx,
            )

            # Pass0: global -> LDS row cache (1-pass global read)
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS * VEC_WIDTH):
                c_base = rocir.const_index(base_idx_int)
                thread_offset_base = rocir.arith.MulIOp(unwrap(tid), rocir.const_index(VEC_WIDTH)).result
                curr_idx = rocir.arith.AddIOp(unwrap(c_base), unwrap(thread_offset_base)).result

                tile_safe = (base_idx_int + BLOCK_THREADS * VEC_WIDTH) <= N
                if tile_safe:
                    tile_i = base_idx_int // tile_cols  # python int
                    blkIn = gIn[(unwrap(row), tile_i)]
                    blkS = gS[(0, tile_i)]
                    thrIn = thr_copy_e.partition_S(blkIn)
                    thrS = thr_copy_e.partition_S(blkS)
                    rocir.copy(
                        tiled_copy_e,
                        thrIn,
                        thrS,
                        nontemporal=USE_NONTEMPORAL,
                        alignment=VEC_ALIGN,
                    )
                else:
                    c_N = rocir.const_index(N)
                    for k in range_constexpr(VEC_WIDTH):
                        c_k = rocir.const_index(k)
                        idx_k = rocir.arith.AddIOp(unwrap(curr_idx), unwrap(c_k)).result
                        is_valid = rocir.arith.CmpIOp(rocir.arith.CmpIPredicate.ult, unwrap(idx_k), unwrap(c_N)).result
                        if is_valid:
                            v_e = tensor_In[(unwrap(row), unwrap(idx_k))]
                            tensor_S[(unwrap(c0_idx), unwrap(idx_k))] = unwrap(v_e)

            gpu.barrier()

            # Pass1: sumsq (from LDS row cache)
            c_zero = arith.constant(compute_type, 0.0).value
            thread_sumsq = unwrap(c_zero)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS * VEC_WIDTH):
                c_base = rocir.const_index(base_idx_int)
                thread_offset_base = rocir.arith.MulIOp(unwrap(tid), rocir.const_index(VEC_WIDTH)).result
                curr_idx = rocir.arith.AddIOp(unwrap(c_base), unwrap(thread_offset_base)).result

                tile_safe = (base_idx_int + BLOCK_THREADS * VEC_WIDTH) <= N
                if tile_safe:
                    vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
                    vec_e = vector.load(
                        vec_type_e, s_row, [unwrap(c0_idx), unwrap(curr_idx)], alignment=VEC_ALIGN
                    )
                    vec_type_c = ir.VectorType.get([VEC_WIDTH], compute_type)
                    vec = vec_e if dtype_str == "f32" else rocir.arith.extf(vec_type_c, unwrap(vec_e))
                    vec2 = rocir.arith.MulFOp(unwrap(vec), unwrap(vec), fastmath=fm_fast).result
                    red2 = vector.reduction(compute_type, "add", unwrap(vec2), fastmath=fm_fast)
                    thread_sumsq = rocir.arith.AddFOp(unwrap(thread_sumsq), unwrap(red2), fastmath=fm_fast).result
                else:
                    c_N = rocir.const_index(N)
                    for k in range_constexpr(VEC_WIDTH):
                        c_k = rocir.const_index(k)
                        idx_k = rocir.arith.AddIOp(unwrap(curr_idx), unwrap(c_k)).result
                        is_valid = rocir.arith.CmpIOp(rocir.arith.CmpIPredicate.ult, unwrap(idx_k), unwrap(c_N)).result
                        v_e = arith.constant(elem_type, 0.0).value
                        if is_valid:
                            v_e = tensor_S[(unwrap(c0_idx), unwrap(idx_k))]
                        v = unwrap(v_e) if dtype_str == "f32" else rocir.arith.extf(compute_type, unwrap(v_e))
                        v2 = rocir.arith.MulFOp(unwrap(v), unwrap(v), fastmath=fm_fast).result
                        thread_sumsq = rocir.arith.AddFOp(unwrap(thread_sumsq), unwrap(v2), fastmath=fm_fast).result

            sum_sq = block_reduce_add(thread_sumsq, s_red)
            mean_sq = rocir.arith.DivFOp(unwrap(sum_sq), unwrap(n_float.value), fastmath=fm_fast).result

            ms_eps = rocir.arith.AddFOp(unwrap(mean_sq), unwrap(eps.value), fastmath=fm_fast).result
            rrms = math.rsqrt(unwrap(ms_eps))

            # Pass2: normalize + gamma + store
            vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
            vec_type_c = ir.VectorType.get([VEC_WIDTH], compute_type)
            rrms_splat = vector.splat(vec_type_c, unwrap(rrms))

            # Software pipeline for aligned tiles: prefetch Gamma
            g_pref_e = None
            if N >= BLOCK_THREADS * VEC_WIDTH:
                c_base0 = rocir.const_index(0)
                thread_offset0 = rocir.arith.MulIOp(unwrap(tid), rocir.const_index(VEC_WIDTH)).result
                curr0 = rocir.arith.AddIOp(unwrap(c_base0), unwrap(thread_offset0)).result
                vec_type_e0 = ir.VectorType.get([VEC_WIDTH], elem_type)
                g_pref_e = vector.load(vec_type_e0, Gamma, [unwrap(curr0)], alignment=VEC_ALIGN)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS * VEC_WIDTH):
                c_base = rocir.const_index(base_idx_int)
                thread_offset_base = rocir.arith.MulIOp(unwrap(tid), rocir.const_index(VEC_WIDTH)).result
                curr_idx = rocir.arith.AddIOp(unwrap(c_base), unwrap(thread_offset_base)).result

                tile_safe = (base_idx_int + BLOCK_THREADS * VEC_WIDTH) <= N
                if tile_safe:
                    # Prefetch next Gamma early (software pipeline)
                    next_base_int = base_idx_int + (BLOCK_THREADS * VEC_WIDTH)
                    if next_base_int < N:
                        c_base_n = rocir.const_index(next_base_int)
                        curr_idx_n = rocir.arith.AddIOp(unwrap(c_base_n), unwrap(thread_offset_base)).result
                        g_next_e = vector.load(vec_type_e, Gamma, [unwrap(curr_idx_n)], alignment=VEC_ALIGN)
                    else:
                        g_next_e = None

                    x_e = vector.load(
                        vec_type_e, s_row, [unwrap(c0_idx), unwrap(curr_idx)], alignment=VEC_ALIGN
                    )
                    # Gamma is reused across many blocks: do NOT use nontemporal here.
                    g_e = g_pref_e if g_pref_e is not None else vector.load(vec_type_e, Gamma, [unwrap(curr_idx)], alignment=VEC_ALIGN)
                    x = x_e if dtype_str == "f32" else rocir.arith.extf(vec_type_c, unwrap(x_e))
                    g = g_e if dtype_str == "f32" else rocir.arith.extf(vec_type_c, unwrap(g_e))
                    norm = rocir.arith.MulFOp(unwrap(x), unwrap(rrms_splat), fastmath=fm_fast).result
                    y = rocir.arith.MulFOp(unwrap(norm), unwrap(g), fastmath=fm_fast).result
                    y_e = y if dtype_str == "f32" else rocir.arith.truncf(vec_type_e, unwrap(y))
                    tile_i = base_idx_int // tile_cols  # python int
                    blkOut = gOut[(unwrap(row), tile_i)]
                    thrOut = thr_copy_e.partition_S(blkOut)
                    frgOut = rocir.make_fragment_like(thrOut, elem_type)
                    vector.store(unwrap(y_e), frgOut.memref, [c0_idx, c0_idx], alignment=VEC_ALIGN)
                    rocir.copy(
                        tiled_copy_e,
                        frgOut,
                        thrOut,
                        nontemporal=USE_NONTEMPORAL,
                        alignment=VEC_ALIGN,
                    )
                    g_pref_e = g_next_e
                else:
                    c_N = rocir.const_index(N)
                    for k in range_constexpr(VEC_WIDTH):
                        c_k = rocir.const_index(k)
                        idx_k = rocir.arith.AddIOp(unwrap(curr_idx), unwrap(c_k)).result
                        is_valid = rocir.arith.CmpIOp(rocir.arith.CmpIPredicate.ult, unwrap(idx_k), unwrap(c_N)).result
                        if is_valid:
                            x_e = tensor_S[(unwrap(c0_idx), unwrap(idx_k))]
                            g_e = tensor_Gamma[unwrap(idx_k)]
                            x = unwrap(x_e) if dtype_str == "f32" else rocir.arith.extf(compute_type, unwrap(x_e))
                            g = unwrap(g_e) if dtype_str == "f32" else rocir.arith.extf(compute_type, unwrap(g_e))
                            norm = rocir.arith.MulFOp(unwrap(x), unwrap(rrms), fastmath=fm_fast).result
                            y = rocir.arith.MulFOp(unwrap(norm), unwrap(g), fastmath=fm_fast).result
                            y_e = y if dtype_str == "f32" else rocir.arith.truncf(elem_type, unwrap(y))
                            tensor_Out[(unwrap(row), unwrap(idx_k))] = unwrap(y_e)

    return _RMSNorm()


