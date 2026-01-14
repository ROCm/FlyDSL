"""LayerNorm kernel builder used by tests.

This file intentionally keeps the kernel builder logic identical to the version
embedded in `tests/kernels/test_layernorm.py` (before factoring) to preserve
codegen and performance. Only test-only helpers/imports are removed.
"""

import os

from flydsl.dialects.ext import flir, arith
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


BLOCK_THREADS = 256
WARP_SIZE = 64
VEC_WIDTH = 8
USE_NONTEMPORAL = True
VEC_ALIGN = 16


def build_layernorm_module(M: int, N: int, dtype_str: str):
    ROW_PACK = int(os.environ.get("FLYDSL_LAYERNORM_ROW_PACK", "1"))
    if ROW_PACK not in (1, 2, 4):
        raise ValueError(f"FLYDSL_LAYERNORM_ROW_PACK must be 1/2/4, got {ROW_PACK}")
    # Avoid LDS overflow on some chips (e.g., gfx942 64KB limit): row_pack=4 with N==8192 bf16/f16
    # can exceed by a small margin once we include reduction scratch.
    if ROW_PACK == 4 and N >= 8192 and dtype_str in ("bf16", "f16"):
        ROW_PACK = 2

    arch = get_hip_arch()
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
            # Cache one row into LDS.
            _state["smem_row"] = allocator.allocate_tensor((ROW_PACK, N), elem_type)
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
            bid = flir.const_index(flir.block_idx("x"))
            c_rp = flir.const_index(ROW_PACK)
            row_base = (arith.ArithValue(bid) * c_rp).value
            tid = flir.const_index(flir.thread_idx("x"))

            elem_type = _state["elem_type"]
            compute_type = _state["compute_type"]

            zero_idx = flir.const_index(0)
            n_float = arith.constant(float(N), type=compute_type)
            eps = arith.constant(EPS, type=compute_type)
            fm_fast = flir.arith.FastMathFlags.fast

            base_ptr = allocator.get_base()
            s_sum = _state["smem_red_sum"](base_ptr).get()
            s_sumsq = _state["smem_red_sumsq"](base_ptr).get()
            s_row = _state["smem_row"](base_ptr).get()
            # FLIR-style tensor views + tiled copies (like elementwise_add_kernel).
            c0_idx = flir.const_index(0)
            tile_cols = BLOCK_THREADS * VEC_WIDTH  # python int
            tensor_In = flir.make_tensor(Input, shape=(M, N), strides=(N, 1))
            tensor_Out = flir.make_tensor(Output, shape=(M, N), strides=(N, 1))
            tensor_Gamma = flir.make_tensor(Gamma, shape=(N,), strides=(1,))
            tensor_Beta = flir.make_tensor(Beta, shape=(N,), strides=(1,))
            tensor_S = flir.make_tensor(s_row, shape=(ROW_PACK, N), strides=(N, 1))
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
            thr_copy_e = tiled_copy_e.get_slice((tid))
            block_reduce_add = reduce_utils.make_block_reduce_add(
                tid=tid,
                fm_fast=fm_fast,
                WARP_SIZE=WARP_SIZE,
                RED_SLOTS=RED_SLOTS,
                gpu=flir.gpu_ext,
                arith=arith,
                arith_ops=flir.arith,
                flir=flir,
                T=T,
                ir=ir,
                zero_idx=zero_idx,
            )

            # For small/un-aligned N, use simple 2-pass global kernel.
            # Row-pack is implemented too: Gamma/Beta loads are shared across packed rows.
            if N < tile_cols:
                c_N = flir.const_index(N)
                c_M = flir.const_index(M)
                c_zero = arith.constant(0.0, type=compute_type)
                inv_n = arith.constant(1.0 / float(N), type=compute_type)

                means = []
                rstds = []
                rows = []
                valids = []

                # Pass1: compute mean/rstd per row.
                for r in range_constexpr(ROW_PACK):
                    c_r = flir.const_index(r)
                    row_r = (arith.ArithValue(row_base) + c_r).value
                    is_valid_row = arith.ult(row_r, c_M)
                    rows.append(row_r)
                    valids.append(is_valid_row)

                    thread_sum = c_zero
                    thread_sumsq = c_zero
                    for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                        c_base = flir.const_index(base_idx_int)
                        idx = c_base + tid
                        is_valid = arith.ult(idx, c_N)
                        thread_sum_next = thread_sum
                        thread_sumsq_next = thread_sumsq
                        if is_valid_row and is_valid:
                            x_e = flir.memref.load(Input, [(row_r), arith.as_value(idx)])
                            x = (x_e) if dtype_str == "f32" else flir.arith.extf(compute_type, arith.as_value(x_e))
                            x = arith.as_value(x)
                            x2 = (arith.ArithValue(x) * x).value
                            thread_sum_next = thread_sum + x
                            thread_sumsq_next = thread_sumsq + x2
                        thread_sum, thread_sumsq = thread_sum_next, thread_sumsq_next

                    sum_val = block_reduce_add(thread_sum, s_sum)
                    sumsq_val = block_reduce_add(thread_sumsq, s_sumsq)
                    mean = sum_val * inv_n
                    mean_sq = sumsq_val * inv_n
                    var = mean_sq - (mean * mean)
                    c0_f = arith.constant(0.0, type=compute_type)
                    var = arith.select(var < c0_f, c0_f, var)
                    rstd = flir.math.rsqrt(arith.as_value(var + eps), fastmath=fm_fast)
                    means.append(mean)
                    rstds.append(rstd)

                # Pass2: normalize + affine + store (Gamma/Beta shared across packed rows).
                for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                    c_base = flir.const_index(base_idx_int)
                    idx = c_base + tid
                    is_valid = arith.ult(idx, c_N)
                    if is_valid:
                        g_e = flir.memref.load(Gamma, [arith.as_value(idx)])
                        b_e = flir.memref.load(Beta, [arith.as_value(idx)])
                        g = (g_e) if dtype_str == "f32" else flir.arith.extf(compute_type, (g_e))
                        b = (b_e) if dtype_str == "f32" else flir.arith.extf(compute_type, (b_e))
                        for r in range_constexpr(ROW_PACK):
                            if valids[r]:
                                row_r = rows[r]
                                x_e = flir.memref.load(Input, [(row_r), arith.as_value(idx)])
                                x = (x_e) if dtype_str == "f32" else flir.arith.extf(compute_type, (x_e))
                                y = (x - means[r]) * rstds[r] * g + b
                                y_e = y if dtype_str == "f32" else flir.arith.truncf(elem_type, (y))
                                flir.memref.store((y_e), Output, [(row_r), (idx)])
                return

            # Vectorized path: global -> LDS row cache -> reduce from LDS -> affine from LDS.
            thread_offset_base = (arith.ArithValue(tid) * VEC_WIDTH).value
            vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
            vec_type_c = ir.VectorType.get([VEC_WIDTH], compute_type)

            c_M = flir.const_index(M)
            rows = []
            valids = []
            for r in range_constexpr(ROW_PACK):
                c_r = flir.const_index(r)
                row_r = (arith.ArithValue(row_base) + c_r).value
                valids.append(arith.ult(row_r, c_M))
                rows.append(row_r)

            # Pass0: global -> LDS row cache (for each packed row)
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS * VEC_WIDTH):
                c_base = flir.const_index(base_idx_int)
                curr_idx = (arith.ArithValue(c_base) + thread_offset_base).value

                tile_safe = (base_idx_int + BLOCK_THREADS * VEC_WIDTH) <= N
                if tile_safe:
                    tile_i = base_idx_int // tile_cols  # python int
                    for r in range_constexpr(ROW_PACK):
                        if valids[r]:
                            blkIn = gIn[((rows[r]), tile_i)]
                            blkS = gS[(r, tile_i)]
                            thrIn = thr_copy_e.partition_S(blkIn)
                            thrS = thr_copy_e.partition_S(blkS)
                            flir.copy(
                                tiled_copy_e,
                                thrIn,
                                thrS,
                                nontemporal=USE_NONTEMPORAL,
                                alignment=VEC_ALIGN,
                            )
                else:
                    c_N = flir.const_index(N)
                    for k in range_constexpr(VEC_WIDTH):
                        c_k = flir.const_index(k)
                        idx_k = curr_idx + c_k
                        is_valid = arith.ult(idx_k, c_N)
                        if is_valid:
                            for r in range_constexpr(ROW_PACK):
                                if valids[r]:
                                    v_e = tensor_In[((rows[r]), arith.as_value(idx_k))]
                                    tensor_S[((flir.const_index(r)), arith.as_value(idx_k))] = (v_e)

            flir.gpu_ext.barrier()

            inv_n = arith.constant(1.0 / float(N), type=compute_type)
            means = []
            rstds = []
            mean_splats = []
            rstd_splats = []

            # Pass1: sum + sumsq (from LDS row cache), per packed row.
            for r in range_constexpr(ROW_PACK):
                c_zero = arith.constant(0.0, type=compute_type)
                thread_sum = c_zero
                thread_sumsq = c_zero
                for base_idx_int in range_constexpr(0, N, BLOCK_THREADS * VEC_WIDTH):
                    c_base = flir.const_index(base_idx_int)
                    curr_idx = (arith.ArithValue(c_base) + thread_offset_base).value

                    tile_safe = (base_idx_int + BLOCK_THREADS * VEC_WIDTH) <= N
                    if tile_safe:
                        x_e = flir.vector.load(vec_type_e, s_row, [(flir.const_index(r)), (curr_idx)], alignment=VEC_ALIGN)
                        x = x_e if dtype_str == "f32" else flir.arith.extf(vec_type_c, arith.as_value(x_e))
                        x = arith.as_value(x)
                        x2 = (arith.ArithValue(x) * x).value
                        red = flir.vector.reduction(compute_type, "add", (x), fastmath=fm_fast)
                        red2 = flir.vector.reduction(compute_type, "add", (x2), fastmath=fm_fast)
                        thread_sum = thread_sum + red
                        thread_sumsq = thread_sumsq + red2
                    else:
                        c_N = flir.const_index(N)
                        for k in range_constexpr(VEC_WIDTH):
                            c_k = flir.const_index(k)
                            idx_k = curr_idx + c_k
                            is_valid = arith.ult(idx_k, c_N)
                            if is_valid:
                                v_e = tensor_S[((flir.const_index(r)), arith.as_value(idx_k))]
                            else:
                                v_e = arith.constant(0.0, type=elem_type)
                            v = (v_e) if dtype_str == "f32" else flir.arith.extf(compute_type, arith.as_value(v_e))
                            v2 = (arith.ArithValue(v) * v).value
                            thread_sum = thread_sum + v
                            thread_sumsq = thread_sumsq + v2

                sum_val = block_reduce_add(thread_sum, s_sum)
                sumsq_val = block_reduce_add(thread_sumsq, s_sumsq)
                mean = sum_val * inv_n
                mean_sq = sumsq_val * inv_n
                var = mean_sq - (mean * mean)
                c0_f = arith.constant(0.0, type=compute_type)
                var = arith.select(var < c0_f, c0_f, var)
                rstd = flir.math.rsqrt(arith.as_value(var + eps), fastmath=fm_fast)
                means.append(mean)
                rstds.append(rstd)
                mean_splats.append(flir.vector.splat(vec_type_c, arith.as_value(mean)))
                rstd_splats.append(flir.vector.splat(vec_type_c, arith.as_value(rstd)))

            # Pass2: normalize + affine + store (from LDS row cache), with Gamma/Beta prefetch shared across rows.
            g_pref_e = None
            b_pref_e = None
            if N >= BLOCK_THREADS * VEC_WIDTH:
                c_base0 = flir.const_index(0)
                curr0 = (arith.ArithValue(c_base0) + thread_offset_base).value
                g_pref_e = flir.vector.load(vec_type_e, Gamma, [arith.as_value(curr0)], alignment=VEC_ALIGN)
                b_pref_e = flir.vector.load(vec_type_e, Beta, [arith.as_value(curr0)], alignment=VEC_ALIGN)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS * VEC_WIDTH):
                c_base = flir.const_index(base_idx_int)
                curr_idx = (arith.ArithValue(c_base) + thread_offset_base).value

                tile_safe = (base_idx_int + BLOCK_THREADS * VEC_WIDTH) <= N
                if tile_safe:
                    # Prefetch next Gamma/Beta early (software pipeline)
                    next_base_int = base_idx_int + (BLOCK_THREADS * VEC_WIDTH)
                    if next_base_int < N:
                        c_base_n = flir.const_index(next_base_int)
                        curr_idx_n = (arith.ArithValue(c_base_n) + thread_offset_base).value
                        g_next_e = flir.vector.load(vec_type_e, Gamma, [arith.as_value(curr_idx_n)], alignment=VEC_ALIGN)
                        b_next_e = flir.vector.load(vec_type_e, Beta, [arith.as_value(curr_idx_n)], alignment=VEC_ALIGN)
                    else:
                        g_next_e = None
                        b_next_e = None

                    g_e = g_pref_e if g_pref_e is not None else flir.vector.load(vec_type_e, Gamma, [arith.as_value(curr_idx)], alignment=VEC_ALIGN)
                    b_e = b_pref_e if b_pref_e is not None else flir.vector.load(vec_type_e, Beta, [arith.as_value(curr_idx)], alignment=VEC_ALIGN)
                    g_v = g_e if dtype_str == "f32" else flir.arith.extf(vec_type_c, arith.as_value(g_e))
                    b_v = b_e if dtype_str == "f32" else flir.arith.extf(vec_type_c, arith.as_value(b_e))

                    tile_i = base_idx_int // tile_cols  # python int
                    for r in range_constexpr(ROW_PACK):
                        if valids[r]:
                            x_e = flir.vector.load(vec_type_e, s_row, [(flir.const_index(r)), arith.as_value(curr_idx)], alignment=VEC_ALIGN)
                            x = x_e if dtype_str == "f32" else flir.arith.extf(vec_type_c, arith.as_value(x_e))
                            diff = (arith.ArithValue(arith.as_value(x)) - arith.as_value(mean_splats[r])).value
                            norm = (arith.ArithValue(diff) * arith.as_value(rstd_splats[r])).value
                            y = (arith.ArithValue(norm) * arith.as_value(g_v)).value
                            y = (arith.ArithValue(y) + arith.as_value(b_v)).value
                            y_e = y if dtype_str == "f32" else flir.arith.truncf(vec_type_e, arith.as_value(y))

                            blkOut = gOut[((rows[r]), tile_i)]
                            thrOut = thr_copy_e.partition_S(blkOut)
                            frgOut = flir.make_fragment_like(thrOut, elem_type)
                            flir.vector.store(arith.as_value(y_e), frgOut.memref, [c0_idx, c0_idx], alignment=VEC_ALIGN)
                            flir.copy(
                                tiled_copy_e,
                                frgOut,
                                thrOut,
                                nontemporal=USE_NONTEMPORAL,
                                alignment=VEC_ALIGN,
                            )

                    g_pref_e = g_next_e
                    b_pref_e = b_next_e
                else:
                    c_N = flir.const_index(N)
                    for k in range_constexpr(VEC_WIDTH):
                        c_k = flir.const_index(k)
                        idx_k = curr_idx + c_k
                        is_valid = arith.ult(idx_k, c_N)
                        if is_valid:
                            g_e = tensor_Gamma[arith.as_value(idx_k)]
                            b_e = tensor_Beta[arith.as_value(idx_k)]
                            g = (g_e) if dtype_str == "f32" else flir.arith.extf(compute_type, arith.as_value(g_e))
                            b = (b_e) if dtype_str == "f32" else flir.arith.extf(compute_type, arith.as_value(b_e))
                            for r in range_constexpr(ROW_PACK):
                                if valids[r]:
                                    x_e = tensor_S[((flir.const_index(r)), arith.as_value(idx_k))]
                                    x = (x_e) if dtype_str == "f32" else flir.arith.extf(compute_type, arith.as_value(x_e))
                                    y = (x - means[r]) * rstds[r] * g + b
                                    y_e = y if dtype_str == "f32" else flir.arith.truncf(elem_type, arith.as_value(y))
                                    tensor_Out[((rows[r]), arith.as_value(idx_k))] = (y_e)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            Input: lambda: T.memref(M, N, _state["elem_type"]),
            Gamma: lambda: T.memref(N, _state["elem_type"]),
            Beta: lambda: T.memref(N, _state["elem_type"]),
            Output: lambda: T.memref(M, N, _state["elem_type"]),
        ):
            c1 = (flir.arith_ext.index(1))
            gx = (flir.arith_ext.index((M + ROW_PACK - 1) // ROW_PACK))
            bx = (flir.arith_ext.index(BLOCK_THREADS))
            flir.gpu_ext.LaunchFuncOp(
                ["layernorm_module", "layernorm_kernel"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[Input, Gamma, Beta, Output],
            )

    return _LayerNorm()


