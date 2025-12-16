#!/usr/bin/env python3
"""
LayerNorm Operator Test
Implementation of a Block-wise LayerNorm:
- Grid: (M, 1, 1) -> One block per row
- Block: (N, 1, 1) -> Threads handle columns
- Shared Memory: Used for reduction (mean and variance)

LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta
"""

import sys
import os

# Add paths to find rocdsl and mlir packages (prefer embedded MLIR to avoid mixing runtimes)
repo_root = os.path.join(os.path.dirname(__file__), "../../..")
embedded_pkgs = os.path.join(repo_root, "build", "python_packages", "rocdsl")
if os.path.isdir(os.path.join(embedded_pkgs, "_mlir")):
    sys.path.insert(0, embedded_pkgs)
else:
    sys.path.insert(0, os.path.join(os.environ.get('MLIR_PATH', ''), 'tools/mlir/python_packages/mlir_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../build/python_bindings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../python'))
sys.path.insert(0, repo_root)

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.dialects.ext import rocir
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from rocdsl.utils import SmemAllocator
from tests.utils import compile_to_hsaco
from _mlir import ir
import _mlir.extras.types as T
try:
    from hip import hip
except ImportError:
    print("HIP module not found. Skipping GPU tests.")
    sys.exit(0)

import numpy as np
import ctypes
import time

# Expose modules through Rocir interface (keep behavior/perf, avoid mlir.* imports).
gpu = rocir.gpu_ext
scf = rocir.scf_ext
# Keep arith as the raw dialect module here (this file uses arith.constant(Type, value) form).
arith = rocir.arith
mlir_arith = rocir.arith
memref = rocir.memref
vector = rocir.vector
math = rocir.math

# Small helper: unwrap MLIR wrapper values into ir.Value
def unwrap(v):
    if hasattr(v, "value"): return v.value
    if hasattr(v, "_value"): return v._value
    if hasattr(v, "result"): return v.result
    return v

EPS = 1e-5

# bf16 host packing helpers (same convention as test_softmax.py: store bf16 as uint16 payload)
def bf16_to_fp32_cpu(arr_bf16_uint16):
    arr_u32 = arr_bf16_uint16.astype(np.uint32) << 16
    return np.frombuffer(arr_u32.tobytes(), dtype=np.float32).reshape(arr_bf16_uint16.shape)

def fp32_to_bf16_cpu(arr_fp32):
    u32 = np.frombuffer(arr_fp32.astype(np.float32).tobytes(), dtype=np.uint32)
    u32 = u32.reshape(arr_fp32.shape)
    # round to nearest even on the bf16 cut
    lsb = (u32 >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    u32_rounded = u32 + rounding_bias
    return (u32_rounded >> 16).astype(np.uint16)

BLOCK_THREADS = 256
WARP_SIZE = 64
VEC_WIDTH = 8
USE_NONTEMPORAL = True
VEC_ALIGN = 16

def dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f32": return T.f32()
    if dtype_str == "f16": return T.f16()
    if dtype_str == "bf16": return T.bf16()
    raise ValueError(f"unsupported dtype: {dtype_str}")

def build_layernorm_module(M: int, N: int, dtype_str: str):
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    gpu.set_container_module(ctx.module)

    arch = get_hip_arch()
    allocator = SmemAllocator(ctx, arch=arch)

    elem_type = dtype_to_elem_type(dtype_str)
    compute_type = T.f32()  # compute in fp32 for stability (and to keep bf16 safe on backend)

    # Allocate Shared Memory for block reductions (one slot per wave)
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    smem_red_sum = allocator.allocate_array(T.f32(), RED_SLOTS)
    smem_red_sumsq = allocator.allocate_array(T.f32(), RED_SLOTS)
    # Cache row in LDS to avoid 2nd global read of Input
    smem_row = allocator.allocate_array(elem_type, N)

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
            s_row = smem_row(base_ptr).get()
            # Rocir-style tensor views + tiled copies (like elementwise_add_kernel).
            c0_idx = rocir.const_index(0)
            tile_cols = BLOCK_THREADS * VEC_WIDTH  # python int
            tensor_In = rocir.make_tensor(Input, shape=(M, N), strides=(N, 1))
            tensor_Out = rocir.make_tensor(Output, shape=(M, N), strides=(N, 1))
            tensor_Gamma = rocir.make_tensor(Gamma, shape=(N,), strides=(1,))
            tensor_Beta = rocir.make_tensor(Beta, shape=(N,), strides=(1,))
            # Represent LDS row cache as a 2D tensor view (1, N) over a 1D memref.
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

            def block_reduce_add(val_f32, scratch_memref):
                scratch_tv = rocir.make_tensor(scratch_memref, shape=(RED_SLOTS,), strides=(1,))
                tid_i32 = mlir_arith.IndexCastOp(T.i32(), tid.value).result
                c_warp_i32 = arith.constant(T.i32(), WARP_SIZE)
                lane_i32 = mlir_arith.RemUIOp(unwrap(tid_i32), unwrap(c_warp_i32)).result
                wave_i32 = mlir_arith.DivUIOp(unwrap(tid_i32), unwrap(c_warp_i32)).result
                width_i32 = arith.constant(T.i32(), WARP_SIZE)
                # Use Rocir layout algebra to compute LDS indices for the reduction scratch.
                c_num_waves = rocir.const_index(RED_SLOTS)
                c1 = rocir.const_index(1)
                shape_red = rocir.make_shape(unwrap(c_num_waves))
                stride_red = rocir.make_stride(unwrap(c1))
                layout_red = rocir.make_layout(shape_red, stride_red)

                w = unwrap(val_f32)
                for sh in [32, 16, 8, 4, 2, 1]:
                    off = arith.constant(T.i32(), sh)
                    peer = gpu.ShuffleOp(unwrap(w), unwrap(off), unwrap(width_i32), mode="xor").shuffleResult
                    w = mlir_arith.AddFOp(unwrap(w), unwrap(peer), fastmath=fm_fast).result

                is_lane0 = mlir_arith.CmpIOp(
                    mlir_arith.CmpIPredicate.eq,
                    unwrap(lane_i32),
                    unwrap(arith.constant(T.i32(), 0)),
                ).result
                if_lane0 = scf.IfOp(unwrap(is_lane0))
                with ir.InsertionPoint(if_lane0.then_block):
                    wave_idx = mlir_arith.IndexCastOp(T.index(), unwrap(wave_i32)).result
                    red_idx = rocir.crd2idx(rocir.make_coord(unwrap(wave_idx)), layout_red)
                    scratch_tv[unwrap(red_idx)] = unwrap(w)
                    scf.yield_([])
                gpu.barrier()

                NUM_WAVES = RED_SLOTS
                is_wave0 = mlir_arith.CmpIOp(
                    mlir_arith.CmpIPredicate.eq,
                    unwrap(wave_i32),
                    unwrap(arith.constant(T.i32(), 0)),
                ).result
                # Only wave0 does the final reduction and writes scratch[0].
                # Everyone else just waits; avoid useless else-path loads.
                if_wave0 = scf.IfOp(unwrap(is_wave0))
                with ir.InsertionPoint(if_wave0.then_block):
                    in_range = mlir_arith.CmpIOp(
                        mlir_arith.CmpIPredicate.ult,
                        unwrap(lane_i32),
                        unwrap(arith.constant(T.i32(), NUM_WAVES)),
                    ).result
                    if_in = scf.IfOp(unwrap(in_range), [T.f32()], hasElse=True)
                    with ir.InsertionPoint(if_in.then_block):
                        lane_idx = mlir_arith.IndexCastOp(T.index(), unwrap(lane_i32)).result
                        red_idx = rocir.crd2idx(rocir.make_coord(unwrap(lane_idx)), layout_red)
                        v = scratch_tv[unwrap(red_idx)]
                        scf.yield_([unwrap(v)])
                    with ir.InsertionPoint(if_in.else_block):
                        scf.yield_([unwrap(arith.constant(T.f32(), 0.0).value)])

                    ww = if_in.results[0]
                    for sh in [32, 16, 8, 4, 2, 1]:
                        off = arith.constant(T.i32(), sh)
                        peer = gpu.ShuffleOp(unwrap(ww), unwrap(off), unwrap(width_i32), mode="xor").shuffleResult
                        ww = mlir_arith.AddFOp(unwrap(ww), unwrap(peer), fastmath=fm_fast).result

                    is_lane0_2 = mlir_arith.CmpIOp(
                        mlir_arith.CmpIPredicate.eq,
                        unwrap(lane_i32),
                        unwrap(arith.constant(T.i32(), 0)),
                    ).result
                    if_lane0_2 = scf.IfOp(unwrap(is_lane0_2))
                    with ir.InsertionPoint(if_lane0_2.then_block):
                        red_idx0 = rocir.crd2idx(rocir.make_coord(unwrap(zero_idx)), layout_red)
                        scratch_tv[unwrap(red_idx0)] = unwrap(ww)
                        scf.yield_([])
                    scf.yield_([])

                gpu.barrier()
                red_idx0 = rocir.crd2idx(rocir.make_coord(unwrap(zero_idx)), layout_red)
                return scratch_tv[unwrap(red_idx0)]

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

            for base_idx_int in range(0, N, BLOCK_THREADS * VEC_WIDTH):
                c_base = rocir.const_index(base_idx_int)
                thread_offset_base = mlir_arith.MulIOp(unwrap(tid), rocir.const_index(VEC_WIDTH)).result
                curr_idx = mlir_arith.AddIOp(unwrap(c_base), unwrap(thread_offset_base)).result

                tile_safe = (base_idx_int + BLOCK_THREADS * VEC_WIDTH) <= N
                if tile_safe:
                    tile_i = base_idx_int // tile_cols  # python int
                    blkIn = gIn[(unwrap(row), tile_i)]
                    blkS = gS[(0, tile_i)]
                    thrIn = thr_copy_e.partition_S(blkIn)
                    thrS = thr_copy_e.partition_S(blkS)
                    vec_e = rocir.copy(
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
                    c_N = rocir.const_index(N)
                    for k in range(VEC_WIDTH):
                        c_k = rocir.const_index(k)
                        idx_k = mlir_arith.AddIOp(unwrap(curr_idx), unwrap(c_k)).result
                        is_valid = mlir_arith.CmpIOp(mlir_arith.CmpIPredicate.ult, unwrap(idx_k), unwrap(c_N)).result
                        if_store = scf.IfOp(unwrap(is_valid))
                        with ir.InsertionPoint(if_store.then_block):
                            v_e = tensor_In[(unwrap(row), unwrap(idx_k))]
                            tensor_S[(unwrap(c0_idx), unwrap(idx_k))] = unwrap(v_e)
                            scf.yield_([])

            gpu.barrier()

            # Pass1: sum / sumsq (from LDS row cache)
            # If fully tiled, we've already accumulated from Pass0 global loads.
            if not FULL_TILES:
                thread_sum = unwrap(c_zero)
                thread_sumsq = unwrap(c_zero)

                for base_idx_int in range(0, N, BLOCK_THREADS * VEC_WIDTH):
                    c_base = rocir.const_index(base_idx_int)
                    thread_offset_base = mlir_arith.MulIOp(unwrap(tid), rocir.const_index(VEC_WIDTH)).result
                    curr_idx = mlir_arith.AddIOp(unwrap(c_base), unwrap(thread_offset_base)).result

                    tile_safe = (base_idx_int + BLOCK_THREADS * VEC_WIDTH) <= N
                    if tile_safe:
                        vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
                        vec_e = vector.load(vec_type_e, s_row, [unwrap(curr_idx)], alignment=VEC_ALIGN)
                        vec_type_c = ir.VectorType.get([VEC_WIDTH], compute_type)
                        vec = vec_e if dtype_str == "f32" else mlir_arith.extf(vec_type_c, unwrap(vec_e))
                        vec2 = mlir_arith.MulFOp(unwrap(vec), unwrap(vec), fastmath=fm_fast).result
                        red = vector.reduction(compute_type, "add", unwrap(vec), fastmath=fm_fast)
                        red2 = vector.reduction(compute_type, "add", unwrap(vec2), fastmath=fm_fast)
                        thread_sum = mlir_arith.AddFOp(unwrap(thread_sum), unwrap(red), fastmath=fm_fast).result
                        thread_sumsq = mlir_arith.AddFOp(unwrap(thread_sumsq), unwrap(red2), fastmath=fm_fast).result
                    else:
                        c_N = rocir.const_index(N)
                        for k in range(VEC_WIDTH):
                            c_k = rocir.const_index(k)
                            idx_k = mlir_arith.AddIOp(unwrap(curr_idx), unwrap(c_k)).result
                            is_valid = mlir_arith.CmpIOp(mlir_arith.CmpIPredicate.ult, unwrap(idx_k), unwrap(c_N)).result
                            if_load = scf.IfOp(unwrap(is_valid), [elem_type], hasElse=True)
                            with ir.InsertionPoint(if_load.then_block):
                                v_e = tensor_S[(unwrap(c0_idx), unwrap(idx_k))]
                                scf.yield_([unwrap(v_e)])
                            with ir.InsertionPoint(if_load.else_block):
                                # dummy (won't be used)
                                scf.yield_([unwrap(arith.constant(elem_type, 0.0).value)])
                            v_e = if_load.results[0]
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

            for base_idx_int in range(0, N, BLOCK_THREADS * VEC_WIDTH):
                c_base = rocir.const_index(base_idx_int)
                thread_offset_base = mlir_arith.MulIOp(unwrap(tid), rocir.const_index(VEC_WIDTH)).result
                curr_idx = mlir_arith.AddIOp(unwrap(c_base), unwrap(thread_offset_base)).result

                tile_safe = (base_idx_int + BLOCK_THREADS * VEC_WIDTH) <= N
                if tile_safe:
                    x_e = vector.load(vec_type_e, s_row, [unwrap(curr_idx)], alignment=VEC_ALIGN)
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
                        frgOut = rocir.make_fragment_like(thrOut, elem_type)
                        vector.store(out_bf16, frgOut.memref, [c0_idx, c0_idx], alignment=VEC_ALIGN)
                        rocir.copy(
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
                        frgOut = rocir.make_fragment_like(thrOut, elem_type)
                        vector.store(unwrap(y_e), frgOut.memref, [c0_idx, c0_idx], alignment=VEC_ALIGN)
                        rocir.copy(
                            tiled_copy_e,
                            frgOut,
                            thrOut,
                            nontemporal=USE_NONTEMPORAL,
                            alignment=VEC_ALIGN,
                        )
                else:
                    c_N = rocir.const_index(N)
                    # scalar tail
                    for k in range(VEC_WIDTH):
                        c_k = rocir.const_index(k)
                        idx_k = mlir_arith.AddIOp(unwrap(curr_idx), unwrap(c_k)).result
                        is_valid = mlir_arith.CmpIOp(mlir_arith.CmpIPredicate.ult, unwrap(idx_k), unwrap(c_N)).result
                        if_store = scf.IfOp(unwrap(is_valid))
                        with ir.InsertionPoint(if_store.then_block):
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
                            scf.yield_([])

    return ctx

def run_test(M: int, N: int, dtype: str = "f32") -> bool:
    print(f"\nTesting LayerNorm (M={M}, N={N}, dtype={dtype})")

    if hip is None:
        print("HIP not available, skipping...")
        return True

    ctx = build_layernorm_module(M, N, dtype)
    try:
        hsaco = compile_to_hsaco(ctx.module, kernel_name="layernorm")
    except Exception as e:
        print(f"Compilation failed: {e}")
        print(ctx.module)
        raise e

    print(f" HSACO size: {len(hsaco)} bytes")

    np.random.seed(42)
    input_f32 = np.random.randn(M, N).astype(np.float32)
    gamma_f32 = np.random.rand(N).astype(np.float32)
    beta_f32 = np.random.rand(N).astype(np.float32)

    if dtype == "f32":
        input_host = input_f32
        gamma_host = gamma_f32
        beta_host = beta_f32
        output_host = np.zeros((M, N), dtype=np.float32)
        elem_bytes = 4
        input_ref = input_f32
        gamma_ref = gamma_f32
        beta_ref = beta_f32
        atol = 1e-4
    elif dtype == "f16":
        input_host = input_f32.astype(np.float16)
        gamma_host = gamma_f32.astype(np.float16)
        beta_host = beta_f32.astype(np.float16)
        output_host = np.zeros((M, N), dtype=np.float16)
        elem_bytes = 2
        input_ref = input_host.astype(np.float32)
        gamma_ref = gamma_host.astype(np.float32)
        beta_ref = beta_host.astype(np.float32)
        atol = 1e-2
    elif dtype == "bf16":
        input_host = fp32_to_bf16_cpu(input_f32)
        gamma_host = fp32_to_bf16_cpu(gamma_f32)
        beta_host = fp32_to_bf16_cpu(beta_f32)
        output_host = np.zeros((M, N), dtype=np.uint16)
        elem_bytes = 2
        input_ref = bf16_to_fp32_cpu(input_host)
        gamma_ref = bf16_to_fp32_cpu(gamma_host)
        beta_ref = bf16_to_fp32_cpu(beta_host)
        atol = 2e-2
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    # Numpy Reference
    mean = np.mean(input_ref, axis=1, keepdims=True)
    var = np.var(input_ref, axis=1, keepdims=True)
    expected = (input_ref - mean) / np.sqrt(var + EPS) * gamma_ref + beta_ref

    # Allocate GPU Memory
    d_input = hip_check(hip.hipMalloc(M * N * elem_bytes))
    d_gamma = hip_check(hip.hipMalloc(N * elem_bytes))
    d_beta = hip_check(hip.hipMalloc(N * elem_bytes))
    d_output = hip_check(hip.hipMalloc(M * N * elem_bytes))

    hip_check(hip.hipMemcpy(d_input, input_host.ctypes.data, M * N * elem_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(d_gamma, gamma_host.ctypes.data, N * elem_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(d_beta, beta_host.ctypes.data, N * elem_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))

    # Load Kernel
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"layernorm_kernel"))

    # Launch Config
    grid_x, grid_y, grid_z = M, 1, 1
    block_x, block_y, block_z = BLOCK_THREADS, 1, 1
    smem_size = 0

    arg_ptrs = [
        ctypes.c_void_p(int(d_input)),
        ctypes.c_void_p(int(d_gamma)),
        ctypes.c_void_p(int(d_beta)),
        ctypes.c_void_p(int(d_output))
    ]
    args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])

    print("Launching kernel...")
    start_time = time.time()
    hip_check(hip.hipModuleLaunchKernel(kernel_func, grid_x, grid_y, grid_z, block_x, block_y, block_z, smem_size, 0, args, None))
    hip_check(hip.hipDeviceSynchronize())
    end_time = time.time()
    
    print(f"Kernel execution time: {(end_time - start_time)*1000:.4f} ms")

    # Copy back
    hip_check(hip.hipMemcpy(output_host.ctypes.data, d_output, M * N * elem_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

    if dtype == "f32":
        output_ref = output_host
    elif dtype == "f16":
        output_ref = output_host.astype(np.float32)
    else:  # bf16
        output_ref = bf16_to_fp32_cpu(output_host)

    # Verification
    error = np.max(np.abs(output_ref - expected))
    print(f"Max absolute error: {error:.2e} (atol={atol})")

    if error < atol:
        print("✅ PASSED")
        ok = True
    else:
        print("❌ FAILED")
        print("First row Expected:")
        print(expected[0, :5])
        print("First row Actual:")
        print(output_host[0, :5])
        ok = False

    # Cleanup
    hip_check(hip.hipFree(d_input))
    hip_check(hip.hipFree(d_gamma))
    hip_check(hip.hipFree(d_beta))
    hip_check(hip.hipFree(d_output))
    hip_check(hip.hipModuleUnload(hip_module))
    return ok

def test_all():
    print("="*80)
    print("Running LayerNorm Tests")
    print("="*80)

    configs = [
        # (64, 256, "f32"),    # Aligned
        # (128, 1024, "f32"),  # Aligned
        # (32, 128, "f16"),    # Aligned
        # (64, 2000, "f32"),   # Unaligned (tail handling)
        # (16, 512, "bf16"),   # BF16
        # (1024, 8192, "bf16"),# BF16
        (32768, 8192, "bf16"),  # BF16
    ]

    failures = 0
    for M, N, dtype in configs:
        if not run_test(M, N, dtype):
            failures += 1

    print("\n" + "="*80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("="*80)

if __name__ == "__main__":
    test_all()

