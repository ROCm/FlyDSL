#!/usr/bin/env python3
"""MFMA GEMM Test using Flir with @gpu.func decorator pattern.
Supports FP8 (mfma_16x16x32) and FP16 (mfma_16x16x16).
Output is unified to FP16 for both paths."""

import sys
import os
import logging
import functools

# Configure logging to show INFO level messages
logging.basicConfig(level=logging.INFO)



from flydsl.runtime.device import get_rocm_arch
import flydsl
import flydsl.dialects.ext.flir as flir
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.utils import SmemAllocator
from tests.utils import pertoken_quant
from tests.test_common import run_perftest, verify_output
import torch
import torch.nn.functional as F
import pytest
from _mlir import ir
from flydsl.dialects.ext import arith, scf, gpu, buffer_ops
from flydsl.dialects.ext import vector
from flydsl.dialects.ext import memref
from _mlir.dialects import arith as _arith_mlir
from flydsl.dialects.ext import rocdl
import _mlir.extras.types as T
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

# Use torch dtypes directly for test parametrization / configuration.
DTYPE_FP8 = torch.float8_e4m3fnuz
DTYPE_FP16 = torch.float16

def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.float32):
    """
    Torch reference implementation (from aiter project).
    Dequantize FP8 inputs and compute FP32 matmul.
    """
    x = x.to(torch.float32) * x_scale
    weight = weight.to(torch.float32) * w_scale
    out = F.linear(x, weight)
    if bias is not None:
        out = out.to(bias.dtype) + bias
    return out.to(dtype)

@pytest.mark.parametrize("dtype_config", [DTYPE_FP8, DTYPE_FP16])
def test_mfma_gemm_flir(dtype_config, M=1024, N=1024, K=1280, tile_m=128, tile_n=128, tile_k=128):
    """Test MFMA GEMM with configurable dtype (FP8 or FP16)."""
    
    # Configure based on dtype (non-MLIR parameters)
    if dtype_config == DTYPE_FP8:
        print(f"MFMA FP8 GEMM Test (mfma_16x16x32) - {M}x{N}x{K}")
        mfma_k = 32  # FP8 MFMA processes 32 K elements
        torch_dtype = DTYPE_FP8
        use_scales = True
        vec_load_size = 16  # Load 16 FP8 elements (16 bytes)
        dtype_name = "fp8"
    else:  # FP16
        print(f"MFMA FP16 GEMM Test (mfma_16x16x16) - {M}x{N}x{K}")
        mfma_k = 16  # FP16 MFMA processes 16 K elements
        torch_dtype = DTYPE_FP16
        use_scales = False
        # FP16 global->LDS vector load width (in elements).
        # Prefer 16B loads: 8 * f16 = 16 bytes.
        vec_load_size = 8
        dtype_name = "fp16"
    
    gpu_arch = get_rocm_arch()
    def _mlir_dtype():
        return ir.Float8E4M3FNType.get() if dtype_config == DTYPE_FP8 else ir.F16Type.get()

    # We currently assume "full tiles" (no tail guards) for both FP8/FP16.
    # This matches the FP8 path’s grid computation and keeps the kernel simple.
    if (M % tile_m) != 0 or (N % tile_n) != 0 or (K % tile_k) != 0:
        raise ValueError(
            f"Expected M/N/K divisible by tile sizes (no tail support): "
            f"M={M},N={N},K={K}, tile_m={tile_m},tile_n={tile_n},tile_k={tile_k}"
        )

    # Common tile decomposition used by both FP8 and FP16 schedules.
    m_repeat = tile_m // 16
    n_repeat = tile_n // 64  # 64 columns per (4-wave) super-tile along N

    # Workgroup / schedule:
    # We keep both FP8/FP16 on a fixed 256-thread block to match the FP8 kernel style.
    block_size_x = 256
    fp16_waves_m = fp16_waves_n = None
    fp16_base_m = fp16_base_n = None
    fp16_m_reps = fp16_n_reps = None
    fp16_vecs_per_thread_a = fp16_vecs_per_thread_b = None

    if dtype_config == DTYPE_FP8:
        # Mirror assumptions inside the FP8 kernel (fixed schedule).
        if tile_m % 16 != 0:
            raise ValueError(f"FP8 requires tile_m multiple of 16; got tile_m={tile_m}")
        if tile_n % (4 * 16) != 0:
            raise ValueError(f"FP8 fixed num_waves=4 along N requires tile_n multiple of 64; got tile_n={tile_n}")
        if tile_k % 32 != 0:
            raise ValueError(f"FP8 MFMA requires tile_k multiple of 32; got tile_k={tile_k}")
        if (tile_m * tile_k) % block_size_x != 0:
            raise ValueError(
                f"FP8 requires tile_m*tile_k divisible by {block_size_x} threads; got tile_m={tile_m}, tile_k={tile_k}"
            )
    else:
        # FP16 fixed schedule: 4 waves along N => 16x64 base tile.
        fp16_waves_m = 1
        fp16_waves_n = 4
        fp16_base_m = 16
        fp16_base_n = 64

        if tile_m % fp16_base_m != 0:
            raise ValueError(f"FP16 requires tile_m multiple of {fp16_base_m}; got tile_m={tile_m}")
        if tile_n % fp16_base_n != 0:
            raise ValueError(
                f"FP16 fixed num_waves=4 along N requires tile_n multiple of {fp16_base_n}; got tile_n={tile_n}"
            )
        if tile_k % 16 != 0:
            raise ValueError(f"FP16 MFMA requires tile_k multiple of 16; got tile_k={tile_k}")
        if tile_k % vec_load_size != 0:
            raise ValueError(f"FP16 requires tile_k divisible by vec_load_size={vec_load_size}; got tile_k={tile_k}")

        fp16_m_reps = m_repeat
        fp16_n_reps = n_repeat

        # Vectorized LDS loads: assign whole vectors (vec_load_size elems) to threads.
        vecs_per_row = tile_k // vec_load_size
        total_vecs_a = tile_m * vecs_per_row
        total_vecs_b = tile_n * vecs_per_row
        if (total_vecs_a % block_size_x) != 0 or (total_vecs_b % block_size_x) != 0:
            raise ValueError(
                f"FP16 fixed schedule requires A/B vector loads divisible by {block_size_x} threads. "
                f"Got total_vecs_a={total_vecs_a}, total_vecs_b={total_vecs_b}"
            )
        fp16_vecs_per_thread_a = total_vecs_a // block_size_x
        fp16_vecs_per_thread_b = total_vecs_b // block_size_x
    
    size_c = M * N
    size_a = M * K
    size_b = N * K  # Transposed B (NxK)
    
    # One module + one kernel entrypoint for both FP8/FP16.
    #
    # - FP8: uses the existing K32 intrawave + swizzled LDS(A) pipeline.
    # - FP16: uses a simple LDS(A+B) pipeline + mfma_16x16x16f16, stores FP16 output.
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    # FP8 uses XOR swizzle and benefits from a small padding along K to reduce bank conflicts.
    # FP16 path does not swizzle and the default tile (128x128x128) must fit within 64KB LDS
    # on gfx942; keep pad_k=0 for FP16 to avoid allocating >64KB.
    pad_k = 8 if dtype_config == DTYPE_FP8 else 0
    lds_stride = tile_k + pad_k
    lds_size_a = tile_m * lds_stride
    lds_size_b = tile_n * lds_stride

    class _MFMA(flir.MlirModule):
        GPU_MODULE_NAME = "mfma_mod"
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            _state["lds_a_decl"] = allocator.allocate_array(_mlir_dtype(), lds_size_a)
            if dtype_config == DTYPE_FP16:
                _state["lds_b_decl"] = allocator.allocate_array(_mlir_dtype(), lds_size_b)
            allocator.finalize()

        # Always include scale parameters for uniform signature.
        # For FP16, we pass dummy scale arrays but don't use them.
        @flir.kernel
        def kernel(
            self: flir.T.i64,
            arg_c: lambda: T.memref(size_c, T.f16()),
            arg_a: lambda: T.memref(size_a, _mlir_dtype()),
            arg_b: lambda: T.memref(size_b, _mlir_dtype()),
            arg_scale_a: lambda: T.memref(M, T.f32()),
            arg_scale_b: lambda: T.memref(N, T.f32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            # ---- Shared setup (both FP8/FP16) ----
            # Keep common SSA values defined once to reduce duplication between branches.
            f32 = ir.F32Type.get()
            vec4_f32 = ir.VectorType.get([4], f32)
            acc_init = arith.constant_vector(0.0, vec4_f32)
            c0_i32 = 0
            i32_type = ir.IntegerType.get_signless(32)

            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            by = gpu.block_id("y")

            # Global layouts (A/B/C) are shared; FP8 ignores layout_b.
            layout_a = flir.make_layout((m_in, k_in), stride=(k_in, 1))
            layout_b = flir.make_layout((n_in, k_in), stride=(k_in, 1))
            layout_c = flir.make_layout((m_in, n_in), stride=(n_in, 1))

            base_ptr = allocator.get_base()
            lds_a = _state["lds_a_decl"](base_ptr).get()

            # LDS layouts (A always, B only used for FP16).
            layout_lds_a = flir.make_layout((tile_m, tile_k), stride=(lds_stride, 1))

            # Lane mapping (intrawave)
            wave_id = tx / 64
            lane_id = tx % 64
            lane_mod_16 = lane_id % 16
            lane_div_16 = lane_id / 16

            if dtype_config == DTYPE_FP8:
                # --- FP8 pipeline (unchanged structure) ---
                f8 = ir.Float8E4M3FNType.get()
                i32_type = ir.IntegerType.get_signless(32)
                vec8_f8 = ir.VectorType.get([8], f8)
                vec16_f8 = ir.VectorType.get([16], f8)
                vec1_i64 = ir.VectorType.get([1], ir.IntegerType.get_signless(64))
                vec4_i32 = ir.VectorType.get([4], i32_type)

                a_rsrc = buffer_ops.create_buffer_resource(arg_a)
                b_rsrc = buffer_ops.create_buffer_resource(arg_b)
                c_rsrc = buffer_ops.create_buffer_resource(arg_c)
                scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a)
                scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b)

                # Thread mapping for A global loads (same as preshuffle test)
                total_threads = 256
                bytes_per_thread_a = (tile_m * tile_k) // total_threads  # fp8 bytes
                max_bytes_per_load = 16
                num_a_loads = (bytes_per_thread_a + max_bytes_per_load - 1) // max_bytes_per_load

                vec_len_val = arith.constant(bytes_per_thread_a, index=True)
                linear_id = tx * vec_len_val
                c_tile_k_val = arith.constant(tile_k, index=True)
                row_a_local = linear_id / c_tile_k_val
                col_a_local = linear_id % c_tile_k_val
                bx_m = bx * tile_m
                row_a_global = bx_m + row_a_local

                # A base index (in bytes == in elements for fp8)
                coord_a_base = flir.make_coord(row_a_global, col_a_local)
                idx_a_base = flir.crd2idx(coord_a_base, layout_a)
                idx_a_base_div4 = idx_a_base / 4

                row_a_lds = lane_mod_16
                col_offset_base = lane_div_16 * 16

                # Tile decomposition
                k_unroll = tile_k // 32  # K32 micro-steps

                # Dynamic tiling along N (same as preshuffle test)
                num_waves = 4
                n_per_wave = 16 * n_repeat
                num_acc_n = n_repeat
                by_n = by * tile_n
                c_n_per_wave = arith.constant(n_per_wave, index=True)
                wave_mod_4 = wave_id % 4
                n_tile_base = wave_mod_4 * c_n_per_wave

                global_n_list = []
                for ni in range_constexpr(num_acc_n):
                    c_offset = arith.constant(ni * 16, index=True)
                    global_n = by_n + n_tile_base + c_offset + lane_mod_16
                    global_n_list.append(global_n)

                # A split loads (force dwordx4)
                vec_a_parts_lens = []
                rem = bytes_per_thread_a
                for _ in range_constexpr(num_a_loads):
                    curr_bytes = min(rem, max_bytes_per_load)
                    vec_a_parts_lens.append(curr_bytes)
                    rem -= curr_bytes

                def load_a_split(idx_div4):
                    parts = []
                    curr_off_i32 = 0
                    for i in range_constexpr(num_a_loads):
                        curr_bytes = vec_a_parts_lens[i]
                        curr_idx = idx_div4 if curr_off_i32 == 0 else (idx_div4 + curr_off_i32)
                        val = buffer_ops.buffer_load(a_rsrc, curr_idx, vec_width=4, dtype=i32_type)
                        parts.append(val)
                        curr_off_i32 += curr_bytes // 4
                    return parts

                vec_a_parts = load_a_split(idx_a_base_div4)
                accs = [acc_init] * (num_acc_n * m_repeat)

                def swizzle_xor_16b(row_idx, col_idx):
                    # XOR swizzle on K at 16B granularity (matches preshuffle test)
                    k_blocks16 = arith.constant(tile_k // 16, index=True)
                    row_mod = row_idx % k_blocks16
                    xor_mask = row_mod * 16
                    return arith.xori(col_idx, xor_mask)

                def load_b_pack_rowmajor(base_k, ki_step, ni):
                    # Match the preshuffle kernel’s (ki64, half, lane_div_16*16) addressing,
                    # but compute a normal row-major (N,K) byte offset.
                    ki64 = (ki_step // 2) * 64
                    ki64_val = arith.constant(ki64, index=True)
                    half = ki_step % 2
                    half_off = arith.constant(half * 8, index=True)
                    k_start = base_k + ki64_val + col_offset_base + half_off  # bytes

                    cK = arith.constant(K, index=True)
                    idx_bytes = (global_n_list[ni] * cK) + k_start
                    idx_i32 = idx_bytes / 4

                    b8 = buffer_ops.buffer_load(b_rsrc, idx_i32, vec_width=2, dtype=i32_type)
                    b_vec64 = vector.bitcast(vec1_i64, b8)
                    return vector.extract(b_vec64, static_position=[0], dynamic_position=[])

                def emit_tile(k_iv, accs_in, vec_a_in_parts, is_last_tile=False):
                    # Store A parts into LDS with XOR swizzle (same as preshuffle test)
                    curr_store_off = 0
                    for i in range_constexpr(num_a_loads):
                        val_vec = vec_a_in_parts[i]
                        curr_bytes = vec_a_parts_lens[i]
                        curr_i32 = curr_bytes // 4

                        col_0 = col_a_local + curr_store_off
                        col_sw = swizzle_xor_16b(row_a_local, col_0)
                        idx_0 = flir.crd2idx(
                            flir.make_coord(row_a_local, col_sw), layout_lds_a
                        )

                        if curr_i32 == 4:
                            val_16 = vector.bitcast(vec16_f8, val_vec)
                            vector.store(val_16, lds_a, [idx_0])
                        elif curr_i32 == 2:
                            val_2_i32 = vector.shuffle(val_vec, val_vec, [0, 1])
                            val_8 = vector.bitcast(vec8_f8, val_2_i32)
                            vector.store(val_8, lds_a, [idx_0])
                        else:
                            # Conservative fallback (rare for our default tiles)
                            vec_f8 = ir.VectorType.get([curr_bytes], f8)
                            val_1_i32 = vector.shuffle(val_vec, val_vec, [0])
                            val_f8 = vector.bitcast(vec_f8, val_1_i32)
                            vector.store(val_f8, lds_a, [idx_0])

                        curr_store_off += curr_bytes

                    gpu.barrier()

                    vec_a_next_parts = vec_a_in_parts
                    scales_pf = {}

                    if not is_last_tile:
                        next_k = k_iv + arith.constant(tile_k, index=True)
                        next_k_div4 = next_k / 4
                        next_idx_a_div4 = idx_a_base_div4 + next_k_div4
                        vec_a_next_parts = load_a_split(next_idx_a_div4)
                    else:
                        # Prefetch scales for epilogue
                        s_b_vals = []
                        for ni in range_constexpr(num_acc_n):
                            c_offset = arith.constant(ni * 16, index=True)
                            col_g = by_n + n_tile_base + c_offset + lane_mod_16
                            s_b_vals.append(buffer_ops.buffer_load(scale_b_rsrc, col_g, vec_width=1, dtype=f32))
                        scales_pf["s_b_vals"] = s_b_vals

                        s_a_vecs = []
                        row_off_base = lane_div_16 * 4
                        for mi in range_constexpr(m_repeat):
                            row_g_base = (bx_m + (mi * 16)) + row_off_base
                            s_a_vec = buffer_ops.buffer_load(scale_a_rsrc, row_g_base, vec_width=4, dtype=f32)
                            s_a_vecs.append(vector.bitcast(vec4_f32, s_a_vec))
                        scales_pf["s_a_vecs"] = s_a_vecs

                    current_accs_list = list(accs_in)

                    for ki_step in range_constexpr(k_unroll):
                        # Preload B packs once per ki_step
                        b_packs = []
                        for ni in range_constexpr(num_acc_n):
                            b_packs.append(load_b_pack_rowmajor(k_iv, ki_step, ni))

                        ki64 = (ki_step // 2) * 64
                        ki64_val = arith.constant(ki64, index=True)
                        half = ki_step % 2
                        half_off = arith.constant(half * 8, index=True)
                        col_base = col_offset_base + ki64_val

                        for mi in range_constexpr(m_repeat):
                            curr_row_a_lds = row_a_lds + arith.constant(mi * 16, index=True)
                            col_base_sw = swizzle_xor_16b(curr_row_a_lds, col_base)
                            col_sw = col_base_sw + half_off

                            idx_a = flir.crd2idx(
                                flir.make_coord(curr_row_a_lds, col_sw), layout_lds_a
                            )
                            loaded_a8 = vector.load_op(vec8_f8, lds_a, [idx_a])
                            a_vec64 = vector.bitcast(vec1_i64, loaded_a8)
                            a_pack = vector.extract(a_vec64, static_position=[0], dynamic_position=[])

                            for ni in range_constexpr(num_acc_n):
                                acc_idx = mi * num_acc_n + ni
                                current_accs_list[acc_idx] = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                    vec4_f32,
                                    [a_pack, b_packs[ni], current_accs_list[acc_idx], c0_i32, c0_i32, c0_i32],
                                )

                    gpu.barrier()
                    return current_accs_list, vec_a_next_parts, scales_pf

                c0 = arith.constant(0, index=True)
                c_k_main = k_in - arith.constant(tile_k, index=True)

                for k_iv in range(c0, c_k_main, arith.constant(tile_k, index=True)):
                    accs, vec_a_parts, _ = emit_tile(k_iv, accs, vec_a_parts, is_last_tile=False)

                final_accs, _, scales = emit_tile(c_k_main, accs, vec_a_parts, is_last_tile=True)

                s_b_vals = scales["s_b_vals"]
                s_a_vecs = scales["s_a_vecs"]

                # Epilogue: scale + store to f16
                for mi in range_constexpr(m_repeat):
                    row_base_m = bx_m + (mi * 16)
                    s_a_vec4 = s_a_vecs[mi]
                    for i in range_constexpr(4):
                        row_off = (lane_div_16 * 4) + i
                        row_g = row_base_m + row_off
                        s_a = vector.extract(s_a_vec4, static_position=[i], dynamic_position=[])
                        for ni in range_constexpr(num_acc_n):
                            acc = final_accs[mi * num_acc_n + ni]
                            val = vector.extract(acc, static_position=[i], dynamic_position=[])
                            s_b = s_b_vals[ni]
                            val_s = (val * s_a) * s_b
                            val_f16 = arith.trunc_f(T.f16(), val_s)
                            c_offset = arith.constant(ni * 16, index=True)
                            col_g = by_n + n_tile_base + c_offset + lane_mod_16
                            idx = flir.crd2idx(flir.make_coord(row_g, col_g), layout_c)
                            buffer_ops.buffer_store(val_f16, c_rsrc, idx)
            else:
                # --- FP16 pipeline ---
                mlir_dtype = _mlir_dtype()
                c0, c1 = 0, 1
                c_tile_k = arith.constant(tile_k, index=True)
                c16, c4, c64 = 16, 4, 64
                identity_map = ir.AffineMap.get_identity(1)

                c_tile_m = arith.index(tile_m)
                c_tile_n = arith.index(tile_n)

                c_lds_stride = arith.constant(lds_stride, index=True)
                stride_lds = flir.make_stride(c_lds_stride, c1)
                layout_lds_b = flir.make_layout(flir.make_shape(c_tile_n, c_tile_k), stride_lds)

                lds_b = _state["lds_b_decl"](base_ptr).get()

                vec_type = ir.VectorType.get([vec_load_size], mlir_dtype)
                pad_val = arith.constant(0.0, type=mlir_dtype)

                wave_row = wave_id // fp16_waves_n
                wave_col = wave_id % fp16_waves_n
                col_offset_base = lane_div_16 * c4

                vec4_f16 = ir.VectorType.get([4], mlir_dtype)

                def compute_16x16(acc_in, row_a_lds, row_b_lds):
                    acc_curr = acc_in
                    for ki in range_constexpr(0, tile_k, mfma_k):
                        col_lds = ki + col_offset_base
                        idx_a_mfma = flir.crd2idx(flir.make_coord(row_a_lds, col_lds), layout_lds_a)
                        idx_b_mfma = flir.crd2idx(flir.make_coord(row_b_lds, col_lds), layout_lds_b)
                        vec_a_load = vector.load_op(vec4_f16, lds_a, [idx_a_mfma])
                        vec_b_load = vector.load_op(vec4_f16, lds_b, [idx_b_mfma])
                        acc_curr = rocdl.mfma_f32_16x16x16f16(
                            vec4_f32,
                            [vec_a_load, vec_b_load, acc_curr, c0_i32, c0_i32, c0_i32],
                        )
                    return acc_curr

                accs = [acc_init for _ in range(fp16_m_reps * fp16_n_reps)]

                # K loop: compile-time unrolled to avoid dominance issues.
                for k_step in range_constexpr(0, K, tile_k):
                    k_curr = arith.constant(k_step, index=True)
                    c_vecs_per_row = arith.constant(tile_k // vec_load_size, index=True)
                    c_vec_load_size = arith.constant(vec_load_size, index=True)

                    c_vpt_a = arith.constant(fp16_vecs_per_thread_a, index=True)
                    vec_id_a_base = tx * c_vpt_a
                    bx_tile = bx * c_tile_m
                    for vi in range_constexpr(fp16_vecs_per_thread_a):
                        vec_id = vec_id_a_base + arith.constant(vi, index=True)
                        row_l = vec_id // c_vecs_per_row
                        col_v = vec_id % c_vecs_per_row
                        col_l = col_v * c_vec_load_size
                        row_g = bx_tile + row_l
                        col_g = k_curr + col_l
                        idx_a = flir.crd2idx(flir.make_coord(row_g, col_g), layout_a)
                        vec_a = vector.transfer_read(
                            vec_type, arg_a, [idx_a], identity_map, pad_val, [True]
                        )
                        lds_idx = (row_l * c_lds_stride) + col_l
                        vector.store(vec_a, lds_a, [lds_idx])

                    c_vpt_b = arith.constant(fp16_vecs_per_thread_b, index=True)
                    vec_id_b_base = tx * c_vpt_b
                    by_tile = by * c_tile_n
                    for vi in range_constexpr(fp16_vecs_per_thread_b):
                        vec_id = vec_id_b_base + arith.constant(vi, index=True)
                        row_l = vec_id // c_vecs_per_row
                        col_v = vec_id % c_vecs_per_row
                        col_l = col_v * c_vec_load_size
                        row_g = by_tile + row_l
                        col_g = k_curr + col_l
                        idx_b = flir.crd2idx(flir.make_coord(row_g, col_g), layout_b)
                        vec_b = vector.transfer_read(
                            vec_type, arg_b, [idx_b], identity_map, pad_val, [True]
                        )
                        lds_idx = (row_l * c_lds_stride) + col_l
                        vector.store(vec_b, lds_b, [lds_idx])

                    gpu.barrier()

                    for mi in range_constexpr(fp16_m_reps):
                        c_m_off = arith.constant(mi * fp16_base_m, index=True)
                        row_a_lds = c_m_off + (wave_row * c16) + lane_mod_16
                        for ni in range_constexpr(fp16_n_reps):
                            c_n_off = arith.constant(ni * fp16_base_n, index=True)
                            row_b_lds = c_n_off + (wave_col * c16) + lane_mod_16
                            acc_idx = (mi * fp16_n_reps) + ni
                            accs[acc_idx] = compute_16x16(accs[acc_idx], row_a_lds, row_b_lds)

                    gpu.barrier()

                lane_rem_16 = lane_id % c16
                for mi in range_constexpr(fp16_m_reps):
                    c_m_off = arith.constant(mi * fp16_base_m, index=True)
                    row_wave_base = c_m_off + (wave_row * c16)
                    row_base_g = (bx * c_tile_m) + row_wave_base
                    for ni in range_constexpr(fp16_n_reps):
                        c_n_off = arith.constant(ni * fp16_base_n, index=True)
                        col_wave_base = c_n_off + (wave_col * c16)
                        col_base_g = (by * c_tile_n) + col_wave_base
                        acc_idx = (mi * fp16_n_reps) + ni
                        final_acc = accs[acc_idx]
                        for i in range_constexpr(4):
                            val = vector.extract(final_acc, static_position=[i], dynamic_position=[])
                            val_f16 = arith.trunc_f(T.f16(), val)
                            row_offset = (lane_div_16 * c4) + arith.index(i)
                            row_g = row_base_g + row_offset
                            col_g = col_base_g + lane_rem_16
                            idx = flir.crd2idx(flir.make_coord(row_g, col_g), layout_c)
                            memref.store(val_f16, arg_c, [idx])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_c: lambda: T.memref(size_c, T.f16()),
            arg_a: lambda: T.memref(size_a, _mlir_dtype()),
            arg_b: lambda: T.memref(size_b, _mlir_dtype()),
            arg_scale_a: lambda: T.memref(M, T.f32()),
            arg_scale_b: lambda: T.memref(N, T.f32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(block_size_x, index=True)
            gx = arith.constant(M // tile_m, index=True)
            gy = arith.constant(N // tile_n, index=True)
            flir.gpu_ext.LaunchFuncOp(
                ["mfma_mod", "kernel"],
                grid_size=(gx, gy, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, m_in, n_in, k_in],
            )

    m = _MFMA()

    # Register Flir dialect for downstream passes (best-effort).
    try:
        import _flirPassesExt
        _flirPassesExt.register_dialect(m.module.context)
        print("✓ Registered Flir dialect")
    except Exception as e:
        print(f"Warning: Could not register Flir dialect: {e}")

    print("✓ MLIR module constructed via flir.MlirModule/@flir.kernel")
    
    # Set kernel attributes on the GPU function
    gpu_func_op = None
    for op in m.module.body.operations:
        if isinstance(op, ir.OpView) and op.OPERATION_NAME == "gpu.module":
            # op.body is a Region, need to access its first block
            body_block = op.body.blocks[0] if hasattr(op.body, 'blocks') else op.body
            for inner_op in body_block.operations:
                if hasattr(inner_op, 'OPERATION_NAME') and inner_op.OPERATION_NAME == "gpu.func":
                    gpu_func_op = inner_op
                    break
    
    if gpu_func_op:
        with m.module.context:
            gpu_func_op.attributes["rocdl.flat_work_group_size"] = ir.StringAttr.get(f"{block_size_x},{block_size_x}")
            gpu_func_op.attributes["rocdl.reqd_work_group_size"] = ir.DenseI32ArrayAttr.get([block_size_x, 1, 1])
            gpu_func_op.attributes["gpu.kernel"] = ir.UnitAttr.get()
    
    print("Compiling...")
    exe = flydsl.compile(m)
    print("✓ Compiled")

    print("Executing kernel...")
    
    # --- Torch Data Gen & Baseline ---
    device = torch.device('cuda')
    torch.manual_seed(42)

    # 1. Source Data (FP32)
    a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.randn(N, K, device=device, dtype=torch.float32)

    if use_scales:
        # FP8: Quantize with per-token scaling
        a_q, scale_a = pertoken_quant(a_fp32, quant_dtype=torch_dtype)  # (M, K)
        b_q, scale_b = pertoken_quant(b_fp32_t, quant_dtype=torch_dtype)  # (N, K)
        
        # Compute Baseline using AIter style (dequant + matmul)
        c_ref = run_torch(a_q, b_q, scale_a, scale_b, bias=None, dtype=torch.float32)
    else:
        # FP16: Direct conversion, no quantization
        a_q = a_fp32.to(torch_dtype)  # (M, K)
        b_q = b_fp32_t.to(torch_dtype)  # (N, K)
        
        # Compute Baseline: F.linear expects weight as (N, K), input as (M, K)
        c_ref = F.linear(a_q.to(torch.float32), b_q.to(torch.float32)).to(torch.float32)
        
        # Create dummy scale arrays (ones) for uniform kernel signature
        scale_a = torch.ones(M, device=device, dtype=torch.float32)
        scale_b = torch.ones(N, device=device, dtype=torch.float32)

    # 5. Run Kernel
    # Unified output: always FP16 for both FP8 and FP16 kernels.
    c_out_raw = torch.zeros((M, N), dtype=torch.float16, device=device)
    
    def launch_kernel():
        exe(c_out_raw, a_q, b_q, scale_a, scale_b, M, N, K)
    
    launch_kernel()
    torch.cuda.synchronize()
    
    # 7. Verify
    if dtype_config == DTYPE_FP8:
        verify_output(c_out_raw.to(torch.float32), c_ref, rtol=0.1, atol=0.1)
    else:
        verify_output(c_out_raw.to(torch.float32), c_ref, rtol=1e-2, atol=1e-2)
    
    # Benchmark
    warmup = 5
    runs = 20
    
    # Calculate bytes moved based on dtype
    if dtype_config == DTYPE_FP8:
        # fp8 inputs, f16 output
        bytes_moved = size_a * 1 + size_b * 1 + size_c * 2
    else:
        # fp16 inputs, f16 output
        bytes_moved = size_a * 2 + size_b * 2 + size_c * 2
    if use_scales:
        bytes_moved += (M + N) * 4  # Add scale arrays if used
    
    flops = 2 * M * N * K

    # Benchmark using run_perftest
    _, avg_us = run_perftest(
        launch_kernel,
        num_iters=runs,
        num_warmup=warmup,
    )
    
    avg_ms = avg_us / 1000
    gflops = flops / (avg_us / 1e6) / 1e9
    tflops = gflops / 1000.0
    bandwidth_tbs = bytes_moved / (avg_us / 1e6) / 1e12
    
    print(f"\n{'='*80}")
    print(f"Throughput: {avg_ms:.3f} ms, {tflops:.2f} TFLOPS, BW: {bandwidth_tbs:.2f} TB/s")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    torch.set_default_device('cuda')
    print("Running MFMA GEMM Tests with Multiple Dtypes")
    
    # Default: FP8 only (fast path). Enable FP16 explicitly if you’re working on it.
    test_mfma_gemm_flir(DTYPE_FP8, M=2560, N=5120, K=4096, tile_m=128, tile_n=128, tile_k=128)
    
    # FP16 fixed schedule uses 4 waves along N and a 64-col base tile.
    test_mfma_gemm_flir(DTYPE_FP16, M=16, N=5120, K=4096, tile_m=16, tile_n=64, tile_k=256)
    