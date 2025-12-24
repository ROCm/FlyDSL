#!/usr/bin/env python3
"""MFMA FP8 GEMM Test using Flir with B preshuffle (variable tiling) - Optimized Torch Version."""

import sys
import os
import logging
import functools

# Configure logging to show INFO level messages
logging.basicConfig(level=logging.INFO)



from pyflir.runtime.device import get_rocm_arch
import pyflir
import pyflir.dialects.ext.flir as flir
from pyflir.dialects.ext.python_control_flow import range_constexpr
from pyflir.utils import SmemAllocator
from tests.utils import pertoken_quant, shuffle_weight
from tests.test_common import verify_output, run_perftest
import torch
import torch.nn.functional as F
import pytest
from _mlir import ir
from _mlir.dialects import vector, memref, builtin, llvm
from pyflir.dialects.ext import arith, scf, gpu, buffer_ops
from _mlir.dialects import arith as _arith_mlir
import _mlir.dialects.rocdl as rocdl
import _mlir.extras.types as T
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

# Aiter imports
try:
    import aiter
    from aiter.ops.shuffle import shuffle_weight as aiter_shuffle_weight
    HAS_AITER = True
except ImportError:
    print("Warning: Aiter not found, skipping comparison")
    HAS_AITER = False

# Aiter benchmarking can be expensive (JIT build/tuning). Keep it off by default in CI.
RUN_AITER_BENCH = os.environ.get("FLIR_RUN_AITER_BENCH", "0") == "1"


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
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

def unwrap(v):
    if isinstance(v, int): return arith.constant(v, index=True).value
    while hasattr(v, "value"):
            v = v.value
    return v

@pytest.mark.parametrize("M, N, K, tile_m, tile_n, tile_k", [(1024, 7168, 2048, 128, 128, 128)])
def test_mfma_fp8_flir_preshuffle(M, N, K, tile_m, tile_n, tile_k):
    print("="*80)
    print(f"MFMA FP8 GEMM Test (Tile: {tile_m}x{tile_n}x{tile_k}) [Torch Optimized]")
    print("="*80)
    gpu_arch = get_rocm_arch()
    def _f8():
        return ir.Float8E4M3FNType.get()
    
            # NOTE: Kernel correctness here is sensitive to how we pack/load MFMA operands.
            # `mfma_f32_16x16x32_fp8_fp8` consumes K=32 per invocation (per wave).
            # This kernel intentionally issues **two** MFMA ops back-to-back into the same accumulator,
            # making a K=64 micro-step.
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    size_c = M * N
    size_a = M * K
    size_b = N * K
    
    # Vector width calc
    total_threads = 256
    elems_a_per_tile = tile_m * tile_k
    elems_per_thread_a = elems_a_per_tile // total_threads
    # Force min 16 bytes to avoid small vector load issues?
    # NO: If tile is small (16x128), 16 bytes load means we load 4096 bytes (2x tile size).
    # Threads 128-255 would write OOB to LDS (allocated for 2048).
    # So we must use exact size.
    bytes_per_thread_a = elems_per_thread_a 
    vec_width_a_i32 = bytes_per_thread_a // 4
    
    pad_k = 0 # Padding to avoid bank conflicts (stride 136 bytes -> bank inc 2)
    lds_stride = tile_k + pad_k
    
    class _MFMA(flir.MlirModule):
        GPU_MODULE_NAME = "mfma_mod"
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            _state["lds_a_decl"] = allocator.allocate_array(_f8(), tile_m * lds_stride)
            allocator.finalize()

        @flir.kernel
        def kernel_fixed(
            self: flir.T.i64,
            arg_c: lambda: T.memref(size_c, T.f16()),
            arg_a: lambda: T.memref(size_a, _f8()),
            arg_b: lambda: T.memref(size_b, _f8()),
            arg_scale_a: lambda: T.memref(M, T.f32()),
            arg_scale_b: lambda: T.memref(N, T.f32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            f8 = _f8()
            f32 = ir.F32Type.get()
            c_m = m_in
            c_n = n_in
            c_k = k_in
            c0 = arith.constant(0, index=True)
            c_tile_k = arith.constant(tile_k, index=True)
            
            i32_type = ir.IntegerType.get_signless(32)
            index_type = ir.IndexType.get()
            vec4_f32 = ir.VectorType.get([4], f32)
            vec8_f8 = ir.VectorType.get([8], f8)
            vec16_f8 = ir.VectorType.get([16], f8)
            vec1_i64 = ir.VectorType.get([1], ir.IntegerType.get_signless(64))
            vec2_i64 = ir.VectorType.get([2], ir.IntegerType.get_signless(64))
            vec2_i32 = ir.VectorType.get([2], i32_type)
            vec4_i32 = ir.VectorType.get([4], i32_type)
            
            vec_a_load_len = bytes_per_thread_a
            
            zero_attr = ir.DenseElementsAttr.get_splat(vec4_f32, ir.FloatAttr.get(f32, 0.0))
            acc_init = _arith_mlir.ConstantOp(vec4_f32, zero_attr).result
            
            layout_a = flir.make_layout((c_m, c_k), stride=(c_k, 1))
            layout_c = flir.make_layout((c_m, c_n), stride=(c_n, 1))
            
            c2 = arith.constant(2, index=True)
            c0_i32 = arith.i32(0)
            
            c32 = arith.constant(32, index=True)
            c16 = arith.constant(16, index=True)
            c4 = arith.constant(4, index=True)
            c8 = arith.constant(8, index=True)
            c512 = arith.constant(512, index=True)
            c64 = arith.constant(64, index=True)
            c1024 = arith.constant(1024, index=True)

            c_k0 = _arith_mlir.DivUIOp(unwrap(c_k), unwrap(c64)).result
            stride_n0 = _arith_mlir.MulIOp(unwrap(c_k0), unwrap(c1024)).result
            stride_b = (
                stride_n0,  # n0
                c1024,      # k0
                arith.constant(256, index=True),  # k1 (KLane)
                c16,        # n1
                arith.constant(1, index=True),    # k2
            )
            # Shape: (N0, K0, KLane, NLane, KPack)
            layout_b = flir.make_layout((_arith_mlir.DivUIOp(unwrap(c_n), unwrap(c16)).result,
                                          c_k0,
                                          arith.constant(4, index=True),
                                          c16,
                                          c16),
                                         stride=stride_b)
            
            shape_lds = flir.make_shape(tile_m, tile_k)
            stride_lds = flir.make_stride(lds_stride, 1)
            layout_lds = flir.make_layout(shape_lds, stride_lds)
            
            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            by = gpu.block_id("y")
            
            base_ptr = allocator.get_base()
            lds_a = _state["lds_a_decl"](base_ptr).get()
            
            a_rsrc = buffer_ops.create_buffer_resource(arg_a)
            b_rsrc = buffer_ops.create_buffer_resource(arg_b)
            c_rsrc = buffer_ops.create_buffer_resource(arg_c)
            scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a)
            scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b)
            
            tx_idx = unwrap(tx)
            vec_len_val = arith.constant(vec_a_load_len, index=True)
            linear_id = _arith_mlir.MulIOp(unwrap(tx_idx), unwrap(vec_len_val)).result
            
            c_tile_k_val = arith.constant(tile_k, index=True)
            row_a_local = _arith_mlir.DivUIOp(unwrap(linear_id), unwrap(c_tile_k_val)).result
            col_a_local = _arith_mlir.RemUIOp(unwrap(linear_id), unwrap(c_tile_k_val)).result
            
            bx_m = _arith_mlir.MulIOp(unwrap(bx), unwrap(arith.constant(tile_m, index=True))).result
            row_a_global = _arith_mlir.AddIOp(unwrap(bx_m), unwrap(row_a_local)).result
            by_n = _arith_mlir.MulIOp(unwrap(by), unwrap(arith.constant(tile_n, index=True))).result
            
            coord_store = flir.make_coord(unwrap(row_a_local), unwrap(col_a_local))
            lds_write_idx = flir.crd2idx(coord_store, layout_lds)
            
            wave_id = tx / 64
            lane_id = tx % 64
            lane_mod_16 = lane_id % 16
            lane_div_16 = lane_id / 16
            
            # MFMA operand mapping: match the known-good lane mapping (see `t1.py`)
            row_a_lds = lane_mod_16
            # CK mapping: lane_div_16 selects k1 (0..3), each k1 spans 16 fp8 within a 64-wide K block.
            col_offset_base = _arith_mlir.MulIOp(unwrap(lane_div_16), unwrap(c16)).result
            
            row_b_lds = lane_mod_16
            
            c2 = arith.constant(2, index=True)
            c0_i32 = arith.i32(0)
            
            coord_a_base = flir.make_coord(unwrap(row_a_global), unwrap(col_a_local))
            idx_a_base = flir.crd2idx(coord_a_base, layout_a)
            idx_a_base_div4 = _arith_mlir.DivUIOp(unwrap(idx_a_base), unwrap(c4)).result
            
            m_repeat = tile_m // 16
            # K64 micro-step: two MFMA(x32) per step
            k_unroll = tile_k // 64
            
            lds_a_indices = []
            
            # --- Dynamic Tiling Logic ---
            num_waves = 4
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16
            
            c_n_per_wave = arith.constant(n_per_wave, index=True)
            wave_mod_4 = _arith_mlir.RemUIOp(unwrap(wave_id), unwrap(c4)).result
            n_tile_base = _arith_mlir.MulIOp(unwrap(wave_mod_4), unwrap(c_n_per_wave)).result
            
            # Global N calc loop
            n_intra_list = []
            n_blk_list = []
            
            for i in range_constexpr(num_acc_n):
                offset = i * 16
                c_offset = arith.constant(offset, index=True)
                
                # global_n = by_n + n_tile_base + offset + row_b_lds
                tmp1 = _arith_mlir.AddIOp(unwrap(by_n), unwrap(n_tile_base)).result
                tmp2 = _arith_mlir.AddIOp(unwrap(tmp1), unwrap(c_offset)).result
                global_n = _arith_mlir.AddIOp(unwrap(tmp2), unwrap(row_b_lds)).result
                
                n_intra = _arith_mlir.RemUIOp(unwrap(global_n), unwrap(c16)).result
                n_blk = _arith_mlir.DivUIOp(unwrap(global_n), unwrap(c16)).result
                
                n_intra_list.append(n_intra)
                n_blk_list.append(n_blk)

            for mi in range_constexpr(m_repeat):
                mi_val = arith.constant(mi * 16, index=True)
                curr_row_a_lds = row_a_lds + mi_val
                
                for ki_step in range_constexpr(k_unroll):
                    # Each MFMA step advances K by 32 for fp8 x32
                    ki = ki_step * 32
                    ki_val = arith.constant(ki, index=True)
                    
                    col_lds = col_offset_base + ki_val
                    coord_a_lds = flir.make_coord(unwrap(curr_row_a_lds), unwrap(col_lds))
                    idx_a_mfma = flir.crd2idx(coord_a_lds, layout_lds)
                    idx_a_idx = unwrap(idx_a_mfma)
                    lds_a_indices.append(idx_a_idx)
                    
            acc_inits = [acc_init] * (num_acc_n * m_repeat)
            
            # --- B Load Logic ---
            def load_b_tile(base_k):
                b_packs = []
                k0_base = _arith_mlir.DivUIOp(unwrap(base_k), unwrap(c64)).result
                k1 = lane_div_16  # 0..3
                k2_base = c0

                for i in range_constexpr(num_acc_n):
                    n_intra = n_intra_list[i]
                    n_blk = n_blk_list[i]
                    for ki_step in range_constexpr(k_unroll):
                        k0 = _arith_mlir.AddIOp(unwrap(k0_base), unwrap(arith.constant(ki_step, index=True))).result
                        coord_b = flir.make_coord(n_blk, k0, k1, n_intra, k2_base)
                        idx_bytes = flir.crd2idx(coord_b, layout_b)
                        idx_i32 = _arith_mlir.DivUIOp(unwrap(idx_bytes), unwrap(c4)).result

                        b16 = buffer_ops.buffer_load(b_rsrc, idx_i32, vec_width=4, dtype=i32_type)
                        b_vec128 = vector.BitCastOp(vec2_i64, b16).result
                        b0_pack = unwrap(vector.ExtractOp(b_vec128, static_position=[0], dynamic_position=[]).result)
                        b1_pack = unwrap(vector.ExtractOp(b_vec128, static_position=[1], dynamic_position=[]).result)
                        b_packs.append(b0_pack)
                        b_packs.append(b1_pack)
                return b_packs

            # Split A loads logic
            max_bytes_per_load = 16
            num_a_loads = (bytes_per_thread_a + max_bytes_per_load - 1) // max_bytes_per_load
            
            vec_a_parts_types = []
            vec_a_parts_lens = []
            
            remaining_bytes = bytes_per_thread_a
            for i in range_constexpr(num_a_loads):
                curr_bytes = min(remaining_bytes, max_bytes_per_load)
                vec_a_parts_lens.append(curr_bytes)
                # Force global dwordx4 loads: carry vector<4xi32> through the loop state.
                # For parts < 16B (e.g. 8B), we still load 16B but only store/use the first curr_bytes.
                vec_a_parts_types.append(vec4_i32)
                remaining_bytes -= curr_bytes

            # Helper to load A (split)
            def load_a_split(idx_div4):
                parts = []
                curr_off_i32 = 0
                for i in range_constexpr(num_a_loads):
                    curr_bytes = vec_a_parts_lens[i]
                    
                    curr_idx = idx_div4
                    if curr_off_i32 > 0:
                        curr_idx = _arith_mlir.AddIOp(unwrap(idx_div4), unwrap(arith.constant(curr_off_i32, index=True))).result
                    
                    # Force global dwordx4 load (16B).
                    val = buffer_ops.buffer_load(a_rsrc, curr_idx, vec_width=4, dtype=i32_type)
                    parts.append(val)
                    # Advance by the *logical* i32s covered by this part (can be < 4).
                    curr_off_i32 += (curr_bytes // 4)
                return parts

            # Initial Loads (A and B)
            vec_a_inits = load_a_split(idx_a_base_div4)
            b_vals_init_raw = load_b_tile(c0)
            
            # Flatten b_vals for loop-carried state
            b_vals_init = b_vals_init_raw
            # b_vals_init = []
            # for b_list in b_vals_init_raw:
            #     b_vals_init.extend(b_list)
            
            # Loop-carried state (lists of MLIR Values)
            accs = acc_inits
            vec_a_parts = vec_a_inits
            b_vals = b_vals_init
            
            # Helper to emit tile body
            def emit_tile(k_iv, accs_in, vec_a_in_parts, b_vals_in, is_last_tile=False):
                # Store A to LDS (split) with Swizzle (Bit 3 <-> Bit 5)
                # To place K=0..7 at 0, K=32..39 at 8
                
                curr_store_off = 0
                c8 = arith.constant(8, index=True)
                c32 = arith.constant(32, index=True)
                
                for i in range_constexpr(num_a_loads):
                    val_vec = vec_a_in_parts[i]
                    # `val_vec` is a vector<i32> of length (curr_bytes/4).
                    # For many tiles (e.g. 16x128), each thread only owns 8 bytes => vector<2xi32>.
                    # Do not unconditionally assume 16 bytes here.
                    curr_bytes = vec_a_parts_lens[i]
                    curr_i32 = curr_bytes // 4
                    
                    # Addr 0 (offset 0)
                    col_0 = _arith_mlir.AddIOp(unwrap(col_a_local), unwrap(arith.constant(curr_store_off, index=True))).result
                    # Swizzle col_0: swap bit 3 and 5
                    # bit3 = (x & 8) >> 3; bit5 = (x & 32) >> 5
                    # x_new = x & ~40 | (bit3 << 5) | (bit5 << 3)
                    
                    def swizzle_idx(row_idx, col_idx):
                        c16 = arith.constant(16, index=True)
                        # CK-style xor-with-modulo swizzle on K, applied at 16B granularity so
                        # vector<16xf8> loads preserve intra-vector order.
                        # NOTE: we rely on tile_k being a power-of-two so xor stays in-bounds.
                        k_blocks16 = arith.constant(tile_k // 16, index=True)
                        row_mod = _arith_mlir.RemUIOp(unwrap(row_idx), unwrap(k_blocks16)).result
                        xor_mask = _arith_mlir.MulIOp(unwrap(row_mod), unwrap(c16)).result
                        return _arith_mlir.XOrIOp(unwrap(col_idx), unwrap(xor_mask)).result

                    col_swizzled_0 = swizzle_idx(row_a_local, col_0)
                    coord_store_0 = flir.make_coord(unwrap(row_a_local), unwrap(col_swizzled_0))
                    idx_0 = flir.crd2idx(coord_store_0, layout_lds)

                    if curr_i32 == 4:
                        val_16 = vector.BitCastOp(vec16_f8, val_vec).result
                        vector.StoreOp(val_16, lds_a, [unwrap(idx_0)])
                    elif curr_i32 == 2:
                        # 8B chunk: single store.
                        val_2_i32 = vector.ShuffleOp(val_vec, val_vec, [0, 1]).result
                        val_8 = vector.BitCastOp(vec8_f8, val_2_i32).result
                        vector.StoreOp(val_8, lds_a, [unwrap(idx_0)])
                    else:
                        # Fallback: support rare cases (e.g. 4B) without corrupting LDS.
                        vec_f8 = ir.VectorType.get([curr_bytes], f8)
                        if curr_bytes <= 4:
                            val_1_i32 = vector.ShuffleOp(val_vec, val_vec, [0]).result
                            val_f8 = vector.BitCastOp(vec_f8, val_1_i32).result
                        else:
                            val_2_i32 = vector.ShuffleOp(val_vec, val_vec, [0, 1]).result
                            val_f8 = vector.BitCastOp(vec_f8, val_2_i32).result
                        vector.StoreOp(val_f8, lds_a, [unwrap(idx_0)])

                    curr_store_off += curr_bytes

                gpu.barrier()
                
                vec_a_next_parts = vec_a_in_parts # Default placeholder
                b_vals_next = b_vals_in # Default placeholder
                scales_pf = {}
                
                if not is_last_tile:
                    # Next K calculations
                    next_k = _arith_mlir.AddIOp(unwrap(k_iv), unwrap(c_tile_k)).result
                    
                    # Prefetch A
                    next_k_div4 = _arith_mlir.DivUIOp(unwrap(next_k), unwrap(c4)).result
                    next_idx_a_div4 = _arith_mlir.AddIOp(unwrap(idx_a_base_div4), unwrap(next_k_div4)).result
                    vec_a_next_parts = load_a_split(next_idx_a_div4)
                    
                    # Prefetch B
                    b_vals_next = load_b_tile(next_k)
                    # b_vals_next_raw = load_b_tile(next_k)
                    # b_vals_next = []
                    # for b_list in b_vals_next_raw:
                    #     b_vals_next.extend(b_list)
                else:
                    # --- PREFETCH SCALES (Last Iteration) ---
                    # Prefetch Scale B (invariant for thread)
                    s_b_vals = []
                    for ni in range_constexpr(num_acc_n):
                        offset = ni * 16
                        c_offset = arith.constant(offset, index=True)
                        tmp1 = _arith_mlir.AddIOp(unwrap(by_n), unwrap(n_tile_base)).result
                        tmp2 = _arith_mlir.AddIOp(unwrap(tmp1), unwrap(c_offset)).result
                        col_g = _arith_mlir.AddIOp(unwrap(tmp2), unwrap(lane_mod_16)).result
                        
                        val = buffer_ops.buffer_load(scale_b_rsrc, col_g, vec_width=1, dtype=f32)
                        s_b_vals.append(val)
                    
                    scales_pf['s_b_vals'] = s_b_vals
                    scales_pf['s_a_vecs'] = []
                    
                    # Pre-load Scale A vectors
                    row_off_base = (lane_div_16 * 4)
                    for mi in range_constexpr(m_repeat):
                        row_base_m = bx_m + (mi * 16)
                        row_g_base = row_base_m + row_off_base
                        s_a_vec = buffer_ops.buffer_load(scale_a_rsrc, row_g_base, vec_width=4, dtype=f32)
                        s_a_vec4 = vector.BitCastOp(vec4_f32, s_a_vec).result
                        scales_pf['s_a_vecs'].append(s_a_vec4)

                current_accs_list = list(accs_in)
                
                # Loop Swap: Iterate K_step (outer) -> MI (inner)
                # To reuse B
                for ki_step in range_constexpr(k_unroll):
                    # b_vals_in already contains i64 packs per ni:
                    # for each ni: [b0(step0), b1(step0), b0(step1), b1(step1), ...]
                    b0_packs = []
                    b1_packs = []
                    for ni in range_constexpr(num_acc_n):
                        base = ni * (2 * k_unroll)
                        b0_packs.append(b_vals_in[base + (2 * ki_step + 0)])
                        b1_packs.append(b_vals_in[base + (2 * ki_step + 1)])

                    ki = ki_step * 64
                    ki_val = arith.constant(ki, index=True)
                    col_lds0 = col_offset_base + ki_val
                    col_lds1 = col_offset_base + ki_val + c32
                    
                    def swizzle_idx(row_idx, col_idx):
                        c16 = arith.constant(16, index=True)
                        k_blocks16 = arith.constant(tile_k // 16, index=True)
                        row_mod = _arith_mlir.RemUIOp(unwrap(row_idx), unwrap(k_blocks16)).result
                        xor_mask = _arith_mlir.MulIOp(unwrap(row_mod), unwrap(c16)).result
                        return _arith_mlir.XOrIOp(unwrap(col_idx), unwrap(xor_mask)).result

                    for mi in range_constexpr(m_repeat):
                        mi_val = arith.constant(mi * 16, index=True)
                        curr_row_a_lds = row_a_lds + mi_val
                        
                        # Read A from LDS using the same (row,col)->(row,col') xor swizzle as the store.
                        col_lds0_swizzled = swizzle_idx(curr_row_a_lds, col_lds0)
                        coord_a0 = flir.make_coord(unwrap(curr_row_a_lds), unwrap(col_lds0_swizzled))
                        idx_a0 = flir.crd2idx(coord_a0, layout_lds)
                        # idx_a0 is already index-typed.
                        idx_a0_idx = unwrap(idx_a0)

                        loaded_a16 = vector.LoadOp(vec16_f8, lds_a, [unwrap(idx_a0_idx)]).result
                        a_vec128 = vector.BitCastOp(vec2_i64, loaded_a16).result
                        a0_pack = vector.ExtractOp(a_vec128, static_position=[0], dynamic_position=[]).result
                        a1_pack = vector.ExtractOp(a_vec128, static_position=[1], dynamic_position=[]).result
                        
                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            curr_acc = current_accs_list[acc_idx]
                            b0_pack = b0_packs[ni]
                            b1_pack = b1_packs[ni]

                            acc0 = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                vec4_f32,
                                [unwrap(a0_pack), unwrap(b0_pack), unwrap(curr_acc), unwrap(c0_i32), unwrap(c0_i32), unwrap(c0_i32)],
                            ).result
                            acc1 = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                vec4_f32,
                                [unwrap(a1_pack), unwrap(b1_pack), unwrap(acc0), unwrap(c0_i32), unwrap(c0_i32), unwrap(c0_i32)],
                            ).result
                            current_accs_list[acc_idx] = acc1
                
                gpu.barrier()
                return current_accs_list, vec_a_next_parts, b_vals_next, scales_pf

            # Main Loop (runs 0 to K-tile_k)
            # Peel off the last iteration
            c_k_main = _arith_mlir.SubIOp(unwrap(c_k), unwrap(c_tile_k)).result

            # Python `for` + reassignment is auto-lowered into scf.for with
            # iter_args/yield/results by `lower_range_for_loops` (supports list/tuple pytree state).
            for k_iv in range(c0, c_k_main, c_tile_k):
                accs, vec_a_parts, b_vals, _ = emit_tile(k_iv, accs, vec_a_parts, b_vals, is_last_tile=False)
            
            # Epilogue: Run last tile and prefetch scales
            final_accs, _, _, scales = emit_tile(c_k_main, accs, vec_a_parts, b_vals, is_last_tile=True)

            
            s_b_vals = scales['s_b_vals']
            s_a_vecs = scales['s_a_vecs']
            
            for mi in range_constexpr(m_repeat):
                row_base_m = bx_m + (mi * 16)
                s_a_vec4 = s_a_vecs[mi]
                
                for i in range_constexpr(4):
                    row_off = (lane_div_16 * 4) + i
                    row_g = row_base_m + row_off
                    
                    s_a = vector.ExtractOp(s_a_vec4, static_position=[i], dynamic_position=[]).result
                    
                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        acc = final_accs[acc_idx]
                        
                        val = vector.ExtractOp(acc, [], [i]).result
                        
                        offset = ni * 16
                        c_offset = arith.constant(offset, index=True)
                        tmp1 = _arith_mlir.AddIOp(unwrap(by_n), unwrap(n_tile_base)).result
                        tmp2 = _arith_mlir.AddIOp(unwrap(tmp1), unwrap(c_offset)).result
                        col_g = _arith_mlir.AddIOp(unwrap(tmp2), unwrap(lane_mod_16)).result
                        
                        s_b = s_b_vals[ni]
                        
                        val_s = _arith_mlir.MulFOp(unwrap(val), unwrap(s_a)).result
                        val_s = _arith_mlir.MulFOp(unwrap(val_s), unwrap(s_b)).result
                        val_f16 = _arith_mlir.TruncFOp(T.f16(), unwrap(val_s)).result
                        
                        idx = flir.crd2idx(flir.make_coord(unwrap(row_g), unwrap(col_g)), layout_c)
                        buffer_ops.buffer_store(val_f16, c_rsrc, idx)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_c: lambda: T.memref(size_c, T.f16()),
            arg_a: lambda: T.memref(size_a, _f8()),
            arg_b: lambda: T.memref(size_b, _f8()),
            arg_scale_a: lambda: T.memref(M, T.f32()),
            arg_scale_b: lambda: T.memref(N, T.f32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            # Describe launch in a host-side function (IR only).
            c1 = arith.constant(1, index=True).value
            bdx = arith.constant(256, index=True).value
            gx = arith.constant(M // tile_m, index=True).value
            gy = arith.constant(N // tile_n, index=True).value
            flir.gpu_ext.LaunchFuncOp(
                ["mfma_mod", "kernel_fixed"],
                grid_size=(gx, gy, c1),
                block_size=(bdx, c1, c1),
                # Ensure we pass raw MLIR Values (not ArithValue wrappers) to gpu.launch_func.
                kernel_operands=[
                    unwrap(arg_c),
                    unwrap(arg_a),
                    unwrap(arg_b),
                    unwrap(arg_scale_a),
                    unwrap(arg_scale_b),
                    unwrap(m_in),
                    unwrap(n_in),
                    unwrap(k_in),
                ],
            )

    # Request occupancy hint: waves-per-eu=2
    # Use a unique kernel_name so IR/asm dumps don't get overwritten by other tests.
    m = _MFMA()
    try:
        import _flirPassesExt
        _flirPassesExt.register_dialect(m.module.context)
        print("✓ Registered Flir dialect")
    except Exception as e:
        print(f"Warning: Could not register Flir dialect: {e}")

    exe = pyflir.compile(m)
    print("✓ Compiled")
    
    grid_x = M // tile_m
    grid_y = N // tile_n
    
    # --- Torch Data Gen & Baseline (AIter Style) ---
    device = torch.device('cuda')
    
    # 1. Source Data (FP32)
    torch.manual_seed(42)  # For reproducibility
    a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.randn(N, K, device=device, dtype=torch.float32)  # (N, K) for weight
    
    # 2. Per-token Quantize to FP8 (E4M3)
    a_q_fp8, scale_a = pertoken_quant(a_fp32, quant_dtype=torch.float8_e4m3fnuz)  # (M, K)
    b_q_fp8, scale_b = pertoken_quant(b_fp32_t, quant_dtype=torch.float8_e4m3fnuz)  # (N, K)

    # When using forced global dwordx4 (16B) loads, some threads will over-read beyond the
    # logical (M*K) / (N*K) region. Pad the underlying storage so the over-read stays in-bounds.
    # This preserves correctness because we only *use* the required bytes.
    PAD_ELEMS = 64  # bytes for fp8; generous guard for safety
    if a_q_fp8.is_contiguous():
        a_flat = a_q_fp8.view(-1)
    else:
        a_flat = a_q_fp8.contiguous().view(-1)
    a_storage = torch.empty(a_flat.numel() + PAD_ELEMS, device=device, dtype=a_q_fp8.dtype)
    a_storage[: a_flat.numel()] = a_flat
    a_q_fp8 = a_storage[: a_flat.numel()].view(M, K)

    if b_q_fp8.is_contiguous():
        b_flat = b_q_fp8.view(-1)
    else:
        b_flat = b_q_fp8.contiguous().view(-1)
    b_storage = torch.empty(b_flat.numel() + PAD_ELEMS, device=device, dtype=b_q_fp8.dtype)
    b_storage[: b_flat.numel()] = b_flat
    b_q_fp8 = b_storage[: b_flat.numel()].view(N, K)
    
    # Keep the aiter-style preshuffle for optional aiter benchmark at the end.
    b_shuffled = shuffle_weight(b_q_fp8)  # (N, K) -> aiter shuffled layout
    
    # 4. Compute Baseline using AIter style (dequant + matmul)
    c_ref = run_torch(a_q_fp8, b_q_fp8, scale_a, scale_b, bias=None, dtype=torch.float32)
    # 5. Run Kernel (f16 output, in-kernel scaling)
    c_out_raw = torch.zeros((M, N), dtype=torch.float16, device=device)
    
    def launch_kernel(c, a, b, sa, sb):
        exe(c, a, b, sa, sb, M, N, K)

    _, us = run_perftest(launch_kernel, c_out_raw, a_q_fp8, b_shuffled, scale_a, scale_b)
    torch.cuda.synchronize()
    c_out_scaled = c_out_raw.to(torch.float32)

    # 7. Verify
    assert verify_output(c_out_scaled, c_ref, rtol=0.1, atol=0.1)

    # Benchmark
    bytes_moved = size_a + size_b + size_c * 2 + (M + N) * 4
    flops = 2 * M * N * K
    tflops = flops / (us / 1e6) / 1e12
    bw = bytes_moved / 1e9 / (us / 1e6)
    print(f"Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {bw:.2f} GB/s")

    if HAS_AITER and RUN_AITER_BENCH:
        print("-" * 40)
        print("Running Aiter Benchmark...")

        def launch_aiter(a, b, sa, sb):
            return aiter.gemm_a8w8_bpreshuffle(
                a,
                b,
                sa,
                sb,
                None,  # bias
                torch.float16,
            )

        c_aiter, us1 = run_perftest(launch_aiter, a_q_fp8, b_shuffled, scale_a, scale_b)
        verify_output(c_aiter.to(torch.float32), c_ref, rtol=0.1, atol=0.1)

        tflops_aiter = flops / (us1 / 1e6) / 1e12
        bw_aiter = bytes_moved / 1e9 / (us1 / 1e6)
        print(f"Aiter: {us1:.1f} us, {tflops_aiter:.2f} TFLOPS, BW: {bw_aiter:.2f} GB/s")
        print(f"Speedup vs Aiter: {tflops / tflops_aiter:.2f}x, us {us1:.1f} vs {us:.1f}")
        print("-" * 40)
    elif HAS_AITER and not RUN_AITER_BENCH:
        print("-" * 40)
        print("Skipping Aiter benchmark (set FLIR_RUN_AITER_BENCH=1 to enable)")
        print("-" * 40)
    # Executor manages module lifetime.


if __name__ == "__main__":
    torch.set_default_device('cuda')
    # Test cases
    print("Running Tiling Tests...")
    
    test_mfma_fp8_flir_preshuffle(1024, 7168, 2048, tile_m=128, tile_n=128, tile_k=128)
    # test_mfma_fp8_flir_preshuffle(16, 7168, 2048, tile_m=16, tile_n=64, tile_k=256) 
    # test_mfma_fp8_flir_preshuffle(16, 7168, 2048, tile_m=16, tile_n=256, tile_k=256)
    # test_mfma_fp8_flir_preshuffle(1024, 7168, 2048, tile_m=128, tile_n=128, tile_k=128)
    # test_mfma_fp8_flir_preshuffle(32, 7168, 2048, tile_m=32, tile_n=128, tile_k=256)

    # Work around a known finalization crash in some environments where native GPU/MLIR
    # bindings execute callbacks while the interpreter is shutting down and the GIL is
    # already released. Hard-exiting avoids running Python finalizers.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
