#!/usr/bin/env python3
"""MFMA FP8 GEMM Test using Rocir with B preshuffle (variable tiling) - Optimized Torch Version."""

import sys
import os
import logging
import functools

# Configure logging to show INFO level messages
logging.basicConfig(level=logging.INFO)

sys.path.insert(0, os.path.join(os.environ.get('MLIR_PATH'), 'tools/mlir/python_packages/mlir_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../build/python_bindings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../python'))

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
import rocdsl.dialects.ext.rocir as rocir
from rocdsl.utils import SmemAllocator
from tests.utils import compile_to_hsaco, perftest, pertoken_quant, shuffle_weight, verify_output
import torch
import torch.nn.functional as F
import pytest
from mlir import ir
from mlir.dialects import vector, memref, builtin, llvm
from rocdsl.dialects.ext import arith, scf, gpu, buffer_ops
from mlir.dialects import arith as _arith_mlir
import mlir.dialects.rocdl as rocdl
import mlir.extras.types as T
from hip import hip
import ctypes


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
    while hasattr(v, "value") or hasattr(v, "_value"):
        if hasattr(v, "_value"):
            v = v._value
        elif hasattr(v, "value"):
            v = v.value
    return v

@pytest.mark.parametrize("M, N, K", [(16, 4096, 2048)])
def test_mfma_fp8_rocir_preshuffle(M, N, K, tile_m=16, tile_n=128, tile_k=128):
    print("="*80)
    print(f"MFMA FP8 GEMM Test (Tile: {tile_m}x{tile_n}x{tile_k}) [Torch Optimized]")
    print("="*80)
    gpu_arch = get_hip_arch()
    ctx = RAIIMLIRContextModule()
    try:
        import _rocirPassesExt
        _rocirPassesExt.register_dialect(ctx.module.context)
        print("✓ Registered Rocir dialect")
    except Exception as e:
        print(f"Warning: Could not register Rocir dialect: {e}")
    
    f8 = ir.Float8E4M3FNType.get()
    f32 = ir.F32Type.get()
    
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
    
    allocator = SmemAllocator(ctx, arch=gpu_arch)
    lds_a_decl = allocator.allocate_array(f8, tile_m * tile_k)
    
    @gpu.module("mfma_mod", [f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'])
    def gpu_mod():
        allocator.finalize()
        
        @gpu.func(emit=True)
        def kernel_fixed(
            arg_c: T.memref(size_c, T.f16()),
            arg_a: T.memref(size_a, f8),
            arg_b: T.memref(size_b, f8),
            arg_scale_a: T.memref(M, T.f32()),
            arg_scale_b: T.memref(N, T.f32()),
            m_in: T.index(),
            n_in: T.index(),
            k_in: T.index()
        ):
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
            
            vec_a_load_len = bytes_per_thread_a
            vec_a_load_type = ir.VectorType.get([vec_a_load_len], f8)
            
            zero_attr = ir.DenseElementsAttr.get_splat(vec4_f32, ir.FloatAttr.get(f32, 0.0))
            acc_init = _arith_mlir.ConstantOp(vec4_f32, zero_attr).result
            
            layout_a = rocir.make_layout((c_m, c_k), stride=(c_k, 1))
            layout_c = rocir.make_layout((c_m, c_n), stride=(c_n, 1))
            
            c32 = arith.constant(32, index=True)
            c16 = arith.constant(16, index=True)
            c4 = arith.constant(4, index=True)
            
            k_blocks = _arith_mlir.DivUIOp(unwrap(c_k), unwrap(c32)).result
            n_blocks = _arith_mlir.DivUIOp(unwrap(c_n), unwrap(c16)).result
            c512 = arith.constant(512, index=True)
            s_n_blocks = _arith_mlir.MulIOp(unwrap(c512), unwrap(k_blocks)).result
            
            stride_b = (c16, s_n_blocks, arith.constant(1, index=True), arith.constant(256, index=True), arith.constant(512, index=True))
            layout_b = rocir.make_layout((c16, n_blocks, c16, arith.constant(2, index=True), k_blocks), stride=stride_b)
            
            shape_lds = rocir.make_shape(tile_m, tile_k)
            stride_lds = rocir.make_stride(tile_k, 1)
            layout_lds = rocir.make_layout(shape_lds, stride_lds)
            
            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            by = gpu.block_id("y")
            
            base_ptr = allocator.get_base()
            lds_a = lds_a_decl(base_ptr).get()
            
            a_rsrc = buffer_ops.create_buffer_resource(arg_a)
            b_rsrc = buffer_ops.create_buffer_resource(arg_b)
            c_rsrc = buffer_ops.create_buffer_resource(arg_c)
            scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a)
            scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b)
            
            tx_idx = _arith_mlir.IndexCastOp(index_type, unwrap(tx)).result
            vec_len_val = arith.constant(vec_a_load_len, index=True)
            linear_id = _arith_mlir.MulIOp(unwrap(tx_idx), unwrap(vec_len_val)).result
            
            c_tile_k_val = arith.constant(tile_k, index=True)
            row_a_local = _arith_mlir.DivUIOp(unwrap(linear_id), unwrap(c_tile_k_val)).result
            col_a_local = _arith_mlir.RemUIOp(unwrap(linear_id), unwrap(c_tile_k_val)).result
            
            bx_m = _arith_mlir.MulIOp(unwrap(bx), unwrap(arith.constant(tile_m, index=True))).result
            row_a_global = _arith_mlir.AddIOp(unwrap(bx_m), unwrap(row_a_local)).result
            by_n = _arith_mlir.MulIOp(unwrap(by), unwrap(arith.constant(tile_n, index=True))).result
            
            coord_store = rocir.make_coord(unwrap(row_a_local), unwrap(col_a_local))
            lds_write_idx = rocir.crd2idx(coord_store, layout_lds)
            
            wave_id = tx / 64
            lane_id = tx % 64
            lane_mod_16 = lane_id % 16
            lane_div_16 = lane_id / 16
            
            row_a_lds = lane_mod_16
            col_offset_base = lane_div_16 * 8
            row_b_lds = lane_mod_16
            n_tile_base = (wave_id % 4) * 32
            
            c2 = arith.constant(2, index=True)
            c0_i32 = arith.i32(0)
            
            coord_a_base = rocir.make_coord(unwrap(row_a_global), unwrap(col_a_local))
            idx_a_base = rocir.crd2idx(coord_a_base, layout_a)
            idx_a_base_div4 = _arith_mlir.DivUIOp(unwrap(idx_a_base), unwrap(c4)).result
            
            m_repeat = tile_m // 16
            k_unroll = tile_k // 32
            
            lds_a_indices = []
            
            n_off0 = arith.constant(0, index=True)
            n_off1 = arith.constant(16, index=True)
            
            global_n_mfma0 = by_n + n_tile_base + n_off0 + row_b_lds
            global_n_mfma1 = by_n + n_tile_base + n_off1 + row_b_lds
                    
            n_intra0 = global_n_mfma0 % c16
            n_blk0 = global_n_mfma0 // c16
            n_intra1 = global_n_mfma1 % c16
            n_blk1 = global_n_mfma1 // c16

            for mi in range(m_repeat):
                mi_val = arith.constant(mi * 16, index=True)
                curr_row_a_lds = row_a_lds + mi_val
                
                for ki_step in range(k_unroll):
                    ki = ki_step * 32
                    ki_val = arith.constant(ki, index=True)
                    
                    col_lds = col_offset_base + ki_val
                    coord_a_lds = rocir.make_coord(unwrap(curr_row_a_lds), unwrap(col_lds))
                    idx_a_mfma = rocir.crd2idx(coord_a_lds, layout_lds)
                    idx_a_idx = _arith_mlir.IndexCastOp(index_type, unwrap(idx_a_mfma)).result
                    lds_a_indices.append(idx_a_idx)
                    
            acc_inits = [acc_init] * (2 * m_repeat)
            
            v_i32_init = buffer_ops.buffer_load(a_rsrc, idx_a_base_div4, vec_width=vec_width_a_i32, dtype=i32_type)
            vec_a_init = vector.BitCastOp(vec_a_load_type, v_i32_init).result
            
            iter_args = acc_inits + [vec_a_init]
            
            for_op = scf.ForOp(c0, c_k, c_tile_k, iter_args)
            
            with ir.InsertionPoint(for_op.body):
                k_iv = for_op.induction_variable
                args = for_op.inner_iter_args
                accs_iter = args[:-1]
                vec_a_iter = args[-1]
                


                vector.StoreOp(vec_a_iter, lds_a, [unwrap(lds_write_idx)])
                gpu.barrier()
                
                next_k = _arith_mlir.AddIOp(unwrap(k_iv), unwrap(c_tile_k)).result
                next_k_div4 = _arith_mlir.DivUIOp(unwrap(next_k), unwrap(c4)).result
                next_idx_a_div4 = _arith_mlir.AddIOp(unwrap(idx_a_base_div4), unwrap(next_k_div4)).result
                v_i32_next = buffer_ops.buffer_load(a_rsrc, next_idx_a_div4, vec_width=vec_width_a_i32, dtype=i32_type)
                vec_a_next = vector.BitCastOp(vec_a_load_type, v_i32_next).result
                
                current_accs_list = list(accs_iter)
                
                for mi in range(m_repeat):
                    for ki_step in range(k_unroll):
                        ki = ki_step * 32
                        ki_val = arith.constant(ki, index=True)
                        col_lds = col_offset_base + ki_val
                        
                        u_idx = mi * k_unroll + ki_step
                        
                        loaded_a = vector.LoadOp(vec8_f8, lds_a, [unwrap(lds_a_indices[u_idx])]).result
                        a_vec64 = vector.BitCastOp(vec1_i64, loaded_a).result
                        a_pack = vector.ExtractOp(a_vec64, static_position=[0], dynamic_position=[]).result
                        
                        acc_idx0 = mi * 2
                        acc_idx1 = mi * 2 + 1
                        
                        curr_acc0 = current_accs_list[acc_idx0]
                        curr_acc1 = current_accs_list[acc_idx1]
                        
                        # Compute global K position for this sub-tile (use MLIR ops explicitly)
                        col_lds_global = _arith_mlir.AddIOp(unwrap(col_lds), unwrap(k_iv)).result
                        k_intra = _arith_mlir.RemUIOp(unwrap(col_lds_global), unwrap(c16)).result
                        k_rem = _arith_mlir.DivUIOp(unwrap(col_lds_global), unwrap(c16)).result
                        k_pack = _arith_mlir.RemUIOp(unwrap(k_rem), unwrap(c2)).result
                        k_blk_local = _arith_mlir.DivUIOp(unwrap(k_rem), unwrap(c2)).result
                        
                        coord_b_base0 = rocir.make_coord(n_intra0, n_blk0, k_intra, k_pack, k_blk_local)
                        idx_b_base0 = rocir.crd2idx(coord_b_base0, layout_b)
                        idx_b0 = _arith_mlir.DivUIOp(unwrap(idx_b_base0), unwrap(c4)).result
                        loaded_b0 = buffer_ops.buffer_load(b_rsrc, idx_b0, vec_width=2, dtype=i32_type)
                        b_pack0 = vector.ExtractOp(vector.BitCastOp(vec1_i64, loaded_b0).result, static_position=[0], dynamic_position=[]).result
                        
                        new_acc0 = rocdl.mfma_f32_16x16x32_fp8_fp8(
                            vec4_f32, [unwrap(a_pack), unwrap(b_pack0), unwrap(curr_acc0), unwrap(c0_i32), unwrap(c0_i32), unwrap(c0_i32)]
                        ).result
                        current_accs_list[acc_idx0] = new_acc0
                        
                        coord_b_base1 = rocir.make_coord(n_intra1, n_blk1, k_intra, k_pack, k_blk_local)
                        idx_b_base1 = rocir.crd2idx(coord_b_base1, layout_b)
                        idx_b1 = _arith_mlir.DivUIOp(unwrap(idx_b_base1), unwrap(c4)).result
                        loaded_b1 = buffer_ops.buffer_load(b_rsrc, idx_b1, vec_width=2, dtype=i32_type)
                        b_pack1 = vector.ExtractOp(vector.BitCastOp(vec1_i64, loaded_b1).result, static_position=[0], dynamic_position=[]).result
                    
                        new_acc1 = rocdl.mfma_f32_16x16x32_fp8_fp8(
                            vec4_f32, [unwrap(a_pack), unwrap(b_pack1), unwrap(curr_acc1), unwrap(c0_i32), unwrap(c0_i32), unwrap(c0_i32)]
                    ).result
                        current_accs_list[acc_idx1] = new_acc1
                        
                gpu.barrier()
                scf.yield_(current_accs_list + [vec_a_next])
            final_accs = for_op.results[:-1]
            
            for mi in range(m_repeat):
                acc0 = final_accs[mi * 2]
                acc1 = final_accs[mi * 2 + 1]
                
                row_base_m = bx_m + (mi * 16)
                
                col_base0 = by_n + n_tile_base
                col_base1 = by_n + n_tile_base + 16
                
                for i in range(4):
                    row_off = (lane_div_16 * 4) + i
                    row_g = row_base_m + row_off
                    
                    # Load scale_a
                    s_a = buffer_ops.buffer_load(scale_a_rsrc, row_g, vec_width=1, dtype=f32)
                    
                    # val0 path
                    val0 = vector.ExtractOp(acc0, [], [i]).result
                    col_g = col_base0 + lane_mod_16
                    s_b0 = buffer_ops.buffer_load(scale_b_rsrc, col_g, vec_width=1, dtype=f32)
                    val0_s = _arith_mlir.MulFOp(unwrap(val0), unwrap(s_a)).result
                    val0_s = _arith_mlir.MulFOp(unwrap(val0_s), unwrap(s_b0)).result
                    val0_f16 = _arith_mlir.TruncFOp(T.f16(), unwrap(val0_s)).result
                    idx0 = rocir.crd2idx(rocir.make_coord(unwrap(row_g), unwrap(col_g)), layout_c)
                    buffer_ops.buffer_store(val0_f16, c_rsrc, idx0)
                    
                    # val1 path
                    val1 = vector.ExtractOp(acc1, [], [i]).result
                    col_g1 = col_base1 + lane_mod_16
                    s_b1 = buffer_ops.buffer_load(scale_b_rsrc, col_g1, vec_width=1, dtype=f32)
                    val1_s = _arith_mlir.MulFOp(unwrap(val1), unwrap(s_a)).result
                    val1_s = _arith_mlir.MulFOp(unwrap(val1_s), unwrap(s_b1)).result
                    val1_f16 = _arith_mlir.TruncFOp(T.f16(), unwrap(val1_s)).result
                    idx1 = rocir.crd2idx(rocir.make_coord(unwrap(row_g), unwrap(col_g1)), layout_c)
                    buffer_ops.buffer_store(val1_f16, c_rsrc, idx1)

    print("Compiling...")
    hsaco = compile_to_hsaco(ctx.module)
    print(f"✓ Compiled ({len(hsaco)} bytes)")
    print(f"DEBUG: HSACO header: {hsaco[:16]}")    
    
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
    
    b_shuffled = shuffle_weight(b_q_fp8)  # (N, K) -> shuffled
    
    # 4. Compute Baseline using AIter style (dequant + matmul)
    c_ref = run_torch(a_q_fp8, b_q_fp8, scale_a, scale_b, bias=None, dtype=torch.float32)
    # 5. Run Kernel (f16 output, in-kernel scaling)
    c_out_raw = torch.zeros((M, N), dtype=torch.float16, device=device)
    
    arg_ptrs = [
        ctypes.c_void_p(c_out_raw.data_ptr()),
        ctypes.c_void_p(a_q_fp8.data_ptr()),
        ctypes.c_void_p(b_shuffled.data_ptr()),
        ctypes.c_void_p(scale_a.data_ptr()),
        ctypes.c_void_p(scale_b.data_ptr()),
        ctypes.c_long(M), ctypes.c_long(N), ctypes.c_long(K)
    ]
    
    args_array = (ctypes.c_void_p * 8)(*[ctypes.addressof(p) for p in arg_ptrs])
    
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"kernel_fixed"))
    
    def launch_kernel():
        hip_check(hip.hipModuleLaunchKernel(kernel_func, grid_x, grid_y, 1, 256, 1, 1, 0, 0, args_array, None))
    
    launch_kernel()
    hip_check(hip.hipDeviceSynchronize())
    c_out_scaled = c_out_raw.to(torch.float32)
    
    # 7. Verify
    verify_output(c_out_scaled, c_ref, rtol=0.1, atol=0.1) 
    # Benchmark
    warmup = 5
    runs = 20
    # A(1)+B(1)+C(2) + scales
    bytes_moved = size_a + size_b + size_c * 2 + (M + N) * 4
    flops = 2 * M * N * K

    @perftest
    def bench():
        return {
            "launch": launch_kernel,
            "size": size_c,  
            "warmup_iters": warmup,
            "bench_iters": runs,
            "total_bytes": bytes_moved,
        }

    results = bench()
    gflops = flops / (results.avg_ms / 1e3) / 1e9
    print(f"Throughput: {gflops:.2f} GFLOPS, BW: {results.bandwidth_gbs:.2f} GB/s")

if __name__ == "__main__":
    torch.set_default_device('cuda')
    # Test cases
    print("Running Tiling Tests...")
    test_mfma_fp8_rocir_preshuffle(16, 4096, 2048, tile_m=16, tile_k=128) # Baseline
    test_mfma_fp8_rocir_preshuffle(32, 4096, 2048, tile_m=32, tile_k=128) # Double M
    test_mfma_fp8_rocir_preshuffle(32, 4096, 2048, tile_m=16, tile_k=256) # Double K
