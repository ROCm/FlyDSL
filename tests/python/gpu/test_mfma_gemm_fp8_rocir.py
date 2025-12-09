#!/usr/bin/env python3
"""MFMA FP8 GEMM Test using Rocir with @gpu.func decorator pattern."""

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
from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
import rocdsl.dialects.ext.rocir as rocir
from rocdsl.utils import SmemAllocator
from tests.utils import compile_to_hsaco, perftest, pertoken_quant, verify_output
import torch
import torch.nn.functional as F
import pytest
from mlir import ir
from mlir.dialects import vector, memref, builtin
from rocdsl.dialects.ext import arith, scf, gpu
from mlir.dialects import arith as _arith_mlir
import mlir.dialects.rocdl as rocdl
import mlir.extras.types as T
from hip import hip
import ctypes

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

def unwrap(v):
    if isinstance(v, int):
        return arith.constant(v, index=True).value
    if hasattr(v, "value"): return v.value
    if hasattr(v, "_value"): return v._value
    return v


def test_mfma_fp8_rocir():
    print("="*80)
    print("MFMA Real FP8 GEMM Test (Rocir + Decorator) - 1024x1024x1280")
    print("="*80)
    
    gpu_arch = get_hip_arch()
    print(f"Detected HIP Arch: {gpu_arch}")

    # Constants
    M, N, K = 1024, 1024, 1280
    
    ctx = RAIIMLIRContextModule()
    
    # Register Rocir dialect
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
    size_b = N * K  # Transposed B (NxK)
    
    # 1. Initialize Allocator
    allocator = SmemAllocator(ctx, arch=gpu_arch)

    # 2. Allocate Arrays (Tile size 32x128)
    lds_a_decl = allocator.allocate_array(f8, 4096)
    lds_b_decl = allocator.allocate_array(f8, 4096)
    
    @gpu.module("mfma_mod", [f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'])
    def gpu_mod():
        # 3. Finalize: Automatically create underlying memref.GlobalOp
        allocator.finalize()
        
        @gpu.func(emit=True)
        def kernel(
            arg_c: T.memref(size_c, T.f32()),
            arg_a: T.memref(size_a, f8),
            arg_b: T.memref(size_b, f8),
            arg_scale_a: T.memref(M, T.f32()),
            arg_scale_b: T.memref(N, T.f32())
        ):
            c0, c1 = 0, 1
            c_m, c_n, c_k = 1024, 1024, 1280
            c128, c32, c16, c8, c4, c2, c64 = 128, 32, 16, 8, 4, 2, 64
            c0_i32 = arith.i32(0)
            identity_map = ir.AffineMap.get_identity(1)
            
            # Define Layouts using Rocir
            # Layout A: (M, K) with stride (K, 1)
            shape_a = rocir.make_shape(c_m, c_k)
            stride_a = rocir.make_stride(c_k, c1)
            layout_a = rocir.make_layout(shape_a, stride_a)
            
            # Layout B (Transposed): (N, K) with stride (K, 1)
            shape_b = rocir.make_shape(c_n, c_k)
            stride_b = rocir.make_stride(c_k, c1)
            layout_b = rocir.make_layout(shape_b, stride_b)
            
            # Layout C: (M, N) with stride (N, 1)
            shape_c = rocir.make_shape(c_m, c_n)
            stride_c = rocir.make_stride(c_n, c1)
            layout_c = rocir.make_layout(shape_c, stride_c)
            
            # LDS Layout: 32x128 (Row Major)
            shape_lds = rocir.make_shape(c32, c128)
            stride_lds = rocir.make_stride(c128, c1)
            layout_lds = rocir.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            by = gpu.block_id("y")
            
            # 4. Get Shared Memory Pointers
            base_ptr = allocator.get_base()
            lds_a = lds_a_decl(base_ptr).get()
            lds_b = lds_b_decl(base_ptr).get()
            
            # Accumulator Init
            vec4_f32 = ir.VectorType.get([4], f32)
            zero_attr = ir.DenseElementsAttr.get_splat(vec4_f32, ir.FloatAttr.get(f32, 0.0))
            acc_init = _arith_mlir.ConstantOp(vec4_f32, zero_attr).result
            
            # Global Load Indices
            tx_16 = tx * c16
            
            row_a_local = tx_16 // c128
            col_a_local = tx_16 % c128
            
            bx_32 = bx * c32
            row_a_global = bx_32 + row_a_local
            
            # For B (Transposed)
            row_b_local = tx_16 // c128
            col_b_local = tx_16 % c128
            
            by_32 = by * c32
            row_b_global = by_32 + row_b_local
            
            # LDS Write Index
            lds_write_idx = tx_16
            
            vec16_f8 = ir.VectorType.get([16], f8)
            pad_f8 = _arith_mlir.ConstantOp(f8, ir.FloatAttr.get(f8, 0.0)).result
            
            # Pre-calculate LDS read indices
            wave_id = tx // c64
            lane_id = tx % c64
            
            wave_row = wave_id // c2
            wave_col = wave_id % c2
            
            lane_mod_16 = lane_id % c16
            lane_div_16 = lane_id // c16
            
            row_a_lds_base = wave_row * c16
            row_a_lds = row_a_lds_base + lane_mod_16
            
            col_offset_base = lane_div_16 * c8
            
            row_b_lds_base = wave_col * c16
            row_b_lds = row_b_lds_base + lane_mod_16
            
            # Main Loop K
            current_acc = acc_init
            for k in range(0, c_k, c128):
                
                # Load A using Rocir
                col_a_global_k = k + col_a_local
                coord_a = rocir.make_coord(row_a_global, col_a_global_k)
                idx_a = rocir.crd2idx(coord_a, layout_a)
                
                vec_a = vector.TransferReadOp(vec16_f8, arg_a, [unwrap(idx_a)], identity_map, unwrap(pad_f8), [True]).result
                vector.StoreOp(vec_a, lds_a, [unwrap(lds_write_idx)])
                
                # Load B (Transposed) using Rocir
                col_b_global_k = k + col_b_local
                coord_b = rocir.make_coord(row_b_global, col_b_global_k)
                idx_b = rocir.crd2idx(coord_b, layout_b)
                
                vec_b = vector.TransferReadOp(vec16_f8, arg_b, [unwrap(idx_b)], identity_map, unwrap(pad_f8), [True]).result
                vector.StoreOp(vec_b, lds_b, [unwrap(lds_write_idx)])
                
                gpu.barrier()
                
                # Inner Loop
                acc = current_acc
                for ki in range(0, 128, 32):
                    # Calculate LDS indices using Rocir
                    col_lds = ki + col_offset_base
                    
                    # A LDS Index
                    coord_a_lds = rocir.make_coord(row_a_lds, col_lds)
                    idx_a_mfma = rocir.crd2idx(coord_a_lds, layout_lds)
                    
                    # B LDS Index
                    coord_b_lds = rocir.make_coord(row_b_lds, col_lds)
                    idx_b_mfma = rocir.crd2idx(coord_b_lds, layout_lds)
                    
                    vec8_f8 = ir.VectorType.get([8], f8)
                    vec8_i8 = ir.VectorType.get([8], ir.IntegerType.get_signless(8))
                    vec1_i64 = ir.VectorType.get([1], ir.IntegerType.get_signless(64))
                    
                    vec_a_load = vector.LoadOp(vec8_f8, lds_a, [unwrap(idx_a_mfma)]).result
                    vec_b_load = vector.LoadOp(vec8_f8, lds_b, [unwrap(idx_b_mfma)]).result
                    
                    a_bytes = _arith_mlir.BitcastOp(vec8_i8, vec_a_load).result
                    b_bytes = _arith_mlir.BitcastOp(vec8_i8, vec_b_load).result
                    
                    a_vec64 = vector.BitCastOp(vec1_i64, a_bytes).result
                    b_vec64 = vector.BitCastOp(vec1_i64, b_bytes).result
                    
                    a_pack = vector.ExtractOp(a_vec64, static_position=[0], dynamic_position=[]).result
                    b_pack = vector.ExtractOp(b_vec64, static_position=[0], dynamic_position=[]).result
                    
                    acc = rocdl.mfma_f32_16x16x32_fp8_fp8(
                        vec4_f32, [unwrap(a_pack), unwrap(b_pack), unwrap(acc), unwrap(c0_i32), unwrap(c0_i32), unwrap(c0_i32)]
                    ).result
                    
                gpu.barrier()
                current_acc = acc

            final_acc = current_acc
            
            # Store Result using Rocir
            lane_div_16 = lane_id // c16
            lane_rem_16 = lane_id % c16
            
            row_wave_base = wave_row * c16
            col_wave_base = wave_col * c16
            
            bx_32 = bx * c32
            by_32 = by * c32
            
            row_base_g = bx_32 + row_wave_base
            col_base_g = by_32 + col_wave_base
            
            for i in range(4):
                val = vector.ExtractOp(final_acc, [], [i]).result
                
                # Row offset = (lane_div_16 * 4) + i
                c_i = arith.index(i)
                row_offset_base = lane_div_16 * c4
                row_offset = row_offset_base + c_i
                
                # Col offset = lane_rem_16
                col_offset = lane_rem_16
                
                # Global indices
                row_g = row_base_g + row_offset
                col_g = col_base_g + col_offset
                
                # Use Rocir for C store index
                coord_c = rocir.make_coord(row_g, col_g)
                idx = rocir.crd2idx(coord_c, layout_c)
                
                # Load Scale A
                # Since arg_scale_a is 1D (M), index by row_g
                scale_a_val = memref.LoadOp(arg_scale_a, [unwrap(row_g)]).result
                
                # Load Scale B
                # Since arg_scale_b is 1D (N), index by col_g
                scale_b_val = memref.LoadOp(arg_scale_b, [unwrap(col_g)]).result

                # Apply scaling: val = val * scale_a * scale_b
                val_s = _arith_mlir.MulFOp(unwrap(val), unwrap(scale_a_val)).result
                val_s = _arith_mlir.MulFOp(unwrap(val_s), unwrap(scale_b_val)).result
                
                memref.StoreOp(unwrap(val_s), arg_c, [unwrap(idx)])
    
    print("✓ MLIR module constructed via @gpu.func decorator")
    
    # Set kernel attributes on the GPU function
    gpu_func_op = None
    for op in ctx.module.body.operations:
        if isinstance(op, ir.OpView) and op.OPERATION_NAME == "gpu.module":
            # op.body is a Region, need to access its first block
            body_block = op.body.blocks[0] if hasattr(op.body, 'blocks') else op.body
            for inner_op in body_block.operations:
                if hasattr(inner_op, 'OPERATION_NAME') and inner_op.OPERATION_NAME == "gpu.func":
                    gpu_func_op = inner_op
                    break
    
    if gpu_func_op:
        gpu_func_op.attributes["rocdl.flat_work_group_size"] = ir.StringAttr.get("256,256")
        gpu_func_op.attributes["rocdl.reqd_work_group_size"] = ir.DenseI32ArrayAttr.get([256, 1, 1])
        gpu_func_op.attributes["gpu.kernel"] = ir.UnitAttr.get()
    
    print("Compiling...")
    hsaco = compile_to_hsaco(ctx.module)
    print(f"✓ Compiled to HSACO: {len(hsaco)} bytes")
    
    print("Executing kernel...")
    
    # --- Torch Data Gen & Baseline (AIter Style) ---
    device = torch.device('cuda')
    torch.manual_seed(42)

    # 1. Source Data (FP32)
    a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.randn(N, K, device=device, dtype=torch.float32)

    # 2. Quantize (FP8 E4M3FNUZ)
    a_q_fp8, scale_a = pertoken_quant(a_fp32, quant_dtype=torch.float8_e4m3fnuz)  # (M, K)
    b_q_fp8, scale_b = pertoken_quant(b_fp32_t, quant_dtype=torch.float8_e4m3fnuz)  # (N, K)

    # 4. Compute Baseline using AIter style (dequant + matmul)
    c_ref = run_torch(a_q_fp8, b_q_fp8, scale_a, scale_b, bias=None, dtype=torch.float32)

    # 5. Run Kernel (Output F32, in-kernel scaling)
    c_out_raw = torch.zeros((M, N), dtype=torch.float32, device=device)
    
    arg_ptrs = [
        ctypes.c_void_p(c_out_raw.data_ptr()),
        ctypes.c_void_p(a_q_fp8.data_ptr()),
        ctypes.c_void_p(b_q_fp8.data_ptr()),
        ctypes.c_void_p(scale_a.data_ptr()),
        ctypes.c_void_p(scale_b.data_ptr()),
    ]
    args_array = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])
    
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"kernel"))
    
    def launch_kernel():
        # Grid: 32x32 blocks. Block: 256 threads.
        hip_check(hip.hipModuleLaunchKernel(kernel_func, 32, 32, 1, 256, 1, 1, 0, 0, args_array, None))
    
    launch_kernel()
    hip_check(hip.hipDeviceSynchronize())
    
    # 7. Verify
    verify_output(c_out_raw, c_ref, rtol=0.1, atol=0.1)
    
    # Benchmark
    warmup = 5
    runs = 20
    # A(1)+B(1)+C(4) + scales
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
    test_mfma_fp8_rocir()
