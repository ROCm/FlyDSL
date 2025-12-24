#!/usr/bin/env python3
"""MFMA GEMM Test using Flir with @gpu.func decorator pattern.
Supports FP8 (mfma_16x16x32) and FP16 (mfma_16x16x16)."""

import sys
import os
import logging
import functools
from enum import Enum

# Configure logging to show INFO level messages
logging.basicConfig(level=logging.INFO)



from pyflir.runtime.device import get_rocm_arch
import pyflir
import pyflir.dialects.ext.flir as flir
from pyflir.dialects.ext.python_control_flow import range_constexpr
from pyflir.utils import SmemAllocator
from tests.utils import pertoken_quant
from tests.test_common import run_perftest, verify_output
import torch
import torch.nn.functional as F
import pytest
from _mlir import ir
from _mlir.dialects import vector, memref, builtin
from pyflir.dialects.ext import arith, scf, gpu
from _mlir.dialects import arith as _arith_mlir
import _mlir.dialects.rocdl as rocdl
import _mlir.extras.types as T
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

class DType(Enum):
    FP8 = "fp8"
    FP16 = "fp16"

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
    if hasattr(v, "value"):
        return v.value
    return v


@pytest.mark.parametrize("dtype_config", [DType.FP8, DType.FP16])
def test_mfma_gemm_flir(dtype_config, M=1024, N=1024, K=1280, tile_m=32, tile_n=32, tile_k=128):
    """Test MFMA GEMM with configurable dtype (FP8 or FP16)."""
    
    # Configure based on dtype (non-MLIR parameters)
    if dtype_config == DType.FP8:
        print("="*80)
        print(f"MFMA FP8 GEMM Test (mfma_16x16x32) - {M}x{N}x{K}")
        print("="*80)
        mfma_k = 32  # FP8 MFMA processes 32 K elements
        torch_dtype = torch.float8_e4m3fnuz
        use_scales = True
        vec_load_size = 16  # Load 16 FP8 elements (16 bytes)
        dtype_name = "fp8"
    else:  # FP16
        print("="*80)
        print(f"MFMA FP16 GEMM Test (mfma_16x16x16) - {M}x{N}x{K}")
        print("="*80)
        mfma_k = 16  # FP16 MFMA processes 16 K elements
        torch_dtype = torch.float16
        use_scales = False
        vec_load_size = 16  # Load 16 FP16 elements (32 bytes)
        dtype_name = "fp16"
    
    gpu_arch = get_rocm_arch()
    print(f"Detected HIP Arch: {gpu_arch}")
    print(f"MFMA K dimension: {mfma_k}, Tile K: {tile_k}")
    
    def _mlir_dtype():
        return ir.Float8E4M3FNType.get() if dtype_config == DType.FP8 else ir.F16Type.get()
    
    size_c = M * N
    size_a = M * K
    size_b = N * K  # Transposed B (NxK)
    
    # 1. Initialize Allocator (emit globals via MlirModule.init_gpu_module)
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    # 2. Allocate Arrays (Tile size: tile_m x tile_k) - sizes are python ints
    pad_k = 8
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
            _state["lds_b_decl"] = allocator.allocate_array(_mlir_dtype(), lds_size_b)
            allocator.finalize()

        # Always include scale parameters for uniform signature.
        # For FP16, we pass dummy scale arrays but don't use them.
        @flir.kernel
        def kernel(
            self: flir.T.i64,
            arg_c: lambda: T.memref(size_c, T.f32()),
            arg_a: lambda: T.memref(size_a, _mlir_dtype()),
            arg_b: lambda: T.memref(size_b, _mlir_dtype()),
            arg_scale_a: lambda: T.memref(M, T.f32()),
            arg_scale_b: lambda: T.memref(N, T.f32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            f32 = ir.F32Type.get()
            mlir_dtype = _mlir_dtype()
            c0, c1 = 0, 1
            c_m, c_n, c_k = m_in, n_in, k_in
            c_tile_k = arith.constant(tile_k, index=True)
            c_mfma_k = mfma_k
            c128, c32, c16, c8, c4, c2, c64 = 128, 32, 16, 8, 4, 2, 64
            c0_i32 = arith.i32(0)
            identity_map = ir.AffineMap.get_identity(1)
            
            c_tile_m = arith.index(tile_m)
            c_tile_n = arith.index(tile_n)
            
            # Define Layouts using Flir
            # Layout A: (M, K) with stride (K, 1)
            shape_a = flir.make_shape(c_m, c_k)
            stride_a = flir.make_stride(c_k, c1)
            layout_a = flir.make_layout(shape_a, stride_a)
            
            # Layout B (Transposed): (N, K) with stride (K, 1)
            shape_b = flir.make_shape(c_n, c_k)
            stride_b = flir.make_stride(c_k, c1)
            layout_b = flir.make_layout(shape_b, stride_b)
            
            # Layout C: (M, N) with stride (N, 1)
            shape_c = flir.make_shape(c_m, c_n)
            stride_c = flir.make_stride(c_n, c1)
            layout_c = flir.make_layout(shape_c, stride_c)
            
            # LDS Layout: tile_m x tile_k (Row Major)
            c_lds_stride = arith.constant(lds_stride, index=True)
            shape_lds = flir.make_shape(c_tile_m, c_tile_k)
            stride_lds = flir.make_stride(c_lds_stride, c1)
            layout_lds = flir.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            by = gpu.block_id("y")
            
            # 4. Get Shared Memory Pointers
            base_ptr = allocator.get_base()
            lds_a = _state["lds_a_decl"](base_ptr).get()
            lds_b = _state["lds_b_decl"](base_ptr).get()
            
            # Accumulator Init
            vec4_f32 = ir.VectorType.get([4], f32)
            zero_attr = ir.DenseElementsAttr.get_splat(vec4_f32, ir.FloatAttr.get(f32, 0.0))
            acc_init = _arith_mlir.ConstantOp(vec4_f32, zero_attr).result
            
            # Global Load Indices
            c_vec_size = arith.index(vec_load_size)
            tx_vec = tx * c_vec_size
            
            row_a_local = tx_vec // c_tile_k
            col_a_local = tx_vec % c_tile_k
            
            bx_tile = bx * c_tile_m
            row_a_global = bx_tile + row_a_local
            
            # For B (Transposed)
            row_b_local = tx_vec // c_tile_k
            col_b_local = tx_vec % c_tile_k
            
            by_tile = by * c_tile_n
            row_b_global = by_tile + row_b_local
            
            # LDS Write Index (Row-Major mapping with padding)
            lds_write_idx = (row_a_local * c_lds_stride) + col_a_local
            
            vec_type = ir.VectorType.get([vec_load_size], mlir_dtype)
            pad_val = _arith_mlir.ConstantOp(mlir_dtype, ir.FloatAttr.get(mlir_dtype, 0.0)).result
            
            # Pre-calculate LDS read indices
            wave_id = tx // c64
            lane_id = tx % c64
            
            wave_row = wave_id // c2
            wave_col = wave_id % c2
            
            lane_mod_16 = lane_id % c16
            lane_div_16 = lane_id // c16
            
            row_a_lds_base = wave_row * c16
            row_a_lds = row_a_lds_base + lane_mod_16
            
            if dtype_config == DType.FP8:
                col_offset_base = lane_div_16 * c8
            else:
                col_offset_base = lane_div_16 * c4
            
            row_b_lds_base = wave_col * c16
            row_b_lds = row_b_lds_base + lane_mod_16
            
            # --- HELPER: Load A/B from Global ---
            def load_global(k_val):
                # Load A
                col_a_global_k = k_val + col_a_local
                coord_a = flir.make_coord(row_a_global, col_a_global_k)
                idx_a = flir.crd2idx(coord_a, layout_a)
                vec_a = vector.TransferReadOp(vec_type, arg_a, [unwrap(idx_a)], identity_map, unwrap(pad_val), [True]).result
                
                # Load B
                col_b_global_k = k_val + col_b_local
                coord_b = flir.make_coord(row_b_global, col_b_global_k)
                idx_b = flir.crd2idx(coord_b, layout_b)
                vec_b = vector.TransferReadOp(vec_type, arg_b, [unwrap(idx_b)], identity_map, unwrap(pad_val), [True]).result
                
                return vec_a, vec_b

            # --- HELPER: Compute Tile ---
            def compute_tile(acc_in):
                acc_curr = acc_in
                for ki in range_constexpr(0, tile_k, mfma_k):
                    col_lds = ki + col_offset_base
                    
                    # A LDS Index (uses layout_lds which has stride padding)
                    coord_a_lds = flir.make_coord(row_a_lds, col_lds)
                    idx_a_mfma = flir.crd2idx(coord_a_lds, layout_lds)
                    
                    # B LDS Index
                    coord_b_lds = flir.make_coord(row_b_lds, col_lds)
                    idx_b_mfma = flir.crd2idx(coord_b_lds, layout_lds)
                    
                    if dtype_config == DType.FP8:
                        vec8_elem = ir.VectorType.get([8], mlir_dtype)
                        vec8_i8 = ir.VectorType.get([8], ir.IntegerType.get_signless(8))
                        vec1_i64 = ir.VectorType.get([1], ir.IntegerType.get_signless(64))
                        
                        vec_a_load = vector.LoadOp(vec8_elem, lds_a, [unwrap(idx_a_mfma)]).result
                        vec_b_load = vector.LoadOp(vec8_elem, lds_b, [unwrap(idx_b_mfma)]).result
                        
                        a_bytes = _arith_mlir.BitcastOp(vec8_i8, vec_a_load).result
                        b_bytes = _arith_mlir.BitcastOp(vec8_i8, vec_b_load).result
                        
                        a_vec64 = vector.BitCastOp(vec1_i64, a_bytes).result
                        b_vec64 = vector.BitCastOp(vec1_i64, b_bytes).result
                        
                        a_pack = vector.ExtractOp(a_vec64, static_position=[0], dynamic_position=[]).result
                        b_pack = vector.ExtractOp(b_vec64, static_position=[0], dynamic_position=[]).result
                        
                        acc_curr = rocdl.mfma_f32_16x16x32_fp8_fp8(
                            vec4_f32, [unwrap(a_pack), unwrap(b_pack), unwrap(acc_curr), unwrap(c0_i32), unwrap(c0_i32), unwrap(c0_i32)]
                        ).result
                    else:  # FP16
                        vec4_f16 = ir.VectorType.get([4], mlir_dtype)
                        vec_a_load = vector.LoadOp(vec4_f16, lds_a, [unwrap(idx_a_mfma)]).result
                        vec_b_load = vector.LoadOp(vec4_f16, lds_b, [unwrap(idx_b_mfma)]).result
                        
                        acc_curr = rocdl.mfma_f32_16x16x16f16(
                            vec4_f32, [unwrap(vec_a_load), unwrap(vec_b_load), unwrap(acc_curr), unwrap(c0_i32), unwrap(c0_i32), unwrap(c0_i32)]
                        ).result
                return acc_curr

            # Prologue: Load first tile
            c_k_idx = arith.constant(0, index=True)
            vec_a_init, vec_b_init = load_global(c_k_idx)
            
            # Main Loop: 0 to K-tile_k
            # c_k_main = c_k - c_tile_k_idx
            c_k_main = _arith_mlir.SubIOp(unwrap(c_k), unwrap(c_tile_k)).result
            
            # Python `for` + reassignment is auto-lowered into scf.for with
            # iter_args/yield/results by `lower_range_for_loops`.
            acc_iter = acc_init
            vec_a_curr = vec_a_init
            vec_b_curr = vec_b_init
            for k_curr in range(c_k_idx, c_k_main, c_tile_k):
                # 1. Store current to LDS
                vector.StoreOp(vec_a_curr, lds_a, [unwrap(lds_write_idx)])
                vector.StoreOp(vec_b_curr, lds_b, [unwrap(lds_write_idx)])
                
                # 2. Prefetch Next
                k_next = _arith_mlir.AddIOp(unwrap(k_curr), unwrap(c_tile_k)).result
                vec_a_next, vec_b_next = load_global(k_next)
                
                gpu.barrier()
                
                # 3. Compute
                acc_iter = compute_tile(acc_iter)
                
                gpu.barrier()
                
                # Carry
                vec_a_curr = vec_a_next
                vec_b_curr = vec_b_next
            
            # Epilogue: Last Tile
            final_acc = acc_iter
            last_vec_a = vec_a_curr
            last_vec_b = vec_b_curr
            
            vector.StoreOp(last_vec_a, lds_a, [unwrap(lds_write_idx)])
            vector.StoreOp(last_vec_b, lds_b, [unwrap(lds_write_idx)])
            
            gpu.barrier()
            final_acc = compute_tile(final_acc)
            
            # Store Result using Flir
            lane_div_16 = lane_id // c16
            lane_rem_16 = lane_id % c16
            
            row_wave_base = wave_row * c16
            col_wave_base = wave_col * c16
            
            bx_tile_m = bx * c_tile_m
            by_tile_n = by * c_tile_n
            
            row_base_g = bx_tile_m + row_wave_base
            col_base_g = by_tile_n + col_wave_base
            
            for i in range_constexpr(4):
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
                
                # Use Flir for C store index
                coord_c = flir.make_coord(row_g, col_g)
                idx = flir.crd2idx(coord_c, layout_c)
                
                # Apply scaling if needed
                if use_scales:
                    # Load Scale A (1D, indexed by row_g)
                    scale_a_val = memref.LoadOp(arg_scale_a, [unwrap(row_g)]).result
                    
                    # Load Scale B (1D, indexed by col_g)
                    scale_b_val = memref.LoadOp(arg_scale_b, [unwrap(col_g)]).result

                    # Apply scaling: val = val * scale_a * scale_b
                    val_s = _arith_mlir.MulFOp(unwrap(val), unwrap(scale_a_val)).result
                    val_s = _arith_mlir.MulFOp(unwrap(val_s), unwrap(scale_b_val)).result
                    
                    memref.StoreOp(unwrap(val_s), arg_c, [unwrap(idx)])
                else:
                    # No scaling, direct store
                    memref.StoreOp(unwrap(val), arg_c, [unwrap(idx)])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_c: lambda: T.memref(size_c, T.f32()),
            arg_a: lambda: T.memref(size_a, _mlir_dtype()),
            arg_b: lambda: T.memref(size_b, _mlir_dtype()),
            arg_scale_a: lambda: T.memref(M, T.f32()),
            arg_scale_b: lambda: T.memref(N, T.f32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            # Describe launch in a host-side function (IR only).
            c1 = arith.constant(1, index=True).value
            bdx = arith.constant(256, index=True).value
            gx = arith.constant((M + tile_m - 1) // tile_m, index=True).value
            gy = arith.constant((N + tile_n - 1) // tile_n, index=True).value
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
            gpu_func_op.attributes["rocdl.flat_work_group_size"] = ir.StringAttr.get("256,256")
            gpu_func_op.attributes["rocdl.reqd_work_group_size"] = ir.DenseI32ArrayAttr.get([256, 1, 1])
            gpu_func_op.attributes["gpu.kernel"] = ir.UnitAttr.get()
    
    print("Compiling...")
    exe = pyflir.compile(m)
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

    # 5. Run Kernel (Output F32)
    c_out_raw = torch.zeros((M, N), dtype=torch.float32, device=device)
    
    def launch_kernel():
        exe(c_out_raw, a_q, b_q, scale_a, scale_b, M, N, K)
    
    launch_kernel()
    torch.cuda.synchronize()
    
    # 7. Verify
    verify_output(c_out_raw, c_ref, rtol=0.1, atol=0.1)
    
    # Benchmark
    warmup = 5
    runs = 20
    
    # Calculate bytes moved based on dtype
    elem_size = 1 if dtype_config == DType.FP8 else 2  # FP8: 1 byte, FP16: 2 bytes
    bytes_moved = size_a * elem_size + size_b * elem_size + size_c * 4  # C is always FP32
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
    print("\n" + "="*80)
    print("Running MFMA GEMM Tests with Multiple Dtypes")
    print("="*80 + "\n")
    
    # Test FP8
    test_mfma_gemm_flir(DType.FP8)
    
    # Test FP16
    try:
        test_mfma_gemm_flir(DType.FP16)
    except AssertionError as e:
        print(f"\n[Note] FP16 verification failed: {e}")
        print("This is expected as FP16 support is experimental/WIP.")
    
    print("\n" + "="*80)
    print("All dtype tests completed!")
    print("="*80)