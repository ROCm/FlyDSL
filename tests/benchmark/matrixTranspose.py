#!/usr/bin/env python3
"""Matrix Transpose Benchmark - GPU kernel with Rocir Layout integration"""

import sys
import os
sys.path.insert(0, os.path.join(os.environ.get('MLIR_PATH'), 'tools/mlir/python_packages/mlir_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../build/python_bindings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.dialects.ext import gpu, rocir, arith, scf
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from mlir import ir
from mlir.dialects import memref
import mlir.extras.types as T
from hip import hip
import numpy as np
import ctypes

# Import benchmark utilities from shared tests/utils.py
from utils import BenchmarkResults, perftest, compile_to_hsaco
    
def benchmark_matrix_transpose():
    """Benchmark matrix transpose kernel performance"""
    print("\n" + "="*80)
    print("Benchmark: Matrix Transpose Performance (B = A^T)")
    print("Size: 4096×4096 matrix (16M floats, ~256 MB)")
    print("Memory Traffic: 2 × 4096 × 4096 × 4 bytes = ~134 MB per kernel")
    print("="*80)
    
    M, N = 4096, 4096
    
    # Compile kernel (same as test_matrix_transpose)
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    gpu.set_container_module(ctx.module)
    
    @gpu.module("transpose_kernels", ["#rocdl.target<abi = \"500\">"])
    def gpu_mod():
        pass
    
    ip = ir.InsertionPoint.at_block_begin(gpu_mod.regions[0].blocks[0])
    ip.__enter__()
    
    @gpu.func(emit=True)
    def matrixTranspose(A: T.memref(M, N, T.f32()), B: T.memref(N, M, T.f32())):
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        tx = gpu.thread_id("x")
        ty = gpu.thread_id("y")
        
        row = (by * arith.index(16) + ty)._value
        col = (bx * arith.index(16) + tx)._value
        
        m_c = arith.index(M)._value
        n_c = arith.index(N)._value
        one = arith.index(1)._value
        
        # Create row-major layout for matrix A (M x N)
        a_shape = rocir.make_shape(m_c, n_c)
        a_stride = rocir.make_stride(n_c, one)
        a_layout = rocir.make_layout(a_shape, a_stride)
        
        # Create row-major layout for transposed matrix B (N x M)
        b_shape = rocir.make_shape(n_c, m_c)
        b_stride = rocir.make_stride(m_c, one)
        b_layout = rocir.make_layout(b_shape, b_stride)
        
        # Create thread coordinate and convert to indices
        thread_coord = rocir.make_coord(row, col)
        a_idx = rocir.crd2idx(thread_coord, a_layout)
        
        # Transposed coordinate for B
        transposed_coord = rocir.make_coord(col, row)
        b_idx = rocir.crd2idx(transposed_coord, b_layout)
        
        row_valid = (row < m_c)._value
        col_valid = (col < n_c)._value
        valid = (row_valid & col_valid)._value
        
        with ir.InsertionPoint(scf.IfOp(valid.value).then_block):
            val = memref.load(A, [row.value if hasattr(row, "value") else row, col.value if hasattr(col, "value") else col])
            memref.store(val.value if hasattr(val, "value") else val, B, [col.value if hasattr(col, "value") else col, row.value if hasattr(row, "value") else row])
            scf.yield_([])
    
    ip.__exit__(None, None, None)
    
    hsaco = compile_to_hsaco(ctx.module, kernel_name="matrixTranspose")
    print(f"  Compiled to HSACO: {len(hsaco)} bytes")
    
    # Allocate device memory
    np.random.seed(123)
    a_host = np.random.randn(M, N).astype(np.float32)
    b_host = np.zeros((N, M), dtype=np.float32)
    
    d_a = hip_check(hip.hipMalloc(M * N * 4))
    d_b = hip_check(hip.hipMalloc(M * N * 4))
    
    hip_check(hip.hipMemcpy(d_a, a_host.ctypes.data, M * N * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"matrixTranspose"))
    
    block_size = 16
    grid_x = (N + block_size - 1) // block_size
    grid_y = (M + block_size - 1) // block_size
    
    arg_ptrs = [ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_b))]
    args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])
    
    grid_dims = (grid_x, grid_y, 1)
    block_dims = (block_size, block_size, 1)

    def hip_kernel_launch():
        hip.hipModuleLaunchKernel(
            kernel_func,
            *grid_dims,
            *block_dims,
            sharedMemBytes=0,
            stream=None,
            kernelParams=args,
            extra=None,
        )
        hip.hipDeviceSynchronize()

    @perftest
    def run_benchmark():
        return {
            "launch": hip_kernel_launch,
            "size": M * N,
            "total_bytes": 2 * M * N * 4,
        }
    
    # Run benchmark
    results = run_benchmark()
    
    # Verify correctness
    hip_check(hip.hipMemcpy(b_host.ctypes.data, d_b, M * N * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    expected = a_host.T
    error = np.max(np.abs(b_host - expected))
    
    print(f"\n  Correctness Check:")
    print(f"  Max error: {error:.2e}")
    
    # Print benchmark results
    print(f"\n{results}")
    
    # Cleanup
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipModuleUnload(hip_module))
    
    return error < 1e-5

# Pytest test function
def test_benchmark_matrix_transpose():
    """Pytest wrapper for matrix transpose benchmark."""
    print("\n" + "="*80)
    print("ROCm GPU Benchmark - Matrix Transpose with Rocir Layout")
    print(f"GPU: {get_hip_arch()}")
    print("="*80)
    assert benchmark_matrix_transpose(), "Matrix transpose benchmark failed correctness check"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Matrix Transpose Benchmark')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run performance benchmark')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ROCm GPU Benchmark - Matrix Transpose with Rocir Layout")
    print(f"GPU: {get_hip_arch()}")
    print("="*80)
    
    result = benchmark_matrix_transpose()
    
    print("\n" + "="*80)
    if result:
        print("✓ BENCHMARK COMPLETED SUCCESSFULLY")
        sys.exit(0)
    else:
        print("⚠️ BENCHMARK FAILED CORRECTNESS CHECK")
        sys.exit(1)

