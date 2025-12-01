#!/usr/bin/env python3
"""Vector Addition Benchmark - GPU kernel with Rocir Layout integration"""

import sys
import os
sys.path.insert(0, os.path.join(os.environ.get('MLIR_PATH'), 'tools/mlir/python_packages/mlir_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../build/python_bindings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.dialects.ext import gpu, rocir, arith
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from mlir import ir
from mlir.dialects import memref
import mlir.extras.types as T
from hip import hip
import numpy as np
import ctypes

# Import benchmark utilities from shared tests/utils.py
from utils import BenchmarkResults, perftest, compile_to_hsaco
    
def benchmark_vector_add():
    """Benchmark vector addition kernel performance"""
    print("\n" + "="*80)
    print("Benchmark: Vector Addition Performance (C = A + B)")
    print("Size: 2048000 elements (2M floats, ~24.6 MB)")
    print("Memory Traffic: 3 × 2048000 × 4 bytes = 24.6 MB per kernel")
    print("="*80)
    
    SIZE = 204800000
    
    # Compile kernel (same as test_vector_add)
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    gpu.set_container_module(ctx.module)
    
    @gpu.module("vec_kernels", ["#rocdl.target<abi = \"500\">"])
    def gpu_mod():
        pass
    
    ip = ir.InsertionPoint.at_block_begin(gpu_mod.regions[0].blocks[0])
    ip.__enter__()
    
    @gpu.func(emit=True)
    def vecAdd(A: T.memref(SIZE, T.f32()), B: T.memref(SIZE, T.f32()), C: T.memref(SIZE, T.f32())):
        tid = (gpu.block_id("x") * gpu.block_dim("x") + gpu.thread_id("x"))._value
        size_c = arith.index(SIZE)._value
        
        # Create 1D layout for vector (contiguous stride)
        one = arith.index(1)._value
        vec_shape = rocir.make_shape(size_c)
        vec_stride = rocir.make_stride(one)
        vec_layout = rocir.make_layout(vec_shape, vec_stride)
        
        # Create coordinate and convert to linear index using rocir
        thread_coord = rocir.make_coord(tid)
        linear_idx = rocir.crd2idx(thread_coord, vec_layout)
        
        valid = (tid < size_c)._value
        if valid:
            # Use layout-computed linear index for memory access
            a = memref.load(A, [linear_idx.value if hasattr(linear_idx, "value") else linear_idx])
            b = memref.load(B, [linear_idx.value if hasattr(linear_idx, "value") else linear_idx])
            c = (a + b)._value
            memref.store(c.value if hasattr(c, "value") else c, C, [linear_idx.value if hasattr(linear_idx, "value") else linear_idx])
    
    ip.__exit__(None, None, None)
    
    hsaco = compile_to_hsaco(ctx.module)
    print(f"  Compiled to HSACO: {len(hsaco)} bytes")
    
    # Allocate device memory
    np.random.seed(42)
    a_host = np.random.randn(SIZE).astype(np.float32)
    b_host = np.random.randn(SIZE).astype(np.float32)
    c_host = np.zeros(SIZE, dtype=np.float32)
    
    d_a = hip_check(hip.hipMalloc(SIZE * 4))
    d_b = hip_check(hip.hipMalloc(SIZE * 4))
    d_c = hip_check(hip.hipMalloc(SIZE * 4))
    
    hip_check(hip.hipMemcpy(d_a, a_host.ctypes.data, SIZE * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(d_b, b_host.ctypes.data, SIZE * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"vecAdd"))
    
    threads_per_block = 256
    num_blocks = (SIZE + threads_per_block - 1) // threads_per_block
    
    arg_ptrs = [ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_b)), ctypes.c_void_p(int(d_c))]
    args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])
    
    # Benchmark function that returns kernel configuration
    @perftest
    def run_benchmark():
        return (
            kernel_func,
            args,
            (num_blocks, 1, 1),  # grid dimensions
            (threads_per_block, 1, 1),  # block dimensions
            SIZE  # for bandwidth calculation
        )
    
    # Run benchmark
    results = run_benchmark()
    
    # Verify correctness
    hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, SIZE * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    expected = a_host + b_host
    error = np.max(np.abs(c_host - expected))
    
    print(f"\n  Correctness Check:")
    print(f"  Max error: {error:.2e}")
    
    # Print benchmark results
    print(f"\n{results}")
    
    # Cleanup
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))
    
    return error < 1e-5

# Pytest test function
def test_benchmark_vector_add():
    """Pytest wrapper for vector addition benchmark."""
    print("\n" + "="*80)
    print("ROCm GPU Benchmark - Vector Addition with Rocir Layout")
    print(f"GPU: {get_hip_arch()}")
    print("="*80)
    assert benchmark_vector_add(), "Vector addition benchmark failed correctness check"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Vector Addition Benchmark')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run performance benchmark')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ROCm GPU Benchmark - Vector Addition with Rocir Layout")
    print(f"GPU: {get_hip_arch()}")
    print("="*80)
    
    result = benchmark_vector_add()
    
    print("\n" + "="*80)
    if result:
        print("✓ BENCHMARK COMPLETED SUCCESSFULLY")
        sys.exit(0)
    else:
        print("⚠️ BENCHMARK FAILED CORRECTNESS CHECK")
        sys.exit(1)

