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
from mlir.dialects import memref, vector, arith as std_arith
import mlir.extras.types as T
from hip import hip
import numpy as np
import ctypes

# Import benchmark utilities from shared tests/utils.py
from utils import BenchmarkResults, perftest, compile_to_hsaco
    
def benchmark_vector_add():
    """Benchmark vector addition kernel performance"""
    
    # Configuration parameters - change these to experiment
    SIZE = 20480000
    TILE_SIZE = 8  # Each thread processes TILE_SIZE elements
    VEC_WIDTH = 4   # Vector width for vectorized loads/stores (must divide TILE_SIZE evenly)
    ITERS_PER_THREAD = TILE_SIZE // VEC_WIDTH  # Number of vectorized iterations per thread
    
    print("\n" + "="*80)
    print("Benchmark: Vector Addition Performance (C = A + B) - Optimized")
    print("Optimization: Continuous Thread Indexing + Tiled SIMD Vectorization")
    print(f"  - Threads work continuously with VEC_WIDTH ({VEC_WIDTH} floats)")
    print(f"  - Outer loop handles TILE_SIZE ({TILE_SIZE} elements = {ITERS_PER_THREAD} iterations per thread)")
    print("  - Each iteration: SIMD vector.load/store operations")
    print(f"Size: {SIZE} elements ({SIZE/1e6:.1f}M floats, ~{SIZE*4/1e9:.2f} GB)")
    print(f"Memory Traffic: 3 × {SIZE} × 4 bytes = {3*SIZE*4/1e9:.2f} GB per kernel")
    print("="*80)
    
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
        # Thread index - each thread handles VEC_WIDTH contiguous elements
        tid = (gpu.block_id("x") * gpu.block_dim("x") + gpu.thread_id("x"))._value
        size_c = arith.index(SIZE)._value
        vec_width = arith.index(VEC_WIDTH)._value
        
        # Create 1D layout for vector (contiguous stride)
        one = arith.index(1)._value
        vec_shape = rocir.make_shape(size_c)
        vec_stride = rocir.make_stride(one)
        vec_layout = rocir.make_layout(vec_shape, vec_stride)
        
        # Create vector type for VEC_WIDTH floats
        vec_type = T.vector(VEC_WIDTH, T.f32())
        
        # Threads work continuously with vec_width
        # Each thread's base address is aligned to VEC_WIDTH elements
        base_vec_idx = tid  # Thread index in terms of vector chunks
        
        # Outer loop handles TILE_SIZE: each thread processes multiple vector chunks
        for iter_idx in range(ITERS_PER_THREAD):
            # Calculate which vector chunk this thread should process in this iteration
            # Global vector chunk index = base_vec_idx * ITERS_PER_THREAD + iter_idx
            iter_offset = arith.index(iter_idx)._value
            iters = arith.index(ITERS_PER_THREAD)._value
            vec_chunk_idx = (base_vec_idx * iters + iter_offset)._value
            
            # Convert vector chunk index to element index
            elem_idx = (vec_chunk_idx * vec_width)._value
            
            # Create coordinate and convert to linear index using rocir
            thread_coord = rocir.make_coord(elem_idx)
            linear_idx = rocir.crd2idx(thread_coord, vec_layout)
            idx_val = linear_idx.value if hasattr(linear_idx, "value") else linear_idx
            
            # Check bounds for vectorized access
            last_elem = (elem_idx + vec_width)._value
            valid = (last_elem <= size_c)._value
            
            if valid:
                # Vectorized load operations - load VEC_WIDTH elements in a single operation
                vec_a = vector.load(vec_type, A, [idx_val])
                vec_b = vector.load(vec_type, B, [idx_val])
                
                # Vectorized addition using standard MLIR arith dialect
                vec_c = std_arith.addf(vec_a, vec_b)
                
                # Vectorized store
                vector.store(vec_c, C, [idx_val])
            else:
                # Handle boundary elements one by one (scalar fallback)
                for j in range(VEC_WIDTH):
                    j_offset = arith.index(j)._value
                    idx = (elem_idx + j_offset)._value
                    valid_elem = (idx < size_c)._value
                    if valid_elem:
                        coord = rocir.make_coord(idx)
                        idx_linear = rocir.crd2idx(coord, vec_layout)
                        idx_val_scalar = idx_linear.value if hasattr(idx_linear, "value") else idx_linear
                        # Scalar loads for boundary elements
                        a_elem = memref.load(A, [idx_val_scalar])
                        b_elem = memref.load(B, [idx_val_scalar])
                        c_elem = (a_elem + b_elem)._value
                        memref.store(c_elem.value if hasattr(c_elem, "value") else c_elem, C, [idx_val_scalar])
    
    ip.__exit__(None, None, None)
    
    hsaco = compile_to_hsaco(ctx.module, kernel_name="vecAdd")
    print(f"  Compiled to HSACO: {len(hsaco)} bytes")
    
    # With TILE_SIZE elements per thread, we need fewer threads
    threads_per_block = 256
    total_threads_needed = (SIZE + TILE_SIZE - 1) // TILE_SIZE
    num_blocks = (total_threads_needed + threads_per_block - 1) // threads_per_block
    
    print(f"  Kernel Configuration:")
    print(f"    - Tile Size: {TILE_SIZE} elements per thread")
    print(f"    - SIMD Vector Width: {VEC_WIDTH} floats (using vector.load/store)")
    print(f"    - Iterations per thread: {ITERS_PER_THREAD} (TILE_SIZE / VEC_WIDTH)")
    print(f"    - Memory access pattern: Continuous threads with vec_width stride")
    print(f"    - Total threads needed: {total_threads_needed:,}")
    print(f"    - Blocks: {num_blocks:,} x Threads/Block: {threads_per_block}")
    
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

