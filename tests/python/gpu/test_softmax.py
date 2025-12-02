#!/usr/bin/env python3
"""
Softmax Operator Test
Implementation of a Block-wise Softmax:
- Grid: (M, 1, 1) -> One block per row
- Block: (N, 1, 1) -> Threads handle columns
- Shared Memory: Used for reduction (max and sum)

Softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
"""

import sys
import os

# Add paths to find rocdsl and mlir packages
sys.path.insert(0, os.path.join(os.environ.get('MLIR_PATH', ''), 'tools/mlir/python_packages/mlir_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../build/python_bindings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../python'))

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.dialects.ext import gpu, scf
from rocdsl.dialects.ext.gpu import lds_space
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from utils import compile_to_hsaco
from mlir import ir
from mlir.dialects import arith, memref, math
import mlir.extras.types as T
try:
    from hip import hip
except ImportError:
    print("HIP module not found. Skipping GPU tests.")
    sys.exit(0)

import numpy as np
import ctypes
import time

# Softmax dimensions
M = 64   # Number of rows (batch size)
N = 256  # Number of columns (feature size, matches block size)

ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
gpu.set_container_module(ctx.module)

@gpu.module("softmax_module", [f'#rocdl.target<chip = "{get_hip_arch()}", abi = "500">'])
def gpu_mod():
    pass

ip = ir.InsertionPoint.at_block_begin(gpu_mod.regions[0].blocks[0])
ip.__enter__()

# Shared memory for reduction (size N)
# We reuse this buffer for max reduction and sum reduction
smem_type = T.memref(N, T.f32(), memory_space=lds_space())
memref.global_(sym_name="shared_buffer", type_=smem_type, alignment=16)

@gpu.func(emit=True)
def softmax_kernel(A: T.memref(M, N, T.f32()), C: T.memref(M, N, T.f32())):
    # IDs
    row = gpu.block_id("x")
    tid = gpu.thread_id("x")
    
    # Get shared memory
    smem = memref.get_global(smem_type, "shared_buffer")
    
    # 1. Load data and write to shared memory for Max Reduction
    val = memref.load(A, [row.value, tid.value])
    memref.store(val.value, smem, [tid.value])
    
    gpu.barrier()
    
    # 2. Reduction Max (Naive implementation: Thread 0 scans all)
    # Note: In production, use tree reduction
    zero_idx = arith.constant(T.index(), 0)
    
    # Variable to hold max value, initialized to first element by Thread 0
    # We use a stack allocation (alloca) or just pass via iter_args in SCF
    # Here we use iter_args for cleanliness
    
    # We need a way to broadcast the result. We will store it back to smem[0]
    
    is_thread_0 = arith.cmpi(arith.CmpIPredicate.eq, tid.value, zero_idx.value)
    
    if_op = scf.IfOp(is_thread_0.value)
    with ir.InsertionPoint(if_op.then_block):
        # Thread 0 finds max
        init_max = memref.load(smem, [zero_idx.value])
        
        one_idx = arith.constant(T.index(), 1)
        n_idx = arith.constant(T.index(), N)
        
        loop = scf.ForOp(one_idx.value, n_idx.value, one_idx.value, [init_max.value])
        with ir.InsertionPoint(loop.body):
            i = loop.induction_variable
            curr_max = loop.inner_iter_args[0]
            val_i = memref.load(smem, [i.value])
            new_max = arith.maximumf(curr_max.value, val_i.value)
            scf.yield_([new_max.value])
            
        max_val = loop.results[0]
        memref.store(max_val.value, smem, [zero_idx.value])
        scf.yield_([])
        
    gpu.barrier()
    
    # 3. Broadcast Max
    # All threads read the max value from smem[0]
    row_max = memref.load(smem, [zero_idx.value])
    
    # 4. Compute Exp and write to shared memory for Sum Reduction
    # exp(x - max)
    diff = arith.subf(val.value, row_max.value)
    exp_val = math.exp(diff.value)
    memref.store(exp_val.value, smem, [tid.value])
    
    gpu.barrier()
    
    # 5. Reduction Sum (Naive implementation: Thread 0 sums all)
    if_op_sum = scf.IfOp(is_thread_0.value)
    with ir.InsertionPoint(if_op_sum.then_block):
        # Thread 0 calculates sum
        init_sum = memref.load(smem, [zero_idx.value]) # Load exp(x_0)
        
        one_idx = arith.constant(T.index(), 1)
        n_idx = arith.constant(T.index(), N)
        
        loop_sum = scf.ForOp(one_idx.value, n_idx.value, one_idx.value, [init_sum.value])
        with ir.InsertionPoint(loop_sum.body):
            i = loop_sum.induction_variable
            curr_sum = loop_sum.inner_iter_args[0]
            val_i = memref.load(smem, [i.value])
            new_sum = arith.addf(curr_sum.value, val_i.value)
            scf.yield_([new_sum.value])
            
        sum_val = loop_sum.results[0]
        memref.store(sum_val.value, smem, [zero_idx.value])
        scf.yield_([])

    gpu.barrier()
    
    # 6. Broadcast Sum
    row_sum = memref.load(smem, [zero_idx.value])
    
    # 7. Normalize and Store
    # res = exp_val / row_sum
    res = arith.divf(exp_val.value, row_sum.value)
    memref.store(res.value, C, [row.value, tid.value])

ip.__exit__(None, None, None)

def test_softmax():
    print("\n" + "="*80)
    print("Testing Softmax Operator (M={}, N={})".format(M, N))
    print("="*80)

    if hip is None:
        print("HIP not available, skipping...")
        return

    # Compile
    print("Compiling kernel...")
    try:
        hsaco = compile_to_hsaco(ctx.module)
    except Exception as e:
        print(f"Compilation failed: {e}")
        # Print module for debugging
        print(ctx.module)
        raise e

    print(f" HSACO size: {len(hsaco)} bytes")

    # Prepare Data
    np.random.seed(42)
    # Random inputs [-2, 2]
    a_host = (np.random.rand(M, N).astype(np.float32) * 4.0) - 2.0
    c_host = np.zeros((M, N), dtype=np.float32)

    # Numpy Reference Softmax
    # exp(x - max) / sum(exp(x - max))
    max_vals = np.max(a_host, axis=1, keepdims=True)
    exp_vals = np.exp(a_host - max_vals)
    sum_vals = np.sum(exp_vals, axis=1, keepdims=True)
    expected = exp_vals / sum_vals

    # Allocate GPU Memory
    d_a = hip_check(hip.hipMalloc(M * N * 4))
    d_c = hip_check(hip.hipMalloc(M * N * 4))

    hip_check(hip.hipMemcpy(d_a, a_host.ctypes.data, M * N * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))

    # Load Kernel
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"softmax_kernel"))

    # Launch Kernel
    # Grid: (M, 1, 1), Block: (N, 1, 1)
    grid_x, grid_y, grid_z = M, 1, 1
    block_x, block_y, block_z = N, 1, 1
    
    # Dynamically allocated shared mem size (if used) - here we use static global shared
    smem_size = 0 

    arg_ptrs = [ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_c))]
    args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])

    print("Launching kernel...")
    start_time = time.time()
    
    hip_check(hip.hipModuleLaunchKernel(kernel_func, grid_x, grid_y, grid_z, block_x, block_y, block_z, smem_size, 0, args, None))
    hip_check(hip.hipDeviceSynchronize())
    
    end_time = time.time()
    print(f"Kernel execution time: {(end_time - start_time)*1000:.4f} ms")

    # Copy back
    hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, M * N * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

    # Verification
    # Use a slightly looser tolerance for float32 exp/div operations
    error = np.max(np.abs(c_host - expected))
    print(f"Max absolute error: {error:.2e}")
    
    if np.allclose(c_host, expected, atol=1e-5):
        print("✅ PASSED: Softmax implementation is correct.")
    else:
        print("❌ FAILED: Results do not match reference.")
        # Debug prints
        print("First row Expected:")
        print(expected[0, :5])
        print("First row Actual:")
        print(c_host[0, :5])

    # Cleanup
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))

if __name__ == "__main__":
    test_softmax()

