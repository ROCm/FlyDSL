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

# LayerNorm dimensions
M = 64   # Batch size
N = 256  # Feature size
EPS = 1e-5

ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
gpu.set_container_module(ctx.module)

@gpu.module("layernorm_module", [f'#rocdl.target<chip = "{get_hip_arch()}", abi = "500">'])
def gpu_mod():
    pass

ip = ir.InsertionPoint.at_block_begin(gpu_mod.regions[0].blocks[0])
ip.__enter__()

# Shared memory for reduction (size N)
smem_type = T.memref(N, T.f32(), memory_space=lds_space())
memref.global_(sym_name="shared_buffer", type_=smem_type, alignment=16)

@gpu.func(emit=True)
def layernorm_kernel(
    Input: T.memref(M, N, T.f32()),
    Gamma: T.memref(N, T.f32()),
    Beta: T.memref(N, T.f32()),
    Output: T.memref(M, N, T.f32())
):
    # IDs
    row = gpu.block_id("x")
    tid = gpu.thread_id("x")
    
    # Constants
    zero_idx = arith.constant(T.index(), 0)
    one_idx = arith.constant(T.index(), 1)
    n_idx = arith.constant(T.index(), N)
    n_float = arith.constant(T.f32(), float(N))
    eps = arith.constant(T.f32(), EPS)
    
    smem = memref.get_global(smem_type, "shared_buffer")
    
    # Load Input
    val = memref.load(Input, [row.value, tid.value])
    
    # -----------------------------------------------------
    # 1. Calculate Mean
    # -----------------------------------------------------
    memref.store(val.value, smem, [tid.value])
    gpu.barrier()
    
    is_thread_0 = arith.cmpi(arith.CmpIPredicate.eq, tid.value, zero_idx.value)
    
    # Reduction Sum for Mean (Thread 0 only)
    if_op = scf.IfOp(is_thread_0.value)
    with ir.InsertionPoint(if_op.then_block):
        init_sum = memref.load(smem, [zero_idx.value])
        
        loop = scf.ForOp(one_idx.value, n_idx.value, one_idx.value, [init_sum.value])
        with ir.InsertionPoint(loop.body):
            i = loop.induction_variable
            curr_sum = loop.inner_iter_args[0]
            v = memref.load(smem, [i.value])
            new_sum = arith.addf(curr_sum.value, v.value)
            scf.yield_([new_sum.value])
        
        sum_val = loop.results[0]
        mean_val = arith.divf(sum_val.value, n_float.value)
        # Store mean back to smem[0] to broadcast
        memref.store(mean_val.value, smem, [zero_idx.value])
        scf.yield_([])
    
    gpu.barrier()
    
    # Broadcast Mean
    mean = memref.load(smem, [zero_idx.value])
    
    # -----------------------------------------------------
    # 2. Calculate Variance
    # -----------------------------------------------------
    # (val - mean)^2
    diff = arith.subf(val.value, mean.value)
    sq_diff = arith.mulf(diff.value, diff.value)
    
    # Store to shared mem for reduction
    memref.store(sq_diff.value, smem, [tid.value])
    gpu.barrier()
    
    # Reduction Sum for Variance (Thread 0 only)
    if_op_var = scf.IfOp(is_thread_0.value)
    with ir.InsertionPoint(if_op_var.then_block):
        init_var_sum = memref.load(smem, [zero_idx.value])
        
        loop_var = scf.ForOp(one_idx.value, n_idx.value, one_idx.value, [init_var_sum.value])
        with ir.InsertionPoint(loop_var.body):
            i = loop_var.induction_variable
            curr_sum = loop_var.inner_iter_args[0]
            v = memref.load(smem, [i.value])
            new_sum = arith.addf(curr_sum.value, v.value)
            scf.yield_([new_sum.value])
            
        var_sum = loop_var.results[0]
        var_val = arith.divf(var_sum.value, n_float.value)
        memref.store(var_val.value, smem, [zero_idx.value])
        scf.yield_([])
        
    gpu.barrier()
    
    # Broadcast Variance
    variance = memref.load(smem, [zero_idx.value])
    
    # -----------------------------------------------------
    # 3. Normalize and Scale
    # -----------------------------------------------------
    # rstd = 1 / sqrt(var + eps)
    var_eps = arith.addf(variance.value, eps.value)
    rstd = math.rsqrt(var_eps.value)
    
    # We still have 'diff' = (val - mean) in register
    # norm = diff * rstd
    norm = arith.mulf(diff.value, rstd.value)
    
    # Load Gamma, Beta
    g = memref.load(Gamma, [tid.value])
    b = memref.load(Beta, [tid.value])
    
    # result = norm * gamma + beta
    scaled = arith.mulf(norm.value, g.value)
    result = arith.addf(scaled.value, b.value)
    
    memref.store(result.value, Output, [row.value, tid.value])

ip.__exit__(None, None, None)

def test_layernorm():
    print("\n" + "="*80)
    print("Testing LayerNorm Operator (M={}, N={})".format(M, N))
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
        print(ctx.module)
        raise e

    print(f" HSACO size: {len(hsaco)} bytes")

    # Prepare Data
    np.random.seed(42)
    input_host = np.random.randn(M, N).astype(np.float32)
    gamma_host = np.random.rand(N).astype(np.float32)
    beta_host = np.random.rand(N).astype(np.float32)
    output_host = np.zeros((M, N), dtype=np.float32)

    # Numpy Reference
    mean = np.mean(input_host, axis=1, keepdims=True)
    var = np.var(input_host, axis=1, keepdims=True)
    expected = (input_host - mean) / np.sqrt(var + EPS) * gamma_host + beta_host

    # Allocate GPU Memory
    d_input = hip_check(hip.hipMalloc(M * N * 4))
    d_gamma = hip_check(hip.hipMalloc(N * 4))
    d_beta = hip_check(hip.hipMalloc(N * 4))
    d_output = hip_check(hip.hipMalloc(M * N * 4))

    hip_check(hip.hipMemcpy(d_input, input_host.ctypes.data, M * N * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(d_gamma, gamma_host.ctypes.data, N * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(d_beta, beta_host.ctypes.data, N * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))

    # Load Kernel
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"layernorm_kernel"))

    # Launch Config
    grid_x, grid_y, grid_z = M, 1, 1
    block_x, block_y, block_z = N, 1, 1
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
    hip_check(hip.hipMemcpy(output_host.ctypes.data, d_output, M * N * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

    # Verification
    error = np.max(np.abs(output_host - expected))
    print(f"Max absolute error: {error:.2e}")

    if np.allclose(output_host, expected, atol=1e-4):
        print("✅ PASSED: LayerNorm implementation is correct.")
    else:
        print("❌ FAILED: Results do not match reference.")
        print("First row Expected:")
        print(expected[0, :5])
        print("First row Actual:")
        print(output_host[0, :5])

    # Cleanup
    hip_check(hip.hipFree(d_input))
    hip_check(hip.hipFree(d_gamma))
    hip_check(hip.hipFree(d_beta))
    hip_check(hip.hipFree(d_output))
    hip_check(hip.hipModuleUnload(hip_module))

if __name__ == "__main__":
    test_layernorm()

