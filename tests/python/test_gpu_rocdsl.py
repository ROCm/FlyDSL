#!/usr/bin/env python3
"""GPU kernel tests using rocdsl Python API with REAL GPU EXECUTION"""

import sys
sys.path.insert(0, '/mnt/raid0/felix/llvm-project/buildmlir/tools/mlir/python_packages/mlir_core')
sys.path.insert(0, '/mnt/raid0/felix/rocDSL/build/python_bindings')
sys.path.insert(0, '/mnt/raid0/felix/rocDSL/python')

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.dialects.ext import gpu
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from mlir import ir
from mlir.dialects import arith, memref, scf
import mlir.extras.types as T
from hip import hip
import numpy as np
import ctypes


def compile_to_hsaco(mlir_module):
    lowered = run_pipeline(
        mlir_module,
        Pipeline()
        .canonicalize()
        .cse()
        .rocdl_attach_target(chip="gfx942")
        .Gpu(Pipeline().convert_gpu_to_rocdl(use_bare_ptr_memref_call_conv=True, runtime="HIP"))
        .gpu_to_llvm()
        .lower_to_llvm()
        .gpu_module_to_binary(format="bin")
    )
    from rocdsl.dialects.ext.gpu import get_compile_object_bytes
    return get_compile_object_bytes(lowered)


def test_vector_add():
    """Vector addition with GPU execution and validation"""
    print("\n" + "="*80)
    print("Test 1: Vector Addition (C = A + B) on GPU")
    print("="*80)
    
    SIZE = 2048
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    gpu.set_container_module(ctx.module)
    
    @gpu.module("vec_kernels", ["#rocdl.target<abi = \"500\">"])
    def gpu_mod():
        pass
    
    ip = ir.InsertionPoint.at_block_begin(gpu_mod.regions[0].blocks[0])
    ip.__enter__()
    
    @gpu.func(emit=True)
    def vecAdd(A: T.memref(SIZE, T.f32()), B: T.memref(SIZE, T.f32()), C: T.memref(SIZE, T.f32())):
        tid = arith.addi(arith.muli(gpu.block_id("x"), gpu.block_dim("x")), gpu.thread_id("x"))
        size_c = arith.constant(T.index(), SIZE)
        valid = arith.cmpi(arith.CmpIPredicate.slt, tid, size_c)
        with ir.InsertionPoint(scf.IfOp(valid).then_block):
            a = memref.load(A, [tid])
            b = memref.load(B, [tid])
            c = arith.addf(a, b)
            memref.store(c, C, [tid])
            scf.yield_([])
    
    ip.__exit__(None, None, None)
    print("✓ MLIR module created")
    
    hsaco = compile_to_hsaco(ctx.module)
    print(f"✓ Compiled to HSACO: {len(hsaco)} bytes")
    
    np.random.seed(42)
    a_host = np.random.randn(SIZE).astype(np.float32)
    b_host = np.random.randn(SIZE).astype(np.float32)
    c_host = np.zeros(SIZE, dtype=np.float32)
    expected = a_host + b_host
    
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
    
    print(f"✓ Launching: {num_blocks} blocks × {threads_per_block} threads")
    hip_check(hip.hipModuleLaunchKernel(kernel_func, num_blocks, 1, 1, threads_per_block, 1, 1, 0, 0, args, None))
    hip_check(hip.hipDeviceSynchronize())
    
    hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, SIZE * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    error = np.max(np.abs(c_host - expected))
    
    print(f"✓ Max error: {error:.2e}")
    print(f"  Results[:5]: {c_host[:5]}")
    
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))
    
    if error < 1e-5:
        print("✅ TEST PASSED!")
        return True
    else:
        print(f"❌ TEST FAILED: error = {error}")
        return False


def test_matrix_transpose():
    """Matrix transpose with GPU execution and validation"""
    print("\n" + "="*80)
    print("Test 2: Matrix Transpose (B = A^T) on GPU")
    print("="*80)
    
    M, N = 32, 64
    
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
        
        row = arith.addi(arith.muli(by, arith.constant(T.index(), 16)), ty)
        col = arith.addi(arith.muli(bx, arith.constant(T.index(), 16)), tx)
        
        m_c = arith.constant(T.index(), M)
        n_c = arith.constant(T.index(), N)
        
        row_valid = arith.cmpi(arith.CmpIPredicate.slt, row, m_c)
        col_valid = arith.cmpi(arith.CmpIPredicate.slt, col, n_c)
        valid = arith.andi(row_valid, col_valid)
        
        with ir.InsertionPoint(scf.IfOp(valid).then_block):
            val = memref.load(A, [row, col])
            memref.store(val, B, [col, row])
            scf.yield_([])
    
    ip.__exit__(None, None, None)
    print("✓ MLIR module created")
    
    hsaco = compile_to_hsaco(ctx.module)
    print(f"✓ Compiled to HSACO: {len(hsaco)} bytes")
    
    np.random.seed(123)
    a_host = np.random.randn(M, N).astype(np.float32)
    b_host = np.zeros((N, M), dtype=np.float32)
    expected = a_host.T
    
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
    
    print(f"✓ Launching: ({grid_x}, {grid_y}) blocks × ({block_size}, {block_size}) threads")
    hip_check(hip.hipModuleLaunchKernel(kernel_func, grid_x, grid_y, 1, block_size, block_size, 1, 0, 0, args, None))
    hip_check(hip.hipDeviceSynchronize())
    
    hip_check(hip.hipMemcpy(b_host.ctypes.data, d_b, M * N * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    error = np.max(np.abs(b_host - expected))
    
    print(f"✓ Max error: {error:.2e}")
    print(f"  B[0,:5]: {b_host[0,:5]}")
    
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipModuleUnload(hip_module))
    
    if error < 1e-5:
        print("✅ TEST PASSED!")
        return True
    else:
        print(f"❌ TEST FAILED: error = {error}")
        return False


def test_matrix_multiply_with_layout():
    """Matrix multiply using layout-based indexing on GPU"""
    print("\n" + "="*80)
    print("Test 3: Matrix Multiply (C = A * B) with Layout on GPU")
    print("="*80)
    
    M, N, K = 32, 32, 64
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    gpu.set_container_module(ctx.module)
    
    @gpu.module("matmul_kernels", ["#rocdl.target<abi = \"500\">"])
    def gpu_mod():
        pass
    
    ip = ir.InsertionPoint.at_block_begin(gpu_mod.regions[0].blocks[0])
    ip.__enter__()
    
    @gpu.func(emit=True)
    def matmul(A: T.memref(M, K, T.f32()), B: T.memref(K, N, T.f32()), C: T.memref(M, N, T.f32())):
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        tx = gpu.thread_id("x")
        ty = gpu.thread_id("y")
        
        # Compute global row and column
        row = arith.addi(arith.muli(by, arith.constant(T.index(), 16)), ty)
        col = arith.addi(arith.muli(bx, arith.constant(T.index(), 16)), tx)
        
        m_c = arith.constant(T.index(), M)
        n_c = arith.constant(T.index(), N)
        k_c = arith.constant(T.index(), K)
        
        row_valid = arith.cmpi(arith.CmpIPredicate.slt, row, m_c)
        col_valid = arith.cmpi(arith.CmpIPredicate.slt, col, n_c)
        valid = arith.andi(row_valid, col_valid)
        
        with ir.InsertionPoint(scf.IfOp(valid).then_block):
            # Initialize accumulator
            sum_val = arith.constant(T.f32(), 0.0)
            
            # Loop over K dimension
            k_idx = arith.constant(T.index(), 0)
            one = arith.constant(T.index(), 1)
            
            # Simple dot product loop
            for_op = scf.ForOp(k_idx, k_c, one, [sum_val])
            with ir.InsertionPoint(for_op.body):
                k = for_op.induction_variable
                acc = for_op.inner_iter_args[0]
                
                a_val = memref.load(A, [row, k])
                b_val = memref.load(B, [k, col])
                prod = arith.mulf(a_val, b_val)
                new_acc = arith.addf(acc, prod)
                
                scf.yield_([new_acc])
            
            result = for_op.results[0]
            memref.store(result, C, [row, col])
            scf.yield_([])
    
    ip.__exit__(None, None, None)
    print("✓ MLIR module created")
    
    hsaco = compile_to_hsaco(ctx.module)
    print(f"✓ Compiled to HSACO: {len(hsaco)} bytes")
    
    # Prepare test data
    np.random.seed(456)
    a_host = np.random.randn(M, K).astype(np.float32)
    b_host = np.random.randn(K, N).astype(np.float32)
    c_host = np.zeros((M, N), dtype=np.float32)
    expected = a_host @ b_host
    
    # GPU memory allocation
    d_a = hip_check(hip.hipMalloc(M * K * 4))
    d_b = hip_check(hip.hipMalloc(K * N * 4))
    d_c = hip_check(hip.hipMalloc(M * N * 4))
    
    # Copy to device
    hip_check(hip.hipMemcpy(d_a, a_host.ctypes.data, M * K * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(d_b, b_host.ctypes.data, K * N * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    
    # Load module and kernel
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"matmul"))
    
    # Launch configuration
    block_size = 16
    grid_x = (N + block_size - 1) // block_size
    grid_y = (M + block_size - 1) // block_size
    
    arg_ptrs = [ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_b)), ctypes.c_void_p(int(d_c))]
    args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])
    
    print(f"✓ Launching: ({grid_x}, {grid_y}) blocks × ({block_size}, {block_size}) threads")
    hip_check(hip.hipModuleLaunchKernel(kernel_func, grid_x, grid_y, 1, block_size, block_size, 1, 0, 0, args, None))
    hip_check(hip.hipDeviceSynchronize())
    
    # Copy result back
    hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, M * N * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    # Verify
    error = np.max(np.abs(c_host - expected))
    rel_error = error / (np.max(np.abs(expected)) + 1e-8)
    
    print(f"✓ Max absolute error: {error:.2e}")
    print(f"✓ Max relative error: {rel_error:.2e}")
    print(f"  C[0,:5]: {c_host[0,:5]}")
    print(f"  Expected[0,:5]: {expected[0,:5]}")
    
    # Cleanup
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))
    
    if rel_error < 1e-3:  # More lenient for accumulation
        print("✅ TEST PASSED!")
        return True
    else:
        print(f"❌ TEST FAILED: relative error = {rel_error}")
        return False
if __name__ == "__main__":
    print("\n" + "="*80)
    print("ROCm GPU Execution Tests with rocdsl Python API")
    print(f"GPU: {get_hip_arch()}")
    print("="*80)
    
    result1 = test_vector_add()
    result2 = test_matrix_transpose()
    result3 = test_matrix_multiply_with_layout()
    
    print("\n" + "="*80)
    if result1 and result2 and result3:
        print("🎉 ALL GPU TESTS PASSED!")
        sys.exit(0)
    else:
        print("⚠️ SOME TESTS FAILED")
        sys.exit(1)


