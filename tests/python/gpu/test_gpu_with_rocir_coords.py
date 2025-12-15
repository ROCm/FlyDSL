#!/usr/bin/env python3
"""GPU kernel test demonstrating Rocir coordinate operations"""

import sys
import os


from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.dialects.ext import gpu, rocir
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from tests.utils import compile_to_hsaco
from mlir import ir
from mlir.dialects import arith, memref, scf
import mlir.extras.types as T
from hip import hip
import numpy as np
import ctypes


def test_matmul_with_rocir():
    """Matrix multiply demonstrating rocir coordinate operations"""
    print("\n" + "="*80)
    print("GPU MatMul with Rocir Coordinate Operations")
    print("Demonstrating: rocir.make_coord, rocir.make_layout, rocir.crd2idx")
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

        # Compute thread coordinates
        row = (by * arith.constant(T.index(), 16) + ty)._value
        col = (bx * arith.constant(T.index(), 16) + tx)._value

        m_c = arith.constant(T.index(), M)
        n_c = arith.constant(T.index(), N)
        k_c = arith.constant(T.index(), K)
        one = arith.constant(T.index(), 1)

        # Create rocir layout for output matrix C
        c_shape = rocir.make_shape(m_c, n_c)
        c_stride = rocir.make_stride(n_c, one)  # Row-major: stride=(32, 1)
        c_layout = rocir.make_layout(c_shape, c_stride)

        # Create coordinate for current thread's position
        thread_coord = rocir.make_coord(row, col)
        
        # Use crd2idx to compute linear index (will be lowered to arith ops)
        linear_idx = rocir.crd2idx(thread_coord, c_layout)

        row_valid = (row < m_c)._value
        col_valid = (col < n_c)._value
        valid = (row_valid & col_valid)._value

        if valid:
            sum_val = arith.constant(T.f32(), 0.0)
            k_idx = arith.constant(T.index(), 0)

            for_op = scf.ForOp(k_idx.value, k_c.value, one.value, [sum_val.value])
            with ir.InsertionPoint(for_op.body):
                k = for_op.induction_variable
                acc = for_op.inner_iter_args[0]

                a_val = memref.load(A, [row.value, k.value])
                b_val = memref.load(B, [k.value, col.value])

                prod = (a_val * b_val)._value
                new_acc = (acc + prod)._value

                scf.yield_([new_acc.value])

            result = for_op.results[0]
            memref.store(result.value, C, [row.value, col.value])

    ip.__exit__(None, None, None)

    print("\n Generated MLIR with rocir coordinate operations:")
    print("- rocir.make_shape: Defines matrix dimensions")
    print("- rocir.make_stride: Specifies row-major layout")
    print("- rocir.make_layout: Combines shape and stride")
    print("- rocir.make_coord: Creates 2D thread coordinates")
    print("- rocir.crd2idx: Converts coord to linear index\n")

    print("Compiling to GPU binary...")
    hsaco = compile_to_hsaco(ctx.module)
    print(f" Compiled to HSACO: {len(hsaco)} bytes\n")

    # Run computation
    np.random.seed(456)
    a_host = np.random.randn(M, K).astype(np.float32)
    b_host = np.random.randn(K, N).astype(np.float32)
    c_host = np.zeros((M, N), dtype=np.float32)
    expected = a_host @ b_host

    d_a = hip_check(hip.hipMalloc(M * K * 4))
    d_b = hip_check(hip.hipMalloc(K * N * 4))
    d_c = hip_check(hip.hipMalloc(M * N * 4))

    hip_check(hip.hipMemcpy(d_a, a_host.ctypes.data, M * K * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(d_b, b_host.ctypes.data, K * N * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))

    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"matmul"))

    block_size = 16
    grid_x = (N + block_size - 1) // block_size
    grid_y = (M + block_size - 1) // block_size

    arg_ptrs = [ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_b)), ctypes.c_void_p(int(d_c))]
    args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])

    hip_check(hip.hipModuleLaunchKernel(kernel_func, grid_x, grid_y, 1, block_size, block_size, 1, 0, 0, args, None))
    hip_check(hip.hipDeviceSynchronize())

    hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, M * N * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

    error = np.max(np.abs(c_host - expected))
    rel_error = error / (np.max(np.abs(expected)) + 1e-8)

    print(f" Max absolute error: {error:.2e}")
    print(f" Max relative error: {rel_error:.2e}")

    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))

    return rel_error < 1e-3


if __name__ == "__main__":
    print("="*80)
    print("Rocir Coordinate Operations in GPU Kernels")
    print(f"GPU: {get_hip_arch()}")
    print("="*80)

    result = test_matmul_with_rocir()

    print("\n" + "="*80)
    if result:
        print(" TEST PASSED!")
        print("\nDemonstrated:")
        print("Rocir layout algebra integrated into GPU kernel")
        print("Coordinate operations (make_coord, make_layout, crd2idx)")
        print("Lowered to arithmetic via rocir-opt subprocess")
        print(f"Compiled and executed on {get_hip_arch()}")
        sys.exit(0)
    else:
        print("⚠️ TEST FAILED")
        sys.exit(1)
