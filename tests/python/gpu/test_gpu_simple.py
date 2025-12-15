"""
Simple GPU kernel tests using rocdsl Python API
Vector addition test with clean, readable syntax
"""

import sys


from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.dialects.ext import gpu, arith

import numpy as np
from mlir import ir
from mlir.dialects import memref
import mlir.extras.types as T


def test_vector_add():
    """Vector addition test: C = A + B"""
    print("\n" + "="*80)
    print("Test: Vector Addition (C = A + B)")
    print("="*80)
    
    M, N = 32, 64
    
    ctx = RAIIMLIRContextModule()
    gpu.set_container_module(ctx.module)
    
    @gpu.module("kernels", ["#rocdl.target<abi = \"500\">"])
    def gpu_module():
        pass
    
    ip = ir.InsertionPoint.at_block_begin(gpu_module.regions[0].blocks[0])
    ip.__enter__()
    
    @gpu.func(emit=True)
    def vecAdd(A: T.memref(M, N, T.f32()), B: T.memref(M, N, T.f32()), C: T.memref(M, N, T.f32())):
        # Get block/thread IDs and dimensions
        bx, by = gpu.block_id("x"), gpu.block_id("y")
        tx, ty = gpu.thread_id("x"), gpu.thread_id("y")
        bdx, bdy = gpu.block_dim("x"), gpu.block_dim("y")
        
        # Calculate global thread index
        row = (by * bdy + ty)._value
        col = (bx * bdx + tx)._value
        
        # Vector addition: C[row,col] = A[row,col] + B[row,col]
        a = memref.load(A, [row.value, col.value])
        b = memref.load(B, [row.value, col.value])
        c = (a + b)._value
        memref.store(c.value, C, [row.value, col.value])
    
    ip.__exit__(None, None, None)
    assert gpu_module.operation.verify()
    
    print(" GPU module created successfully!")
    print(ctx.module)
    
    print("\nCompiling...")
    compiled = run_pipeline(ctx.module, Pipeline().canonicalize().cse())
    print(" Compilation successful!")
    
    print("\n" + "="*80)
    print("PASSED: Vector Addition Test")
    print("="*80)


if __name__ == "__main__":
    test_vector_add()
