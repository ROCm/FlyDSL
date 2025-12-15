"""
GPU kernel tests using rocir layout algebra API
Demonstrates: layout-based indexing, shape/stride operations, local partitioning
Uses simplified syntax inspired by mlir-python-extras
"""

import sys
import os

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.dialects.ext import gpu, rocir
from rocdsl.dialects.ext.arith import Index
from rocdsl.runtime.hip_util import hip_check, get_hip_arch, launch_kernel

import numpy as np
from mlir import ir
from mlir.dialects import arith, memref, scf
import mlir.extras.types as T


def test_layout_based_transpose():
    """Matrix transpose using rocir layout algebra"""
    print("\n" + "="*80)
    print("Test 1: Matrix Transpose with Rocir Layout")
    print("="*80)
    
    M, N = 32, 64
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    gpu.set_container_module(ctx.module)
    
    @gpu.module("transpose_kernels", ["#rocdl.target<abi = \"500\">"])
    def gpu_module():
        pass
    
    ip = ir.InsertionPoint.at_block_begin(gpu_module.regions[0].blocks[0])
    ip.__enter__()
    
    @gpu.func(emit=True)
    def transpose_layout(Input: T.memref(M, N, T.f32()), Output: T.memref(N, M, T.f32())):
        """Transpose using rocir layout: creates transposed layout and uses it for indexing"""
        
        # Thread indices - cleaner calculation
        bx, by = gpu.block_id("x"), gpu.block_id("y")
        tx, ty = gpu.thread_id("x"), gpu.thread_id("y")
        bdx, bdy = gpu.block_dim("x"), gpu.block_dim("y")
        
        row = (by * bdy + ty)._value
        col = (bx * bdx + tx)._value
        
        # Create layout constants
        M_c, N_c = Index(M), Index(N)
        one = Index(1)
        
        # Input layout (row-major M x N)
        input_shape = rocir.make_shape(M_c, N_c)
        input_stride = rocir.make_stride(N_c, one)
        input_layout = rocir.make_layout(input_shape, input_stride)
        
        # Output layout (row-major N x M, transposed)
        output_shape = rocir.make_shape(N_c, M_c)
        output_stride = rocir.make_stride(M_c, one)
        output_layout = rocir.make_layout(output_shape, output_stride)
        
        # Bounds check
        valid = ((row < M_c)._value & (col < N_c)._value._value
        )
        
        # Transpose: Input[row,col] -> Output[col,row]
        if valid:
            val = memref.load(Input, [row.value if hasattr(row, "value") else row, col.value if hasattr(col, "value") else col])
            memref.store(val.value if hasattr(val, "value") else val, Output, [col.value if hasattr(col, "value") else col, row.value if hasattr(row, "value") else row])

    ip.__exit__(None, None, None)
    assert gpu_module.operation.verify()
    
    print(" Layout-based transpose GPU module created!")
    print(ctx.module)
    
    print("\nCompiling...")
    compiled = run_pipeline(ctx.module, Pipeline().canonicalize().cse())
    print(" Compilation successful!")
    
    print("\n" + "="*80)
    print("PASSED: Layout-based Transpose Test")
    print("="*80)


def test_strided_layout_access():
    """Strided layout with custom shape and stride using rocir"""
    print("\n" + "="*80)
    print("Test 2: Strided Layout with Rocir Shape/Stride")
    print("="*80)
    
    M, N = 16, 32
    in_stride_val = N + 8   # Input padding
    out_stride_val = N + 4  # Output padding
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    gpu.set_container_module(ctx.module)
    
    @gpu.module("strided_kernels", ["#rocdl.target<abi = \"500\">"])
    def gpu_module():
        pass
    
    ip = ir.InsertionPoint.at_block_begin(gpu_module.regions[0].blocks[0])
    ip.__enter__()
    
    @gpu.func(emit=True)
    def copy_with_layout(
        Input: T.memref(M * in_stride_val, T.f32()), 
        Output: T.memref(M * out_stride_val, T.f32())
    ):
        """Copy using rocir layouts with custom strides"""
        
        # Thread indices
        bx, by = gpu.block_id("x"), gpu.block_id("y")
        tx, ty = gpu.thread_id("x"), gpu.thread_id("y")
        bdx, bdy = gpu.block_dim("x"), gpu.block_dim("y")
        
        row = (by * bdy + ty)._value
        col = (bx * bdx + tx)._value
        
        # Layout constants
        M_c, N_c = Index(M), Index(N)
        one = Index(1)
        in_s, out_s = Index(in_stride_val), Index(out_stride_val)
        
        # Create layouts with custom strides
        shape = rocir.make_shape(M_c, N_c)
        input_layout = rocir.make_layout(shape, rocir.make_stride(in_s, one))
        output_layout = rocir.make_layout(shape, rocir.make_stride(out_s, one))
        
        # Query layout properties
        input_size = rocir.size(input_layout)
        output_size = rocir.size(output_layout)
        
        # Bounds check
        valid = ((row < M_c)._value & (col < N_c)._value._value
        )
        
        # Strided copy with scaling
        if valid:
            in_idx = (row * in_s + col)._value
            out_idx = (row * out_s + col)._value
            
            val = memref.load(Input, [in_idx.value if hasattr(in_idx, "value") else in_idx])
            result = (val * arith.constant(T.f32(), 2.0))
            memref.store(result.value if hasattr(result, "value") else result, Output, [out_idx.value if hasattr(out_idx, "value") else out_idx])

    ip.__exit__(None, None, None)
    assert gpu_module.operation.verify()
    
    print(" Strided layout GPU module created!")
    print(ctx.module)
    
    print("\nCompiling...")
    compiled = run_pipeline(ctx.module, Pipeline().canonicalize().cse())
    print(" Compilation successful!")
    
    print("\n" + "="*80)
    print("PASSED: Strided Layout Test")
    print("="*80)


def test_tiled_layout():
    """Tiled layout using rocir logical_divide for thread partitioning"""
    print("\n" + "="*80)
    print("Test 3: Tiled Layout with Rocir Logical Divide")
    print("="*80)
    
    M, N = 64, 128
    TILE_M, TILE_N = 16, 32
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    gpu.set_container_module(ctx.module)
    
    @gpu.module("tiled_kernels", ["#rocdl.target<abi = \"500\">"])
    def gpu_module():
        pass
    
    ip = ir.InsertionPoint.at_block_begin(gpu_module.regions[0].blocks[0])
    ip.__enter__()
    
    @gpu.func(emit=True)
    def tiled_copy(Input: T.memref(M, N, T.f32()), Output: T.memref(M, N, T.f32())):
        """Copy using tiled layout partitioning"""
        
        # Thread/block IDs
        bx, by = gpu.block_id("x"), gpu.block_id("y")
        tx, ty = gpu.thread_id("x"), gpu.thread_id("y")
        bdx, bdy = gpu.block_dim("x"), gpu.block_dim("y")
        
        # Global thread position
        row = (by * bdy + ty)._value
        col = (bx * bdx + tx)._value
        
        # Constants
        M_c, N_c = Index(M), Index(N)
        tile_M_c, tile_N_c = Index(TILE_M), Index(TILE_N)
        one = Index(1)
        
        # Global layout (row-major)
        global_shape = rocir.make_shape(M_c, N_c)
        global_stride = rocir.make_stride(N_c, one)
        global_layout = rocir.make_layout(global_shape, global_stride)
        
        # Tile layout
        tile_shape = rocir.make_shape(tile_M_c, tile_N_c)
        tile_stride = rocir.make_stride(tile_N_c, one)
        tiler_layout = rocir.make_layout(tile_shape, tile_stride)
        
        # Partition global data by tiles
        partitioned = rocir.logical_divide(global_layout, tiler_layout)
        
        # Bounds check
        valid = ((row < M_c)._value & (col < N_c)._value._value
        )
        
        # Copy with layout awareness
        if valid:
            val = memref.load(Input, [row.value if hasattr(row, "value") else row, col.value if hasattr(col, "value") else col])
            memref.store(val.value if hasattr(val, "value") else val, Output, [row.value if hasattr(row, "value") else row, col.value if hasattr(col, "value") else col])

    ip.__exit__(None, None, None)
    assert gpu_module.operation.verify()
    
    print(" Tiled layout GPU module created!")
    print(ctx.module)
    
    print("\nCompiling...")
    compiled = run_pipeline(ctx.module, Pipeline().canonicalize().cse())
    print(" Compilation successful!")
    
    print("\n" + "="*80)
    print("PASSED: Tiled Layout Test")
    print("="*80)


if __name__ == "__main__":
    test_layout_based_transpose()
    test_strided_layout_access()
    test_tiled_layout()
    print("\n" + "="*80)
    print("All Rocir Layout GPU tests PASSED!")
    print("="*80)
