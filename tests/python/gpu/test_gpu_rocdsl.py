#!/usr/bin/env python3
"""GPU kernel tests demonstrating integration with Rocir Layout concepts"""

import sys
import os
from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.dialects.ext import gpu, rocir, arith, scf
from rocdsl.dialects.ext.gpu import lds_space
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from rocdsl.utils import SmemAllocator
from tests.utils import compile_to_hsaco
from mlir import ir
from mlir.dialects import memref
import mlir.extras.types as T
from hip import hip
import numpy as np
import ctypes

def demonstrate_rocir_layouts():
    """
    Demonstrate Rocir layout algebra concepts at host level.
    Layouts describe how multi-dimensional tensors map to linear memory.
    """
    print("\n" + "="*80)
    print("Rocir Layout Algebra Demo (Host-Side)")
    print("="*80)
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    with ctx.context:
        # Example 1: 1D contiguous vector layout
        size_1d = 1024
        stride_1d = 1
        
        shape_1d = rocir.make_shape(size_1d)
        stride_vec = rocir.make_stride(stride_1d)
        layout_1d = rocir.make_layout(shape_1d, stride_vec)
        
        print(" 1D Vector Layout:")
        print("- Shape: (1024,)")
        print("- Stride: (1,) [contiguous]")
        print("- Usage: Maps vector index i to memory offset i*1")
        
        # Example 2: 2D row-major matrix layout
        m = 32
        n = 64
        one = 1
        
        shape_2d = rocir.make_shape(m, n)
        stride_row_major = rocir.make_stride(n, one)  # (64, 1)
        layout_row_major = rocir.make_layout(shape_2d, stride_row_major)
        
        print("\n 2D Row-Major Matrix Layout (32×64):")
        print("- Shape: (32, 64)")
        print("- Stride: (64, 1)")
        print("- Usage: A[i,j] maps to offset i*64 + j*1")
        
        # Example 3: 2D column-major matrix layout
        stride_col_major = rocir.make_stride(one, m)  # (1, 32)
        layout_col_major = rocir.make_layout(shape_2d, stride_col_major)
        
        print("\n 2D Column-Major Matrix Layout (32×64):")
        print("- Shape: (32, 64)")
        print("- Stride: (1, 32)")
        print("- Usage: A[i,j] maps to offset i*1 + j*32")
        
        # Layout composition: Tiling
        tile_m = 16
        tile_n = 16
        tile_shape = rocir.make_shape(tile_m, tile_n)
        
        print("\n Layout Composition (Tiling):")
        print("- Original: (32, 64) with stride (64, 1)")
        print("- Tile: (16, 16)")
        print("- Composition creates hierarchical layout for blocked access")
        print("- Available: logical_product, tiled_product, blocked_product")
        
        print("\n Available Rocir Layout Operations:")
        print("- Construction: make_shape, make_stride, make_layout")
        print("- Query: size, cosize, rank, get_shape, get_stride")
        print("- Composition: composition, logical_product, zipped_product,")
        print("tiled_product, flat_product, raked_product, blocked_product")
        print("- Partitioning: logical_divide, zipped_divide, tiled_divide, flat_divide,")
        print("local_partition, local_tile")

def test_vector_add():
    """Vector addition using rocir 1D layout"""
    print("\n" + "="*80)
    print("Test 1: Vector Addition (C = A + B) with Rocir Layout")
    print("Layout: 1D contiguous, shape=(2048,), stride=(1,)")
    print("Using: rocir.make_coord, rocir.crd2idx")
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
        tid = (gpu.block_id("x") * gpu.block_dim("x") + gpu.thread_id("x"))._value
        size_c = SIZE
        
        # Create 1D layout for vector (contiguous stride)
        one = 1
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
    print(f" Compiled to HSACO: {len(hsaco)} bytes")
    
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
    
    hip_check(hip.hipModuleLaunchKernel(kernel_func, num_blocks, 1, 1, threads_per_block, 1, 1, 0, 0, args, None))
    hip_check(hip.hipDeviceSynchronize())
    
    hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, SIZE * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    error = np.max(np.abs(c_host - expected))
    print(f" Max error: {error:.2e}")
    
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))
    
    return error < 1e-5

def test_matrix_transpose():
    """Matrix transpose using rocir 2D layouts"""
    print("\n" + "="*80)
    print("Test 2: Matrix Transpose (B = A^T) with Rocir Layouts")
    print("Layout A: shape=(32,64), stride=(64,1) [row-major]")
    print("Layout B: shape=(64,32), stride=(32,1) [row-major, transposed]")
    print("Using: rocir.make_layout, rocir.crd2idx")
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
        
        row = (by * arith.index(16) + ty)._value
        col = (bx * arith.index(16) + tx)._value
        
        m_c = M
        n_c = N
        one = 1
        
        # Create row-major layout for matrix A (M x N)
        a_shape = rocir.make_shape(m_c, n_c)
        a_stride = rocir.make_stride(n_c, one)  # Row-major: stride=(N, 1)
        a_layout = rocir.make_layout(a_shape, a_stride)
        
        # Create row-major layout for transposed matrix B (N x M)
        b_shape = rocir.make_shape(n_c, m_c)
        b_stride = rocir.make_stride(m_c, one)  # Row-major: stride=(M, 1)
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
        
        if valid:
            val = memref.load(A, [row.value if hasattr(row, "value") else row, col.value if hasattr(col, "value") else col])
            memref.store(val.value if hasattr(val, "value") else val, B, [col.value if hasattr(col, "value") else col, row.value if hasattr(row, "value") else row])

    ip.__exit__(None, None, None)
    
    hsaco = compile_to_hsaco(ctx.module)
    print(f" Compiled to HSACO: {len(hsaco)} bytes")
    
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
    
    hip_check(hip.hipModuleLaunchKernel(kernel_func, grid_x, grid_y, 1, block_size, block_size, 1, 0, 0, args, None))
    hip_check(hip.hipDeviceSynchronize())
    
    hip_check(hip.hipMemcpy(b_host.ctypes.data, d_b, M * N * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    error = np.max(np.abs(b_host - expected))
    print(f" Max error: {error:.2e}")
    
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipModuleUnload(hip_module))
    
    return error < 1e-5

def test_matmul():
    """Matrix multiply using rocir layouts for all matrices"""
    print("\n" + "="*80)
    print("Test 3: Matrix Multiply (C = A * B) with Rocir Layouts")
    print("Layout A (32×64): shape=(32,64), stride=(64,1) [row-major]")
    print("Layout B (64×32): shape=(64,32), stride=(32,1) [row-major]")
    print("Layout C (32×32): shape=(32,32), stride=(32,1) [row-major]")
    print("Using: rocir.make_layout, rocir.crd2idx for coordinate computation")
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
    def matmul(A: T.memref(M * K, T.f32()), B: T.memref(K * N, T.f32()), C: T.memref(M * N, T.f32())):
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        tx = gpu.thread_id("x")
        ty = gpu.thread_id("y")
        
        row = (by * arith.index(16) + ty)._value
        col = (bx * arith.index(16) + tx)._value
        
        m_c = M
        n_c = N
        k_c = K
        one = 1
        
        # Create row-major layouts for matrices A, B, C
        a_shape = rocir.make_shape(m_c, k_c)
        a_stride = rocir.make_stride(k_c, one)  # A: (M x K) row-major
        a_layout = rocir.make_layout(a_shape, a_stride)
        
        b_shape = rocir.make_shape(k_c, n_c)
        b_stride = rocir.make_stride(n_c, one)  # B: (K x N) row-major
        b_layout = rocir.make_layout(b_shape, b_stride)
        
        c_shape = rocir.make_shape(m_c, n_c)
        c_stride = rocir.make_stride(n_c, one)  # C: (M x N) row-major
        c_layout = rocir.make_layout(c_shape, c_stride)
        
        # Create coordinate for current thread
        thread_coord = rocir.make_coord(row, col)
        c_idx = rocir.crd2idx(thread_coord, c_layout)
        
        row_valid = (row < m_c)._value
        col_valid = (col < n_c)._value
        valid = (row_valid & col_valid)._value
        
        if valid:
            sum_val = arith.f32(0.0)
            k_idx = arith.index(0)._value
            
            for_op = scf.ForOp(k_idx.value, arith.index(k_c).value, arith.index(one).value, [sum_val.value])
            with ir.InsertionPoint(for_op.body):
                k = for_op.induction_variable
                acc = for_op.inner_iter_args[0]
                
                # Use layout to compute A[row, k] linear address
                a_coord = rocir.make_coord(row, k)
                a_idx = rocir.crd2idx(a_coord, a_layout)
                a_val = memref.load(A, [a_idx.value if hasattr(a_idx, "value") else a_idx])
                
                # Use layout to compute B[k, col] linear address
                b_coord = rocir.make_coord(k, col)
                b_idx = rocir.crd2idx(b_coord, b_layout)
                b_val = memref.load(B, [b_idx.value if hasattr(b_idx, "value") else b_idx])
                
                prod = (a_val * b_val)._value
                new_acc = (acc + prod)._value
                
                scf.yield_([new_acc.value])
            
            result = for_op.results[0]
            # Use layout-computed linear index for C
            memref.store(result.value if hasattr(result, "value") else result, C, [c_idx.value if hasattr(c_idx, "value") else c_idx])

    ip.__exit__(None, None, None)
    
    hsaco = compile_to_hsaco(ctx.module)
    print(f" Compiled to HSACO: {len(hsaco)} bytes")
    
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

def test_matmul_shared_memory():
    """Matrix multiply with shared memory tiling optimization using SmemAllocator"""
    print("\n" + "="*80)
    print("Test 4: Optimized Matrix Multiply (Shared Memory Tiling)")
    print("Using: SmemAllocator for LDS shared memory")
    print("="*80)
    
    M, N, K = 256, 256, 256
    TILE_SIZE = 16
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    gpu.set_container_module(ctx.module)
    
    # Initialize Allocator
    allocator = SmemAllocator(ctx, arch=get_hip_arch())
    
    # Allocate Shared Memory for Tiles
    s_a_decl = allocator.allocate_array(T.f32(), TILE_SIZE * TILE_SIZE)
    s_b_decl = allocator.allocate_array(T.f32(), TILE_SIZE * TILE_SIZE)
    
    @gpu.module("matmul_shared", [f'#rocdl.target<chip = "{get_hip_arch()}", abi = "500">'])
    def gpu_mod():
        allocator.finalize()
    
    ip = ir.InsertionPoint.at_block_begin(gpu_mod.regions[0].blocks[0])
    ip.__enter__()
    
    @gpu.func(emit=True)
    def matmul_shared(A: T.memref(M, K, T.f32()), B: T.memref(K, N, T.f32()), C: T.memref(M, N, T.f32())):
        base_ptr = allocator.get_base()
        As = s_a_decl(base_ptr)
        Bs = s_b_decl(base_ptr)
        
        # TILE_SIZE is int, so use T.index() constructor properly or arith.constant with type
        # arith.constant(type, value)
        
        # Fix: T.index() returns a type, arith.constant(value, type=type)
        # Or arith.constant(type, value) if pos args correct.
        # But error says constant() takes 1 pos arg. It likely expects arith.constant(T.index(), value) -> arith.constant(value, type=type)
        # Actually arith.constant signature in python bindings: (value, type=None) or (type, value) depending on implementation.
        # Let's try standard mlir.dialects.arith.ConstantOp
        
        tile_size_const = arith.index(TILE_SIZE)
        row = (gpu.block_id("y") * tile_size_const + gpu.thread_id("y"))._value
        col = (gpu.block_id("x") * tile_size_const + gpu.thread_id("x"))._value
        
        tx = gpu.thread_id("x")
        ty = gpu.thread_id("y")
        
        zero = arith.index(0)
        one = arith.index(1)
        tile_c = tile_size_const
        k_c = arith.index(K)
        zero_f = arith.f32(0.0)
        
        acc = zero_f
        num_tiles = arith.index(K // TILE_SIZE)
        
        # Rocir Layout definition
        tile_size_idx = tile_size_const
        one_idx = one
        
        # Shape: (TILE_SIZE, TILE_SIZE)
        tile_shape = rocir.make_shape(tile_size_idx, tile_size_idx)
        # Stride: (TILE_SIZE, 1) -> Row Major
        tile_stride = rocir.make_stride(tile_size_idx, one_idx)
        tile_layout = rocir.make_layout(tile_shape, tile_stride)

        def get_tile_idx(y, x):
            coord = rocir.make_coord(y, x)
            idx_val = rocir.crd2idx(coord, tile_layout)
            return idx_val.value if hasattr(idx_val, 'value') else idx_val
        
        for_tiles = scf.ForOp(zero.value, num_tiles.value, one.value, [acc.value])
        with ir.InsertionPoint(for_tiles.body):
            t = for_tiles.induction_variable
            acc_val = for_tiles.inner_iter_args[0]
            k_base = (t * tile_c)._value
            
            a_col = (k_base + tx)._value
            a_val = memref.load(A, [row.value if hasattr(row, "value") else row, a_col.value if hasattr(a_col, "value") else a_col])
            # Store to As[ty, tx] -> As[ty * TILE + tx]
            As.store(a_val.value if hasattr(a_val, "value") else a_val, [get_tile_idx(ty.value, tx.value)])
            
            b_row = (k_base + ty)._value
            b_val = memref.load(B, [b_row.value if hasattr(b_row, "value") else b_row, col.value if hasattr(col, "value") else col])
            # Store to Bs[ty, tx] -> Bs[ty * TILE + tx]
            Bs.store(b_val.value if hasattr(b_val, "value") else b_val, [get_tile_idx(ty.value, tx.value)])
            
            gpu.barrier()
            
            for_k = scf.ForOp(zero.value, tile_c.value, one.value, [acc_val.value if hasattr(acc_val, "value") else acc_val])
            with ir.InsertionPoint(for_k.body):
                k_local = for_k.induction_variable
                acc_k = for_k.inner_iter_args[0]
                
                # Load from As[ty, k_local]
                a_smem = As.load([get_tile_idx(ty.value, k_local.value)])
                # Load from Bs[k_local, tx]
                b_smem = Bs.load([get_tile_idx(k_local.value, tx.value)])
                
                new_acc = (acc_k + a_smem * b_smem)._value
                
                scf.yield_([new_acc.value])
            
            gpu.barrier()
            scf.yield_([for_k.results[0].value if hasattr(for_k.results[0], "value") else for_k.results[0]])
        
        memref.store(for_tiles.results[0].value if hasattr(for_tiles.results[0], "value") else for_tiles.results[0], C, [row.value if hasattr(row, "value") else row, col.value if hasattr(col, "value") else col])
    
    ip.__exit__(None, None, None)
    
    hsaco = compile_to_hsaco(ctx.module)
    print(f" Compiled to HSACO: {len(hsaco)} bytes")
    print(f" Shared memory per block: {2 * TILE_SIZE * TILE_SIZE * 4} bytes")
    
    np.random.seed(789)
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
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"matmul_shared"))
    
    grid_x = (N + TILE_SIZE - 1) // TILE_SIZE
    grid_y = (M + TILE_SIZE - 1) // TILE_SIZE
    
    arg_ptrs = [ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_b)), ctypes.c_void_p(int(d_c))]
    args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])
    
    hip_check(hip.hipModuleLaunchKernel(kernel_func, grid_x, grid_y, 1, TILE_SIZE, TILE_SIZE, 1, 0, 0, args, None))
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
    print("\n" + "="*80)
    print("ROCm GPU Tests - Rocir Coordinate Operations in GPU Kernels")
    print(f"GPU: {get_hip_arch()}")
    print("="*80)
    
    # Demonstrate Rocir layout concepts
    demonstrate_rocir_layouts()
    
    # Run GPU tests with rocir operations
    result1 = test_vector_add()
    result2 = test_matrix_transpose()
    result3 = test_matmul()
    result4 = test_matmul_shared_memory()
    
    print("\n" + "="*80)
    if result1 and result2 and result3 and result4:
        print(" ALL GPU TESTS PASSED!")
        print("\n Rocir Coordinate Operations Fully Integrated:")
        print("• Vector operations use rocir 1D layouts")
        print("• Matrix operations use rocir 2D row-major layouts")
        print("• Coordinate indexing lowered to arithmetic via rocir-opt")
        print(f"• All tests verified on {get_hip_arch()} GPU")
        sys.exit(0)
    else:
        print("⚠️ SOME TESTS FAILED")
        sys.exit(1)
