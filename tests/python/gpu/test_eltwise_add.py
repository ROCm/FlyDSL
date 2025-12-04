#!/usr/bin/env python3
"""Elementwise Addition Example using RocDSL
This example demonstrates the RocDSL API pattern
- make_ordered_layout, make_layout_tv
- make_copy_atom, make_tiled_copy_tv
- get_slice, partition operations

The actual kernel uses a simplified implementation for AMD GPU.
"""

import sys
import os
import argparse
import numpy as np
import ctypes

# Setup paths
sys.path.insert(0, os.path.join(os.environ.get('MLIR_PATH', ''), 'tools/mlir/python_packages/mlir_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../build/python_bindings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../python'))

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.dialects.ext import func, gpu, rocir, rocm
from rocdsl.dialects.ext.arith import Index
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from mlir.ir import F16Type, F32Type, IndexType, InsertionPoint, Value
from mlir.dialects import memref, arith, scf
import mlir.extras.types as T
from hip import hip


def create_elementwise_add_kernel(M: int, N: int, dtype=F32Type):
    """Create elementwise addition kernel demonstrating RocDSL API.
    
    Args:
        M, N: Tensor dimensions
        dtype: Element type
        
    Returns:
        Compiled kernel module
    """
    print(f"\n[RocDSL INFO] Creating elementwise add kernel for {M}x{N}")
    print(f"[RocDSL INFO] Element type: {dtype}")
    
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    gpu.set_container_module(ctx.module)
    
    # Create GPU module
    @gpu.module("elementwise_kernels", ["#rocdl.target<abi = \"500\">"])
    def gpu_mod():
        pass
    
    # Set insertion point
    ip = InsertionPoint.at_block_begin(gpu_mod.regions[0].blocks[0])
    ip.__enter__()
    
    # Demonstrate RocDSL API usage (compile-time)
    print("\n[RocDSL Layout Algebra API Demo] - 展示所有核心API:")
    
    # ===== 1. Layout 创建 =====
    # make_ordered_layout: 创建带维度顺序的layout
    # 用于定义线程和数据的多维排布
    THR_M, THR_N = 4, 32  # Thread layout: 4行 x 32列的线程排布
    VAL_M, VAL_N = 4, 4    # Value layout: 每个线程处理4x4个元素
    
    # 创建Thread Layout (描述线程在块内的排布方式)
    thr_layout = rocir.make_ordered_layout((THR_M, THR_N), order=(1, 0))
    print(f"  ✓ make_ordered_layout: thread ({THR_M}, {THR_N})")
    
    # 创建Value Layout (描述每个线程处理的数据排布)
    val_layout = rocir.make_ordered_layout((VAL_M, VAL_N), order=(1, 0))
    print(f"  ✓ make_ordered_layout: value ({VAL_M}, {VAL_N})")
    
    # ===== 2. TV Layout 组合 =====
    # make_layout_tv: 组合thread和value layout
    # 这是Layout Algebra的核心：thread-value分解
    tiler_mn, tv_layout = rocir.make_layout_tv(thr_layout, val_layout)
    print(f"  ✓ make_layout_tv: combined thread+value layouts")
    
    # ===== 3. Copy Atom 创建 =====
    # make_copy_atom: 定义原子拷贝操作
    # 指定数据类型和向量化宽度
    copy_atom_load = rocir.make_copy_atom(dtype.get(), vector_size=8)
    copy_atom_store = rocir.make_copy_atom(dtype.get(), vector_size=8)
    print(f"  ✓ make_copy_atom: vector_size=8 for G2R and R2G")
    
    # ===== 4. Tiled Copy 创建 =====
    # make_tiled_copy_tv: 基于TV layout创建tiled copy
    # 这定义了如何在线程间分配和拷贝数据
    tiled_copy_A = rocir.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_B = rocir.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_C = rocir.make_tiled_copy_tv(copy_atom_store, thr_layout, val_layout)
    print(f"  ✓ make_tiled_copy_tv: created for A, B, C tensors")
    
    # ===== 5. Per-Thread Slice =====
    # get_slice: 获取特定线程的slice
    # 用于将全局tiled copy分解为每个线程的操作
    tidx_demo = Index(0)
    thr_copy_A = tiled_copy_A.get_slice(tidx_demo)
    print(f"  ✓ get_slice: per-thread copy slice")
    
    print("\n[RocDSL API Demo] All RocDSL API operations completed successfully!")
    print("[RocDSL INFO] Now generating actual GPU kernel...\n")
    
    # Create actual GPU kernel - 对齐 CUTLASS elementwise_add 实现
    @gpu.func(emit=True)
    def elementwise_add_kernel(A: T.memref(M, N, dtype.get()), 
                               B: T.memref(M, N, dtype.get()), 
                               C: T.memref(M, N, dtype.get())):
        
        # Helper to unwrap ArithValue to raw MLIR Value
        def unwrap(val):
            return val.value if hasattr(val, "value") else val
        
        def to_value(val):
            if isinstance(val, Value):
                return val
            if hasattr(val, "value"):
                return val.value
            return Index(val).value
        
        # ===== Step 1: Thread and Block IDs =====
        tid_x = gpu.thread_id("x")
        tid_y = gpu.thread_id("y")
        bid_x = gpu.block_id("x")
        bid_y = gpu.block_id("y")
        
        # Calculate linear thread index
        bdim_x = gpu.block_dim("x")
        tidx = (tid_y * bdim_x + tid_x)._value
        
        # Block coordinates
        blk_coord_y = bid_y._value
        blk_coord_x = bid_x._value
        
        # ===== Step 2: Re-create TiledCopy and Layouts (Kernel Scope) =====
        # Since we need these for logic inside the kernel
        THR_M, THR_N = 4, 32
        VAL_M, VAL_N = 4, 4
        
        thr_layout = rocir.make_ordered_layout((THR_M, THR_N), order=(1, 0))
        val_layout = rocir.make_ordered_layout((VAL_M, VAL_N), order=(1, 0))
        
        # Atoms
        copy_atom_load = rocir.make_copy_atom(dtype.get(), vector_size=8)
        copy_atom_store = rocir.make_copy_atom(dtype.get(), vector_size=8)
        
        # Tiled Copies
        tiled_copy_A = rocir.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout,
                                               thr_shape=(THR_M, THR_N), val_shape=(VAL_M, VAL_N))
        tiled_copy_B = rocir.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout,
                                               thr_shape=(THR_M, THR_N), val_shape=(VAL_M, VAL_N))
        tiled_copy_C = rocir.make_tiled_copy_tv(copy_atom_store, thr_layout, val_layout,
                                               thr_shape=(THR_M, THR_N), val_shape=(VAL_M, VAL_N))
        
        global_shape = rocir.make_shape(M, N)
        global_stride = rocir.make_stride(N, 1)
        global_layout = rocir.make_layout(global_shape, global_stride)
        
        tensor_A = rocir.make_tensor(A, global_layout, shape=(M, N))
        tensor_B = rocir.make_tensor(B, global_layout, shape=(M, N))
        tensor_C = rocir.make_tensor(C, global_layout, shape=(M, N))
        
        # ===== Step 3: Tiling and Partitioning =====
        TILE_M = THR_M * VAL_M
        TILE_N = THR_N * VAL_N
        
        # Block start
        blk_start_row = unwrap(arith.MulIOp(to_value(blk_coord_y), to_value(Index(TILE_M).value)).result)
        blk_start_col = unwrap(arith.MulIOp(to_value(blk_coord_x), to_value(Index(TILE_N).value)).result)
        
        # Thread mapping (manual for now, derived from layouts)
        thr_in_tile = tidx % Index(THR_M * THR_N)
        thr_row = unwrap((thr_in_tile // Index(THR_N)).value)
        thr_col = unwrap((thr_in_tile % Index(THR_N)).value)
        
        # Precompute bounds
        limit_M = to_value(Index(M).value)
        limit_N = to_value(Index(N).value)
        
        val_shape = tiled_copy_A.val_shape
        
        thrA = rocir.partition_src(tiled_copy_A, tensor_A, tidx)
        thrB = rocir.partition_src(tiled_copy_B, tensor_B, tidx)
        thrC = rocir.partition_dst(tiled_copy_C, tensor_C, tidx)
        
        cond = None
        global_limits = [limit_M, limit_N]
        for dim in range(len(val_shape)):
            base = to_value(thrA.base_coords[dim])
            extent = to_value(Index(val_shape[dim]).value)
            end_coord = unwrap(arith.AddIOp(base, extent).result)
            limit = global_limits[dim]
            dim_cond = unwrap(arith.CmpIOp(arith.CmpIPredicate.ule, end_coord, limit).result)
            cond = dim_cond if cond is None else unwrap(arith.AndIOp(cond, dim_cond).result)
        
        if_op = scf.IfOp(cond)
        with InsertionPoint(if_op.then_block):
            frgA = rocir.make_fragment_like(tiled_copy_A, dtype.get())
            frgB = rocir.make_fragment_like(tiled_copy_B, dtype.get())
            frgC = rocir.make_fragment_like(tiled_copy_C, dtype.get())
            
            rocir.copy(tiled_copy_A, thrA, frgA)
            rocir.copy(tiled_copy_B, thrB, frgB)
            
            for i in range(val_shape[0]):
                idx_i = unwrap(Index(i).value)
                for j in range(val_shape[1]):
                    idx_j = unwrap(Index(j).value)
                    idx_coords = [idx_i, idx_j]
                    a_val = unwrap(memref.load(frgA.memref, idx_coords))
                    b_val = unwrap(memref.load(frgB.memref, idx_coords))
                    c_val = unwrap(arith.AddFOp(a_val, b_val).result)
                    memref.store(c_val, frgC.memref, idx_coords)
            
            rocir.copy(tiled_copy_C, frgC, thrC)
            scf.YieldOp([])
    
    ip.__exit__(None, None, None)
    
    print(f"[RocDSL INFO] Generated MLIR module")
    
    return ctx.module


def compile_and_run(M, N, dtype=F32Type, benchmark=False, iterations=100):
    """Compile and run the elementwise add kernel."""
    
    print("\n" + "="*80)
    print("RocDSL Elementwise Addition Test")
    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Element type: {dtype}")
    print(f"GPU: {get_hip_arch()}")
    print("="*80)
    
    # Create kernel
    module = create_elementwise_add_kernel(M, N, dtype)
    
    # Compile to HSACO  
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from utils import compile_to_hsaco
    from rocdsl.compiler.pipeline import run_pipeline, Pipeline
    
    # Run optimization pipeline
    print(f"[RocDSL INFO] Running optimization pipeline...")
    optimized = run_pipeline(module, Pipeline().canonicalize().cse())
    
    # Compile to HSACO
    print(f"[RocDSL INFO] Compiling to HSACO...")
    hsaco = compile_to_hsaco(optimized, kernel_name="elementwise_add_kernel")
    print(f"\n[RocDSL INFO] Compiled to HSACO: {len(hsaco)} bytes")
    
    # Prepare data
    np.random.seed(42)
    torch_dtype = np.float32 if dtype == F32Type else np.float16
    a_host = np.random.randn(M, N).astype(torch_dtype)
    b_host = np.random.randn(M, N).astype(torch_dtype)
    c_host = np.zeros((M, N), dtype=torch_dtype)
    
    # Allocate device memory
    size_bytes = M * N * a_host.itemsize
    d_a = hip_check(hip.hipMalloc(size_bytes))
    d_b = hip_check(hip.hipMalloc(size_bytes))
    d_c = hip_check(hip.hipMalloc(size_bytes))
    
    # Copy to device
    hip_check(hip.hipMemcpy(d_a, a_host.ctypes.data, size_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(d_b, b_host.ctypes.data, size_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    
    # Load kernel
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"elementwise_add_kernel"))
    
    # Launch configuration
    BLOCK_X, BLOCK_Y = 16, 16
    grid_x = (N + BLOCK_X - 1) // BLOCK_X
    grid_y = (M + BLOCK_Y - 1) // BLOCK_Y
    
    print(f"\n[RocDSL INFO] Launch configuration:")
    print(f"  Grid: ({grid_x}, {grid_y}, 1)")
    print(f"  Block: ({BLOCK_X}, {BLOCK_Y}, 1)")
    
    # Prepare arguments
    arg_ptrs = [
        ctypes.c_void_p(int(d_a)),
        ctypes.c_void_p(int(d_b)),
        ctypes.c_void_p(int(d_c))
    ]
    args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])
    
    # Launch kernel
    hip_check(hip.hipModuleLaunchKernel(
        kernel_func,
        grid_x, grid_y, 1,  # grid dim
        BLOCK_X, BLOCK_Y, 1,   # block dim
        0,  # shared mem
        None,  # stream
        args,
        None
    ))
    
    # Copy result back
    hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, size_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    # Verify results
    expected = a_host + b_host
    error = np.max(np.abs(c_host - expected))
    
    print(f"\n[RocDSL INFO] Verification:")
    print(f"  Max error: {error:.2e}")
    if M <= 4 and N <= 16:
        print(f"  Input A:\n{a_host}")
        print(f"  Input B:\n{b_host}")
        print(f"  Result C:\n{c_host}")
        print(f"  Expected:\n{expected}")
    else:
        print(f"  First few elements:")
        print(f"    A: {a_host[:2, :4]}")
        print(f"    B: {b_host[:2, :4]}")
        print(f"    C (result): {c_host[:2, :4]}")
        print(f"    Expected: {expected[:2, :4]}")
    
    # Benchmark if requested
    if benchmark:
        print(f"\n[RocDSL INFO] Running benchmark ({iterations} iterations)...")
        import time
        
        start_event = hip_check(hip.hipEventCreate())
        stop_event = hip_check(hip.hipEventCreate())
        
        # Warmup
        for _ in range(10):
            hip_check(hip.hipModuleLaunchKernel(
                kernel_func, grid_x, grid_y, 1, BLOCK_X, BLOCK_Y, 1,
                0, None, args, None
            ))
        hip_check(hip.hipDeviceSynchronize())
        
        # Benchmark
        hip_check(hip.hipEventRecord(start_event, None))
        for _ in range(iterations):
            hip_check(hip.hipModuleLaunchKernel(
                kernel_func, grid_x, grid_y, 1, BLOCK_X, BLOCK_Y, 1,
                0, None, args, None
            ))
        hip_check(hip.hipEventRecord(stop_event, None))
        hip_check(hip.hipEventSynchronize(stop_event))
        
        elapsed_ms = ctypes.c_float()
        hip_check(hip.hipEventElapsedTime(ctypes.byref(elapsed_ms), start_event, stop_event))
        
        avg_time_ms = elapsed_ms.value / iterations
        bandwidth_gb = (3 * M * N * a_host.itemsize) / (avg_time_ms / 1e3) / 1e9
        
        print(f"  Average time: {avg_time_ms:.4f} ms")
        print(f"  Bandwidth: {bandwidth_gb:.2f} GB/s")
        
        hip_check(hip.hipEventDestroy(start_event))
        hip_check(hip.hipEventDestroy(stop_event))
    
    # Cleanup
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))
    
    passed = error < 1e-4
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Elementwise add example using RocDSL"
    )
    parser.add_argument("--M", default=4, type=int, help="Number of rows")
    parser.add_argument("--N", default=16, type=int, help="Number of columns")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--iterations", default=100, type=int, help="Benchmark iterations")
    
    args = parser.parse_args()
    
    passed = compile_and_run(
        args.M, args.N,
        dtype=F32Type,
        benchmark=args.benchmark,
        iterations=args.iterations
    )
    
    print("\n" + "="*80)
    if passed:
        print("✅ PASS - Elementwise add test completed successfully!")
        print("\nThis example demonstrated the RocDSL API pattern:")
        print("  1. make_ordered_layout - create layouts with dimension ordering")
        print("  2. make_layout_tv - create TV layout from thread and value layouts")
        print("  3. make_copy_atom - create copy operation descriptors")
        print("  4. make_tiled_copy_tv - create tiled copies")
        print("  5. get_slice - get per-thread slices")
    else:
        print("❌ FAIL - Verification failed!")
    print("="*80)
    
    sys.exit(0 if passed else 1)
