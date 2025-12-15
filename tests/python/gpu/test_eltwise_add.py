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
import pytest

# Setup paths


from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.dialects.ext import func, gpu, rocir, rocm
from rocdsl.dialects.ext.arith import Index
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from mlir.ir import F16Type, F32Type, IntegerType, InsertionPoint
from mlir.dialects import arith
import mlir.extras.types as T
from hip import hip
from tests.test_common import run_perftest


THR_M, THR_N = 4, 32
VAL_M, VAL_N = 4, 4
COPY_VEC = 8

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
    
    @gpu.func(emit=True)
    def elementwise_add_kernel(A: T.memref(M, N, dtype.get()), 
                               B: T.memref(M, N, dtype.get()), 
                               C: T.memref(M, N, dtype.get())):
        # ===== Step 1: Thread and Block IDs =====
        tid_x = rocir.thread_idx("x")
        tid_y = rocir.thread_idx("y")
        bid_x = rocir.block_idx("x")
        bid_y = rocir.block_idx("y")
        
        # Calculate linear thread index
        bdim_x = rocir.block_dim("x")
        tidx = (tid_y * bdim_x + tid_x).value
        
        # Block coordinates
        blk_coord_y = bid_y
        blk_coord_x = bid_x
        
        # ===== Step 2: TiledCopy + Layouts =====
        thr_layout = rocir.make_ordered_layout((THR_M, THR_N), order=(1, 0))
        val_layout = rocir.make_ordered_layout((VAL_M, VAL_N), order=(1, 0))
        
        # Atoms
        copy_atom_load = rocir.make_copy_atom(dtype.get(), vector_size=COPY_VEC)
        copy_atom_store = rocir.make_copy_atom(dtype.get(), vector_size=COPY_VEC)
        
        # Tiled Copies
        tiled_copy_A = rocir.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout,
                                               thr_shape=(THR_M, THR_N), val_shape=(VAL_M, VAL_N))
        tiled_copy_B = rocir.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout,
                                               thr_shape=(THR_M, THR_N), val_shape=(VAL_M, VAL_N))
        tiled_copy_C = rocir.make_tiled_copy_tv(copy_atom_store, thr_layout, val_layout,
                                               thr_shape=(THR_M, THR_N), val_shape=(VAL_M, VAL_N))
        
        tensor_A = rocir.make_tensor(A, shape=(M, N), strides=(N, 1))
        tensor_B = rocir.make_tensor(B, shape=(M, N), strides=(N, 1))
        tensor_C = rocir.make_tensor(C, shape=(M, N), strides=(N, 1))
        
        TILE_M = THR_M * VAL_M
        TILE_N = THR_N * VAL_N
        tile_shape = (TILE_M, TILE_N)
        gA = rocir.zipped_divide(tensor_A, tile_shape)
        gB = rocir.zipped_divide(tensor_B, tile_shape)
        gC = rocir.zipped_divide(tensor_C, tile_shape)
        idC = rocir.make_identity_tensor((M, N))
        cC = rocir.zipped_divide(idC, tile_shape)
        
        blk_coord = (blk_coord_y, blk_coord_x)
        blkA = gA[blk_coord]
        blkB = gB[blk_coord]
        blkC = gC[blk_coord]
        blkCrd = cC[blk_coord]
        
        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        thr_copy_C = tiled_copy_C.get_slice(tidx)
        
        thrA = thr_copy_A.partition_S(blkA)
        thrB = thr_copy_B.partition_S(blkB)
        thrC = thr_copy_C.partition_S(blkC)
        thrCrd = thr_copy_C.partition_S(blkCrd)
        
        val_shape = tiled_copy_A.val_shape
        frgA = rocir.make_fragment_like(thrA, dtype.get())
        frgB = rocir.make_fragment_like(thrB, dtype.get())
        frgC = rocir.make_fragment_like(thrC, dtype.get())
        
        pred_ty = IntegerType.get_signless(1)
        frgPred = rocir.make_rmem_tensor(val_shape, pred_ty)
        total_vals = val_shape[0] * val_shape[1]
        for linear in range(total_vals):
            lin_idx = rocir.const_index(linear)
            coords = thrCrd.coords_from_linear(lin_idx)
            pred_val = rocir.elem_less(coords, (M, N))
            pred_offsets = tuple(frgPred.offsets_from_linear(lin_idx))
            frgPred[pred_offsets] = pred_val
        
        rocir.copy(tiled_copy_A, thrA, frgA, pred=frgPred)
        rocir.copy(tiled_copy_B, thrB, frgB, pred=frgPred)
        
        for i in range(val_shape[0]):
            idx_i = rocir.const_index(i)
            for j in range(val_shape[1]):
                idx_j = rocir.const_index(j)
                coords = (idx_i, idx_j)
                a_val = frgA[coords]
                b_val = frgB[coords]
                c_val = a_val + b_val
                frgC[coords] = c_val
        
        rocir.copy(tiled_copy_C, frgC, thrC, pred=frgPred)
    
    ip.__exit__(None, None, None)
    
    print(f"[RocDSL INFO] Generated MLIR module")
    
    return ctx.module


# Test cases
TEST_SHAPES = [
    (1, 3),
    (129, 255),
    (1021, 515),
    # (5120, 4000),
]

# Compute Max Dims for shared kernel
MAX_M = max(s[0] for s in TEST_SHAPES)
MAX_N = max(s[1] for s in TEST_SHAPES)

# Cache compiled kernel (singleton)
_compiled_kernel_hsaco = None

def get_or_compile_kernel(dtype=F32Type):
    """Get or compile the single shared kernel for MAX dimensions."""
    global _compiled_kernel_hsaco
    
    if _compiled_kernel_hsaco is None:
        print(f"\n[RocDSL INFO] Compiling SHARED kernel for max dimensions: {MAX_M}x{MAX_N}")
        
        # Create kernel with MAX dimensions
        module = create_elementwise_add_kernel(MAX_M, MAX_N, dtype)
        
        # Compile to HSACO  
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
        from tests.utils import compile_to_hsaco
        from rocdsl.compiler.pipeline import run_pipeline, Pipeline
        
        # Run optimization pipeline
        print(f"[RocDSL INFO] Running optimization pipeline...")
        optimized = run_pipeline(module, Pipeline().canonicalize().cse())
        
        # Compile to HSACO
        print(f"[RocDSL INFO] Compiling to HSACO...")
        hsaco = compile_to_hsaco(optimized, kernel_name="elementwise_add_kernel")
        print(f"[RocDSL INFO] Compiled to HSACO: {len(hsaco)} bytes")
        _compiled_kernel_hsaco = hsaco
    else:
        print(f"[RocDSL INFO] Reusing SHARED kernel (max: {MAX_M}x{MAX_N})")
    
    return _compiled_kernel_hsaco


@pytest.mark.parametrize(
    ("M", "N"),
    TEST_SHAPES,
)
def test_compile_and_run(M, N, dtype=F32Type, benchmark=False, iterations=100):
    """Compile and run the elementwise add kernel."""
    
    print("\n" + "="*80)
    print("RocDSL Elementwise Addition Test")
    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Element type: {dtype}")
    print(f"GPU: {get_hip_arch()}")
    print("="*80)
    
    # Kernel selection:
    # - If the requested MxN fits within TEST_SHAPES' MAX_M/MAX_N, reuse the shared MAX kernel
    #   and use padded buffers.
    # - If the requested M or N exceeds MAX, compile an exact-shape kernel and skip padding.
    use_shared_max_kernel = (M <= MAX_M) and (N <= MAX_N)
    if use_shared_max_kernel:
        hsaco = get_or_compile_kernel(dtype)
        compile_M, compile_N = MAX_M, MAX_N
    else:
        print(f"\n[RocDSL INFO] Requested shape {M}x{N} exceeds shared max {MAX_M}x{MAX_N}; compiling exact-shape kernel.")
        from rocdsl.compiler.pipeline import run_pipeline, Pipeline
        from tests.utils import compile_to_hsaco
        module = create_elementwise_add_kernel(M, N, dtype)
        optimized = run_pipeline(module, Pipeline().canonicalize().cse())
        hsaco = compile_to_hsaco(optimized, kernel_name="elementwise_add_kernel")
        compile_M, compile_N = M, N
    
    # Prepare data
    np.random.seed(42)
    torch_dtype = np.float32 if dtype == F32Type else np.float16
    a_host = np.random.randn(M, N).astype(torch_dtype)
    b_host = np.random.randn(M, N).astype(torch_dtype)
    c_host = np.zeros((M, N), dtype=torch_dtype)
    
    if use_shared_max_kernel:
        # Padded data for shared kernel (must match MAX_M x MAX_N strides)
        # The kernel expects buffers of size MAX_M * MAX_N with stride MAX_N
        a_padded = np.zeros((MAX_M, MAX_N), dtype=torch_dtype)
        b_padded = np.zeros((MAX_M, MAX_N), dtype=torch_dtype)
        c_padded = np.zeros((MAX_M, MAX_N), dtype=torch_dtype)
        
        a_padded[:M, :N] = a_host
        b_padded[:M, :N] = b_host
        
        # Allocate device memory (full MAX size)
        max_size_bytes = MAX_M * MAX_N * a_host.itemsize
        d_a = hip_check(hip.hipMalloc(max_size_bytes))
        d_b = hip_check(hip.hipMalloc(max_size_bytes))
        d_c = hip_check(hip.hipMalloc(max_size_bytes))
        
        # Copy to device (padded)
        hip_check(hip.hipMemcpy(d_a, a_padded.ctypes.data, max_size_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        hip_check(hip.hipMemcpy(d_b, b_padded.ctypes.data, max_size_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    else:
        # Exact-shape kernel: no padding
        max_size_bytes = M * N * a_host.itemsize
        d_a = hip_check(hip.hipMalloc(max_size_bytes))
        d_b = hip_check(hip.hipMalloc(max_size_bytes))
        d_c = hip_check(hip.hipMalloc(max_size_bytes))
        hip_check(hip.hipMemcpy(d_a, a_host.ctypes.data, max_size_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        hip_check(hip.hipMemcpy(d_b, b_host.ctypes.data, max_size_bytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    
    # Load kernel
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"elementwise_add_kernel"))
    
    # Launch configuration (must match kernel tiling)
    # - blockDim should provide exactly THR_M*THR_N threads (tidx in [0, THR_M*THR_N))
    # - gridDim should be computed in units of the kernel's tile coverage
    BLOCK_X, BLOCK_Y = THR_N, THR_M
    TILE_M = THR_M * VAL_M
    TILE_N = THR_N * VAL_N
    grid_x = (N + TILE_N - 1) // TILE_N
    grid_y = (M + TILE_M - 1) // TILE_M
    
    print(f"\n[RocDSL INFO] Launch configuration:")
    kernel_tag = f"{compile_M}x{compile_N}"
    print(f"  Grid: ({grid_x}, {grid_y}, 1) [Tiles: {TILE_M}x{TILE_N}, kernel={kernel_tag}]")
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
        grid_x, grid_y, 1,  # grid dim (partial)
        BLOCK_X, BLOCK_Y, 1,   # block dim
        0,  # shared mem
        None,  # stream
        args,
        None
    ))
    
    if use_shared_max_kernel:
        # Copy result back (full padded buffer)
        hip_check(hip.hipMemcpy(c_padded.ctypes.data, d_c, max_size_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
        # Extract relevant part
        c_host = c_padded[:M, :N]
    else:
        hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, max_size_bytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    # Verify results
    expected = a_host + b_host
    error = np.max(np.abs(c_host - expected))
    
    print(f"\n[RocDSL INFO] Verification:")
    print(f"  Max error: {error:.2e}")
    
    # Benchmark if requested
    if benchmark:
        # Use the shared perf harness (same as vecAdd benchmark) so results are comparable.
        # NOTE: run_perftest uses torch.profiler on ROCm; it expects torch to be installed.
        print(f"\n[RocDSL INFO] Running benchmark via run_perftest ({iterations} iterations)...")
        try:
            import torch  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "Benchmark requested but torch is unavailable. "
                "Install torch ROCm build or run without --benchmark."
            ) from e

        def hip_kernel_launch():
            hip_check(
                hip.hipModuleLaunchKernel(
                    kernel_func,
                    grid_x,
                    grid_y,
                    1,
                    BLOCK_X,
                    BLOCK_Y,
                    1,
                    0,
                    None,
                    args,
                    None,
                )
            )

        # run_perftest returns (data, avg_us)
        _, avg_us = run_perftest(hip_kernel_launch, num_iters=iterations, num_warmup=10)
        hip_check(hip.hipDeviceSynchronize())

        total_bytes = 3 * M * N * a_host.itemsize
        bandwidth_gb = total_bytes / (avg_us / 1e6) / 1e9
        avg_time_ms = avg_us / 1000.0

        print(f"  Average time: {avg_time_ms:.4f} ms")
        print(f"  Bandwidth: {bandwidth_gb:.2f} GB/s")

        # Standardized bandwidth line so run_tests.sh can pick it up (like matrixTranspose / vecAdd).
        print(f"\nBandwidth: {bandwidth_gb:.2f} GB/s")

        results = {
            "avg_ms": avg_time_ms,
            "avg_us": avg_us,
            "bandwidth_gbs": bandwidth_gb,
            "size": M * N,
            "total_bytes": total_bytes,
        }
        print(f"\n{results}")
    
    # Cleanup
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))
    assert error < 1e-4


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Elementwise add example using RocDSL"
    )
    parser.add_argument("--M", default=129, type=int, help="Number of rows")
    parser.add_argument("--N", default=255, type=int, help="Number of columns")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--iterations", default=100, type=int, help="Benchmark iterations")
    parser.add_argument("--run-all-shapes", action="store_true", help="Run all test shapes")
    
    args = parser.parse_args()
    
    if args.run_all_shapes:
        # Run all shapes to demonstrate kernel caching
        print("\n" + "="*80)
        print("Running all test shapes (demonstrates kernel caching)")
        print("="*80)
        for M, N in TEST_SHAPES:
            print(f"\n--- Testing shape {M}x{N} (1st run) ---")
            test_compile_and_run(M, N, dtype=F32Type, benchmark=False)
            
            # Run again to show kernel is reused
            print(f"\n--- Testing shape {M}x{N} (2nd run - should reuse kernel) ---")
            test_compile_and_run(M, N, dtype=F32Type, benchmark=False)
        
        print("\nPASS - All shapes tested with kernel caching!")
    else:
        test_compile_and_run(
            args.M, args.N,
            dtype=F32Type,
            benchmark=args.benchmark,
            iterations=args.iterations
        )
        
        print("PASS - Elementwise add test completed successfully!")
