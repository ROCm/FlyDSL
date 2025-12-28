#!/usr/bin/env python3
"""Elementwise Addition Example using FLIR
This example demonstrates the FLIR API pattern
- make_ordered_layout, make_layout_tv
- make_copy_atom, make_tiled_copy_tv
- get_slice, partition operations

The actual kernel uses a simplified implementation for AMD GPU.
"""

import sys
import os
import argparse
import numpy as np
import pytest
import torch
import pyflir

# Setup paths


from pyflir.dialects.ext import flir
from pyflir.dialects.ext.python_control_flow import range_constexpr
from pyflir.dialects.ext.arith import Index
from pyflir.runtime.device import get_rocm_arch
from _mlir.ir import F16Type, F32Type, IntegerType
from _mlir.dialects import arith
import _mlir.extras.types as T
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)
from tests.test_common import run_perftest


THR_M, THR_N = 4, 32
VAL_M, VAL_N = 4, 4
COPY_VEC = 8

def create_elementwise_add_kernel(M: int, N: int, dtype=F32Type):
    """Create elementwise addition kernel demonstrating FLIR API.
    
    Args:
        M, N: Tensor dimensions
        dtype: Element type
        
    Returns:
        Compiled kernel module
    """
    print(f"\n[FLIR INFO] Creating elementwise add kernel for {M}x{N}")
    print(f"[FLIR INFO] Element type: {dtype}")
    
    class _EltwiseAdd(flir.MlirModule):
        GPU_MODULE_NAME = "elementwise_kernels"
        GPU_MODULE_TARGETS = ['#rocdl.target<abi = "500">']

        @flir.kernel
        def elementwise_add_kernel(
            self: flir.T.i64,
            A: lambda: T.memref(M, N, dtype.get()),
            B: lambda: T.memref(M, N, dtype.get()),
            C: lambda: T.memref(M, N, dtype.get()),
        ):
            # ===== Step 1: Thread and Block IDs =====
            tid_x = flir.thread_idx("x")
            tid_y = flir.thread_idx("y")
            bid_x = flir.block_idx("x")
            bid_y = flir.block_idx("y")

            # Calculate linear thread index
            bdim_x = flir.block_dim("x")
            tidx = (tid_y * bdim_x + tid_x).value

            # Block coordinates
            blk_coord_y = bid_y
            blk_coord_x = bid_x

            # ===== Step 2: TiledCopy + Layouts =====
            thr_layout = flir.make_ordered_layout((THR_M, THR_N), order=(1, 0))
            val_layout = flir.make_ordered_layout((VAL_M, VAL_N), order=(1, 0))

            # Atoms
            copy_atom_load = flir.make_copy_atom(dtype.get(), vector_size=COPY_VEC)
            copy_atom_store = flir.make_copy_atom(dtype.get(), vector_size=COPY_VEC)

            # Tiled Copies
            tiled_copy_A = flir.make_tiled_copy_tv(
                copy_atom_load,
                thr_layout,
                val_layout,
                thr_shape=(THR_M, THR_N),
                val_shape=(VAL_M, VAL_N),
            )
            tiled_copy_B = flir.make_tiled_copy_tv(
                copy_atom_load,
                thr_layout,
                val_layout,
                thr_shape=(THR_M, THR_N),
                val_shape=(VAL_M, VAL_N),
            )
            tiled_copy_C = flir.make_tiled_copy_tv(
                copy_atom_store,
                thr_layout,
                val_layout,
                thr_shape=(THR_M, THR_N),
                val_shape=(VAL_M, VAL_N),
            )

            tensor_A = flir.make_tensor(A, shape=(M, N), strides=(N, 1))
            tensor_B = flir.make_tensor(B, shape=(M, N), strides=(N, 1))
            tensor_C = flir.make_tensor(C, shape=(M, N), strides=(N, 1))

            TILE_M = THR_M * VAL_M
            TILE_N = THR_N * VAL_N
            tile_shape = (TILE_M, TILE_N)
            gA = flir.zipped_divide(tensor_A, tile_shape)
            gB = flir.zipped_divide(tensor_B, tile_shape)
            gC = flir.zipped_divide(tensor_C, tile_shape)
            idC = flir.make_identity_tensor((M, N))
            cC = flir.zipped_divide(idC, tile_shape)

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
            frgA = flir.make_fragment_like(thrA, dtype.get())
            frgB = flir.make_fragment_like(thrB, dtype.get())
            frgC = flir.make_fragment_like(thrC, dtype.get())

            pred_ty = IntegerType.get_signless(1)
            frgPred = flir.make_rmem_tensor(val_shape, pred_ty)
            total_vals = val_shape[0] * val_shape[1]
            for linear in range_constexpr(total_vals):
                lin_idx = flir.const_index(linear)
                coords = thrCrd.coords_from_linear(lin_idx)
                pred_val = flir.elem_less(coords, (M, N))
                pred_offsets = tuple(frgPred.offsets_from_linear(lin_idx))
                frgPred[pred_offsets] = pred_val

            flir.copy(tiled_copy_A, thrA, frgA, pred=frgPred)
            flir.copy(tiled_copy_B, thrB, frgB, pred=frgPred)

            for i in range_constexpr(val_shape[0]):
                idx_i = flir.const_index(i)
                for j in range_constexpr(val_shape[1]):
                    idx_j = flir.const_index(j)
                    coords = (idx_i, idx_j)
                    a_val = frgA[coords]
                    b_val = frgB[coords]
                    c_val = a_val + b_val
                    frgC[coords] = c_val

            flir.copy(tiled_copy_C, frgC, thrC, pred=frgPred)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A: lambda: T.memref(M, N, dtype.get()),
            B: lambda: T.memref(M, N, dtype.get()),
            C: lambda: T.memref(M, N, dtype.get()),
        ):
            c1 = Index(1).value
            tile_m = THR_M * VAL_M
            tile_n = THR_N * VAL_N
            gx = Index((N + tile_n - 1) // tile_n).value
            gy = Index((M + tile_m - 1) // tile_m).value
            bdx = Index(THR_N).value
            bdy = Index(THR_M).value
            flir.gpu_ext.LaunchFuncOp(
                ["elementwise_kernels", "elementwise_add_kernel"],
                grid_size=(gx, gy, c1),
                block_size=(bdx, bdy, c1),
                kernel_operands=[A, B, C],
            )

    print("[FLIR INFO] Generated MLIR module")
    return _EltwiseAdd()


# Test cases
TEST_SHAPES = [
    (1, 3),
    (129, 255),
    # (1021, 515),
    # (5120, 4000),
]

# Compute Max Dims for shared kernel
MAX_M = max(s[0] for s in TEST_SHAPES)
MAX_N = max(s[1] for s in TEST_SHAPES)

# Cache compiled kernel executor (singleton)
_compiled_kernel_exe = None

def get_or_compile_kernel(dtype=F32Type):
    """Get or compile the single shared kernel for MAX dimensions."""
    global _compiled_kernel_exe
    
    if _compiled_kernel_exe is None:
        print(f"\n[FLIR INFO] Compiling SHARED kernel for max dimensions: {MAX_M}x{MAX_N}")
        
        # Create kernel with MAX dimensions
        m = create_elementwise_add_kernel(MAX_M, MAX_N, dtype)
        print(f"[FLIR INFO] Compiling via pyflir.compile...")
        _compiled_kernel_exe = pyflir.compile(m)
    else:
        print(f"[FLIR INFO] Reusing SHARED kernel (max: {MAX_M}x{MAX_N})")
    
    return _compiled_kernel_exe


@pytest.mark.parametrize(
    ("M", "N"),
    TEST_SHAPES,
)
def test_compile_and_run(M, N, dtype=F32Type, benchmark=False, iterations=100):
    """Compile and run the elementwise add kernel."""
    
    print("\n" + "="*80)
    print("FLIR Elementwise Addition Test")
    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Element type: {dtype}")
    print(f"GPU: {get_rocm_arch()}")
    print("="*80)
    
    # Kernel selection:
    # - If the requested MxN fits within TEST_SHAPES' MAX_M/MAX_N, reuse the shared MAX kernel
    #   and use padded buffers.
    # - If the requested M or N exceeds MAX, compile an exact-shape kernel and skip padding.
    use_shared_max_kernel = (M <= MAX_M) and (N <= MAX_N)
    if use_shared_max_kernel:
        exe = get_or_compile_kernel(dtype)
        compile_M, compile_N = MAX_M, MAX_N
    else:
        print(f"\n[FLIR INFO] Requested shape {M}x{N} exceeds shared max {MAX_M}x{MAX_N}; compiling exact-shape kernel.")
        m = create_elementwise_add_kernel(M, N, dtype)
        exe = pyflir.compile(m)
        compile_M, compile_N = M, N
    
    # Prepare data
    np.random.seed(42)
    torch_dtype = np.float32 if dtype == F32Type else np.float16
    a_host = np.random.randn(M, N).astype(torch_dtype)
    b_host = np.random.randn(M, N).astype(torch_dtype)
    expected = a_host + b_host
    
    if use_shared_max_kernel:
        # Padded data for shared kernel (must match MAX_M x MAX_N strides)
        # The kernel expects buffers of size MAX_M * MAX_N with stride MAX_N
        a_padded = np.zeros((MAX_M, MAX_N), dtype=torch_dtype)
        b_padded = np.zeros((MAX_M, MAX_N), dtype=torch_dtype)
        c_padded = np.zeros((MAX_M, MAX_N), dtype=torch_dtype)
        
        a_padded[:M, :N] = a_host
        b_padded[:M, :N] = b_host
        
        # Device tensors (full MAX size)
        t_dtype = torch.float32 if dtype == F32Type else torch.float16
        A = torch.tensor(a_padded, device="cuda", dtype=t_dtype)
        B = torch.tensor(b_padded, device="cuda", dtype=t_dtype)
        C = torch.empty((MAX_M, MAX_N), device="cuda", dtype=t_dtype)
    else:
        t_dtype = torch.float32 if dtype == F32Type else torch.float16
        A = torch.tensor(a_host, device="cuda", dtype=t_dtype)
        B = torch.tensor(b_host, device="cuda", dtype=t_dtype)
        C = torch.empty((M, N), device="cuda", dtype=t_dtype)
    
    # Launch configuration (must match kernel tiling)
    # - blockDim should provide exactly THR_M*THR_N threads (tidx in [0, THR_M*THR_N))
    # - gridDim should be computed in units of the kernel's tile coverage
    BLOCK_X, BLOCK_Y = THR_N, THR_M
    TILE_M = THR_M * VAL_M
    TILE_N = THR_N * VAL_N
    grid_x = (N + TILE_N - 1) // TILE_N
    grid_y = (M + TILE_M - 1) // TILE_M
    
    print(f"\n[FLIR INFO] Launch configuration:")
    kernel_tag = f"{compile_M}x{compile_N}"
    print(f"  Grid: ({grid_x}, {grid_y}, 1) [Tiles: {TILE_M}x{TILE_N}, kernel={kernel_tag}]")
    print(f"  Block: ({BLOCK_X}, {BLOCK_Y}, 1)")
    
    # Launch kernel via executor
    exe(A, B, C)
    torch.cuda.synchronize()
    
    if use_shared_max_kernel:
        c_host = C[:M, :N].cpu().numpy()
    else:
        c_host = C.cpu().numpy()
    
    # Verify results
    error = np.max(np.abs(c_host - expected))
    
    print(f"\n[FLIR INFO] Verification:")
    print(f"  Max error: {error:.2e}")
    
    # Benchmark if requested
    if benchmark:
        # Use the shared perf harness (same as vecAdd benchmark) so results are comparable.
        # NOTE: run_perftest uses torch.profiler on ROCm; it expects torch to be installed.
        print(f"\n[FLIR INFO] Running benchmark via run_perftest ({iterations} iterations)...")

        def kernel_launch():
            exe(A, B, C)

        # run_perftest returns (data, avg_us)
        _, avg_us = run_perftest(kernel_launch, num_iters=iterations, num_warmup=10)
        torch.cuda.synchronize()

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
    
    assert error < 1e-4


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Elementwise add example using FLIR"
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
