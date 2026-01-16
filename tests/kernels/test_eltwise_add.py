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
import flydsl

# Setup paths


from flydsl.dialects.ext import flir
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.dialects.ext import arith
from flydsl.dialects.ext.arith import Index
from flydsl.runtime.device import get_rocm_arch
from _mlir import ir
from _mlir.ir import F16Type, F32Type, IntegerType
from _mlir.dialects import arith
from flydsl.dialects.ext import arith as arith_ext
import _mlir.extras.types as T
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)
from tests.test_common import run_perftest


THR_M, THR_N = 4, 32
VAL_M, VAL_N = 4, 4
COPY_VEC = 8

def create_elementwise_add_kernel(N: int, dtype=F32Type):
    """Create elementwise addition kernel demonstrating FLIR API.
    
    Args:
        N: Compile-time constant leading dimension (row stride)
        dtype: Element type
        
    Returns:
        Compiled kernel module
    """
    print(f"\n[FLIR INFO] Creating elementwise add kernel (dynamic M, static N={N})")
    print(f"[FLIR INFO] Element type: {dtype}")

    # NOTE: Kernel operands in the lowered module use dynamic memref types.
    # Keep the host stub signature dynamic too so gpu.launch_func types match.
    S = ir.ShapedType.get_dynamic_size()
    
    class _EltwiseAdd(flir.MlirModule):
        GPU_MODULE_NAME = "elementwise_kernels"
        GPU_MODULE_TARGETS = ['#rocdl.target<abi = "500">']

        @flir.kernel
        def elementwise_add_kernel(
            self: flir.T.i64,
            A: lambda: T.memref(S, S, dtype.get()),
            B: lambda: T.memref(S, S, dtype.get()),
            C: lambda: T.memref(S, S, dtype.get()),
            m_in: lambda: T.index(),
        ):
            # ===== Step 1: Thread and Block IDs =====
            tid_x = flir.thread_idx("x")
            tid_y = flir.thread_idx("y")
            bid_x = flir.block_idx("x")
            bid_y = flir.block_idx("y")

            # Calculate linear thread index
            bdim_x = flir.block_dim("x")
            tidx = tid_y * bdim_x + tid_x

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

            # Strides must be compile-time constants because downstream helpers like
            # `make_fragment_like(TensorView)` require statically-known template.strides.
            # This means the kernel is specialized on the leading dimension `N`.
            tensor_A = flir.make_tensor(A, shape=(m_in, N), strides=(N, 1))
            tensor_B = flir.make_tensor(B, shape=(m_in, N), strides=(N, 1))
            tensor_C = flir.make_tensor(C, shape=(m_in, N), strides=(N, 1))

            TILE_M = THR_M * VAL_M
            TILE_N = THR_N * VAL_N
            tile_shape = (TILE_M, TILE_N)
            gA = flir.zipped_divide(tensor_A, tile_shape)
            gB = flir.zipped_divide(tensor_B, tile_shape)
            gC = flir.zipped_divide(tensor_C, tile_shape)
            idC = flir.make_identity_tensor((m_in, N))
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
                pred_val = flir.elem_less(coords, (m_in, N))
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
            A: lambda: T.memref(S, S, dtype.get()),
            B: lambda: T.memref(S, S, dtype.get()),
            C: lambda: T.memref(S, S, dtype.get()),
            m_in: lambda: T.index(),
        ):
            c1 = Index(1)
            tile_m = THR_M * VAL_M
            tile_n = THR_N * VAL_N
            gx = (Index(N) + Index(tile_n) - c1) // Index(tile_n)
            gy = (m_in + Index(tile_m) - c1) // Index(tile_m)
            bdx = Index(THR_N)
            bdy = Index(THR_M)
            flir.gpu_ext.LaunchFuncOp(
                ["elementwise_kernels", "elementwise_add_kernel"],
                grid_size=(gx, gy, c1),
                block_size=(bdx, bdy, c1),
                kernel_operands=[A, B, C, m_in],
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
    
    print(f"\n[FLIR INFO] Compiling kernel for dynamic M, static N={N} (may hit compile cache)...")
    m = create_elementwise_add_kernel(N, dtype)
    exe = flydsl.compile(m)
    
    # Prepare data
    np.random.seed(42)
    torch_dtype = np.float32 if dtype == F32Type else np.float16
    a_host = np.random.randn(M, N).astype(torch_dtype)
    b_host = np.random.randn(M, N).astype(torch_dtype)
    expected = a_host + b_host
    
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
    kernel_tag = f"{M}x{N}"
    print(f"  Grid: ({grid_x}, {grid_y}, 1) [Tiles: {TILE_M}x{TILE_N}, kernel={kernel_tag}]")
    print(f"  Block: ({BLOCK_X}, {BLOCK_Y}, 1)")
    
    # Launch kernel via executor
    exe(A, B, C, M)
    torch.cuda.synchronize()
    
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
            exe(A, B, C, M, N)

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
        # Run all shapes to demonstrate compile caching behavior (if enabled).
        print("\n" + "="*80)
        print("Running all test shapes (demonstrates compile caching)")
        print("="*80)
        for M, N in TEST_SHAPES:
            print(f"\n--- Testing shape {M}x{N} (1st run) ---")
            test_compile_and_run(M, N, dtype=F32Type, benchmark=False)
            
            # Run again: flydsl.compile may hit its cache and avoid redoing heavy work.
            print(f"\n--- Testing shape {M}x{N} (2nd run - should hit compile cache) ---")
            test_compile_and_run(M, N, dtype=F32Type, benchmark=False)
        
        print("\nPASS - All shapes tested with compile caching!")
    else:
        test_compile_and_run(
            args.M, args.N,
            dtype=F32Type,
            benchmark=args.benchmark,
            iterations=args.iterations
        )
        
        print("PASS - Elementwise add test completed successfully!")
