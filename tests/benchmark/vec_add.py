#!/usr/bin/env python3
"""Vector Addition Benchmark - GPU kernel with Flir Layout integration"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pyflir
from pyflir.compiler.pipeline import Pipeline, run_pipeline
from pyflir.dialects.ext import flir
from pyflir.dialects.ext import arith
from pyflir.runtime.device import get_rocm_arch
from _mlir.ir import F32Type, IntegerType
import _mlir.extras.types as T
import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU benchmarks.", allow_module_level=True)

from tests.test_common import run_perftest


def create_vec_add_kernel(
    size: int,
    tile_size: int = 8,
    dtype=F32Type,
    vec_width: int = 4,
    threads_per_block: int = 256,
):
    """Create a RocIR tiled copy kernel mirroring 1-D vec_add behavior."""
    if tile_size % vec_width != 0:
        raise ValueError("tile_size must be divisible by vec_width")

    TILE_SIZE = tile_size
    THREADS_PER_BLOCK = threads_per_block
    VEC_WIDTH = vec_width
    TILE_ELEMS = THREADS_PER_BLOCK * TILE_SIZE
    ITERS_PER_THREAD = TILE_SIZE // VEC_WIDTH

    gpu_arch = get_rocm_arch()
    num_blocks = (size + TILE_ELEMS - 1) // TILE_ELEMS

    class _VecAdd(flir.MlirModule):
        GPU_MODULE_NAME = "vec_kernels"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        @flir.kernel
        def vec_add(
            self: flir.T.i64,
            A: lambda: T.memref(size, dtype.get()),
            B: lambda: T.memref(size, dtype.get()),
            C: lambda: T.memref(size, dtype.get()),
        ):
            tid_x = flir.thread_idx("x")
            tid_y = flir.thread_idx("y")
            bid_x = flir.block_idx("x")
            bdim_x = flir.block_dim("x")
            tid_linear = (tid_y * bdim_x + tid_x).value

            thr_layout = flir.make_ordered_layout((THREADS_PER_BLOCK,), order=(0,))
            val_layout = flir.make_ordered_layout((TILE_SIZE,), order=(0,))

            copy_atom_load = flir.make_copy_atom(dtype.get(), vector_size=VEC_WIDTH)
            copy_atom_store = flir.make_copy_atom(dtype.get(), vector_size=VEC_WIDTH)

            tiled_copy_A = flir.make_tiled_copy_tv(
                copy_atom_load,
                thr_layout,
                val_layout,
                thr_shape=(THREADS_PER_BLOCK,),
                val_shape=(TILE_SIZE,),
            )
            tiled_copy_B = flir.make_tiled_copy_tv(
                copy_atom_load,
                thr_layout,
                val_layout,
                thr_shape=(THREADS_PER_BLOCK,),
                val_shape=(TILE_SIZE,),
            )
            tiled_copy_C = flir.make_tiled_copy_tv(
                copy_atom_store,
                thr_layout,
                val_layout,
                thr_shape=(THREADS_PER_BLOCK,),
                val_shape=(TILE_SIZE,),
            )

            tensor_A = flir.make_tensor(A, shape=(size,), strides=(1,))
            tensor_B = flir.make_tensor(B, shape=(size,), strides=(1,))
            tensor_C = flir.make_tensor(C, shape=(size,), strides=(1,))

            tile_shape = (TILE_ELEMS,)

            gA = flir.zipped_divide(tensor_A, tile_shape)
            gB = flir.zipped_divide(tensor_B, tile_shape)
            gC = flir.zipped_divide(tensor_C, tile_shape)

            idC = flir.make_identity_tensor((size,))
            cC = flir.zipped_divide(idC, tile_shape)

            blk_coord = (bid_x,)
            blkA = gA[blk_coord]
            blkB = gB[blk_coord]
            blkC = gC[blk_coord]
            blkCrd = cC[blk_coord]

            thr_copy_A = tiled_copy_A.get_slice(tid_linear)
            thr_copy_B = tiled_copy_B.get_slice(tid_linear)
            thr_copy_C = tiled_copy_C.get_slice(tid_linear)

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
            total_vals = val_shape[0]

            for linear in range(total_vals):
                lin_idx = flir.const_index(linear)
                coords = thrCrd.coords_from_linear(lin_idx)
                pred_val = flir.elem_less(coords, (size,))
                pred_offsets = tuple(frgPred.offsets_from_linear(lin_idx))
                frgPred[pred_offsets] = pred_val

            flir.copy(tiled_copy_A, thrA, frgA, pred=frgPred)
            flir.copy(tiled_copy_B, thrB, frgB, pred=frgPred)

            for iter_idx in range(ITERS_PER_THREAD):
                iter_base = iter_idx * VEC_WIDTH
                for lane in range(VEC_WIDTH):
                    lin = iter_base + lane
                    idx = flir.const_index(lin)
                    coords = (idx,)
                    a_val = frgA[coords]
                    b_val = frgB[coords]
                    c_val = a_val + b_val
                    frgC[coords] = c_val

            flir.copy(tiled_copy_C, frgC, thrC, pred=frgPred)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A: lambda: T.memref(size, dtype.get()),
            B: lambda: T.memref(size, dtype.get()),
            C: lambda: T.memref(size, dtype.get()),
        ):
            c1 = arith.index(1).value
            gx = arith.index(num_blocks).value
            bx = arith.index(THREADS_PER_BLOCK).value
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "vec_add"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[A, B, C],
            )

    return _VecAdd().module


def benchmark_pytorch_add(size: int):
    """Measure torch.add performance for the same problem size."""
    if torch is None:
        print("\nPyTorch not installed; skipping torch.add baseline.")
        return None
    if not torch.cuda.is_available():
        print("\nPyTorch CUDA backend unavailable; skipping torch.add baseline.")
        return None

    device = torch.device("cuda")
    dtype = torch.float32
    a = torch.randn(size, dtype=dtype, device=device)
    b = torch.randn(size, dtype=dtype, device=device)
    c = torch.empty_like(a)

    def torch_launch():
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.add(a, b, out=c)
        stop.record()
        torch.cuda.synchronize()
        return start.elapsed_time(stop)

    _, avg_us = run_perftest(torch_launch, num_iters=20, num_warmup=2)
    
    total_bytes = 3 * size * a.element_size()
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9
    avg_ms = avg_us / 1000
    
    results = {
        "avg_ms": avg_ms,
        "avg_us": avg_us,
        "bandwidth_gbs": bandwidth_gbs,
        "size": size,
        "total_bytes": total_bytes,
    }
    
    return results
    
def benchmark_vector_add(tile_size: int = 4):
    """Benchmark vector addition kernel performance."""
    
    # Configuration parameters - change these to experiment
    SIZE = 20480000
    TILE_SIZE = tile_size  # Each thread processes TILE_SIZE elements
    VEC_WIDTH = 4   # Vector width for vectorized loads/stores (must divide TILE_SIZE evenly)
    THREADS_PER_BLOCK = 256
    ITERS_PER_THREAD = TILE_SIZE // VEC_WIDTH  # Number of vectorized iterations per thread
    TILE_ELEMS = THREADS_PER_BLOCK * TILE_SIZE
    
    print("\n" + "="*80)
    print("Benchmark: Vector Addition Performance (C = A + B) - Optimized")
    print("Optimization: Continuous Thread Indexing + Tiled SIMD Vectorization")
    print(f"  - Threads work continuously with VEC_WIDTH ({VEC_WIDTH} floats)")
    print(f"  - Outer loop handles TILE_SIZE ({TILE_SIZE} elements = {ITERS_PER_THREAD} iterations per thread)")
    print("  - Each iteration: SIMD vector.load/store operations")
    print(f"Size: {SIZE} elements ({SIZE/1e6:.1f}M floats, ~{SIZE*4/1e9:.2f} GB)")
    print(f"Memory Traffic: 3 × {SIZE} × 4 bytes = {3*SIZE*4/1e9:.2f} GB per kernel")
    print("="*80)
    
    module = create_vec_add_kernel(SIZE, tile_size=TILE_SIZE, dtype=F32Type)
    print("  Running canonicalize + CSE pipeline...")
    optimized = run_pipeline(module, Pipeline().canonicalize().cse())
    exe = pyflir.compile(optimized)
    
    threads_per_block = THREADS_PER_BLOCK
    num_blocks = (SIZE + TILE_ELEMS - 1) // TILE_ELEMS
    
    total_threads_needed = (SIZE + TILE_SIZE - 1) // TILE_SIZE

    print(f"  Kernel Configuration:")
    print(f"    - Tile Size: {TILE_SIZE} elements per thread")
    print(f"    - SIMD Vector Width: {VEC_WIDTH} floats (using vector.load/store)")
    print(f"    - Iterations per thread: {ITERS_PER_THREAD} (TILE_SIZE / VEC_WIDTH)")
    print(f"    - Memory access pattern: Continuous threads with vec_width stride")
    print(f"    - Total threads needed: {total_threads_needed:,}")
    print(f"    - Blocks: {num_blocks:,} x Threads/Block: {threads_per_block}")
    
    # Allocate device memory
    a_host = np.random.randn(SIZE).astype(np.float32)
    b_host = np.random.randn(SIZE).astype(np.float32)
    a_dev = torch.tensor(a_host, device="cuda", dtype=torch.float32)
    b_dev = torch.tensor(b_host, device="cuda", dtype=torch.float32)
    c_dev = torch.empty_like(a_dev)

    def kernel_launch():
        exe(a_dev, b_dev, c_dev)
        torch.cuda.synchronize()

    # Run benchmark
    _, avg_us = run_perftest(kernel_launch, num_iters=20, num_warmup=2)
    
    total_bytes = 3 * SIZE * 4
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9
    avg_ms = avg_us / 1000
    
    results = {
        "avg_ms": avg_ms,
        "avg_us": avg_us,
        "bandwidth_gbs": bandwidth_gbs,
        "size": SIZE,
        "total_bytes": total_bytes,
    }
    
    # Verify correctness
    c_host = c_dev.cpu().numpy()
    expected = a_host + b_host
    error = np.max(np.abs(c_host - expected))
    
    print(f"\n  Correctness Check:")
    print(f"  Max error: {error:.2e}")
    
    # Print a standardized bandwidth line so run_tests.sh can pick it up (like matrixTranspose).
    print(f"\nBandwidth: {bandwidth_gbs:.2f} GB/s")

    # Print benchmark results
    print(f"\n{results}")

    torch_results = benchmark_pytorch_add(SIZE)
    if torch_results:
        print("\nPyTorch torch.add baseline:")
        print(torch_results)
        bw_ratio = results["bandwidth_gbs"] / torch_results["bandwidth_gbs"]
        # Avoid "Bandwidth:" here so run_tests.sh doesn't accidentally pick the baseline.
        print(f"  PyTorch BW: {torch_results['bandwidth_gbs']:.2f} GB/s")
        print(f"  Bandwidth ratio (FLIR / PyTorch): {bw_ratio:.2f}x")
    
    return error < 1e-5

# Pytest test function
def test_benchmark_vector_add():
    """Pytest wrapper for vector addition benchmark."""
    print("\n" + "="*80)
    print("ROCm GPU Benchmark - Vector Addition with Flir Layout")
    print(f"GPU: {get_rocm_arch()}")
    print("="*80)
    assert benchmark_vector_add(), "Vector addition benchmark failed correctness check"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Vector Addition Benchmark')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run performance benchmark')
    parser.add_argument('--tile', type=int, default=4,
                       help='Elements handled per thread (default: 8)')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ROCm GPU Benchmark - Vector Addition with Flir Layout")
    print(f"GPU: {get_rocm_arch()}")
    print("="*80)
    
    result = benchmark_vector_add(tile_size=args.tile)
    
    print("\n" + "="*80)
    if result:
        print("✓ BENCHMARK COMPLETED SUCCESSFULLY")
        sys.exit(0)
    else:
        print("[ERROR]BENCHMARK FAILED CORRECTNESS CHECK")
        sys.exit(1)

