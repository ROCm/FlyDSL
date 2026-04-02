#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Vector Addition Benchmark - GPU kernel with flydsl API"""

import sys
import os
import numpy as np
import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU benchmarks.", allow_module_level=True)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch
from tests.test_common import checkAllclose, run_perftest


@flyc.kernel
def vecAddKernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x

    tile_elems = block_dim * vec_width

    tA = fx.logical_divide(A, fx.make_layout(tile_elems, 1))
    tB = fx.logical_divide(B, fx.make_layout(tile_elems, 1))
    tC = fx.logical_divide(C, fx.make_layout(tile_elems, 1))

    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))
    tC = fx.slice(tC, (None, bid))

    tA = fx.logical_divide(tA, fx.make_layout(vec_width, 1))
    tB = fx.logical_divide(tB, fx.make_layout(vec_width, 1))
    tC = fx.logical_divide(tC, fx.make_layout(vec_width, 1))

    copy_bits = vec_width * 32
    RABMemRefTy = fx.MemRefType.get(
        fx.T.f32(), fx.LayoutType.get(vec_width, 1), fx.AddressSpace.Register
    )
    copyAtom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), fx.Float32)

    rA = fx.memref_alloca(RABMemRefTy, fx.make_layout(vec_width, 1))
    rB = fx.memref_alloca(RABMemRefTy, fx.make_layout(vec_width, 1))
    rC = fx.memref_alloca(RABMemRefTy, fx.make_layout(vec_width, 1))

    fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)

    vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
    fx.memref_store_vec(vC, rC)

    fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))


@flyc.jit
def vecAdd(
    A: fx.Tensor,
    B: fx.Tensor,
    C,
    n: fx.Int32,
    const_n: fx.Constexpr[int],
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    tile_elems = block_dim * vec_width
    grid_x = (n + tile_elems - 1) // tile_elems
    vecAddKernel(A, B, C, block_dim, vec_width).launch(
        grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream
    )


def benchmark_pytorch_add(size: int):
    """Measure torch.add performance for the same problem size."""
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

    return {
        "avg_ms": avg_ms,
        "avg_us": avg_us,
        "bandwidth_gbs": bandwidth_gbs,
        "size": size,
        "total_bytes": total_bytes,
    }


def benchmark_vector_add(vec_width: int = 4):
    """Benchmark vector addition kernel performance."""

    THREADS_PER_BLOCK = 256
    VEC_WIDTH = vec_width
    TILE_ELEMS = THREADS_PER_BLOCK * VEC_WIDTH
    SIZE = TILE_ELEMS * 10000  # align to tile boundary

    print("\n" + "=" * 80)
    print("Benchmark: Vector Addition (C = A + B) - flydsl API")
    print(f"  - Threads per block: {THREADS_PER_BLOCK}")
    print(f"  - Vec width: {VEC_WIDTH} floats ({VEC_WIDTH * 32} bits)")
    print(f"  - Tile elems: {TILE_ELEMS}")
    print(f"Size: {SIZE} elements ({SIZE / 1e6:.1f}M floats, ~{SIZE * 4 / 1e9:.2f} GB)")
    print(f"Memory Traffic: 3 x {SIZE} x 4 bytes = {3 * SIZE * 4 / 1e9:.2f} GB per kernel")
    print("=" * 80)

    a_dev = torch.randn(SIZE, device="cuda", dtype=torch.float32)
    b_dev = torch.randn(SIZE, device="cuda", dtype=torch.float32)
    c_dev = torch.empty_like(a_dev)

    stream = torch.cuda.Stream()

    tA = flyc.from_dlpack(a_dev).mark_layout_dynamic(leading_dim=0, divisibility=VEC_WIDTH)

    vecAdd(tA, b_dev, c_dev, SIZE, SIZE, THREADS_PER_BLOCK, VEC_WIDTH, stream=stream)
    torch.cuda.synchronize()

    error = checkAllclose(c_dev, a_dev + b_dev)
    print(f"  Correctness: max error = {error:.2e}")

    def kernel_launch():
        vecAdd(tA, b_dev, c_dev, SIZE, SIZE, THREADS_PER_BLOCK, VEC_WIDTH, stream=stream)
        torch.cuda.synchronize()

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

    print(f"\n  FlyDSL kernel: {avg_ms:.4f} ms, Bandwidth: {bandwidth_gbs:.2f} GB/s")

    torch_results = benchmark_pytorch_add(SIZE)
    if torch_results:
        bw_ratio = results["bandwidth_gbs"] / torch_results["bandwidth_gbs"]
        print(f"  PyTorch BW: {torch_results['bandwidth_gbs']:.2f} GB/s")
        print(f"  Bandwidth ratio (FlyDSL / PyTorch): {bw_ratio:.2f}x")

    return error < 1e-5


def test_benchmark_vector_add():
    """Pytest wrapper for vector addition benchmark."""
    print("\n" + "=" * 80)
    print("ROCm GPU Benchmark - Vector Addition with flydsl API")
    print(f"GPU: {get_rocm_arch()}")
    print("=" * 80)
    assert benchmark_vector_add(), "Vector addition benchmark failed correctness check"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vector Addition Benchmark")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument(
        "--vec-width", type=int, default=4, help="Vector width (default: 4)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("ROCm GPU Benchmark - Vector Addition with flydsl API")
    print(f"GPU: {get_rocm_arch()}")
    print("=" * 80)

    result = benchmark_vector_add(vec_width=args.vec_width)

    print("\n" + "=" * 80)
    if result:
        print("BENCHMARK COMPLETED SUCCESSFULLY")
        sys.exit(0)
    else:
        print("[ERROR] BENCHMARK FAILED CORRECTNESS CHECK")
        sys.exit(1)
