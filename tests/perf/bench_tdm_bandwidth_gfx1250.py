#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""TDM load bandwidth benchmark for gfx1250.

Three modes:
  --mode read-only : TDM loads only (no store). Measures peak TDM HBM read bandwidth.
  --mode unique    : Each WG reads unique tiles + add + store. Measures raw HBM R/W bandwidth.
  --mode multicast : GEMM-like shared tiles with cluster multicast. Measures
                     multicast L2→LDS throughput amplification.

Usage:
    python tests/perf/bench_tdm_bandwidth_gfx1250.py                      # run all modes
    python tests/perf/bench_tdm_bandwidth_gfx1250.py --mode read-only     # peak read BW only
    python tests/perf/bench_tdm_bandwidth_gfx1250.py --mode unique        # R/W BW only
    python tests/perf/bench_tdm_bandwidth_gfx1250.py --mode multicast     # multicast only
"""

import argparse
import sys

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, tdm_ops, vector
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, get_op_result_or_value

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TILE = 128
BLOCK_DIM = 256
WAVE_SIZE = 32
NUM_WARPS = BLOCK_DIM // WAVE_SIZE
VEC_WIDTH = 8
ELEM_BYTES = 2  # bf16
TILE_BYTES = TILE * TILE * ELEM_BYTES  # 32 KB
ELEMS_PER_THREAD = (TILE * TILE) // BLOCK_DIM  # 64
VECS_PER_THREAD = ELEMS_PER_THREAD // VEC_WIDTH  # 8


# ---------------------------------------------------------------------------
# Kernel: unique tiles per WG (mode=unique)
# ---------------------------------------------------------------------------


def _compile_tdm_unique_add(grid_m, grid_n):
    """Each WG reads unique A/B tiles — no L2 reuse, all reads hit HBM."""
    lds_a_offset = 0
    lds_b_offset = ((TILE_BYTES + 127) // 128) * 128
    total_lds = lds_b_offset + TILE_BYTES

    smem_alloc = SmemAllocator(None, arch="gfx1250", global_sym_name="tdm_unique_add_smem")
    smem_alloc.ptr = total_lds

    tile_elems = TILE * TILE

    @flyc.kernel
    def tdm_unique_add_kernel(arg_a: fx.Tensor, arg_b: fx.Tensor, arg_c: fx.Tensor):
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        tid = gpu.thread_id("x")

        bid_idx = bx * arith.index(grid_n) + by
        blk_row = bid_idx * arith.index(TILE)
        k_base = arith.index(0)

        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            smem_alloc.finalized = False
            smem_alloc.finalize()
        smem_base = smem_alloc.get_base()
        bf16_ty = T.bf16
        smem_a = SmemPtr(smem_base, lds_a_offset, bf16_ty, shape=(TILE * TILE,))
        smem_b = SmemPtr(smem_base, lds_b_offset, bf16_ty, shape=(TILE * TILE,))
        lds_a_memref = get_op_result_or_value(smem_a.get())
        lds_b_memref = get_op_result_or_value(smem_b.get())

        # A, B: (total_wgs * TILE, TILE), each WG reads unique rows
        desc_a = tdm_ops.make_tensor_descriptor_2d(
            global_ptr=arg_a,
            lds_memref=lds_a_memref,
            global_offset=(blk_row, k_base),
            tensor_shape=(TILE, TILE),
            strides=(TILE, 1),
            tile_shape=(TILE, TILE),
            elem_bytes=ELEM_BYTES,
            num_warps=NUM_WARPS,
            workgroup_mask=0,
        )
        desc_b = tdm_ops.make_tensor_descriptor_2d(
            global_ptr=arg_b,
            lds_memref=lds_b_memref,
            global_offset=(blk_row, k_base),
            tensor_shape=(TILE, TILE),
            strides=(TILE, 1),
            tile_shape=(TILE, TILE),
            elem_bytes=ELEM_BYTES,
            num_warps=NUM_WARPS,
            workgroup_mask=0,
        )

        tdm_ops.tensor_load_2d(desc_a)
        tdm_ops.tensor_load_2d(desc_b)
        tdm_ops.tensor_wait(0)
        gpu.barrier()

        bid = fx.block_idx.x * grid_n + fx.block_idx.y
        C = fx.rocdl.make_buffer_tensor(arg_c)
        tC = fx.logical_divide(C, fx.make_layout(tile_elems, 1))
        tC = fx.slice(tC, (None, bid))
        tC = fx.logical_divide(tC, fx.make_layout(VEC_WIDTH, 1))
        copyAtom = fx.make_copy_atom(fx.rocdl.BufferCopy(VEC_WIDTH * fx.BFloat16.width), fx.BFloat16)
        rC = fx.make_rmem_tensor(VEC_WIDTH, fx.BFloat16)
        vec_ty = T.vec(VEC_WIDTH, bf16_ty)
        base_elem = tid * arith.index(ELEMS_PER_THREAD)
        for v in range_constexpr(VECS_PER_THREAD):
            elem_off = base_elem + arith.index(v * VEC_WIDTH)
            va = vector.load_op(vec_ty, lds_a_memref, [elem_off])
            vb = vector.load_op(vec_ty, lds_b_memref, [elem_off])
            vc = arith.addf(va, vb)
            fx.memref_store_vec(vc, rC)
            store_idx = fx.thread_idx.x * VECS_PER_THREAD + v
            fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, store_idx)))

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        L = tdm_unique_add_kernel(A, B, C)
        L.launch(grid=(grid_m, grid_n, 1), block=(BLOCK_DIM, 1, 1), stream=stream)

    return launch


# ---------------------------------------------------------------------------
# Kernel: read-only TDM loads (mode=read-only)
# ---------------------------------------------------------------------------


def _compile_tdm_read_only(grid_m, grid_n):
    """Each WG does TDM load of 2 tiles to LDS, no store. Measures pure TDM read BW."""
    lds_a_offset = 0
    lds_b_offset = ((TILE_BYTES + 127) // 128) * 128
    total_lds = lds_b_offset + TILE_BYTES

    smem_alloc = SmemAllocator(None, arch="gfx1250", global_sym_name="tdm_read_only_smem")
    smem_alloc.ptr = total_lds

    @flyc.kernel
    def tdm_read_only_kernel(arg_a: fx.Tensor, arg_b: fx.Tensor, arg_c: fx.Tensor):
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        _tid = gpu.thread_id("x")

        bid_idx = bx * arith.index(grid_n) + by
        blk_row = bid_idx * arith.index(TILE)
        k_base = arith.index(0)

        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            smem_alloc.finalized = False
            smem_alloc.finalize()
        smem_base = smem_alloc.get_base()
        bf16_ty = T.bf16
        smem_a = SmemPtr(smem_base, lds_a_offset, bf16_ty, shape=(TILE * TILE,))
        smem_b = SmemPtr(smem_base, lds_b_offset, bf16_ty, shape=(TILE * TILE,))
        lds_a_memref = get_op_result_or_value(smem_a.get())
        lds_b_memref = get_op_result_or_value(smem_b.get())

        desc_a = tdm_ops.make_tensor_descriptor_2d(
            global_ptr=arg_a,
            lds_memref=lds_a_memref,
            global_offset=(blk_row, k_base),
            tensor_shape=(TILE, TILE),
            strides=(TILE, 1),
            tile_shape=(TILE, TILE),
            elem_bytes=ELEM_BYTES,
            num_warps=NUM_WARPS,
            workgroup_mask=0,
        )
        desc_b = tdm_ops.make_tensor_descriptor_2d(
            global_ptr=arg_b,
            lds_memref=lds_b_memref,
            global_offset=(blk_row, k_base),
            tensor_shape=(TILE, TILE),
            strides=(TILE, 1),
            tile_shape=(TILE, TILE),
            elem_bytes=ELEM_BYTES,
            num_warps=NUM_WARPS,
            workgroup_mask=0,
        )

        tdm_ops.tensor_load_2d(desc_a)
        tdm_ops.tensor_load_2d(desc_b)
        tdm_ops.tensor_wait(0)
        gpu.barrier()

        # Every thread reads a vector from each LDS tile, adds, stores to C[bid]
        # to prevent dead code elimination of the TDM loads.
        vec_ty = T.vec(VEC_WIDTH, bf16_ty)
        va = vector.load_op(vec_ty, lds_a_memref, [arith.index(0)])
        vb = vector.load_op(vec_ty, lds_b_memref, [arith.index(0)])
        vc = arith.addf(va, vb)
        bid = fx.block_idx.x * grid_n + fx.block_idx.y
        C = fx.rocdl.make_buffer_tensor(arg_c)
        tC = fx.logical_divide(C, fx.make_layout(VEC_WIDTH, 1))
        copyAtom = fx.make_copy_atom(fx.rocdl.BufferCopy(VEC_WIDTH * fx.BFloat16.width), fx.BFloat16)
        rC = fx.make_rmem_tensor(VEC_WIDTH, fx.BFloat16)
        fx.memref_store_vec(vc, rC)
        fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, bid)))

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        L = tdm_read_only_kernel(A, B, C)
        L.launch(grid=(grid_m, grid_n, 1), block=(BLOCK_DIM, 1, 1), stream=stream)

    return launch


# ---------------------------------------------------------------------------
# Kernel: shared tiles with multicast (mode=multicast)
# ---------------------------------------------------------------------------


def _compile_tdm_shared_add(grid_m, grid_n, cluster_m, cluster_n):
    """GEMM-like tiling: A shared by row, B shared by col, with cluster multicast."""
    use_cluster = cluster_m > 1 or cluster_n > 1

    lds_a_offset = 0
    lds_b_offset = ((TILE_BYTES + 127) // 128) * 128
    total_lds = lds_b_offset + TILE_BYTES

    smem_alloc = SmemAllocator(None, arch="gfx1250", global_sym_name="tdm_shared_add_smem")
    smem_alloc.ptr = total_lds

    stride_a = TILE
    stride_b = grid_n * TILE
    tile_elems = TILE * TILE

    @flyc.kernel
    def tdm_shared_add_kernel(arg_a: fx.Tensor, arg_b: fx.Tensor, arg_c: fx.Tensor):
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        tid = gpu.thread_id("x")

        if const_expr(use_cluster):
            local_x, local_y = cluster.compute_cluster_position()
            a_mcast_mask, b_mcast_mask = cluster.compute_mcast_masks(local_x, local_y, cluster_m, cluster_n)
        else:
            a_mcast_mask = 0
            b_mcast_mask = 0

        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            smem_alloc.finalized = False
            smem_alloc.finalize()
        smem_base = smem_alloc.get_base()
        bf16_ty = T.bf16
        smem_a = SmemPtr(smem_base, lds_a_offset, bf16_ty, shape=(TILE * TILE,))
        smem_b = SmemPtr(smem_base, lds_b_offset, bf16_ty, shape=(TILE * TILE,))
        lds_a_memref = get_op_result_or_value(smem_a.get())
        lds_b_memref = get_op_result_or_value(smem_b.get())

        blk_m = bx * arith.index(TILE)
        blk_n = by * arith.index(TILE)
        k_base = arith.index(0)

        desc_a = tdm_ops.make_tensor_descriptor_2d(
            global_ptr=arg_a,
            lds_memref=lds_a_memref,
            global_offset=(blk_m, k_base),
            tensor_shape=(TILE, TILE),
            strides=(stride_a, 1),
            tile_shape=(TILE, TILE),
            elem_bytes=ELEM_BYTES,
            num_warps=NUM_WARPS,
            workgroup_mask=a_mcast_mask,
        )
        desc_b = tdm_ops.make_tensor_descriptor_2d(
            global_ptr=arg_b,
            lds_memref=lds_b_memref,
            global_offset=(k_base, blk_n),
            tensor_shape=(TILE, TILE),
            strides=(stride_b, 1),
            tile_shape=(TILE, TILE),
            elem_bytes=ELEM_BYTES,
            num_warps=NUM_WARPS,
            workgroup_mask=b_mcast_mask,
        )

        tdm_ops.tensor_load_2d(desc_a)
        tdm_ops.tensor_load_2d(desc_b)
        tdm_ops.tensor_wait(0)
        if const_expr(use_cluster):
            cluster.cluster_barrier()
        else:
            gpu.barrier()

        bid = fx.block_idx.x * grid_n + fx.block_idx.y
        C = fx.rocdl.make_buffer_tensor(arg_c)
        tC = fx.logical_divide(C, fx.make_layout(tile_elems, 1))
        tC = fx.slice(tC, (None, bid))
        tC = fx.logical_divide(tC, fx.make_layout(VEC_WIDTH, 1))
        copyAtom = fx.make_copy_atom(fx.rocdl.BufferCopy(VEC_WIDTH * fx.BFloat16.width), fx.BFloat16)
        rC = fx.make_rmem_tensor(VEC_WIDTH, fx.BFloat16)
        vec_ty = T.vec(VEC_WIDTH, bf16_ty)
        base_elem = tid * arith.index(ELEMS_PER_THREAD)
        for v in range_constexpr(VECS_PER_THREAD):
            elem_off = base_elem + arith.index(v * VEC_WIDTH)
            va = vector.load_op(vec_ty, lds_a_memref, [elem_off])
            vb = vector.load_op(vec_ty, lds_b_memref, [elem_off])
            vc = arith.addf(va, vb)
            fx.memref_store_vec(vc, rC)
            store_idx = fx.thread_idx.x * VECS_PER_THREAD + v
            fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, store_idx)))

    cluster_dims_str = f"{cluster_m},{cluster_n},1"

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        L = tdm_shared_add_kernel(
            A,
            B,
            C,
            value_attrs={"rocdl.cluster_dims": cluster_dims_str} if use_cluster else {},
        )
        if use_cluster:
            L.launch(
                grid=(grid_m, grid_n, 1),
                block=(BLOCK_DIM, 1, 1),
                stream=stream,
                cluster=(cluster_m, cluster_n, 1),
            )
        else:
            L.launch(grid=(grid_m, grid_n, 1), block=(BLOCK_DIM, 1, 1), stream=stream)

    return launch


# ---------------------------------------------------------------------------
# Benchmark timing (adapted from benchmark_common.py:bench_kernel_us)
# ---------------------------------------------------------------------------


def _bench_kernel_us(run_fn, warmup=10, iters=50, flush_mb=512):
    """Per-iteration CUDA events timer with L2 flush and IQR-filtered median."""
    l2_bytes = getattr(
        torch.cuda.get_device_properties(torch.cuda.current_device()),
        "L2_cache_size",
        256 * 1024 * 1024,  # fallback if L2_cache_size unavailable (cmodel reports 96 MB)
    )
    alloc_bytes = max(l2_bytes * 2, flush_mb * 1024 * 1024)
    flush_buf = torch.empty(alloc_bytes, dtype=torch.uint8, device="cuda")

    for _ in range(warmup):
        flush_buf.zero_()
        run_fn()
    torch.cuda.synchronize()

    start_ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        flush_buf.zero_()
        start_ev[i].record()
        run_fn()
        end_ev[i].record()

    torch.cuda.synchronize()
    latencies = sorted(start_ev[i].elapsed_time(end_ev[i]) * 1e3 for i in range(iters))

    n = len(latencies)
    if n >= 8:
        q1, q3 = latencies[n // 4], latencies[3 * n // 4]
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        filtered = [x for x in latencies if lo <= x <= hi]
        if filtered:
            latencies = filtered

    del flush_buf
    return latencies[len(latencies) // 2]


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

HBM_PEAK_TBS = 22.0


def _run_unique(grid_m, grid_n, warmup, iters, flush_mb):
    """Mode=unique: each WG reads unique tiles, all from HBM."""
    total_wgs = grid_m * grid_n
    tile_elems = TILE * TILE

    torch.manual_seed(42)
    a_dev = torch.randn((total_wgs * TILE, TILE), device="cuda", dtype=torch.bfloat16)
    b_dev = torch.randn((total_wgs * TILE, TILE), device="cuda", dtype=torch.bfloat16)
    c_dev = torch.zeros(total_wgs * tile_elems, device="cuda", dtype=torch.bfloat16)

    launch_fn = _compile_tdm_unique_add(grid_m, grid_n)

    median_us = _bench_kernel_us(lambda: launch_fn(a_dev, b_dev, c_dev), warmup, iters, flush_mb)

    read_bytes = total_wgs * 2 * TILE * TILE * ELEM_BYTES
    write_bytes = total_wgs * TILE * TILE * ELEM_BYTES
    total_bytes = read_bytes + write_bytes
    bw_tbs = total_bytes / (median_us / 1e6) / 1e12 if median_us > 0 else 0.0
    return median_us, bw_tbs


def _run_read_only(grid_m, grid_n, warmup, iters, flush_mb):
    """Mode=read-only: TDM loads only, minimal store. Measures pure TDM read BW."""
    total_wgs = grid_m * grid_n

    torch.manual_seed(42)
    a_dev = torch.randn((total_wgs * TILE, TILE), device="cuda", dtype=torch.bfloat16)
    b_dev = torch.randn((total_wgs * TILE, TILE), device="cuda", dtype=torch.bfloat16)
    c_dev = torch.zeros(total_wgs * VEC_WIDTH, device="cuda", dtype=torch.bfloat16)

    launch_fn = _compile_tdm_read_only(grid_m, grid_n)

    median_us = _bench_kernel_us(lambda: launch_fn(a_dev, b_dev, c_dev), warmup, iters, flush_mb)

    read_bytes = total_wgs * 2 * TILE * TILE * ELEM_BYTES
    bw_tbs = read_bytes / (median_us / 1e6) / 1e12 if median_us > 0 else 0.0
    return median_us, bw_tbs


def _run_multicast(grid_m, grid_n, cluster_x, cluster_y, warmup, iters, flush_mb):
    """Mode=multicast: GEMM-like shared tiles, measures multicast throughput."""
    total_wgs = grid_m * grid_n
    tile_elems = TILE * TILE

    M = grid_m * TILE
    K = TILE
    N = grid_n * TILE

    torch.manual_seed(42)
    a_dev = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b_dev = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
    c_dev = torch.zeros(total_wgs * tile_elems, device="cuda", dtype=torch.bfloat16)

    launch_fn = _compile_tdm_shared_add(grid_m, grid_n, cluster_x, cluster_y)

    median_us = _bench_kernel_us(lambda: launch_fn(a_dev, b_dev, c_dev), warmup, iters, flush_mb)

    # Effective throughput: count all bytes each WG processes (L2 serves repeats)
    eff_bytes = total_wgs * 2 * TILE * TILE * ELEM_BYTES
    eff_tbs = eff_bytes / (median_us / 1e6) / 1e12 if median_us > 0 else 0.0
    return median_us, eff_tbs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="TDM load bandwidth benchmark for gfx1250")
    parser.add_argument(
        "--mode",
        choices=["all", "unique", "multicast", "read-only"],
        default="all",
        help="all: run all modes (default); unique: TDM load+add+store; read-only: TDM load only; multicast: cluster multicast",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations (default: 50)")
    parser.add_argument(
        "--peak-bw", type=float, default=HBM_PEAK_TBS, help=f"Peak HBM BW in TB/s (default: {HBM_PEAK_TBS})"
    )
    parser.add_argument("--flush-mb", type=int, default=512, help="L2 flush buffer size in MB (default: 512)")
    args = parser.parse_args()

    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        print(f"ERROR: TDM benchmark requires gfx1250, got {arch}", file=sys.stderr)
        sys.exit(1)

    # MI450: 8 XCDs × 256 CUs = 2048 CUs total
    grid_sizes = [(4, 4), (8, 8), (16, 16), (32, 32), (32, 64), (64, 64), (64, 128), (128, 128), (256, 128), (256, 256)]
    # Multicast mode: square grids only (fewer combos, cleaner comparison)
    mcast_grid_sizes = [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)]

    run_all = args.mode == "all"
    if run_all or args.mode == "multicast":
        _run_multicast_sweep(mcast_grid_sizes, args)
    if run_all or args.mode == "read-only":
        _run_read_only_sweep(grid_sizes, args)
    if run_all or args.mode == "unique":
        _run_unique_sweep(grid_sizes, args)


def _run_read_only_sweep(grid_sizes, args):
    """Sweep grid sizes with read-only kernel — measures peak TDM read bandwidth."""
    print(f"\nTDM Pure Read Bandwidth (gfx1250, tile={TILE}x{TILE} bf16, TDM load only)")
    print(f"  warmup={args.warmup}, iters={args.iters}, peak={args.peak_bw} TB/s")
    print("  BW = read_bytes / time (2 tiles/WG, no store)")
    print("=" * 72)
    print(f"  {'Grid':>8s}  {'WGs':>6s}  {'Read(MB)':>9s}  {'Median(us)':>12s}  {'BW(TB/s)':>10s}  {'Util%':>7s}")
    print("-" * 72)

    for gm, gn in grid_sizes:
        total_wgs = gm * gn
        read_mb = total_wgs * 2 * TILE * TILE * ELEM_BYTES / (1024 * 1024)
        try:
            median_us, bw_tbs = _run_read_only(gm, gn, args.warmup, args.iters, args.flush_mb)
        except Exception as e:
            print(f"  {gm}x{gn:>3d}  {total_wgs:>6d}  {read_mb:>7.0f}    ERROR: {e}")
            continue

        util = bw_tbs / args.peak_bw * 100.0
        print(f"  {gm}x{gn:>3d}  {total_wgs:>6d}  {read_mb:>7.0f}    {median_us:10.1f}    {bw_tbs:8.3f}   {util:5.1f}%")

    print("=" * 72)


def _run_unique_sweep(grid_sizes, args):
    """Sweep grid sizes with unique tiles — measures raw HBM bandwidth."""
    print(f"\nTDM Load HBM Bandwidth (gfx1250, tile={TILE}x{TILE} bf16, unique tiles per WG)")
    print(f"  warmup={args.warmup}, iters={args.iters}, peak={args.peak_bw} TB/s")
    print("  BW = (read + write) / time;  read = 2 tiles/WG, write = 1 tile/WG")
    print("=" * 72)
    print(f"  {'Grid':>8s}  {'WGs':>6s}  {'Data(MB)':>9s}  {'Median(us)':>12s}  {'BW(TB/s)':>10s}  {'Util%':>7s}")
    print("-" * 72)

    for gm, gn in grid_sizes:
        total_wgs = gm * gn
        data_mb = total_wgs * 3 * TILE * TILE * ELEM_BYTES / (1024 * 1024)
        try:
            median_us, bw_tbs = _run_unique(gm, gn, args.warmup, args.iters, args.flush_mb)
        except Exception as e:
            print(f"  {gm}x{gn:>3d}  {total_wgs:>6d}  {data_mb:>7.0f}    ERROR: {e}")
            continue

        util = bw_tbs / args.peak_bw * 100.0
        print(f"  {gm}x{gn:>3d}  {total_wgs:>6d}  {data_mb:>7.0f}    {median_us:10.1f}    {bw_tbs:8.3f}   {util:5.1f}%")

    print("=" * 72)


def _run_multicast_sweep(grid_sizes, args):
    """Sweep grid × cluster configs — measures multicast L2→LDS throughput."""
    cluster_configs = [(1, 1), (2, 1), (1, 2), (2, 2), (4, 2), (2, 4), (4, 4), (8, 2), (2, 8)]

    print(f"\nTDM Multicast Throughput (gfx1250, tile={TILE}x{TILE} bf16, multicast tiles)")
    print(f"  warmup={args.warmup}, iters={args.iters}")
    print("  Eff.BW = total_wgs * 2 tiles / time (effective L2→LDS throughput)")
    print("=" * 78)
    print(f"  {'Grid':>8s}  {'WGs':>6s}  {'Cluster':>8s}  {'Median(us)':>12s}  {'Eff.BW(TB/s)':>14s}  {'Speedup':>8s}")
    print("-" * 78)

    for gm, gn in grid_sizes:
        baseline_us = None
        for cx, cy in cluster_configs:
            if gm % cx != 0 or gn % cy != 0:
                continue
            total_wgs = gm * gn
            try:
                median_us, eff_tbs = _run_multicast(gm, gn, cx, cy, args.warmup, args.iters, args.flush_mb)
            except Exception as e:
                print(f"  {gm}x{gn:>3d}  {total_wgs:>6d}    ({cx},{cy})    ERROR: {e}")
                continue

            if baseline_us is None:
                baseline_us = median_us
            speedup = baseline_us / median_us if median_us > 0 else 0.0

            print(
                f"  {gm}x{gn:>3d}  {total_wgs:>6d}    ({cx},{cy})  "
                f"  {median_us:10.1f}      {eff_tbs:10.3f}    {speedup:6.2f}x"
            )

    print("=" * 78)


if __name__ == "__main__":
    main()
