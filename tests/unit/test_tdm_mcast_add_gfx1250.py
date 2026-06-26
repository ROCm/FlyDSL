#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""TDM multicast + cluster launch bandwidth test for gfx1250.

Validates that TDM async loads with cluster multicast masks work
correctly and measures the bandwidth benefit of multicast.

The kernel mimics a GEMM tiling pattern:
  A: (grid_m * T, T)  — WGs in the same row share one A tile (multicast)
  B: (T, grid_n * T)  — WGs in the same col share one B tile (multicast)
  C: (grid_m * T, grid_n * T) = broadcast(A_tile) + broadcast(B_tile)
"""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, tdm_ops, vector
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, get_op_result_or_value

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402

_arch = str(get_rocm_arch())
if _arch != "gfx1250":
    pytest.skip(f"TDM multicast requires gfx1250, got {_arch}", allow_module_level=True)

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
# Kernel compilation
# ---------------------------------------------------------------------------


def _compile_tdm_mcast_add(grid_m, grid_n, cluster_m, cluster_n):
    """Compile the TDM multicast add kernel for given grid and cluster dims."""
    use_cluster = cluster_m > 1 or cluster_n > 1

    lds_a_offset = 0
    lds_b_offset = ((TILE_BYTES + 127) // 128) * 128
    total_lds = lds_b_offset + TILE_BYTES

    smem_alloc = SmemAllocator(None, arch="gfx1250", global_sym_name="tdm_mcast_add_smem")
    smem_alloc.ptr = total_lds

    stride_a = TILE
    stride_b = grid_n * TILE
    tile_elems = TILE * TILE

    @flyc.kernel
    def tdm_mcast_add_kernel(
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_c: fx.Tensor,
    ):
        # index-typed ids for TDM descriptors and LDS addressing
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        tid = gpu.thread_id("x")

        # --- Cluster multicast masks ---
        if const_expr(use_cluster):
            local_x, local_y = cluster.compute_cluster_position()
            a_mcast_mask, b_mcast_mask = cluster.compute_mcast_masks(local_x, local_y, cluster_m, cluster_n)
        else:
            a_mcast_mask = 0
            b_mcast_mask = 0

        # --- LDS setup ---
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

        # --- TDM descriptors ---
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

        # --- Issue TDM loads and sync ---
        tdm_ops.tensor_load_2d(desc_a)
        tdm_ops.tensor_load_2d(desc_b)
        tdm_ops.tensor_wait(0)
        if const_expr(use_cluster):
            cluster.cluster_barrier()
        else:
            gpu.barrier()

        # --- Read from LDS, add, store to global via buffer store ---
        # i32-typed ids for CuTE layout ops (fly.make_int_tuple requires i32/i64)
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
        L = tdm_mcast_add_kernel(
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
            L.launch(
                grid=(grid_m, grid_n, 1),
                block=(BLOCK_DIM, 1, 1),
                stream=stream,
            )

    return launch


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


def _run_tdm_mcast_add(grid_m, grid_n, cluster_x, cluster_y, n_warmup=0, n_iters=1):
    """Launch the TDM multicast add kernel and verify correctness.

    Returns (max_err, avg_us) where avg_us is 0 if n_warmup == 0.
    """
    assert grid_m % cluster_x == 0 and grid_n % cluster_y == 0

    M = grid_m * TILE
    K = TILE
    N = grid_n * TILE

    tile_elems = TILE * TILE
    total_c_elems = grid_m * grid_n * tile_elems

    torch.manual_seed(42)
    a_dev = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b_dev = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
    c_dev = torch.zeros(total_c_elems, device="cuda", dtype=torch.bfloat16)

    launch_fn = _compile_tdm_mcast_add(grid_m, grid_n, cluster_x, cluster_y)
    stream = torch.cuda.Stream()

    # Warmup
    for _ in range(n_warmup):
        launch_fn(a_dev, b_dev, c_dev, stream=stream)
    torch.cuda.synchronize()

    avg_us = 0.0
    if n_iters > 1 and n_warmup > 0:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        for _ in range(n_iters):
            launch_fn(a_dev, b_dev, c_dev, stream=stream)
        end_event.record(stream)
        torch.cuda.synchronize()
        avg_us = start_event.elapsed_time(end_event) * 1000.0 / n_iters
    else:
        launch_fn(a_dev, b_dev, c_dev, stream=stream)
        torch.cuda.synchronize()

    # Reference: flat tile bid = bx * grid_n + by, each tile = flatten(A_tile + B_tile)
    c_ref = torch.zeros_like(c_dev)
    for bx in range(grid_m):
        a_tile = a_dev[bx * TILE : (bx + 1) * TILE, :]
        for by in range(grid_n):
            b_tile = b_dev[:, by * TILE : (by + 1) * TILE]
            bid = bx * grid_n + by
            c_ref[bid * tile_elems : (bid + 1) * tile_elems] = (a_tile + b_tile).flatten()

    max_err = (c_dev.float() - c_ref.float()).abs().max().item()
    return max_err, avg_us


# ---------------------------------------------------------------------------
# Test 1: Correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cluster_x, cluster_y",
    [
        (2, 1),
        (1, 2),
        (2, 2),
    ],
)
def test_tdm_mcast_add_correctness(cluster_x, cluster_y):
    """TDM multicast add: verify correctness with cluster multicast."""
    grid_m = max(cluster_x, 2) * 2
    grid_n = max(cluster_y, 2) * 2
    max_err, _ = _run_tdm_mcast_add(grid_m, grid_n, cluster_x, cluster_y)
    assert max_err < 1e-2, f"cluster=({cluster_x},{cluster_y}) max_err={max_err}"


# ---------------------------------------------------------------------------
# Test 2: Bandwidth comparison
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_tdm_mcast_add_bandwidth():
    """Compare TDM multicast vs no-multicast bandwidth."""
    grid_m, grid_n = 8, 8
    n_warmup, n_iters = 5, 20

    # No multicast baseline
    max_err_base, us_base = _run_tdm_mcast_add(grid_m, grid_n, 1, 1, n_warmup, n_iters)
    assert max_err_base < 1e-2, f"baseline max_err={max_err_base}"

    # With multicast
    max_err_mcast, us_mcast = _run_tdm_mcast_add(grid_m, grid_n, 2, 2, n_warmup, n_iters)
    assert max_err_mcast < 1e-2, f"multicast max_err={max_err_mcast}"

    total_wgs = grid_m * grid_n
    read_bytes = total_wgs * 2 * TILE * TILE * ELEM_BYTES
    write_bytes = total_wgs * TILE * TILE * ELEM_BYTES
    total_bytes = read_bytes + write_bytes

    bw_base = total_bytes / (us_base / 1e6) / 1e9 if us_base > 0 else 0
    bw_mcast = total_bytes / (us_mcast / 1e6) / 1e9 if us_mcast > 0 else 0
    speedup = us_base / us_mcast if us_mcast > 0 else 0

    print(f"\n{'='*60}")
    print(f"TDM Multicast Add Bandwidth (grid={grid_m}x{grid_n}, tile={TILE})")
    print(f"  No multicast:   {us_base:8.1f} us  {bw_base:8.1f} GB/s")
    print(f"  Multicast 2x2:  {us_mcast:8.1f} us  {bw_mcast:8.1f} GB/s")
    print(f"  Speedup:        {speedup:.2f}x")
    print(f"{'='*60}")
