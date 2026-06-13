#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Cluster launch tests for gfx1250.

Validates the mgpuLaunchClusterKernel runtime path (hipDrvLaunchKernelEx)
and cluster synchronization intrinsics.

Tests:
  1. Smoke: vec_add launched with cluster dims — proves the runtime launch path works.
  2. Barrier: vec_add with cluster_barrier() — validates cluster sync intrinsics.

See also: test_cluster_mcast_gemm_gfx1250.py for TDM multicast GEMM tests (deferred).
"""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx

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
    pytest.skip(f"Cluster launch requires gfx1250, got {_arch}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Test 1 & 2: Simple vec_add kernel with cluster launch
# ---------------------------------------------------------------------------

VEC_WIDTH = 4
BLOCK_DIM = 256
TILE_ELEMS = BLOCK_DIM * VEC_WIDTH


@flyc.kernel
def vec_add_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
    grid_dim_x: fx.Constexpr[int],
):
    """Element-wise C = A + B with buffer load/store."""
    bid = fx.block_idx.y * grid_dim_x + fx.block_idx.x
    tid = fx.thread_idx.x

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    tA = fx.logical_divide(A, fx.make_layout(block_dim * vec_width, 1))
    tB = fx.logical_divide(B, fx.make_layout(block_dim * vec_width, 1))
    tC = fx.logical_divide(C, fx.make_layout(block_dim * vec_width, 1))

    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))
    tC = fx.slice(tC, (None, bid))

    tA = fx.logical_divide(tA, fx.make_layout(vec_width, 1))
    tB = fx.logical_divide(tB, fx.make_layout(vec_width, 1))
    tC = fx.logical_divide(tC, fx.make_layout(vec_width, 1))

    copyAtom = fx.make_copy_atom(fx.rocdl.BufferCopy(vec_width * fx.Float32.width), fx.Float32)
    rA = fx.make_rmem_tensor(vec_width, fx.Float32)
    rB = fx.make_rmem_tensor(vec_width, fx.Float32)
    rC = fx.make_rmem_tensor(vec_width, fx.Float32)

    fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)

    vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
    fx.memref_store_vec(vC, rC)

    fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))


@flyc.kernel
def vec_add_cluster_barrier_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
    grid_dim_x: fx.Constexpr[int],
):
    """Vec_add with a cluster_barrier() call to exercise cluster sync intrinsics."""
    from flydsl.expr.rocdl.cluster import cluster_barrier

    bid = fx.block_idx.y * grid_dim_x + fx.block_idx.x
    tid = fx.thread_idx.x

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    tA = fx.logical_divide(A, fx.make_layout(block_dim * vec_width, 1))
    tB = fx.logical_divide(B, fx.make_layout(block_dim * vec_width, 1))
    tC = fx.logical_divide(C, fx.make_layout(block_dim * vec_width, 1))

    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))
    tC = fx.slice(tC, (None, bid))

    tA = fx.logical_divide(tA, fx.make_layout(vec_width, 1))
    tB = fx.logical_divide(tB, fx.make_layout(vec_width, 1))
    tC = fx.logical_divide(tC, fx.make_layout(vec_width, 1))

    copyAtom = fx.make_copy_atom(fx.rocdl.BufferCopy(vec_width * fx.Float32.width), fx.Float32)
    rA = fx.make_rmem_tensor(vec_width, fx.Float32)
    rB = fx.make_rmem_tensor(vec_width, fx.Float32)
    rC = fx.make_rmem_tensor(vec_width, fx.Float32)

    fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)

    # Cluster barrier: sync all WGs in the cluster before computing.
    # This exercises rocdl.s_barrier_signal / s_barrier_wait intrinsics.
    cluster_barrier()

    vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
    fx.memref_store_vec(vC, rC)

    fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))


def _run_vec_add_cluster(cluster_x, cluster_y, use_barrier=False):
    """Launch vec_add with cluster dims and verify correctness."""
    # Grid dims must be >= cluster dims and divisible by them.
    # Use a 2D grid: gx tiles along x, gy tiles along y.
    gx = max(cluster_x, 2) * 2  # ensure gx divisible by cluster_x
    gy = max(cluster_y, 2) * 2  # ensure gy divisible by cluster_y
    total_tiles = gx * gy
    size = TILE_ELEMS * total_tiles

    a_dev = torch.randn(size, device="cuda", dtype=torch.float32)
    b_dev = torch.randn(size, device="cuda", dtype=torch.float32)
    c_dev = torch.empty_like(a_dev)

    assert gx % cluster_x == 0 and gy % cluster_y == 0
    cluster_dims_str = f"{cluster_x},{cluster_y},1"
    kernel_fn = vec_add_cluster_barrier_kernel if use_barrier else vec_add_kernel

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        L = kernel_fn(
            A,
            B,
            C,
            BLOCK_DIM,
            VEC_WIDTH,
            gx,
            value_attrs={
                "rocdl.cluster_dims": cluster_dims_str,
            },
        )
        L.launch(
            grid=(gx, gy, 1),
            block=(BLOCK_DIM, 1, 1),
            stream=stream,
            cluster=(cluster_x, cluster_y, 1),
        )

    stream = torch.cuda.Stream()
    tA = flyc.from_dlpack(a_dev).mark_layout_dynamic(leading_dim=0, divisibility=VEC_WIDTH)
    launch(tA, b_dev, c_dev, size, stream=stream)
    torch.cuda.synchronize()

    max_err = (c_dev - (a_dev + b_dev)).abs().max().item()
    assert max_err < 1e-5, f"vec_add cluster=({cluster_x},{cluster_y},1) max_err={max_err}"


# --- Test 1: Cluster launch smoke test ---
@pytest.mark.parametrize(
    "cluster_x, cluster_y",
    [
        (2, 1),
        (1, 2),
        (2, 2),
        (4, 1),
    ],
)
def test_cluster_launch_vec_add(cluster_x, cluster_y):
    """Smoke test: vec_add with cluster dims exercises mgpuLaunchClusterKernel."""
    _run_vec_add_cluster(cluster_x, cluster_y, use_barrier=False)


# --- Test 2: Cluster launch with cluster_barrier ---
@pytest.mark.parametrize(
    "cluster_x, cluster_y",
    [
        (2, 1),
        (2, 2),
    ],
)
def test_cluster_launch_with_barrier(cluster_x, cluster_y):
    """Vec_add with cluster_barrier() validates cluster sync intrinsics."""
    _run_vec_add_cluster(cluster_x, cluster_y, use_barrier=True)
