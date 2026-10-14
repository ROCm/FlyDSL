# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared utilities for gfx1250 GEMM kernels (fp16 / mxfp4 / mxfp8)."""

import math as _math

import flydsl.expr as fx
from flydsl.expr import gpu, rocdl
from flydsl.expr.rocdl import cluster, tdm_ops
from flydsl.expr.typing import T


def make_lds_copy_ops(bits):
    """Create one reusable layout/copy atom and return its load/store callables."""
    if bits not in (32, 64, 128):
        raise ValueError(f"bits must be 32/64/128, got {bits}")
    elem_count = bits // fx.Int32.width
    layout = fx.make_layout(elem_count, 1)
    atom = fx.make_copy_atom(fx.UniversalCopy(bits), fx.Int32)
    ptr_ty = fx.PointerType.get(
        elem_ty=fx.Int32.ir_type,
        address_space=fx.AddressSpace.Shared,
        alignment=bits // 8,
    )

    def _view(lds_base_idx, byte_offset):
        addr_i32 = fx.Int32(lds_base_idx) + fx.Int32(byte_offset)
        ptr = fx.inttoptr(ptr_ty, addr_i32)
        return fx.Tensor(fx.make_view(ptr, layout))

    def load(lds_base_idx, byte_offset):
        rmem = fx.make_rmem_tensor(layout, fx.Int32)
        fx.copy(atom, _view(lds_base_idx, byte_offset), rmem)
        return rmem.load()

    def store(lds_base_idx, byte_offset, data):
        rmem = fx.make_rmem_tensor(layout, fx.Int32)
        rmem.store(data)
        fx.copy(atom, rmem, _view(lds_base_idx, byte_offset))

    return load, store


def make_gemm_grid(M, N, tile_m, tile_n, cluster_m, *, split_k=1, persistent_n_tiles=1, fused_splitk=False):
    """Pad M to whole clusters and split long grids without duplicating tiles.

    Fused split-K reserves grid.z for the K splits of one tile. Other grids
    may put exact chunks of M there, keeping grid.x in whole cluster rows.
    """
    rows = fx.ceildiv(M, tile_m)
    rows = fx.ceildiv(fx.max(rows, 1), cluster_m) * cluster_m
    run = rows
    if not fused_splitk:
        divisor = rows & -rows
        capped = fx.min(divisor, 32)
        fits = (capped % cluster_m) == 0
        run = ((rows > 32) & (divisor >= 8) & fits).select(capped, rows)
    return run, fx.ceildiv(N, tile_n) // persistent_n_tiles, (rows // run) * split_k


def workgroup_barrier(use_cluster=False):
    """Issue the appropriate barrier for LDS visibility.

    Cluster mode layers an inter-workgroup barrier on top of the regular
    workgroup barrier protocol, so call sites can treat it as a single
    "LDS is now readable" fence.
    """
    if use_cluster:
        cluster.cluster_barrier()
    else:
        gpu.barrier()


def pipeline_fence(outstanding=0, use_cluster=False):
    """Fused READY+REUSE fence for gfx1250 multi-buffer pipeline.

    Issues ``s_wait_tensorcnt`` followed by the appropriate barrier.
    """
    tdm_ops.tensor_wait(outstanding)
    workgroup_barrier(use_cluster=use_cluster)


WGP_BARRIER_ID = -1


def pipeline_fence_signal(outstanding=0, use_cluster=False):
    """Signal half of a split barrier fence.

    Issues ``s_wait_tensorcnt`` then ``s_barrier_signal -1``.
    The matching ``pipeline_fence_wait`` must be called later
    (typically mid-compute) before reading the LDS data.

    When *use_cluster* is True the intra-WG barrier is still required
    so that all waves' TDM loads are visible before any wave reads LDS.
    The cluster barrier is layered on top for inter-WG synchronisation.
    """
    tdm_ops.tensor_wait(outstanding)
    rocdl.s_barrier_signal(WGP_BARRIER_ID)
    if use_cluster:
        cluster.cluster_signal_once_per_wg()


def pipeline_fence_wait(use_cluster=False):
    """Wait half of a split barrier fence.

    Issues ``s_barrier_wait -1``.  Must be preceded by a matching
    ``pipeline_fence_signal`` from all waves in the workgroup.
    """
    rocdl.s_barrier_wait(WGP_BARRIER_ID)
    if use_cluster:
        cluster.cluster_wait()


LOG2E = _math.log2(_math.e)


def fmin_f32(a, b):
    """Scalar f32 min (select-based, no NaN handling)."""
    import flydsl.expr as _fx

    return _fx.Float32((a < b).select(a, b))


def fmax_f32(a, b):
    """Scalar f32 max (select-based, no NaN handling)."""
    import flydsl.expr as _fx

    return _fx.Float32((a > b).select(a, b))


def fused_silu_swiglu_elem(g, u, *, swiglu, limit_f32, neg_limit_f32):
    """One (gate, up) pair -> fused silu or swiglu scalar (gpt-oss clamp)."""
    import flydsl.expr as _fx

    _one = _fx.Float32(1.0)
    g = fmin_f32(g, limit_f32)
    u = fmin_f32(fmax_f32(u, neg_limit_f32), limit_f32)
    if swiglu:
        nlog2e = _fx.Float32(-1.702 * LOG2E)
        sig = _fx.Float32(rocdl.rcp(T.f32, _one + (g * nlog2e).exp2()))
        return g * sig * (u + _one)
    nlog2e = _fx.Float32(-LOG2E)
    sig = _fx.Float32(rocdl.rcp(T.f32, _one + (g * nlog2e).exp2()))
    return g * sig * u


__all__ = [
    # LDS helpers
    "make_lds_copy_ops",
    # Launch grid
    "make_gemm_grid",
    # Pipeline
    "workgroup_barrier",
    "pipeline_fence",
    "pipeline_fence_signal",
    "pipeline_fence_wait",
    # Scalar math
    "LOG2E",
    "fmin_f32",
    "fmax_f32",
    "fused_silu_swiglu_elem",
]
