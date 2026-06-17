# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""ROCDL cluster helpers for gfx1250 workgroup clustering."""

from ..._mlir import ir
from ..._mlir.dialects import gpu, rocdl, scf
from .. import arith as _arith_ext
from ..meta import dsl_loc_tracing
from ..typing import T
from . import cluster_workgroup_id_x, cluster_workgroup_id_y, wave_id

CLUSTER_BARRIER_ID = -3
# For cluster sync, wait on the cluster user barrier itself.
CLUSTER_WAIT_ALL = CLUSTER_BARRIER_ID


@dsl_loc_tracing
def is_wave_leader():
    """Return true for wave-0 inside the workgroup."""
    return _arith_ext.cmpi(
        _arith_ext.CmpIPredicate.eq,
        wave_id(),
        _arith_ext.constant(0, type=T.i32),
    )


@dsl_loc_tracing
def cluster_signal_once_per_wg():
    """Signal cluster barrier from exactly one wave per workgroup."""
    if_op = scf.IfOp(is_wave_leader(), [], has_else=False)
    if len(if_op.regions[0].blocks) == 0:
        if_op.regions[0].blocks.append(*[])
    with ir.InsertionPoint(if_op.regions[0].blocks[0]):
        rocdl.s_barrier_signal(CLUSTER_BARRIER_ID)
        scf.YieldOp([])


@dsl_loc_tracing
def cluster_wait():
    """Wait on the cluster user barrier."""
    rocdl.s_barrier_wait(CLUSTER_WAIT_ALL)


@dsl_loc_tracing
def cluster_barrier():
    """Workgroup + cluster barrier with one-wave signal semantics.

    This is the safe default for kernels using cluster multicast:
      1) synchronize waves inside each workgroup
      2) signal cluster barrier once per workgroup (wave-0 only)
      3) wait for all workgroups in the cluster
    """
    gpu.barrier()
    cluster_signal_once_per_wg()
    cluster_wait()


@dsl_loc_tracing
def compute_cluster_position():
    """Compute a workgroup's (row, col) position within its cluster.

    Returns:
        (local_x, local_y) as MLIR index values -- position within the cluster.
    """
    local_x = _arith_ext.index_cast(T.index, cluster_workgroup_id_x())
    local_y = _arith_ext.index_cast(T.index, cluster_workgroup_id_y())
    return local_x, local_y


@dsl_loc_tracing
def compute_mcast_masks(local_x, local_y, cluster_m: int, cluster_n: int):
    """Compute MCAST workgroup_mask values for A and B matrices.

    Hardware flat WG index within a cluster uses X-inner ordering
    (MI400 Shader Programming, TTMP6 layout, section 3.5.5.1):

        flat_wg_id = wg_x + wg_y * nwg_x = local_x + local_y * cluster_m

    where cluster_dims = (cluster_m, cluster_n, 1), so nwg_x = cluster_m.

    A mask: WGs sharing the same M-tile row (same local_x, varying local_y).
        Bits: {local_x + ly * cluster_m : ly in 0..cluster_n-1}
    B mask: WGs sharing the same N-tile column (same local_y, varying local_x).
        Bits: {lx + local_y * cluster_m : lx in 0..cluster_m-1}

    Args:
        local_x: WG row within cluster (MLIR index, 0..cluster_m-1).
        local_y: WG column within cluster (MLIR index, 0..cluster_n-1).
        cluster_m: Cluster rows (Python int).
        cluster_n: Cluster columns (Python int).

    Returns:
        (a_mask, b_mask) as MLIR i32 values for TDM workgroup_mask.
    """
    local_x_i32 = _arith_ext.index_cast(T.i32, local_x)
    local_y_i32 = _arith_ext.index_cast(T.i32, local_y)
    cluster_m_i32 = _arith_ext.constant(cluster_m, type=T.i32)

    # A mask: pattern has bits at strides of cluster_m, shifted by local_x.
    a_pattern_val = 0
    for ly in range(cluster_n):
        a_pattern_val |= 1 << (ly * cluster_m)
    a_pattern = _arith_ext.constant(a_pattern_val, type=T.i32)
    a_mask = _arith_ext.shli(a_pattern, local_x_i32)

    # B mask: cluster_m contiguous low bits, shifted by local_y * cluster_m.
    b_pattern = _arith_ext.constant((1 << cluster_m) - 1, type=T.i32)
    col_base = _arith_ext.muli(local_y_i32, cluster_m_i32)
    b_mask = _arith_ext.shli(b_pattern, col_base)

    return a_mask, b_mask


__all__ = [
    "CLUSTER_BARRIER_ID",
    "CLUSTER_WAIT_ALL",
    "is_wave_leader",
    "cluster_signal_once_per_wg",
    "cluster_wait",
    "cluster_barrier",
    "compute_cluster_position",
    "compute_mcast_masks",
]
