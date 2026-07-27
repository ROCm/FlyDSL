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


__all__ = [
    "CLUSTER_BARRIER_ID",
    "CLUSTER_WAIT_ALL",
    "is_wave_leader",
    "cluster_signal_once_per_wg",
    "cluster_wait",
    "cluster_barrier",
    "compute_cluster_position",
]
