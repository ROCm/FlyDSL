# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GPU intrinsics and address space helpers.

Provides thread/block indexing (``thread_idx``, ``block_idx``),
synchronization (``barrier``), and shared memory address space
(``smem_space``/``lds_space``) for kernel authoring.

Usage::

    import flydsl.expr as fx

    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    fx.gpu.barrier()
"""

from .._mlir.dialects import gpu
from .._mlir.ir import Attribute
from .primitive import get_dyn_shared
from .struct import Arena
from .typing import Tuple3D

thread_id = gpu.thread_id
block_id = gpu.block_id

thread_idx = Tuple3D(gpu.thread_id)
block_idx = Tuple3D(gpu.block_id)
block_dim = Tuple3D(gpu.block_dim)
grid_dim = Tuple3D(gpu.grid_dim)

barrier = gpu.barrier

_int = int


def smem_space(int=False):
    """Return the GPU shared memory (LDS/workgroup) address space.

    Args:
        int: If True, return the integer value; otherwise return an
             MLIR ``#gpu.address_space<workgroup>`` attribute.
    """
    a = gpu.AddressSpace.Workgroup
    if int:
        return _int(a)
    return Attribute.parse(f"#gpu.address_space<{a}>")


lds_space = smem_space


CLUSTER_BARRIER_ID = -3
CLUSTER_WAIT_ALL = CLUSTER_BARRIER_ID


def _rocdl_cluster():
    from .rocdl import cluster

    return cluster


def is_wave_leader():
    """Return true for wave-0 inside the workgroup."""
    return _rocdl_cluster().is_wave_leader()


def cluster_signal_once_per_wg():
    """Signal cluster barrier from exactly one wave per workgroup."""
    return _rocdl_cluster().cluster_signal_once_per_wg()


def cluster_wait():
    """Wait on the cluster user barrier."""
    return _rocdl_cluster().cluster_wait()


def cluster_barrier():
    """Workgroup + cluster barrier with one-wave signal semantics.

    This is the safe default for kernels using cluster multicast:
      1) synchronize waves inside each workgroup
      2) signal cluster barrier once per workgroup (wave-0 only)
      3) wait for all workgroups in the cluster
    """
    return _rocdl_cluster().cluster_barrier()


def compute_cluster_position():
    """Compute a workgroup's (row, col) position within its cluster.

    Returns:
        (local_x, local_y) as MLIR index values — position within the cluster.
    """
    return _rocdl_cluster().compute_cluster_position()


def compute_mcast_masks(local_x, local_y, cluster_m: _int, cluster_n: _int):
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
    return _rocdl_cluster().compute_mcast_masks(local_x, local_y, cluster_m, cluster_n)


class SharedAllocator(Arena):
    def __init__(self, base_alignment: int = Arena.DEFAULT_BASE_ALIGNMENT):
        super().__init__(base_alignment=base_alignment)

        from ..compiler.kernel_function import KernelFunction

        kf = KernelFunction.get_current()
        if kf is None:
            raise RuntimeError("SharedAllocator can only be created inside a @kernel function")
        kf.register_shared_allocator(self)
        self._base = get_dyn_shared()

    @property
    def base_ptr(self):
        return self._base


__all__ = [
    "thread_id",
    "block_id",
    "thread_idx",
    "block_idx",
    "block_dim",
    "grid_dim",
    "barrier",
    "smem_space",
    "lds_space",
    "is_wave_leader",
    "cluster_signal_once_per_wg",
    "cluster_wait",
    "cluster_barrier",
    "compute_cluster_position",
    "compute_mcast_masks",
    "CLUSTER_BARRIER_ID",
    "CLUSTER_WAIT_ALL",
    "SharedAllocator",
]
