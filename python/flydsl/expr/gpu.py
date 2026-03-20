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

from .._mlir import ir
from .._mlir.dialects import gpu
from .._mlir.ir import Attribute
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


class SharedAllocator:
    """Placeholder for shared memory allocation (see ``flydsl.utils.smem_allocator``)."""
    pass


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
    "SharedAllocator",
]
