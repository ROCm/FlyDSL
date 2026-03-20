# Copyright (c) 2025 FlyDSL Project Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
