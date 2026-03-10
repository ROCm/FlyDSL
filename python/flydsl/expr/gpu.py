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
    a = gpu.AddressSpace.Workgroup
    if int:
        return _int(a)
    return Attribute.parse(f"#gpu.address_space<{a}>")


lds_space = smem_space


class SharedAllocator:
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
