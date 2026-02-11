from .._mlir.dialects import gpu
from .typing import Tuple3D

thread_idx = Tuple3D(gpu.thread_id)
block_idx = Tuple3D(gpu.block_id)
block_dim = Tuple3D(gpu.block_dim)
grid_dim = Tuple3D(gpu.grid_dim)


class SharedAllocator:
    pass


__all__ = [
    "thread_idx",
    "block_idx",
    "block_dim",
    "grid_dim",
    "SharedAllocator",
]
