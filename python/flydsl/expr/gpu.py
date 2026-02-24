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


def find_ops(op, pred, single=False):
    if isinstance(op, (ir.OpView, ir.Module)):
        op = op.operation

    matching = []

    def _walk(op: ir.Operation):
        if single and matching:
            return
        for r in op.regions:
            for b in r.blocks:
                for o in b.operations:
                    if pred(o):
                        matching.append(o)
                    _walk(o)

    _walk(op)
    if single and matching:
        matching = matching[0]
    return matching


def get_compile_object_bytes(compiled_module):
    binary = find_ops(compiled_module, lambda o: isinstance(o, gpu.BinaryOp), single=True)
    objects = list(map(gpu.ObjectAttr, binary.objects))
    return objects[-1].object


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
    "find_ops",
    "get_compile_object_bytes",
    "SharedAllocator",
]
