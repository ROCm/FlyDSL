import ctypes
from typing import Generic, TypeVar

from .._mlir import ir
from .._mlir.dialects import gpu
from .numeric import (
    BFloat16,
    Boolean,
    Float,
    Float8E4M3,
    Float8E4M3FN,
    Float8E5M2,
    Float16,
    Float32,
    Float64,
    Int4,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    as_numeric,
)

__all__ = [
    # DSL Type
    "Boolean",
    "Float",
    "BFloat16",
    "Float8E4M3",
    "Float8E4M3FN",
    "Float8E5M2",
    "Float16",
    "Float32",
    "Float64",
    "Int4",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Uint8",
    "Uint16",
    "Uint32",
    "Uint64",
    "Constexpr",
    "Tensor",
    "Stream",
    "Tuple3D",
    # Utility functions
    "as_numeric",
]


ValueT = TypeVar("ValueT")


class Constexpr(Generic[ValueT]):
    pass


class IntTuple:
    pass


class Basis:
    pass


class Layout:
    pass


class ComposedLayout:
    pass


class CoordTensor:
    pass


class Swizzle:
    pass


class Tensor:
    def __init__(self, value: ir.Value):
        self.value = value

    def __str__(self):
        return f"Tensor({self.value})"

    @classmethod
    def __new_from_ir_values__(cls, values):
        return Tensor(values[0])

    def __extract_ir_values__(self):
        return [self.value]


class Stream:
    def __init__(self, value=None):
        self.value = value

    def __ir_types__(self):
        return [gpu.AsyncTokenType.get()]

    def __c_pointers__(self):
        if self.value is None:
            # default nullptr stream
            return [ctypes.cast(ctypes.pointer(ctypes.c_void_p(0)), ctypes.c_void_p)]
        return [ctypes.cast(ctypes.pointer(ctypes.c_void_p(self.value.cuda_stream)), ctypes.c_void_p)]

    @classmethod
    def __new_from_ir_values__(cls, values):
        return Stream(values[0])

    def __extract_ir_values__(self):
        return [self.value]


class Tuple3D:
    def __init__(self, factory, dtype=Int32):
        self.factory = factory
        self.dtype = dtype

    def __getattr__(self, name):
        if name in ("x", "y", "z"):
            return self.dtype(self.factory(name))
        raise AttributeError(name)

    def __iter__(self):
        return iter((self.x, self.y, self.z))
