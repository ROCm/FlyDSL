import ctypes
from typing import Generic, TypeVar

from .._mlir import ir
from .._mlir.dialects import gpu, llvm
from .._mlir.extras import types as T

ValueT = TypeVar("ValueT")


class Constexpr(Generic[ValueT]):
    """
    Constexpr is transparent for mlir, it will be replaced by the actual value at compile time.
    """

    pass


class Int32:
    def __init__(self, value):
        self.value = value

    def __ir_types__(self):
        return [T.i32()]

    def __c_pointers__(self):
        return [ctypes.cast(ctypes.pointer(ctypes.c_int32(self.value)), ctypes.c_void_p)]

    @classmethod
    def __new_from_ir_values__(self, values):
        return Int32(values[0])

    def __extract_ir_values__(self):
        return [self.value]


class Tensor:
    def __init__(self, value: ir.Value):
        self.value = value

    @classmethod
    def __new_from_ir_values__(cls, values):
        return Tensor(values[0])

    def __extract_ir_values__(self):
        return [self.value]


class Stream:
    def __init__(self, value):
        self.value = value

    def __ir_types__(self):
        return [gpu.AsyncTokenType.get()]

    def __c_pointers__(self):
        if self.value is None:
            return [ctypes.cast(ctypes.pointer(ctypes.c_void_p(0)), ctypes.c_void_p)]
        return [ctypes.cast(ctypes.pointer(ctypes.c_void_p(self.value.cuda_stream)), ctypes.c_void_p)]

    @classmethod
    def __new_from_ir_values__(cls, values):
        return Stream(values[0])

    def __extract_ir_values__(self):
        return [self.value]
