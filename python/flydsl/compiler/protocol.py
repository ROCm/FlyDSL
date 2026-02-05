import ctypes
from itertools import chain
from typing import List, Protocol, runtime_checkable

from .._mlir import ir


@runtime_checkable
class DslType(Protocol):
    @classmethod
    def __new_from_ir_values__(cls, values: List[ir.Value]) -> "DslType": ...
    def __extract_ir_values__(self) -> List[ir.Value]: ...


@runtime_checkable
class JitArgument(Protocol):
    def __ir_types__(self) -> List[ir.Type]: ...
    def __c_pointers__(self) -> List[ctypes.c_void_p]: ...


def get_ir_types(obj) -> List[ir.Type]:
    if isinstance(obj, ir.Value):
        return [obj.type]
    elif hasattr(obj, "__ir_types__"):
        return obj.__ir_types__()
    elif hasattr(obj, "__extract_ir_values__"):
        return [v.type for v in obj.__extract_ir_values__()]
    elif isinstance(obj, (tuple, list)):
        return list(chain.from_iterable(get_ir_types(x) for x in obj))
    raise TypeError(f"Cannot get IR types from {obj}")


def get_c_pointers(obj) -> List[ctypes.c_void_p]:
    if hasattr(obj, "__c_pointers__"):
        return obj.__c_pointers__()
    elif isinstance(obj, (tuple, list)):
        return list(chain.from_iterable(get_c_pointers(x) for x in obj))
    raise TypeError(f"Cannot get C pointers from {obj}")


def extract_ir_values(obj) -> List[ir.Value]:
    if isinstance(obj, ir.Value):
        return [obj]
    elif hasattr(obj, "__extract_ir_values__"):
        return obj.__extract_ir_values__()
    elif isinstance(obj, (tuple, list)):
        return list(chain.from_iterable(extract_ir_values(x) for x in obj))
    raise TypeError(f"Cannot extract IR values from {obj}")


def new_from_ir_values(dsl_type, args, values: List[ir.Value]) -> DslType:
    if hasattr(dsl_type, "__new_from_ir_values__"):
        return dsl_type.__new_from_ir_values__(values)
    elif isinstance(dsl_type, (tuple, list)):
        elem = []
        for ty, arg in zip(dsl_type, args, strict=True):
            val_num = len(get_ir_types(arg))
            elem.append(new_from_ir_values(ty, arg, values[:val_num]))
            values = values[val_num:]
        return type(dsl_type)(elem)
    raise TypeError(f"Cannot construct from IR values for {dsl_type}")
