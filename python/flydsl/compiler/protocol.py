# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ctypes
from itertools import chain
from types import SimpleNamespace
from typing import List, Protocol, runtime_checkable

from .._mlir import ir


@runtime_checkable
class DslType(Protocol):
    @classmethod
    def __construct_from_ir_values__(cls, values: List[ir.Value]) -> "DslType": ...
    def __extract_to_ir_values__(self) -> List[ir.Value]: ...


@runtime_checkable
class JitArgument(Protocol):
    def __get_ir_types__(self) -> List[ir.Type]: ...
    def __get_c_pointers__(self) -> List[ctypes.c_void_p]: ...


def get_ir_types(obj) -> List[ir.Type]:
    if isinstance(obj, ir.Value):
        return [obj.type]
    if hasattr(obj, "__get_ir_types__"):
        return obj.__get_ir_types__()
    if hasattr(obj, "__extract_to_ir_values__"):
        return [v.type for v in obj.__extract_to_ir_values__()]
    if isinstance(obj, SimpleNamespace):
        return list(chain.from_iterable(get_ir_types(v) for v in vars(obj).values()))
    if isinstance(obj, (tuple, list)):
        return list(chain.from_iterable(get_ir_types(x) for x in obj))
    raise TypeError(f"Cannot derive IR types from {obj}")


def get_c_pointers(obj) -> List[ctypes.c_void_p]:
    if hasattr(obj, "__get_c_pointers__"):
        return obj.__get_c_pointers__()
    if isinstance(obj, SimpleNamespace):
        return list(chain.from_iterable(get_c_pointers(v) for v in vars(obj).values()))
    if isinstance(obj, (tuple, list)):
        return list(chain.from_iterable(get_c_pointers(x) for x in obj))
    raise TypeError(f"Cannot derive C pointers from {obj}")


def extract_to_ir_values(obj) -> List[ir.Value]:
    if isinstance(obj, ir.Value):
        return [obj]
    if hasattr(obj, "__extract_to_ir_values__"):
        return obj.__extract_to_ir_values__()
    if isinstance(obj, SimpleNamespace):
        return list(chain.from_iterable(extract_to_ir_values(v) for v in vars(obj).values()))
    if isinstance(obj, (tuple, list)):
        return list(chain.from_iterable(extract_to_ir_values(x) for x in obj))
    raise TypeError(f"Cannot extract IR values from {obj}")


def construct_from_ir_values(dsl_type, args, values: List[ir.Value]) -> DslType:
    if isinstance(args, SimpleNamespace):
        rebuilt = {}
        cursor = 0
        for name, value in vars(args).items():
            n = len(get_ir_types(value))
            sub_type = type(value)
            rebuilt[name] = construct_from_ir_values(sub_type, value, values[cursor : cursor + n])
            cursor += n
        if cursor != len(values):
            raise ValueError(f"SimpleNamespace expected {cursor} ir.Values, got {len(values)}")
        return SimpleNamespace(**rebuilt)
    if hasattr(dsl_type, "__construct_from_ir_values__"):
        return dsl_type.__construct_from_ir_values__(values)
    if isinstance(dsl_type, (tuple, list)):
        elems = []
        for ty, arg in zip(dsl_type, args, strict=True):
            count = len(get_ir_types(arg))
            elems.append(construct_from_ir_values(ty, arg, values[:count]))
            values = values[count:]
        return type(dsl_type)(elems)
    raise TypeError(f"Cannot construct DSL value for {dsl_type}")
