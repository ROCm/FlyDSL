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

import ctypes
from itertools import chain
from typing import List, Protocol, runtime_checkable

from .._mlir import ir


@runtime_checkable
class DslType(Protocol):
    @classmethod
    def __fly_construct__(cls, values: List[ir.Value]) -> "DslType": ...
    def __fly_values__(self) -> List[ir.Value]: ...


@runtime_checkable
class JitArgument(Protocol):
    def __fly_types__(self) -> List[ir.Type]: ...
    def __fly_ptrs__(self) -> List[ctypes.c_void_p]: ...


def fly_types(obj) -> List[ir.Type]:
    if isinstance(obj, ir.Value):
        return [obj.type]
    if hasattr(obj, "__fly_types__"):
        return obj.__fly_types__()
    if hasattr(obj, "__fly_values__"):
        return [v.type for v in obj.__fly_values__()]
    if isinstance(obj, (tuple, list)):
        return list(chain.from_iterable(fly_types(x) for x in obj))
    raise TypeError(f"Cannot derive IR types from {obj}")


def fly_pointers(obj) -> List[ctypes.c_void_p]:
    if hasattr(obj, "__fly_ptrs__"):
        return obj.__fly_ptrs__()
    if isinstance(obj, (tuple, list)):
        return list(chain.from_iterable(fly_pointers(x) for x in obj))
    raise TypeError(f"Cannot derive C pointers from {obj}")


def fly_values(obj) -> List[ir.Value]:
    if isinstance(obj, ir.Value):
        return [obj]
    if hasattr(obj, "__fly_values__"):
        return obj.__fly_values__()
    if isinstance(obj, (tuple, list)):
        return list(chain.from_iterable(fly_values(x) for x in obj))
    raise TypeError(f"Cannot extract IR values from {obj}")


def fly_construct(dsl_type, args, values: List[ir.Value]) -> DslType:
    if hasattr(dsl_type, "__fly_construct__"):
        return dsl_type.__fly_construct__(values)
    if isinstance(dsl_type, (tuple, list)):
        elems = []
        for ty, arg in zip(dsl_type, args, strict=True):
            count = len(fly_types(arg))
            elems.append(fly_construct(ty, arg, values[:count]))
            values = values[count:]
        return type(dsl_type)(elems)
    raise TypeError(f"Cannot construct DSL value for {dsl_type}")
