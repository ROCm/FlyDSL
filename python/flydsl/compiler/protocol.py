# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from itertools import chain
from types import SimpleNamespace
from typing import Callable, List, Protocol, Tuple, runtime_checkable

from .._mlir import ir


@runtime_checkable
class DslType(Protocol):
    @classmethod
    def __construct_from_ir_values__(cls, values: List[ir.Value]) -> "DslType": ...
    def __extract_to_ir_values__(self) -> List[ir.Value]: ...


@runtime_checkable
class JitArgument(Protocol):
    def __get_ir_types__(self) -> List[ir.Type]: ...
    def __cache_signature__(self) -> object: ...
    def __c_abi_spec__(self) -> List[Tuple[type, Callable]]: ...


@runtime_checkable
class Storable(Protocol):
    @classmethod
    def __dsl_size_of__(cls) -> int: ...
    @classmethod
    def __dsl_align_of__(cls) -> int: ...
    @classmethod
    def __peek_from_ptr__(cls, ptr: ir.Value): ...
    @classmethod
    def __poke_into_ptr__(cls, ptr: ir.Value, value) -> None: ...


def get_ir_types(obj) -> List[ir.Type]:
    if isinstance(obj, ir.Value):
        return [obj.type]
    if hasattr(obj, "__get_ir_types__"):
        return obj.__get_ir_types__()
    if isinstance(obj, SimpleNamespace):
        return list(chain.from_iterable(get_ir_types(v) for v in vars(obj).values()))
    if isinstance(obj, (tuple, list)):
        return list(chain.from_iterable(get_ir_types(x) for x in obj))
    # derive IR types from extract_to_ir_values if possible
    try:
        ir_values = extract_to_ir_values(obj)
        return [v.type for v in ir_values]
    except TypeError as exc:
        raise TypeError(f"Cannot derive IR types from {obj}: {exc}") from exc


def cache_signature(obj) -> object:
    if hasattr(obj, "__cache_signature__"):
        return obj.__cache_signature__()
    if isinstance(obj, SimpleNamespace):
        return tuple((name, cache_signature(value)) for name, value in vars(obj).items())
    if isinstance(obj, (tuple, list)):
        return tuple(cache_signature(x) for x in obj)
    raise TypeError(
        f"Cannot derive cache signature for {obj!r}: type {type(obj).__name__} does not "
        "implement __cache_signature__."
    )


def c_abi_spec(obj) -> List[Tuple[type, Callable]]:
    if hasattr(obj, "__c_abi_spec__"):
        return obj.__c_abi_spec__()
    # TODO: support SimpleNamespace / tuple / list here?
    # if isinstance(obj, SimpleNamespace):
    #     return list(chain.from_iterable(c_abi_spec(v) for v in vars(obj).values()))
    # if isinstance(obj, (tuple, list)):
    #     return list(chain.from_iterable(c_abi_spec(x) for x in obj))
    raise TypeError(f"Cannot derive C-ABI spec for {obj!r}: type {type(obj).__name__}.")


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


def dsl_size_of(dsl_type) -> int:
    if hasattr(dsl_type, "__dsl_size_of__"):
        return dsl_type.__dsl_size_of__()
    raise TypeError(f"type {dsl_type} does not implement the Storable protocol")


def dsl_align_of(dsl_type) -> int:
    if hasattr(dsl_type, "__dsl_align_of__"):
        return dsl_type.__dsl_align_of__()
    raise TypeError(f"type {dsl_type} does not implement the Storable protocol")


def peek_from_ptr(dsl_type, ptr: ir.Value):
    if hasattr(dsl_type, "__peek_from_ptr__"):
        return dsl_type.__peek_from_ptr__(ptr)
    raise TypeError(f"type {dsl_type} does not implement the Storable protocol")


def poke_into_ptr(dsl_type, ptr: ir.Value, value) -> None:
    if hasattr(dsl_type, "__poke_into_ptr__"):
        dsl_type.__poke_into_ptr__(ptr, value)
        return
    raise TypeError(f"type {dsl_type} does not implement the Storable protocol")
