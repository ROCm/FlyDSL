# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Ordered global-memory and synchronization operations for AMD GPUs."""

from ..._mlir import ir
from ..._mlir.dialects import llvm
from ..meta import dsl_loc_tracing
from ..numeric import Float, Integer, Numeric
from ..typing import Vector, as_ir_value
from .enum import MemoryOrder

__all__ = [
    "atomic_fetch_add",
    "global_load",
    "global_store",
    "memory_fence",
    "sleep",
]

_ORDERINGS = {
    MemoryOrder.Unordered: llvm.AtomicOrdering.unordered,
    MemoryOrder.Monotonic: llvm.AtomicOrdering.monotonic,
    MemoryOrder.Acquire: llvm.AtomicOrdering.acquire,
    MemoryOrder.Release: llvm.AtomicOrdering.release,
    MemoryOrder.SequentiallyConsistent: llvm.AtomicOrdering.seq_cst,
}


def _ordering_kwargs(memory_order, syncscope):
    if memory_order == MemoryOrder.NotAtomic:
        if syncscope is not None:
            raise ValueError("syncscope requires an atomic memory order")
        return {}
    try:
        ordering = _ORDERINGS[memory_order]
    except KeyError as exc:
        raise ValueError(f"unsupported memory order: {memory_order!r}") from exc
    kwargs = {"ordering": ordering}
    if syncscope is not None:
        kwargs["syncscope"] = syncscope
    return kwargs


def _global_ptr(address):
    address = as_ir_value(address)
    if not isinstance(address.type, ir.IntegerType) or address.type.width != 64:
        raise TypeError("global-memory address must be an i64 value")
    return llvm.IntToPtrOp(llvm.PointerType.get(address_space=1), address).result


def _atomic_ordering(memory_order):
    if memory_order in (MemoryOrder.NotAtomic, MemoryOrder.Unordered):
        raise ValueError(f"invalid atomic memory order: {memory_order}")
    try:
        return _ORDERINGS[memory_order]
    except KeyError as exc:
        raise ValueError(f"unsupported memory order: {memory_order!r}") from exc


@dsl_loc_tracing
def atomic_fetch_add(
    address,
    value,
    *,
    memory_order=MemoryOrder.Monotonic,
    syncscope=None,
    alignment=None,
):
    """Atomically add ``value`` at an i64 global address and return the old value."""
    value = as_ir_value(value)
    try:
        dtype = Numeric.from_ir_type(value.type)
    except (TypeError, ValueError) as exc:
        raise TypeError("atomic value must be a FlyDSL integer or floating-point scalar") from exc
    if issubclass(dtype, Integer):
        bin_op = llvm.AtomicBinOp.add
    elif issubclass(dtype, Float):
        bin_op = llvm.AtomicBinOp.fadd
    else:
        raise TypeError("atomic value must be a FlyDSL integer or floating-point scalar")
    kwargs = {}
    if syncscope is not None:
        kwargs["syncscope"] = syncscope
    if alignment is not None:
        kwargs["alignment"] = alignment
    result = llvm.AtomicRMWOp(
        bin_op,
        _global_ptr(address),
        value,
        _atomic_ordering(memory_order),
        **kwargs,
    ).result
    return dtype(result)


@dsl_loc_tracing
def memory_fence(
    memory_order=MemoryOrder.SequentiallyConsistent,
    *,
    syncscope=None,
):
    """Synchronize memory accesses by the calling thread at the requested scope."""
    kwargs = {}
    if syncscope is not None:
        kwargs["syncscope"] = syncscope
    llvm.FenceOp(_atomic_ordering(memory_order), **kwargs)


@dsl_loc_tracing
def global_load(
    address,
    dtype,
    *,
    vector_width=1,
    alignment=None,
    memory_order=MemoryOrder.NotAtomic,
    syncscope=None,
    nontemporal=False,
):
    """Load a scalar or flat vector from an i64 global-memory address."""
    if not isinstance(vector_width, int) or vector_width < 1:
        raise ValueError("vector_width must be a positive compile-time integer")
    if memory_order == MemoryOrder.Release:
        raise ValueError(f"invalid load memory order: {memory_order}")
    try:
        element_type = dtype.ir_type
    except AttributeError as exc:
        raise TypeError("dtype must be a FlyDSL scalar type") from exc
    result_type = element_type if vector_width == 1 else ir.VectorType.get([vector_width], element_type)
    kwargs = _ordering_kwargs(memory_order, syncscope)
    if alignment is not None:
        kwargs["alignment"] = alignment
    if nontemporal:
        kwargs["nontemporal"] = True
    result = llvm.LoadOp(result_type, _global_ptr(address), **kwargs).result
    return Vector(result) if vector_width > 1 else dtype(result)


@dsl_loc_tracing
def global_store(
    address,
    value,
    *,
    alignment=None,
    memory_order=MemoryOrder.NotAtomic,
    syncscope=None,
    nontemporal=False,
):
    """Store a scalar or flat vector to an i64 global-memory address."""
    if memory_order == MemoryOrder.Acquire:
        raise ValueError(f"invalid store memory order: {memory_order}")
    kwargs = _ordering_kwargs(memory_order, syncscope)
    if alignment is not None:
        kwargs["alignment"] = alignment
    if nontemporal:
        kwargs["nontemporal"] = True
    llvm.StoreOp(as_ir_value(value), _global_ptr(address), **kwargs)


@dsl_loc_tracing
def sleep(cycles=1):
    """Pause a wavefront for a compile-time number of sleep cycles."""
    if not isinstance(cycles, int) or not 0 <= cycles <= 127:
        raise ValueError("cycles must be a compile-time integer in [0, 127]")
    llvm.InlineAsmOp(None, [], f"s_sleep {cycles}", "", has_side_effects=True)
