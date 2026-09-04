# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Ordered global-memory and synchronization operations for AMD GPUs."""

from ..._mlir import ir
from ..._mlir.dialects import llvm
from ..meta import dsl_loc_tracing
from ..typing import Vector, as_ir_value
from .enum import MemoryOrder

__all__ = ["global_load", "global_store", "sleep"]

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
