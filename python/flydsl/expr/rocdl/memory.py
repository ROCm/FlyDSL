# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Ordered global-memory and synchronization operations for AMD GPUs."""

from ..._mlir.dialects import llvm
from ..enum import AtomicOrdering
from ..meta import dsl_loc_tracing
from ..primitive import AddressSpace
from ..typing import Pointer, as_ir_value, is_generic_address_space

__all__ = [
    "global_load",
    "global_store",
    "sleep",
]

_ORDERINGS = {
    AtomicOrdering.Unordered: llvm.AtomicOrdering.unordered,
    AtomicOrdering.Monotonic: llvm.AtomicOrdering.monotonic,
    AtomicOrdering.Acquire: llvm.AtomicOrdering.acquire,
    AtomicOrdering.Release: llvm.AtomicOrdering.release,
    AtomicOrdering.AcqRel: llvm.AtomicOrdering.acq_rel,
    AtomicOrdering.SeqCst: llvm.AtomicOrdering.seq_cst,
}


def _ordering_kwargs(memory_order, syncscope):
    if memory_order == AtomicOrdering.NotAtomic:
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


def _global_ptr(ptr):
    if not isinstance(ptr, Pointer):
        raise TypeError(f"global memory operation requires an fx.Pointer, got {ptr!r}")
    if not is_generic_address_space(ptr.address_space, AddressSpace.Global):
        raise ValueError(f"global memory operation requires a global-address-space pointer, got {ptr.address_space}")
    return ptr.llvm_ptr


@dsl_loc_tracing
def global_load(
    ptr,
    dtype,
    *,
    memory_order=AtomicOrdering.NotAtomic,
    syncscope=None,
    nontemporal=False,
):
    """Load a scalar from a global ``fly.ptr``."""
    llvm_ptr = _global_ptr(ptr)
    if memory_order == AtomicOrdering.Release:
        raise ValueError(f"invalid load memory order: {memory_order}")
    try:
        result_type = dtype.ir_type
    except AttributeError as exc:
        raise TypeError("dtype must be a FlyDSL scalar type") from exc
    kwargs = _ordering_kwargs(memory_order, syncscope)
    kwargs["alignment"] = ptr.alignment
    if nontemporal:
        kwargs["nontemporal"] = True
    result = llvm.LoadOp(result_type, llvm_ptr, **kwargs).result
    return dtype(result)


@dsl_loc_tracing
def global_store(
    ptr,
    value,
    *,
    memory_order=AtomicOrdering.NotAtomic,
    syncscope=None,
    nontemporal=False,
):
    """Store a scalar to a global ``fly.ptr``."""
    llvm_ptr = _global_ptr(ptr)
    if memory_order == AtomicOrdering.Acquire:
        raise ValueError(f"invalid store memory order: {memory_order}")
    value = as_ir_value(value)
    kwargs = _ordering_kwargs(memory_order, syncscope)
    kwargs["alignment"] = ptr.alignment
    if nontemporal:
        kwargs["nontemporal"] = True
    llvm.StoreOp(value, llvm_ptr, **kwargs)


@dsl_loc_tracing
def sleep(cycles=1):
    """Pause a wavefront for a compile-time number of sleep cycles."""
    if not isinstance(cycles, int) or not 0 <= cycles <= 127:
        raise ValueError("cycles must be a compile-time integer in [0, 127]")
    llvm.InlineAsmOp(None, [], f"s_sleep {cycles}", "", has_side_effects=True)
