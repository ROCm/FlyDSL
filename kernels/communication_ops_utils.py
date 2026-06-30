# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Low-level cross-card (P2P) communication primitives for communication kernels.

These wrap LLVM-dialect global memory ops with explicit memory ordering and
syncscope -- which the high-level FlyDSL APIs (buffer_ops / Pointer) do not
expose -- so dispatch/combine can publish and observe data across cards.
"""

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl.expr import arith

__all__ = [
    "store_i32_system",
    "store_i64_global_system",
    "fence_system_acquire",
    "load_i64_global",
    "atomic_add_global_at",
]


def _to_ptr_global(v):
    """Cast an i64 address to ``!llvm.ptr<1>`` (global address space)."""
    return _llvm_d.IntToPtrOp(_llvm_d.PointerType.get(address_space=1), arith.unwrap(v)).result


def store_i32_system(addr_i64, offset, val):
    """System-scope release i32 store at ``addr_i64 + offset*4``."""
    base = arith.unwrap(addr_i64)
    off = arith.unwrap(offset)
    val_ = arith.unwrap(val)
    _i64 = ir.IntegerType.get_signless(64)
    _i32 = ir.IntegerType.get_signless(32)
    _nuw = ir.Attribute.parse("#llvm.overflow<none>")
    off64 = _llvm_d.ZExtOp(_i64, off).res if off.type == _i32 else off
    byte_off = _llvm_d.MulOp(off64, _llvm_d.ConstantOp(_i64, ir.IntegerAttr.get(_i64, 4)).result, _nuw).result
    addr = _llvm_d.AddOp(base, byte_off, _nuw).result
    gptr = _llvm_d.IntToPtrOp(_llvm_d.PointerType.get(address_space=1), addr).result
    _llvm_d.StoreOp(val_, gptr, alignment=4, ordering=_llvm_d.AtomicOrdering.release, syncscope="one-as")


def store_i64_global_system(addr_i64, val):
    """System-scope release i64 store to ``addr_i64``."""
    gptr = _to_ptr_global(addr_i64)
    _llvm_d.StoreOp(arith.unwrap(val), gptr, alignment=8, ordering=_llvm_d.AtomicOrdering.release, syncscope="one-as")


def fence_system_acquire():
    """System-scope acquire fence."""
    _llvm_d.FenceOp(_llvm_d.AtomicOrdering.acquire, syncscope="one-as")


def load_i64_global(addr_i64):
    """Relaxed global i64 load from ``addr_i64``."""
    ptr = _to_ptr_global(addr_i64)
    _i64 = ir.IntegerType.get_signless(64)
    return _llvm_d.LoadOp(_i64, ptr, alignment=8).result


def atomic_add_global_at(addr_i64, val):
    """Monotonic global ``atomic fetch-and-add``; returns the old value."""
    ptr = _to_ptr_global(addr_i64)
    return _llvm_d.AtomicRMWOp(_llvm_d.AtomicBinOp.add, ptr, arith.unwrap(val), _llvm_d.AtomicOrdering.monotonic).res
