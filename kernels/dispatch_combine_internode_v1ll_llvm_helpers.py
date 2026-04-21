# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""LLVM global-memory helpers for internode V1LL FlyDSL kernels (raw ``i64`` addresses).

``flydsl.expr`` is reserved for the public DSL. Raw ``data_ptr()``-style kernels
should keep ``IntToPtr`` / ``llvm.load`` / ``llvm.store`` glue next to the
kernels—same idea as ``dispatch_combine_intranode_kernel.py`` (``_lv_unwrap``,
``_to_ptr_global``, etc.), factored here only because two V1LL modules share it.
"""
from __future__ import annotations

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as arith_d
from flydsl._mlir.dialects import llvm as llvm_d
from flydsl._mlir.extras import types as T
from flydsl._mlir.ir import IntegerAttr, IntegerType
from flydsl.expr import arith as fx_arith
from flydsl.expr.numeric import Int64
from flydsl.expr.utils.arith import ArithValue

_i32 = lambda: IntegerType.get_signless(32)
_i64 = lambda: IntegerType.get_signless(64)
_ptr1 = lambda: llvm_d.PointerType.get(address_space=1)
_nuw = lambda: ir.Attribute.parse("#llvm.overflow<none>")


def _unwrap(v):
    if isinstance(v, ir.Value):
        return v
    if isinstance(v, ArithValue):
        return v
    if hasattr(v, "__fly_values__"):
        vals = v.__fly_values__()
        if len(vals) != 1:
            raise ValueError(f"expected a single SSA value, got {len(vals)}")
        x = vals[0]
        if isinstance(x, ArithValue):
            return x
        if isinstance(x, ir.Value):
            return x
        raise TypeError(f"unsupported fly value slot type: {type(x)}")
    if isinstance(v, int) and not isinstance(v, bool):
        return llvm_d.ConstantOp(_i32(), IntegerAttr.get(_i32(), int(v))).result
    raise TypeError(f"cannot unwrap {type(v).__name__} to ir.Value")


def _to_i64_addr(v) -> ir.Value:
    x = _unwrap(v)
    if x.type == _i64():
        return x
    if x.type == _i32():
        return arith_d.ExtUIOp(_i64(), x).result
    raise TypeError(f"expected i64 address (or i32), got {x.type}")


def _to_ptr_global(addr_i64: ir.Value) -> ir.Value:
    return llvm_d.IntToPtrOp(_ptr1(), addr_i64).result


def const_i32(n: int) -> ir.Value:
    ty = _i32()
    return arith_d.ConstantOp(ty, IntegerAttr.get(ty, int(n))).result


def const_i64(n: int) -> Int64:
    ty = _i64()
    return Int64(arith_d.ConstantOp(ty, IntegerAttr.get(ty, int(n))).result)


def zext_i32_to_i64(v) -> Int64:
    return Int64(arith_d.ExtUIOp(_i64(), _unwrap(v)).result)


def as_index(val):
    v = _unwrap(val)
    if isinstance(v, int) and not isinstance(v, bool):
        return arith_d.ConstantOp(ir.IndexType.get(), int(v)).result
    if v.type == ir.IndexType.get():
        return v
    return arith_d.IndexCastOp(ir.IndexType.get(), v).result


def idx_to_i32(iv):
    v = _unwrap(iv)
    ty = _i32()
    if v.type == ty:
        return v
    if v.type == ir.IndexType.get():
        return arith_d.IndexCastOp(ty, v).result
    raise TypeError(f"idx_to_i32: expected index or i32, got {v.type}")


def icmp_eq_i32(a, b) -> ir.Value:
    return arith_d.CmpIOp(arith_d.CmpIPredicate.eq, _unwrap(a), _unwrap(b)).result


def icmp_ult_i32(a, b) -> ir.Value:
    return arith_d.CmpIOp(arith_d.CmpIPredicate.ult, _unwrap(a), _unwrap(b)).result


def select_i32(cond, a, b) -> ir.Value:
    return arith_d.SelectOp(_unwrap(cond), _unwrap(a), _unwrap(b)).result


def divui(lhs, rhs):
    return fx_arith.divui(lhs, rhs)


def remui(lhs, rhs):
    return fx_arith.remui(lhs, rhs)


def load_i32_global_at(base, idx) -> ir.Value:
    base_i = _to_i64_addr(base)
    off_i = _unwrap(idx)
    off64 = arith_d.ExtUIOp(_i64(), off_i).result if off_i.type != _i64() else off_i
    byte_off = llvm_d.MulOp(
        off64,
        llvm_d.ConstantOp(_i64(), IntegerAttr.get(_i64(), 4)).result,
        _nuw(),
    ).result
    addr = llvm_d.AddOp(base_i, byte_off, _nuw()).result
    ptr = _to_ptr_global(addr)
    return llvm_d.LoadOp(_i32(), ptr, alignment=4).result


def load_f32_global_at(base, idx) -> ir.Value:
    base_i = _to_i64_addr(base)
    off_i = _unwrap(idx)
    off64 = arith_d.ExtUIOp(_i64(), off_i).result if off_i.type != _i64() else off_i
    byte_off = llvm_d.MulOp(
        off64,
        llvm_d.ConstantOp(_i64(), IntegerAttr.get(_i64(), 4)).result,
        _nuw(),
    ).result
    addr = llvm_d.AddOp(base_i, byte_off, _nuw()).result
    ptr = _to_ptr_global(addr)
    return llvm_d.LoadOp(T.f32(), ptr, alignment=4).result


def load_i32_global(addr) -> ir.Value:
    ptr = _to_ptr_global(_to_i64_addr(addr))
    return llvm_d.LoadOp(_i32(), ptr, alignment=4).result


def store_i32_global_at(base, idx, val) -> None:
    base_i = _to_i64_addr(base)
    off_i = _unwrap(idx)
    off64 = arith_d.ExtUIOp(_i64(), off_i).result if off_i.type != _i64() else off_i
    byte_off = llvm_d.MulOp(
        off64,
        llvm_d.ConstantOp(_i64(), IntegerAttr.get(_i64(), 4)).result,
        _nuw(),
    ).result
    addr = llvm_d.AddOp(base_i, byte_off, _nuw()).result
    ptr = _to_ptr_global(addr)
    llvm_d.StoreOp(_unwrap(val), ptr, alignment=4)


def store_i32_global(base, idx, val) -> None:
    store_i32_global_at(base, idx, val)


def load_v4i32_global(addr) -> ir.Value:
    addr_i = _to_i64_addr(addr)
    vec_ty = ir.VectorType.get([4], _i32())
    ptr = _to_ptr_global(addr_i)
    return llvm_d.LoadOp(vec_ty, ptr, alignment=16).result


def store_v4i32_global(vec, addr) -> None:
    addr_i = _to_i64_addr(addr)
    ptr = _to_ptr_global(addr_i)
    llvm_d.StoreOp(_unwrap(vec), ptr, alignment=16)
