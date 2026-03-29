"""Compatibility memory/signal ops for custom all-reduce kernels.

This module provides a small subset of legacy `mem_ops` APIs used by
`/home/xzhu/custom_all_reduce_kernel.py`, implemented on top of current
FlyDSL/ROCDL primitives.
"""

from __future__ import annotations

from .._mlir import ir
from .._mlir.dialects import arith as _arith
from .._mlir.dialects import llvm, rocdl, scf
from .typing import T


def _i32():
    return T.i32


def _i64():
    return T.i64


def _v4i32():
    return T.i32x4


def load_i32_uncached(addr_i64):
    v = llvm.InlineAsmOp(
        _i32(),
        [addr_i64],
        "global_load_dword $0, $1, off sc1",
        "=v,v",
        has_side_effects=True,
    ).result
    rocdl.s_waitcnt(0)
    return v


def store_i32_uncached_flush(addr_i64, val_i32):
    llvm.InlineAsmOp(None, [], "buffer_wbl2 sc0 sc1", "", has_side_effects=True)
    llvm.InlineAsmOp(
        None,
        [addr_i64, val_i32],
        "global_store_dword $0, $1, off sc0 sc1",
        "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


def store_i32_uncached(addr_i64, val_i32):
    llvm.InlineAsmOp(
        None,
        [addr_i64, val_i32],
        "global_store_dword $0, $1, off sc0 sc1",
        "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


def store_i32(addr_i64, val_i32):
    llvm.InlineAsmOp(
        None,
        [addr_i64, val_i32],
        "global_store_dword $0, $1, off",
        "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


def load_v4i32(addr_i64):
    v = llvm.InlineAsmOp(
        _v4i32(),
        [addr_i64],
        "flat_load_dwordx4 $0, $1",
        "=v,v",
        has_side_effects=True,
    ).result
    rocdl.s_waitcnt(0)
    return v


def store_v4i32(addr_i64, v4i32_val):
    llvm.InlineAsmOp(
        None,
        [addr_i64, v4i32_val],
        "global_store_dwordx4 $0, $1, off",
        "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


def load_device_ptr(array_base_i64, index_i32):
    from . import arith as ea

    elem_addr = array_base_i64 + _arith.ExtUIOp(_i64(), index_i32).result * ea.constant(8, type=_i64())
    ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr"), elem_addr).result
    return llvm.LoadOp(_i64(), ptr).result


def poll_until_ge(addr_i64, target_u32):
    init_cur = load_i32_uncached(addr_i64)
    w = scf.WhileOp([_i32()], [init_cur])
    before = ir.Block.create_at_start(w.before, [_i32()])
    after = ir.Block.create_at_start(w.after, [_i32()])
    with ir.InsertionPoint(before):
        cur = before.arguments[0]
        need_wait = _arith.CmpIOp(_arith.CmpIPredicate.ult, cur, target_u32).result
        scf.ConditionOp(need_wait, [cur])
    with ir.InsertionPoint(after):
        scf.YieldOp([load_i32_uncached(addr_i64)])

