"""Signal buffer and uncached memory operations for multi-GPU allreduce.

Provides high-level wrappers around GFX942 inline assembly for:
- Uncached global loads/stores (signal buffers, bypasses L1/TCP cache)
- XGMI cross-GPU stores with L2 flush
- 16-byte global loads/stores (vectorized data movement)
- Device-side pointer array access
- Spin-wait synchronization on signal buffers

All functions operate on raw ``ir.Value`` (with ArithValue operator overloading
enabled via ``from flydsl.expr import arith``).
"""

from __future__ import annotations

from .._mlir import ir
from .._mlir.dialects import llvm, rocdl, arith as _arith, scf
from .typing import T


def _i32() -> ir.Type:
    return T.i32


def _i64() -> ir.Type:
    return T.i64


def _v4i32() -> ir.Type:
    return T.i32x4


# ---------------------------------------------------------------------------
# Uncached global memory primitives
# ---------------------------------------------------------------------------

def ld_uncached_u32(addr_i64):
    """Load u32 from uncached global address (signal buffer, bypasses L1).

    Uses ``global_load_dword ... sc1`` on GFX942.
    """
    v = llvm.InlineAsmOp(
        _i32(), [addr_i64],
        "global_load_dword $0, $1, off sc1", "=v,v",
        has_side_effects=True,
    ).result
    rocdl.s_waitcnt(0)
    return v


def st_xgmi_u32(addr_i64, val_i32):
    """Store u32 to peer GPU signal buffer, flushing L2 for XGMI visibility.

    Issues ``buffer_wbl2 sc0 sc1`` before the store for cache coherence.
    """
    llvm.InlineAsmOp(None, [], "buffer_wbl2 sc0 sc1", "", has_side_effects=True)
    llvm.InlineAsmOp(
        None, [addr_i64, val_i32],
        "global_store_dword $0, $1, off sc0 sc1", "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


def st_local_u32(addr_i64, val_i32):
    """Store u32 to local (same-GPU) signal buffer, no XGMI flush needed."""
    llvm.InlineAsmOp(
        None, [addr_i64, val_i32],
        "global_store_dword $0, $1, off", "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


def ld_global_16b(addr_i64):
    """Load 16 bytes (vector<4xi32>) from global address."""
    v = llvm.InlineAsmOp(
        _v4i32(), [addr_i64],
        "flat_load_dwordx4 $0, $1", "=v,v",
        has_side_effects=True,
    ).result
    rocdl.s_waitcnt(0)
    return v


def st_global_16b(addr_i64, v4i32_val):
    """Store 16 bytes (vector<4xi32>) to global address."""
    llvm.InlineAsmOp(
        None, [addr_i64, v4i32_val],
        "global_store_dwordx4 $0, $1, off", "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


# ---------------------------------------------------------------------------
# Pointer arithmetic helpers
# ---------------------------------------------------------------------------

def load_ptr_from_array(array_base_i64, index_i32):
    """Load an i64 pointer from a device-side array at *index_i32*.

    Computes ``base + index * 8``, casts to ``!llvm.ptr``, and loads i64.
    """
    from . import arith as ea

    i64 = _i64()
    elem_addr = array_base_i64 + _arith.ExtUIOp(i64, index_i32).result * ea.constant(8, type=i64)
    ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr"), elem_addr).result
    return llvm.LoadOp(i64, ptr).result


def select_by_lane(lane_i32, vals_i64):
    """Select one of *vals_i64[0..N]* by *lane_i32* via chained ``arith.select``."""
    from . import arith as ea

    i32 = _i32()
    out = vals_i64[0]
    for i in range(1, len(vals_i64)):
        pred = _arith.CmpIOp(_arith.CmpIPredicate.eq, lane_i32, ea.constant(i, type=i32)).result
        out = _arith.SelectOp(pred, vals_i64[i], out).result
    return out


# ---------------------------------------------------------------------------
# Signal synchronization
# ---------------------------------------------------------------------------

def spin_wait_ge(addr_i64, target_u32):
    """Spin-wait until ``*addr >= target`` (uncachable signal buffer)."""
    i32 = _i32()
    init_cur = ld_uncached_u32(addr_i64)
    w = scf.WhileOp([i32], [init_cur])
    before = ir.Block.create_at_start(w.before, [i32])
    after = ir.Block.create_at_start(w.after, [i32])
    with ir.InsertionPoint(before):
        cur = before.arguments[0]
        need_wait = _arith.CmpIOp(_arith.CmpIPredicate.ult, cur, target_u32).result
        scf.ConditionOp(need_wait, [cur])
    with ir.InsertionPoint(after):
        scf.YieldOp([ld_uncached_u32(addr_i64)])
