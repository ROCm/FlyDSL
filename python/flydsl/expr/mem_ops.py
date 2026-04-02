# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Low-level memory operations via inline assembly for multi-GPU kernels.

Provides wrappers around GFX942 inline assembly instructions for:

- **Uncached** loads/stores (``sc0 sc1`` — system-scope coherent,
  for cross-GPU signal buffers allocated with ``hipDeviceMallocUncached``)
- **Nontemporal** stores (``nt`` — bypasses L1/L2 cache, works on any
  memory type including regular ``hipMalloc`` / IPC-mapped addresses)
- **Cached** vector loads/stores (16-byte / ``v4i32``)
- Device-side pointer access

All functions operate on raw ``ir.Value`` (i32/i64/vector<4xi32>).

Example::

    from flydsl.expr import mem_ops

    val = mem_ops.load_i32_uncached(addr)
    mem_ops.store_i32_uncached_flush(peer_addr, flag)
    data = mem_ops.load_v4i32(data_addr)
"""

from __future__ import annotations

from .._mlir import ir
from .._mlir.dialects import arith as _arith, llvm, rocdl
from .meta import traced_op
from .typing import T


# ---------------------------------------------------------------------------
# Uncached i32 operations (system-scope coherent, for signal buffers)
# ---------------------------------------------------------------------------

@traced_op
def load_i32_uncached(addr_i64):
    """Load i32 from global address, bypassing L1 cache (system-scope).

    Emits ``global_load_dword ... sc1`` on GFX942.
    Typically used to poll cross-GPU signal buffers.
    """
    v = llvm.InlineAsmOp(
        T.i32, [addr_i64],
        "global_load_dword $0, $1, off sc1", "=v,v",
        has_side_effects=True,
    ).result
    rocdl.s_waitcnt(0)
    return v


@traced_op
def store_i32_uncached_flush(addr_i64, val_i32):
    """Store i32 with L2 flush + system-scope coherence for XGMI visibility.

    Emits ``buffer_wbl2 sc0 sc1`` followed by ``global_store_dword ... sc0 sc1``.
    Use after cached data stores (``store_v4i32``) to ensure L2 dirty lines
    reach HBM before the signal becomes visible to peer GPUs.
    """
    llvm.InlineAsmOp(None, [], "buffer_wbl2 sc0 sc1", "", has_side_effects=True)
    llvm.InlineAsmOp(
        None, [addr_i64, val_i32],
        "global_store_dword $0, $1, off sc0 sc1", "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


@traced_op
def store_i32_uncached(addr_i64, val_i32):
    """Store i32 with system-scope coherence (no L2 flush).

    Emits ``global_store_dword ... sc0 sc1``.
    Use after nontemporal data stores (``store_v4i32_nt``) which already
    bypass L2 — no ``buffer_wbl2`` is needed.
    """
    llvm.InlineAsmOp(
        None, [addr_i64, val_i32],
        "global_store_dword $0, $1, off sc0 sc1", "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


@traced_op
def store_i32(addr_i64, val_i32):
    """Store i32 to global address (normal cached store).

    Emits ``global_store_dword ... off`` with no cache coherence flags.
    Use for writes visible only to the local GPU (e.g. updating own signal).
    """
    llvm.InlineAsmOp(
        None, [addr_i64, val_i32],
        "global_store_dword $0, $1, off", "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


# ---------------------------------------------------------------------------
# v4i32 (16-byte) vector operations
# ---------------------------------------------------------------------------

@traced_op
def load_v4i32(addr_i64):
    """Load 16 bytes (``vector<4xi32>``) from global address.

    Emits ``flat_load_dwordx4``.
    """
    v = llvm.InlineAsmOp(
        T.i32x4, [addr_i64],
        "flat_load_dwordx4 $0, $1", "=v,v",
        has_side_effects=True,
    ).result
    rocdl.s_waitcnt(0)
    return v


@traced_op
def store_v4i32(addr_i64, v4i32_val):
    """Store 16 bytes (``vector<4xi32>``) to global address.

    Emits ``global_store_dwordx4 ... off``.
    """
    llvm.InlineAsmOp(
        None, [addr_i64, v4i32_val],
        "global_store_dwordx4 $0, $1, off", "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


@traced_op
def store_v4i32_nt(addr_i64, v4i32_val):
    """Store 16 bytes with nontemporal hint, bypassing L1/L2 cache.

    Emits ``flat_store_dwordx4 ... nt``.
    Suitable for large data writes across XGMI — works on any memory type
    (regular ``hipMalloc``, IPC-mapped coarse-grained memory).
    """
    llvm.InlineAsmOp(
        None, [addr_i64, v4i32_val],
        "flat_store_dwordx4 $0, $1 nt", "v,v",
        has_side_effects=True,
    )
    rocdl.s_waitcnt(0)


# ---------------------------------------------------------------------------
# Pointer helpers
# ---------------------------------------------------------------------------

@traced_op
def load_device_ptr(array_base_i64, index):
    """Load an i64 pointer from a device-side pointer array.

    Computes ``base + index * 8``, casts to ``!llvm.ptr``, and loads i64.

    Args:
        array_base_i64: Base address of the pointer array (i64).
        index: Array index (i32 or i64).
    """
    from . import arith as ea

    i64 = T.i64
    if hasattr(index, 'type') and isinstance(index.type, ir.IntegerType) and index.type.width == 32:
        index = _arith.ExtUIOp(i64, index).result
    elem_addr = array_base_i64 + index * ea.constant(8, type=i64)
    ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr"), elem_addr).result
    return llvm.LoadOp(i64, ptr).result


@traced_op
def invalidate_l1():
    """Invalidate L1 scalar cache (``buffer_inv sc1``).

    Use inside a polling loop after a remote-visible load to discard stale
    L1 cache lines so the next iteration sees fresh data from L2/HBM.
    """
    llvm.InlineAsmOp(None, [], "buffer_inv sc1", "", has_side_effects=True)


__all__ = [
    # Uncached i32 (system-scope coherent)
    "load_i32_uncached",
    "store_i32_uncached_flush",
    "store_i32_uncached",
    "store_i32",
    # v4i32 (16-byte vector)
    "load_v4i32",
    "store_v4i32",
    "store_v4i32_nt",
    # Cache control
    "invalidate_l1",
    # Pointer helpers
    "load_device_ptr",
]
