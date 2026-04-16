"""
FlyDSL low-level GPU ops for dispatch/combine kernels.

These wrap ROCDL and LLVM dialect operations that are not exposed by the
standard ``flydsl.expr`` (``fx.*``) interface but are required for warp-level
communication primitives used in dispatch/combine.

Usage inside ``@flyc.kernel``::

    from flydsl.expr.lowlevel import ballot_i64, readlane, ptrtoint

    @flyc.kernel
    def my_kernel(A: fx.Tensor):
        tid  = fx.thread_idx.x
        lane = tid & 63
        cond = lane < 32
        mask = ballot_i64(cond)     # rocdl.ballot.i64
        tid0 = readlane(tid, 0)     # rocdl.readlane (broadcast lane-0)
        addr = ptrtoint(A)          # llvm.ptrtoint → i64
"""

from __future__ import annotations

from typing import Any

from .._mlir import ir
from .._mlir.dialects import llvm
from .._mlir.ir import (
    IntegerAttr,
    IntegerType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _i1()  -> ir.Type: return IntegerType.get_signless(1)
def _i32() -> ir.Type: return IntegerType.get_signless(32)
def _i64() -> ir.Type: return IntegerType.get_signless(64)
def _ptr() -> ir.Type: return llvm.PointerType.get()


def _unwrap(v: Any) -> ir.Value:
    """Extract raw ir.Value from a DSL object or ir.Value."""
    if isinstance(v, ir.Value):
        return v
    if hasattr(v, "__fly_values__"):
        vals = v.__fly_values__()
        if len(vals) == 1:
            return vals[0]
        raise ValueError(f"Expected 1 ir.Value, got {len(vals)}")
    if isinstance(v, int):
        c = llvm.ConstantOp(_i32(), IntegerAttr.get(_i32(), v)).result
        return c
    raise TypeError(f"Cannot convert {type(v).__name__} to ir.Value")


def _const_i32(val: int) -> ir.Value:
    return llvm.ConstantOp(_i32(), IntegerAttr.get(_i32(), val)).result


def _const_i64(val: int) -> ir.Value:
    return llvm.ConstantOp(_i64(), IntegerAttr.get(_i64(), val)).result


# ---------------------------------------------------------------------------
# Pointer ↔ integer conversions (LLVM)
# ---------------------------------------------------------------------------
def ptrtoint(ptr: Any) -> ir.Value:
    """Convert a tensor/pointer to i64 address.

    Handles:
    - ``fx.Tensor`` (wraps ``!fly.memref<...>``) → uses
      ``fly.extract_aligned_pointer_as_index`` first, then ``arith.index_cast``
    - LLVM pointer (``!llvm.ptr`` or ``!llvm.ptr<N>``) → direct ``llvm.ptrtoint``

    Returns:
        i64 integer address.
    """
    ptr_val = _unwrap(ptr)
    type_str = str(ptr_val.type)

    if "fly.memref" in type_str or "memref<" in type_str:
        # Extract aligned pointer as index, then cast index → i64
        from .._mlir.dialects import _fly_ops_gen as _fly
        from .._mlir.dialects import arith as _arith
        from .._mlir.ir import IndexType
        idx_val = _fly.ExtractAlignedPointerAsIndexOp(IndexType.get(), ptr_val).result
        return _arith.IndexCastUIOp(_i64(), idx_val).result

    if "llvm.ptr" in type_str:
        return llvm.PtrToIntOp(_i64(), ptr_val).result

    raise TypeError(
        f"ptrtoint: unsupported value type '{ptr_val.type}'. "
        f"Expected fly.memref or llvm.ptr."
    )


def inttoptr(addr: Any) -> ir.Value:
    """``llvm.inttoptr`` i64 → ptr — convert an integer address to a pointer.

    Returns:
        Opaque ``!llvm.ptr`` value.
    """
    addr_val = _unwrap(addr)
    return llvm.IntToPtrOp(_ptr(), addr_val).result


# ---------------------------------------------------------------------------
# Global memory load/store (addrspace 1)
# ---------------------------------------------------------------------------
def load_global(addr_i64: Any, *, dtype=None, nt: bool = False) -> ir.Value:
    """Load from a single global (addrspace 1) address.

    Args:
        dtype: MLIR type to load (default i32). Pass e.g.
               ``VectorType.get([4], i32)`` for 128-bit vector loads.
        nt:    Nontemporal hint — bypasses L2 cache.
               WARNING: do NOT use nt for XGMI P2P reads (causes ~13x slowdown).
    """
    if dtype is None:
        dtype = _i32()
    kw = {}
    if nt:
        kw["nontemporal"] = ir.BoolAttr.get(True)
    gptr = _to_ptr_global(_unwrap(addr_i64))
    return llvm.LoadOp(dtype, gptr, alignment=4, **kw).result


def store_global(val: Any, addr_i64: Any, *, nt: bool = False) -> None:
    """Store to a single global (addrspace 1) address.

    Args:
        nt: Nontemporal hint — ``global_store nt`` bypasses L2 cache.
    """
    val_ = _unwrap(val)
    kw = {}
    if nt:
        kw["nontemporal"] = ir.BoolAttr.get(True)
    gptr = _to_ptr_global(_unwrap(addr_i64))
    llvm.StoreOp(val_, gptr, alignment=4, **kw)


def store_i64_system(addr_i64: Any, val: Any) -> None:
    """Atomic store i64 with system scope using flat (generic) address space.

    Equivalent to mori's ``AtomicStoreRelaxedSystem<uint64_t>``:
    ``__hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)``

    Uses addrspace(0) (flat/generic) rather than addrspace(1) (global).
    In AMDGPU, HIP raw pointers use the flat address space, generating
    ``flat_store_b64`` instructions. addrspace(1) generates ``global_store_dwordx2``.
    For P2P (XGMI) addresses mapped via hipIpcOpenMemHandle, ``flat_store``
    is the correct instruction matching mori's ``__hip_atomic_store`` behavior.

    Args:
        addr_i64: i64 integer address (P2P-mapped remote GPU memory via XGMI).
        val:      i64 value to store.
    """
    addr_val = _unwrap(addr_i64)
    val_val  = _unwrap(val)
    # addrspace(0) = flat/generic: matches HIP's raw pointer behavior for P2P writes
    ptr_flat_ty = llvm.PointerType.get(address_space=0)
    gptr = llvm.IntToPtrOp(ptr_flat_ty, addr_val).result
    llvm.StoreOp(
        val_val, gptr,
        alignment=8,
        ordering=llvm.AtomicOrdering.monotonic,
        syncscope="one-as",
    )


def store_i32_system(addr_i64: Any, offset: Any, val: Any) -> None:
    """Atomic store i32 with system scope (``syncscope("one-as") monotonic``).

    Equivalent to mori's ``AtomicStoreRelaxedSystem``:
    ``__hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)``

    Uses global pointer (addrspace 1) with monotonic ordering and one-as
    syncscope so the store is immediately visible across all GPUs via XGMI.
    """
    base = _unwrap(addr_i64)
    off  = _unwrap(offset)
    val_ = _unwrap(val)
    off64 = llvm.ZExtOp(_i64(), off).res if off.type == _i32() else off
    byte_off = llvm.MulOp(off64, _const_i64(4), ir.Attribute.parse("#llvm.overflow<none>")).result
    addr = llvm.AddOp(base, byte_off, ir.Attribute.parse("#llvm.overflow<none>")).result
    ptr_global_ty = llvm.PointerType.get(address_space=1)
    gptr = llvm.IntToPtrOp(ptr_global_ty, addr).result
    llvm.StoreOp(
        val_, gptr,
        alignment=4,
        ordering=llvm.AtomicOrdering.monotonic,
        syncscope="one-as",
    )


# ---------------------------------------------------------------------------
# Vectorized memory ops (128-bit load/store for token data)
# ---------------------------------------------------------------------------
def load_v4i32(ptr: Any) -> ir.Value:
    """Load 128-bit (4 × i32) vector from *ptr*.

    Args:
        ptr: Pointer value (any ptr type or i64 address).

    Returns:
        ``vector<4xi32>`` value.
    """
    from .._mlir.ir import VectorType
    v4i32 = VectorType.get([4], _i32())
    ptr_val = _to_ptr(_unwrap(ptr))
    return llvm.LoadOp(v4i32, ptr_val, alignment=4).result


def store_v4i32(vec: Any, ptr: Any) -> None:
    """Store 128-bit (4 × i32) vector *vec* to *ptr* (flat address space).

    Args:
        vec: ``vector<4xi32>`` value.
        ptr: Destination pointer (flat).
    """
    vec_val = _unwrap(vec)
    ptr_val = _unwrap(ptr)
    llvm.StoreOp(vec_val, ptr_val, alignment=4)


# ---------------------------------------------------------------------------
# Address computation helpers
# ---------------------------------------------------------------------------
def _to_ptr(v: ir.Value) -> ir.Value:
    """Convert i64 to LLVM ptr, or pass through if already a ptr."""
    type_str = str(v.type)
    if "i64" in type_str or "index" in type_str:
        return llvm.IntToPtrOp(_ptr(), v).result
    return v


def _to_ptr_global(v: ir.Value) -> ir.Value:
    """Convert i64 to LLVM ptr addrspace(1) (global address space)."""
    ptr_g = llvm.PointerType.get(address_space=1)
    return llvm.IntToPtrOp(ptr_g, _unwrap(v)).result


def _type_align(dtype) -> int:
    """Return natural alignment in bytes for an MLIR scalar/vector type."""
    if dtype == _i64():
        return 8
    return 4


def _addr_at(base_i64: Any, offset: Any, elem_bytes: int) -> ir.Value:
    """Compute ``base_i64 + offset * elem_bytes`` as i64."""
    base = _unwrap(base_i64)
    off  = _unwrap(offset)
    off64 = llvm.ZExtOp(_i64(), off).res if off.type == _i32() else off
    _nuw = ir.Attribute.parse("#llvm.overflow<none>")
    byte_off = llvm.MulOp(off64, _const_i64(elem_bytes), _nuw).result
    return llvm.AddOp(base, byte_off, _nuw).result


def load_global_at(base_i64: Any, offset: Any, *, dtype=None, nt: bool = False) -> ir.Value:
    """Load from ``base + offset * elem_bytes`` using global addrspace(1).

    Args:
        dtype: MLIR element type (default i32). Determines offset scale and alignment.
        nt:    Nontemporal hint — ``global_load nt`` bypasses L2 cache.
    """
    if dtype is None:
        dtype = _i32()
    eb = _type_align(dtype)
    addr = _addr_at(base_i64, offset, eb)
    kw = {}
    if nt:
        kw["nontemporal"] = ir.BoolAttr.get(True)
    return llvm.LoadOp(dtype, _to_ptr_global(addr), alignment=eb, **kw).result


def store_global_at(base_i64: Any, offset: Any, val: Any, *, nt: bool = False) -> None:
    """Store to ``base + offset * elem_bytes`` using global addrspace(1).

    Element size is inferred from *val*'s type.
    """
    val_ = _unwrap(val)
    eb = _type_align(val_.type)
    addr = _addr_at(base_i64, offset, eb)
    kw = {}
    if nt:
        kw["nontemporal"] = ir.BoolAttr.get(True)
    llvm.StoreOp(val_, _to_ptr_global(addr), alignment=eb, **kw)


def atomic_add_global_at(addr_i64: Any, val: Any) -> ir.Value:
    """Atomic fetch-and-add on global (addrspace 1) memory, returns old value.

    The value type (i32/i64) is inferred from *val*.
    AMDGPU generates ``global_atomic_add`` (i32) or
    ``global_atomic_add_u64`` (i64).
    """
    addr = _unwrap(addr_i64)
    val_ = _unwrap(val)
    ptr  = _to_ptr_global(addr)
    return llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        ptr,
        val_,
        llvm.AtomicOrdering.monotonic,
    ).res


def store_i64_global_system(addr_i64: Any, val: Any) -> None:
    """Atomic store i64 with system scope using global addrspace(1).

    Like store_i64_system but generates global_store_dwordx2 instead of
    flat_store_b64.  For P2P (XGMI) cross-device barrier signaling.
    """
    addr_val = _unwrap(addr_i64)
    val_val  = _unwrap(val)
    ptr_global_ty = llvm.PointerType.get(address_space=1)
    gptr = llvm.IntToPtrOp(ptr_global_ty, addr_val).result
    llvm.StoreOp(
        val_val, gptr,
        alignment=8,
        ordering=llvm.AtomicOrdering.monotonic,
        syncscope="one-as",
    )


def load_at(base_i64: Any, offset: Any, *, dtype=None) -> ir.Value:
    """Load from ``base + offset * elem_bytes`` (flat addrspace).

    Args:
        dtype: MLIR element type (default i32). Determines offset scale and alignment.
    """
    if dtype is None:
        dtype = _i32()
    eb = _type_align(dtype)
    addr = _addr_at(base_i64, offset, eb)
    return llvm.LoadOp(dtype, _to_ptr(addr), alignment=eb).result


def idx_to_i32(v: Any) -> ir.Value:
    """Cast MLIR ``index``-typed induction variable to ``i32``.

    ``scf.ForOp`` induction variables have ``index`` type.  Use this at the
    top of a dynamic loop body to get a regular ``i32`` for arithmetic::

        for i_ix in range(as_index(start), as_index(stop), as_index(step)):
            i = idx_to_i32(i_ix)
            # Use i in i32 arithmetic

    Returns an ``ArithValue`` so that arithmetic operators work correctly.
    """
    from .._mlir.dialects import arith as _arith
    from .._mlir.ir import Value
    v_ = _unwrap(v)
    if v_.type == _i32():
        result = v_
    else:
        result = _arith.IndexCastUIOp(_i32(), v_).result
    # Wrap as ArithValue so that Python arithmetic operators (//, +, etc.) work
    # correctly when combined with other ArithValues.
    try:
        from ..expr.arith import ArithValue
        return ArithValue(result)
    except Exception:
        try:
            from ..expr.utils.arith import ArithValue
            return ArithValue(result)
        except Exception:
            return result


def as_index(v: Any) -> ir.Value:
    """Cast i32/i64 MLIR value to ``index`` type for use in ``range()`` loops.

    FlyDSL's ``scf_range`` calls ``scf.ForOp(start, stop, step)`` which
    requires ``index``-typed operands.  Use this when loop bounds are computed
    from dynamic i32/i64 values (e.g. ``lane * 4``).
    """
    from .._mlir.dialects import arith as _arith
    from .._mlir.ir import IndexType
    if isinstance(v, int):
        return _arith.ConstantOp(IndexType.get(), v).result
    v_ = _unwrap(v)
    if v_.type == IndexType.get():
        return v_
    return _arith.IndexCastOp(IndexType.get(), v_).result


