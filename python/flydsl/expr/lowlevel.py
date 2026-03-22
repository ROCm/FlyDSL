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
from .._mlir.dialects import llvm, rocdl
from .._mlir.ir import (
    DenseI32ArrayAttr,
    IntegerAttr,
    IntegerType,
    InsertionPoint,
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
# Warp voting / shuffles (ROCDL)
# ---------------------------------------------------------------------------
def ballot_i64(cond: Any) -> ir.Value:
    """``rocdl.ballot.i64(cond)`` — warp ballot, returns 64-bit lane mask.

    Args:
        cond: i1 condition value (one per lane).

    Returns:
        i64 bitmask: bit ``lane`` is set iff ``cond[lane]`` is true.
    """
    pred = _unwrap(cond)
    # Ensure i1
    if pred.type != _i1():
        pred = llvm.TruncOp(_i1(), pred).result
    return rocdl.BallotOp(_i64(), pred).result


def readlane(val: Any, lane: Any) -> ir.Value:
    """``rocdl.readlane(val, lane)`` — read a value from a specific warp lane.

    Commonly used to broadcast lane-0's value to all lanes::

        dest_tok_all = readlane(dest_tok, 0)

    Args:
        val:  The value whose lane-*lane* register content is read.
        lane: Source lane index (i32).

    Returns:
        Same type as *val*, but holding lane-*lane*'s value in all lanes.
    """
    src  = _unwrap(val)
    lane_v = _const_i32(lane) if isinstance(lane, int) else _unwrap(lane)
    return rocdl.ReadlaneOp(_i32(), src, lane_v).result


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
# Atomic operations (LLVM)
# ---------------------------------------------------------------------------
def _to_i32_value(v: Any) -> ir.Value:
    """Convert a value to an i32 ir.Value (for divui/remui)."""
    if isinstance(v, int):
        from .._mlir.dialects import arith as _arith
        from .._mlir.ir import IntegerAttr as _IA
        i32 = _i32()
        return _arith.ConstantOp(i32, _IA.get(i32, v)).result
    return _unwrap(v)


def divui(a: Any, b: Any) -> ir.Value:
    """Unsigned integer divide ``a // b``.

    Generates ``arith.divui`` which compiles to efficient ``udiv`` on AMD GPU
    (vs the default ``arith.floordivsi`` → ``sdiv`` from Python's ``//``).
    For non-negative values, ``udiv`` is significantly faster than ``sdiv``.

    Args:
        a: i32 dividend (ArithValue or int).
        b: i32 divisor (ArithValue or compile-time int constant).
    """
    from .._mlir.dialects import arith as _arith
    a_ = _unwrap(a)
    b_ = _to_i32_value(b) if isinstance(b, int) else _unwrap(b)
    return _arith.DivUIOp(a_, b_).result


def remui(a: Any, b: Any) -> ir.Value:
    """Unsigned integer remainder ``a % b``.

    Generates ``arith.remui`` which compiles to efficient ``urem`` on AMD GPU.

    Args:
        a: i32 dividend (ArithValue or int).
        b: i32 divisor (ArithValue or compile-time int constant).
    """
    from .._mlir.dialects import arith as _arith
    a_ = _unwrap(a)
    b_ = _to_i32_value(b) if isinstance(b, int) else _unwrap(b)
    return _arith.RemUIOp(a_, b_).result


def load_i32_global(addr_i64: Any) -> ir.Value:
    """Load i32 from a global (addrspace 1) address.

    Used for XGMI P2P reads: ``ptr_p2p`` returns an XGMI-mapped address
    which is in global memory (addrspace 1).  Reading with addrspace(1)
    loads uses the GPU's global memory load path, not flat memory.
    """
    from .._mlir.ir import IntegerType
    addr  = _unwrap(addr_i64)
    ptr_g = llvm.PointerType.get(address_space=1)
    gptr  = llvm.IntToPtrOp(ptr_g, addr).result
    return llvm.LoadOp(_i32(), gptr, alignment=4).result


def atomic_add_i32_at(addr_i64: Any, val: Any) -> ir.Value:
    """GPU atomic add to i32 located at i64 address (local device memory).

    Equivalent to ``atomicrmw add i32* ptr, val monotonic`` in LLVM IR.
    Use for local GPU barriers and counters (NOT shmem atomics).
    """
    addr = _unwrap(addr_i64)
    val_ = _unwrap(val)
    ptr  = _to_ptr(addr)
    return llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        ptr,
        val_,
        llvm.AtomicOrdering.monotonic,
    ).res


def atomic_add_i64_at(addr_i64: Any, val: Any) -> ir.Value:
    """GPU atomic add to i64 located at i64 address (local device memory).

    Equivalent to ``atomicrmw add i64* ptr, val monotonic`` in LLVM IR.
    Use for generation counters that need i64 range (e.g. barrier flag).
    AMDGPU generates: ``flat_atomic_add_u64``.
    """
    addr = _unwrap(addr_i64)
    val_ = _unwrap(val)
    ptr  = _to_ptr(addr)
    return llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        ptr,
        val_,
        llvm.AtomicOrdering.monotonic,
    ).res


def atomic_add_monotonic(ptr: Any, val: Any) -> ir.Value:
    """``llvm.atomicrmw add ptr, val monotonic`` — atomic fetch-and-add.

    Suitable for local GPU memory grid-barrier counters.

    Returns:
        Old value (i32) before the add.
    """
    ptr_val = _unwrap(ptr)
    val_val = _unwrap(val)
    return llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        ptr_val,
        val_val,
        llvm.AtomicOrdering.monotonic,
    ).res


# ---------------------------------------------------------------------------
# Memory fence (LLVM)
# ---------------------------------------------------------------------------
def fence_one_as_seq_cst() -> None:
    """``fence syncscope("one-as") seq_cst`` — system-scope memory fence.

    Ensures all prior memory ops (including XGMI shmem writes) are ordered
    before subsequent signal writes.  Required between data writes and
    signal sends in the dispatch Phase 2 protocol.

    Note: seq_cst is the strongest (and most expensive) ordering.
    Prefer ``fence_one_as_release()`` when a global total order is not needed.
    """
    llvm.FenceOp(
        llvm.AtomicOrdering.seq_cst,
        syncscope="one-as",
    )


def fence_one_as_release() -> None:
    """``fence syncscope("one-as") release`` — lightweight system-scope fence.

    Equivalent to HIP's ``__threadfence_system()`` which mori uses in
    dispatch Phase 2 before the signal write
    (``core::AtomicStoreRelaxedSystem``).

    Semantics: all prior memory operations are globally visible before any
    subsequent store.  Unlike ``fence_one_as_seq_cst()``, this does NOT
    establish a global total order (seq_cst), so it avoids the costly
    ``s_memrealtime`` instruction on AMDGPU, saving ~10-20 μs per dispatch.

    Safety: sufficient for the Phase 2 signaling pattern — we only need
    Phase 1 XGMI writes to be visible to remote GPUs before the signal
    write, which release ordering guarantees.
    """
    llvm.FenceOp(
        llvm.AtomicOrdering.release,
        syncscope="one-as",
    )


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


def store_i32_shmem(addr_i64: Any, offset: Any, val: Any) -> None:
    """System-scope i32 store to fine-grained shmem via flat (addrspace 0) pointer.

    Generates ``flat_store_dword sc0`` on AMDGPU.  The sc0 bit bypasses the L2
    cache, making writes immediately visible to remote GPUs via XGMI — identical
    in effect to mori's ``WarpCopy`` to ``hipDeviceMallocUncached`` memory.

    Use this (rather than ``store_i32_system`` which uses addrspace(1)) when the
    destination is a local shmem buffer obtained via ``data_ptr()`` / ShmemMalloc,
    whose pointer lives in the flat address space (addrspace 0).

    Args:
        addr_i64: i64 base address (flat, from shmem ``data_ptr()``).
        offset:   i32 element offset; byte address = ``addr_i64 + offset * 4``.
        val:      i32 value to store.
    """
    base = _unwrap(addr_i64)
    off  = _unwrap(offset)
    val_ = _unwrap(val)
    off64 = llvm.ZExtOp(_i64(), off).res if off.type == _i32() else off
    byte_off = llvm.MulOp(off64, _const_i64(4), ir.Attribute.parse("#llvm.overflow<none>")).result
    addr = llvm.AddOp(base, byte_off, ir.Attribute.parse("#llvm.overflow<none>")).result
    ptr_flat_ty = llvm.PointerType.get(address_space=0)
    gptr = llvm.IntToPtrOp(ptr_flat_ty, addr).result
    llvm.StoreOp(
        val_, gptr,
        alignment=4,
        ordering=llvm.AtomicOrdering.monotonic,
        syncscope="one-as",
    )


def store_i32_global(addr_i64: Any, offset: Any, val: Any) -> None:
    """Plain store i32 via ``ptr addrspace(1)`` (global address space).

    Like mori's direct P2P writes: ``memObj->GetAs<int*>(destPe)[offset] = val``.
    Uses global addrspace(1) pointer so that XGMI P2P addresses work correctly.
    Does NOT use atomic ordering — caller must ensure data-race safety.
    """
    base = _unwrap(addr_i64)
    off  = _unwrap(offset)
    val_ = _unwrap(val)
    off64 = llvm.ZExtOp(_i64(), off).res if off.type == _i32() else off
    byte_off = llvm.MulOp(off64, _const_i64(4), ir.Attribute.parse("#llvm.overflow<none>")).result
    addr = llvm.AddOp(base, byte_off, ir.Attribute.parse("#llvm.overflow<none>")).result
    ptr_global_ty = llvm.PointerType.get(address_space=1)
    gptr = llvm.IntToPtrOp(ptr_global_ty, addr).result
    llvm.StoreOp(val_, gptr, alignment=4)


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


def load_v4i32_global(addr_i64: Any) -> ir.Value:
    """Load 128-bit (4 × i32) vector from a global (addrspace 1) P2P address.

    Used in combine Stage 3 to load 16 bytes (4 × bf16 pairs) per lane from
    remote GPU's shmem_comb_inp via XGMI P2P, replacing 4 separate
    load_i32_global calls and reducing P2P load instruction count by 4×.

    Args:
        addr_i64: i64 integer address in global (addrspace 1) memory.

    Returns:
        ``vector<4xi32>`` value.
    """
    from .._mlir.ir import VectorType
    v4i32 = VectorType.get([4], _i32())
    ptr_global_ty = llvm.PointerType.get(address_space=1)
    gptr = llvm.IntToPtrOp(ptr_global_ty, _unwrap(addr_i64)).result
    return llvm.LoadOp(v4i32, gptr, alignment=4).result


def store_v4i32_global(vec: Any, addr_i64: Any) -> None:
    """Store 128-bit (4 × i32) vector to a global (addrspace 1) address.

    The address is given as i64; this function converts to ``ptr addrspace(1)``
    before storing, which the AMD backend can translate to ``global_store_dwordx4``.

    Args:
        vec:      ``vector<4xi32>`` value.
        addr_i64: Destination i64 integer address.
    """
    from .._mlir.ir import VectorType
    vec_val  = _unwrap(vec)
    addr_val = _unwrap(addr_i64)
    # inttoptr → ptr addrspace(1)
    ptr_global_ty = llvm.PointerType.get(address_space=1)
    gptr = llvm.IntToPtrOp(ptr_global_ty, addr_val).result
    llvm.StoreOp(vec_val, gptr, alignment=4)


# ---------------------------------------------------------------------------
# Convenience: sync_threads (alias for fx.gpu.barrier)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Raw typed load/store via pointer arithmetic
# ---------------------------------------------------------------------------
def _to_ptr(v: ir.Value) -> ir.Value:
    """Convert i64 to LLVM ptr, or pass through if already a ptr."""
    type_str = str(v.type)
    if "i64" in type_str or "index" in type_str:
        return llvm.IntToPtrOp(_ptr(), v).result
    return v


def load_i32_at(base_i64: Any, offset: Any) -> ir.Value:
    """Load i32 from ``base_i64 + offset * 4``."""
    base = _unwrap(base_i64)
    off  = _unwrap(offset)
    off64 = llvm.ZExtOp(_i64(), off).res if off.type == _i32() else off
    byte_off = llvm.MulOp(off64, _const_i64(4), ir.Attribute.parse("#llvm.overflow<none>")).result
    addr = llvm.AddOp(base, byte_off, ir.Attribute.parse("#llvm.overflow<none>")).result
    ptr  = _to_ptr(addr)
    return llvm.LoadOp(_i32(), ptr, alignment=4).result


def load_f32_at(base_i64: Any, offset: Any) -> ir.Value:
    """Load f32 from ``base_i64 + offset * 4``."""
    from .._mlir.dialects import llvm as _llvm
    base = _unwrap(base_i64)
    off  = _unwrap(offset)
    off64 = _llvm.ZExtOp(_i64(), off).res if off.type == _i32() else off
    byte_off = _llvm.MulOp(off64, _const_i64(4), ir.Attribute.parse("#llvm.overflow<none>")).result
    addr = _llvm.AddOp(base, byte_off, ir.Attribute.parse("#llvm.overflow<none>")).result
    ptr  = _to_ptr(addr)
    f32  = ir.F32Type.get()
    return _llvm.LoadOp(f32, ptr, alignment=4).result


def store_i32_at(base_i64: Any, offset: Any, val: Any) -> None:
    """Store i32 *val* to ``base_i64 + offset * 4``."""
    base = _unwrap(base_i64)
    off  = _unwrap(offset)
    val_ = _unwrap(val)
    off64 = llvm.ZExtOp(_i64(), off).res if off.type == _i32() else off
    byte_off = llvm.MulOp(off64, _const_i64(4), ir.Attribute.parse("#llvm.overflow<none>")).result
    addr = llvm.AddOp(base, byte_off, ir.Attribute.parse("#llvm.overflow<none>")).result
    ptr  = _to_ptr(addr)
    llvm.StoreOp(val_, ptr, alignment=4)


def add_i64(a: Any, b: Any) -> ir.Value:
    """``llvm.add i64 a, b`` — 64-bit integer addition."""
    a_ = _unwrap(a)
    b_ = _unwrap(b)
    return llvm.AddOp(a_, b_, ir.Attribute.parse("#llvm.overflow<none>")).result


def mul_i64(a: Any, b: int) -> ir.Value:
    """``llvm.mul i64 a, const(b)`` — 64-bit multiply by constant."""
    a_ = _unwrap(a)
    return llvm.MulOp(
        a_, _const_i64(b), ir.Attribute.parse("#llvm.overflow<none>")
    ).result


def zext_i32_to_i64(v: Any) -> ir.Value:
    """Zero-extend i32 to i64."""
    v_ = _unwrap(v)
    if v_.type == _i64():
        return v_
    return llvm.ZExtOp(_i64(), v_).result


def load_i64_at(base_i64: Any, offset: Any) -> ir.Value:
    """Load i64 from ``base_i64 + offset * 8``."""
    base = _unwrap(base_i64)
    off  = _unwrap(offset)
    off64 = llvm.ZExtOp(_i64(), off).res if off.type == _i32() else off
    byte_off = llvm.MulOp(off64, _const_i64(8), ir.Attribute.parse("#llvm.overflow<none>")).result
    addr = llvm.AddOp(base, byte_off, ir.Attribute.parse("#llvm.overflow<none>")).result
    ptr  = _to_ptr(addr)
    return llvm.LoadOp(_i64(), ptr, alignment=8).result


def store_i64_at(base_i64: Any, offset: Any, val: Any) -> None:
    """Store i64 *val* to ``base_i64 + offset * 8``."""
    base = _unwrap(base_i64)
    off  = _unwrap(offset)
    val_ = _unwrap(val)
    off64 = llvm.ZExtOp(_i64(), off).res if off.type == _i32() else off
    byte_off = llvm.MulOp(off64, _const_i64(8), ir.Attribute.parse("#llvm.overflow<none>")).result
    addr = llvm.AddOp(base, byte_off, ir.Attribute.parse("#llvm.overflow<none>")).result
    ptr  = _to_ptr(addr)
    llvm.StoreOp(val_, ptr, alignment=8)


def const_i32(v: int) -> ir.Value:
    """Materialize a Python int as an i32 MLIR constant."""
    return llvm.ConstantOp(_i32(), ir.IntegerAttr.get(_i32(), v)).result


def const_i64(v: int) -> ir.Value:
    """Materialize a Python int as an i64 MLIR constant."""
    return llvm.ConstantOp(_i64(), ir.IntegerAttr.get(_i64(), v)).result


def select_i64(cond: Any, a: Any, b: Any) -> ir.Value:
    """LLVM select: ``a if cond else b`` for i64."""
    return llvm.SelectOp(_unwrap(cond), _unwrap(a), _unwrap(b)).result


def select_i32(cond: Any, a: Any, b: Any) -> ir.Value:
    """Scalar select: ``a if cond else b`` for i32 (via arith.select)."""
    from .._mlir.dialects import arith as _arith
    return _arith.SelectOp(_unwrap(cond), _unwrap(a), _unwrap(b)).result


def icmp_ult_i32(a: Any, b: Any) -> ir.Value:
    """Unsigned less-than comparison for i32."""
    return llvm.ICmpOp(llvm.ICmpPredicate.ult, _unwrap(a), _unwrap(b)).res


def icmp_eq_i32(a: Any, b: Any) -> ir.Value:
    """Equality comparison for i32."""
    return llvm.ICmpOp(llvm.ICmpPredicate.eq, _unwrap(a), _unwrap(b)).res


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


def sync_threads() -> None:
    """``gpu.barrier()`` — block-level thread synchronization.

    Equivalent to ``__syncthreads()`` / ``s.barrier``.
    """
    from .._mlir.dialects import gpu
    gpu.barrier()


def atomic_fetch_add_i32_global(addr_i64, val):
    """Atomic fetch-add on XGMI global (addrspace 1) address, returns old value.

    Generates a single global_atomic_add_ret AMD GPU instruction on the
    XGMI-mapped remote address returned by ptr_p2p() or _sel_pe().
    This is vastly faster than mori_shmem_uint32_atomic_fetch_add_thread
    which uses a software CAS-retry loop (~10+ instructions per lane).

    Performance note:
        For intranode dispatch, call this from lane0 exec-masked block and
        broadcast via readlane(result, 0) to avoid 64x unnecessary XGMI ops.

    Args:
        addr_i64: XGMI address from ptr_p2p() / _sel_pe(), addrspace(1).
        val:      i32 value to add (0 = fetch only, 1 = increment).

    Returns:
        Old i32 value before the add.
    """
    addr = _unwrap(addr_i64)
    val_ = _unwrap(val)
    ptr_g = llvm.PointerType.get(address_space=1)
    gptr  = llvm.IntToPtrOp(ptr_g, addr).result
    return llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        gptr,
        val_,
        llvm.AtomicOrdering.monotonic,
    ).res
