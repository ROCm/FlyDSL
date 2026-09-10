# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""LLVM-dialect API — thin DSL wrappers over the MLIR ``llvm`` dialect."""

from .._mlir import ir
from .._mlir.dialects import llvm as _llvm
from .enum import AtomicOrdering, SyncScope
from .meta import dsl_loc_tracing
from .numeric import BFloat16, Float16, Float32, Float64, Int32, Int64, Integer, Uint32, Uint64, as_numeric
from .typing import Numeric, Pointer, Vector, as_dsl_value, as_ir_value

__all__ = [
    "atomic_add",
    "atomic_sub",
    "atomic_min",
    "atomic_max",
    "atomic_and",
    "atomic_or",
    "atomic_xor",
    "atomic_xchg",
    "atomic_cas",
    "atomic_fmin",
    "atomic_fmax",
    "generic_load",
    "generic_store",
    "memory_fence",
]


# Float form of a binop. Absent means LLVM has no float atomicrmw for it, e.g.
# the bitwise ops.
_FLOAT_FORMS = {
    _llvm.AtomicBinOp.add: _llvm.AtomicBinOp.fadd,
    _llvm.AtomicBinOp.sub: _llvm.AtomicBinOp.fsub,
    _llvm.AtomicBinOp.min: _llvm.AtomicBinOp.fmin,
    _llvm.AtomicBinOp.max: _llvm.AtomicBinOp.fmax,
    _llvm.AtomicBinOp.xchg: _llvm.AtomicBinOp.xchg,
}

# Unsigned form. Only the comparisons have one; add / and / ... are sign-agnostic.
_UNSIGNED_FORMS = {
    _llvm.AtomicBinOp.min: _llvm.AtomicBinOp.umin,
    _llvm.AtomicBinOp.max: _llvm.AtomicBinOp.umax,
}

# AtomicOrdering -> LLVM ordering.
_ATOMIC_ORDERINGS = {
    AtomicOrdering.NotAtomic: _llvm.AtomicOrdering.not_atomic,
    AtomicOrdering.Unordered: _llvm.AtomicOrdering.unordered,
    AtomicOrdering.Monotonic: _llvm.AtomicOrdering.monotonic,
    AtomicOrdering.Acquire: _llvm.AtomicOrdering.acquire,
    AtomicOrdering.Release: _llvm.AtomicOrdering.release,
    AtomicOrdering.AcqRel: _llvm.AtomicOrdering.acq_rel,
    AtomicOrdering.SeqCst: _llvm.AtomicOrdering.seq_cst,
}

# The types LLVM does an atomicrmw on.
_ATOMIC_FLOATS = (Float16, BFloat16, Float32, Float64)
_ATOMIC_INT_WIDTHS = (8, 16, 32, 64)
# Only these take a vector operand, and only a vector of floats.
_VECTOR_FORMS = frozenset(
    [
        _llvm.AtomicBinOp.fadd,
        _llvm.AtomicBinOp.fsub,
        _llvm.AtomicBinOp.fmin,
        _llvm.AtomicBinOp.fmax,
    ]
)


def _atomic_operand(value, ptr):
    """Turn a raw Python number or bare MLIR value into a DSL scalar.

    A plain ``1`` / ``1.0`` takes the type ``ptr`` points to when there is one,
    so ``atomic_add(f32_ptr, 1)`` adds a float; otherwise it falls back to the
    default Python mapping (int -> Int32, float -> Float32), as does a bare
    signless MLIR value, which carries no signedness of its own.
    """
    if isinstance(value, (Numeric, Vector)):
        return value
    if isinstance(value, ir.Value):
        return Numeric.from_ir_type(value.type)(value)
    if isinstance(ptr, Pointer):
        return ptr.element_type(value)
    return as_numeric(value)


def _atomic_ordering(ordering) -> _llvm.AtomicOrdering:
    """Map ``ordering`` onto the LLVM ordering."""
    if isinstance(ordering, _llvm.AtomicOrdering):
        return ordering
    if isinstance(ordering, AtomicOrdering):
        return _ATOMIC_ORDERINGS[ordering]
    raise TypeError(f"ordering must be an fx.AtomicOrdering, got {ordering!r}")


def _normalize_to_llvm_ptr(ptr, *, fly_ptr_only=False) -> ir.Value:
    if fly_ptr_only and not isinstance(ptr, Pointer):
        raise TypeError(f"requires an fx.Pointer, got {ptr!r}")
    if isinstance(ptr, Pointer):
        ptr = ptr.llvm_ptr
    if not isinstance(ptr, ir.Value) or not isinstance(ptr.type, _llvm.PointerType):
        raise TypeError(f"pointer must be a fly.ptr or !llvm.ptr, got {ptr}")
    return ptr


def _emit_atomic_rmw(bin_op, ptr, value, syncscope, ordering, alignment=None):
    value = _atomic_operand(value, ptr)
    value_type = value.dtype

    if isinstance(ptr, Pointer) and value_type.ir_type != ptr.element_type.ir_type:
        raise TypeError(
            f"atomic value type {value_type.__name__} must match pointer element type {ptr.element_type.__name__}"
        )
    op = str(bin_op).lstrip("_")  # MLIR spells the bitwise ops _and / _or / _xor

    if value_type.is_float:
        if value_type not in _ATOMIC_FLOATS:
            raise TypeError(f"atomic {op} got unsupported float type: {value_type.__name__}")
        bin_op = _FLOAT_FORMS.get(bin_op)
        if bin_op is None:
            raise TypeError(f"atomic {op} has no floating-point form")
    elif issubclass(value_type, Integer):
        if value_type.width not in _ATOMIC_INT_WIDTHS:
            raise TypeError(f"atomic {op} takes an 8 / 16 / 32 / 64-bit integer, got {value_type.__name__}")
        if not value_type.signed:
            bin_op = _UNSIGNED_FORMS.get(bin_op, bin_op)

    if isinstance(value, Vector) and bin_op not in _VECTOR_FORMS:
        raise TypeError(f"atomic {op} takes no vector operand, got {value.type}")

    if alignment is None:
        alignment = ptr.alignment if isinstance(ptr, Pointer) else None
    result = _llvm.atomicrmw(
        bin_op,
        _normalize_to_llvm_ptr(ptr),
        as_ir_value(value),
        _atomic_ordering(ordering),
        syncscope=syncscope or None,
        alignment=alignment,
    )
    return as_dsl_value(result, value)


@dsl_loc_tracing
def generic_load(
    ptr: Pointer,
    *,
    dtype: type[Numeric] | type[Vector] | None = None,
    count: int | None = None,
    memory_order: AtomicOrdering = AtomicOrdering.NotAtomic,
    syncscope: SyncScope | str | None = None,
    nontemporal: bool = False,
    volatile: bool = False,
) -> Numeric | Vector:
    """Load a scalar or vector through a pointer.

    Args:
        ptr: Generic-, global-, or shared-address-space ``fx.Pointer`` to load
            from. Its alignment is attached to the generated LLVM load.
        dtype: scalar or concrete vector result type. When omitted, the
            pointer's element type is used.
        count: Number of scalar elements to load. ``None`` and ``1`` produce a
            scalar when ``dtype`` is scalar; values greater than ``1`` produce
            a flat vector. Must be ``None`` when ``dtype`` is a vector type.
        memory_order: Atomic ordering for the load. Defaults to
            ``fx.AtomicOrdering.NotAtomic``.
        syncscope: Optional LLVM or target-specific synchronization scope.
            Requires an atomic ``memory_order``.
        nontemporal: Whether to mark the load as non-temporal.
        volatile: Whether to mark the load as volatile.

    Returns:
        A scalar or vector matching the requested ``dtype`` and
        ``count`` combination.

    Raises:
        TypeError: If ``ptr`` is not an ``fx.Pointer`` or ``dtype`` is not a
            supported scalar or vector type.
        ValueError: If ``count`` is invalid or the atomic ordering options are
            incompatible.

    Examples:
        The supported scalar and vector forms are::

            scalar = fx.generic_load(ptr)                             # pointer element type
            scalar = fx.generic_load(ptr, dtype=fx.Float32)           # explicit scalar type
            scalar = fx.generic_load(ptr, dtype=fx.Float32, count=1)  # explicit scalar count
            vector = fx.generic_load(ptr, count=4)                    # inferred float32x4
            vector = fx.generic_load(ptr, dtype=fx.Float32, count=4)  # explicit element type
            vector = fx.generic_load(ptr, dtype=fx.Float32x4)         # concrete vector type
    """
    llvm_ptr = _normalize_to_llvm_ptr(ptr, fly_ptr_only=True)
    llvm_ordering = _atomic_ordering(memory_order)
    if llvm_ordering in (_llvm.AtomicOrdering.release, _llvm.AtomicOrdering.acq_rel):
        raise ValueError(f"invalid load memory order: {memory_order}")
    if llvm_ordering == _llvm.AtomicOrdering.not_atomic and syncscope is not None:
        raise ValueError("syncscope requires an atomic memory order")

    if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 1):
        raise ValueError(f"count must be a positive integer or None, got {count!r}")

    if dtype is None:
        dtype = ptr.element_type

    vector_shape = None
    if isinstance(dtype, type) and issubclass(dtype, Numeric):
        if count is None or count == 1:
            result_type = dtype.ir_type
        else:
            result_type = Vector.make_type(count, dtype)
            vector_shape = (count,)
    elif isinstance(dtype, type) and issubclass(dtype, Vector):
        if count is not None:
            raise ValueError("count must be None when dtype is a vector type")
        try:
            result_type = dtype.ir_type
        except AttributeError as exc:
            raise TypeError("dtype must be a concrete vector type") from exc
    else:
        raise TypeError("dtype must be a scalar or vector type")

    kwargs = {}
    if llvm_ordering != _llvm.AtomicOrdering.not_atomic:
        kwargs["ordering"] = llvm_ordering
    if syncscope is not None:
        kwargs["syncscope"] = syncscope
    kwargs["alignment"] = ptr.alignment
    if nontemporal:
        kwargs["nontemporal"] = True
    if volatile:
        kwargs["volatile_"] = True

    result = _llvm.LoadOp(result_type, llvm_ptr, **kwargs).result

    if vector_shape is not None:
        return Vector(result, vector_shape, dtype)
    return dtype(result)


@dsl_loc_tracing
def generic_store(
    ptr: Pointer,
    value,
    *,
    memory_order: AtomicOrdering = AtomicOrdering.NotAtomic,
    syncscope: SyncScope | str | None = None,
    nontemporal: bool = False,
    volatile: bool = False,
) -> None:
    """Store a scalar or vector through a pointer.

    Args:
        ptr: Generic-, global-, or shared-address-space ``fx.Pointer`` to store
            through. Its alignment is attached to the generated LLVM store.
        value: scalar or vector value to store.
        memory_order: Atomic ordering for the store. Defaults to
            ``fx.AtomicOrdering.NotAtomic``.
        syncscope: Optional LLVM or target-specific synchronization scope.
            Requires an atomic ``memory_order``.
        nontemporal: Whether to mark the store as non-temporal.
        volatile: Whether to mark the store as volatile.

    Raises:
        TypeError: If ``ptr`` is not an ``fx.Pointer``.
        ValueError: If the atomic ordering options are incompatible.

    Examples:
        Store scalar and vector values using the pointer's alignment::

            fx.generic_store(ptr, scalar)
            fx.generic_store(ptr, vector, volatile=True)
    """
    llvm_ptr = _normalize_to_llvm_ptr(ptr, fly_ptr_only=True)
    llvm_ordering = _atomic_ordering(memory_order)
    if llvm_ordering in (_llvm.AtomicOrdering.acquire, _llvm.AtomicOrdering.acq_rel):
        raise ValueError(f"invalid store memory order: {memory_order}")
    if llvm_ordering == _llvm.AtomicOrdering.not_atomic and syncscope is not None:
        raise ValueError("syncscope requires an atomic memory order")

    kwargs = {}
    if llvm_ordering != _llvm.AtomicOrdering.not_atomic:
        kwargs["ordering"] = llvm_ordering
    if syncscope is not None:
        kwargs["syncscope"] = syncscope
    kwargs["alignment"] = ptr.alignment
    if nontemporal:
        kwargs["nontemporal"] = True
    if volatile:
        kwargs["volatile_"] = True

    _llvm.StoreOp(as_ir_value(value), llvm_ptr, **kwargs)


@dsl_loc_tracing
def atomic_add(
    ptr: Pointer,
    value,
    *,
    syncscope: SyncScope | str | None = None,
    ordering: AtomicOrdering = AtomicOrdering.Monotonic,
):
    """Atomically add ``value`` and return the previous value.

    Args:
        ptr: Destination ``fly.ptr`` or MLIR ``!llvm.ptr``.
        value: Value to add. Raw Python numbers are converted to the pointer's
            element type when it is available.
        syncscope: LLVM scope or target-specific synchronization scope. ``None``
            selects LLVM's system scope; target-specific scopes include
            ``fx.rocdl.SyncScope.Agent`` and ``fx.rocdl.SyncScope.Workgroup``.
        ordering: An ``fx.AtomicOrdering`` value. LLVM requires ``Monotonic``
            or a stronger ordering for ``atomicrmw`` operations.

    Returns:
        The value stored at ``ptr`` before the addition, represented using the
        same DSL scalar or vector type as ``value``.

    Raises:
        TypeError: If the pointer, value type, or ordering is unsupported.
    """
    return _emit_atomic_rmw(_llvm.AtomicBinOp.add, ptr, value, syncscope, ordering)


@dsl_loc_tracing
def atomic_sub(
    ptr: Pointer,
    value,
    *,
    syncscope: SyncScope | str | None = None,
    ordering: AtomicOrdering = AtomicOrdering.Monotonic,
):
    """Atomically subtract ``value`` and return the previous value.

    Args:
        ptr: Destination ``fly.ptr`` or MLIR ``!llvm.ptr``.
        value: Value to subtract. Integer and floating-point scalar values are
            supported, as are supported floating-point vectors.
        syncscope: LLVM scope or target-specific synchronization scope. ``None``
            selects LLVM's system scope.
        ordering: An ``fx.AtomicOrdering`` value. LLVM requires ``Monotonic``
            or a stronger ordering for ``atomicrmw`` operations.

    Returns:
        The value stored at ``ptr`` before the subtraction.

    Raises:
        TypeError: If the pointer, value type, or ordering is unsupported.
    """
    return _emit_atomic_rmw(_llvm.AtomicBinOp.sub, ptr, value, syncscope, ordering)


@dsl_loc_tracing
def atomic_min(
    ptr: Pointer,
    value,
    *,
    syncscope: SyncScope | str | None = None,
    ordering: AtomicOrdering = AtomicOrdering.Monotonic,
):
    """Atomically take the minimum and return the previous value.

    Args:
        ptr: Destination ``fly.ptr`` or MLIR ``!llvm.ptr``.
        value: Candidate minimum. Its DSL type selects the signed, unsigned,
            or floating-point LLVM operation.
        syncscope: LLVM scope or target-specific synchronization scope. ``None``
            selects LLVM's system scope.
        ordering: An ``fx.AtomicOrdering`` value. LLVM requires ``Monotonic``
            or a stronger ordering for ``atomicrmw`` operations.

    Returns:
        The value stored at ``ptr`` before the minimum operation.

    Raises:
        TypeError: If the pointer, value type, or ordering is unsupported.
    """
    return _emit_atomic_rmw(_llvm.AtomicBinOp.min, ptr, value, syncscope, ordering)


@dsl_loc_tracing
def atomic_max(
    ptr: Pointer,
    value,
    *,
    syncscope: SyncScope | str | None = None,
    ordering: AtomicOrdering = AtomicOrdering.Monotonic,
):
    """Atomically take the maximum and return the previous value.

    Args:
        ptr: Destination ``fly.ptr`` or MLIR ``!llvm.ptr``.
        value: Candidate maximum. Its DSL type selects the signed, unsigned,
            or floating-point LLVM operation.
        syncscope: LLVM scope or target-specific synchronization scope. ``None``
            selects LLVM's system scope.
        ordering: An ``fx.AtomicOrdering`` value. LLVM requires ``Monotonic``
            or a stronger ordering for ``atomicrmw`` operations.

    Returns:
        The value stored at ``ptr`` before the maximum operation.

    Raises:
        TypeError: If the pointer, value type, or ordering is unsupported.
    """
    return _emit_atomic_rmw(_llvm.AtomicBinOp.max, ptr, value, syncscope, ordering)


@dsl_loc_tracing
def atomic_and(
    ptr: Pointer,
    value,
    *,
    syncscope: SyncScope | str | None = None,
    ordering: AtomicOrdering = AtomicOrdering.Monotonic,
):
    """Atomically apply bitwise AND and return the previous value.

    Args:
        ptr: Destination ``fly.ptr`` or MLIR ``!llvm.ptr``.
        value: Integer value to combine with the value stored at ``ptr``.
        syncscope: LLVM scope or target-specific synchronization scope. ``None``
            selects LLVM's system scope.
        ordering: An ``fx.AtomicOrdering`` value. LLVM requires ``Monotonic``
            or a stronger ordering for ``atomicrmw`` operations.

    Returns:
        The value stored at ``ptr`` before the bitwise operation.

    Raises:
        TypeError: If ``value`` is not a supported scalar integer, or if the
            pointer or ordering is unsupported.
    """
    return _emit_atomic_rmw(_llvm.AtomicBinOp._and, ptr, value, syncscope, ordering)


@dsl_loc_tracing
def atomic_or(
    ptr: Pointer,
    value,
    *,
    syncscope: SyncScope | str | None = None,
    ordering: AtomicOrdering = AtomicOrdering.Monotonic,
):
    """Atomically apply bitwise OR and return the previous value.

    Args:
        ptr: Destination ``fly.ptr`` or MLIR ``!llvm.ptr``.
        value: Integer value to combine with the value stored at ``ptr``.
        syncscope: LLVM scope or target-specific synchronization scope. ``None``
            selects LLVM's system scope.
        ordering: An ``fx.AtomicOrdering`` value. LLVM requires ``Monotonic``
            or a stronger ordering for ``atomicrmw`` operations.

    Returns:
        The value stored at ``ptr`` before the bitwise operation.

    Raises:
        TypeError: If ``value`` is not a supported scalar integer, or if the
            pointer or ordering is unsupported.
    """
    return _emit_atomic_rmw(_llvm.AtomicBinOp._or, ptr, value, syncscope, ordering)


@dsl_loc_tracing
def atomic_xor(
    ptr: Pointer,
    value,
    *,
    syncscope: SyncScope | str | None = None,
    ordering: AtomicOrdering = AtomicOrdering.Monotonic,
):
    """Atomically apply bitwise XOR and return the previous value.

    Args:
        ptr: Destination ``fly.ptr`` or MLIR ``!llvm.ptr``.
        value: Integer value to combine with the value stored at ``ptr``.
        syncscope: LLVM scope or target-specific synchronization scope. ``None``
            selects LLVM's system scope.
        ordering: An ``fx.AtomicOrdering`` value. LLVM requires ``Monotonic``
            or a stronger ordering for ``atomicrmw`` operations.

    Returns:
        The value stored at ``ptr`` before the bitwise operation.

    Raises:
        TypeError: If ``value`` is not a supported scalar integer, or if the
            pointer or ordering is unsupported.
    """
    return _emit_atomic_rmw(_llvm.AtomicBinOp._xor, ptr, value, syncscope, ordering)


@dsl_loc_tracing
def atomic_xchg(
    ptr: Pointer,
    value,
    *,
    syncscope: SyncScope | str | None = None,
    ordering: AtomicOrdering = AtomicOrdering.Monotonic,
):
    """Atomically replace the pointed-to value and return the previous value.

    Args:
        ptr: Destination ``fly.ptr`` or MLIR ``!llvm.ptr``.
        value: New scalar integer or floating-point value to store.
        syncscope: LLVM scope or target-specific synchronization scope. ``None``
            selects LLVM's system scope.
        ordering: An ``fx.AtomicOrdering`` value. LLVM requires ``Monotonic``
            or a stronger ordering for ``atomicrmw`` operations.

    Returns:
        The value stored at ``ptr`` before the exchange.

    Raises:
        TypeError: If the pointer, value type, or ordering is unsupported.
    """
    return _emit_atomic_rmw(_llvm.AtomicBinOp.xchg, ptr, value, syncscope, ordering)


def _emit_float_minmax(minimum, ptr, value, is_positive, syncscope, ordering):
    """Apply fmin / fmax through the integer atomics, on the IEEE-754 bit pattern.

    A non-negative float's bit pattern orders like a signed integer; a negative
    one's order reverses, and an unsigned comparison restores it. So the op to
    issue depends on the sign of ``value``:

        value >= 0 -> signed max / min
        value <  0 -> unsigned min / max

    Either covers the case where memory holds the opposite sign, because a
    negative bit pattern is >= 2**31 unsigned and < 0 signed.
    """
    from .._mlir.dialects import scf
    from .typing import as_dsl_value, as_ir_value

    value = _atomic_operand(value, ptr)

    if isinstance(ptr, Pointer) and value.dtype.ir_type != ptr.element_type.ir_type:
        raise TypeError(
            f"atomic value type {value.dtype.__name__} must match pointer element type {ptr.element_type.__name__}"
        )
    alignment = ptr.alignment if isinstance(ptr, Pointer) else None

    ptr = _normalize_to_llvm_ptr(ptr)
    if value.dtype not in (Float32, Float64):
        raise TypeError(f"atomic_fmin / atomic_fmax take an fx.Float32 or fx.Float64 value, got {value!r}")

    int_type, unsigned_type = (Int32, Uint32) if value.dtype is Float32 else (Int64, Uint64)
    bits = _llvm.bitcast(int_type.ir_type, as_ir_value(value))
    # The unsigned operand promotes min to umin inside _emit_atomic_rmw.
    signed_op = _llvm.AtomicBinOp.min if minimum else _llvm.AtomicBinOp.max
    unsigned_op = _llvm.AtomicBinOp.max if minimum else _llvm.AtomicBinOp.min

    def emit(bin_op, dtype):
        return as_ir_value(_emit_atomic_rmw(bin_op, ptr, dtype(bits), syncscope, ordering, alignment))

    if is_positive is None:
        zero = _llvm.mlir_constant(ir.IntegerAttr.get(int_type.ir_type, 0))
        if_op = scf.IfOp(_llvm.icmp(_llvm.ICmpPredicate.slt, bits, zero), [int_type.ir_type], has_else=True)
        with ir.InsertionPoint(if_op.then_block):
            scf.YieldOp([emit(unsigned_op, unsigned_type)])
        with ir.InsertionPoint(if_op.else_block):
            scf.YieldOp([emit(signed_op, int_type)])
        old_bits = if_op.results[0]
    elif is_positive:
        old_bits = emit(signed_op, int_type)
    else:
        old_bits = emit(unsigned_op, unsigned_type)

    return as_dsl_value(_llvm.bitcast(value.dtype.ir_type, old_bits), value)


@dsl_loc_tracing
def atomic_fmax(
    ptr: Pointer,
    value,
    *,
    is_positive: bool | None = None,
    syncscope: SyncScope | str | None = None,
    ordering: AtomicOrdering = AtomicOrdering.Monotonic,
):
    """Atomically take the float maximum through the integer atomics.

    NaN is ordered by its bit pattern rather than ignored, so this is not a
    drop-in replacement for IEEE-754 ``maxNum`` when NaN can reach the operands.

    Args:
        ptr: Destination ``fly.ptr`` or MLIR ``!llvm.ptr``.
        value: ``Float32`` or ``Float64`` candidate maximum.
        is_positive: Whether the sign bit of ``value`` is known to be clear.
            ``True`` selects the non-negative path, ``False`` selects the
            negative path, and ``None`` emits a runtime sign test. The supplied
            value is not validated against ``value``.
        syncscope: LLVM scope or target-specific synchronization scope. ``None``
            selects LLVM's system scope. A device scope is preferable on CDNA
            when applicable.
        ordering: An ``fx.AtomicOrdering`` value. LLVM requires ``Monotonic``
            or a stronger ordering for the underlying integer atomics.

    Returns:
        The floating-point value stored at ``ptr`` before the maximum operation.

    Raises:
        TypeError: If ``value`` is not ``Float32`` or ``Float64``, or if the
            pointer or ordering is unsupported.

    Note:
        NaNs are ordered by their bit patterns rather than handled with IEEE-754
        ``maxNum`` semantics. An incorrect ``is_positive`` silently produces an
        incorrect result.
    """
    return _emit_float_minmax(False, ptr, value, is_positive, syncscope, ordering)


@dsl_loc_tracing
def atomic_fmin(
    ptr: Pointer,
    value,
    *,
    is_positive: bool | None = None,
    syncscope: SyncScope | str | None = None,
    ordering: AtomicOrdering = AtomicOrdering.Monotonic,
):
    """Atomically take the float minimum through the integer atomics.

    This is the minimum counterpart of :func:`atomic_fmax` and has the same
    architecture and NaN trade-offs.

    Args:
        ptr: Destination ``fly.ptr`` or MLIR ``!llvm.ptr``.
        value: ``Float32`` or ``Float64`` candidate minimum.
        is_positive: Whether the sign bit of ``value`` is known to be clear.
            ``True`` selects the non-negative path, ``False`` selects the
            negative path, and ``None`` emits a runtime sign test. The supplied
            value is not validated against ``value``.
        syncscope: LLVM scope or target-specific synchronization scope. ``None``
            selects LLVM's system scope. A device scope is preferable on CDNA
            when applicable.
        ordering: An ``fx.AtomicOrdering`` value. LLVM requires ``Monotonic``
            or a stronger ordering for the underlying integer atomics.

    Returns:
        The floating-point value stored at ``ptr`` before the minimum operation.

    Raises:
        TypeError: If ``value`` is not ``Float32`` or ``Float64``, or if the
            pointer or ordering is unsupported.

    Note:
        NaNs are ordered by their bit patterns rather than handled with IEEE-754
        ``minNum`` semantics. An incorrect ``is_positive`` silently produces an
        incorrect result.
    """
    return _emit_float_minmax(True, ptr, value, is_positive, syncscope, ordering)


@dsl_loc_tracing
def atomic_cas(
    ptr: Pointer,
    cmp,
    val,
    *,
    syncscope: SyncScope | str | None = None,
    success_ordering: AtomicOrdering = AtomicOrdering.Monotonic,
    failure_ordering: AtomicOrdering = AtomicOrdering.Monotonic,
    weak: bool = False,
):
    """Atomically compare and exchange, returning ``(old_value, success)``.

    LLVM only compare-exchanges integers and pointers, so a floating-point
    ``cmp`` / ``val`` is rejected; bitcast it to an integer first.

    Args:
        ptr: Destination ``fly.ptr`` or MLIR ``!llvm.ptr``.
        cmp: Integer value to compare with the value stored at ``ptr``.
        val: Integer value to store when the comparison succeeds. It must
            have the same DSL type as ``cmp``.
        syncscope: LLVM scope or target-specific synchronization scope. ``None``
            selects LLVM's system scope.
        success_ordering: An ``fx.AtomicOrdering`` value used when the exchange
            succeeds. LLVM requires ``Monotonic`` or a stronger ordering.
        failure_ordering: An ``fx.AtomicOrdering`` value used when the
            comparison fails. LLVM requires ``Monotonic`` or a stronger
            ordering, but it cannot be stronger than ``success_ordering`` or be
            ``Release`` or ``AcqRel`` because a failed exchange does not
            perform a write.
        weak: Whether LLVM may allow a spurious comparison failure.

    Returns:
        A pair containing the value previously stored at ``ptr`` and a DSL
        Boolean indicating whether the exchange succeeded.

    Raises:
        TypeError: If the operands are floating-point values, their types do
            not match, or the pointer or an ordering is unsupported.
    """
    from .numeric import Boolean
    from .typing import as_dsl_value, as_ir_value

    cmp = _atomic_operand(cmp, ptr)
    if isinstance(ptr, Pointer) and cmp.dtype.ir_type != ptr.element_type.ir_type:
        raise TypeError(
            f"atomic value type {cmp.dtype.__name__} must match pointer element type {ptr.element_type.__name__}"
        )
    val = val if isinstance(val, Numeric) else cmp.dtype(val)

    if cmp.dtype.is_float:
        raise TypeError(f"atomic_cas takes an integer operand, got {cmp!r}")
    if cmp.dtype is not val.dtype:
        raise TypeError(f"cmp and val must have the same type, got {cmp!r} and {val!r}")

    pair = _llvm.cmpxchg(
        _normalize_to_llvm_ptr(ptr),
        as_ir_value(cmp),
        as_ir_value(val),
        _atomic_ordering(success_ordering),
        _atomic_ordering(failure_ordering),
        syncscope=syncscope or None,
        alignment=ptr.alignment if isinstance(ptr, Pointer) else None,
        weak=weak,
    )
    old = _llvm.extractvalue(cmp.dtype.ir_type, pair, [0])
    success = _llvm.extractvalue(ir.IntegerType.get_signless(1), pair, [1])
    return as_dsl_value(old, cmp), Boolean(success)


@dsl_loc_tracing
def memory_fence(
    *,
    syncscope=None,
    ordering=AtomicOrdering.SeqCst,
):
    """Emit an LLVM memory fence for ``syncscope`` and ``ordering``.

    A fence orders other accesses rather than carrying a value of its own, so
    ``ordering`` must be ``Acquire`` or stronger.

    Args:
        syncscope: LLVM scope or target-specific synchronization scope. ``None``
            selects LLVM's system scope; target-specific scopes may narrow the
            participating agents.
        ordering: An ``fx.AtomicOrdering`` value. LLVM fences require
            ``Acquire``, ``Release``, ``AcqRel``, or ``SeqCst``.

    Raises:
        TypeError: If ``ordering`` is not a supported atomic-ordering value.
    """
    _llvm.fence(_atomic_ordering(ordering), syncscope=syncscope or None)
