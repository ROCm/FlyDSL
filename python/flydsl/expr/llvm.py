# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""LLVM-dialect API — thin DSL wrappers over the MLIR ``llvm`` dialect."""

from .._mlir import ir
from .._mlir.dialects import llvm as _llvm
from .enum import AtomicOrdering
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


def _atomic_ordering(ordering):
    """Map ``ordering`` onto the LLVM ordering."""
    if isinstance(ordering, _llvm.AtomicOrdering):
        return ordering
    if isinstance(ordering, AtomicOrdering):
        return _ATOMIC_ORDERINGS[ordering]
    raise TypeError(f"ordering must be an fx.AtomicOrdering, got {ordering!r}")


def _normalize_to_llvm_ptr(ptr):
    if isinstance(ptr, Pointer):
        ptr = ptr.llvm_ptr
    if not isinstance(ptr, ir.Value) or not isinstance(ptr.type, _llvm.PointerType):
        raise TypeError(f" pointer must be a fly.ptr or !llvm.ptr, got {ptr}")
    return ptr


def _emit_atomic_rmw(bin_op, ptr, value, syncscope, ordering):
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

    result = _llvm.atomicrmw(
        bin_op,
        _normalize_to_llvm_ptr(ptr),
        as_ir_value(value),
        _atomic_ordering(ordering),
        syncscope=syncscope or None,
    )
    return as_dsl_value(result, value)


@dsl_loc_tracing
def atomic_add(
    ptr: Pointer,
    value,
    *,
    syncscope=None,
    ordering=AtomicOrdering.Monotonic,
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
def atomic_sub(ptr: Pointer, value, *, syncscope=None, ordering=AtomicOrdering.Monotonic):
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
def atomic_min(ptr: Pointer, value, *, syncscope=None, ordering=AtomicOrdering.Monotonic):
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
def atomic_max(ptr: Pointer, value, *, syncscope=None, ordering=AtomicOrdering.Monotonic):
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
def atomic_and(ptr: Pointer, value, *, syncscope=None, ordering=AtomicOrdering.Monotonic):
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
def atomic_or(ptr: Pointer, value, *, syncscope=None, ordering=AtomicOrdering.Monotonic):
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
def atomic_xor(ptr: Pointer, value, *, syncscope=None, ordering=AtomicOrdering.Monotonic):
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
def atomic_xchg(ptr: Pointer, value, *, syncscope=None, ordering=AtomicOrdering.Monotonic):
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
    ptr = _normalize_to_llvm_ptr(ptr)
    if value.dtype not in (Float32, Float64):
        raise TypeError(f"atomic_fmin / atomic_fmax take an fx.Float32 or fx.Float64 value, got {value!r}")

    int_type, unsigned_type = (Int32, Uint32) if value.dtype is Float32 else (Int64, Uint64)
    bits = _llvm.bitcast(int_type.ir_type, as_ir_value(value))
    # The unsigned operand promotes min to umin inside _emit_atomic_rmw.
    signed_op = _llvm.AtomicBinOp.min if minimum else _llvm.AtomicBinOp.max
    unsigned_op = _llvm.AtomicBinOp.max if minimum else _llvm.AtomicBinOp.min

    def emit(bin_op, dtype):
        return as_ir_value(_emit_atomic_rmw(bin_op, ptr, dtype(bits), syncscope, ordering))

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
    is_positive=None,
    syncscope=None,
    ordering=AtomicOrdering.Monotonic,
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
    is_positive=None,
    syncscope=None,
    ordering=AtomicOrdering.Monotonic,
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
    syncscope=None,
    success_ordering=AtomicOrdering.Monotonic,
    failure_ordering=AtomicOrdering.Monotonic,
    weak=False,
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
            succeeds.
        failure_ordering: An ``fx.AtomicOrdering`` value used when the
            comparison fails. It cannot be ``Release`` or ``AcqRel`` because a
            failed exchange does not perform a write.
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
