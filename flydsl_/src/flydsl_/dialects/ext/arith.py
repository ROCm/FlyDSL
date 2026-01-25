"""Arith dialect extensions with operator overloading for Pythonic syntax."""

from functools import partialmethod
from typing import Optional, Union, Tuple
import weakref
import os
import numpy as np

from _mlir.ir import (
    Type, Value, IntegerType, IndexType, F32Type, F64Type, F16Type,
    DenseElementsAttr, Location, InsertionPoint, ShapedType, VectorType
)
from _mlir.dialects import arith as _arith
from _mlir.dialects._ods_common import get_op_result_or_op_results

from ._loc import maybe_default_loc

def _is_integer_like_type(t: Type) -> bool:
    """Check if type is integer-like (including index)."""
    return IntegerType.isinstance(t) or IndexType.isinstance(t)

def _is_floating_point_type(t: Type) -> bool:
    """Check if type is floating point."""
    return F32Type.isinstance(t) or F64Type.isinstance(t) or F16Type.isinstance(t)

def _is_index_type(t: Type) -> bool:
    """Check if type is index."""
    return IndexType.isinstance(t)

def _is_vector_type(t: Type) -> bool:
    """Check if type is a vector."""
    return VectorType.isinstance(t)

def _get_element_type(t: Type) -> Type:
    """Get element type from vector or return the type itself for scalars."""
    if _is_vector_type(t):
        return VectorType(t).element_type
    return t

def _infer_mlir_type(value, vector=False):
    """Infer MLIR type from Python value."""
    if isinstance(value, bool):
        return IntegerType.get_signless(1)
    elif isinstance(value, int):
        # Default to i64 for Python ints
        return IntegerType.get_signless(64)
    elif isinstance(value, float):
        return F64Type.get()
    elif isinstance(value, np.ndarray):
        # TODO: Implement tensor type inference
        raise NotImplementedError("Tensor type inference not yet implemented")
    else:
        raise ValueError(f"Cannot infer MLIR type from {type(value)}")

def constant(
    value: Union[int, float, bool],
    *,
    type: Optional[Type] = None,
    index: bool = False,
    loc: Location = None,
    ip: InsertionPoint = None,
) -> "ArithValue":
    """Create a constant with type inference.
    
    Args:
        value: Python value (int, float, bool)
        type: Optional explicit MLIR type
        index: If True, create index type constant
        loc: Location for the operation
        ip: Insertion point
        
    Returns:
        ArithValue wrapping the constant
    """
    if index:
        mlir_type = IndexType.get()
    elif type is not None:
        mlir_type = type
    else:
        mlir_type = _infer_mlir_type(value)
    
    if _is_floating_point_type(mlir_type) and not isinstance(value, float):
        value = float(value)

    if loc is None:
        # Prefer a file/line location pointing at user code for better IR dumps.
        try:
            from flydsl.dialects.ext.func import get_user_code_loc

            loc = get_user_code_loc()
        except Exception:
            loc = None
    # If we still only have `unknown`, try inheriting the active Location context.
    try:
        if loc is not None and str(loc) == "loc(unknown)":
            loc = None
    except Exception:
        pass
    if loc is None:
        try:
            loc = Location.current
        except Exception:
            loc = None

    # -------------------------------------------------------------------------
    # Constant emission
    # -------------------------------------------------------------------------
    # NOTE: We intentionally do not cache/unique constants here. In some test and
    # multi-module scenarios, reusing cached MLIR Value handles can lead to
    # interpreter-level crashes if an underlying IR object becomes invalid.
    try:
        ip_eff = ip or InsertionPoint.current
        blk = getattr(ip_eff, "block", None)
    except Exception:
        blk = None

    if blk is not None:
        # Insert at block begin to dominate all uses in this block.
        try:
            with InsertionPoint.at_block_begin(blk):
                result = _arith.ConstantOp(mlir_type, value, loc=loc).result
        except Exception:
            result = _arith.ConstantOp(mlir_type, value, loc=loc, ip=ip).result
        return ArithValue(result)

    result = _arith.ConstantOp(mlir_type, value, loc=loc, ip=ip).result
    return ArithValue(result)

def index(value: int, *, loc: Location = None, ip: InsertionPoint = None) -> "ArithValue":
    """Create an index constant."""
    return constant(value, index=True, loc=loc, ip=ip)

def i32(value: int, *, loc: Location = None, ip: InsertionPoint = None) -> "ArithValue":
    """Create an i32 constant."""
    return constant(value, type=IntegerType.get_signless(32), loc=loc, ip=ip)

def i64(value: int, *, loc: Location = None, ip: InsertionPoint = None) -> "ArithValue":
    """Create an i64 constant."""
    return constant(value, type=IntegerType.get_signless(64), loc=loc, ip=ip)

def f16(value: float, *, loc: Location = None, ip: InsertionPoint = None) -> "ArithValue":
    """Create an f16 constant."""
    return constant(value, type=F16Type.get(), loc=loc, ip=ip)

def f16(value: float, *, loc: Location = None, ip: InsertionPoint = None) -> "ArithValue":
    """Create an f16 constant."""
    return constant(value, type=F16Type.get(), loc=loc, ip=ip)

def f32(value: float, *, loc: Location = None, ip: InsertionPoint = None) -> "ArithValue":
    """Create an f32 constant."""
    return constant(value, type=F32Type.get(), loc=loc, ip=ip)

def f64(value: float, *, loc: Location = None, ip: InsertionPoint = None) -> "ArithValue":
    """Create an f64 constant."""
    return constant(value, type=F64Type.get(), loc=loc, ip=ip)

def maximum(lhs: Union["ArithValue", Value], rhs: Union["ArithValue", Value], *, loc: Location = None) -> "ArithValue":
    """Compute maximum of two values (automatically handles float/int types).
    
    Args:
        lhs: Left operand (ArithValue, Value, or Python number)
        rhs: Right operand (ArithValue, Value, or Python number)
        loc: Optional source location
        
    Returns:
        ArithValue wrapping the maximum result
        
    Example:
        >>> a = arith.f32(1.5)
        >>> b = arith.f32(2.3)
        >>> c = arith.maximum(a, b)  # Function style
        >>> d = a.max(b)              # Method style (equivalent)
    """
    return _minmax_op(lhs, rhs, op_type="max", loc=loc)

def minimum(lhs: Union["ArithValue", Value], rhs: Union["ArithValue", Value], *, loc: Location = None) -> "ArithValue":
    """Compute minimum of two values (automatically handles float/int types).
    
    Args:
        lhs: Left operand (ArithValue, Value, or Python number)
        rhs: Right operand (ArithValue, Value, or Python number)
        loc: Optional source location
        
    Returns:
        ArithValue wrapping the minimum result
        
    Example:
        >>> a = arith.f32(1.5)
        >>> b = arith.f32(2.3)
        >>> c = arith.minimum(a, b)  # Function style
        >>> d = a.min(b)              # Method style (equivalent)
    """
    return _minmax_op(lhs, rhs, op_type="min", loc=loc)

def select(condition: Union["ArithValue", Value], true_value: Union["ArithValue", Value], 
           false_value: Union["ArithValue", Value], *, loc: Location = None) -> "ArithValue":
    """Select between two values based on a condition (ternary operator).
    
    Args:
        condition: Boolean condition (i1 type)
        true_value: Value to return if condition is true
        false_value: Value to return if condition is false
        loc: Optional source location
        
    Returns:
        ArithValue wrapping the selected value
        
    Example:
        >>> cond = a < b
        >>> result = arith.select(cond, a, b)  # Equivalent to: a if cond else b
    """
    cond_val = _unwrap_value(condition) if isinstance(condition, ArithValue) else condition
    true_val = _unwrap_value(true_value) if isinstance(true_value, ArithValue) else true_value
    false_val = _unwrap_value(false_value) if isinstance(false_value, ArithValue) else false_value
    
    result = _arith.SelectOp(cond_val, true_val, false_val, loc=loc).result
    return ArithValue(result)

def extf(result_type: Type, value: Union["ArithValue", Value], *, loc: Location = None) -> "ArithValue":
    """Extend floating point value to a wider type (e.g., f16 -> f32).
    
    Args:
        result_type: Target floating point type
        value: Value to extend
        loc: Optional source location
        
    Returns:
        ArithValue wrapping the extended value
        
    Example:
        >>> f16_val = ...  # some f16 value
        >>> f32_val = arith.extf(T.vector(32, T.f32()), f16_val)
    """
    val = _unwrap_value(value) if isinstance(value, ArithValue) else value
    result = _arith.ExtFOp(result_type, val, loc=loc).result
    return ArithValue(result)

def fptosi(result_type: Type, value: Union["ArithValue", Value], *, loc: Location = None) -> "ArithValue":
    """Convert floating point value to signed integer.
    
    Args:
        result_type: Target integer type
        value: Floating point value to convert
        loc: Optional source location
        
    Returns:
        ArithValue wrapping the integer result
        
    Example:
        >>> f32_val = arith.f32(3.7)
        >>> i32_val = arith.fptosi(T.i32(), f32_val)  # Result: 3
    """
    val = _unwrap_value(value) if isinstance(value, ArithValue) else value
    result = _arith.FPToSIOp(result_type, val, loc=loc).result
    return ArithValue(result)


def sitofp(result_type: Type, value: Union["ArithValue", Value], *, loc: Location = None) -> "ArithValue":
    """Convert signed integer value to floating point.

    Args:
        result_type: Target floating point type (e.g., f32)
        value: Signed integer value to convert
        loc: Optional source location

    Returns:
        ArithValue wrapping the floating point result

    Example:
        >>> i32_val = ...
        >>> f32_val = arith.sitofp(T.f32(), i32_val)
    """
    val = _unwrap_value(value) if isinstance(value, ArithValue) else value
    result = _arith.SIToFPOp(result_type, val, loc=loc).result
    return ArithValue(result)

def constant_vector(element_value: Union[int, float], vector_type: Type, *, loc: Location = None) -> "ArithValue":
    """Create a constant vector with all elements set to the same value.
    
    Args:
        element_value: Scalar value to splat across the vector
        vector_type: Vector type (e.g., T.vector(32, T.f32()))
        loc: Optional source location
        
    Returns:
        ArithValue wrapping the constant vector
        
    Example:
        >>> vec_zero = arith.constant_vector(0.0, T.vector(32, T.f16()))
        >>> vec_ones = arith.constant_vector(1.0, T.vector(16, T.f32()))
    """
    from _mlir.ir import FloatAttr, IntegerAttr, DenseElementsAttr
    
    # Get element type from vector type
    element_type = VectorType(vector_type).element_type
    
    # Create attribute for the element value
    if _is_floating_point_type(element_type):
        elem_attr = FloatAttr.get(element_type, float(element_value))
    elif _is_integer_like_type(element_type):
        elem_attr = IntegerAttr.get(element_type, int(element_value))
    else:
        raise ValueError(f"Unsupported element type for constant vector: {element_type}")
    
    # Create dense elements attribute (splat)
    dense_attr = DenseElementsAttr.get_splat(vector_type, elem_attr)
    
    result = _arith.ConstantOp(vector_type, dense_attr, loc=loc).result
    return ArithValue(result)

def absf(value: Union["ArithValue", Value], *, loc: Location = None) -> "ArithValue":
    """Calculate absolute value (floating point).
    
    Args:
        value: Input value (float or vector of floats)
        loc: Optional source location
        
    Returns:
        Absolute value result wrapped in ArithValue
    """
    from _mlir.dialects import math as _math
    val = _unwrap_value(value)
    result = _math.AbsFOp(val, loc=loc).result
    return ArithValue(result)

def andi(lhs: Union["ArithValue", Value, int], rhs: Union["ArithValue", Value, int], *, loc: Location = None) -> "ArithValue":
    """Bitwise AND operation on integers.
    
    Args:
        lhs: Left operand
        rhs: Right operand
        loc: Optional source location
        
    Returns:
        ArithValue wrapping the AND result
    """
    loc = maybe_default_loc(loc)
    if isinstance(lhs, int):
        lhs = constant(lhs, loc=loc)
    if isinstance(rhs, int):
        rhs = constant(rhs, loc=loc)
    lhs_val = _unwrap_value(lhs)
    rhs_val = _unwrap_value(rhs)
    result = _arith.AndIOp(lhs_val, rhs_val, loc=loc).result
    return ArithValue(result)

def ori(lhs: Union["ArithValue", Value, int], rhs: Union["ArithValue", Value, int], *, loc: Location = None) -> "ArithValue":
    """Bitwise OR operation on integers.
    
    Args:
        lhs: Left operand
        rhs: Right operand
        loc: Optional source location
        
    Returns:
        ArithValue wrapping the OR result
    """
    loc = maybe_default_loc(loc)
    if isinstance(lhs, int):
        lhs = constant(lhs, loc=loc)
    if isinstance(rhs, int):
        rhs = constant(rhs, loc=loc)
    lhs_val = _unwrap_value(lhs)
    rhs_val = _unwrap_value(rhs)
    result = _arith.OrIOp(lhs_val, rhs_val, loc=loc).result
    return ArithValue(result)

def xori(lhs: Union["ArithValue", Value, int], rhs: Union["ArithValue", Value, int], *, loc: Location = None) -> "ArithValue":
    """Bitwise XOR operation on integers.
    
    Args:
        lhs: Left operand
        rhs: Right operand
        loc: Optional source location
        
    Returns:
        ArithValue wrapping the XOR result
    """
    loc = maybe_default_loc(loc)
    if isinstance(lhs, int):
        lhs = constant(lhs, loc=loc)
    if isinstance(rhs, int):
        rhs = constant(rhs, loc=loc)
    lhs_val = _unwrap_value(lhs)
    rhs_val = _unwrap_value(rhs)
    result = _arith.XOrIOp(lhs_val, rhs_val, loc=loc).result
    return ArithValue(result)

def shrui(lhs: Union["ArithValue", Value, int], rhs: Union["ArithValue", Value, int], *, loc: Location = None) -> "ArithValue":
    """Logical (unsigned) right shift operation on integers.
    
    Args:
        lhs: Value to shift
        rhs: Number of bits to shift
        loc: Optional source location
        
    Returns:
        ArithValue wrapping the shift result
    """
    loc = maybe_default_loc(loc)
    if isinstance(lhs, int):
        lhs = constant(lhs, loc=loc)
    if isinstance(rhs, int):
        rhs = constant(rhs, loc=loc)
    lhs_val = _unwrap_value(lhs)
    rhs_val = _unwrap_value(rhs)
    result = _arith.ShRUIOp(lhs_val, rhs_val, loc=loc).result
    return ArithValue(result)

def shli(lhs: Union["ArithValue", Value, int], rhs: Union["ArithValue", Value, int], *, loc: Location = None) -> "ArithValue":
    """Left shift operation on integers.
    
    Args:
        lhs: Value to shift
        rhs: Number of bits to shift
        loc: Optional source location
        
    Returns:
        ArithValue wrapping the shift result
    """
    loc = maybe_default_loc(loc)
    if isinstance(lhs, int):
        lhs = constant(lhs, loc=loc)
    if isinstance(rhs, int):
        rhs = constant(rhs, loc=loc)
    lhs_val = _unwrap_value(lhs)
    rhs_val = _unwrap_value(rhs)
    result = _arith.ShLIOp(lhs_val, rhs_val, loc=loc).result
    return ArithValue(result)

def index_cast(target_type: Type, value: Union["ArithValue", Value, int], *, loc: Location = None) -> "ArithValue":
    """Cast between index and integer types.
    
    Args:
        target_type: Target type (index or integer type)
        value: Value to cast
        loc: Optional source location
        
    Returns:
        ArithValue wrapping the cast result
    """
    loc = maybe_default_loc(loc)
    if isinstance(value, int):
        value = constant(value, loc=loc)
    val = _unwrap_value(value)
    result = _arith.IndexCastOp(target_type, val, loc=loc).result
    return ArithValue(result)

def index_cast_ui(target_type: Type, value: Union["ArithValue", Value, int], *, loc: Location = None) -> "ArithValue":
    """Cast between index and unsigned integer types.

    Args:
        target_type: Target type (index or unsigned integer type)
        value: Value to cast
        loc: Optional source location

    Returns:
        ArithValue wrapping the cast result
    """
    loc = maybe_default_loc(loc)
    if isinstance(value, int):
        value = constant(value, loc=loc)
    val = _unwrap_value(value)
    result = _arith.IndexCastUIOp(target_type, val, loc=loc).result
    return ArithValue(result)


def trunc_f(target_type: Type, value: Union["ArithValue", Value], *, loc: Location = None) -> "ArithValue":
    """Truncate floating point value to narrower type (e.g., f32 -> f16).
    
    Args:
        target_type: Target floating point type
        value: Value to truncate
        loc: Optional source location
        
    Returns:
        ArithValue wrapping the truncated result
    """
    loc = maybe_default_loc(loc)
    val = _unwrap_value(value)
    result = _arith.TruncFOp(target_type, val, loc=loc).result
    return ArithValue(result)

def reduce(value: Union["ArithValue", Value], kind: str = "add", *, acc: Optional[Value] = None, loc: Location = None) -> "ArithValue":
    """Perform vector reduction.
    
    Args:
        value: Input vector value
        kind: Reduction kind ("add", "mul", "min", "max", "and", "or", "xor")
        acc: Optional accumulator
        loc: Optional source location
        
    Returns:
        Reduced scalar value wrapped in ArithValue
    """
    from _mlir.dialects import vector as _vector
    
    val = _unwrap_value(value)
    
    # Map string kind to CombiningKind enum
    kind = kind.lower()
    val_type = val.type
    elem_type = _get_element_type(val_type)
    
    is_float = _is_floating_point_type(elem_type)
    
    kind_map = {
        "add": _vector.CombiningKind.ADD,
        "mul": _vector.CombiningKind.MUL,
        "and": _vector.CombiningKind.AND,
        "or": _vector.CombiningKind.OR,
        "xor": _vector.CombiningKind.XOR,
    }
    
    if kind in ["min", "max"]:
        if is_float:
            kind_map["min"] = _vector.CombiningKind.MINIMUMF
            kind_map["max"] = _vector.CombiningKind.MAXIMUMF
        else:
            kind_map["min"] = _vector.CombiningKind.MINSI # Default to signed
            kind_map["max"] = _vector.CombiningKind.MAXSI
            
    if kind not in kind_map:
        raise ValueError(f"Unsupported reduction kind: {kind}")
        
    combining_kind = kind_map[kind]
    
    if acc is not None:
        acc = _unwrap_value(acc)
        op = _vector.ReductionOp(elem_type, combining_kind, val, acc=acc, loc=loc)
    else:
        op = _vector.ReductionOp(elem_type, combining_kind, val, loc=loc)
        
    return ArithValue(op.result)

def _unwrap_value(val):
    """递归unwrap ArithValue，获取底层的 ir.Value"""
    while isinstance(val, ArithValue):
        val = val._value
    return val


def unwrap(
    val: Union["ArithValue", Value, int, float, bool],
    *,
    type: Optional[Type] = None,
    index: bool = False,
    loc: Location = None,
) -> Value:
    """Public helper to unwrap `ArithValue` (and related wrappers) to the underlying `ir.Value`.

    This is the preferred replacement for direct `.value` / `._value` access in tests.
    """
    if isinstance(val, (int, float, bool)):
        return _unwrap_value(constant(val, type=type, index=index, loc=loc))
    return _unwrap_value(val)


def as_value(
    val: Union["ArithValue", Value, int, float, bool],
    *,
    type: Optional[Type] = None,
    index: bool = False,
    loc: Location = None,
) -> Value:
    """Alias for `unwrap`, intended for readability at MLIR builder boundaries."""
    return unwrap(val, type=type, index=index, loc=loc)

def _binary_op(
    lhs: "ArithValue",
    rhs: "ArithValue",
    op: str,
    *,
    loc: Location = None,
) -> "ArithValue":
    """Execute binary operation based on operand types."""
    loc = maybe_default_loc(loc)
    # Coerce operands to ArithValue with matching types
    # If one is already ArithValue, use its type for the other
    if isinstance(lhs, ArithValue) and not isinstance(rhs, ArithValue):
        if isinstance(rhs, (int, float)):
            rhs = constant(rhs, type=lhs._value.type, loc=loc)
        else:
            rhs = ArithValue(rhs)
    elif isinstance(rhs, ArithValue) and not isinstance(lhs, ArithValue):
        if isinstance(lhs, (int, float)):
            lhs = constant(lhs, type=rhs._value.type, loc=loc)
        else:
            lhs = ArithValue(lhs)
    elif not isinstance(lhs, ArithValue) and not isinstance(rhs, ArithValue):
        # Both are raw values, convert both
        if isinstance(lhs, (int, float)):
            lhs = constant(lhs, loc=loc)
        else:
            lhs = ArithValue(lhs)
        if isinstance(rhs, (int, float)):
            rhs = constant(rhs, type=lhs._value.type, loc=loc)
        else:
            rhs = ArithValue(rhs)
    
    # Determine operation suffix based on type
    # For vectors, check the element type
    lhs_type = lhs._value.type if isinstance(lhs, ArithValue) else lhs.type
    element_type = _get_element_type(lhs_type)
    
    op_name = op.capitalize()
    if _is_floating_point_type(element_type):
        op_name += "F"
    elif _is_integer_like_type(element_type):
        if op == "div":
            op_name = "DivSI"  # Signed integer division
        elif op == "mod":
            op_name = "RemSI"  # Signed integer remainder
        else:
            if op in ["and", "or", "xor"]:
                # MLIR op spellings are a bit inconsistent:
                # - AndIOp / OrIOp
                # - XOrIOp (capital 'O')
                if op == "xor":
                    op_name = "XOrI"
                else:
                    op_name = op.capitalize() + "I"  # AndI, OrI
            else:
                op_name += "I"
    else:
        raise NotImplementedError(f"Unsupported operand types for {op}: {lhs_type} (element type: {element_type})")
    
    # Get the operation class
    op_class = getattr(_arith, f"{op_name}Op")

    lhs_val = _unwrap_value(lhs) if isinstance(lhs, ArithValue) else lhs
    rhs_val = _unwrap_value(rhs) if isinstance(rhs, ArithValue) else rhs

    # Vector-scalar promotion (broadcast): MLIR arith ops require identical types.
    # Some bindings don't reliably expose `VectorType` for isinstance checks, so
    # detect vectors structurally.
    try:
        lhs_et = getattr(lhs_val.type, "element_type", None)
        rhs_et = getattr(rhs_val.type, "element_type", None)

        if lhs_et is not None and lhs_et == rhs_val.type:
            rhs_val = _vector.BroadcastOp(lhs_val.type, rhs_val, loc=loc).result
        elif rhs_et is not None and rhs_et == lhs_val.type:
            lhs_val = _vector.BroadcastOp(rhs_val.type, lhs_val, loc=loc).result
    except Exception:
        pass

    result = op_class(lhs_val, rhs_val, loc=loc).result
    
    return ArithValue(result)


def _shift_op(lhs: "ArithValue", rhs: "ArithValue", op: str, *, loc: Location = None) -> "ArithValue":
    """Shift operation for `ArithValue`.

    Notes:
    - `<<` maps to `arith.shli`
    - `>>` maps to `arith.shrui` (logical / unsigned right shift)
    - We keep this separate from `_binary_op` because shifts are not implemented there.
    """
    loc = maybe_default_loc(loc)
    # Coerce operands similar to `_binary_op` so `v << 3` works.
    if isinstance(lhs, ArithValue) and not isinstance(rhs, ArithValue):
        if isinstance(rhs, (int, float)):
            rhs = constant(rhs, type=lhs._value.type, loc=loc)
        else:
            rhs = ArithValue(rhs)
    elif isinstance(rhs, ArithValue) and not isinstance(lhs, ArithValue):
        if isinstance(lhs, (int, float)):
            lhs = constant(lhs, type=rhs._value.type, loc=loc)
        else:
            lhs = ArithValue(lhs)
    elif not isinstance(lhs, ArithValue) and not isinstance(rhs, ArithValue):
        lhs = constant(lhs, loc=loc) if isinstance(lhs, (int, float)) else ArithValue(lhs)
        rhs = constant(rhs, type=lhs._value.type, loc=loc) if isinstance(rhs, (int, float)) else ArithValue(rhs)

    lhs_val = _unwrap_value(lhs)
    rhs_val = _unwrap_value(rhs)

    lhs_type = lhs_val.type
    element_type = _get_element_type(lhs_type)
    if not _is_integer_like_type(element_type):
        raise NotImplementedError(f"Shift not supported for type: {lhs_type} (element type: {element_type})")

    if op == "shl":
        op_class = _arith.ShLIOp
    elif op == "shrui":
        op_class = _arith.ShRUIOp
    else:
        raise ValueError(f"Unknown shift op: {op}")

    return ArithValue(op_class(lhs_val, rhs_val, loc=loc).result)

def _rbinary_op(rhs: "ArithValue", lhs: "ArithValue", op: str, *, loc: Location = None) -> "ArithValue":
    """Reverse binary operation (for right-hand operations)."""
    return _binary_op(lhs, rhs, op, loc=loc)

def _comparison_op(
    lhs: "ArithValue",
    rhs: "ArithValue",
    predicate: str,
    *,
    loc: Location = None,
) -> "ArithValue":
    """Execute comparison operation."""
    loc = maybe_default_loc(loc)
    # Coerce rhs to ArithValue if needed
    if not isinstance(rhs, ArithValue):
        if isinstance(rhs, (int, float)):
            rhs = constant(rhs, type=lhs._value.type if isinstance(lhs, ArithValue) else lhs.type, loc=loc)
        else:
            rhs = ArithValue(rhs)
    
    if _is_floating_point_type(lhs._value.type if isinstance(lhs, ArithValue) else lhs.type):
        # Ordered float comparison
        if predicate in {"eq", "ne"}:
            pred_name = "O" + predicate.upper()  # OEQ, ONE
        else:
            pred_name = "O" + predicate.upper()  # OGT, OLT, OGE, OLE
        pred_attr = getattr(_arith.CmpFPredicate, pred_name)
        lhs_val = _unwrap_value(lhs) if isinstance(lhs, ArithValue) else lhs
        rhs_val = _unwrap_value(rhs) if isinstance(rhs, ArithValue) else rhs
        result = _arith.CmpFOp(pred_attr, lhs_val, rhs_val, loc=loc).result
    elif _is_integer_like_type(lhs._value.type if isinstance(lhs, ArithValue) else lhs.type):
        # Signed integer comparison
        if predicate in {"eq", "ne"}:
            pred_name = predicate  # eq, ne (lowercase)
        else:
            pred_name = "s" + predicate  # slt, sle, sgt, sge (lowercase)
        pred_attr = getattr(_arith.CmpIPredicate, pred_name)
        lhs_val = _unwrap_value(lhs) if isinstance(lhs, ArithValue) else lhs
        rhs_val = _unwrap_value(rhs) if isinstance(rhs, ArithValue) else rhs
        result = _arith.CmpIOp(pred_attr, lhs_val, rhs_val, loc=loc).result
    else:
        raise NotImplementedError(f"Comparison not supported for type: {lhs._value.type if isinstance(lhs, ArithValue) else lhs.type}")
    
    return ArithValue(result)


def cmpu(
    lhs: Union["ArithValue", Value, int],
    rhs: Union["ArithValue", Value, int],
    predicate: str,
    *,
    loc: Location = None,
) -> "ArithValue":
    """Unsigned integer/index comparison.

    This is the "safe" alternative to Python `< <= > >=` on `ArithValue`, which currently lowers
    to **signed** integer predicates (slt/sle/sgt/sge). For address math / indices we usually want
    unsigned predicates (ult/ule/ugt/uge).

    Supported predicates:
    - "ult", "ule", "ugt", "uge"
    """
    # Coerce inputs similarly to `_comparison_op`
    if not isinstance(lhs, ArithValue):
        lhs = constant(lhs, loc=loc) if isinstance(lhs, (int, float)) else ArithValue(lhs)
    if not isinstance(rhs, ArithValue):
        rhs = constant(rhs, type=lhs._value.type, loc=loc) if isinstance(rhs, (int, float)) else ArithValue(rhs)

    lhs_val = _unwrap_value(lhs)
    rhs_val = _unwrap_value(rhs)

    if not _is_integer_like_type(_get_element_type(lhs_val.type)):
        raise NotImplementedError(f"Unsigned compare not supported for type: {lhs_val.type}")

    pred_attr = getattr(_arith.CmpIPredicate, predicate)
    return ArithValue(_arith.CmpIOp(pred_attr, lhs_val, rhs_val, loc=loc).result)


def ult(lhs, rhs, *, loc: Location = None) -> "ArithValue":
    return cmpu(lhs, rhs, "ult", loc=loc)


def ule(lhs, rhs, *, loc: Location = None) -> "ArithValue":
    return cmpu(lhs, rhs, "ule", loc=loc)


def ugt(lhs, rhs, *, loc: Location = None) -> "ArithValue":
    return cmpu(lhs, rhs, "ugt", loc=loc)


def uge(lhs, rhs, *, loc: Location = None) -> "ArithValue":
    return cmpu(lhs, rhs, "uge", loc=loc)
def _minmax_op(
    lhs: "ArithValue",
    rhs: "ArithValue",
    op_type: str,  # "max" or "min"
    *,
    loc: Location = None,
) -> "ArithValue":
    """Execute min/max operation based on operand types."""
    loc = maybe_default_loc(loc)
    # Coerce rhs to ArithValue if needed
    if not isinstance(rhs, ArithValue):
        if isinstance(rhs, (int, float)):
            rhs = constant(rhs, type=lhs._value.type if isinstance(lhs, ArithValue) else lhs.type, loc=loc)
        else:
            rhs = ArithValue(rhs)
    
    # Unwrap values
    lhs_val = _unwrap_value(lhs) if isinstance(lhs, ArithValue) else lhs
    rhs_val = _unwrap_value(rhs) if isinstance(rhs, ArithValue) else rhs
    
    if _is_floating_point_type(lhs_val.type):
        # Float min/max
        if op_type == "max":
            op_class = _arith.MaximumFOp
        else:
            op_class = _arith.MinimumFOp
        result = op_class(lhs_val, rhs_val, loc=loc).result
    elif _is_integer_like_type(lhs_val.type):
        # Integer min/max (signed/unsigned logic could be tricky, default to signed for now)
        # TODO: Add unsigned support if needed
        if op_type == "max":
            op_class = _arith.MaxSIOp
        else:
            op_class = _arith.MinSIOp
        result = op_class(lhs_val, rhs_val, loc=loc).result
    else:
        raise NotImplementedError(f"{op_type} not supported for type: {lhs_val.type}")
    
    return ArithValue(result)

class ArithValue:
    """Value wrapper with operator overloading for Pythonic arithmetic.
    
    Allows writing natural Python expressions like:
        c = a + b       # instead of arith.AddIOp(a, b)
        d = a * 2       # instead of arith.MulIOp(a, constant(2))
        e = a < b       # instead of arith.CmpIOp(...)
    """
    
    def __init__(self, value: Value):
        """Wrap an MLIR Value.

        Note: In some environments we register `ArithValue` as a global MLIR value
        caster. This can cause values returned from MLIR to already be `ArithValue`
        instances. To keep `. _value` stable (always an `ir.Value`), we unwrap
        nested wrappers here.
        """
        object.__setattr__(self, "_value", _unwrap_value(value))
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped value."""
        return getattr(self._value, name)
    
    def __repr__(self):
        return f"ArithValue({self._value})"
    
    # Arithmetic operators
    __add__ = partialmethod(_binary_op, op="add")
    __sub__ = partialmethod(_binary_op, op="sub")
    __mul__ = partialmethod(_binary_op, op="mul")
    __truediv__ = partialmethod(_binary_op, op="div")
    __floordiv__ = partialmethod(_binary_op, op="div")
    __mod__ = partialmethod(_binary_op, op="mod")

    # Bitwise operators
    __and__ = partialmethod(_binary_op, op="and")
    __or__ = partialmethod(_binary_op, op="or")
    __xor__ = partialmethod(_binary_op, op="xor")

    # Shift operators
    __lshift__ = partialmethod(_shift_op, op="shl")
    __rshift__ = partialmethod(_shift_op, op="shrui")

    # Min/Max methods
    max = partialmethod(_minmax_op, op_type="max")
    min = partialmethod(_minmax_op, op_type="min")
    
    # Reverse arithmetic operators (for when left operand is Python type)
    __radd__ = partialmethod(_rbinary_op, op="add")
    __rsub__ = partialmethod(_rbinary_op, op="sub")
    __rmul__ = partialmethod(_rbinary_op, op="mul")
    __rtruediv__ = partialmethod(_rbinary_op, op="div")
    __rfloordiv__ = partialmethod(_rbinary_op, op="div")
    __rmod__ = partialmethod(_rbinary_op, op="mod")

    
    # Comparison operators
    __eq__ = partialmethod(_comparison_op, predicate="eq")
    __ne__ = partialmethod(_comparison_op, predicate="ne")
    __lt__ = partialmethod(_comparison_op, predicate="lt")
    __le__ = partialmethod(_comparison_op, predicate="le")
    __gt__ = partialmethod(_comparison_op, predicate="gt")
    __ge__ = partialmethod(_comparison_op, predicate="ge")
    
    # Allow unwrapping for MLIR operations
    @property
    def value(self) -> Value:
        """Get the underlying MLIR Value (递归unwrap)."""
        return _unwrap_value(self)

# Re-export commonly used arith operations
from _mlir.dialects.arith import (
    AddIOp, AddFOp, SubIOp, SubFOp, MulIOp, MulFOp,
    DivSIOp, DivFOp, RemSIOp, RemFOp,
    CmpIOp, CmpFOp, CmpIPredicate, CmpFPredicate,
    IndexCastOp, ExtSIOp, TruncIOp, ExtFOp, TruncFOp,
    SIToFPOp, FPToSIOp, SelectOp,
)

__all__ = [
    "constant", "unwrap", "as_value", "index", "i32", "i64", "f16", "f32", "f64", "Index",
    "maximum", "minimum", "select", "extf", "fptosi", "sitofp", "absf", "reduce", "constant_vector",
    "andi", "ori", "xori", "shrui", "shli", "index_cast", "index_cast_ui", "trunc_f",
    "ArithValue",
    "AddIOp", "AddFOp", "SubIOp", "SubFOp", "MulIOp", "MulFOp",
    "DivSIOp", "DivFOp", "RemSIOp", "RemFOp",
    "CmpIOp", "CmpFOp", "CmpIPredicate", "CmpFPredicate",
    "IndexCastOp", "ExtSIOp", "TruncIOp", "ExtFOp", "TruncFOp",
    "SIToFPOp", "FPToSIOp", "SelectOp",
]

# Alias for convenience
Index = index

try:
    # Register `ArithValue` as an automatic wrapper for common scalar types so
    # op results can participate in Python operator overloading.
    from _mlir._mlir_libs._mlir import register_value_caster

    # Also include VectorType so vector-typed op results participate in operator overloading.
    for t in [F32Type, F64Type, IndexType, IntegerType, VectorType]:
        register_value_caster(t.static_typeid)(ArithValue)
except Exception:
    # Best-effort only.
    pass
