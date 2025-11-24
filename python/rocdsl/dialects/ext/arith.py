"""Arith dialect extensions with operator overloading for Pythonic syntax."""

from functools import partialmethod
from typing import Optional, Union, Tuple
import numpy as np

from mlir.ir import (
    Type, Value, IntegerType, IndexType, F32Type, F64Type,
    DenseElementsAttr, Location, InsertionPoint, ShapedType
)
from mlir.dialects import arith as _arith
from mlir.dialects._ods_common import get_op_result_or_op_results


def _is_integer_like_type(t: Type) -> bool:
    """Check if type is integer-like (including index)."""
    return IntegerType.isinstance(t) or IndexType.isinstance(t)


def _is_floating_point_type(t: Type) -> bool:
    """Check if type is floating point."""
    return F32Type.isinstance(t) or F64Type.isinstance(t)


def _is_index_type(t: Type) -> bool:
    """Check if type is index."""
    return IndexType.isinstance(t)


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


def f32(value: float, *, loc: Location = None, ip: InsertionPoint = None) -> "ArithValue":
    """Create an f32 constant."""
    return constant(value, type=F32Type.get(), loc=loc, ip=ip)


def f64(value: float, *, loc: Location = None, ip: InsertionPoint = None) -> "ArithValue":
    """Create an f64 constant."""
    return constant(value, type=F64Type.get(), loc=loc, ip=ip)


def _binary_op(
    lhs: "ArithValue",
    rhs: "ArithValue",
    op: str,
    *,
    loc: Location = None,
) -> "ArithValue":
    """Execute binary operation based on operand types."""
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
    op_name = op.capitalize()
    if _is_floating_point_type(lhs._value.type if isinstance(lhs, ArithValue) else lhs.type):
        op_name += "F"
    elif _is_integer_like_type(lhs._value.type if isinstance(lhs, ArithValue) else lhs.type):
        if op == "div":
            op_name = "DivSI"  # Signed integer division
        elif op == "mod":
            op_name = "RemSI"  # Signed integer remainder
        else:
            op_name += "I"
    else:
        raise NotImplementedError(f"Unsupported operand types for {op}: {lhs._value.type if isinstance(lhs, ArithValue) else lhs.type}")
    
    # Get the operation class
    op_class = getattr(_arith, f"{op_name}Op")
    result = op_class(lhs._value if isinstance(lhs, ArithValue) else lhs, rhs._value if isinstance(rhs, ArithValue) else rhs, loc=loc).result
    
    return ArithValue(result)


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
        lhs_val = lhs._value if isinstance(lhs, ArithValue) else lhs
        rhs_val = rhs._value if isinstance(rhs, ArithValue) else rhs
        result = _arith.CmpFOp(pred_attr, lhs_val, rhs_val, loc=loc).result
    elif _is_integer_like_type(lhs._value.type if isinstance(lhs, ArithValue) else lhs.type):
        # Signed integer comparison
        if predicate in {"eq", "ne"}:
            pred_name = predicate  # eq, ne (lowercase)
        else:
            pred_name = "s" + predicate  # slt, sle, sgt, sge (lowercase)
        pred_attr = getattr(_arith.CmpIPredicate, pred_name)
        lhs_val = lhs._value if isinstance(lhs, ArithValue) else lhs
        rhs_val = rhs._value if isinstance(rhs, ArithValue) else rhs
        result = _arith.CmpIOp(pred_attr, lhs_val, rhs_val, loc=loc).result
    else:
        raise NotImplementedError(f"Comparison not supported for type: {lhs._value.type if isinstance(lhs, ArithValue) else lhs.type}")
    
    return ArithValue(result)


class ArithValue:
    """Value wrapper with operator overloading for Pythonic arithmetic.
    
    Allows writing natural Python expressions like:
        c = a + b       # instead of arith.AddIOp(a, b)
        d = a * 2       # instead of arith.MulIOp(a, constant(2))
        e = a < b       # instead of arith.CmpIOp(...)
    """
    
    def __init__(self, value: Value):
        """Wrap an MLIR Value."""
        # Don't call super().__init__() - Value is not meant to be subclassed normally
        # Instead, we just store the value and delegate attribute access
        object.__setattr__(self, '_value', value)
    
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
        """Get the underlying MLIR Value."""
        return self._value


# Re-export commonly used arith operations
from mlir.dialects.arith import (
    AddIOp, AddFOp, SubIOp, SubFOp, MulIOp, MulFOp,
    DivSIOp, DivFOp, RemSIOp, RemFOp,
    CmpIOp, CmpFOp, CmpIPredicate, CmpFPredicate,
    IndexCastOp, ExtSIOp, TruncIOp, ExtFOp, TruncFOp,
    SIToFPOp, FPToSIOp, SelectOp,
)

__all__ = [
    "constant", "index", "i32", "i64", "f32", "f64",
    "ArithValue",
    "AddIOp", "AddFOp", "SubIOp", "SubFOp", "MulIOp", "MulFOp",
    "DivSIOp", "DivFOp", "RemSIOp", "RemFOp",
    "CmpIOp", "CmpFOp", "CmpIPredicate", "CmpFPredicate",
    "IndexCastOp", "ExtSIOp", "TruncIOp", "ExtFOp", "TruncFOp",
    "SIToFPOp", "FPToSIOp", "SelectOp",
]
