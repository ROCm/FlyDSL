"""Utility functions for IR tests."""


def unwrap_values(*values):
    """Unwrap ArithValue objects to get underlying MLIR Values.
    
    This is needed because ArithValue wraps MLIR Values for operator
    overloading, but some MLIR op builders expect raw MLIR Values.
    
    Args:
        *values: One or more values (may be ArithValue or MLIR Value)
    
    Returns:
        Tuple of unwrapped MLIR Values
    
    Example:
        >>> # In a @flir.jit body, return raw Values:
        >>> # return unwrap_values(result1, result2)
    """
    unwrapped = []
    for val in values:
        # Prefer the unified helper in arith ext (handles ints/bools too).
        from pyflir.dialects.ext import arith

        unwrapped.append(arith.unwrap(val))
    return tuple(unwrapped)

