"""Extended scf dialect with convenience wrappers."""

from typing import Optional, Sequence, Callable
from contextlib import contextmanager

from mlir.ir import (
    Value,
    Location,
    InsertionPoint,
    IndexType,
    Block,
)
from mlir.dialects import scf as _scf

from .arith import constant


def canonicalize_range(start, stop=None, step=None):
    """Canonicalize range parameters similar to Python range()."""
    if step is None:
        step = 1
    if stop is None:
        stop = start
        start = 0
    
    # Convert to Value if needed
    params = []
    for p in [start, stop, step]:
        if isinstance(p, int):
            p = constant(p, index=True)
        params.append(p)
    
    return params[0], params[1], params[2]


@contextmanager
def range_(
    start,
    stop=None,
    step=None,
    iter_args: Optional[Sequence[Value]] = None,
    loc: Location = None,
    ip: InsertionPoint = None,
):
    """Create an scf.for loop with Python-like range semantics.
    
    Args:
        start: Loop start (or stop if stop is None)
        stop: Loop stop (exclusive)
        step: Loop step (default 1)
        iter_args: Loop-carried values
        loc: Location for the operation
        ip: Insertion point
        
    Yields:
        (index, *iter_args) if iter_args provided, else just index
        
    Examples:
        >>> with range_(10) as i:
        ...     # Loop body with index i
        
        >>> with range_(0, 10, 2) as i:
        ...     # Loop from 0 to 10 with step 2
        
        >>> with range_(10, iter_args=[init_val]) as (i, val):
        ...     # Loop with carried value
    """
    if loc is None:
        loc = Location.unknown()
    
    start, stop, step = canonicalize_range(start, stop, step)
    
    iter_args = iter_args or []
    for_op = _scf.ForOp(start, stop, step, iter_args, loc=loc, ip=ip)
    
    with InsertionPoint(for_op.body):
        # Yield induction variable and iter args
        if iter_args:
            yield (for_op.induction_variable, *for_op.inner_iter_args)
        else:
            yield for_op.induction_variable


@contextmanager  
def if_(
    condition: Value,
    results: Optional[Sequence] = None,
    *,
    hasElse: bool = False,
    loc: Location = None,
    ip: InsertionPoint = None,
):
    """Create an scf.if operation.
    
    Args:
        condition: Boolean condition value
        results: Result types for the if operation
        hasElse: Whether to include an else block
        loc: Location for the operation
        ip: Insertion point
        
    Yields:
        (then_block, else_block) if hasElse, else just then_block
        
    Examples:
        >>> with if_(condition) as then_block:
        ...     # Then block code
        
        >>> with if_(condition, hasElse=True) as (then_block, else_block):
        ...     with then_block:
        ...         # Then code
        ...     with else_block:
        ...         # Else code
    """
    if loc is None:
        loc = Location.unknown()
    
    results = results or []
    if_op = _scf.IfOp(condition, results, hasElse=hasElse, loc=loc, ip=ip)
    
    if hasElse:
        yield (if_op.then_block, if_op.else_block)
    else:
        yield if_op.then_block


@contextmanager
def while_(
    before_args: Sequence[Value],
    *,
    loc: Location = None,
    ip: InsertionPoint = None,
):
    """Create an scf.while loop.
    
    Args:
        before_args: Arguments to the before region
        loc: Location for the operation
        ip: Insertion point
        
    Yields:
        (before_block, after_block)
        
    Examples:
        >>> with while_([init]) as (before, after):
        ...     with before:
        ...         # Condition check
        ...     with after:
        ...         # Loop body
    """
    if loc is None:
        loc = Location.unknown()
    
    while_op = _scf.WhileOp(before_args, loc=loc, ip=ip)
    yield (while_op.before, while_op.after)


def yield_(
    operands: Sequence[Value] = None,
    loc: Location = None,
    ip: InsertionPoint = None,
):
    """Create an scf.yield operation.
    
    Args:
        operands: Values to yield
        loc: Location for the operation
        ip: Insertion point
    """
    if loc is None:
        loc = Location.unknown()
    
    operands = operands or []
    return _scf.YieldOp(operands, loc=loc, ip=ip)


# Re-export common scf operations
from mlir.dialects.scf import (
    ForOp,
    IfOp,
    WhileOp,
    YieldOp,
    ExecuteRegionOp,
)

__all__ = [
    "range_",
    "if_",
    "while_",
    "yield_",
    "ForOp",
    "IfOp",
    "WhileOp",
    "YieldOp",
    "ExecuteRegionOp",
]
