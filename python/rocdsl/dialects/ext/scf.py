"""Extended scf dialect with convenience wrappers.

This module intentionally provides context-manager wrappers that manage
insertion points and terminators so callers don't need to use
`ir.InsertionPoint(...)` directly.
"""

from typing import Optional, Sequence
from contextlib import contextmanager

from _mlir.ir import (
    Value,
    Location,
    InsertionPoint,
    Block,
)
from _mlir.dialects import scf as _scf

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
    # Unwrap ArithValue-like wrappers.
    if hasattr(start, "value"):
        start = start.value
    if hasattr(stop, "value"):
        stop = stop.value
    if hasattr(step, "value"):
        step = step.value
    
    iter_args = iter_args or []
    iter_args = [a.value if hasattr(a, "value") else a for a in iter_args]
    for_op = _scf.ForOp(start, stop, step, iter_args, loc=loc, ip=ip)

    # Enter the for-op body insertion point for the duration of the context.
    with InsertionPoint(for_op.body):
        try:
            # Yield induction variable and iter args
            if iter_args:
                yield (for_op.induction_variable, *for_op.inner_iter_args)
            else:
                yield for_op.induction_variable
        finally:
            # Ensure scf.for body is terminated.
            block = for_op.body
            if (not block.operations) or not isinstance(block.operations[-1], _scf.YieldOp):
                _scf.YieldOp(list(for_op.inner_iter_args))


@contextmanager
def for_(
    start,
    stop=None,
    step=None,
    iter_args: Optional[Sequence[Value]] = None,
    *,
    loc: Location = None,
    ip: InsertionPoint = None,
):
    """Create an scf.for op and enter its body insertion point.

    This is like `range_`, but yields the `for_op` so callers can access `.results`.
    """
    if loc is None:
        loc = Location.unknown()

    start, stop, step = canonicalize_range(start, stop, step)
    # Unwrap ArithValue-like wrappers.
    if hasattr(start, "value"):
        start = start.value
    if hasattr(stop, "value"):
        stop = stop.value
    if hasattr(step, "value"):
        step = step.value
    iter_args = iter_args or []
    iter_args = [a.value if hasattr(a, "value") else a for a in iter_args]
    for_op = _scf.ForOp(start, stop, step, iter_args, loc=loc, ip=ip)

    with InsertionPoint(for_op.body):
        try:
            yield for_op
        finally:
            block = for_op.body
            if (not block.operations) or not isinstance(block.operations[-1], _scf.YieldOp):
                _scf.YieldOp(list(for_op.inner_iter_args))


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


class IfOp:
    """Context-manager wrapper around scf.if that manages insertion point + yield.

    Typical usage:

      if_op = scf.IfOp(cond)
      with if_op:
          ...  # inserts into then_block

      if_op = scf.IfOp(cond, hasElse=True)
      with if_op.then():
          ...
      with if_op.else_():
          ...
    """

    def __init__(
        self,
        condition: Value,
        results: Optional[Sequence] = None,
        *,
        hasElse: bool = False,
        loc: Location = None,
        ip: InsertionPoint = None,
    ):
        if loc is None:
            loc = Location.unknown()
        results = results or []
        self.op = _scf.IfOp(condition, results, hasElse=hasElse, loc=loc, ip=ip)
        self._ip = None

    def __getattr__(self, name):
        return getattr(self.op, name)

    @contextmanager
    def then(self):
        with InsertionPoint(self.op.then_block):
            try:
                yield self.op.then_block
            finally:
                blk = self.op.then_block
                if (not blk.operations) or not isinstance(blk.operations[-1], _scf.YieldOp):
                    _scf.YieldOp([])

    @contextmanager
    def else_(self):
        if not hasattr(self.op, "else_block") or self.op.else_block is None:
            raise RuntimeError("IfOp has no else block (use hasElse=True)")
        with InsertionPoint(self.op.else_block):
            try:
                yield self.op.else_block
            finally:
                blk = self.op.else_block
                if (not blk.operations) or not isinstance(blk.operations[-1], _scf.YieldOp):
                    _scf.YieldOp([])

    def __enter__(self):
        # Default context is then-block.
        self._ip = InsertionPoint(self.op.then_block)
        self._ip.__enter__()
        return self.op

    def __exit__(self, exc_type, exc, tb):
        if self._ip is not None:
            # Ensure then-block is terminated.
            if exc_type is None:
                blk = self.op.then_block
                if (not blk.operations) or not isinstance(blk.operations[-1], _scf.YieldOp):
                    _scf.YieldOp([])
            self._ip.__exit__(exc_type, exc, tb)
        self._ip = None
        return False


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
from _mlir.dialects.scf import (
    WhileOp,
    YieldOp,
    ExecuteRegionOp,
)

class ForOp(_scf.ForOp):
    """Wrapper around scf.ForOp that supports int arguments and ArithValue."""
    def __init__(self, start, stop, step, iter_args=None, *, loc=None, ip=None):
        # Convert ints to index constants
        if isinstance(start, int):
            start = constant(start, index=True)
        if isinstance(stop, int):
            stop = constant(stop, index=True)
        if isinstance(step, int):
            step = constant(step, index=True)
            
        # Unwrap ArithValues
        if hasattr(start, "value"): start = start.value
        if hasattr(stop, "value"): stop = stop.value
        if hasattr(step, "value"): step = step.value
        
        # Unwrap iter_args
        if iter_args:
            iter_args = [arg.value if hasattr(arg, "value") else arg for arg in iter_args]
            
        super().__init__(start, stop, step, iter_args=iter_args, loc=loc, ip=ip)

__all__ = [
    "range_",
    "for_",
    "if_",
    "IfOp",
    "while_",
    "yield_",
    "ForOp",
    "WhileOp",
    "YieldOp",
    "ExecuteRegionOp",
]
