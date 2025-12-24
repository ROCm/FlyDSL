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
from _mlir.dialects import arith as _arith

from .arith import constant


def _normalize_if_condition(condition):
    """Best-effort normalization for scf.if conditions.

    Accepts:
    - MLIR Value
    - ArithValue-like wrappers with `.value`
    - Python bool / int (materializes an i1 constant)
    """
    if hasattr(condition, "value"):
        return condition.value
    if isinstance(condition, bool):
        return constant(condition).value
    if isinstance(condition, int):
        return constant(bool(condition)).value
    return condition


def canonicalize_range(start, stop=None, step=None):
    """Canonicalize range parameters similar to Python range()."""
    if step is None:
        step = 1
    if stop is None:
        stop = start
        start = 0

    # Match Python semantics: range(step=0) is an error.
    if isinstance(step, int) and step == 0:
        raise ValueError("range() arg 3 must not be zero")
    
    # Convert to Value if needed
    params = []
    for p in [start, stop, step]:
        if isinstance(p, int):
            p = constant(p, index=True)
        params.append(p)
    
    return params[0], params[1], params[2]


class _ForOpView:
    """A small facade over scf.ForOp that can override the exposed induction var.

    Used to implement full Python `range` semantics (e.g. negative step) while still
    providing a scf.ForOp-like surface area to callers.
    """

    def __init__(self, op: _scf.ForOp, *, induction_variable: Value):
        self.op = op
        self.induction_variable = induction_variable

    def __getattr__(self, name):
        # Delegate everything else to the underlying op.
        return getattr(self.op, name)

    @property
    def inner_iter_args(self):
        return self.op.inner_iter_args

    @property
    def results(self):
        return self.op.results

    @property
    def body(self):
        return self.op.body


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
        # Prefer a file/line location pointing at user code for better IR dumps.
        try:
            from pyflir.dialects.ext.func import get_user_code_loc

            loc = get_user_code_loc()
        except Exception:
            loc = None
        if loc is None:
            loc = Location.unknown()
    
    # Implement Python `range` semantics for negative constant steps.
    # - For positive steps, we emit scf.for(start, stop, step) directly.
    # - For negative steps with *int* bounds, we synthesize:
    #     for t in range(0, trip_count, 1):
    #         i = start + t * step
    #   and expose `i` as the induction variable to the user's body builder.
    # - For negative steps with dynamic bounds, we currently do not implement the
    #   trip-count computation (would require signed comparisons/division).
    if step is None:
        raw_step = 1
    else:
        raw_step = step
    raw_start = start
    raw_stop = stop
    if raw_stop is None:
        raw_stop = raw_start
        raw_start = 0

    if isinstance(raw_step, int) and raw_step < 0:
        if not (isinstance(raw_start, int) and isinstance(raw_stop, int)):
            raise NotImplementedError(
                "scf.range_ negative step currently supports only int bounds; "
                "use a positive step or an explicit scf.while_ for dynamic bounds."
            )

        trip_count = len(range(raw_start, raw_stop, raw_step))
        # Build an increasing scf.for over [0, trip_count).
        start_t = constant(0, index=True).value
        stop_t = constant(trip_count, index=True).value
        step_t = constant(1, index=True).value
        iter_args = iter_args or []
        iter_args = [a.value if hasattr(a, "value") else a for a in iter_args]
        for_op = _scf.ForOp(start_t, stop_t, step_t, iter_args, loc=loc, ip=ip)

        start_i = constant(raw_start, index=True).value
        step_i = constant(raw_step, index=True).value  # negative

        with InsertionPoint(for_op.body):
            try:
                t = for_op.induction_variable
                i = _arith.AddIOp(start_i, _arith.MulIOp(t, step_i).result).result
                if iter_args:
                    yield (i, *for_op.inner_iter_args)
                else:
                    yield i
            finally:
                block = for_op.body
                if (not block.operations) or not isinstance(block.operations[-1], _scf.YieldOp):
                    _scf.YieldOp(list(for_op.inner_iter_args))
        return

    start, stop, step = canonicalize_range(start, stop, step)

    # Unwrap various "Value-like" wrappers down to a real `_mlir.ir.Value`.
    # We need this because our arithmetic helpers often return wrapper objects
    # (e.g. `ArithValue`) which are not accepted as operands by generated op
    # builders (like `_scf.ForOp`).
    def _as_value(v):
        seen = set()
        while True:
            if isinstance(v, Value):
                return v
            obj_id = id(v)
            if obj_id in seen:
                return v
            seen.add(obj_id)
            if hasattr(v, "_value"):
                v = v._value
                continue
            if hasattr(v, "value"):
                v = v.value
                continue
            if hasattr(v, "result"):
                v = v.result
                continue
            return v

    start = _as_value(start)
    stop = _as_value(stop)
    step = _as_value(step)
    
    iter_args = iter_args or []
    iter_args = [_as_value(a) for a in iter_args]
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

    # Mirror `range_` negative-step handling.
    raw_step = 1 if step is None else step
    raw_start = start
    raw_stop = stop
    if raw_stop is None:
        raw_stop = raw_start
        raw_start = 0
    if isinstance(raw_step, int) and raw_step < 0:
        if not (isinstance(raw_start, int) and isinstance(raw_stop, int)):
            raise NotImplementedError(
                "scf.for_ negative step currently supports only int bounds; "
                "use a positive step or an explicit scf.while_ for dynamic bounds."
            )
        trip_count = len(range(raw_start, raw_stop, raw_step))
        start_t = constant(0, index=True).value
        stop_t = constant(trip_count, index=True).value
        step_t = constant(1, index=True).value
        iter_args = iter_args or []
        iter_args = [a.value if hasattr(a, "value") else a for a in iter_args]
        for_op = _scf.ForOp(start_t, stop_t, step_t, iter_args, loc=loc, ip=ip)

        start_i = constant(raw_start, index=True).value
        step_i = constant(raw_step, index=True).value  # negative
        with InsertionPoint(for_op.body):
            try:
                t = for_op.induction_variable
                i = _arith.AddIOp(start_i, _arith.MulIOp(t, step_i).result).result
                yield _ForOpView(for_op, induction_variable=i)
            finally:
                block = for_op.body
                if (not block.operations) or not isinstance(block.operations[-1], _scf.YieldOp):
                    _scf.YieldOp(list(for_op.inner_iter_args))
        return

    start, stop, step = canonicalize_range(start, stop, step)
    # Unwrap various "Value-like" wrappers down to a real `_mlir.ir.Value`.
    def _as_value(v):
        seen = set()
        while True:
            if isinstance(v, Value):
                return v
            obj_id = id(v)
            if obj_id in seen:
                return v
            seen.add(obj_id)
            if hasattr(v, "_value"):
                v = v._value
                continue
            if hasattr(v, "value"):
                v = v.value
                continue
            if hasattr(v, "result"):
                v = v.result
                continue
            return v

    start = _as_value(start)
    stop = _as_value(stop)
    step = _as_value(step)
    iter_args = iter_args or []
    iter_args = [_as_value(a) for a in iter_args]
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
    
    condition = _normalize_if_condition(condition)
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
        condition = _normalize_if_condition(condition)
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
