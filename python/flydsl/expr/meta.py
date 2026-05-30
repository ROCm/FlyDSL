# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import inspect
import threading
from functools import wraps

from .._mlir import ir


def _to_raw_value(obj):
    if isinstance(obj, ir.Value):
        return obj
    if isinstance(obj, type):
        return obj
    if hasattr(obj, "__extract_to_ir_values__"):
        values = obj.__extract_to_ir_values__()
        if len(values) != 1:
            raise ValueError(f"Primitive function expects 1 value, got {len(values)}")
        return values[0]
    if isinstance(obj, tuple):
        return tuple(_to_raw_value(e) for e in obj)
    if isinstance(obj, list):
        return [_to_raw_value(e) for e in obj]
    return obj


def _flatten_args(args, kwargs):
    new_args = tuple(_to_raw_value(a) for a in args)
    new_kwargs = {k: _to_raw_value(v) if k not in ("loc", "ip") else v for k, v in kwargs.items()}
    return new_args, new_kwargs


def _caller_location(depth=1):
    """Build an MLIR Location from the Python call-site *depth* frames up."""
    frame = inspect.currentframe()
    for _ in range(depth + 1):
        if frame is not None:
            frame = frame.f_back
    if frame is None:
        return ir.Location.unknown()

    info = inspect.getframeinfo(frame)
    pos = getattr(info, "positions", None)
    line = pos.lineno if pos is not None else info.lineno
    col = (pos.col_offset or 0) if pos is not None else 0
    file_loc = ir.Location.file(info.filename, line, col)

    if info.code_context:
        label = " ".join(ln.strip() for ln in info.code_context)
    else:
        label = info.function
    return ir.Location.name(label, childLoc=file_loc)


_scope = threading.local()


def _pinned_loc():
    """Return the call-site Location pinned by an enclosing ``source_loc`` scope, or None."""
    return getattr(_scope, "cur", None)


class source_loc:
    """Pin the user call-site location for every op emitted inside the ``with`` block.

    A kernel helper that emits device ops is itself one (or more) Python frames above the
    user's scheduling line, so ``traced_op``'s ``_caller_location(depth=1)`` would attribute
    those ops to the helper body instead of the call site. Wrapping the helper body in
    ``with source_loc():`` captures the caller once and makes both untraced ODS builders
    (via MLIR's ambient location) and ``traced_op`` leaves (via the pin) resolve to it.

    Re-entrant: a nested ``source_loc`` is a no-op so the outermost scope wins.
    """

    def __init__(self):
        self._own = _pinned_loc() is None
        # depth 2: hop past __init__ and the helper/wrapper frame to the user call site.
        self._loc = _caller_location(2) if self._own else None

    def __enter__(self):
        if self._own:
            self._loc.__enter__()
            _scope.cur = self._loc
        return self

    def __exit__(self, *exc):
        if self._own:
            try:
                self._loc.__exit__(*exc)
            finally:
                _scope.cur = None
        return False


def source_loc_scope(fn):
    """Decorator form of :class:`source_loc` for a kernel helper.

    Runs the whole helper body inside ``source_loc()`` so every op it emits attributes to
    the helper's call site, without reindenting the body. The ``wrapper`` frame occupies
    the same stack slot the helper body otherwise would, so the captured location resolves
    to the helper's call site.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        with source_loc():
            return fn(*args, **kwargs)

    return wrapper


def traced_op(op):
    @wraps(op)
    def wrapper(*args, **kwargs):
        loc = kwargs.pop("loc", None)
        if loc is None:
            loc = _pinned_loc() or _caller_location(depth=1)
        args, kwargs = _flatten_args(args, kwargs)
        with loc:
            return op(*args, **kwargs)

    return wrapper
