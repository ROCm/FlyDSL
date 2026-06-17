# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import contextlib
import inspect
import os
import threading
from functools import lru_cache, wraps

from .._mlir import ir
from ..utils import env

__all__ = [
    "capture_user_location",
    "dsl_loc_tracing",
    "dsl_wrap_result",
    "tracing_context",
]

# Package root for the ``flydsl`` Python package: ``.../python/flydsl``.
# Any frame whose file lives under this prefix is treated as DSL library code
# and skipped when locating the user's source position.
_FLYDSL_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@lru_cache(maxsize=1024)
def _is_framework_file(filename: str) -> bool:
    """True if *filename* belongs to DSL library code (or is not locatable)."""
    if not filename or filename[0] == "<":
        # ``<string>``, ``<frozen ...>`` and similar synthetic names are not
        # user source we can point at.
        return True
    return os.path.abspath(filename).startswith(_FLYDSL_PKG_ROOT)


# --------------------------------------------------------------------------- #
# Tracing Variable (thread-local)
# --------------------------------------------------------------------------- #
_tls = threading.local()


def _stack():
    s = getattr(_tls, "stack", None)
    if s is None:
        s = _tls.stack = []
    return s


@contextlib.contextmanager
def tracing_context(func):
    _stack().append(getattr(func, "__code__", None))
    try:
        yield
    finally:
        stack = _stack()
        if stack:
            stack.pop()


def file_location(filename: str, line: int, col: int = 0, context=None) -> ir.Location:
    ctx = context or ir.Context.current
    if filename and not filename.startswith("<"):
        filename = os.path.abspath(filename)
    return ir.Location.file(filename, line, col, context=ctx)


def capture_user_location() -> ir.Location:
    """Build a ``CallSiteLoc`` chain over the *user* frames.

    Walks up from the op-building site, skips DSL-library frames, and records
    every user frame from the innermost (where the op is written) up to the
    tracing boundary.
    """
    stack = getattr(_tls, "stack", None)
    boundary = stack[-1] if stack else None
    max_depth = env.debug.max_loc_depth
    ctx = ir.Context.current
    locs = []
    boundary_loc = None
    dropped = 0

    frame = inspect.currentframe().f_back
    try:
        while frame is not None:
            code = frame.f_code
            is_boundary = boundary is not None and code is boundary
            if not _is_framework_file(code.co_filename):
                keep = len(locs) < max_depth
                if keep or is_boundary:
                    info = inspect.getframeinfo(frame, context=0)
                    # ``Traceback.positions`` only exists on Python 3.11+; fall
                    # back to ``f_lineno`` / col 0 on 3.8-3.10.
                    pos = getattr(info, "positions", None)
                    line = pos.lineno if pos is not None and pos.lineno is not None else frame.f_lineno
                    col = pos.col_offset if pos is not None and pos.col_offset is not None else 0
                    floc = file_location(info.filename, line, col, context=ctx)
                    if keep:
                        locs.append(floc)
                    else:
                        # Always keep the kernel frame: it is the top of the stack.
                        boundary_loc = floc
                else:
                    dropped += 1
            if is_boundary:
                break
            frame = frame.f_back
    finally:
        del frame

    # The kernel boundary frame is always kept as the outermost call-site
    if boundary_loc is not None:
        locs.append(boundary_loc)

    if not locs:
        return ir.Location.unknown()
    callee, callers = locs[0], locs[1:]
    if not callers:
        return callee
    return ir.Location.callsite(callee, callers)


def dsl_loc_tracing(fn):
    """Attach a source ``Location`` to the op(s) a primitive builds.

    Location policy (single source of truth for the whole ``expr`` layer):

    * The location is the **full user call-site chain** -- a ``CallSiteLoc``
      from the innermost user frame (where the op is written) up to the
      boundary, each frame a ``FileLineColLoc`` (line + column).  A lone user
      frame collapses to a plain ``FileLineColLoc``.
    * It is captured **once** and entered as a dynamic ``with loc:`` scope, so
      every op the decorated function builds inherits it via
      ``Location.current``.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if getattr(_tls, "active_loc", None) is not None:
            # Already inside a captured scope
            return fn(*args, **kwargs)
        loc = capture_user_location()
        _tls.active_loc = loc
        try:
            with loc:
                return fn(*args, **kwargs)
        finally:
            _tls.active_loc = None

    return wrapper


def dsl_wrap_result(target=None):
    """Wrap the op result(s) back into DslType values.

    - ``target=None`` (default): dispatch by the result's ``ir.Type``.
    - ``target=SomeClass``: force ``SomeClass(value)`` — useful when the result
      type cannot be uniquely determined from the ``ir.Type`` (vectors, …).

    Multi-value returns (tuples / lists) are wrapped element-wise.
    """

    def decorator(op, target):
        @wraps(op)
        def wrapper(*args, **kwargs):
            from .typing import as_dsl_value

            return as_dsl_value(op(*args, **kwargs), target)

        return wrapper

    if inspect.isfunction(target):
        return decorator(target, None)
    return lambda op: decorator(op, target)
