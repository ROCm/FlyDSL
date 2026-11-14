# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import contextlib
import inspect
import itertools
import os
import threading
import types
from functools import lru_cache, wraps
from typing import TYPE_CHECKING, overload

from .._mlir import ir
from ..utils import env

if TYPE_CHECKING:
    from typing import Any, Callable, ParamSpec, TypeVar

    _P = ParamSpec("_P")
    _R = TypeVar("_R")

__all__ = [
    "capture_user_location",
    "dsl_loc_tracing",
    "dsl_wrap_result",
    "tracing_context",
    "tracing_option",
]

# Package root for the ``flydsl`` Python package: ``.../python/flydsl``.
# Any frame whose file lives under this prefix is treated as DSL library code
# and skipped when locating the user's source position.
_FLYDSL_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Python 3.10 has no column positions.
_HAS_CO_POSITIONS = hasattr(types.CodeType, "co_positions")


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


def _source_path_and_positions(code) -> tuple:
    """Absolute source path and co_positions() of *code*. Positions are None unless a frame with a func is open."""
    cache = getattr(_tls, "source_position_cache", None)

    # Keyed by id, since code objects from different files can compare equal.
    entry = cache.get(id(code)) if cache is not None else None
    if entry is not None:
        return entry[1], entry[2]

    file = inspect.getsourcefile(code) or inspect.getfile(code)
    source_path = os.path.abspath(file)
    if cache is None:
        return source_path, None

    positions = tuple(code.co_positions()) if _HAS_CO_POSITIONS else ()
    # code is part of cache value to keep it alive
    cache[id(code)] = (code, source_path, positions)
    return source_path, positions


def _frame_position(frame) -> tuple:
    """Source path, line and column of *frame*'s current instruction, as inspect.getframeinfo reports them."""
    code = frame.f_code
    path, positions = _source_path_and_positions(code)

    line = None
    col = None
    # co_positions() has one (line, end_line, col, end_col) entry per 2-byte code unit; Python 3.10 has none.
    if _HAS_CO_POSITIONS and frame.f_lasti >= 0:
        index = frame.f_lasti // 2
        if positions is None:
            line, _, col, _ = next(itertools.islice(code.co_positions(), index, None))
        else:
            line, _, col, _ = positions[index]

    return (
        path,
        frame.f_lineno if line is None else line,
        0 if col is None else col,
    )


@contextlib.contextmanager
def tracing_context(func=None, **options):
    """Push a tracing frame for the duration of the ``with`` block.

    *func*, when given, marks the call-site boundary used by
    ``capture_user_location``; a frame entered without one inherits the
    enclosing boundary.  Keyword *options* become ambient settings readable
    with ``tracing_option(key)``.  Every key passed is set, ``None`` included
    -- an explicit ``None`` shadows an enclosing setting.
    """
    stack = _stack()
    # Only the outermost trace owns the cache; a fastmath scope can span traces and would pin their code objects.
    owns_cache = func is not None and getattr(_tls, "source_position_cache", None) is None
    if owns_cache:
        _tls.source_position_cache = {}
        # Read once per trace: capture_user_location runs for every op.
        _tls.max_loc_depth = env.debug.max_loc_depth
        _tls.frame_location_cache = {}
    boundary = getattr(func, "__code__", None)
    if boundary is None and stack:
        boundary = stack[-1][0]
    stack.append((boundary, options))
    try:
        yield
    finally:
        stack = _stack()
        if stack:
            stack.pop()
        if owns_cache:
            _tls.source_position_cache = None
            _tls.max_loc_depth = None
            _tls.frame_location_cache = None


def tracing_option(key, default=None):
    """Return the ambient value of *key*, innermost tracing frame first."""
    stack = getattr(_tls, "stack", None)
    if stack:
        for _, options in reversed(stack):
            if key in options:
                return options[key]
    return default


# Runtime helpers that evaluate rewriter-made thunks (lambdas standing in for operands of a user expression).
_THUNK_RUNNERS = set()


def thunk_runner(fn):
    """Mark *fn* as calling operand thunks, so their frames are not reported as extra call sites."""
    _THUNK_RUNNERS.add(fn.__code__)
    return fn


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
    boundary = stack[-1][0] if stack else None
    max_depth = getattr(_tls, "max_loc_depth", None)
    if max_depth is None:
        max_depth = env.debug.max_loc_depth
    ctx = ir.Context.current
    frame_locs = getattr(_tls, "frame_location_cache", None)
    locs = []
    boundary_loc = None
    dropped = 0
    # Innermost operand thunk seen: its position stands in for the frame that created it.
    thunk = None

    frame = inspect.currentframe().f_back
    try:
        while frame is not None:
            code = frame.f_code
            is_boundary = boundary is not None and code is boundary
            if not _is_framework_file(code.co_filename):
                caller = frame.f_back
                if caller is not None and caller.f_code in _THUNK_RUNNERS:
                    if thunk is None:
                        thunk = frame
                    frame = caller
                    continue
                pos_frame = frame if thunk is None else thunk
                thunk = None
                keep = len(locs) < max_depth
                if keep or is_boundary:
                    # Ops share call sites, so reuse each frame's location within a trace.
                    key = (id(pos_frame.f_code), pos_frame.f_lasti)
                    entry = frame_locs.get(key) if frame_locs is not None else None
                    if entry is None:
                        path, line, col = _frame_position(pos_frame)
                        floc = ir.Location.file(path, line, col, context=ctx)
                        if frame_locs is not None:
                            # code is part of the value to keep it alive
                            frame_locs[key] = (pos_frame.f_code, floc)
                    else:
                        floc = entry[1]
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
        del frame, thunk

    # The kernel boundary frame is always kept as the outermost call-site
    if boundary_loc is not None:
        locs.append(boundary_loc)

    if not locs:
        return ir.Location.unknown()
    callee, callers = locs[0], locs[1:]
    if not callers:
        return callee
    return ir.Location.callsite(callee, callers)


def dsl_loc_tracing(fn: "Callable[_P, _R]") -> "Callable[_P, _R]":
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
    def wrapper(*args: "_P.args", **kwargs: "_P.kwargs") -> "_R":
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


@overload
def dsl_wrap_result(target: "Callable[_P, Any]") -> "Callable[_P, Any]": ...
@overload
def dsl_wrap_result(target: "type | None" = None) -> "Callable[[Callable[_P, Any]], Callable[_P, Any]]": ...
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
