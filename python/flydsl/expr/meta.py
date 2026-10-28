# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import contextlib
import inspect
import os
import sysconfig
import threading
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

# The Python standard library is skipped too: ``ArithValue``'s operators are
# ``functools.partialmethod``s, whose wrapper frame would otherwise be the
# innermost "user" frame of every arithmetic op.
_STDLIB_ROOTS = tuple(
    os.path.abspath(p) + os.sep
    for p in {sysconfig.get_paths().get("stdlib"), sysconfig.get_paths().get("platstdlib")}
    if p
)


@lru_cache(maxsize=1024)
def _is_framework_file(filename: str) -> bool:
    """True if *filename* belongs to DSL library code or the Python standard
    library (or is not locatable)."""
    if not filename or filename[0] == "<":
        # ``<string>``, ``<frozen ...>`` and similar synthetic names are not
        # user source we can point at.
        return True
    path = os.path.abspath(filename)
    if path.startswith(_FLYDSL_PKG_ROOT):
        return True
    return path.startswith(_STDLIB_ROOTS) and f"{os.sep}site-packages{os.sep}" not in path


def _escape_attr_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


# Scopes built for the current context, so a function has one subprogram per
# compilation; only the last context is held.
_scope_cache = [None, {}]


def _frame_scope(code, context) -> ir.Attribute:
    """A debug scope for the Python function of *code*: a ``DILexicalBlockFile``
    in a ``DISubprogram`` named after it.

    MLIR's debug translation gives a call-site frame a scope only if its
    location carries one, and drops the frames that have none, so without it
    every frame between the innermost and the kernel's collapses into the
    kernel's call line. A ``DILexicalBlockFile`` rather than the subprogram
    itself: ``ensure-debug-info-scope-on-llvm-func`` takes any
    ``FusedLoc<DISubprogramAttr>`` inside a function op's location as that
    function's subprogram, which breaks a function declaration created under a
    call chain. Each subprogram carries its own compile unit, as a distinct id
    in parsed attribute text is local to that parse.
    """
    if _scope_cache[0] is not context:
        _scope_cache[:] = [context, {}]
    key = (code.co_filename, code.co_firstlineno, code.co_name)
    if key not in _scope_cache[1]:
        _scope_cache[1][key] = _parse_scope(
            code.co_filename, getattr(code, "co_qualname", code.co_name), code.co_firstlineno, context
        )
    return _scope_cache[1][key]


def _parse_scope(filename: str, name: str, line: int, context) -> ir.Attribute:
    path = os.path.abspath(filename)
    f = (
        f'#llvm.di_file<"{_escape_attr_string(os.path.basename(path))}" in '
        f'"{_escape_attr_string(os.path.dirname(path))}">'
    )
    cu = (
        f"#llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = {f}, "
        f'producer = "FlyDSL", isOptimized = true, emissionKind = LineTablesOnly>'
    )
    return ir.Attribute.parse(
        f"#llvm.di_lexical_block_file<scope = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = {cu}, "
        f'scope = {f}, name = "{_escape_attr_string(name)}", file = {f}, line = {line}, scopeLine = {line}, '
        f'subprogramFlags = "Definition|Optimized", type = #llvm.di_subroutine_type<callingConvention = '
        f"DW_CC_normal>>, file = {f}, discriminator = 0>",
        context=context,
    )


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
def tracing_context(func=None, **options):
    """Push a tracing frame for the duration of the ``with`` block.

    *func*, when given, marks the call-site boundary used by
    ``capture_user_location``; a frame entered without one inherits the
    enclosing boundary.  Keyword *options* become ambient settings readable
    with ``tracing_option(key)``.  Every key passed is set, ``None`` included
    -- an explicit ``None`` shadows an enclosing setting.
    """
    stack = _stack()
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


def tracing_option(key, default=None):
    """Return the ambient value of *key*, innermost tracing frame first."""
    stack = getattr(_tls, "stack", None)
    if stack:
        for _, options in reversed(stack):
            if key in options:
                return options[key]
    return default


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
    max_depth = env.debug.max_loc_depth
    ctx = ir.Context.current
    locs = []
    codes = []  # the code object of each entry of ``locs``
    boundary_loc = None
    boundary_code = None
    reached_boundary = False
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
                        codes.append(code)
                    else:
                        # Always keep the kernel frame: it is the top of the stack.
                        boundary_loc = floc
                        boundary_code = code
                else:
                    dropped += 1
            if is_boundary:
                reached_boundary = True
                break
            frame = frame.f_back
    finally:
        del frame

    # The kernel boundary frame is always kept as the outermost call-site
    if boundary_loc is not None:
        locs.append(boundary_loc)
        codes.append(boundary_code)

    # With debug info, every frame but the kernel's (the last) carries the scope
    # of its own function, so each becomes one inlined level in the DWARF. Only
    # when the chain reaches the kernel frame: an outermost frame with a foreign
    # subprogram would not belong to the kernel function.
    if env.debug.enable_debug_info and reached_boundary and len(locs) > 1:
        locs = [
            ir.Location.fused([loc], metadata=_frame_scope(c, ctx), context=ctx)
            for loc, c in zip(locs[:-1], codes[:-1])
        ] + [locs[-1]]

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
