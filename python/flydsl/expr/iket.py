# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""IKET -- in-kernel event tracing for gfx942/gfx950.

A kernel author annotates phases with :func:`mark`, :func:`range_push` /
:func:`range_pop` and :func:`range_start` / :func:`range_end`; each annotation
records a timestamped event, and a post-processing step turns those events into a
timeline::

    fx.iket.range_push("mainloop")
    for k_tile in range(0, k_tiles, 1):
        fx.iket.range_push("k_tile", k_tile)
        ...
        fx.iket.range_pop()
    fx.iket.range_pop()

**Disabled by default.**  Without ``FLYDSL_IKET_ENABLE=1`` (or the ``iket`` compile
hint) every entry point returns before emitting anything, so an annotated kernel
compiles to the same code as an unannotated one.

Records land in a device buffer that the runtime allocates and reads back; see
``flydsl.utils.iket_runtime``.  Each wave claims its slot range once at kernel
entry, so the per-event cost is a timestamp and a store -- no atomic.
"""

from __future__ import annotations

from typing import Any

from .meta import dsl_loc_tracing

# Record layout, 32 bytes, two dwordx4 stores.  The two 64-bit fields are adjacent
# so the backend can merge them; see docs.
#
#   u64 ts          offset 0
#   u64 payload     offset 8
#   u32 hw_id       offset 16
#   u32 xcc_id      offset 20
#   u32 event_id:24 | kind:8   offset 24
#   u32 start_slot  offset 28
RECORD_BYTES = 32

# Event kinds, stored in the top byte of the packed dword.
KIND_MARK = 0
KIND_RANGE_PUSH = 1
KIND_RANGE_POP = 2
KIND_RANGE_START = 3
KIND_RANGE_END = 4

# The trace buffer arrives as an implicit kernel argument; its first record slot holds
# the wave slot counter, so no device global is needed at all.
CURSOR_SLOT_BYTES = RECORD_BYTES


def _enabled() -> bool:
    """Whether this compilation should emit instrumentation.

    Single source of truth: the ``iket`` compile hint wins, the ``FLYDSL_IKET_ENABLE``
    environment variable is the default.  Nothing copies the env var into the hints
    dict, so reading only the hints would make ``FLYDSL_IKET_ENABLE=1`` silently do
    nothing -- a clean run and an empty trace.
    """
    from ..compiler.kernel_function import CompilationContext
    from ..utils import env

    hints = CompilationContext.get_compile_hints() or {}
    if "iket" in hints:
        return bool(hints["iket"])
    return bool(env.iket.enable)


def tracing_enabled() -> bool:
    """Whether this compilation emits instrumentation.  See :func:`_enabled`."""
    return _enabled()


class RangeToken:
    """Handle pairing a :func:`range_start` with its :func:`range_end`.

    Carries two SSA values -- the buffer slot its START record occupies, and an ``i1``
    marking a sentinel -- plus a Python-side event name that never becomes IR.

    Implements the ``DslType`` protocol (``flydsl.compiler.protocol``) so the token can
    be **loop-carried through** ``scf.for``.  That is the whole point of the token model
    and is what makes the cross-iteration ``sentinel_token`` idiom work.

    Both values must be real ``ir.Value``s, including on the disabled path: FlyDSL calls
    ``get_ir_types`` on carried variables at the loop boundary, and a bare Python ``bool``
    there raises ``AttributeError`` before ``range_end`` is ever reached.
    """

    __slots__ = ("_slot", "_is_none", "_event_name")

    def __init__(self, slot, is_none, event_name: str | None = None):
        self._slot = slot
        self._is_none = is_none
        self._event_name = event_name

    # -- DslType protocol ---------------------------------------------------

    def __extract_to_ir_values__(self) -> list:
        return [self._slot, self._is_none]

    @classmethod
    def __construct_from_ir_values__(cls, values: list, exemplar: "RangeToken | None" = None) -> "RangeToken":
        return cls(values[0], values[1], exemplar._event_name if exemplar is not None else None)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RangeToken(event_name={self._event_name!r})"


def _i32(value: int):
    from .numeric import Int32

    return Int32(value)


def _i1_const(value: bool):
    from .._mlir import ir
    from .._mlir.dialects import arith

    i1 = ir.IntegerType.get_signless(1)
    return arith.ConstantOp(i1, ir.IntegerAttr.get(i1, int(value))).result


def _disabled_token(event_name: str | None) -> RangeToken:
    """A token that emits nothing but still threads through ``scf.for``.

    ``_slot`` and ``_is_none`` are real SSA constants, not Python values: the loop
    machinery reads their ``ir.Type`` before any ``range_end`` runs.  A statically-true
    ``_is_none`` const-folds the guard in :func:`range_end`, and the dead constants are
    removed by the ``canonicalize`` already in the pipeline.
    """
    return RangeToken(_i32(0).ir_value(), _i1_const(True), event_name)


@dsl_loc_tracing
def mark(event_name: str, payload: Any = None) -> None:
    """Record a point event.

    :param event_name: name shown in the trace; reuse it for a recurring phase.
    :param payload: optional scalar recorded with the event.  Wave-uniform values are
        expected; under divergence the first active lane's value is recorded.
    """
    if not _enabled():
        return
    _emit_record(event_name, KIND_MARK, payload=payload)


@dsl_loc_tracing
def range_push(event_name: str, payload: Any = None) -> None:
    """Open a stack range, closed by the next :func:`range_pop`.

    Push/pop pair LIFO along an executed path.  Pushing in one branch of an ``if`` and
    popping in the other is fine: each executed path pushes once and pops once.
    """
    if not _enabled():
        return
    _emit_record(event_name, KIND_RANGE_PUSH, payload=payload)


@dsl_loc_tracing
def range_pop() -> None:
    """Close the innermost open :func:`range_push`."""
    if not _enabled():
        return
    _emit_record(None, KIND_RANGE_POP)


@dsl_loc_tracing
def range_start(event_name: str, payload: Any = None) -> RangeToken:
    """Open a token range, closed by :func:`range_end` with the returned token.

    Prefer this over push/pop when the close site is not lexically nested -- notably
    across loop iterations, where the token is carried by ``scf.for``.
    """
    if not _enabled():
        return _disabled_token(event_name)
    slot = _emit_record(event_name, KIND_RANGE_START, payload=payload)
    return RangeToken(slot, _i1_const(False), event_name)


@dsl_loc_tracing
def range_end(token: RangeToken, payload: Any = None) -> None:
    """Close the range opened by *token*.  A sentinel token emits nothing."""
    if not _enabled():
        return
    _emit_record(token._event_name, KIND_RANGE_END, payload=payload, start_slot=token, guard=token)


@dsl_loc_tracing
def sentinel_token(event_name: str) -> RangeToken:
    """A token that emits no event, for initialising a loop-carried range.

    Use when a ``range_end`` precedes the ``range_start`` it pairs with in source order::

        tok = fx.iket.sentinel_token("mma_k_tile")
        for k_tile in range(0, k_tiles, 1):
            fx.iket.range_end(tok)          # no-op on the first iteration
            tok = fx.iket.range_start("mma_k_tile")
            ...
        fx.iket.range_end(tok)
    """
    if not _enabled():
        return _disabled_token(event_name)
    return RangeToken(_i32(0).ir_value(), _i1_const(True), event_name)


def _emit_record(event_name, kind, *, payload=None, start_slot=None, guard=None):
    """Emit one trace record; returns its slot as an ``Int32`` for START events."""
    from .._mlir import ir
    from .._mlir.dialects import scf
    from .iket_emit import emit_record

    if guard is None:
        return emit_record(event_name, kind, payload=payload, start_slot=start_slot)

    # A sentinel token must not record.  The flag is a compile-time constant at each
    # construction site but becomes a genuine loop-carried block argument once scf.for
    # merges a sentinel initialiser with an in-body range_start, so the guard has to be
    # real IR rather than a Python `if`.
    from .numeric import Boolean

    cond = Boolean(guard._is_none) == Boolean(False)
    if_op = scf.IfOp(cond.ir_value(), [], has_else=False)
    if len(if_op.regions[0].blocks) == 0:
        if_op.regions[0].blocks.append(*[])
    with ir.InsertionPoint(if_op.regions[0].blocks[0]):
        emit_record(event_name, kind, payload=payload, start_slot=start_slot)
        scf.YieldOp([])
    return None


def collect(kernel: str = "kernel", out_dir=None) -> str | None:
    """Read back the last traced launch and write a Chrome Trace file.

    Call after the launch has completed (the read synchronises implicitly through
    ``hipMemcpy``).  Returns the trace path, or None when nothing was traced.

    Raises if waves claimed more slots than the buffer holds: a partial trace has
    unbalanced push/pop and would render as a corrupt timeline, so it is rejected
    rather than trimmed.
    """
    from ..utils import iket_runtime, iket_trace
    from . import iket_emit

    buf = iket_emit.current_buffer()
    if buf is None:
        return None

    records = buf.read_records()
    if not records:
        return None
    return iket_trace.write_trace(
        records,
        iket_emit.name_table(),
        kernel=kernel,
        out_dir=out_dir or iket_runtime.dump_dir(),
    )


def summary() -> dict | None:
    """Per-phase totals for the last traced launch, without writing a file."""
    from ..utils import iket_trace
    from . import iket_emit

    buf = iket_emit.current_buffer()
    if buf is None:
        return None
    return iket_trace.summarize(buf.read_records(), iket_emit.name_table())
