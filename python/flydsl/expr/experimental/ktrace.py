# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""KTRACE -- in-kernel wave tracing for gfx942/gfx950.

**Experimental.**  Lives under ``flydsl.expr.experimental``: the API, the 32-byte
record layout and the ``FLYDSL_KTRACE_*`` settings may change in any minor release.

A kernel author annotates phases with :func:`mark`, :func:`range_push` /
:func:`range_pop` and :func:`range_start` / :func:`range_end`; each annotation
records a timestamped event, and a post-processing step turns those events into a
timeline::

    from flydsl.expr.experimental import ktrace

    ktrace.range_push("mainloop")
    for k_tile in range(0, k_tiles, 1):
        ktrace.range_push("k_tile", k_tile)
        ...
        ktrace.range_pop()
    ktrace.range_pop()

**Disabled by default.**  Without ``FLYDSL_KTRACE_ENABLE=1`` (or the ``ktrace`` compile
hint) every entry point returns before emitting anything, so an annotated kernel
compiles to the same code as an unannotated one.

This module only *annotates*: each entry point emits one ``fly_ktrace`` op, and
``convert-fly-ktrace-to-rocdl`` expands it into the code that writes a record --
the leader guard, the slot claim, the bounds checks and the stores.  See
``lib/Conversion/FlyKtraceToROCDL`` for that expansion and
``compiler/backends/rocm.py`` for where the pass runs.

Records land in a device buffer that the runtime allocates and reads back; see
``flydsl.runtime.device_runtime.ktrace_buffer``.  The one *contended* atomic is hoisted to the
kernel entry, where each wave claims its whole slot range at once; per event the
cost is a timestamp, an uncontended bump of that wave's own counter, and the
record store.
"""

from __future__ import annotations

import threading
from typing import Any

from ..meta import dsl_loc_tracing

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

# Slots each wave reserves with its one prologue atomic. One of them holds the wave's
# own event counter, so a wave records at most EVENTS_PER_WAVE - 1 events.
#
# Fixed rather than configurable. It was an env var, and being read independently by the
# emitter, the decoder and the JIT cache key made it a recurring source of silent wrong
# data: a value changed between launch and collect() made the decoder skip real records
# as counter words, and a value of 1 emitted the guard `local < 0`, suppressing every
# record while the counter still climbed. Users cannot judge a good value anyway -- the
# knob that matters when a trace overflows is the buffer size, which stays configurable.
#
# Mirrored by kEventsPerWave in lib/Conversion/FlyKtraceToROCDL/FlyKtraceToROCDL.cpp;
# tests/unit/test_ktrace_emit.py pins the two together.
EVENTS_PER_WAVE = 256

# Slot id for a range whose START record was never written -- the store was suppressed by
# a bounds guard, or the lane was not the recording leader. It must be a value no real
# slot can take: start_slot is decoded as an unsigned 32-bit field, and a buffer large
# enough to reach this id would need 128 GiB, far past any device. Pairing on it is what
# a stale, in-range slot number would corrupt, so the host drops these instead.
UNPAIRED_SLOT = 0xFFFFFFFF

# Records read back from the device, held until the next launch drops them (the jit layer
# calls invalidate_records below). Lets collect() and summary() both describe the SAME
# launch: the device read is destructive, so without this the second call saw an empty
# buffer.
_CACHED_RECORDS: list | None = None

# Serialises a read of the buffer against the launches that fill it. read_records() and
# the reset() that re-arms it are two separate device operations, and a traced launch
# landing between them writes records that reset() then wipes -- the caller keeps the
# records from BEFORE its own launch, and the next read returns them again, so nothing
# reveals the loss. hipDeviceSynchronize cannot close that window: it waits for work
# already submitted, not for work submitted while it runs.
#
# Held across the whole read-reset pair, and taken again by the jit layer around a traced
# dispatch, so the two cannot interleave. Reentrant because a launch from inside a
# collection path would otherwise deadlock on a lock this thread already holds.
_COLLECT_LOCK = threading.RLock()


def collection_lock():
    """The mutex guarding a trace read against the launches that fill the buffer.

    Exposed for the jit layer, which takes it around a traced dispatch. See
    :data:`_COLLECT_LOCK` for what the window is.
    """
    return _COLLECT_LOCK


def _enabled() -> bool:
    """Whether this compilation should emit instrumentation.

    Single source of truth: the ``ktrace`` compile hint wins, the ``FLYDSL_KTRACE_ENABLE``
    environment variable is the default.  Nothing copies the env var into the hints
    dict, so reading only the hints would make ``FLYDSL_KTRACE_ENABLE=1`` silently do
    nothing -- a clean run and an empty trace.
    """
    from ...compiler.kernel_function import CompilationContext
    from ...utils import env

    hints = CompilationContext.get_compile_hints() or {}
    if "ktrace" in hints:
        return bool(hints["ktrace"])
    return bool(env.ktrace.enable)


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


def _i1_const(value: bool):
    from ..._mlir import ir
    from ..._mlir.dialects import arith

    i1 = ir.IntegerType.get_signless(1)
    return arith.ConstantOp(i1, ir.IntegerAttr.get(i1, int(value))).result


def _disabled_token(event_name: str | None) -> RangeToken:
    """A token that emits nothing but still threads through ``scf.for``.

    ``_slot`` and ``_is_none`` are real SSA values, not Python values: the loop
    machinery reads their ``ir.Type`` before any ``range_end`` runs.  A statically-true
    ``_is_none`` const-folds the guard in :func:`range_end`, and the dead values are
    removed by the ``canonicalize`` already in the pipeline.

    A plain i32 constant rather than ``fly_ktrace.sentinel_token``: on this path every
    annotation is skipped, so there is no ``range_start`` for ``scf.for`` to merge this
    with and no type to agree on. Emitting the op here would leave a ``fly_ktrace``
    annotation in a build that is supposed to be byte-identical to an unannotated one,
    and the conversion pass does not run to remove it.
    """
    from ..._mlir import ir
    from ..._mlir.dialects import arith

    i32 = ir.IntegerType.get_signless(32)
    slot = arith.ConstantOp(i32, ir.IntegerAttr.get(i32, 0)).result
    return RangeToken(slot, _i1_const(True), event_name)


def mark(event_name: str, payload: Any = None) -> None:
    """Record a point event.

    :param event_name: name shown in the trace; reuse it for a recurring phase.
    :param payload: optional scalar recorded with the event.  Wave-uniform values are
        expected; under divergence the first active lane's value is recorded.
    """
    if not _enabled():
        return
    _mark_impl(event_name, payload)


@dsl_loc_tracing
def _mark_impl(event_name, payload) -> None:
    _emit_record(event_name, KIND_MARK, payload=payload)


def range_push(event_name: str, payload: Any = None) -> None:
    """Open a stack range, closed by the next :func:`range_pop`.

    Push/pop pair LIFO along an executed path.  Pushing in one branch of an ``if`` and
    popping in the other is fine: each executed path pushes once and pops once.
    """
    if not _enabled():
        return
    _range_push_impl(event_name, payload)


@dsl_loc_tracing
def _range_push_impl(event_name, payload) -> None:
    _emit_record(event_name, KIND_RANGE_PUSH, payload=payload)


def range_pop() -> None:
    """Close the innermost open :func:`range_push`."""
    if not _enabled():
        return
    _range_pop_impl()


@dsl_loc_tracing
def _range_pop_impl() -> None:
    _emit_record(None, KIND_RANGE_POP)


def range_start(event_name: str, payload: Any = None) -> RangeToken:
    """Open a token range, closed by :func:`range_end` with the returned token.

    Prefer this over push/pop when the close site is not lexically nested -- notably
    across loop iterations, where the token is carried by ``scf.for``.
    """
    if not _enabled():
        return _disabled_token(event_name)
    return _range_start_impl(event_name, payload)


@dsl_loc_tracing
def _range_start_impl(event_name, payload) -> RangeToken:
    slot = _emit_record(event_name, KIND_RANGE_START, payload=payload)
    return RangeToken(slot, _i1_const(False), event_name)


def range_end(token: RangeToken, payload: Any = None) -> None:
    """Close the range opened by *token*.  A sentinel token emits nothing."""
    if not _enabled():
        return
    _range_end_impl(token, payload)


@dsl_loc_tracing
def _range_end_impl(token, payload) -> None:
    _emit_record(token._event_name, KIND_RANGE_END, payload=payload, start_slot=token, guard=token)


@dsl_loc_tracing
def sentinel_token(event_name: str) -> RangeToken:
    """A token that emits no event, for initialising a loop-carried range.

    Use when a ``range_end`` precedes the ``range_start`` it pairs with in source
    order, which is what measuring one iteration *up to* the next needs::

        stage = ktrace.sentinel_token("k_tile")
        for i in range(0, k_tiles, 1):
            # Closes the previous iteration's range; the sentinel makes the first
            # pass through here record nothing, since there is no range open yet.
            ktrace.range_end(stage)
            stage = ktrace.range_start("k_tile")
            ...
        ktrace.range_end(stage)
    """
    if not _enabled():
        return _disabled_token(event_name)
    from ..._mlir.dialects import fly_ktrace

    return RangeToken(fly_ktrace.sentinel_token(_token_type()), _i1_const(True), event_name)


def _token_type():
    from ..._mlir import ir

    return ir.Type.parse("!fly_ktrace.token")


def _dialect_payload(payload):
    """Coerce a payload to one i64 operand, or None to omit it.

    The op declares ``Optional<I64>``, so an absent payload omits the operand rather
    than passing a zero: the pass materialises the zero itself, and passing one here
    would make "no payload" indistinguishable from "payload 0" in the IR.
    """
    if payload is None:
        return None
    from ..rocdl.ktrace_emit import _payload_value

    return _payload_value(payload)


def _build_annotation(event_name, kind, *, payload=None, start_slot=None):
    """Build one fly_ktrace annotation. Returns the token value for START events.

    The record-writing form -- leader guard, slot claim, bounds checks, stores -- is
    the pass's job now; see lib/Conversion/FlyKtraceToROCDL.
    """
    from ..._mlir.dialects import fly_ktrace

    operand = _dialect_payload(payload)

    if kind == KIND_MARK:
        fly_ktrace.mark(event_name, payload=operand)
        return None
    if kind == KIND_RANGE_PUSH:
        fly_ktrace.range_push(event_name, payload=operand)
        return None
    if kind == KIND_RANGE_POP:
        # No name: the host takes it from the matching push.
        fly_ktrace.range_pop(payload=operand)
        return None
    if kind == KIND_RANGE_START:
        return fly_ktrace.range_start(_token_type(), event_name, payload=operand)
    if kind == KIND_RANGE_END:
        fly_ktrace.range_end(start_slot._slot, event_name, payload=operand)
        return None
    raise AssertionError(f"unknown ktrace event kind {kind}")


def _emit_record(event_name, kind, *, payload=None, start_slot=None, guard=None):
    """Emit one trace annotation; returns its token for START events."""
    from ..._mlir import ir
    from ..._mlir.dialects import scf

    if guard is None:
        return _build_annotation(event_name, kind, payload=payload, start_slot=start_slot)

    # A sentinel token must not record. The flag is a compile-time constant at each
    # construction site but becomes a loop-carried block argument once scf.for merges a
    # sentinel initialiser with an in-body range_start, so the guard has to be real IR
    # rather than a Python `if` -- and for the same reason the token type does not carry
    # it. See FlyKtrace_TokenType.
    from ..numeric import Boolean

    cond = Boolean(guard._is_none) == Boolean(False)
    if_op = scf.IfOp(cond.ir_value(), [], has_else=False)
    if len(if_op.regions[0].blocks) == 0:
        if_op.regions[0].blocks.append(*[])
    with ir.InsertionPoint(if_op.regions[0].blocks[0]):
        _build_annotation(event_name, kind, payload=payload, start_slot=start_slot)
        scf.YieldOp([])
    return None


def _take_records() -> list | None:
    """Read every record written since the last read, and re-arm the buffer.

    Launches accumulate: the device buffer is zeroed on READ, not on launch, so a
    benchmark loop or autotune sweep that traces repeatedly sees all of its launches
    in one trace rather than only the final one.

    Reading is destructive on the device side: the buffer is zeroed once the records
    are in host memory, so the cursor and every wave counter start the next launch at
    zero.  Without that, each launch's claims accumulate in a process-wide buffer and a
    repeated trace -- a benchmark loop, an autotune sweep -- eventually fails with
    "trace overflowed" even though no single launch came close to the capacity.

    The records are cached host-side until the next launch, so :func:`collect` and
    :func:`summary` can both be called after one launch and describe the same records
    without reading the device twice.

    Returns None when nothing has been traced yet, so the caller can stay quiet.
    """
    global _CACHED_RECORDS
    from ..rocdl import ktrace_emit

    buf = ktrace_emit.current_buffer()
    if buf is None:
        return None

    # The cache is dropped by the jit layer on every launch (invalidate_records),
    # so reaching here with it set means no launch has intervened. The device cursor
    # cannot serve as that signal: a launch whose every wave is excluded by
    # FLYDSL_KTRACE_BLOCKS claims no slots and leaves the cursor at zero, which reads
    # identically to "nothing launched" -- and the caller would then be handed the
    # PREVIOUS kernel's records as if they described this launch.
    # Inside the lock, and so is the cache check above it: a traced launch landing between
    # the check and the read would drop the cache and write records this read then wipes.
    with _COLLECT_LOCK:
        if _CACHED_RECORDS is not None:
            return _CACHED_RECORDS

        try:
            _CACHED_RECORDS = buf.read_records()
            return _CACHED_RECORDS
        finally:
            # Also on the overflow path: leaving a full buffer behind would make every
            # later call raise the same error, hiding whichever launch actually overflowed.
            #
            # A failing reset must not replace the exception on its way out. The two are
            # correlated -- an overflow means the device just wrote out of bounds, which is
            # exactly when hipMemset is likely to fail -- so letting it propagate from the
            # finally would swap the actionable "claimed N slots, buffer holds M, re-run
            # with FLYDSL_KTRACE_BUFFER_BYTES>=..." for a bare HIP error code, precisely
            # when the diagnosis matters most.
            try:
                buf.reset()
            except Exception as exc:  # noqa: BLE001 - the in-flight exception wins
                from ...utils import log

                log().warning(f"ktrace: failed to re-arm the trace buffer: {exc}")


# --- Called by the jit layer -------------------------------------------------------
#
# These three own nothing but ktrace's own state, so they live here rather than in
# jit_function: the cache below is module-private, and the name table and capacity both
# belong to the emitter. The jit layer calls them at the points only it can see -- before
# a launch, and when a cached artifact is restored.


def invalidate_records() -> None:
    """Drop the host-side record cache. Called by the jit layer before every launch.

    Runs on every jit call, so it stays cheap: an uncontended RLock acquire and a
    global store.

    Always takes the collection lock, rather than checking the global first and
    returning early. A read in progress has not published its records yet, so an
    unlocked check would see None, skip, and let that read install records describing a
    buffer state from before this launch -- which is the stale trace this call exists to
    prevent. The lock is uncontended unless a read is actually running.
    """
    global _CACHED_RECORDS
    with _COLLECT_LOCK:
        _CACHED_RECORDS = None


def restore_names(artifact) -> None:
    """Re-publish a cached artifact's ktrace names into the emitter's table."""
    names = getattr(artifact, "_ktrace_names", None)
    if not names:
        return
    from ..rocdl import ktrace_emit

    ktrace_emit.merge_names(names)


def require_matching_capacity(artifact) -> None:
    """Refuse a cached binary whose store bound does not match the live buffer.

    The bound is compiled in, the buffer is allocated once per process, and the cache
    key cannot stand in for either: it records FLYDSL_KTRACE_BUFFER_BYTES, while the
    bound now comes from the buffer that actually existed at compile time. So one key
    can name a binary built for 2047 slots while the process holds a 511-slot buffer,
    and the guard would pass on slots the allocation does not own.
    """
    compiled = getattr(artifact, "_ktrace_capacity", -1)
    if compiled < 0 or not getattr(artifact, "_ktrace_traced", False):
        return
    from ...utils import env as _env
    from ..rocdl import ktrace_emit

    if _env.compile.compile_only:
        return  # nothing launches, so no buffer is allocated to compare against

    # Allocate rather than skip when there is no buffer yet. A fresh process hitting
    # the disk cache reaches here before the first launch, which is exactly the case
    # the check exists for: the binary's bound is fixed and the buffer about to be
    # created may be smaller.
    buffer = ktrace_emit.ensure_buffer()
    if buffer.capacity_slots == compiled:
        return
    raise RuntimeError(
        f"ktrace: this kernel was compiled to bound its stores at {compiled} slots but the "
        f"process holds a {buffer.capacity_slots}-slot buffer. The bound is baked into the "
        "binary and the buffer is allocated once, so re-run with a single "
        "FLYDSL_KTRACE_BUFFER_BYTES, or with FLYDSL_RUNTIME_ENABLE_CACHE=0 to rebuild "
        "against the buffer in use."
    )


def collect(kernel: str = "kernel", out_dir=None) -> str | None:
    """Read back every launch since the last read and write a Chrome Trace file.

    Call after the launch has completed (the read synchronises implicitly through
    ``hipMemcpy``).  Returns the trace path, or None when nothing was traced.

    The records are cached host-side, so :func:`summary` may be called after this (or
    before) and both describe the same set of records.

    Raises if waves claimed more slots than the buffer holds: a partial trace has
    unbalanced push/pop and would render as a corrupt timeline, so it is rejected
    rather than trimmed.
    """
    from ...runtime.device_runtime import ktrace_buffer
    from ...utils import ktrace_trace
    from ..rocdl import ktrace_emit

    records = _take_records()
    if not records:
        # None (never traced) and [] (traced, recorded nothing) both yield no file.
        return None
    return ktrace_trace.write_trace(
        records,
        ktrace_emit.name_table(),
        kernel=kernel,
        out_dir=out_dir or ktrace_buffer.dump_dir(),
    )


def summary() -> dict | None:
    """Per-phase totals for every launch since the last read, without writing a file.

    Reads back the same records :func:`collect` does; either order works, and both
    describe the same set.
    """
    from ...utils import ktrace_trace
    from ..rocdl import ktrace_emit

    records = _take_records()
    if not records:
        # Matches collect(): an empty trace is not a launch with zero phases. Reporting
        # {"phases": {}} read as "the kernel ran and had none", hiding a too-narrow
        # FLYDSL_KTRACE_BLOCKS filter.
        return None
    return ktrace_trace.summarize(records, ktrace_emit.name_table())
