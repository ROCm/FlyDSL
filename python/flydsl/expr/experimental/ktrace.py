# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""KTRACE kernel annotations and host capture API.

**Experimental.**  Lives under ``flydsl.expr.experimental``: the API, the 32-byte
record layout and the ``FLYDSL_KTRACE_*`` settings may change in any minor release.

A kernel author annotates phases with :func:`mark`, :func:`range`,
:func:`range_push` / :func:`range_pop` and :func:`range_start` / :func:`range_end`; each annotation
records a timestamped event, and a post-processing step turns those events into a
timeline::

    from flydsl.expr.experimental import ktrace

    ktrace.range_push("mainloop")
    for k_tile in range(0, k_tiles, 1):
        ktrace.range_push("k_tile", k_tile)
        ...
        ktrace.range_pop()
    ktrace.range_pop()

At the host call site, ``with ktrace.capture("kernel") as trace:`` enables the
annotations for specializations compiled in that block and makes the resulting
snapshot available through ``trace.summary()`` and ``trace.export()``.

**Disabled by default.**  Without ``FLYDSL_KTRACE_ENABLE=1`` (or the ``ktrace`` compile
hint) every entry point returns before emitting anything, so an annotated kernel
compiles to the same code as an unannotated one.

Each annotation emits one ``fly_ktrace`` op. The active compile backend supplies
the pass that expands it, the device buffer, record decoder, and trace writer.
The ROCm implementation supports gfx942/gfx950.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any

from ..meta import dsl_loc_tracing

# Logical event kinds shared by annotations and trace reconstruction.
KIND_MARK = 0
KIND_RANGE_PUSH = 1
KIND_RANGE_POP = 2
KIND_RANGE_START = 3
KIND_RANGE_END = 4

# Records read back from the device, held until the next traced launch drops them.
# Lets collect() and summary() both describe the same traced launches: the device read
# is destructive, so without this the second call saw an empty buffer.
_CACHED_RECORDS: list | None = None
_ACTIVE_CAPTURE = None

# Serialises a read of the buffer against the launches that fill it. read_records() and
# the reset() that re-arms it are two separate device operations, and a traced launch
# landing between them writes records that reset() then wipes -- the caller keeps the
# records from BEFORE its own launch, and the next read returns them again, so nothing
# reveals the loss. Synchronizing the device cannot close that window: another
# thread can submit work between the read and reset.
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


def _trace_backend():
    from ...compiler.backends import get_ktrace_backend

    return get_ktrace_backend()


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

    Carries two SSA values -- the backend's START pairing token, and an ``i1``
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


@contextmanager
def range(event_name: str, payload: Any = None):
    """Record the work in a ``with`` block as one named range.

    This can be used directly in a kernel body. The push and pop follow the
    executed path, including when the body raises during compilation.
    """
    range_push(event_name, payload)
    try:
        yield
    finally:
        range_pop()


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
    from .ktrace_common import payload_value

    return payload_value(payload)


def _build_annotation(event_name, kind, *, payload=None, start_slot=None):
    """Build one fly_ktrace annotation. Returns the token value for START events.

    The record-writing form is the active target's lowering pass's job.
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


def _take_records(backend=None) -> list | None:
    """Read every record written since the last read, and re-arm the buffer.

    Launches accumulate: the device buffer is zeroed on READ, not on launch, so a
    benchmark loop or autotune sweep that traces repeatedly sees all of its launches
    in one trace rather than only the final one.

    Reading is destructive on the device side: the provider resets its buffer after
    copying the records. Without that, repeated launches can exhaust the buffer even
    when no single launch comes close to its capacity.

    The records are cached host-side until the next launch, so :func:`collect` and
    :func:`summary` can both be called after one launch and describe the same records
    without reading the device twice.

    Returns None when nothing has been traced yet, so the caller can stay quiet.
    """
    global _CACHED_RECORDS
    backend = backend or _trace_backend()
    buf = backend.current_buffer()
    if buf is None:
        return None

    # The JIT drops the cache on an eager launch. Graph replay bypasses the JIT, so
    # the provider's launch counter also detects a replay whose filter recorded
    # no events.
    with _COLLECT_LOCK:
        if _CACHED_RECORDS is not None:
            if not buf.launch_count():
                return _CACHED_RECORDS
            _CACHED_RECORDS = None

        try:
            _CACHED_RECORDS = buf.read_records()
            return _CACHED_RECORDS
        finally:
            # Also on the overflow path: leaving a full buffer behind would make every
            # later call raise the same error, hiding whichever launch actually overflowed.
            #
            # A failing reset must not replace the read exception on its way out.
            # The read error gives the actionable overflow diagnosis; the reset
            # error may only report a low-level device failure.
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
    """Drop the host-side record cache before a traced dispatch.

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
    from .ktrace_common import merge_names

    merge_names(names)


def require_compatible_buffer(artifact) -> None:
    """Ask the target to validate a cached binary against its live trace buffer."""
    if not getattr(artifact, "_ktrace_traced", False):
        return
    compiled = getattr(artifact, "_ktrace_buffer_config", None)
    if compiled is None:
        raise RuntimeError("ktrace: cached traced binary has no buffer settings; rebuild its compile cache")
    from ...utils import env as _env

    if _env.compile.compile_only:
        return  # nothing launches, so no buffer is allocated to compare against

    _trace_backend().require_compatible_buffer(compiled)


def collect(kernel: str = "kernel", out_dir=None) -> str | None:
    """Read back every launch since the last read and write a Chrome Trace file.

    Call after the launch has completed. The provider's read synchronizes with
    device work. Returns the trace path, or None when nothing was traced.

    The records are cached host-side, so :func:`summary` may be called after this (or
    before) and both describe the same set of records.

    The provider rejects incomplete captures rather than exporting a timeline
    with unbalanced ranges.
    """
    from .ktrace_common import name_table

    backend = _trace_backend()
    records = _take_records(backend)
    if not records:
        # None (never traced) and [] (traced, recorded nothing) both yield no file.
        return None
    return backend.write_trace(
        records,
        name_table(),
        kernel=kernel,
        out_dir=out_dir or backend.dump_dir(),
    )


def summary() -> dict | None:
    """Per-phase totals for every launch since the last read, without writing a file.

    Reads back the same records :func:`collect` does; either order works, and both
    describe the same set.
    """
    from .ktrace_common import name_table

    backend = _trace_backend()
    records = _take_records(backend)
    if not records:
        # Matches collect(): an empty trace is not a launch with zero phases. Reporting
        # {"phases": {}} read as "the kernel ran and had none", hiding a too-narrow
        # FLYDSL_KTRACE_BLOCKS filter.
        return None
    return backend.summarize(records, name_table())


def _blocks_hint(blocks) -> str | None:
    if blocks is None:
        return None
    if isinstance(blocks, str):
        return blocks
    if isinstance(blocks, (tuple, list)) and len(blocks) == 3 and all(isinstance(x, int) and x >= 0 for x in blocks):
        return ",".join(map(str, blocks))
    raise ValueError("ktrace blocks must be (x, y, z), a backend filter string, or None")


class Capture:
    """One host-side tracing window for launches and graph replays.

    Entering enables tracing in JIT specializations compiled inside the window.
    The context exits with a snapshot of the device records. ``clear()`` discards
    warmup or graph-construction records before replay, and ``read()`` can take a
    fresh snapshot after a later replay of a graph that retains the traced binary.
    """

    def __init__(self, kernel: str = "kernel", *, blocks=None, out_dir=None):
        self.kernel = kernel
        self.blocks = _blocks_hint(blocks)
        self.out_dir = out_dir
        self._backend = None
        self._scope = None
        self._records = None
        self._names = None

    def _provider(self):
        return self._backend or _trace_backend()

    def __enter__(self):
        global _ACTIVE_CAPTURE
        from ...compiler.kernel_function import CompilationContext

        self._backend = _trace_backend()
        self._backend.check_outside_capture()
        with _COLLECT_LOCK:
            if _ACTIVE_CAPTURE is not None:
                raise RuntimeError("ktrace: capture windows cannot be nested")
            _ACTIVE_CAPTURE = self
            try:
                self.clear()
                hints = {"ktrace": True}
                if self.blocks is not None:
                    hints["ktrace_blocks"] = self.blocks
                self._scope = CompilationContext.compile_hints(hints)
                self._scope.__enter__()
            except BaseException:
                _ACTIVE_CAPTURE = None
                raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global _ACTIVE_CAPTURE
        try:
            if exc_type is None:
                self._snapshot()
        finally:
            try:
                self._scope.__exit__(exc_type, exc_value, traceback)
            finally:
                self._scope = None
                with _COLLECT_LOCK:
                    _ACTIVE_CAPTURE = None
        return False

    def clear(self) -> None:
        """Discard records collected so far, for example before graph replay."""
        global _CACHED_RECORDS
        backend = self._provider()

        backend.check_outside_capture()
        with _COLLECT_LOCK:
            if _ACTIVE_CAPTURE is not None and _ACTIVE_CAPTURE is not self:
                raise RuntimeError("ktrace: cannot clear another capture's active trace buffer")
            buf = backend.current_buffer()
            if buf is not None:
                # Warmup may have launched on a non-default stream. Wait for it
                # before zeroing the shared buffer on the default stream.
                buf.synchronize()
                buf.reset()
            _CACHED_RECORDS = None
            self._records = None
            self._names = None

    def read(self):
        """Read and reset the device buffer after launches have finished."""
        global _CACHED_RECORDS
        from .ktrace_common import name_table

        backend = self._provider()
        backend.check_outside_capture()
        with _COLLECT_LOCK:
            if _ACTIVE_CAPTURE is not None and _ACTIVE_CAPTURE is not self:
                raise RuntimeError("ktrace: cannot read another capture's active trace buffer")
            # A graph replay is invisible to JIT cache invalidation. Read the live
            # buffer even when the previous snapshot is still cached host-side.
            _CACHED_RECORDS = None
            self._records = list(_take_records(backend) or [])
            self._names = name_table()
        return self

    def _snapshot(self):
        backend = self._provider()

        backend.check_outside_capture()
        if self._records is None:
            self.read()
        elif _ACTIVE_CAPTURE is self:
            buf = backend.current_buffer()
            if buf is not None and buf.launch_count():
                self.read()
        return self._records

    def summary(self) -> dict | None:
        """Return per-phase totals for this window."""

        records = self._snapshot()
        if not records:
            return None
        return self._provider().summarize(records, self._names)

    def launches(self) -> list[dict]:
        """Return separate phase totals for each kernel launch in this window.

        The provider groups repeated calls, including graph replays. Results
        are ordered by their first recorded timestamp and receive portable
        ``launch_id`` values. Launches with no recorded events are absent.
        """
        backend = self._provider()
        records = self._snapshot() or []
        return [
            {**backend.launch_metadata(identity), **backend.summarize(group, self._names), "launch_id": ordinal}
            for ordinal, (identity, group) in enumerate(backend.group_launches(records), 1)
        ]

    def export_launches(self, names: list[str] | None = None) -> list[str]:
        """Write one Perfetto-compatible file per launch, optionally named by layer.

        ``names`` must contain one label per recorded launch in timestamp order.
        Each file has its own raw-record companion. The usual ``export()`` still
        writes one timeline containing every launch on separate tracks.
        """
        backend = self._provider()
        groups = backend.group_launches(self._snapshot() or [])
        if names is not None:
            if isinstance(names, str) or len(names) != len(groups) or not all(isinstance(name, str) for name in names):
                raise ValueError("ktrace: names must contain one string per recorded launch")
        out_dir = self.out_dir or backend.dump_dir()
        return [
            backend.write_trace(
                group,
                self._names,
                kernel=names[i] if names is not None else f"{self.kernel}_launch_{i + 1}",
                out_dir=out_dir,
            )
            for i, (_identity, group) in enumerate(groups)
        ]

    def export(self) -> str | None:
        """Write this window to a Perfetto-compatible trace file."""
        backend = self._provider()
        records = self._snapshot()
        if not records:
            return None
        return backend.write_trace(records, self._names, kernel=self.kernel, out_dir=self.out_dir or backend.dump_dir())


def capture(kernel: str = "kernel", *, blocks=None, out_dir=None) -> Capture:
    """Enable and collect in-kernel tracing around user launches.

    ``blocks=(x, y, z)`` records one workgroup without changing the grid.
    Enter before creating a device graph, then call ``clear()`` after graph
    construction and before replay to exclude warmup and construction records.
    """
    return Capture(kernel, blocks=blocks, out_dir=out_dir)
