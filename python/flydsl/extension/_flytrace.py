# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Target-neutral in-kernel event tracing.

Annotations are inert unless a capture supplies compiler hints. The compiler
discovers event sites and delegates record emission to the backend selected by
the normal FlyDSL compiler target. Capture and export remain independent of a
particular GPU instruction set or record layout.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import NamedTuple

from .. import expr as fx
from .._mlir import ir
from ..compiler.kernel_function import CompilationContext
from ..expr.meta import dsl_loc_tracing
from ._flytrace_backend import get_trace_backend
from ._flytrace_export import export_perfetto_json, export_perfetto_per_kernel, save_raw_trace
from ._flytrace_schema import MAX_EVENTS
from ._flytrace_schema import option as _option

_active = ContextVar("flytrace_capture", default=None)
_DEFAULT_BLOCK = object()


class TraceOptions(NamedTuple):
    version: str
    mode: str
    blocks: object
    exclude: tuple[str, ...]
    max_events: int
    max_blocks: int | None
    hardware: bool


def _normalize_blocks(block):
    if block is None:
        return None
    if isinstance(block, (tuple, list)) and len(block) == 3 and all(type(number) is int for number in block):
        blocks = (tuple(block),)
    else:
        try:
            blocks = tuple(tuple(item) for item in block)
        except (TypeError, ValueError) as exc:
            raise ValueError("flytrace block must be an (x, y, z) tuple, a sequence of tuples, or None") from exc
    if not blocks:
        raise ValueError("flytrace block selection cannot be empty")
    if any(len(item) != 3 or any(type(number) is not int or number < 0 for number in item) for item in blocks):
        raise ValueError("flytrace blocks must contain non-negative compile-time (x, y, z) tuples")
    if len(set(blocks)) != len(blocks):
        raise ValueError("flytrace block selection contains duplicates")
    return blocks


def _normalize_exclude(exclude):
    if isinstance(exclude, str):
        raise TypeError("flytrace exclude must be an iterable of event names, not a string")
    try:
        names = tuple(exclude)
    except TypeError as exc:
        raise TypeError("flytrace exclude must be an iterable of event names") from exc
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("flytrace excluded event names must be nonempty strings")
    return tuple(sorted(set(names)))


@contextmanager
def configure(*, block):
    """Scope block selection to launches inside a top-level ``@jit`` body.

    Pass one compile-time ``(x, y, z)`` coordinate, a sequence of coordinates,
    or ``None`` for a bounded all-grid capture. An explicit ``capture(block=)``
    overrides this selection. Nested contexts restore their parent selection.
    """

    from ..compiler.kernel_function import KernelFunction

    ctx = CompilationContext.get_current()
    if ctx is None or KernelFunction.get_current() is not None:
        raise RuntimeError("flytrace.configure belongs in a @jit before kernel launch")
    if ir.InsertionPoint.current.block.owner.operation.name != "func.func":
        raise ValueError("flytrace.configure requires top-level JIT or compile-time control flow")
    selection = _normalize_blocks(block)
    previous = ctx.trace_blocks
    ctx.trace_blocks = selection
    try:
        yield
    finally:
        ctx.trace_blocks = previous


@dsl_loc_tracing
def _event(name, kind, payload=None):
    from ..compiler.kernel_function import KernelFunction

    ctx = CompilationContext.get_current()
    if ctx is None or KernelFunction.get_current() is None:
        raise RuntimeError("flytrace annotations belong inside a @kernel")
    if not isinstance(name, str) or not name:
        raise ValueError("flytrace event name must be a nonempty compile-time string")
    if ctx.trace_spec is None or name in _option(ctx.trace_spec["options"], "exclude"):
        return
    operands = [] if payload is None else [fx.Int32(payload).ir_value()]
    ir.Operation.create(
        "fly.trace_event",
        operands=operands,
        attributes={"event_name": ir.StringAttr.get(name), "kind": ir.StringAttr.get(kind)},
    )


@dataclass(frozen=True)
class RangeToken:
    """Compile-time handle pairing ``range_start`` with ``range_end``."""

    name: str
    has_payload: bool
    identifier: int


def mark(name, payload=None):
    """Emit an instantaneous event from every participating wave."""

    _event(name, "mark", payload)


def range_start(name, payload=None):
    """Start a named range and return the token required to close it."""

    ctx = CompilationContext.get_current()
    identifier = ctx.trace_range_counter if ctx is not None else -1
    if ctx is not None:
        ctx.trace_range_counter += 1
    _event(name, f"range_start:{identifier}", payload)
    return RangeToken(name, payload is not None, identifier)


def range_end(token, payload=None):
    """Close a range created by ``range_start``."""

    if not isinstance(token, RangeToken):
        raise TypeError("flytrace.range_end() requires a RangeToken from range_start()")
    if token.has_payload != (payload is not None):
        raise ValueError("flytrace range start/end payload forms must match")
    _event(token.name, f"range_end:{token.identifier}", payload)


def range_push(name, payload=None):
    """Start a LIFO nested range."""

    _event(name, "push", payload)


def range_pop():
    """Close the most recently pushed range."""

    _event("__pop", "pop")


def boundary(name, payload=None):
    """End the current adjacent phase and begin the named phase."""

    _event(name, "boundary", payload)


def end():
    """Close the current adjacent phase."""

    _event("__end", "end")


# Short names remain source-compatible with the initial API.
push = range_push
pop = range_pop


def lower_kernel(func, ctx, grid, block, stream):
    """Delegate trace lowering to the active compiler target's backend."""

    get_trace_backend().lower_kernel(func, ctx, grid, block, stream)


class TraceCallState:
    def __init__(self, state, spec):
        self.state, self.spec = state, spec

    def __call__(self, args):
        cap = _active.get()
        if cap is None:
            raise RuntimeError("a trace-enabled compiled function must run inside flytrace.capture()")
        if cap.options != self.spec["options"]:
            raise ValueError("compiled trace configuration differs from the active capture")
        compiled_backend = self.spec.get("backend")
        if compiled_backend is not None and compiled_backend != cap.backend.name:
            raise ValueError("compiled trace backend differs from the active capture backend")
        return self.state(args)


def fill_buffer(spec, _ignored, storage):
    cap = _active.get()
    if cap is None:
        raise RuntimeError("trace launch requires an active flytrace.capture()")
    storage.value = cap._buffer(spec)


class capture:
    """Capture annotated events from arbitrary compiled kernel launches.

    The active compiler backend supplies device recording and decoding. The
    default block selection is ``(0, 0, 0)``; use ``block=None`` with a finite
    ``max_blocks`` to capture a runtime-sized grid. ``mode='auto'`` chooses the
    compact static schedule when possible and otherwise records dynamic control
    flow and payloads into a bounded per-wave buffer. ``per_kernel=True`` makes
    an automatic path export write a directory of aligned kernel traces.
    """

    def __init__(
        self,
        path=None,
        *,
        block=_DEFAULT_BLOCK,
        exclude=(),
        hardware=False,
        per_kernel=False,
        mode="auto",
        max_events=65536,
        max_blocks=None,
    ):
        if mode not in ("auto", "static", "dynamic"):
            raise ValueError("flytrace mode must be 'auto', 'static', or 'dynamic'")
        if type(max_events) is not int or not 0 < max_events <= MAX_EVENTS:
            raise ValueError("flytrace max_events must be in [1, 2**20]")
        if max_blocks is not None and (type(max_blocks) is not int or max_blocks <= 0):
            raise ValueError("flytrace max_blocks must be a positive integer or None")
        if type(hardware) is not bool:
            raise TypeError("flytrace hardware must be a bool")
        if type(per_kernel) is not bool:
            raise TypeError("flytrace per_kernel must be a bool")
        blocks = "jit" if block is _DEFAULT_BLOCK else _normalize_blocks(block)
        self.path = path
        self.per_kernel = per_kernel
        self.options = TraceOptions(
            "flytrace-v4",
            mode,
            blocks,
            _normalize_exclude(exclude),
            max_events,
            max_blocks,
            hardware,
        )
        self._records = {}
        self._graph_bound = set()
        self._allocated_words = 0
        self._entered = False
        self._captured = False

    @property
    def backend(self):
        if not self._entered and not self._captured:
            raise RuntimeError("flytrace capture has not been entered")
        return self._backend

    def __enter__(self):
        if _active.get() is not None or self._entered:
            raise RuntimeError("nested/reentrant flytrace captures are unsupported")
        backend = get_trace_backend()
        if backend.is_current_stream_capturing():
            raise RuntimeError(
                "enter flytrace.capture() before device graph capture; "
                "trace launches must be warmed once outside graph capture"
            )
        device = backend.current_device()
        if self._captured and (backend.name != self._backend.name or device != self.device):
            raise RuntimeError("a flytrace capture cannot move between backends or devices")
        token = _active.set(self)
        hints = CompilationContext.compile_hints({"flytrace": self.options})
        try:
            hints.__enter__()
        except BaseException:
            _active.reset(token)
            raise
        self._backend = backend
        self.device = device
        self._token = token
        self._hints = hints
        self._entered = True
        self._captured = True
        return self

    def __exit__(self, typ, value, tb):
        try:
            self.backend.synchronize(self.device)
            if typ is None and self.path is not None:
                self.export(self.path)
        finally:
            try:
                self._hints.__exit__(typ, value, tb)
            finally:
                self._entered = False
                _active.reset(self._token)

    def _buffer(self, spec):
        compiled_backend = spec.get("backend")
        if compiled_backend is not None and compiled_backend != self.backend.name:
            raise ValueError("compiled trace backend differs from the active capture backend")
        if self.backend.current_device() != self.device:
            raise ValueError("cannot change device inside flytrace capture")
        if not spec["words"]:
            return 0
        key = id(spec)
        if self.backend.is_current_stream_capturing():
            previous = self._records.get(key)
            if previous is None:
                raise RuntimeError(
                    "flytrace launch was not prepared for graph capture; "
                    "run this traced specialization once before beginning graph capture"
                )
            self._graph_bound.add(key)
            return self.backend.buffer_pointer(previous[1])
        if key in self._graph_bound:
            previous = self._records.get(key)
            if previous is None:
                raise RuntimeError("flytrace lost storage retained by a captured graph")
            # A captured graph owns this address for its lifetime. Reuse it for
            # later eager launches, but clear stale rows before an arbitrary
            # user stream can write the next recording.
            self.backend.synchronize(self.device)
            self.backend.clear_buffer(previous[1])
            self.backend.synchronize(self.device)
            return self.backend.buffer_pointer(previous[1])
        previous_words = self._records[key][0]["words"] if key in self._records else 0
        allocated_words = self._allocated_words - previous_words + spec["words"]
        if allocated_words * 4 > 512 * 1024**2:
            raise ValueError("flytrace capture exceeds the 512 MiB allocation limit; select fewer blocks/events")
        # Synchronize before replacing an older record that may still be used
        # by an arbitrary user stream, then synchronize the fresh zeroed buffer
        # before that stream consumes it.
        if previous_words:
            self.backend.synchronize(self.device)
            self._records.pop(key)
            self._allocated_words -= previous_words
        buffer = self.backend.allocate_buffer(spec["words"], self.device)
        self.backend.synchronize(self.device)
        self._allocated_words += spec["words"]
        self._records[key] = (spec, buffer)
        return self.backend.buffer_pointer(buffer)

    def _decode_recordings(self):
        backend = self.backend
        backend.synchronize(self.device)
        recordings = []
        for recording, (spec, buffer) in enumerate(self._records.values()):
            words = backend.buffer_words(buffer)
            recordings.append((recording, backend.decode(spec, words)))
        return recordings

    def decode(self):
        return [wave for _, waves in self._decode_recordings() for wave in waves]

    def save(self, path):
        """Save absolute ticks and hardware identities for offline analysis."""

        waves = self.decode()
        specs = [spec for spec, _ in self._records.values()]
        return save_raw_trace(waves, path, self.backend.raw_metadata(specs))

    def export_per_kernel(self, directory):
        """Export one aligned Perfetto JSON file per compiled kernel."""

        groups = []
        for recording, waves in self._decode_recordings():
            by_kernel = {}
            for wave in waves:
                by_kernel.setdefault(wave["kernel"], []).append(wave)
            groups.extend(
                {"recording": recording, "kernel": kernel, "waves": kernel_waves}
                for kernel, kernel_waves in by_kernel.items()
            )
        return export_perfetto_per_kernel(groups, directory, self.backend.clock_hz)

    def export(self, path, *, per_kernel=None):
        """Export one combined trace, or one file per compiled kernel."""

        if per_kernel is None:
            per_kernel = self.per_kernel
        if type(per_kernel) is not bool:
            raise TypeError("flytrace export per_kernel must be a bool or None")
        if per_kernel:
            return self.export_per_kernel(path)
        return export_perfetto_json(self.decode(), path, self.backend.clock_hz)
