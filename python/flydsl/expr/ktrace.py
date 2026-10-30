# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors


from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any

from .meta import dsl_loc_tracing

KIND_MARK = 0
KIND_RANGE_PUSH = 1
KIND_RANGE_POP = 2
KIND_RANGE_START = 3
KIND_RANGE_STOP = 4

_CACHED_RECORDS: list | None = None
_ACTIVE_CAPTURE = None

_COLLECT_LOCK = threading.RLock()


def collection_lock():
    return _COLLECT_LOCK


def _trace_backend():
    from ..compiler.backends import get_ktrace_backend

    return get_ktrace_backend()


def tracing_enabled() -> bool:
    from ..compiler.kernel_function import CompilationContext
    from ..utils import env

    hints = CompilationContext.get_compile_hints() or {}
    if "ktrace" in hints:
        return bool(hints["ktrace"])
    return bool(env.ktrace.enable)


class RangeToken:

    __slots__ = ("_slot", "_event_name")

    def __init__(self, slot, event_name: str | None = None):
        self._slot = slot
        self._event_name = event_name

    def __extract_to_ir_values__(self) -> list:
        return [self._slot]

    @classmethod
    def __construct_from_ir_values__(cls, values: list, exemplar: "RangeToken | None" = None) -> "RangeToken":
        return cls(values[0], exemplar._event_name if exemplar is not None else None)

    def __repr__(self) -> str:  # pragma: no cover
        return f"RangeToken(event_name={self._event_name!r})"


def _disabled_token(event_name: str | None) -> RangeToken:
    from .._mlir import ir
    from .._mlir.dialects import arith

    i32 = ir.IntegerType.get_signless(32)
    slot = arith.ConstantOp(i32, ir.IntegerAttr.get(i32, 0)).result
    return RangeToken(slot, event_name)


def mark(event_name: str, payload: Any = None) -> None:
    if not tracing_enabled():
        return
    _mark_impl(event_name, payload)


@dsl_loc_tracing
def _mark_impl(event_name, payload) -> None:
    _build_annotation(event_name, KIND_MARK, payload=payload)


def range_push(event_name: str, payload: Any = None) -> None:
    if not tracing_enabled():
        return
    _range_push_impl(event_name, payload)


@dsl_loc_tracing
def _range_push_impl(event_name, payload) -> None:
    _build_annotation(event_name, KIND_RANGE_PUSH, payload=payload)


def range_pop() -> None:
    if not tracing_enabled():
        return
    _range_pop_impl()


@contextmanager
def range(event_name: str, payload: Any = None):
    range_push(event_name, payload)
    try:
        yield
    finally:
        range_pop()


@dsl_loc_tracing
def _range_pop_impl() -> None:
    _build_annotation(None, KIND_RANGE_POP)


def range_start(event_name: str, payload: Any = None) -> RangeToken:
    if not tracing_enabled():
        return _disabled_token(event_name)
    return _range_start_impl(event_name, payload)


@dsl_loc_tracing
def _range_start_impl(event_name, payload) -> RangeToken:
    slot = _build_annotation(event_name, KIND_RANGE_START, payload=payload)
    return RangeToken(slot, event_name)


def range_stop(token: RangeToken, payload: Any = None) -> None:
    if not tracing_enabled():
        return
    _range_stop_impl(token, payload)


@dsl_loc_tracing
def _range_stop_impl(token, payload) -> None:
    _build_annotation(token._event_name, KIND_RANGE_STOP, payload=payload, start_slot=token)


@dsl_loc_tracing
def inactive_token(event_name: str) -> RangeToken:
    if not tracing_enabled():
        return _disabled_token(event_name)
    from .._mlir.dialects import fly_ktrace

    return RangeToken(fly_ktrace.inactive_token(_token_type()), event_name)


def _token_type():
    from .._mlir import ir

    return ir.Type.parse("!fly_ktrace.token")


def _build_annotation(event_name, kind, *, payload=None, start_slot=None):
    from .._mlir.dialects import fly_ktrace
    from .ktrace_common import payload_operand

    operand, payload_type = (None, "uint64") if payload is None else payload_operand(payload)
    kwargs = {"payload": operand, "payload_type": payload_type}

    if kind == KIND_MARK:
        fly_ktrace.mark(event_name, **kwargs)
        return None
    if kind == KIND_RANGE_PUSH:
        fly_ktrace.range_push(event_name, **kwargs)
        return None
    if kind == KIND_RANGE_POP:
        fly_ktrace.range_pop(**kwargs)
        return None
    if kind == KIND_RANGE_START:
        return fly_ktrace.range_start(_token_type(), event_name, **kwargs)
    if kind == KIND_RANGE_STOP:
        fly_ktrace.range_stop(start_slot._slot, event_name, **kwargs)
        return None
    raise AssertionError(f"unknown ktrace event kind {kind}")


def _take_records(backend=None) -> list | None:
    global _CACHED_RECORDS
    backend = backend or _trace_backend()
    buf = backend.current_buffer()
    if buf is None:
        return None

    with _COLLECT_LOCK:
        if _CACHED_RECORDS is not None:
            if not buf.launch_count():
                return _CACHED_RECORDS
            _CACHED_RECORDS = None

        try:
            _CACHED_RECORDS = buf.read_records()
            return _CACHED_RECORDS
        finally:
            try:
                buf.reset()
            except Exception as exc:  # noqa: BLE001
                from ..utils import log

                log().warning(f"ktrace: failed to re-arm the trace buffer: {exc}")


def invalidate_records() -> None:
    global _CACHED_RECORDS
    with _COLLECT_LOCK:
        _CACHED_RECORDS = None


def restore_names(artifact) -> None:
    names = getattr(artifact, "_ktrace_names", None)
    if not names:
        return
    from .ktrace_common import merge_names

    merge_names(names)


def require_compatible_buffer(artifact) -> None:
    if not getattr(artifact, "_ktrace_traced", False):
        return
    compiled = getattr(artifact, "_ktrace_buffer_config", None)
    if compiled is None:
        raise RuntimeError("ktrace: cached traced binary has no buffer settings; rebuild its compile cache")
    _trace_backend().require_compatible_buffer(compiled)


def _require_no_capture(name: str, method: str) -> None:
    if _ACTIVE_CAPTURE is not None:
        raise RuntimeError(f"ktrace: {name}() would drain the active capture's records; use its {method}() instead")


def collect(kernel: str = "kernel", out_dir=None) -> str | None:
    from .ktrace_common import name_table

    _require_no_capture("collect", "export")
    backend = _trace_backend()
    records = _take_records(backend)
    if not records:
        return None
    return backend.write_trace(
        records,
        name_table(),
        kernel=kernel,
        out_dir=out_dir or backend.dump_dir(),
    )


def summary() -> dict | None:
    from .ktrace_common import name_table

    _require_no_capture("summary", "summary")
    backend = _trace_backend()
    records = _take_records(backend)
    if not records:
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
        from ..compiler.kernel_function import CompilationContext

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
            if exc_type is None and not self._snapshot():
                from ..utils import log

                log().warning(
                    "ktrace: the capture recorded no events. Only kernels compiled inside the capture are "
                    "instrumented; a CompiledFunction built before it runs untraced."
                )
        finally:
            try:
                self._scope.__exit__(exc_type, exc_value, traceback)
            finally:
                self._scope = None
                with _COLLECT_LOCK:
                    _ACTIVE_CAPTURE = None
        return False

    def clear(self) -> None:
        global _CACHED_RECORDS
        backend = self._provider()

        backend.check_outside_capture()
        with _COLLECT_LOCK:
            if _ACTIVE_CAPTURE is not None and _ACTIVE_CAPTURE is not self:
                raise RuntimeError("ktrace: cannot clear another capture's active trace buffer")
            buf = backend.current_buffer()
            if buf is not None:
                buf.synchronize()
                buf.reset()
            _CACHED_RECORDS = None
            self._records = None
            self._names = None

    def read(self):
        return self._read(keep=False)

    def _read(self, *, keep: bool):
        global _CACHED_RECORDS
        from .ktrace_common import name_table

        backend = self._provider()
        backend.check_outside_capture()
        with _COLLECT_LOCK:
            if _ACTIVE_CAPTURE is not None and _ACTIVE_CAPTURE is not self:
                raise RuntimeError("ktrace: cannot read another capture's active trace buffer")
            _CACHED_RECORDS = None
            records = list(_take_records(backend) or [])
            self._records = (self._records or []) + records if keep else records
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
                # Reading re-arms the buffer: keep what an earlier summary() or export() took.
                self._read(keep=True)
        return self._records

    def summary(self) -> dict | None:

        records = self._snapshot()
        if not records:
            return None
        return self._provider().summarize(records, self._names)

    def launches(self) -> list[dict]:
        backend = self._provider()
        records = self._snapshot() or []
        return [
            {**backend.launch_metadata(identity), **backend.summarize(group, self._names), "launch_id": ordinal}
            for ordinal, (identity, group) in enumerate(backend.group_launches(records), 1)
        ]

    def export_launches(self, names: list[str] | None = None) -> list[str]:
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
        backend = self._provider()
        records = self._snapshot()
        if not records:
            return None
        return backend.write_trace(records, self._names, kernel=self.kernel, out_dir=self.out_dir or backend.dump_dir())


def capture(kernel: str = "kernel", *, blocks=None, out_dir=None) -> Capture:
    return Capture(kernel, blocks=blocks, out_dir=out_dir)
