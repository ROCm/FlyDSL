# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Contract between shared tracing code and a target's trace implementation.

Records are opaque to the shared capture code. A provider owns their device
format, launch identity, timebase, summary, export, and buffer compatibility
rule. A compile backend returns its provider through
``BaseBackend.ktrace_backend``.
"""

from __future__ import annotations

from typing import Any, Protocol


class TraceBuffer(Protocol):
    """A provider's buffer, including replay detection and destructive readback."""

    device_ptr: int

    def synchronize(self) -> None: ...

    def launch_count(self) -> int: ...

    def read_records(self) -> list[Any]: ...

    def reset(self) -> None: ...


class KtraceBackend(Protocol):
    """Target hook used by the shared JIT and capture code.

    ``configure_module`` publishes target settings and the known event names.
    The lowering pass must publish ``fly_ktrace.event_names`` for cached binaries.
    ``buffer_config`` returns a picklable value that identifies the device buffer
    settings compiled into a binary; the provider checks it on a cache hit.
    ``group_launches`` returns groups in recording order. ``launch_metadata``
    supplies target fields; the shared API adds the ordinal ``launch_id``.
    """

    def ensure_buffer(self) -> TraceBuffer: ...

    def current_buffer(self) -> TraceBuffer | None: ...

    def require_dispatchable_buffer(self) -> None: ...

    def buffer_config(self) -> Any: ...

    def require_compatible_buffer(self, compiled_config: Any) -> None: ...

    def configure_module(
        self,
        module: Any,
        *,
        target_arch: str,
        known_names: dict[str, int],
        blocks: str,
    ) -> None: ...

    def check_outside_capture(self) -> None: ...

    def summarize(self, records: list[Any], names: dict[str, int]) -> dict: ...

    def group_launches(self, records: list[Any]) -> list[tuple[Any, list[Any]]]: ...

    def launch_metadata(self, identity: Any) -> dict: ...

    def write_trace(self, records: list[Any], names: dict[str, int], *, kernel: str, out_dir: str) -> str: ...

    def dump_dir(self) -> str: ...
