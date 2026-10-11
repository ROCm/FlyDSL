# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors


from __future__ import annotations

from typing import Any, Protocol


class TraceBuffer(Protocol):

    device_ptr: int

    def synchronize(self) -> None: ...

    def launch_count(self) -> int: ...

    def read_records(self) -> list[Any]: ...

    def reset(self) -> None: ...


class KtraceBackend(Protocol):

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
