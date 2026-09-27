# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Backend boundary for in-kernel tracing.

The public tracing API, compiler integration, and exporters are target-neutral.
Backends own device allocation, synchronization, record decoding, and device IR
lowering because those details depend on the compiler target and runtime ABI.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Callable

from ..compiler.backends import compile_backend_name


class TraceBackend(ABC):
    """Platform implementation used by a trace capture."""

    name: str
    clock_hz: int

    @abstractmethod
    def lower_kernel(self, func, ctx, grid, block, stream) -> None:
        """Lower trace annotations in one device kernel."""

    @abstractmethod
    def current_device(self) -> int:
        """Return the runtime device ordinal used for capture."""

    @abstractmethod
    def synchronize(self, device: int) -> None:
        """Wait for work submitted to *device*."""

    @abstractmethod
    def allocate_buffer(self, words: int, device: int):
        """Allocate a zero-initialized device buffer of 32-bit words."""

    @abstractmethod
    def buffer_pointer(self, buffer) -> int:
        """Return the device address passed through the hidden trace ABI."""

    @abstractmethod
    def buffer_words(self, buffer) -> list[int]:
        """Copy a device buffer to unsigned host words."""

    @abstractmethod
    def decode(self, spec: dict, words: list[int]) -> list[dict]:
        """Decode backend records described by one compiled trace spec."""

    @abstractmethod
    def raw_metadata(self, specs: list[dict]) -> dict:
        """Describe clocks and target identity for the raw trace format."""


_FACTORIES: dict[str, Callable[[], TraceBackend]] = {}
_INSTANCES: dict[str, TraceBackend] = {}


def register_trace_backend(name: str, factory: Callable[[], TraceBackend], *, force: bool = False) -> None:
    """Register a tracing backend for a compiler backend identifier."""

    if not isinstance(name, str):
        raise TypeError("trace backend name must be a string")
    key = name.lower()
    if not key or not callable(factory):
        raise TypeError("trace backend registration requires a nonempty name and callable factory")
    if key in _FACTORIES and not force:
        raise ValueError(f"trace backend {key!r} is already registered")
    _FACTORIES[key] = factory
    _INSTANCES.pop(key, None)


def get_trace_backend(name: str | None = None) -> TraceBackend:
    """Resolve the tracing backend matching the active compiler backend."""

    if name is not None and not isinstance(name, str):
        raise TypeError("trace backend name must be a string or None")
    key = compile_backend_name() if name is None else name.lower()
    if key not in _FACTORIES and key.isidentifier():
        module_name = f"{__package__}._flytrace_{key}"
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
    factory = _FACTORIES.get(key)
    if factory is None:
        available = ", ".join(sorted(_FACTORIES)) or "(none)"
        raise ValueError(f"no flytrace backend for compiler backend {key!r}; registered backends: {available}")
    if key not in _INSTANCES:
        backend = factory()
        if not isinstance(backend, TraceBackend):
            raise TypeError(f"trace backend factory for {key!r} returned {type(backend).__name__}")
        if backend.name != key:
            raise ValueError(f"trace backend registered as {key!r} identifies itself as {backend.name!r}")
        _INSTANCES[key] = backend
    return _INSTANCES[key]


__all__ = ["TraceBackend", "get_trace_backend", "register_trace_backend"]
