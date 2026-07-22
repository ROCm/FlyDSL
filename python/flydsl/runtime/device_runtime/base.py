# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Abstract device runtime : single native GPU stack per process."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Device:
    """Logical device identity within one process-wide runtime stack."""

    kind: str
    index: int


class DeviceRuntime(metaclass=ABCMeta):
    """Vendor-neutral runtime: one implementation per process (HIP, CUDA, …).

    Opaque stream handles live in :mod:`flydsl.expr.typing` at the DSL boundary;
    concrete APIs stay in native glue (e.g. ROCm wrappers).
    """

    kind: ClassVar[str]
    """Stable runtime identifier (e.g. ``\"rocm\"`` for HIP/ROCm)."""

    @abstractmethod
    def device_count(self) -> int:
        """Number of visible devices for this runtime."""

    @abstractmethod
    def current_device_id(self) -> int:
        """Current device id for this runtime."""

    def set_device_id(self, device_id: int) -> None:
        """Make *device_id* current for the calling thread."""
        raise NotImplementedError(f"{type(self).__name__} does not implement set_device_id()")

    def device_arch(self, device_id: int) -> str:
        """Return the compile architecture for one logical device."""
        raise NotImplementedError(f"{type(self).__name__} does not implement device_arch()")

    @contextmanager
    def device_guard(self, device_id: int):
        """Run with *device_id* current, restoring the caller's device."""
        previous = self.current_device_id()
        if previous != device_id:
            self.set_device_id(device_id)
        try:
            yield
        finally:
            if previous != device_id:
                self.set_device_id(previous)
