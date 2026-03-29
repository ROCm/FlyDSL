# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Abstract device runtime : single native GPU stack per process."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Device:
    """Logical GPU device (ordinal only in v1; capabilities may extend later)."""

    ordinal: int = 0
    """Device index for the active runtime (e.g. HIP device id)."""


class DeviceRuntime(metaclass=ABCMeta):
    """Vendor-neutral runtime: one implementation per process (HIP, CUDA, …).

    Stream and Event handles stay opaque at the Python/C boundary; concrete
    APIs live in native glue (e.g. ROCm wrappers).
    """

    kind: ClassVar[str]
    """Stable runtime identifier (e.g. ``\"rocm\"`` for HIP/ROCm)."""

    @abstractmethod
    def device_count(self) -> int:
        """Number of visible devices for this runtime."""

    def default_device(self) -> Device:
        """Default device for launch when none is specified."""
        return Device(ordinal=0)


class Event:
    """Placeholder for future opaque event handles.

    Kernel launch paths may synchronize via streams; explicit events can be
    added without changing the compile-backend layer.
    """

    __slots__ = ()
