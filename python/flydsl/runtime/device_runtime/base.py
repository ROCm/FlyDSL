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


def device_from_dlpack(device_type: int, device_id: int) -> Device | None:
    """Map canonical DLPack GPU device types to a FlyDSL device."""
    if int(device_type) == 10:  # kDLROCM
        return Device(kind="rocm", index=int(device_id))
    if int(device_type) == 2:  # kDLCUDA
        return Device(kind="cuda", index=int(device_id))
    return None


def device_from_argument(value) -> Device | None:
    """Read framework-neutral device metadata from a call argument."""
    provider = getattr(value, "__flydsl_device__", None)
    if provider is not None:
        device = provider()
        if device is not None:
            if not isinstance(device, Device):
                raise TypeError(f"__flydsl_device__ must return Device or None, got {device!r}")
            return device

    dlpack_device = getattr(value, "__dlpack_device__", None)
    if callable(dlpack_device):
        try:
            device_type, device_id = dlpack_device()
        except Exception:
            return None
        return device_from_dlpack(device_type, device_id)
    return None


def device_from_stream_argument(value, *, cuda_kind: str = "cuda") -> Device | None:
    """Read a CUDA/HIP-style stream's logical device without importing a framework."""
    raw = value.value if getattr(value, "_is_stream_param", False) else value
    stream_device = getattr(raw, "device", None)
    if getattr(stream_device, "type", None) != "cuda":
        return None
    index = getattr(stream_device, "index", None)
    if index is None:
        index = getattr(raw, "device_index", None)
    if index is None:
        return None
    return Device(kind=cuda_kind, index=int(index))


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
