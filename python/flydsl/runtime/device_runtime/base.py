# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Abstract device runtime : single native GPU stack per process."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import ClassVar


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
