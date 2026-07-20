# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""ROCm / HIP device runtime (default FlyDSL GPU stack)."""

from __future__ import annotations

from typing import ClassVar

from ..device import get_rocm_device_count
from .base import DeviceRuntime


class RocmDeviceRuntime(DeviceRuntime):
    """HIP-based runtime; matches compile backend ``rocm``.

    ``device_count()`` delegates to :func:`get_rocm_device_count` in ``device.py``
    (``@lru_cache``; ``rocm_agent_enumerator``, same style as arch detection there).
    """

    kind: ClassVar[str] = "rocm"

    def device_count(self) -> int:
        return get_rocm_device_count()
