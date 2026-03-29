# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""ROCm / HIP device runtime (default FlyDSL GPU stack)."""

from __future__ import annotations

from typing import ClassVar, Optional

from ..device import get_rocm_device_count
from .base import DeviceRuntime


class RocmDeviceRuntime(DeviceRuntime):
    """HIP-based runtime; matches compile backend ``rocm``.

    ``device_count()`` uses :func:`get_rocm_device_count` in ``device.py``
    (``rocm_agent_enumerator``, same style as arch detection there).
    """

    kind: ClassVar[str] = "rocm"

    def __init__(self) -> None:
        self._device_count_cache: Optional[int] = None

    def _lazy_device_count(self) -> int:
        if self._device_count_cache is not None:
            return self._device_count_cache
        n = get_rocm_device_count()
        self._device_count_cache = n
        return self._device_count_cache

    def device_count(self) -> int:
        return self._lazy_device_count()
