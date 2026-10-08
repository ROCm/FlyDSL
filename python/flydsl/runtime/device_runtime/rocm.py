# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""ROCm / HIP device runtime (default FlyDSL GPU stack)."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import ClassVar

from ..device import get_rocm_device_count
from .base import DeviceRuntime

# Cached HIP runtime handle (``libamdhip64``); cached once.
_HIP_LIB = None
_HIP_LIB_TRIED = False
_HIP_DLL_DIRS = []


def _hip_library_candidates():
    if os.name != "nt":
        yield from ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5")
        return

    roots = [os.environ.get("ROCM_PATH"), os.environ.get("HIP_PATH")]
    for entry in sys.path:
        roots.extend((str(Path(entry) / "_rocm_sdk_core"), str(Path(entry) / "_rocm_sdk_devel")))
    seen = set()
    for root in roots:
        if not root:
            continue
        for name in ("amdhip64_7.dll", "amdhip64.dll"):
            candidate = Path(root) / "bin" / name
            if candidate.is_file() and candidate not in seen:
                seen.add(candidate)
                yield str(candidate)
    yield "amdhip64_7.dll"
    yield "amdhip64.dll"


def _hip_get_device() -> int:
    """Active HIP device index via ``hipGetDevice``."""
    global _HIP_LIB, _HIP_LIB_TRIED
    if not _HIP_LIB_TRIED:
        _HIP_LIB_TRIED = True
        for soname in _hip_library_candidates():
            try:
                if os.name == "nt" and os.path.isabs(soname):
                    _HIP_DLL_DIRS.append(os.add_dll_directory(str(Path(soname).parent)))
                lib = ctypes.CDLL(soname)
                lib.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
                lib.hipGetDevice.restype = ctypes.c_int
                _HIP_LIB = lib
                break
            except OSError:
                continue
    if _HIP_LIB is None:
        return 0
    dev = ctypes.c_int(0)
    try:
        if _HIP_LIB.hipGetDevice(ctypes.byref(dev)) == 0:
            return int(dev.value)
    except Exception:
        pass
    return 0


class RocmDeviceRuntime(DeviceRuntime):
    """HIP-based runtime; matches compile backend ``rocm``.

    ``device_count()`` delegates to ``rocm_agent_enumerator`` in ``device.py``;
    ``current_device_id()`` queries HIP ``hipGetDevice`` directly.
    """

    kind: ClassVar[str] = "rocm"

    def device_count(self) -> int:
        return get_rocm_device_count()

    def current_device_id(self) -> int:
        return _hip_get_device()
