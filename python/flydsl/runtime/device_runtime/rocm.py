# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""ROCm / HIP device runtime (default FlyDSL GPU stack)."""

from __future__ import annotations

import ctypes
import functools
import threading
from pathlib import Path
from typing import ClassVar

from .base import DeviceRuntime

# Cached HIP runtime handle (``libamdhip64``); cached once.
_HIP_LIB = None
_HIP_LIB_LOCK = threading.Lock()
_FLY_RUNTIME_LIB = None
_FLY_RUNTIME_LIB_LOCK = threading.Lock()


def _get_hip_lib():
    global _HIP_LIB
    if _HIP_LIB is not None:
        return _HIP_LIB

    with _HIP_LIB_LOCK:
        if _HIP_LIB is not None:
            return _HIP_LIB

        last_error = None
        for soname in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
            try:
                lib = ctypes.CDLL(soname)
                lib.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
                lib.hipGetDevice.restype = ctypes.c_int
                lib.hipSetDevice.argtypes = [ctypes.c_int]
                lib.hipSetDevice.restype = ctypes.c_int
                lib.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
                lib.hipGetDeviceCount.restype = ctypes.c_int
                _HIP_LIB = lib
                return lib
            except OSError as exc:
                last_error = exc

        raise RuntimeError("Unable to load libamdhip64; cannot resolve the active ROCm device") from last_error


def _get_fly_runtime_lib():
    global _FLY_RUNTIME_LIB
    if _FLY_RUNTIME_LIB is not None:
        return _FLY_RUNTIME_LIB

    with _FLY_RUNTIME_LIB_LOCK:
        if _FLY_RUNTIME_LIB is not None:
            return _FLY_RUNTIME_LIB

        # Resolve against this concrete FlyDSL package, not the importable
        # _mlir_libs module: editable/CI environments can also contain an older
        # installed FlyDSL namespace whose runtime lacks the new symbol.
        path = Path(__file__).resolve().parents[2] / "_mlir" / "_mlir_libs" / "libfly_jit_runtime.so"
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as exc:
            raise RuntimeError(f"Unable to load FlyDSL ROCm runtime at {path}: {exc}") from exc
        try:
            lib.mgpuGetDeviceArch.argtypes = [ctypes.c_int32, ctypes.c_char_p, ctypes.c_size_t]
            lib.mgpuGetDeviceArch.restype = ctypes.c_int32
        except AttributeError as exc:
            raise RuntimeError(
                f"FlyDSL ROCm runtime at {path} does not export mgpuGetDeviceArch; rebuild FlyDSL"
            ) from exc
        _FLY_RUNTIME_LIB = lib
        return lib


def _hip_get_device() -> int:
    """Active HIP device index via ``hipGetDevice``."""
    lib = _get_hip_lib()
    dev = ctypes.c_int(0)
    result = lib.hipGetDevice(ctypes.byref(dev))
    if result != 0:
        raise RuntimeError(f"hipGetDevice failed with error code {result}")
    return int(dev.value)


def _hip_set_device(device_id: int) -> None:
    result = _get_hip_lib().hipSetDevice(int(device_id))
    if result != 0:
        raise RuntimeError(f"hipSetDevice({device_id}) failed with error code {result}")


def _hip_get_device_count() -> int:
    count = ctypes.c_int(0)
    result = _get_hip_lib().hipGetDeviceCount(ctypes.byref(count))
    if result != 0:
        raise RuntimeError(f"hipGetDeviceCount failed with error code {result}")
    return int(count.value)


@functools.lru_cache(maxsize=None)
def _hip_device_arch(device_id: int) -> str:
    arch = ctypes.create_string_buffer(256)
    result = _get_fly_runtime_lib().mgpuGetDeviceArch(device_id, arch, len(arch))
    if result != 0:
        raise RuntimeError(f"hipGetDeviceProperties({device_id}) failed with error code {result}")
    value = arch.value.decode("ascii").split(":", 1)[0]
    if not value.startswith("gfx"):
        raise RuntimeError(f"HIP returned an invalid architecture for device {device_id}: {value!r}")
    return value


class RocmDeviceRuntime(DeviceRuntime):
    """HIP-based runtime; matches compile backend ``rocm``.

    Device identity and architecture come from HIP so logical device indices
    respect the runtime's visibility/remapping rules.
    """

    kind: ClassVar[str] = "rocm"

    def device_count(self) -> int:
        return _hip_get_device_count()

    def current_device_id(self) -> int:
        return _hip_get_device()

    def set_device_id(self, device_id: int) -> None:
        _hip_set_device(device_id)

    def device_arch(self, device_id: int) -> str:
        return _hip_device_arch(device_id)
