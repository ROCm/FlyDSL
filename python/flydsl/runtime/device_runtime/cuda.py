# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CUDA driver-API device runtime (NVIDIA GPU stack)."""

from __future__ import annotations

import ctypes
import functools
import os
from typing import ClassVar, Optional

from .base import DeviceRuntime

# Cached CUDA driver handle (``libcuda``); cached once.
_CUDA_LIB = None
_CUDA_LIB_TRIED = False

CUDA_SUCCESS = 0


def _load_cuda():
    global _CUDA_LIB, _CUDA_LIB_TRIED
    if not _CUDA_LIB_TRIED:
        _CUDA_LIB_TRIED = True
        for soname in ("libcuda.so", "libcuda.so.1"):
            try:
                _CUDA_LIB = ctypes.CDLL(soname)
                break
            except OSError:
                continue
    return _CUDA_LIB


@functools.lru_cache(maxsize=1)
def _driver() -> Optional[ctypes.CDLL]:
    """Return an initialized ``libcuda`` handle, or None when unavailable.

    Everything here goes through the driver rather than ``nvidia-smi`` on
    purpose: the driver enumerates *visible* devices, so ``CUDA_VISIBLE_DEVICES``
    (and its ordering / UUID forms) is honored automatically and agrees with
    what PyTorch sees. ``nvidia-smi`` reports physical devices and would
    disagree with the runtime.
    """
    lib = _load_cuda()
    if lib is None:
        return None
    try:
        if lib.cuInit(0) != CUDA_SUCCESS:
            return None
    except Exception:
        return None
    return lib


@functools.lru_cache(maxsize=1)
def get_cuda_device_count() -> int:
    """Number of CUDA devices visible to this process. 0 when unavailable."""
    lib = _driver()
    if lib is None:
        return 0
    count = ctypes.c_int(0)
    try:
        if lib.cuDeviceGetCount(ctypes.byref(count)) != CUDA_SUCCESS:
            return 0
    except Exception:
        return 0
    return int(count.value)


def _cuda_current_device() -> int:
    """Active CUDA device index via ``cuCtxGetDevice`` (falls back to 0)."""
    lib = _driver()
    if lib is None:
        return 0
    try:
        dev = ctypes.c_int(0)
        if lib.cuCtxGetDevice(ctypes.byref(dev)) == CUDA_SUCCESS:
            return int(dev.value)
    except Exception:
        pass
    return 0


def _compute_capability(ordinal: int) -> Optional[str]:
    """Compute capability of visible device ``ordinal`` as ``sm_XX``."""
    lib = _driver()
    if lib is None:
        return None
    try:
        dev = ctypes.c_int(0)
        if lib.cuDeviceGet(ctypes.byref(dev), ordinal) != CUDA_SUCCESS:
            return None
        major, minor = ctypes.c_int(0), ctypes.c_int(0)
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_{MAJOR,MINOR}
        if lib.cuDeviceGetAttribute(ctypes.byref(major), 75, dev) != CUDA_SUCCESS:
            return None
        if lib.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev) != CUDA_SUCCESS:
            return None
    except Exception:
        return None
    return f"sm_{major.value}{minor.value}"


@functools.lru_cache(maxsize=None)
def _cuda_arch_from_hardware(ordinal: int) -> str:
    """Cached compute capability of a visible device."""
    return _compute_capability(ordinal)


def get_cuda_arch() -> str:
    """Best-effort CUDA GPU arch string (e.g. ``'sm_90'``).

    Honors ``ARCH`` / ``FLYDSL_GPU_ARCH`` when they name an ``sm_*`` target,
    otherwise reports the capability of the *currently selected* device.
    """
    env = os.environ.get("ARCH") or os.environ.get("FLYDSL_GPU_ARCH")
    if env and env.startswith("sm_"):
        return env
    return _cuda_arch_from_hardware(_cuda_current_device())


class CudaDeviceRuntime(DeviceRuntime):
    """CUDA driver-API runtime; matches compile backend ``cuda``.

    Both ``device_count()`` and ``current_device_id()`` query the CUDA driver,
    so they agree with ``CUDA_VISIBLE_DEVICES`` and with PyTorch.
    """

    kind: ClassVar[str] = "cuda"

    def device_count(self) -> int:
        return get_cuda_device_count()

    def current_device_id(self) -> int:
        return _cuda_current_device()
