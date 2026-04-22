# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL runtime: device runtime, GPU detection helpers, etc."""

from .device_runtime import (
    COMPILE_BACKEND_TO_RUNTIME_KIND,
    DeviceRuntime,
    RocmDeviceRuntime,
    ensure_compile_runtime_compatible,
    ensure_compile_runtime_pairing_from_env,
    get_device_runtime,
    register_compile_runtime_mapping,
    register_device_runtime,
)

__all__ = [
    "COMPILE_BACKEND_TO_RUNTIME_KIND",
    "DeviceRuntime",
    "RocmDeviceRuntime",
    "ensure_compile_runtime_compatible",
    "ensure_compile_runtime_pairing_from_env",
    "get_device_runtime",
    "register_compile_runtime_mapping",
    "register_device_runtime",
]
