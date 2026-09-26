# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from __future__ import annotations

import pytest

from kernels.comm.custom_all_reduce import FlyDSLAllreduce
from kernels.common import hip_ipc
from kernels.common.hip_ipc import (
    HIP_DEVICE_MALLOC_UNCACHED,
    HIP_IPC_HANDLE_BYTES,
    HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS,
    HipRuntime,
)

pytestmark = pytest.mark.l0_backend_agnostic


class _FakeFunction:
    def __init__(self, implementation):
        self.implementation = implementation
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        return self.implementation(*args)


class _FakeHip:
    def __init__(self) -> None:
        self.exported_pointer = None
        self.opened_handle = None
        self.open_flags = None
        self.closed_pointer = None
        self.pointer_query = None
        self.allocated = None
        self.memset = None
        self.freed_pointer = None
        self.open_error = 0
        self.memset_error = 0
        self.hipIpcGetMemHandle = _FakeFunction(self._get_mem_handle)
        self.hipIpcOpenMemHandle = _FakeFunction(self._open_mem_handle)
        self.hipIpcCloseMemHandle = _FakeFunction(self._close_mem_handle)
        self.hipGetErrorString = _FakeFunction(lambda error: b"invalid handle" if error else b"success")
        self.hipPointerGetAttribute = _FakeFunction(self._pointer_get_attribute)
        self.hipExtMallocWithFlags = _FakeFunction(self._malloc_with_flags)
        self.hipMemset = _FakeFunction(self._memset)
        self.hipFree = _FakeFunction(self._free)

    def _get_mem_handle(self, handle_pointer, device_pointer):
        self.exported_pointer = device_pointer.value
        handle = handle_pointer._obj
        for index in range(HIP_IPC_HANDLE_BYTES):
            handle.reserved[index] = index
        return 0

    def _open_mem_handle(self, output_pointer, handle, flags):
        self.opened_handle = bytes(handle.reserved)
        self.open_flags = flags.value
        output_pointer._obj.value = 0xCAFE0000
        return self.open_error

    def _close_mem_handle(self, device_pointer):
        self.closed_pointer = device_pointer.value
        return 0

    def _pointer_get_attribute(self, output_pointer, attribute, device_pointer):
        self.pointer_query = (attribute.value, device_pointer.value)
        output_pointer._obj.value = 0xABCD0000
        return 0

    def _malloc_with_flags(self, output_pointer, size, flags):
        self.allocated = (size.value, flags.value)
        output_pointer._obj.value = 0xDEAD0000
        return 0

    def _memset(self, device_pointer, value, size):
        self.memset = (device_pointer.value, value, size.value)
        return self.memset_error

    def _free(self, device_pointer):
        self.freed_pointer = device_pointer.value
        return 0


def test_hip_ipc_round_trip_preserves_handle_and_pointer_offsets():
    hip = _FakeHip()
    runtime = HipRuntime(hip)

    allocation_base = runtime.get_allocation_base(0xABCD1234)
    handle = runtime.get_ipc_handle(allocation_base)
    mapped_base = runtime.open_ipc_handle(handle)
    runtime.close_ipc_handle(mapped_base)

    assert allocation_base == 0xABCD0000
    assert hip.pointer_query == (11, 0xABCD1234)
    assert hip.exported_pointer == allocation_base
    assert handle == bytes(range(HIP_IPC_HANDLE_BYTES))
    assert hip.opened_handle == handle
    assert hip.open_flags == HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
    assert mapped_base == 0xCAFE0000
    assert hip.closed_pointer == mapped_base


def test_uncached_allocation_is_zeroed_and_freed():
    hip = _FakeHip()
    runtime = HipRuntime(hip)

    device_pointer = runtime.allocate_uncached(4096)
    runtime.free_device_memory(device_pointer)

    assert device_pointer == 0xDEAD0000
    assert hip.allocated == (4096, HIP_DEVICE_MALLOC_UNCACHED)
    assert hip.memset == (device_pointer, 0, 4096)
    assert hip.freed_pointer == device_pointer


def test_uncached_allocation_is_freed_when_zeroing_fails():
    hip = _FakeHip()
    hip.memset_error = 17
    runtime = HipRuntime(hip)

    with pytest.raises(RuntimeError, match="hipMemset failed: invalid handle"):
        runtime.allocate_uncached(4096)

    assert hip.freed_pointer == 0xDEAD0000


def test_open_ipc_handle_reports_invalid_size_and_hip_error():
    hip = _FakeHip()
    runtime = HipRuntime(hip)

    with pytest.raises(ValueError, match="expected a 64-byte HIP IPC handle, got 3 bytes"):
        runtime.open_ipc_handle(b"bad")

    hip.open_error = 17
    with pytest.raises(RuntimeError, match="hipIpcOpenMemHandle failed: invalid handle"):
        runtime.open_ipc_handle(bytes(HIP_IPC_HANDLE_BYTES))


def test_allreduce_compatibility_entrypoints_use_the_shared_runtime(monkeypatch):
    hip = _FakeHip()
    monkeypatch.setattr(hip_ipc, "_DEFAULT_RUNTIME", HipRuntime(hip))

    allocation_base = FlyDSLAllreduce._get_alloc_base_ptr(0xABCD1234)
    handle = FlyDSLAllreduce._get_mem_handle_bytes(allocation_base)
    mapped_base = FlyDSLAllreduce._open_mem_handle(handle)
    FlyDSLAllreduce._close_mem_handle(mapped_base)
    uncached = FlyDSLAllreduce._alloc_uncached(2048)
    FlyDSLAllreduce._free_device_mem(uncached)

    assert allocation_base == 0xABCD0000
    assert mapped_base == 0xCAFE0000
    assert hip.closed_pointer == mapped_base
    assert hip.allocated == (2048, HIP_DEVICE_MALLOC_UNCACHED)
    assert hip.freed_pointer == uncached
