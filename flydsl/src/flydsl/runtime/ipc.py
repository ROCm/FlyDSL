"""HIP IPC helpers (experimental).

This module provides a minimal "IPC runtime" in Python:
- Export a device allocation as an IPC handle (bytes)
- Open/close an IPC handle to obtain a peer-accessible device pointer

Notes
-----
1) HIP IPC is a host-side feature. GPU kernels cannot call these APIs.
2) This module is intentionally small and does not try to manage process
   rendezvous (that's typically done via torch.distributed, sockets, etc.).
3) In FlyDSL, compiled host entrypoints are lowered with bare-pointer calling
   conventions (`gpu-to-llvm{use-bare-pointers-for-host=true,...}`), so you can
   pass a raw device pointer (int) as a kernel argument where a memref is
   expected, as long as the shape/strides match the compiled kernel.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Optional, Tuple, Union


# HIP IPC handle is an opaque 64-byte blob on both CUDA and HIP in practice.
_HIP_IPC_HANDLE_BYTES = 64


class HipIpcError(RuntimeError):
    pass


def _load_hip() -> ctypes.CDLL:
    # ROCm typically ships libamdhip64.so (with or without SONAME suffix).
    for name in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise HipIpcError(
        "Failed to load HIP runtime library (libamdhip64.so). "
        "Make sure ROCm is installed and LD_LIBRARY_PATH includes the HIP libs."
    )


_hip = _load_hip()


class hipIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * _HIP_IPC_HANDLE_BYTES)]


# hipIpcMemLazyEnablePeerAccess
HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1


def _hip_check(err: int, *, what: str):
    if int(err) == 0:
        return
    # Try to decode error string if available.
    try:
        _hip.hipGetErrorString.restype = ctypes.c_char_p
        _hip.hipGetErrorString.argtypes = [ctypes.c_int]
        s = _hip.hipGetErrorString(int(err))
        msg = s.decode("utf-8", errors="replace") if s else f"hipError({err})"
    except Exception:
        msg = f"hipError({err})"
    raise HipIpcError(f"{what} failed: {msg}")


# Bind prototypes (best-effort; HIP uses C ABI).
_hip.hipIpcGetMemHandle.restype = ctypes.c_int
_hip.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(hipIpcMemHandle_t), ctypes.c_void_p]

_hip.hipIpcOpenMemHandle.restype = ctypes.c_int
_hip.hipIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), hipIpcMemHandle_t, ctypes.c_uint]

_hip.hipIpcCloseMemHandle.restype = ctypes.c_int
_hip.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]


def _as_handle(handle_bytes: Union[bytes, bytearray, memoryview]) -> hipIpcMemHandle_t:
    b = bytes(handle_bytes)
    if len(b) != _HIP_IPC_HANDLE_BYTES:
        raise ValueError(f"Expected IPC handle size {_HIP_IPC_HANDLE_BYTES} bytes, got {len(b)}")
    h = hipIpcMemHandle_t()
    ctypes.memmove(ctypes.byref(h), b, _HIP_IPC_HANDLE_BYTES)
    return h


def _handle_to_bytes(h: hipIpcMemHandle_t) -> bytes:
    return bytes(ctypes.string_at(ctypes.byref(h), _HIP_IPC_HANDLE_BYTES))


@dataclass(frozen=True)
class IpcHandle:
    """Opaque handle bytes + byte offset into the underlying allocation."""

    handle: bytes
    offset_bytes: int = 0


def get_ipc_handle_from_base_ptr(base_ptr: int) -> bytes:
    """Export an IPC handle for a device allocation base pointer."""
    h = hipIpcMemHandle_t()
    err = _hip.hipIpcGetMemHandle(ctypes.byref(h), ctypes.c_void_p(int(base_ptr)))
    _hip_check(err, what="hipIpcGetMemHandle")
    return _handle_to_bytes(h)


def open_ipc_handle(handle: Union[bytes, bytearray, memoryview], *, flags: int = HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS) -> int:
    """Open an IPC handle and return the base device pointer (as int)."""
    h = _as_handle(handle)
    out_ptr = ctypes.c_void_p()
    err = _hip.hipIpcOpenMemHandle(ctypes.byref(out_ptr), h, ctypes.c_uint(int(flags)))
    _hip_check(err, what="hipIpcOpenMemHandle")
    return int(out_ptr.value)


def close_ipc_handle(base_ptr: int) -> None:
    err = _hip.hipIpcCloseMemHandle(ctypes.c_void_p(int(base_ptr)))
    _hip_check(err, what="hipIpcCloseMemHandle")


class IpcMapping:
    """Context manager to open/close an IPC handle."""

    def __init__(self, handle: Union[bytes, bytearray, memoryview], *, offset_bytes: int = 0, flags: int = HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS):
        self._handle = bytes(handle)
        self._offset = int(offset_bytes)
        self._flags = int(flags)
        self._base_ptr: Optional[int] = None

    def __enter__(self) -> int:
        self._base_ptr = open_ipc_handle(self._handle, flags=self._flags)
        return int(self._base_ptr + self._offset)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._base_ptr is not None:
            close_ipc_handle(self._base_ptr)
            self._base_ptr = None


def get_ipc_handle(tensor) -> IpcHandle:
    """Get IPC handle for a torch tensor (base allocation) and the tensor's byte offset.

    Requirements:
    - `tensor` must be a CUDA/ROCm tensor
    - This uses tensor storage base pointer to compute offset, so views are OK
      as long as the underlying storage is a single allocation.
    """
    # Import torch lazily to keep flydsl core import-light.
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("get_ipc_handle(tensor) requires PyTorch installed") from e

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got: {type(tensor)}")
    if not tensor.is_cuda:
        raise ValueError("Tensor must be on CUDA/ROCm device")

    # Storage base pointer (device).
    storage = tensor.untyped_storage() if hasattr(tensor, "untyped_storage") else tensor.storage()
    base_ptr = int(storage.data_ptr())
    ptr = int(tensor.data_ptr())
    offset = ptr - base_ptr
    if offset < 0:
        raise RuntimeError("Unexpected negative offset between tensor.data_ptr and storage.data_ptr")

    handle = get_ipc_handle_from_base_ptr(base_ptr)
    return IpcHandle(handle=handle, offset_bytes=offset)


def open_ipc_tensor_ptr(ipc: IpcHandle, *, flags: int = HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS) -> IpcMapping:
    """Open an IPC mapping and return a context manager yielding `device_ptr = base+offset`."""
    return IpcMapping(ipc.handle, offset_bytes=ipc.offset_bytes, flags=flags)


def split_ptr_to_ranked_1d(ptr: int, *, elem_size: int, offset_elems: int = 0) -> int:
    """Convenience: compute ptr + offset_elems*elem_size (for manual views)."""
    return int(ptr) + int(offset_elems) * int(elem_size)


