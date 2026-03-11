"""HIP-based kernel launcher that bypasses MLIR ExecutionEngine.

Extracts the GPU binary from compiled MLIR IR and uses HIP API
directly to load and launch kernels. This avoids LLVM symbol
conflicts when running inside a PyTorch/ROCm process.
"""

import ctypes
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import torch


@lru_cache(maxsize=1)
def _get_hip_lib():
    """Load libamdhip64.so for HIP API calls."""
    lib = ctypes.CDLL("libamdhip64.so")
    return lib


def _hip_check(status, msg="HIP API call"):
    if status != 0:
        raise RuntimeError(f"{msg} failed with error code {status}")


class HipKernelLauncher:
    """Launch a GPU kernel from compiled MLIR IR using HIP API directly."""

    def __init__(self, ir_text: str, kernel_name: str):
        self._ir_text = ir_text
        self._kernel_name = kernel_name
        self._module = None
        self._function = None
        self._lock = threading.Lock()

    @staticmethod
    def _extract_gpu_binary(ir_text: str) -> Optional[bytes]:
        """Extract the raw ELF binary from gpu.binary MLIR attribute."""
        pattern = r'bin\s*=\s*"((?:[^"\\]|\\.)*)"'
        match = re.search(pattern, ir_text, re.DOTALL)
        if not match:
            return None
        escaped = match.group(1)
        raw_bytes = bytearray()
        i = 0
        while i < len(escaped):
            if escaped[i] == '\\':
                i += 1
                if i < len(escaped):
                    c = escaped[i]
                    if c == 'n':
                        raw_bytes.append(0x0A)
                    elif c == 't':
                        raw_bytes.append(0x09)
                    elif c == '\\':
                        raw_bytes.append(0x5C)
                    elif c == '"':
                        raw_bytes.append(0x22)
                    elif c == '0' and i + 1 < len(escaped) and escaped[i + 1] == '0':
                        raw_bytes.append(0x00)
                        i += 1
                    else:
                        if len(escaped) > i + 1:
                            try:
                                raw_bytes.append(int(escaped[i:i + 2], 16))
                                i += 1
                            except ValueError:
                                raw_bytes.append(ord(c))
                        else:
                            raw_bytes.append(ord(c))
            else:
                raw_bytes.append(ord(escaped[i]))
            i += 1
        return bytes(raw_bytes)

    def _ensure_loaded(self):
        with self._lock:
            if self._function is not None:
                return

            binary = self._extract_gpu_binary(self._ir_text)
            if binary is None:
                raise RuntimeError("Could not extract GPU binary from MLIR IR")
            if not binary.startswith(b'\x7fELF'):
                raise RuntimeError(
                    f"Extracted binary is not ELF (starts with {binary[:4].hex()})"
                )

            hip = _get_hip_lib()
            module = ctypes.c_void_p()
            buf = ctypes.create_string_buffer(binary)
            status = hip.hipModuleLoadData(ctypes.byref(module), buf)
            _hip_check(status, "hipModuleLoadData")
            self._module = module

            func = ctypes.c_void_p()
            name = self._kernel_name.encode()
            status = hip.hipModuleGetFunction(ctypes.byref(func), module, name)
            _hip_check(status, f"hipModuleGetFunction({self._kernel_name})")
            self._function = func

    def launch(
        self,
        args: list,
        grid: tuple,
        block: tuple = (256, 1, 1),
        shared_mem: int = 0,
        stream: Optional[int] = None,
    ):
        """Launch the kernel with given arguments.

        Args:
            args: list of (ctypes value, ctypes type) pairs or raw ctypes values.
                  Tensors are automatically converted to device pointers.
            grid: (gx, gy, gz) grid dimensions
            block: (bx, by, bz) block dimensions
            shared_mem: shared memory size in bytes
            stream: HIP stream pointer (int). If None, uses current PyTorch stream.
        """
        self._ensure_loaded()

        c_args = []
        for a in args:
            if isinstance(a, torch.Tensor):
                c_args.append(ctypes.c_void_p(a.data_ptr()))
            elif isinstance(a, int):
                c_args.append(ctypes.c_int32(a))
            elif isinstance(a, ctypes._SimpleCData):
                c_args.append(a)
            else:
                c_args.append(ctypes.c_void_p(int(a)))

        arg_ptrs = (ctypes.c_void_p * len(c_args))()
        for i, a in enumerate(c_args):
            arg_ptrs[i] = ctypes.cast(ctypes.pointer(a), ctypes.c_void_p)

        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream

        hip = _get_hip_lib()
        gx, gy, gz = grid
        bx, by, bz = block
        status = hip.hipModuleLaunchKernel(
            self._function,
            ctypes.c_uint(int(gx)),
            ctypes.c_uint(int(gy)),
            ctypes.c_uint(int(gz)),
            ctypes.c_uint(int(bx)),
            ctypes.c_uint(int(by)),
            ctypes.c_uint(int(bz)),
            ctypes.c_uint(int(shared_mem)),
            ctypes.c_void_p(int(stream)),
            arg_ptrs,
            ctypes.c_void_p(0),
        )
        _hip_check(status, "hipModuleLaunchKernel")
