# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Load AOT-exported .o or .so files back into Python for execution.

Supports both .o and .so files:
- .o files loaded via FlyBinaryLoader (LLVM JITLink, no gcc needed)
- .so files loaded via dlopen
- ``__getattr__`` for ergonomic function lookup

Usage::

    from flydsl.compiler.export import load_module

    mod = load_module("./build/my_kernel.o")  # or .so
    mod.gemm_fp16(packed_args)
"""

import ctypes
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional


@lru_cache(maxsize=1)
def _get_binary_loader():
    """Load libfly_binary_loader.so (LLVM LLJIT-based .o loader)."""
    lib_dir = Path(__file__).resolve().parent.parent.parent / "_mlir" / "_mlir_libs"
    loader_path = lib_dir / "libfly_binary_loader.so"
    if not loader_path.exists():
        return None

    lib = ctypes.CDLL(str(loader_path))

    lib.flyBinaryModuleCreate.argtypes = [
        ctypes.c_char_p,        # objData
        ctypes.c_size_t,        # objSize
        ctypes.POINTER(ctypes.c_char_p),  # sharedLibs
        ctypes.c_size_t,        # numLibs
    ]
    lib.flyBinaryModuleCreate.restype = ctypes.c_void_p

    lib.flyBinaryModuleLookup.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.flyBinaryModuleLookup.restype = ctypes.c_void_p

    lib.flyBinaryModuleDestroy.argtypes = [ctypes.c_void_p]
    lib.flyBinaryModuleDestroy.restype = None

    return lib


class BinaryKernelModule:
    """A loaded AOT module from an exported .o or .so file.

    For .o files: uses FlyBinaryLoader (LLVM JITLink) to load in-process.
    For .so files: loaded directly via dlopen.

    """

    def __init__(self, file_path: str):
        self._path = file_path
        self._lib: Optional[ctypes.CDLL] = None
        self._jit_handle: Optional[int] = None  # for .o via FlyBinaryLoader
        self._func_cache: dict = {}

    def _ensure_loaded(self):
        if self._lib is not None or self._jit_handle is not None:
            return
        if self._path.endswith(".so"):
            self._lib = ctypes.CDLL(self._path)
        elif self._path.endswith(".o"):
            self._load_object_file(self._path)
        else:
            raise ValueError(f"Unsupported file type: {self._path}. Use .o or .so")

    def _load_object_file(self, obj_path: str):
        """Load .o via FlyBinaryLoader (LLVM JITLink), fallback to gcc."""
        loader = _get_binary_loader()
        if loader is not None:
            self._load_via_jitlink(obj_path, loader)
        else:
            self._load_via_gcc(obj_path)

    def _load_via_jitlink(self, obj_path: str, loader):
        """Load .o in-process using LLVM LLJIT."""
        from ..jit_executor import _resolve_runtime_libs

        with open(obj_path, "rb") as f:
            obj_data = f.read()

        runtime_libs = _resolve_runtime_libs()
        c_libs = (ctypes.c_char_p * len(runtime_libs))(
            *[lib.encode() for lib in runtime_libs]
        )

        handle = loader.flyBinaryModuleCreate(
            obj_data, len(obj_data), c_libs, len(runtime_libs)
        )
        if not handle:
            raise RuntimeError(f"FlyBinaryLoader failed to load {obj_path}")
        self._jit_handle = handle
        self._loader = loader

    def _load_via_gcc(self, obj_path: str):
        """Fallback: link .o → .so using gcc, then dlopen."""
        import subprocess

        from ..jit_executor import _resolve_runtime_libs
        lib_dir = str(Path(_resolve_runtime_libs()[0]).parent)
        so_path = obj_path[:-2] + ".so"

        cmd = [
            "gcc", "-shared", "-o", so_path, obj_path,
            f"-L{lib_dir}", "-lfly_jit_runtime", "-lmlir_c_runner_utils",
            f"-Wl,-rpath,{lib_dir}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to link {obj_path} (gcc fallback):\n"
                f"  {' '.join(cmd)}\n  {result.stderr}"
            )
        self._lib = ctypes.CDLL(so_path)

    def _lookup_symbol(self, name: str) -> Optional[int]:
        """Look up a symbol, returns raw pointer or None."""
        if self._jit_handle is not None:
            ptr = self._loader.flyBinaryModuleLookup(
                self._jit_handle, name.encode()
            )
            return ptr if ptr else None
        elif self._lib is not None:
            try:
                return ctypes.cast(
                    getattr(self._lib, name), ctypes.c_void_p
                ).value
            except AttributeError:
                return None
        return None

    def _read_string_global(self, name: str) -> Optional[str]:
        """Read a null-terminated string global (llvm.array<N x i8>)."""
        if self._jit_handle is not None:
            ptr = self._lookup_symbol(name)
            if ptr:
                return ctypes.string_at(ptr).decode("utf-8")
            return None
        elif self._lib is not None:
            try:
                sym = ctypes.c_byte.in_dll(self._lib, name)
                return ctypes.string_at(ctypes.addressof(sym)).decode("utf-8")
            except (ValueError, OSError):
                return None
        return None

    def get_function(self, prefix: str):
        """Look up a function by its export prefix.

        Returns a ctypes callable: ``fn(packed_args: ctypes.c_void_p) -> None``.
        """
        if prefix in self._func_cache:
            return self._func_cache[prefix]

        self._ensure_loaded()

        func_name = self._read_string_global(f"{prefix}_entry_name")
        if func_name is None:
            raise KeyError(
                f"Function prefix '{prefix}' not found in module {self._path}."
            )

        ciface_name = f"_mlir_{func_name}"

        if self._jit_handle is not None:
            ptr = self._lookup_symbol(ciface_name)
            if not ptr:
                raise KeyError(f"Symbol '{ciface_name}' not found in module.")
            func_ptr = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(ptr)
        else:
            try:
                func_ptr = getattr(self._lib, ciface_name)
            except AttributeError:
                raise KeyError(f"Symbol '{ciface_name}' not found in module.")
            func_ptr.argtypes = [ctypes.c_void_p]
            func_ptr.restype = None

        self._func_cache[prefix] = func_ptr
        return func_ptr

    def __getattr__(self, prefix: str):
        """Ergonomic function lookup: ``mod.gemm_fp16(packed_args)``."""
        if prefix.startswith("_"):
            raise AttributeError(prefix)
        try:
            return self.get_function(prefix)
        except KeyError:
            raise AttributeError(
                f"No function with prefix '{prefix}' in {self._path}"
            )

    def __del__(self):
        if self._jit_handle is not None and hasattr(self, '_loader'):
            self._loader.flyBinaryModuleDestroy(self._jit_handle)
            self._jit_handle = None

    def __repr__(self):
        if self._jit_handle:
            status = "loaded (JITLink)"
        elif self._lib:
            status = "loaded (dlopen)"
        else:
            status = "not loaded"
        return f"<BinaryKernelModule path='{self._path}' {status}>"


def load_module(file_path: str) -> BinaryKernelModule:
    """Load an AOT-exported .o or .so file.

    For .o files: uses LLVM JITLink (in-process, no gcc needed).
    For .so files: loaded directly via dlopen.

    Falls back to gcc linking if FlyBinaryLoader is not available.


    Args:
        file_path: Path to the .o or .so file.

    Returns:
        BinaryKernelModule with ``.get_function(prefix)`` or ``__getattr__``.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Module file not found: {file_path}")
    return BinaryKernelModule(file_path)
