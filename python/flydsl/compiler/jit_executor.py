# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ctypes
import threading
from functools import lru_cache
from pathlib import Path
from typing import List

from .._mlir import ir
from .._mlir.execution_engine import ExecutionEngine
from .protocol import fly_pointers


@lru_cache(maxsize=1)
def _resolve_runtime_libs() -> List[str]:
    from .backends import get_backend

    backend = get_backend()
    mlir_libs_dir = Path(__file__).resolve().parent.parent / "_mlir" / "_mlir_libs"
    libs = [mlir_libs_dir / name for name in backend.jit_runtime_lib_basenames()]
    for lib in libs:
        if not lib.exists():
            raise FileNotFoundError(
                f"Required JIT runtime library not found: {lib}\n"
                f"Please rebuild the project."
            )
    return [str(p) for p in libs]


class _ArgPacker:
    """Thread-local buffer for packing C pointer arguments."""

    def __init__(self):
        self._tls = threading.local()

    def pack(self, ptrs: List[ctypes.c_void_p]):
        size = len(ptrs)
        buf = getattr(self._tls, "packed_args", None)
        capacity = getattr(self._tls, "capacity", 0)
        if buf is None or capacity < size:
            buf = (ctypes.c_void_p * size)()
            self._tls.packed_args = buf
            self._tls.capacity = size
        for i, ptr in enumerate(ptrs):
            buf[i] = ptr
        return buf


class CompiledArtifact:
    def __init__(
        self,
        compiled_module: ir.Module,
        func_name: str,
        source_ir: str = None,
    ):
        self._ir_text = str(compiled_module)
        self._entry = func_name
        self._source_ir = source_ir
        self._module = None
        self._engine = None
        self._func_exe = None
        self._lock = threading.Lock()
        self._packer = _ArgPacker()

    def __getstate__(self):
        return {
            "ir_text": self._ir_text,
            "entry": self._entry,
            "source_ir": self._source_ir,
        }

    def __setstate__(self, state):
        self._ir_text = state["ir_text"]
        self._entry = state["entry"]
        self._source_ir = state["source_ir"]
        self._module = None
        self._engine = None
        self._func_exe = None
        self._lock = threading.Lock()
        self._packer = _ArgPacker()

    def _ensure_engine(self):
        with self._lock:
            if self._engine is not None:
                return

            # Create context and immediately exit the with-block, but
            # keep a reference so the context is not garbage-collected.
            # Destroying the context while ExecutionEngine still holds
            # HSA code objects causes GPU memory access faults.
            ctx = ir.Context()
            with ctx:
                ctx.load_all_available_dialects()
                self._module = ir.Module.parse(self._ir_text)
                self._engine = ExecutionEngine(
                    self._module,
                    opt_level=3,
                    shared_libs=_resolve_runtime_libs(),
                )
                self._engine.initialize()
            # Store ctx to prevent GC (but no longer the active context)
            self._ctx = ctx

    def _get_func_exe(self):
        if self._func_exe is None:
            if self._engine is None:
                self._ensure_engine()
            func_ptr = self._engine.raw_lookup(self._entry)
            self._func_exe = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(func_ptr)
        return self._func_exe

    def __call__(self, *args, **kwargs):
        func_exe = self._get_func_exe()

        owned: list = []
        all_c_ptrs: List[ctypes.c_void_p] = []
        for arg in args:
            ptrs = fly_pointers(arg)
            owned.append(ptrs)
            owned.append(arg)
            all_c_ptrs.extend(ptrs)

        packed_args = self._packer.pack(all_c_ptrs)

        result = func_exe(packed_args)
        del owned
        return result

    def dump(self, compiled: bool = True):
        if compiled:
            print("=" * 60)
            print("Compiled MLIR IR:")
            print("=" * 60)
            print(self._ir_text)
        else:
            if self._source_ir is None:
                print("Original IR not available")
            else:
                print("=" * 60)
                print("Original MLIR IR:")
                print("=" * 60)
                print(self._source_ir)

    def dump_to_object(self, function_prefix: str, arch: str = "") -> bytes:
        """Export this artifact as a relocatable ELF .o file.

        The .o contains x86 host wrapper code + embedded GPU binary (HSACO),
        linkable into a shared library for C/C++ deployment without Python.

        Args:
            function_prefix: Prefix for exported symbols, e.g. "gemm_fp16_256x128".
            arch: Target GPU architecture. Auto-detected if empty.

        Returns:
            bytes: Contents of the ELF .o file.
        """
        from .export.object_export import dump_to_object

        return dump_to_object(self, function_prefix, arch=arch)

    def export_to_c(
        self,
        file_path: str,
        file_name: str,
        function_prefix: str,
        arch: str = "",
    ) -> None:
        """Export this artifact as C header + object file for HIP deployment.

        Produces {file_name}.h and {file_name}.o in the specified directory.
        Use ``load_module("{file_name}.o")`` to load in Python (links .o → .so
        automatically), or ``gcc -shared`` to link for C/C++ use.

        Args:
            file_path: Directory to write output files.
            file_name: Base name for output files (without extension).
            function_prefix: Symbol prefix, e.g. "gemm_fp16_256x128".
            arch: Target GPU architecture. Auto-detected if empty.
        """
        from .export.hip_header import export_to_c

        export_to_c(self, file_path, file_name, function_prefix, arch=arch)

    @property
    def ir(self) -> str:
        return self._ir_text

    @property
    def source_ir(self) -> str:
        return self._source_ir
