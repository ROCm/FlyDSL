# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ctypes
import os
import subprocess
import threading
from functools import lru_cache
from pathlib import Path
from typing import List

from .._mlir import ir
from .._mlir.execution_engine import ExecutionEngine
from ..utils import env
from .protocol import fly_pointers


def _find_asan_runtime_lib() -> str:
    """Find the AddressSanitizer runtime library path.

    Searches common locations for libasan.so on ROCm/AMD platforms.
    Returns empty string if not found.
    """
    # Common ASan library names (versioned)
    asan_names = [
        "libasan.so",
        "libasan.so.8",
        "libasan.so.7",
        "libasan.so.6",
        "libasan.so.5",
        "libasan.so.4",
    ]

    # Search paths
    search_paths = []

    # 1. Check ROCM_PATH
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    if rocm_path:
        search_paths.extend([
            Path(rocm_path) / "lib",
            Path(rocm_path) / "llvm" / "lib",
        ])

    # 2. Check system library paths
    search_paths.extend([
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/lib64"),
        Path("/usr/lib"),
        Path("/lib/x86_64-linux-gnu"),
        Path("/lib64"),
    ])

    # 3. Check LD_LIBRARY_PATH
    ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
    for path in ld_lib_path.split(":"):
        if path:
            search_paths.append(Path(path))

    # Try to find the library
    for search_dir in search_paths:
        if not search_dir.exists():
            continue
        for name in asan_names:
            lib_path = search_dir / name
            if lib_path.exists():
                return str(lib_path)

    # Try using ldconfig as a last resort
    try:
        result = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "libasan.so" in line:
                    parts = line.split("=>")
                    if len(parts) >= 2:
                        path = parts[-1].strip()
                        if os.path.exists(path):
                            return path
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return ""


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

    result = [str(p) for p in libs]

    # Add ASan runtime library if enabled
    if env.debug.enable_asan:
        asan_lib = _find_asan_runtime_lib()
        if asan_lib:
            result.append(asan_lib)
        else:
            import warnings

            warnings.warn(
                "FLYDSL_DEBUG_ENABLE_ASAN is set but libasan.so was not found. "
                "ASan instrumentation may not work. "
                "Please ensure ROCm ASan is installed (e.g., install rocm-llvm packages) "
                "or set LD_LIBRARY_PATH to include the ASan runtime library path.",
                RuntimeWarning,
                stacklevel=2,
            )

    return result


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

    @property
    def ir(self) -> str:
        return self._ir_text

    @property
    def source_ir(self) -> str:
        return self._source_ir
