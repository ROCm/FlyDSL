# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ctypes
import importlib
import threading
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional

from .._mlir import ir
from .._mlir.execution_engine import ExecutionEngine
from .protocol import fly_pointers


# C prototype for the runtime callback installed by
# mgpuSetModuleLoadCallback.  Keep at module scope so ctypes generates
# the type once; per-compilation wrappers simply wrap a fresh Python
# closure in this CFUNCTYPE.
_ModuleLoadCb = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)


def _qualname(fn: Callable) -> Optional[str]:
    """Serialise a callable as ``module:qualname``; return None if not possible."""
    mod = getattr(fn, "__module__", None)
    qn = getattr(fn, "__qualname__", None)
    if not mod or not qn:
        return None
    return f"{mod}:{qn}"


def _resolve_qualname(ref: str) -> Optional[Callable]:
    """Inverse of _qualname; silently returns None on failure."""
    try:
        mod_name, qn = ref.split(":", 1)
        obj = importlib.import_module(mod_name)
        for part in qn.split("."):
            obj = getattr(obj, part)
        return obj
    except Exception:
        return None


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
        source_ir: Optional[str] = None,
        post_load_processors: Optional[List[Callable]] = None,
    ):
        self._ir_text = str(compiled_module)
        self._entry = func_name
        self._source_ir = source_ir
        self._post_load_processors = post_load_processors or []
        self._module = None
        self._engine = None
        self._func_exe = None
        self._lock = threading.Lock()
        self._packer = _ArgPacker()

    def __getstate__(self):
        # Serialise post-load processors by fully-qualified name so the
        # pickle stream contains no concrete callables.  Callables that
        # lack a module:qualname pair are dropped; the unpickler will
        # warn once so missing hooks are visible.
        refs: List[str] = []
        dropped = 0
        for p in self._post_load_processors:
            ref = _qualname(p)
            if ref is None:
                dropped += 1
            else:
                refs.append(ref)
        return {
            "ir_text": self._ir_text,
            "entry": self._entry,
            "source_ir": self._source_ir,
            "processor_refs": refs,
            "processors_dropped": dropped,
        }

    def __setstate__(self, state):
        self._ir_text = state["ir_text"]
        self._entry = state["entry"]
        self._source_ir = state["source_ir"]
        self._post_load_processors = []
        missing: List[str] = []
        for ref in state.get("processor_refs", []):
            fn = _resolve_qualname(ref)
            if fn is None:
                missing.append(ref)
            else:
                self._post_load_processors.append(fn)
        if missing:
            warnings.warn(
                "CompiledArtifact was unpickled without some post-load "
                f"processors (could not resolve: {missing}); kernels that "
                "rely on them may fail at launch time.",
                RuntimeWarning,
                stacklevel=2,
            )
        if state.get("processors_dropped"):
            warnings.warn(
                f"{state['processors_dropped']} post-load processor(s) could "
                "not be pickled (no module:qualname); they will not be "
                "re-attached on unpickle.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._module = None
        self._engine = None
        self._func_exe = None
        self._lock = threading.Lock()
        self._packer = _ArgPacker()

    def _ensure_engine(self):
        with self._lock:
            if self._engine is not None:
                return

            loaded_modules: List[int] = []
            cb_ref: Optional[ctypes._CFuncPtr] = None
            runtime_lib: Optional[ctypes.CDLL] = None

            if self._post_load_processors:
                runtime_lib = ctypes.CDLL(str(_resolve_runtime_libs()[0]))
                # argtypes use c_void_p so we can pass both a ctypes
                # CFUNCTYPE instance (via ctypes.cast) and None (NULL)
                # through the same entry point.
                runtime_lib.mgpuSetModuleLoadCallback.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p,
                ]
                runtime_lib.mgpuSetModuleLoadCallback.restype = None

                @_ModuleLoadCb
                def _on_module_load(module, _user):
                    loaded_modules.append(module)

                # Keep a strong reference to the ctypes trampoline for
                # the entire engine-init window; letting it be GC'd
                # while the C++ runtime still holds the pointer would
                # SIGSEGV on the next hipModuleLoadData.
                cb_ref = _on_module_load
                runtime_lib.mgpuSetModuleLoadCallback(
                    ctypes.cast(cb_ref, ctypes.c_void_p), None,
                )

            try:
                # Create context and immediately exit the with-block, but
                # keep a reference (self._ctx) so the context is not
                # garbage-collected.  Destroying the context while
                # ExecutionEngine still holds HSA code objects causes GPU
                # memory access faults.
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
                self._ctx = ctx
            finally:
                if runtime_lib is not None:
                    runtime_lib.mgpuSetModuleLoadCallback(None, None)
                # Release the ctypes trampoline now that the C++ runtime
                # no longer holds a pointer to it.
                del cb_ref

            for proc in self._post_load_processors:
                for module in loaded_modules:
                    proc(module)

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
