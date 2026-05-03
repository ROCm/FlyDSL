# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ctypes
import importlib
import pickle
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, List, Optional

from .._mlir import ir
from .._mlir.execution_engine import ExecutionEngine
from .protocol import fly_pointers

_GPU_MODULE_INIT = "flydsl_gpu_module_init"
_GPU_MODULE_LOAD_TO_DEVICE = "flydsl_gpu_module_load_to_device"


def _qualname(fn: Callable) -> Optional[str]:
    """Serialise a callable as ``module:qualname``; return None if not possible.

    Rejects anything that can't round-trip through ``_resolve_qualname``:

    - Missing ``__module__`` / ``__qualname__``.
    - ``__qualname__`` containing ``<`` (lambdas → ``<lambda>``,
      nested / closure functions → ``<locals>``, comprehensions, etc.).
    - Bound methods: ``__qualname__`` looks fine (``Class.method``) but
      resolving it yields the *unbound* function, silently dropping ``self``.

    As a final safety net we verify the generated ref actually resolves
    back to *fn* in the current process — if not, we refuse.
    """
    mod = getattr(fn, "__module__", None)
    qn = getattr(fn, "__qualname__", None)
    if not mod or not qn:
        return None
    if "<" in qn:
        return None
    if getattr(fn, "__self__", None) is not None:
        return None
    ref = f"{mod}:{qn}"
    # Round-trip check: guarantees the symbol is actually reachable
    # under the name we plan to write into the pickle stream.
    if _resolve_qualname(ref) is not fn:
        return None
    return ref


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
            raise FileNotFoundError(f"Required JIT runtime library not found: {lib}\n" f"Please rebuild the project.")
    return [str(p) for p in libs]


def _pack_ciface_args(*args) -> ctypes.c_void_p:
    """Pack argument addresses for MLIR's `_mlir_ciface_*` wrapper ABI."""
    packed = (ctypes.c_void_p * len(args))()
    for i, arg in enumerate(args):
        packed[i] = ctypes.addressof(arg)
    packed_addr = ctypes.c_void_p(ctypes.addressof(packed))
    packed_addr_ptr = ctypes.pointer(packed_addr)
    packed_addr_ptr_ptr = ctypes.pointer(packed_addr_ptr)
    # Keep the intermediate ctypes objects alive by attaching them to the
    # returned c_void_p for the duration of the call expression.
    packed_ciface_arg = ctypes.cast(packed_addr_ptr_ptr, ctypes.c_void_p)
    packed_ciface_arg._keepalive = (  # type: ignore[attr-defined]
        packed,
        packed_addr,
        packed_addr_ptr,
        packed_addr_ptr_ptr,
        args,
    )
    return packed_ciface_arg


class GpuJitModule:
    """Owns GPU modules loaded for one ExecutionEngine."""

    def __init__(self, engine: ExecutionEngine, modules: List[int]):
        self.engine = engine
        self.modules = list(modules)
        self._runtime_lib = ctypes.CDLL(str(_resolve_runtime_libs()[0]))
        self._runtime_lib.mgpuModuleUnload.argtypes = [ctypes.c_void_p]
        self._runtime_lib.mgpuModuleUnload.restype = None
        self._unloaded = False

    def unload(self) -> None:
        if self._unloaded:
            return
        try:
            for module in self.modules:
                if module:
                    self._runtime_lib.mgpuModuleUnload(ctypes.c_void_p(module))
            self.modules.clear()
        finally:
            self._unloaded = True

    def __del__(self):
        self.unload()


def _load_gpu_modules(engine: ExecutionEngine) -> List[int]:
    """Load embedded GPU modules through explicit symbols emitted by FlyDSL."""
    init_ptr = engine.raw_lookup(_GPU_MODULE_INIT)
    load_ptr = engine.raw_lookup(_GPU_MODULE_LOAD_TO_DEVICE)
    if not init_ptr or not load_ptr:
        raise RuntimeError(
            "compiled module does not expose FlyDSL ROCm module loader symbols; "
            "make sure the FlyDSL ROCm explicit-module offloading handler is registered"
        )

    init = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(init_ptr)
    load = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(load_ptr)

    module = ctypes.c_void_p()
    err = ctypes.c_int32(0)
    packed_ciface_arg = _pack_ciface_args(module, err)
    init(packed_ciface_arg)
    if err.value != 0:
        raise RuntimeError(f"{_GPU_MODULE_INIT} failed with error code {err.value}")
    load(packed_ciface_arg)
    if err.value != 0 or not module.value:
        raise RuntimeError(f"{_GPU_MODULE_LOAD_TO_DEVICE} failed with error code {err.value}")
    return [module.value]


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
        link_libs: Optional[List[str]] = None,
        uses_explicit_module: bool = False,
    ):
        self._ir_text = str(compiled_module)
        self._entry = func_name
        self._source_ir = source_ir
        self._post_load_processors = post_load_processors or []
        self._link_libs = link_libs or []
        self._uses_explicit_module = uses_explicit_module
        self._module = None
        self._engine = None
        self._jit_module = None
        self._func_exe = None
        self._lock = threading.Lock()
        self._packer = _ArgPacker()

    def __getstate__(self):
        # Serialise post-load processors by fully-qualified name so the
        # pickle stream carries no concrete callables.
        #
        # If any processor cannot be represented as module:qualname
        # (e.g. lambdas, functools.partial, bound methods) we *refuse*
        # to pickle instead of silently dropping it.  Silently dropping
        # would let the disk cache round-trip a kernel that later
        # launches without running its required initialiser, and the
        # first kernel launch on the next process would GPU-fault on
        # uninitialised device-side globals — with a stack that gives
        # no hint about the missing processor.  Raising here means:
        # (a) the failure is visible during cache-write, before any
        # user sees a mysterious fault, and (b) the caller (typically
        # the on-disk cache layer) can fall back to "memory cache
        # only" for this artifact without crashing the run.
        refs: List[str] = []
        unpicklable: List[str] = []
        for p in self._post_load_processors:
            ref = _qualname(p)
            if ref is None:
                unpicklable.append(repr(p))
            else:
                refs.append(ref)
        if unpicklable:
            raise pickle.PicklingError(
                "CompiledArtifact has post-load processors that are not "
                "representable as module:qualname: "
                f"{unpicklable}.  Wrap them in a top-level function in a "
                "regular Python module so the on-disk cache can re-import "
                "them after pickle round-trip (lambdas, partial, and bound "
                "methods are not supported).  If this artifact genuinely "
                "must not be cached to disk, suppress the disk-cache write "
                "path for it."
            )
        return {
            "ir_text": self._ir_text,
            "entry": self._entry,
            "source_ir": self._source_ir,
            "processor_refs": refs,
            "link_libs": self._link_libs,
            "uses_explicit_module": self._uses_explicit_module,
        }

    def __setstate__(self, state):
        self._ir_text = state["ir_text"]
        self._entry = state["entry"]
        self._source_ir = state["source_ir"]
        self._link_libs = state.get("link_libs", [])
        self._uses_explicit_module = state.get("uses_explicit_module", False)
        self._post_load_processors = []
        missing: List[str] = []
        for ref in state.get("processor_refs", []):
            fn = _resolve_qualname(ref)
            if fn is None:
                missing.append(ref)
            else:
                self._post_load_processors.append(fn)
        if missing:
            raise pickle.UnpicklingError(
                "CompiledArtifact could not resolve required post-load " f"processors: {missing}"
            )
        self._module = None
        self._engine = None
        self._jit_module = None
        self._func_exe = None
        self._lock = threading.Lock()
        self._packer = _ArgPacker()

    def _ensure_engine(self):
        with self._lock:
            if self._engine is not None:
                return

            # Keep the Context alive on self._ctx: destroying it
            # while ExecutionEngine still holds HSA code objects
            # causes GPU memory access faults.
            ctx = ir.Context()
            with ctx:
                ctx.load_all_available_dialects()
                module = ir.Module.parse(self._ir_text)
                engine = ExecutionEngine(
                    module,
                    opt_level=3,
                    shared_libs=_resolve_runtime_libs(),
                )
                engine.initialize()

            if self._uses_explicit_module:
                loaded_modules = _load_gpu_modules(engine)

                # Post-condition: if callers registered post-load
                # processors, at least one module load MUST have been
                # observed.  Zero observations means the explicit module
                # loader did not run; fail loud here rather than let
                # kernel launch segfault on uninitialised device globals.
                if self._post_load_processors and not loaded_modules:
                    raise RuntimeError(
                        "post_load_processors registered but no hipModuleLoad "
                        "was observed during ExecutionEngine.initialize(). "
                        "Device-side globals (e.g. mori shmem's "
                        "globalGpuStates) will be uninitialised at kernel "
                        "launch.  Check that the compiled module contains a GPU "
                        "binary and that mgpuModuleLoad is still the ROCm runtime "
                        "loader symbol used by MLIR."
                    )

                jit_module = GpuJitModule(engine, loaded_modules)
                for proc in self._post_load_processors:
                    for mod_handle in loaded_modules:
                        proc(mod_handle)
            else:
                jit_module = None

            self._module = module
            self._engine = engine
            self._jit_module = jit_module
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
