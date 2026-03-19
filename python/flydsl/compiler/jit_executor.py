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
    mlir_libs_dir = Path(__file__).resolve().parent.parent / "_mlir" / "_mlir_libs"
    return [str(mlir_libs_dir / "libfly_jit_runtime.so")]


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


_shmem_hook_installed = False
_shmem_hook_lock = threading.Lock()


def _ensure_shmem_hook():
    """Register mori shmem module-load hook in libfly_jit_runtime.so.

    After registration, every hipModuleLoadData performed by the MLIR
    ExecutionEngine will trigger ``mori.shmem.shmem_module_init`` so that
    ``globalGpuStates`` is injected into the loaded GPU module.
    """
    global _shmem_hook_installed
    if _shmem_hook_installed:
        return

    with _shmem_hook_lock:
        if _shmem_hook_installed:
            return

        import mori.shmem as ms

        runtime_lib = ctypes.CDLL(str(_resolve_runtime_libs()[0]))

        HOOK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

        def _on_module_load(hip_module_ptr):
            ms.shmem_module_init(hip_module_ptr)

        # Must keep a reference to prevent GC of the callback.
        _ensure_shmem_hook._callback = HOOK_TYPE(_on_module_load)
        runtime_lib.mgpuSetModuleLoadHook(_ensure_shmem_hook._callback)
        _shmem_hook_installed = True


class CompiledArtifact:
    def __init__(
        self,
        compiled_module: ir.Module,
        func_name: str,
        source_ir: str = None,
        needs_shmem: bool = False,
    ):
        self._ir_text = str(compiled_module)
        self._entry = func_name
        self._source_ir = source_ir
        self._needs_shmem = needs_shmem
        self._module = None
        self._engine = None
        self._func_exe = None   # cached ctypes function object (~30 μs/call saving)
        self._lock = threading.Lock()
        self._packer = _ArgPacker()

    def __getstate__(self):
        return {
            "ir_text": self._ir_text,
            "entry": self._entry,
            "source_ir": self._source_ir,
            "needs_shmem": self._needs_shmem,
        }

    def __setstate__(self, state):
        self._ir_text = state["ir_text"]
        self._entry = state["entry"]
        self._source_ir = state["source_ir"]
        self._needs_shmem = state.get("needs_shmem", False)
        self._module = None
        self._engine = None
        self._func_exe = None
        self._lock = threading.Lock()
        self._packer = _ArgPacker()

    def _ensure_engine(self):
        with self._lock:
            if self._engine is not None:
                return

            if self._needs_shmem:
                _ensure_shmem_hook()

            with ir.Context() as ctx:
                ctx.load_all_available_dialects()
                self._module = ir.Module.parse(self._ir_text)
                self._engine = ExecutionEngine(
                    self._module,
                    opt_level=3,
                    shared_libs=_resolve_runtime_libs(),
                )
                self._engine.initialize()

            # Cache the ctypes function object once (inside the lock to avoid
            # races). Avoids two expensive operations per __call__:
            # raw_lookup (~10 μs) and CFUNCTYPE(...)(func_ptr) (~20 μs),
            # saving ~30 μs per dispatch.
            func_ptr = self._engine.raw_lookup(self._entry)
            self._func_exe = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(func_ptr)

    def __call__(self, *args, **kwargs):
        if self._engine is None:
            self._ensure_engine()

        all_c_ptrs: List[ctypes.c_void_p] = []
        for arg in args:
            all_c_ptrs.extend(fly_pointers(arg))

        packed_args = self._packer.pack(all_c_ptrs)

        return self._func_exe(packed_args)

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
