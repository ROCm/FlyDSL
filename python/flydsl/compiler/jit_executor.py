import ctypes
import threading
from functools import lru_cache
from pathlib import Path
from typing import List

from .._mlir import ir
from .._mlir.execution_engine import ExecutionEngine
from .protocol import get_c_pointers


@lru_cache(maxsize=1)
def _get_mlir_runtime_libs() -> List[str]:
    mlir_libs_dir = Path(__file__).resolve().parent.parent / "_mlir" / "_mlir_libs"
    lib_names = ["libmlir_rocm_runtime.so", "libmlir_c_runner_utils.so"]
    return [str(mlir_libs_dir / name) for name in lib_names]


class JitCompiledFunction:
    def __init__(
        self,
        compiled_module: ir.Module,
        func_name: str,
        original_ir: str = None,
    ):
        self._compiled_ir = str(compiled_module)
        self._func_name = func_name
        self._original_ir = original_ir
        self._module = None
        self._engine = None
        self._engine_lock = threading.Lock()
        self._tls = threading.local()

    def __getstate__(self):
        return {
            "compiled_ir": self._compiled_ir,
            "func_name": self._func_name,
            "original_ir": self._original_ir,
        }

    def __setstate__(self, state):
        self._compiled_ir = state["compiled_ir"]
        self._func_name = state["func_name"]
        self._original_ir = state["original_ir"]
        self._module = None
        self._engine = None
        self._engine_lock = threading.Lock()
        self._tls = threading.local()

    def _init_engine(self):
        with self._engine_lock:
            if self._engine is not None:
                return

            with ir.Context() as ctx:
                ctx.load_all_available_dialects()
                self._module = ir.Module.parse(self._compiled_ir)
                self._engine = ExecutionEngine(
                    self._module,
                    opt_level=3,
                    shared_libs=_get_mlir_runtime_libs(),
                )
                self._engine.initialize()

    def _get_packed_args_buffer(self, size: int):
        buf = getattr(self._tls, "packed_args", None)
        capacity = getattr(self._tls, "capacity", 0)
        if buf is None or capacity < size:
            buf = (ctypes.c_void_p * size)()
            self._tls.packed_args = buf
            self._tls.capacity = size
        return buf

    def __call__(self, *args, **kwargs):
        if self._engine is None:
            self._init_engine()

        all_c_ptrs: List[ctypes.c_void_p] = []
        for arg in args:
            all_c_ptrs.extend(get_c_pointers(arg))

        if not hasattr(self, '_func_exe'):
            func_ptr = self._engine.raw_lookup(self._func_name)
            self._func_exe = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(func_ptr)

        num_args = len(all_c_ptrs)
        packed_args = self._get_packed_args_buffer(num_args)
        for i, ptr in enumerate(all_c_ptrs):
            packed_args[i] = ptr

        return self._func_exe(packed_args)

    def print_ir(self, compiled: bool = True):
        if compiled:
            print("=" * 60)
            print("Compiled MLIR IR:")
            print("=" * 60)
            print(self._compiled_ir)
        else:
            if self._original_ir is None:
                print("Original IR not available")
            else:
                print("=" * 60)
                print("Original MLIR IR:")
                print("=" * 60)
                print(self._original_ir)

    @property
    def ir(self) -> str:
        return self._compiled_ir

    @property
    def original_ir(self) -> str:
        return self._original_ir
