import ctypes
import threading
from functools import lru_cache
from pathlib import Path
from typing import List

from .._mlir import ir
from .._mlir.execution_engine import ExecutionEngine
from ..expr.typing import Stream
from .protocol import fly_pointers


def _get_current_gpu_stream() -> int:
    """Return the raw stream handle (hipStream_t / cudaStream_t) for the
    current PyTorch CUDA stream.  Falls back to the default stream (0)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.current_stream().cuda_stream
    except Exception:
        pass
    return 0


@lru_cache(maxsize=1)
def _resolve_runtime_libs() -> List[str]:
    mlir_libs_dir = Path(__file__).resolve().parent.parent / "_mlir" / "_mlir_libs"
    rocm_rt = mlir_libs_dir / "libfly_jit_runtime.so"
    if not rocm_rt.exists():
        rocm_rt = mlir_libs_dir / "libmlir_rocm_runtime.so"
    return [str(rocm_rt), str(mlir_libs_dir / "libmlir_c_runner_utils.so")]


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
        self._lock = threading.Lock()
        self._packer = _ArgPacker()

    def _ensure_engine(self):
        with self._lock:
            if self._engine is not None:
                return

            with ir.Context() as ctx:
                ctx.load_all_available_dialects()
                self._module = ir.Module.parse(self._ir_text)
                self._engine = ExecutionEngine(
                    self._module,
                    opt_level=3,
                    shared_libs=_resolve_runtime_libs(),
                )
                self._engine.initialize()

    def __call__(self, *args, **kwargs):
        if self._engine is None:
            self._ensure_engine()

        all_c_ptrs: List[ctypes.c_void_p] = []
        has_stream_arg = False
        for arg in args:
            if isinstance(arg, Stream):
                has_stream_arg = True
            all_c_ptrs.extend(fly_pointers(arg))

        if not has_stream_arg:
            stream_ptr = kwargs.pop("stream", None)
            if stream_ptr is None:
                stream_ptr = _get_current_gpu_stream()
            stream_val = ctypes.c_void_p(stream_ptr)
            self._tls.stream_val = stream_val
            all_c_ptrs.append(ctypes.cast(ctypes.pointer(stream_val), ctypes.c_void_p))

        func_ptr = self._engine.raw_lookup(self._entry)
        func_exe = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(func_ptr)

        packed_args = self._packer.pack(all_c_ptrs)

        return func_exe(packed_args)

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
