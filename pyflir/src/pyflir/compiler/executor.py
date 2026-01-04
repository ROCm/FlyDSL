"""Executor for FLIR via MLIR ExecutionEngine."""

from __future__ import annotations

import ctypes
import importlib.util
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SharedLibs:
    rocm_runtime: str
    runner_utils: str

    def as_list(self) -> List[str]:
        # De-duplicate while preserving order (ExecutionEngine just needs the
        # symbols to be available once).
        out: List[str] = []
        for p in (self.rocm_runtime, self.runner_utils):
            if p and p not in out:
                out.append(p)
        return out


def _default_mlir_lib_dir() -> Optional[Path]:
    try:
        spec = importlib.util.find_spec("_mlir._mlir_libs")
        if spec:
            if spec.submodule_search_locations:
                embedded_lib_dir = Path(next(iter(spec.submodule_search_locations)))
            elif spec.origin:
                embedded_lib_dir = Path(spec.origin).parent
            else:
                embedded_lib_dir = None
        else:
            embedded_lib_dir = None
        if embedded_lib_dir:
            for cand in (embedded_lib_dir, embedded_lib_dir / "lib"):
                if not cand.exists():
                    continue
                if (cand / "libflir_jit_runtime.so").exists() or any(cand.glob("libflir_jit_runtime.so.*")):
                    return cand
    except Exception:
        pass

    return None


def default_shared_libs(lib_dir: Optional[Path] = None) -> SharedLibs:
    if lib_dir is None:
        lib_dir = _default_mlir_lib_dir()
    if lib_dir is None:
        raise FileNotFoundError(
            "Could not locate FLIR JIT runtime library (expected `libflir_jit_runtime.so` "
            "under the embedded `_mlir/_mlir_libs/`).\n\n"
            "Fix:\n"
            "  - Build with `./build.sh` and use the embedded package root on PYTHONPATH, or\n"
            "  - Install the built wheel so `_mlir/_mlir_libs/libflir_jit_runtime.so` is present."
        )

    flir_rt = lib_dir / "libflir_jit_runtime.so"
    if not flir_rt.exists():
        cands = sorted(lib_dir.glob("libflir_jit_runtime.so.*"))
        if cands:
            flir_rt = cands[-1]
    if flir_rt.exists():
        # Thin ROCm runtime (mgpu* wrappers).
        return SharedLibs(str(flir_rt), str(flir_rt))

    raise FileNotFoundError(
        f"Missing FLIR JIT runtime lib in {lib_dir}. Expected "
        "`libflir_jit_runtime.so` (or `libflir_jit_runtime.so.*`)."
    )


class ExecutionEngineExecutor:
    """Execute host-side entrypoints compiled by FLIR via MLIR ExecutionEngine."""

    def __init__(
        self,
        jit_module,
        *,
        opt_level: int = 3,
        shared_libs: Optional[Sequence[str]] = None,
    ):
        from _mlir._mlir_libs._mlirExecutionEngine import ExecutionEngine  # type: ignore

        if shared_libs is None:
            shared_libs = default_shared_libs().as_list()

        self._llvm_sigs = self._extract_llvm_func_sigs(jit_module)
        self.engine = ExecutionEngine(jit_module, opt_level=opt_level, shared_libs=list(shared_libs))
        self.engine.initialize()

    @staticmethod
    def _extract_llvm_func_sigs(jit_module) -> Dict[str, List[str]]:
        """Parse `llvm.func` argument type strings from the lowered module."""
        asm = jit_module.operation.get_asm(enable_debug_info=False)
        pat = re.compile(r"llvm\.func\s+@([^\s(]+)\(([^)]*)\)")
        sigs: Dict[str, List[str]] = {}
        for m in pat.finditer(asm):
            name = m.group(1)
            args = m.group(2).strip()
            if not args:
                sigs[name] = []
                continue
            arg_types: List[str] = []
            for a in args.split(","):
                a = a.strip()
                if not a:
                    continue
                if ":" in a:
                    ty = a.split(":", 1)[1].strip()
                else:
                    ty = a
                arg_types.append(ty)
            sigs[name] = arg_types
        return sigs

    @staticmethod
    def _ctype_for_llvm_type(ty: str):
        ty = ty.strip()
        if ty == "!llvm.ptr":
            return ctypes.c_void_p
        if ty == "i1":
            return ctypes.c_bool
        if ty == "i8":
            return ctypes.c_int8
        if ty == "i16":
            return ctypes.c_int16
        if ty == "i32":
            return ctypes.c_int32
        if ty == "i64":
            return ctypes.c_int64
        if ty == "f32":
            return ctypes.c_float
        if ty == "f64":
            return ctypes.c_double
        return ctypes.c_void_p

    def __getattr__(self, name: str):
        # `ExecutionEngine.raw_lookup(name)` returns the packed-call interface,
        # i.e. a function pointer with signature `void(void**)`.
        sym = f"_mlir_ciface_{name}"
        func_ptr = 0
        sig_name = name

        # Prefer `_mlir_ciface_*` if it exists in the lowered module assembly.
        if sym in self._llvm_sigs:
            func_ptr = int(self.engine.raw_lookup(sym))
            sig_name = sym
        if func_ptr == 0:
            func_ptr = int(self.engine.raw_lookup(name))
            sig_name = name
        if func_ptr == 0:
            raise AttributeError(f"No such function: {name}") from None

        # Packed-call wrapper: void(void**)
        func_exe = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(func_ptr)

        def wrapper(*args):
            llvm_arg_tys = self._llvm_sigs.get(sig_name) or self._llvm_sigs.get(name) or []
            if len(args) != len(llvm_arg_tys):
                # Best-effort for 0-arg functions when signature couldn't be parsed.
                if len(args) == 0 and len(llvm_arg_tys) == 0:
                    empty = (ctypes.c_void_p * 0)()
                    return func_exe(empty)
                raise TypeError(f"{name} expects {len(llvm_arg_tys)} args, got {len(args)}")

            owned = []  # keep ctypes temporaries alive for the duration of the call
            arg_ptrs = []

            for a, ty in zip(args, llvm_arg_tys):
                ty = ty.strip()
                if ty == "!llvm.ptr":
                    # Tensor-like: any object with a `.data_ptr()` method returning an int.
                    if hasattr(a, "data_ptr") and callable(getattr(a, "data_ptr")):
                        v = ctypes.c_void_p(int(a.data_ptr()))
                    elif isinstance(a, ctypes.c_void_p):
                        v = a
                    elif isinstance(a, int):
                        v = ctypes.c_void_p(int(a))
                    else:
                        raise TypeError(f"Unsupported pointer arg type: {type(a)}")
                else:
                    cty = self._ctype_for_llvm_type(ty)
                    if isinstance(a, bool):
                        v = cty(bool(a))
                    elif isinstance(a, int):
                        v = cty(int(a))
                    elif isinstance(a, float):
                        v = cty(float(a))
                    else:
                        raise TypeError(f"Unsupported scalar arg type: {type(a)} for {ty}")

                owned.append(v)
                arg_ptrs.append(ctypes.cast(ctypes.pointer(v), ctypes.c_void_p))

            c_args = (ctypes.c_void_p * len(arg_ptrs))(*arg_ptrs)
            owned.append(c_args)
            return func_exe(c_args)

        return wrapper

    def __call__(self, *args):
        return self.__getattr__("__call__")(*args)


Executor = ExecutionEngineExecutor


