"""Executor for FLIR via MLIR ExecutionEngine."""

from __future__ import annotations

import ctypes
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
        return [self.rocm_runtime, self.runner_utils]


def _default_mlir_lib_dir() -> Optional[Path]:
    # Highest priority: explicit override.
    env_dir = os.environ.get("FLIR_MLIR_LIB_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists():
            return p

    # Next: MLIR_PATH from build.sh convention.
    mlir_path = os.environ.get("MLIR_PATH")
    if mlir_path:
        p = Path(mlir_path) / "lib"
        if p.exists():
            return p

    # Next: try to locate a sibling llvm-project next to FLIR_BUILD_DIR (or its parents).
    # This matches common build layouts where dsl2/ and llvm-project/ live side-by-side.
    build_dir = os.environ.get("FLIR_BUILD_DIR")
    if build_dir:
        b = Path(build_dir).resolve()
        for base in [b] + list(b.parents)[:3]:
            cand = base / "llvm-project" / "buildmlir" / "lib"
            if cand.exists():
                return cand

    # Repo-local default from build.sh: <repo>/../llvm-project/buildmlir/lib
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root.parent / "llvm-project" / "buildmlir" / "lib"
    if candidate.exists():
        return candidate

    return None


def default_shared_libs(lib_dir: Optional[Path] = None) -> SharedLibs:
    if lib_dir is None:
        lib_dir = _default_mlir_lib_dir()
    if lib_dir is None:
        raise FileNotFoundError(
            "Could not locate MLIR runtime libraries. Set `FLIR_MLIR_LIB_DIR=/path/to/mlir/lib` "
            "or `MLIR_PATH=/path/to/mlir/build`."
        )

    rocm_rt = lib_dir / "libmlir_rocm_runtime.so"
    runner = lib_dir / "libmlir_runner_utils.so"
    if not rocm_rt.exists() or not runner.exists():
        raise FileNotFoundError(
            f"Missing MLIR runtime libs in {lib_dir}. Expected "
            f"`{rocm_rt.name}` and `{runner.name}`."
        )

    return SharedLibs(str(rocm_rt), str(runner))


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
        #
        # Therefore we must always call it with a single pointer to an array of
        # pointers-to-arguments (each entry points to host memory containing the
        # value to be passed).
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


