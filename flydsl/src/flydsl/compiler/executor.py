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
            "  - Build with `./flir/build.sh` and use the embedded package root on PYTHONPATH, or\n"
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
        # Memref descriptor (LLVM dialect prints a concrete struct type when bare-pointers are disabled).
        # Example:
        #   !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        if ty.startswith("!llvm.struct<"):
            return _ctype_for_llvm_struct(ty)
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
                    # Struct args: allow passing torch tensors / tensor-like objects, which we pack
                    # into a StridedMemRef descriptor, and also allow passing pre-built ctypes structs.
                    cty = self._ctype_for_llvm_type(ty)
                    if issubclass(cty, ctypes.Structure):
                        if isinstance(a, cty):
                            v = a
                        elif hasattr(a, "data_ptr") and callable(getattr(a, "data_ptr")) and hasattr(a, "shape"):
                            v = _pack_tensor_as_memref_descriptor(a, cty)
                        else:
                            raise TypeError(f"Unsupported struct arg type: {type(a)} for {ty}")
                    else:
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


# ---- Minimal LLVM-dialect struct parsing for memref descriptors ----
_STRUCT_CACHE: Dict[str, type] = {}


def _split_top_level_commas(s: str) -> List[str]:
    out: List[str] = []
    cur = []
    depth_angle = 0
    depth_paren = 0
    for ch in s:
        if ch == "<":
            depth_angle += 1
        elif ch == ">":
            depth_angle -= 1
        elif ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        if ch == "," and depth_angle == 0 and depth_paren == 0:
            out.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if cur:
        out.append("".join(cur).strip())
    return [x for x in out if x]


def _parse_array_ty(ty: str) -> Tuple[int, str]:
    # array<k x i64>
    m = re.match(r"array<\s*(\d+)\s*x\s*([a-z0-9]+)\s*>", ty.strip())
    if not m:
        raise ValueError(f"Unsupported array type: {ty}")
    return int(m.group(1)), m.group(2)


def _ctype_for_llvm_struct(ty: str):
    ty = ty.strip()
    c = _STRUCT_CACHE.get(ty)
    if c is not None:
        return c
    # Expect: !llvm.struct<( ... )>
    inner = ty
    inner = inner[len("!llvm.struct<") :].rstrip(">")
    inner = inner.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1].strip()
    fields = _split_top_level_commas(inner)

    c_fields = []
    for i, fty in enumerate(fields):
        fty = fty.strip()
        if fty in {"ptr", "!llvm.ptr"}:
            c_fields.append((f"f{i}", ctypes.c_void_p))
        elif fty in {"i64"}:
            c_fields.append((f"f{i}", ctypes.c_int64))
        elif fty.startswith("array<"):
            n, et = _parse_array_ty(fty)
            if et != "i64":
                raise ValueError(f"Unsupported array element type: {fty}")
            c_fields.append((f"f{i}", ctypes.c_int64 * n))
        else:
            # best-effort: treat unknowns as void*
            c_fields.append((f"f{i}", ctypes.c_void_p))

    cls_name = f"LLVMStruct_{abs(hash(ty))}"
    cls = type(cls_name, (ctypes.Structure,), {"_fields_": c_fields})
    _STRUCT_CACHE[ty] = cls
    return cls


def _pack_tensor_as_memref_descriptor(t, cty_struct: type) -> ctypes.Structure:
    """Pack a tensor-like object into a StridedMemRef descriptor.

    Supports the canonical MLIR ABI layout:
      (ptr basePtr, ptr dataPtr, i64 offset, array<rank x i64> sizes, array<rank x i64> strides)
    """
    # Rank inferred from the struct field types.
    f = getattr(cty_struct, "_fields_", [])
    if len(f) != 5:
        raise TypeError("Only StridedMemRef descriptors are supported (5-field struct)")
    sizes_ty = f[3][1]
    strides_ty = f[4][1]
    if not (hasattr(sizes_ty, "_length_") and hasattr(strides_ty, "_length_")):
        raise TypeError("Invalid memref descriptor struct (missing sizes/strides arrays)")
    rank = int(sizes_ty._length_)

    # Tensor metadata
    shape = tuple(int(x) for x in getattr(t, "shape"))
    if len(shape) != rank:
        raise TypeError(f"Tensor rank {len(shape)} does not match descriptor rank {rank}")

    # Strides are in elements.
    strides = tuple(int(x) for x in getattr(t, "stride")()) if callable(getattr(t, "stride", None)) else None
    if strides is None:
        raise TypeError("Tensor-like object must provide stride()")

    # Use storage base ptr + storage_offset if available, else fall back to data_ptr.
    base_ptr = None
    offset_elems = 0
    try:
        if hasattr(t, "untyped_storage"):
            base_ptr = int(t.untyped_storage().data_ptr())
        elif hasattr(t, "storage"):
            base_ptr = int(t.storage().data_ptr())
        offset_elems = int(getattr(t, "storage_offset", lambda: 0)())
    except Exception:
        base_ptr = None
        offset_elems = 0

    data_ptr = int(t.data_ptr())
    if base_ptr is None:
        base_ptr = data_ptr
        offset_elems = 0

    desc = cty_struct()
    desc.f0 = ctypes.c_void_p(int(base_ptr))
    desc.f1 = ctypes.c_void_p(int(data_ptr))
    desc.f2 = ctypes.c_int64(int(offset_elems))
    for i in range(rank):
        desc.f3[i] = ctypes.c_int64(int(shape[i]))
        desc.f4[i] = ctypes.c_int64(int(strides[i]))
    return desc


