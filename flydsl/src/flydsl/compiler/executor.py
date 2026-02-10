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
        spec = importlib.util.find_spec("flydsl._mlir._mlir_libs")
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
        from flydsl._mlir._mlir_libs._mlirExecutionEngine import ExecutionEngine  # type: ignore

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
                # Fallback: try to expand tensor-like args into flattened ranked memref ABI.
                def _is_tensor_like(x) -> bool:
                    return (
                        hasattr(x, "data_ptr")
                        and callable(getattr(x, "data_ptr"))
                        and hasattr(x, "numel")
                        and callable(getattr(x, "numel"))
                        and hasattr(x, "is_contiguous")
                        and callable(getattr(x, "is_contiguous"))
                    )

                def _tensor_rank(x) -> int:
                    if hasattr(x, "dim") and callable(getattr(x, "dim")):
                        try:
                            return int(x.dim())
                        except Exception:
                            return 1
                    if hasattr(x, "shape"):
                        try:
                            return int(len(x.shape))
                        except Exception:
                            return 1
                    return 1

                def _tensor_shape(x):
                    try:
                        return tuple(int(d) for d in x.shape)
                    except Exception:
                        return None

                def _tensor_strides(x):
                    if hasattr(x, "stride") and callable(getattr(x, "stride")):
                        try:
                            return tuple(int(s) for s in x.stride())
                        except Exception:
                            return None
                    return None

                def _try_expand_flattened_memrefs(user_args, sig_tys):
                    out = []
                    i = 0  # sig index
                    j = 0  # user arg index
                    while i < len(sig_tys) and j < len(user_args):
                        # Detect start of a flattened memref descriptor.
                        if (
                            _is_tensor_like(user_args[j])
                            and i + 2 < len(sig_tys)
                            and sig_tys[i].strip() == "!llvm.ptr"
                            and sig_tys[i + 1].strip() == "!llvm.ptr"
                            and sig_tys[i + 2].strip() == "i64"
                        ):
                            # How many consecutive i64s after the first 3?
                            k = i + 3
                            num_i64 = 0
                            while k < len(sig_tys) and sig_tys[k].strip() == "i64":
                                num_i64 += 1
                                k += 1
                            # Need an even number: sizes+strides.
                            if num_i64 < 2 or (num_i64 % 2) != 0:
                                return None
                            max_rank = num_i64 // 2
                            r = _tensor_rank(user_args[j])
                            if r < 1 or r > max_rank:
                                # Fall back to rank-1 if available.
                                if max_rank >= 1:
                                    r = 1
                                else:
                                    return None
                            span = 3 + 2 * r

                            t = user_args[j]
                            if not bool(t.is_contiguous()):
                                raise ValueError("Non-contiguous tensor passed to memref arg; call `.contiguous()` first.")
                            base = int(t.data_ptr())
                            out.append(ctypes.c_void_p(base))  # allocated
                            out.append(ctypes.c_void_p(base))  # aligned
                            out.append(int(0))                 # offset (elements)

                            if r == 1:
                                out.append(int(t.numel()))
                                out.append(int(1))
                            else:
                                shape = _tensor_shape(t)
                                strides = _tensor_strides(t)
                                if shape is None or strides is None or len(shape) < r or len(strides) < r:
                                    return None
                                # Use the last r dims/strides (supports passing a flattened view tensor too).
                                shape_r = tuple(int(d) for d in shape[-r:])
                                strides_r = tuple(int(s) for s in strides[-r:])
                                out.extend(list(shape_r))
                                out.extend(list(strides_r))

                            i += span
                            j += 1
                            continue

                        # Default: 1:1 mapping (scalar or pointer)
                        out.append(user_args[j])
                        i += 1
                        j += 1

                    if i == len(sig_tys) and j == len(user_args):
                        return out
                    return None

                expanded = _try_expand_flattened_memrefs(args, llvm_arg_tys)
                if expanded is None:
                    raise TypeError(f"{name} expects {len(llvm_arg_tys)} args, got {len(args)}")
                args = tuple(expanded)

            owned = []  # keep ctypes temporaries alive for the duration of the call
            arg_ptrs = []

            # If we're calling a ciface wrapper and the raw function uses a flattened
            # memref ABI, then the ciface `!llvm.ptr` arguments are *pointers to
            # memref descriptors*, not raw data pointers.
            #
            # We infer each memref rank from the raw signature and build a matching
            # descriptor from the torch tensor argument.
            is_ciface = sig_name.startswith("_mlir_ciface_")
            raw_sig = self._llvm_sigs.get(name, [])
            ciface_sig = self._llvm_sigs.get(sig_name, [])
            ciface_uses_desc_ptrs = bool(is_ciface and raw_sig and ciface_sig and len(raw_sig) > len(ciface_sig))
            memref_ranks = []
            if ciface_uses_desc_ptrs:
                # Number of scalar (non-ptr) args in ciface.
                scalar_count = sum(1 for t in ciface_sig if t.strip() != "!llvm.ptr")
                # Number of memref descriptor pointers in ciface.
                memref_count = sum(1 for t in ciface_sig if t.strip() == "!llvm.ptr")
                idx = 0
                for mi in range(memref_count):
                    if idx + 2 >= len(raw_sig) or raw_sig[idx].strip() != "!llvm.ptr" or raw_sig[idx + 1].strip() != "!llvm.ptr" or raw_sig[idx + 2].strip() != "i64":
                        # Best-effort fallback.
                        memref_ranks = []
                        break
                    if mi < memref_count - 1:
                        # Find next memref descriptor start.
                        nxt = idx + 3
                        while nxt + 2 < len(raw_sig):
                            if raw_sig[nxt].strip() == "!llvm.ptr" and raw_sig[nxt + 1].strip() == "!llvm.ptr" and raw_sig[nxt + 2].strip() == "i64":
                                break
                            nxt += 1
                        d = nxt - (idx + 3)
                        memref_ranks.append(max(1, d // 2) if d >= 2 and d % 2 == 0 else 1)
                        idx = nxt
                    else:
                        # Last memref: remaining (excluding trailing scalars) encodes sizes+strides.
                        d = (len(raw_sig) - scalar_count) - (idx + 3)
                        memref_ranks.append(max(1, d // 2) if d >= 2 and d % 2 == 0 else 1)

            def _make_memref_desc_type(rank: int):
                class _MemRefDesc(ctypes.Structure):
                    _fields_ = [
                        ("allocated", ctypes.c_void_p),
                        ("aligned", ctypes.c_void_p),
                        ("offset", ctypes.c_int64),
                        ("sizes", ctypes.c_int64 * rank),
                        ("strides", ctypes.c_int64 * rank),
                    ]
                return _MemRefDesc

            memref_i = 0
            for a, ty in zip(args, llvm_arg_tys):
                ty = ty.strip()
                if ty == "!llvm.ptr":
                    # Tensor-like: any object with a `.data_ptr()` method returning an int.
                    if hasattr(a, "data_ptr") and callable(getattr(a, "data_ptr")):
                        if ciface_uses_desc_ptrs and memref_ranks and memref_i < len(memref_ranks):
                            r = int(memref_ranks[memref_i])
                            memref_i += 1
                            if hasattr(a, "is_contiguous") and callable(getattr(a, "is_contiguous")) and not bool(a.is_contiguous()):
                                raise ValueError("Non-contiguous tensor passed to memref argument; call `.contiguous()` first.")
                            base = int(a.data_ptr())
                            # Shape/stride in elements.
                            shape = tuple(int(d) for d in getattr(a, "shape", (int(a.numel()),)))
                            stride = tuple(int(s) for s in a.stride()) if hasattr(a, "stride") and callable(getattr(a, "stride")) else (1,)
                            if r == 1:
                                sizes = (int(a.numel()),)
                                strides = (1,)
                            else:
                                # Use the last r dims (row-major torch contiguous expected).
                                sizes = shape[-r:]
                                strides = stride[-r:]
                            DescT = _make_memref_desc_type(r)
                            desc = DescT()
                            desc.allocated = ctypes.c_void_p(base)
                            desc.aligned = ctypes.c_void_p(base)
                            desc.offset = ctypes.c_int64(0)
                            for ii in range(r):
                                desc.sizes[ii] = int(sizes[ii])
                                desc.strides[ii] = int(strides[ii])
                            owned.append(desc)
                            # IMPORTANT: packed-call expects a pointer to the *argument value*.
                            # For `!llvm.ptr` args, the argument value is itself a pointer,
                            # so we must pass a pointer-to-(c_void_p) holding `&desc`, not
                            # a pointer-to-struct.
                            desc_ptr = ctypes.c_void_p(ctypes.addressof(desc))
                            owned.append(desc_ptr)
                            arg_ptrs.append(ctypes.cast(ctypes.pointer(desc_ptr), ctypes.c_void_p))
                            continue

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
