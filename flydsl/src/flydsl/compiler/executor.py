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

    # Class-level caches for ctypes (shared across all instances)
    _CTYPE_MAP = {
        "!llvm.ptr": ctypes.c_void_p,
        "i1": ctypes.c_bool,
        "i8": ctypes.c_int8,
        "i16": ctypes.c_int16,
        "i32": ctypes.c_int32,
        "i64": ctypes.c_int64,
        "f32": ctypes.c_float,
        "f64": ctypes.c_double,
    }
    _MEMREF_DESC_CACHE: Dict[int, type] = {}  # rank -> DescType

    @classmethod
    def _get_memref_desc_type(cls, rank: int):
        """Get or create cached memref descriptor type for given rank."""
        if rank not in cls._MEMREF_DESC_CACHE:
            class _MemRefDesc(ctypes.Structure):
                _fields_ = [
                    ("allocated", ctypes.c_void_p),
                    ("aligned", ctypes.c_void_p),
                    ("offset", ctypes.c_int64),
                    ("sizes", ctypes.c_int64 * rank),
                    ("strides", ctypes.c_int64 * rank),
                ]
            cls._MEMREF_DESC_CACHE[rank] = _MemRefDesc
        return cls._MEMREF_DESC_CACHE[rank]

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
        # Instance-level cache for wrapper functions
        self._wrapper_cache: Dict[str, callable] = {}

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

    @classmethod
    def _ctype_for_llvm_type(cls, ty: str):
        """Convert LLVM type string to ctypes type (uses cached mapping)."""
        ty = ty.strip()
        return cls._CTYPE_MAP.get(ty, ctypes.c_void_p)

    def __getattr__(self, name: str):
        # Check wrapper cache first for fast path
        if name in self._wrapper_cache:
            return self._wrapper_cache[name]

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

        # Packed-call wrapper: void(void**) - cache the CFUNCTYPE result
        func_exe = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(func_ptr)

        # Pre-compute signature info for this function (avoid repeated lookups)
        llvm_arg_tys = self._llvm_sigs.get(sig_name) or self._llvm_sigs.get(name) or []
        is_ciface = sig_name.startswith("_mlir_ciface_")
        raw_sig = self._llvm_sigs.get(name, [])
        ciface_sig = self._llvm_sigs.get(sig_name, [])
        ciface_uses_desc_ptrs = bool(is_ciface and raw_sig and ciface_sig and len(raw_sig) > len(ciface_sig))

        # Pre-compute memref ranks if using descriptor pointers
        memref_ranks = []
        if ciface_uses_desc_ptrs:
            scalar_count = sum(1 for t in ciface_sig if t.strip() != "!llvm.ptr")
            memref_count = sum(1 for t in ciface_sig if t.strip() == "!llvm.ptr")
            idx = 0
            for mi in range(memref_count):
                if idx + 2 >= len(raw_sig) or raw_sig[idx].strip() != "!llvm.ptr" or raw_sig[idx + 1].strip() != "!llvm.ptr" or raw_sig[idx + 2].strip() != "i64":
                    memref_ranks = []
                    break
                if mi < memref_count - 1:
                    nxt = idx + 3
                    while nxt + 2 < len(raw_sig):
                        if raw_sig[nxt].strip() == "!llvm.ptr" and raw_sig[nxt + 1].strip() == "!llvm.ptr" and raw_sig[nxt + 2].strip() == "i64":
                            break
                        nxt += 1
                    d = nxt - (idx + 3)
                    memref_ranks.append(max(1, d // 2) if d >= 2 and d % 2 == 0 else 1)
                    idx = nxt
                else:
                    d = (len(raw_sig) - scalar_count) - (idx + 3)
                    memref_ranks.append(max(1, d // 2) if d >= 2 and d % 2 == 0 else 1)

        # Pre-compute ctypes for each argument type
        arg_ctypes = [self._ctype_for_llvm_type(ty) for ty in llvm_arg_tys]
        
        # Pre-strip type strings to avoid repeated strip() calls
        llvm_arg_tys_stripped = [ty.strip() for ty in llvm_arg_tys]
        
        # Pre-compute which args are pointers vs scalars (static info from signature)
        arg_is_ptr = [ty == "!llvm.ptr" for ty in llvm_arg_tys_stripped]

        # Pre-allocate arrays for common case (reused across calls)
        num_args = len(llvm_arg_tys)
        
        # Pre-allocate c_args array type for this function
        c_args_type = ctypes.c_void_p * num_args if num_args > 0 else ctypes.c_void_p * 1
        
        # Capture executor reference for use in wrapper
        executor_self = self
        
        # Pre-fetch memref descriptor types for known ranks
        memref_desc_types = {}
        if memref_ranks:
            for r in memref_ranks:
                memref_desc_types[r] = executor_self._get_memref_desc_type(r)
        
        # Cache c_void_p type for fast checks
        c_void_p_type = ctypes.c_void_p

        def wrapper(*args):
            # Use pre-computed llvm_arg_tys from closure
            if len(args) != num_args:
                # Best-effort for 0-arg functions when signature couldn't be parsed.
                if len(args) == 0 and num_args == 0:
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
                            and sig_tys[i] == "!llvm.ptr"
                            and sig_tys[i + 1] == "!llvm.ptr"
                            and sig_tys[i + 2] == "i64"
                        ):
                            # How many consecutive i64s after the first 3?
                            k = i + 3
                            num_i64 = 0
                            while k < len(sig_tys) and sig_tys[k] == "i64":
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

                expanded = _try_expand_flattened_memrefs(args, llvm_arg_tys_stripped)
                if expanded is None:
                    raise TypeError(f"{name} expects {len(llvm_arg_tys)} args, got {len(args)}")
                args = tuple(expanded)

            # Pre-allocate arrays with known size
            owned = [None] * (num_args * 3)  # Max: desc + desc_ptr + v per arg
            owned_idx = 0
            arg_ptrs = [None] * num_args
            arg_ptr_idx = 0

            memref_i = 0
            for i in range(num_args):
                a = args[i]
                
                if arg_is_ptr[i]:
                    # Pointer argument - use fast type checks
                    a_type = type(a)
                    if a_type is int:
                        v = c_void_p_type(a)
                    elif a_type is c_void_p_type:
                        v = a
                    elif hasattr(a, "data_ptr"):
                        # Tensor-like object
                        if ciface_uses_desc_ptrs and memref_ranks and memref_i < len(memref_ranks):
                            r = memref_ranks[memref_i]
                            memref_i += 1
                            base = int(a.data_ptr())
                            DescT = memref_desc_types.get(r) or executor_self._get_memref_desc_type(r)
                            desc = DescT()
                            desc.allocated = c_void_p_type(base)
                            desc.aligned = c_void_p_type(base)
                            desc.offset = ctypes.c_int64(0)
                            if r == 1:
                                desc.sizes[0] = int(a.numel())
                                desc.strides[0] = 1
                            else:
                                shape = a.shape
                                stride = a.stride()
                                for ii in range(r):
                                    desc.sizes[ii] = int(shape[-r + ii])
                                    desc.strides[ii] = int(stride[-r + ii])
                            owned[owned_idx] = desc
                            owned_idx += 1
                            desc_ptr = c_void_p_type(ctypes.addressof(desc))
                            owned[owned_idx] = desc_ptr
                            owned_idx += 1
                            arg_ptrs[arg_ptr_idx] = ctypes.cast(ctypes.pointer(desc_ptr), c_void_p_type)
                            arg_ptr_idx += 1
                            continue
                        v = c_void_p_type(int(a.data_ptr()))
                    else:
                        raise TypeError(f"Unsupported pointer arg type: {type(a)}")
                else:
                    # Scalar argument - use fast type checks
                    cty = arg_ctypes[i]
                    a_type = type(a)
                    if a_type is int:
                        v = cty(a)
                    elif a_type is float:
                        v = cty(a)
                    elif a_type is bool:
                        v = cty(a)
                    else:
                        raise TypeError(f"Unsupported scalar arg type: {a_type} for {llvm_arg_tys_stripped[i]}")

                owned[owned_idx] = v
                owned_idx += 1
                arg_ptrs[arg_ptr_idx] = ctypes.cast(ctypes.pointer(v), c_void_p_type)
                arg_ptr_idx += 1

            c_args = c_args_type(*arg_ptrs[:arg_ptr_idx])
            return func_exe(c_args)

        # Cache the wrapper for subsequent calls
        self._wrapper_cache[name] = wrapper
        return wrapper

    def __call__(self, *args):
        return self.__getattr__("__call__")(*args)


Executor = ExecutionEngineExecutor
