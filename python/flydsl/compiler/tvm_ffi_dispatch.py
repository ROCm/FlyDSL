# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""TVM FFI dispatch path for fast kernel launch.

Generates an LLVM IR wrapper function that bridges TVM FFI's packed calling
convention to the host function's typed parameters.  The wrapper is injected
into the compiled MLIR module before JIT compilation, so it's compiled and
linked together with the kernel — no separate C++ compilation step.

The hot path becomes a single ``tvm_ffi.Function.__call__(*args)`` which
goes through Cython directly to the LLVM IR wrapper, bypassing all Python
argument marshalling.

Safety: a lightweight guard check validates compile-affecting parameters
(constexpr values, tensor dtypes, type params) on every call.  If the
guard fails, ``prepare_and_call`` returns ``(False, None)`` and the caller
falls back to the full compilation path.
"""

import inspect
from typing import List, Optional, Tuple

from ..utils import log

# Slot kinds — mirror the arg types we handle
_KIND_TENSOR = "tensor"
_KIND_INT32 = "int32"
_KIND_INT64 = "int64"
_KIND_STREAM = "stream"
_KIND_VOID_PTR = "void_ptr"  # auto-stream (NULL)


def _get_dtype(arg):
    """Extract dtype from a tensor-like object."""
    return arg.dtype


def _slot_specs_from_sig(sig, args_tuple):
    """Build slot specs from a function signature and argument tuple.

    Returns a list of (arg_idx, kind) tuples, or None if any parameter
    can't be fast-pathed.

    arg_idx: index into the *original* args_tuple (-1 for auto-stream)
    kind: one of _KIND_TENSOR, _KIND_INT32, _KIND_INT64, _KIND_STREAM, _KIND_VOID_PTR
    """
    import ctypes

    from .jit_argument import _is_constexpr_annotation, _is_type_param_annotation

    slot_specs = []
    has_user_stream = False

    for i, (param_name, param) in enumerate(sig.parameters.items()):
        annotation = param.annotation

        if annotation is not inspect.Parameter.empty and _is_constexpr_annotation(annotation):
            continue
        if annotation is not inspect.Parameter.empty and _is_type_param_annotation(annotation):
            continue

        if getattr(annotation, '_is_stream_param', False):
            has_user_stream = True
            slot_specs.append((i, _KIND_STREAM))
            continue

        arg = args_tuple[i]

        if hasattr(arg, 'data_ptr'):
            slot_specs.append((i, _KIND_TENSOR))
        elif isinstance(arg, int):
            # Check the ctype from the existing slot spec protocol
            from .jit_function import _resolve_jit_arg_type
            jit_arg_type = _resolve_jit_arg_type(arg, annotation)
            if jit_arg_type is not None and hasattr(jit_arg_type, '_reusable_slot_spec'):
                spec = jit_arg_type._reusable_slot_spec(arg)
                if spec is not None:
                    ctype, _ = spec
                    if ctype == ctypes.c_int or ctype == ctypes.c_int32:
                        slot_specs.append((i, _KIND_INT32))
                    else:
                        slot_specs.append((i, _KIND_INT64))
                    continue
            slot_specs.append((i, _KIND_INT32))
        else:
            return None  # unsupported arg type

    if not has_user_stream:
        slot_specs.append((-1, _KIND_VOID_PTR))

    return slot_specs


# ---------------------------------------------------------------------------
# LLVM IR wrapper generation
# ---------------------------------------------------------------------------

def _llvm_param_type(kind):
    """Map a slot kind to its MLIR LLVM dialect type string."""
    if kind == _KIND_TENSOR:
        return "!llvm.ptr<1>"
    elif kind == _KIND_INT32:
        return "i32"
    elif kind == _KIND_INT64:
        return "i64"
    elif kind in (_KIND_STREAM, _KIND_VOID_PTR):
        return "!llvm.ptr"
    raise ValueError(f"Unknown kind: {kind}")


def generate_tvm_ffi_wrapper_ir(func_name: str, slot_specs: List[Tuple[int, str]]) -> str:
    """Generate MLIR LLVM dialect text for the TVM FFI wrapper function.

    The wrapper function signature matches TVM FFI's ``Invoke`` layout::

        void __tvm_ffi_<name>(ptr handle, ptr args, i32 num_args,
                              ptr result, i32 ret_code)

    The MLIR JIT engine's ``packFunctionArguments`` automatically creates a
    ``_mlir___tvm_ffi_<name>(void**)`` wrapper that unpacks these 5 args
    from the packed ``void**`` array — which is exactly the ``mlir_args``
    layout that ``TVMFFIPyMLIRPackedSafeCall::Invoke`` constructs::

        void* mlir_args[] = {&handle, &args, &num_args, &rv, &ret_code};

    The packed wrapper dereferences each ``mlir_args[i]`` to recover the
    typed value, then calls our function.  ``raw_lookup("__tvm_ffi_<name>")``
    finds the packed wrapper via ``lookupPacked`` (which prepends ``_mlir_``).

    TVMFFIAny struct: {i32 type_index, i32 padding, i64 value_union} (16 bytes).
    For tensors, value_union holds a TensorObj*; DLTensor is embedded at offset
    +24 (sizeof(TVMFFIObject)); field 0 of DLTensor is the data ptr.
    For ints, value_union holds i64.

    ret_code: received by value (0 = success) from the packed wrapper.
    Since we don't modify it, ``Invoke`` returns 0 automatically.
    """
    lines = []
    ssa_idx = [0]

    def ssa():
        name = f"%v{ssa_idx[0]}"
        ssa_idx[0] += 1
        return name

    # Collect host function parameter types
    param_types = [_llvm_param_type(kind) for _, kind in slot_specs]

    # 5-arg signature matching TVM FFI Invoke layout:
    #   %handle:   !llvm.ptr  (unused, NULL)
    #   %args:     !llvm.ptr  (TVMFFIAny*)
    #   %num_args: i32        (unused)
    #   %result:   !llvm.ptr  (unused)
    #   %ret_code: i32        (unused, stays 0)
    wrapper_name = f"__tvm_ffi_{func_name}"
    lines.append(
        f"  llvm.func @{wrapper_name}("
        f"%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, "
        f"%arg3: !llvm.ptr, %arg4: i32) {{"
    )

    # %arg1 = TVMFFIAny* args (already dereferenced by packFunctionArguments)

    # --- Extract each argument from the TVMFFIAny array ---
    call_args = []
    tvm_arg_idx = 0  # index into TVM FFI packed args (skips VOID_PTR)

    for _, (arg_idx, kind) in enumerate(slot_specs):
        if kind == _KIND_VOID_PTR:
            # Auto-stream: null pointer
            val = ssa()
            lines.append(f"    {val} = llvm.mlir.zero : !llvm.ptr")
            call_args.append(val)
            continue

        # Byte offset to value_union of TVMFFIAny[tvm_arg_idx]:
        #   Each TVMFFIAny = {i32, i32, i64} = 16 bytes
        #   value_union is at offset 8 from struct start
        byte_offset = tvm_arg_idx * 16 + 8

        off = ssa()
        lines.append(f"    {off} = llvm.mlir.constant({byte_offset} : i64) : i64")

        value_ptr = ssa()
        lines.append(
            f"    {value_ptr} = llvm.getelementptr %arg1[{off}]"
            f" : (!llvm.ptr, i64) -> !llvm.ptr, i8"
        )

        if kind == _KIND_TENSOR:
            # Load TensorObj* from value_union
            tensor_obj_ptr = ssa()
            lines.append(f"    {tensor_obj_ptr} = llvm.load {value_ptr} : !llvm.ptr -> !llvm.ptr")

            # GEP +24 bytes to skip TVMFFIObject header → DLTensor
            # sizeof(TVMFFIObject) = 24: {i64 combined_ref, i32 type_index, i32 pad, ptr deleter/align}
            obj_offset = ssa()
            lines.append(f"    {obj_offset} = llvm.mlir.constant(24 : i64) : i64")
            dltensor_ptr = ssa()
            lines.append(
                f"    {dltensor_ptr} = llvm.getelementptr {tensor_obj_ptr}[{obj_offset}]"
                f" : (!llvm.ptr, i64) -> !llvm.ptr, i8"
            )

            # Load data pointer from DLTensor (field 0, offset 0)
            data_ptr = ssa()
            lines.append(f"    {data_ptr} = llvm.load {dltensor_ptr} : !llvm.ptr -> !llvm.ptr")

            # Addrspacecast to address space 1 (GPU global memory)
            gpu_ptr = ssa()
            lines.append(f"    {gpu_ptr} = llvm.addrspacecast {data_ptr} : !llvm.ptr to !llvm.ptr<1>")
            call_args.append(gpu_ptr)

        elif kind == _KIND_INT32:
            # Load i64 from value_union, truncate to i32
            i64_val = ssa()
            lines.append(f"    {i64_val} = llvm.load {value_ptr} : !llvm.ptr -> i64")

            i32_val = ssa()
            lines.append(f"    {i32_val} = llvm.trunc {i64_val} : i64 to i32")
            call_args.append(i32_val)

        elif kind == _KIND_INT64:
            # Load i64 directly
            i64_val = ssa()
            lines.append(f"    {i64_val} = llvm.load {value_ptr} : !llvm.ptr -> i64")
            call_args.append(i64_val)

        elif kind == _KIND_STREAM:
            # Load i64 from value_union, convert to pointer
            i64_val = ssa()
            lines.append(f"    {i64_val} = llvm.load {value_ptr} : !llvm.ptr -> i64")

            ptr_val = ssa()
            lines.append(f"    {ptr_val} = llvm.inttoptr {i64_val} : i64 to !llvm.ptr")
            call_args.append(ptr_val)

        tvm_arg_idx += 1

    # --- Call the host function ---
    call_args_str = ", ".join(call_args)
    param_types_str = ", ".join(param_types)
    lines.append(f"    llvm.call @{func_name}({call_args_str}) : ({param_types_str}) -> ()")

    # ret_code stays 0 (success) — no need to write it back since
    # packFunctionArguments passes it by value, and TVM FFI's Invoke
    # reads it from the original stack variable which stays 0.

    lines.append("    llvm.return")
    lines.append("  }")

    return "\n".join(lines)


def inject_tvm_ffi_wrapper(ir_text: str, func_name: str, slot_specs: List[Tuple[int, str]]) -> str:
    """Insert the TVM FFI wrapper function into compiled module IR text.

    Finds the module's closing ``}`` and inserts the wrapper just before it.
    """
    wrapper = generate_tvm_ffi_wrapper_ir(func_name, slot_specs)
    idx = ir_text.rfind("}")
    if idx < 0:
        raise ValueError("Cannot find module closing '}' in IR text")
    return ir_text[:idx] + wrapper + "\n" + ir_text[idx:]


# ---------------------------------------------------------------------------
# TVMFFIDispatcher
# ---------------------------------------------------------------------------

class TVMFFIDispatcher:
    """Dispatches kernel calls through a TVM FFI function wrapping an LLVM IR
    wrapper that was compiled together with the kernel.

    Safety invariant: ``prepare_and_call`` validates a lightweight guard
    before dispatching.  The guard captures all compile-affecting parameters
    (constexpr values, tensor dtypes, type params) at build time and
    compares them on each call.
    """

    __slots__ = ('_slot_specs', '_dispatch_fn', '_sig',
                 '_arg_indices', '_arg_extractors',
                 '_n_params', '_defaults', '_kwarg_map', '_guards')

    def __init__(self, dispatch_fn, slot_specs: List[Tuple[int, str]],
                 sig: inspect.Signature, args_tuple: tuple):
        self._dispatch_fn = dispatch_fn
        self._slot_specs = slot_specs
        self._sig = sig

        # Pre-compute parameter info for the fast path (skip sig.bind)
        params = list(sig.parameters.values())
        self._n_params = len(params)
        defaults = []
        kwarg_map = {}
        for i, p in enumerate(params):
            if p.default is not inspect.Parameter.empty:
                defaults.append(p.default)
            else:
                defaults.append(None)
            kwarg_map[p.name] = i
        self._defaults = tuple(defaults)
        self._kwarg_map = kwarg_map

        # Build extractors for preparing call args from the original args_tuple
        self._arg_indices = []
        self._arg_extractors = []
        for packed_idx, (arg_idx, kind) in enumerate(slot_specs):
            if kind == _KIND_VOID_PTR:
                # auto-stream: not a user arg
                continue
            self._arg_indices.append(arg_idx)
            if kind == _KIND_TENSOR:
                self._arg_extractors.append(None)  # pass tensor directly
            elif kind in (_KIND_INT32, _KIND_INT64):
                self._arg_extractors.append(_extract_int_value)
            elif kind == _KIND_STREAM:
                self._arg_extractors.append(_extract_stream_value)

        # Build guards: capture compile-affecting parameters.
        from .jit_argument import _is_constexpr_annotation, _is_type_param_annotation

        guards = []
        for i, (param_name, param) in enumerate(sig.parameters.items()):
            ann = param.annotation
            if ann is not inspect.Parameter.empty and _is_constexpr_annotation(ann):
                guards.append((i, None, args_tuple[i]))
            elif ann is not inspect.Parameter.empty and _is_type_param_annotation(ann):
                guards.append((i, None, args_tuple[i]))
            elif i < len(args_tuple) and hasattr(args_tuple[i], 'dtype'):
                guards.append((i, _get_dtype, args_tuple[i].dtype))
        self._guards = tuple(guards)

    def _check_guard(self, args_tuple):
        """Validate that args match the compiled kernel's compile-time assumptions."""
        for idx, extractor, expected in self._guards:
            if idx >= len(args_tuple):
                return False
            arg = args_tuple[idx]
            actual = extractor(arg) if extractor is not None else arg
            if actual != expected:
                return False
        return True

    def prepare_and_call(self, args, kwargs):
        """Assemble args from raw Python args/kwargs, validate guard, and dispatch.

        Returns ``(True, result)`` on success.
        Returns ``(False, None)`` if the guard check fails.
        """
        n_args = len(args)
        if not kwargs:
            if n_args < self._n_params:
                args_tuple = args + self._defaults[n_args:]
            else:
                args_tuple = args
        else:
            vals = list(args) + list(self._defaults[n_args:])
            for name, val in kwargs.items():
                vals[self._kwarg_map[name]] = val
            args_tuple = tuple(vals)

        if not self._check_guard(args_tuple):
            return False, None

        return True, self._dispatch(args_tuple)

    def _dispatch(self, args_tuple):
        """Internal: build call args and invoke TVM FFI wrapper."""
        call_args = []
        for arg_idx, extractor in zip(self._arg_indices, self._arg_extractors):
            arg = args_tuple[arg_idx]
            if extractor is None:
                call_args.append(arg)
            else:
                call_args.append(extractor(arg))

        return self._dispatch_fn(*call_args)

    def __call__(self, args_tuple):
        """Direct dispatch without guard check."""
        return self._dispatch(args_tuple)


def _extract_int_value(arg):
    """Extract int value, handling both raw ints and DSL numeric types."""
    if hasattr(arg, 'value'):
        return int(arg.value)
    return int(arg)


def _extract_stream_value(arg):
    """Extract stream pointer as int."""
    from ..expr.typing import Stream
    if isinstance(arg, Stream):
        raw = arg.value
    else:
        raw = arg
    if raw is None:
        return 0
    elif isinstance(raw, int):
        return raw
    return raw.cuda_stream


def build_tvm_ffi_dispatcher(sig, args_tuple, compiled_func) -> Optional[TVMFFIDispatcher]:
    """Build a TVMFFIDispatcher using the __tvm_ffi_<name> symbol from the engine.

    Returns None if the args can't be fast-pathed or tvm_ffi is unavailable.
    """
    try:
        import tvm_ffi  # noqa: F401
    except ImportError:
        log().debug("[TVMFFIDispatcher] tvm_ffi not available, skipping")
        return None

    slot_specs = _slot_specs_from_sig(sig, args_tuple)
    if slot_specs is None:
        return None

    try:
        compiled_func._ensure_engine()
        wrapper_sym = "__tvm_ffi_" + compiled_func._entry
        ffi_ptr = compiled_func._engine.raw_lookup(wrapper_sym)
        if not ffi_ptr:
            log().debug(f"[TVMFFIDispatcher] {wrapper_sym} not found in module")
            return None
        dispatch_fn = tvm_ffi.Function.__from_mlir_packed_safe_call__(
            ffi_ptr, keep_alive_object=compiled_func._engine
        )
        return TVMFFIDispatcher(dispatch_fn, slot_specs, sig, args_tuple)
    except Exception as e:
        log().debug(f"[TVMFFIDispatcher] build failed: {e}")
        return None
