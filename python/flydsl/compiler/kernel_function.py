# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import inspect
import threading
from contextlib import contextmanager, nullcontext
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .._mlir import ir
from .._mlir.dialects import arith, gpu
from ..expr.meta import capture_user_location, file_location, tracing_context
from ..expr.numeric import Index, Integer
from ..expr.typing import Constexpr, as_ir_value
from ..expr.utils.arith import fastmath as fastmath_ctx
from .ast_rewriter import ASTRewriter
from .diagnostics import install_excepthook, warn_annotation_value_mismatch, warn_invalid_annotations
from .jit_argument import is_type_param_annotation, resolve_signature
from .mlir_utils import convert_to_mlir_attr
from .protocol import construct_from_ir_values, extract_to_ir_values, get_ir_types

# =============================================================================
# GPU Operation Helpers
# =============================================================================


def create_gpu_module(
    sym_name: str,
    targets: Optional[List[str]] = None,
    *,
    use_explicit_module: bool = False,
    loc=None,
    ip=None,
) -> gpu.GPUModuleOp:
    target_attrs = []
    if targets:
        for t in targets:
            if isinstance(t, str):
                target_attrs.append(ir.Attribute.parse(t))
            else:
                target_attrs.append(t)
    offloading = ir.Attribute.parse("#fly.explicit_module") if use_explicit_module else None
    module_op = gpu.GPUModuleOp(
        sym_name,
        targets=ir.ArrayAttr.get(target_attrs) if target_attrs else None,
        offloadingHandler=offloading,
        loc=loc,
        ip=ip,
    )
    module_op.regions[0].blocks.append()
    return module_op


def get_gpu_module_body(module_op: gpu.GPUModuleOp):
    return module_op.regions[0].blocks[0]


def _validate_known_block_size(value):
    """Validate and normalize *known_block_size* to a list of 3 positive ints.

    Returns ``None`` when *value* is ``None`` (attribute should be omitted).

    Raises:
        TypeError: if *value* is not a sequence of integers.
        ValueError: if the length is not 3 or any element is not positive.
    """
    if value is None:
        return None

    try:
        elems = list(value)
    except TypeError:
        raise TypeError(
            f"known_block_size must be a sequence of 3 positive integers, got {type(value).__name__}"
        ) from None

    if len(elems) != 3:
        raise ValueError(f"known_block_size must have exactly 3 elements (x, y, z), got {len(elems)}")

    for i, v in enumerate(elems):
        v = as_ir_value(v, keep_static=True)
        if not isinstance(v, int):
            raise TypeError(f"known_block_size[{i}] must be an int, got {type(v).__name__}")
        if v <= 0:
            raise ValueError(f"known_block_size[{i}] must be positive, got {v}")
        elems[i] = v

    return elems


def create_gpu_func(
    sym_name: str,
    function_type: ir.TypeAttr,
    *,
    known_block_size=None,
    loc=None,
    ip=None,
) -> gpu.GPUFuncOp:
    return gpu.GPUFuncOp(
        function_type,
        sym_name=sym_name,
        kernel=True,
        known_block_size=known_block_size,
        loc=loc,
        ip=ip,
    )


def _attach_attrs(op, unit_attrs: Optional[List[str]], value_attrs: Optional[Dict[str, Any]]) -> None:
    if unit_attrs:
        unit = ir.UnitAttr.get()
        for name in unit_attrs:
            op.attributes[name] = unit
    if value_attrs:
        for name, value in value_attrs.items():
            if value is None:
                continue
            op.attributes[name] = convert_to_mlir_attr(value)


# =============================================================================
# Location Tracking Utilities
# =============================================================================


def func_def_location(func: Callable, context=None) -> ir.Location:
    """File location of *func*'s ``def`` line (the kernel/jit definition)."""
    try:
        line = inspect.getsourcelines(func)[1]
    except (OSError, TypeError):
        line = 0
    return file_location(inspect.getfile(func), line, 0, context)


# =============================================================================
# Launch Configuration
# =============================================================================

DimValueType = Union[int, ir.Value, Integer]
DimType = Union[int, ir.Value, Integer, Tuple[DimValueType, ...], List[DimValueType]]


def _normalize_dim(dim: DimType) -> Tuple[DimValueType, DimValueType, DimValueType]:
    if isinstance(dim, (int, ir.Value, Integer)):
        return (dim, 1, 1)
    elif len(dim) == 1:
        return (dim[0], 1, 1)
    elif len(dim) == 2:
        return (dim[0], dim[1], 1)
    return (dim[0], dim[1], dim[2])


# =============================================================================
# Compilation Context (per-compilation state)
# =============================================================================


def merge_compile_hints(*layers) -> dict:
    """Shallow-merge hint layers; later non-None values win."""
    merged = {}
    for layer in layers:
        if layer:
            merged.update((key, value) for key, value in layer.items() if value is not None)
    return merged


class CompilationContext:
    """Context for tracking compilation state within a @jit function.

    Manages:
    - GPU module op for kernel definitions
    - Kernel counter for unique naming
    - Location trackers for debugging
    """

    _current = threading.local()

    # Thread-local storage for compile hints (waves_per_eu, maxnreg, etc.)
    _compile_hints = threading.local()

    @classmethod
    @contextmanager
    def compile_hints(cls, hints: dict):
        """Set thread-local hints, shallow-merging nested contexts.

        Usage:
            with CompilationContext.compile_hints({"waves_per_eu": 2}):
                fn(*args, **kwargs)
        """
        prev = getattr(cls._compile_hints, "data", None)
        cls._compile_hints.data = merge_compile_hints(prev, hints)
        try:
            yield
        finally:
            cls._compile_hints.data = prev

    @classmethod
    def get_compile_hints(cls):
        """Get compiler hints for the current thread, or empty dict."""
        return getattr(cls._compile_hints, "data", None) or {}

    def __init__(self):
        self.gpu_module_op = None
        self.kernel_counter = 0
        self._kernel_names: set[str] = set()
        self.stream_arg = None
        self.link_libs: list = []
        self._link_libs_seen: set = set()
        # Callables invoked on each GPU hipModule_t after ExecutionEngine
        # loads it.  Populated by ExternFunction when module_init_fn is set.
        self.post_load_processors: list = []

    @classmethod
    def get_current(cls) -> Optional["CompilationContext"]:
        return getattr(cls._current, "value", None)

    @classmethod
    @contextmanager
    def create(cls):
        prev = getattr(cls._current, "value", None)
        ctx = CompilationContext()
        cls._current.value = ctx
        try:
            yield ctx
        finally:
            cls._current.value = prev

    def add_link_lib(self, path: str) -> None:
        if path in self._link_libs_seen:
            return
        self._link_libs_seen.add(path)
        self.link_libs.append(path)

    def next_kernel_id(self) -> int:
        """Get next unique kernel ID."""
        kid = self.kernel_counter
        self.kernel_counter += 1
        return kid

    def unique_kernel_name(self, base_name: str, kernel_id: int) -> str:
        """Reserve a kernel symbol, suffixing repeated explicit names."""
        name = base_name
        suffix = kernel_id
        while name in self._kernel_names:
            name = f"{base_name}_{suffix}"
            suffix += 1
        self._kernel_names.add(name)
        return name


def effective_fastmath_hint(hints: dict):
    """Resolve the ambient fastmath hint for traced JIT/kernel bodies."""
    if "fastmath" in hints:
        return hints["fastmath"]
    if hints.get("fast_fp_math"):
        return "fast"
    return None


# =============================================================================
# Kernel Launcher
# =============================================================================


class KernelLauncher:
    """Holds a pending kernel call and materializes it on launch().

    Created by calling a @kernel decorated function. Call .launch() to emit
    both the specialized gpu.func and its gpu.launch_func operation.
    """

    def __init__(
        self,
        kernel_function: "KernelFunction",
        ctx: CompilationContext,
        args: Tuple,
        kwargs: Dict[str, Any],
        bound_self: Any = None,
        unit_attrs: Optional[List[str]] = None,
        value_attrs: Optional[Dict[str, Any]] = None,
    ):
        self._kernel_function = kernel_function
        self._ctx = ctx
        self._args = args
        self._kwargs = kwargs
        self._bound_self = bound_self
        self._unit_attrs = list(unit_attrs) if unit_attrs is not None else None
        self._value_attrs = dict(value_attrs) if value_attrs is not None else None
        self._emitted_kernels = {}

    def _kernel_display_name(self) -> str:
        return self._kernel_function._name or self._kernel_function._func.__name__

    def _resolve_known_block_size(self, block_dims: Tuple) -> Optional[List[int]]:
        """Return an explicit block size or infer one from static launch dims."""
        explicit = self._kernel_function._known_block_size
        if explicit is not None:
            return _validate_known_block_size(explicit)

        raw_dims = tuple(as_ir_value(value, keep_static=True) for value in block_dims)
        if all(isinstance(value, int) for value in raw_dims):
            return _validate_known_block_size(raw_dims)
        return None

    def _check_block_vs_known(self, block_dims: Tuple, known_block_size: Optional[List[int]]) -> None:
        """Raise when static launch dimensions contradict an exact block size."""
        if known_block_size is None:
            return

        labels = ("x", "y", "z")
        for i, (launch_val, declared) in enumerate(zip(block_dims, known_block_size)):
            launch_val = as_ir_value(launch_val, keep_static=True)
            if isinstance(launch_val, int) and launch_val != declared:
                raise ValueError(
                    f"launch block {labels[i]}={launch_val} differs from "
                    f"known_block_size {labels[i]}={declared} declared on "
                    f"kernel '{self._kernel_display_name()}'. "
                    f"This produces an internally-inconsistent IR and is "
                    f"undefined behavior on AMDGPU."
                )

    def launch(
        self,
        *,
        grid: DimType = (1, 1, 1),
        block: DimType = (1, 1, 1),
        smem: Optional[Union[int, ir.Value]] = None,
        stream: Optional[ir.Value] = None,
        cluster: Optional[DimType] = None,
        unit_attrs: Optional[List[str]] = None,
        value_attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit gpu.launch_func operation with the given configuration.

        Args:
            grid: Grid dimensions (x, y, z). Can be int, integer Numeric,
                ir.Value, tuple, or list.
            block: Block dimensions (x, y, z). Can be int, integer Numeric,
                ir.Value, tuple, or list. Static integer dimensions
                automatically specialize the emitted kernel's
                ``known_block_size``.
            smem: Dynamic shared memory size in bytes. ``None`` (default)
                auto-infers from ``SharedAllocator.allocated_bytes`` when one
                was used inside the kernel body. Explicit values are allowed
                when they are >= the auto-inferred size.
            stream: CUDA/HIP stream as ir.Value. None means default stream.
            cluster: Cluster dimensions (x, y, z) for workgroup clustering.
                     None means no clustering. Enables MCAST and cluster barriers.
            unit_attrs: Unit attributes to attach to gpu.launch_func.
            value_attrs: Value attributes to attach to gpu.launch_func.
        """
        launch_loc = capture_user_location()
        grid_dims = _normalize_dim(grid)
        block_dims = _normalize_dim(block)
        known_block_size = self._resolve_known_block_size(block_dims)
        self._check_block_vs_known(block_dims, known_block_size)

        specialization_key = tuple(known_block_size) if known_block_size is not None else None
        emitted_kernel = self._emitted_kernels.get(specialization_key)
        if emitted_kernel is None:
            kernel_name, kernel_args, gpu_func_op, smem_bytes = self._kernel_function._emit_kernel(
                self._ctx,
                self._args,
                self._kwargs,
                bound_self=self._bound_self,
                known_block_size=known_block_size,
            )
            _attach_attrs(gpu_func_op, self._unit_attrs, self._value_attrs)
            emitted_kernel = (kernel_name, kernel_args, smem_bytes)
            self._emitted_kernels[specialization_key] = emitted_kernel
        else:
            kernel_name, kernel_args, smem_bytes = emitted_kernel

        if smem is None:
            smem = smem_bytes if smem_bytes is not None else 0
        elif smem_bytes is not None:
            smem_int = None
            try:
                smem_int = int(as_ir_value(smem, keep_static=True))
            except (TypeError, ValueError):
                pass
            if smem_int is not None and smem_int < smem_bytes:
                raise ValueError(
                    f"launch smem={smem_int} is less than the {smem_bytes} bytes "
                    f"allocated by SharedAllocator in kernel '{kernel_name}'"
                )

        kernel_operands = []
        for arg in kernel_args:
            kernel_operands.extend(extract_to_ir_values(arg))

        with launch_loc:
            grid_x = Index(grid_dims[0]).ir_value()
            grid_y = Index(grid_dims[1]).ir_value()
            grid_z = Index(grid_dims[2]).ir_value()
            block_x = Index(block_dims[0]).ir_value()
            block_y = Index(block_dims[1]).ir_value()
            block_z = Index(block_dims[2]).ir_value()

            smem_val = None
            smem_raw = as_ir_value(smem, keep_static=True)
            if isinstance(smem_raw, ir.Value):
                smem_val = smem_raw
            else:
                smem_py = None
                try:
                    smem_py = int(smem_raw)
                except (TypeError, ValueError):
                    smem_py = None
                if smem_py is not None and smem_py > 0:
                    smem_val = arith.constant(ir.IntegerType.get_signless(32), smem_py)

            if stream is not None:
                stream_val = as_ir_value(stream, keep_static=True)
            else:
                ctx = CompilationContext.get_current()
                stream_val = ctx.stream_arg if ctx and ctx.stream_arg else None

            async_deps = [stream_val] if stream_val is not None else None

            cluster_size = None
            if cluster is not None:
                cx, cy, cz = _normalize_dim(cluster)
                cluster_size = (
                    Index(cx).ir_value(),
                    Index(cy).ir_value(),
                    Index(cz).ir_value(),
                )

            launch_kwargs = {
                "async_dependencies": async_deps,
                "dynamic_shared_memory_size": smem_val,
                "loc": launch_loc,
                "ip": None,
            }
            if cluster_size is not None:
                launch_kwargs["cluster_size"] = cluster_size

            launch_op = gpu.LaunchFuncOp(
                ["kernels", kernel_name],
                (grid_x, grid_y, grid_z),
                (block_x, block_y, block_z),
                kernel_operands,
                **launch_kwargs,
            )
            _attach_attrs(launch_op, unit_attrs, value_attrs)


# =============================================================================
# Kernel Function
# =============================================================================


class KernelFunction:
    """Wrapper for @kernel decorated functions.

    Calling it captures the kernel arguments and returns a KernelLauncher.
    The launcher emits a specialized gpu.func together with its launch op once
    the launch configuration is known.
    """

    _current: Optional["KernelFunction"] = None

    def __init__(self, func: Callable, name: Optional[str] = None, known_block_size=None):
        install_excepthook()
        # ASTRewriter.transform mutates `func.__code__` in place.  To preserve
        # the *pre-rewrite* code object (whose co_names / co_freevars still
        # reference helper callables that the rewriter inlines into IR ops),
        # build a shallow function wrapper around the original __code__ here,
        # BEFORE the transform runs.  The JIT-cache dependency collector
        # uses this to detect changes to closure-captured helpers.
        import types as _types

        _orig_code = func.__code__
        self._original_func = _types.FunctionType(
            _orig_code,
            func.__globals__,
            name=func.__name__,
            argdefs=func.__defaults__,
            closure=func.__closure__,
        )
        self._original_func.__kwdefaults__ = func.__kwdefaults__
        self._original_func.__qualname__ = func.__qualname__
        self._original_func.__module__ = func.__module__
        self._func = ASTRewriter.transform(func)
        self._name = name
        self._known_block_size = _validate_known_block_size(known_block_size)
        self._kernel_name: Optional[str] = None
        self._shared_allocator = None

        full_sig = resolve_signature(self._func)
        params = list(full_sig.parameters.values())

        self._has_self_param = bool(params) and params[0].name == "self"
        if self._has_self_param:
            self._sig = full_sig.replace(parameters=params[1:])
        else:
            self._sig = full_sig

        # Definition-time annotation validity check (once per function, signature-only).
        warn_invalid_annotations(self._sig, context="@kernel")

    @classmethod
    def get_current(cls) -> Optional["KernelFunction"]:
        return cls._current

    def register_shared_allocator(self, alloc) -> None:
        if self._shared_allocator is not None:
            raise RuntimeError(
                "Only one SharedAllocator is allowed per kernel; "
                f"kernel '{self._kernel_name or self._func.__name__}' already has one"
            )
        self._shared_allocator = alloc

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return partial(self.__call__, obj)

    def _emit_kernel(
        self,
        ctx: CompilationContext,
        args: Tuple,
        kwargs: Dict,
        bound_self: Any = None,
        known_block_size: Optional[List[int]] = None,
    ):
        """Emit gpu.func for this kernel into the GPU module."""
        sig = self._sig
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        param_names: List[str] = []
        param_values: List[Any] = []
        constexpr_values: Dict[str, Any] = {}

        for param_name, value in bound.arguments.items():
            param = sig.parameters[param_name]
            annotation = param.annotation
            if annotation is not inspect.Parameter.empty and (
                Constexpr.is_constexpr_annotation(annotation) or is_type_param_annotation(annotation)
            ):
                constexpr_values[param_name] = value
            else:
                if (
                    annotation is not inspect.Parameter.empty
                    and isinstance(annotation, type)
                    and not isinstance(value, annotation)
                ):
                    warn_annotation_value_mismatch(param_name, annotation, type(value), context="@kernel")
                param_names.append(param_name)
                param_values.append(value)

        kernel_arg_types = []
        for value in param_values:
            kernel_arg_types.extend(get_ir_types(value))

        kernel_id = ctx.next_kernel_id()
        if self._name is not None:
            self._kernel_name = ctx.unique_kernel_name(self._name, kernel_id)
        else:
            self._kernel_name = ctx.unique_kernel_name(f"{self._func.__name__}_{kernel_id}", kernel_id)

        kernel_loc = func_def_location(self._func)

        self._shared_allocator = None
        KernelFunction._current = self
        try:
            with ir.InsertionPoint(ctx.gpu_module_body):
                func_type = ir.FunctionType.get(kernel_arg_types, [])
                with kernel_loc:
                    gpu_func = create_gpu_func(
                        self._kernel_name,
                        ir.TypeAttr.get(func_type),
                        known_block_size=known_block_size,
                    )
                gpu_func.regions[0].blocks.append(*kernel_arg_types)
                entry_block = gpu_func.regions[0].blocks[0]

                with ir.InsertionPoint(entry_block), kernel_loc:
                    block_args = list(entry_block.arguments)
                    dsl_args: Dict[str, Any] = {}
                    idx = 0
                    for param_name, value in zip(param_names, param_values):
                        n = len(get_ir_types(value))
                        dsl_args[param_name] = construct_from_ir_values(
                            type(value), value, list(block_args[idx : idx + n])
                        )
                        idx += n

                    dsl_args.update(constexpr_values)

                    fastmath_flag = effective_fastmath_hint(CompilationContext.get_compile_hints())
                    fastmath_scope = fastmath_ctx(fastmath_flag) if fastmath_flag is not None else nullcontext()
                    # Bound the call-site boundary at the kernel body.
                    with tracing_context(self._func), fastmath_scope:
                        if bound_self is not None:
                            self._func(bound_self, **dsl_args)
                        else:
                            self._func(**dsl_args)
                    gpu.ReturnOp([])
        finally:
            KernelFunction._current = None

        # The static memory size is handled by compiler.
        if self._shared_allocator is not None and not self._shared_allocator.is_static:
            smem_bytes = self._shared_allocator.allocated_bytes
        else:
            smem_bytes = None

        return self._kernel_name, tuple(param_values), gpu_func, smem_bytes

    def __call__(
        self,
        *args,
        unit_attrs: Optional[List[str]] = None,
        value_attrs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> KernelLauncher:
        ctx = CompilationContext.get_current()
        if ctx is None:
            raise RuntimeError("@kernel can only be called inside @jit function")

        bound_self = None
        if self._has_self_param:
            if not args:
                raise TypeError(f"{self._func.__name__}() missing 'self' argument")
            bound_self, args = args[0], args[1:]

        return KernelLauncher(
            self,
            ctx,
            args,
            kwargs,
            bound_self=bound_self,
            unit_attrs=unit_attrs,
            value_attrs=value_attrs,
        )


# =============================================================================
# Kernel Decorator
# =============================================================================


def kernel(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    known_block_size=None,
) -> KernelFunction:
    """Decorator for GPU kernel functions.

    Usage:
        @kernel
        def my_kernel(a: Tensor, b: Tensor):
            # kernel body
            ...

        # With explicit kernel name (visible in profiler):
        @kernel(name="gemm_m16n128k128_bf16")
        def my_kernel(a: Tensor):
            ...

        # With an explicit block size contract (normally inferred at launch):
        @kernel(known_block_size=[512, 1, 1])
        def my_kernel(a: Tensor):
            ...

    The decorated function can be called inside a @jit function to
    capture a pending kernel call. ``.launch(config)`` emits the specialized
    kernel definition and launch op together.

    Args:
        func: Function to decorate
        name: Optional kernel name override; shown in profiler instead of the
              Python function name. Tile/dtype info can be embedded here.
        known_block_size: Optional list of [x, y, z] block dimensions. Sets
              the ``known_block_size`` attribute on the GPU function, which the
              AMDGPU backend uses to derive ``max_flat_workgroup_size``.
              When omitted, static integer launch dimensions are inferred
              automatically. Explicit values remain useful as a contract for
              dynamic launch dimensions.

    Returns:
        KernelFunction wrapper
    """
    if func is None:
        return lambda f: KernelFunction(f, name=name, known_block_size=known_block_size)
    return KernelFunction(func, name=name, known_block_size=known_block_size)
