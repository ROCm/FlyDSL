import inspect
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, get_origin

from .._mlir import ir
from .._mlir.dialects import arith, gpu
from ..expr.typing import Constexpr
from .ast_rewriter import ASTRewriter
from .protocol import extract_ir_values, get_ir_types, new_from_ir_values

# =============================================================================
# GPU Operation Helpers
# =============================================================================


def create_gpu_module(
    sym_name: str,
    targets: Optional[List[str]] = None,
    *,
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
    module_op = gpu.GPUModuleOp(
        sym_name, targets=ir.ArrayAttr.get(target_attrs) if target_attrs else None, loc=loc, ip=ip
    )
    module_op.regions[0].blocks.append()
    return module_op


def get_gpu_module_body(module_op: gpu.GPUModuleOp):
    return module_op.regions[0].blocks[0]


def create_gpu_func(
    sym_name: str,
    function_type: ir.TypeAttr,
    *,
    loc=None,
    ip=None,
) -> gpu.GPUFuncOp:
    return gpu.GPUFuncOp(function_type, sym_name=sym_name, kernel=True, loc=loc, ip=ip)


# =============================================================================
# Location Tracking Utilities
# =============================================================================


def get_source_location(depth: int = 2) -> Tuple[str, int, int]:
    """Get source file location from call stack.

    Args:
        depth: Stack depth to look up (2 = caller's caller)

    Returns:
        Tuple of (filename, line, column)
    """
    frame = inspect.currentframe()
    try:
        for _ in range(depth):
            if frame is not None:
                frame = frame.f_back
        if frame is not None:
            return (frame.f_code.co_filename, frame.f_lineno, 0)
    finally:
        del frame
    return ("<unknown>", 0, 0)


def create_file_location(filename: str, line: int, col: int = 0, context=None) -> ir.Location:
    """Create an MLIR file location."""
    ctx = context or ir.Context.current
    return ir.Location.file(filename, line, col, context=ctx)


def create_caller_location(depth: int = 2, context=None) -> ir.Location:
    """Create an MLIR location from the caller's source position."""
    filename, line, col = get_source_location(depth + 1)
    return create_file_location(filename, line, col, context)


class FuncLocationTracker:
    """Track source locations for a Python function being traced."""

    def __init__(self, func: Callable):
        self._func = func
        self._filename = inspect.getfile(func)
        try:
            self._source_lines, self._start_line = inspect.getsourcelines(func)
        except (OSError, TypeError):
            self._source_lines = []
            self._start_line = 0

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def start_line(self) -> int:
        return self._start_line

    def get_func_location(self, context=None) -> ir.Location:
        """Get location for the function definition."""
        return create_file_location(self._filename, self._start_line, 0, context)

    @contextmanager
    def func_scope(self):
        """Enter a location scope for this function."""
        loc = self.get_func_location()
        with loc:
            yield loc


# =============================================================================
# Launch Configuration
# =============================================================================

DimValueType = Union[int, ir.Value]
DimType = Union[int, ir.Value, Tuple[DimValueType, ...], List[DimValueType]]


def _unwrap_to_raw(val):
    if isinstance(val, ir.Value):
        return val
    if hasattr(val, "__extract_ir_values__"):
        values = val.__extract_ir_values__()
        if len(values) == 1:
            return values[0]
    return val


def _to_index_value(val: DimValueType) -> ir.Value:
    val = _unwrap_to_raw(val)
    if isinstance(val, ir.Value):
        if val.type == ir.IndexType.get():
            return val
        return arith.index_cast(ir.IndexType.get(), val)
    return arith.constant(ir.IndexType.get(), val)


def _normalize_dim(dim: DimType) -> Tuple[DimValueType, DimValueType, DimValueType]:
    if isinstance(dim, (int, ir.Value)):
        return (dim, 1, 1)
    elif len(dim) == 1:
        return (dim[0], 1, 1)
    elif len(dim) == 2:
        return (dim[0], dim[1], 1)
    return (dim[0], dim[1], dim[2])


# =============================================================================
# Compilation Context (per-compilation state)
# =============================================================================


class CompilationContext:
    """Context for tracking compilation state within a @jit function.

    Manages:
    - GPU module op for kernel definitions
    - Kernel counter for unique naming
    - Location trackers for debugging
    """

    _current: Optional["CompilationContext"] = None

    def __init__(self, func_tracker: Optional[FuncLocationTracker] = None):
        self.gpu_module_op = None
        self.kernel_counter = 0
        self.func_tracker = func_tracker
        self.kernel_trackers: Dict[str, FuncLocationTracker] = {}

    @classmethod
    def get_current(cls) -> Optional["CompilationContext"]:
        return cls._current

    @classmethod
    @contextmanager
    def create(cls, func_tracker: Optional[FuncLocationTracker] = None):
        prev = cls._current
        ctx = CompilationContext(func_tracker)
        cls._current = ctx
        try:
            yield ctx
        finally:
            cls._current = prev

    def next_kernel_id(self) -> int:
        """Get next unique kernel ID."""
        kid = self.kernel_counter
        self.kernel_counter += 1
        return kid

    def register_kernel_tracker(self, name: str, tracker: FuncLocationTracker):
        """Register a location tracker for a kernel function."""
        self.kernel_trackers[name] = tracker

    def get_kernel_tracker(self, name: str) -> Optional[FuncLocationTracker]:
        """Get the location tracker for a kernel function."""
        return self.kernel_trackers.get(name)


# =============================================================================
# Kernel Launcher
# =============================================================================


class KernelLauncher:
    """Holds kernel reference and generates gpu.launch_func on launch().

    Created by calling a @kernel decorated function. Call .launch()
    to emit the actual launch operation.
    """

    def __init__(
        self,
        kernel_name: str,
        kernel_args: Tuple,
        call_location: Optional[ir.Location] = None,
    ):
        self._kernel_name = kernel_name
        self._kernel_args = kernel_args
        self._call_location = call_location

    def launch(
        self,
        *,
        grid: DimType = (1, 1, 1),
        block: DimType = (1, 1, 1),
        smem: Union[int, ir.Value] = 0,
        stream: Optional[ir.Value] = None,
    ) -> None:
        """Emit gpu.launch_func operation with the given configuration.

        Args:
            grid: Grid dimensions (x, y, z). Can be int, ir.Value, tuple, or list.
            block: Block dimensions (x, y, z). Can be int, ir.Value, tuple, or list.
            smem: Dynamic shared memory size in bytes. Can be int or ir.Value.
            stream: CUDA/HIP stream as ir.Value. None means default stream.
        """
        launch_loc = create_caller_location(depth=2)

        kernel_operands = []
        for arg in self._kernel_args:
            kernel_operands.extend(extract_ir_values(arg))

        grid_dims = _normalize_dim(grid)
        block_dims = _normalize_dim(block)

        with launch_loc:
            grid_x = _to_index_value(grid_dims[0])
            grid_y = _to_index_value(grid_dims[1])
            grid_z = _to_index_value(grid_dims[2])
            block_x = _to_index_value(block_dims[0])
            block_y = _to_index_value(block_dims[1])
            block_z = _to_index_value(block_dims[2])

            smem_val = None
            if isinstance(smem, ir.Value):
                smem_val = smem
            elif smem > 0:
                smem_val = arith.constant(ir.IntegerType.get_signless(32), smem)

            async_object = _unwrap_to_raw(stream) if stream is not None else None

            gpu.LaunchFuncOp(
                ["kernels", self._kernel_name],
                (grid_x, grid_y, grid_z),
                (block_x, block_y, block_z),
                kernel_operands,
                dynamic_shared_memory_size=smem_val,
                async_object=async_object,
                loc=launch_loc,
                ip=None,
            )


# =============================================================================
# Kernel Function
# =============================================================================


class KernelFunction:
    """Wrapper for @kernel decorated functions.

    When called, emits a gpu.func and returns a KernelLauncher for
    configuring and launching the kernel.
    """

    def __init__(self, func: Callable, some_args=None):
        self._func = ASTRewriter.transform(func)
        self._some_args = some_args
        self._kernel_name: Optional[str] = None
        self._location_tracker = FuncLocationTracker(func)

    @staticmethod
    def _is_constexpr_annotation(annotation) -> bool:
        if annotation is Constexpr:
            return True
        return get_origin(annotation) is Constexpr

    def _emit_kernel(self, ctx: CompilationContext, args: Tuple, kwargs: Dict) -> Tuple[Any, ...]:
        """Emit gpu.func for this kernel into the GPU module.

        Returns:
            Tuple of non-constexpr argument values for use in launch.
        """
        sig = inspect.signature(self._func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        param_names: List[str] = []
        param_values: List[Any] = []
        constexpr_values: Dict[str, Any] = {}

        for param_name, value in bound.arguments.items():
            param = sig.parameters[param_name]
            annotation = param.annotation
            if annotation is not inspect.Parameter.empty and self._is_constexpr_annotation(annotation):
                constexpr_values[param_name] = value
            else:
                param_names.append(param_name)
                param_values.append(value)

        kernel_arg_types = []
        for value in param_values:
            kernel_arg_types.extend(get_ir_types(value))

        kernel_id = ctx.next_kernel_id()
        self._kernel_name = f"{self._func.__name__}_{kernel_id}"

        ctx.register_kernel_tracker(self._kernel_name, self._location_tracker)

        kernel_loc = self._location_tracker.get_func_location()

        with ir.InsertionPoint(ctx.gpu_module_body):
            func_type = ir.FunctionType.get(kernel_arg_types, [])
            with kernel_loc:
                gpu_func = create_gpu_func(self._kernel_name, ir.TypeAttr.get(func_type))
            gpu_func.regions[0].blocks.append(*kernel_arg_types)
            entry_block = gpu_func.regions[0].blocks[0]

            with ir.InsertionPoint(entry_block), kernel_loc:
                block_args = list(entry_block.arguments)
                dsl_args: Dict[str, Any] = {}
                idx = 0
                for param_name, value in zip(param_names, param_values):
                    n = len(get_ir_types(value))
                    dsl_args[param_name] = new_from_ir_values(type(value), value, list(block_args[idx : idx + n]))
                    idx += n

                dsl_args.update(constexpr_values)
                self._func(**dsl_args)
                gpu.ReturnOp([])

        return tuple(param_values)

    def __call__(self, *args, **kwargs) -> KernelLauncher:
        ctx = CompilationContext.get_current()
        if ctx is None:
            raise RuntimeError("@kernel can only be called inside @jit function")

        call_loc = create_caller_location(depth=2)

        kernel_args = self._emit_kernel(ctx, args, kwargs)

        return KernelLauncher(self._kernel_name, kernel_args, call_loc)


# =============================================================================
# Kernel Decorator
# =============================================================================


def kernel(func: Optional[Callable] = None, *, some_args=None) -> KernelFunction:
    """Decorator for GPU kernel functions.

    Usage:
        @kernel
        def my_kernel(a: Tensor, b: Tensor):
            # kernel body
            ...

        # Or with arguments
        @kernel(some_args=value)
        def my_kernel(a: Tensor):
            ...

    The decorated function can be called inside a @jit function to
    define the kernel, then .launch(config) is called to emit the launch op.

    Args:
        func: Function to decorate
        some_args: Optional kernel-specific arguments

    Returns:
        KernelFunction wrapper
    """
    if func is None:
        return lambda f: KernelFunction(f, some_args=some_args)
    return KernelFunction(func, some_args=some_args)
