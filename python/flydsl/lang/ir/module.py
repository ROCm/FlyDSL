import inspect
from typing import Optional

from ..._mlir import ir
from ..._mlir.extras import types as T
from ..._mlir.dialects import arith, func, _gpu_ops_gen


from .gpu import (
    gpu_func,
    prep_func_types,
    LaunchFuncOp,
    block_idx,
    thread_idx,
    block_dim,
    grid_dim,
)


class GlobalRAIIMLIRContext:
    context: ir.Context
    location: ir.Location

    def __init__(self, allow_unregistered_dialects=False):
        self.context = ir.Context()
        if allow_unregistered_dialects:
            self.context.allow_unregistered_dialects = True
        self.context.__enter__()
        self.location = ir.Location.unknown()
        self.location.__enter__()

    def __del__(self):
        self.location.__exit__(None, None, None)
        self.context.__exit__(None, None, None)


class MlirModule:
    GPU_MODULE_NAME = "kernels"

    cls_kernel_fn = []
    cls_jit_fn = []
    cls_kernel_sym = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Initialize MLIR module for this subclass FIRST
        cls.module = ir.Module.create()
        cls.module.operation.attributes["gpu.container_module"] = ir.UnitAttr.get()

        with ir.InsertionPoint(cls.module.body):
            cls.gpu_module = _gpu_ops_gen.module(cls.GPU_MODULE_NAME)

        # After MLIR module is created, collect functions registered by descriptors
        # Descriptors __set_name__ runs during class creation, adding to temporary lists
        # We need to move them to the class-specific lists
        temp_kernel_fn = []
        temp_jit_fn = []
        temp_kernel_sym = {}

        # Collect from class __dict__ directly (not inherited)
        for name, value in cls.__dict__.items():
            if isinstance(value, _KernelDescriptor):
                # This descriptor belongs to this class
                if hasattr(value, "_wrapper"):
                    temp_kernel_fn.append(value._wrapper)
                    temp_kernel_sym[name] = name
            elif isinstance(value, _JitDescriptor):
                if hasattr(value, "_wrapper"):
                    temp_jit_fn.append(value._wrapper)

        # Set class-specific lists
        cls.cls_kernel_fn = temp_kernel_fn
        cls.cls_jit_fn = temp_jit_fn
        cls.cls_kernel_sym = temp_kernel_sym

    def __init__(self):
        self.kernel_func_op = {}
        for fn in self.cls_jit_fn:
            fn(self)
        for fn in self.cls_kernel_fn:
            fn(self)

    def __repr__(self):
        return str(self.module)

    def __getattr__(self, name: str):
        if name in self.cls_kernel_sym.keys():
            return ir.SymbolRefAttr.get(
                [self.GPU_MODULE_NAME, self.cls_kernel_sym[name]]
            )
        raise AttributeError(f"{name} not found in kernel functions.")

    @classmethod
    def create_gpu_module(cls, module_attrs=None):
        cls.gpu_module = _gpu_ops_gen.module("kernels")

    @classmethod
    def create_from_mlir_source(cls, file_path: str):
        pass

    @classmethod
    def kernel(cls, fn):
        def wrapper(self, *args, **kwargs):
            if len(self.gpu_module.bodyRegion.blocks) == 0:
                self.gpu_module.bodyRegion.blocks.append()
            with ir.InsertionPoint.at_block_begin(self.gpu_module.bodyRegion.blocks[0]):
                self.kernel_func_op[fn.__name__] = gpu_func(fn, emit=True)

        cls.cls_kernel_fn.append(wrapper)
        cls.cls_kernel_sym[fn.__name__] = fn.__name__
        return fn

    @classmethod
    def jit(cls, fn):
        def wrapper(self):
            with ir.InsertionPoint.at_block_begin(self.module.body):
                sig = inspect.signature(fn)
                input_types, return_types, _ = prep_func_types(sig, [])
                func.FuncOp.from_py_func(*input_types)(fn)

        cls.cls_jit_fn.append(wrapper)
        return fn


class _KernelDescriptor:
    """Descriptor that automatically registers kernel to the correct class."""

    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        self._wrapper = None

    def __set_name__(self, owner, name):
        """Called when the descriptor is assigned to a class attribute."""
        # Check if owner is a subclass of MlirModule
        try:
            if issubclass(owner, MlirModule):
                # Capture fn in the closure
                fn = self.fn

                def wrapper(instance_self, *args, **kwargs):
                    if len(instance_self.gpu_module.bodyRegion.blocks) == 0:
                        instance_self.gpu_module.bodyRegion.blocks.append()
                    with ir.InsertionPoint.at_block_begin(
                        instance_self.gpu_module.bodyRegion.blocks[0]
                    ):
                        instance_self.kernel_func_op[fn.__name__] = gpu_func(
                            fn, emit=True
                        )

                # Store the wrapper in the descriptor for later collection
                self._wrapper = wrapper
                self._name = name
        except TypeError:
            # owner is not a class, skip
            pass

    def __get__(self, obj, objtype=None):
        """Return the original function for method access."""
        if obj is None:
            return self.fn
        return self.fn.__get__(obj, objtype)


class _JitDescriptor:
    """Descriptor that automatically registers jit function to the correct class."""

    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        self._wrapper = None

    def __set_name__(self, owner, name):
        """Called when the descriptor is assigned to a class attribute."""
        # Check if owner is a subclass of MlirModule
        try:
            if issubclass(owner, MlirModule):
                # Capture fn in the closure
                fn = self.fn

                def wrapper(instance_self):
                    with ir.InsertionPoint.at_block_begin(instance_self.module.body):
                        sig = inspect.signature(fn)
                        input_types, return_types, _ = prep_func_types(sig, [])
                        func.FuncOp.from_py_func(*input_types)(fn)

                # Store the wrapper in the descriptor for later collection
                self._wrapper = wrapper
        except TypeError:
            # owner is not a class, skip
            pass

    def __get__(self, obj, objtype=None):
        """Return the original function for method access."""
        if obj is None:
            return self.fn
        return self.fn.__get__(obj, objtype)


# Use descriptor-based decorators that return descriptors
def kernel(fn):
    """Decorator that returns a descriptor for automatic class detection."""
    return _KernelDescriptor(fn)


def jit(fn):
    """Decorator that returns a descriptor for automatic class detection."""
    return _JitDescriptor(fn)


_global_ctx = GlobalRAIIMLIRContext()
