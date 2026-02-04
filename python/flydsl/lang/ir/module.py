import inspect
from typing import Optional

# CRITICAL: Use flydsl's own _mlir package following NV Cutlass pattern
# All MLIR functionality is self-contained under flydsl._mlir
from flydsl._mlir import ir
from flydsl._mlir.extras import types as T
from flydsl._mlir.interop import to_upstream, to_upstream_types

# ALSO import upstream mlir for bridging with upstream dialects
from mlir import ir as upstream_ir

# Import dialects from flydsl._mlir (NOT from upstream mlir!)
from flydsl._mlir.dialects import arith, func
from flydsl._mlir.dialects import gpu as _gpu_ops_gen


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
    """
    RAII Context that sets BOTH flydsl and upstream mlir contexts.
    
    This is critical for cross-domain operation: we need both domains
    to see the same underlying MlirContext* as their current context.
    """
    context: ir.Context
    location: ir.Location
    _cleaned_up: bool = False

    def __init__(self, allow_unregistered_dialects=False):
        # STRATEGY: Use ONLY upstream context to avoid domain isolation issues
        # The flydsl dialect will be registered via the shared C pointer
        
        # 1. Create upstream context first (has access to all standard dialects)
        self.upstream_context = upstream_ir.Context()
        self.upstream_context.load_all_available_dialects()
        self.upstream_context.allow_unregistered_dialects = True
        
        # 2. Create flydsl wrapper sharing same underlying MlirContext*
        if hasattr(self.upstream_context, '_CAPIPtr'):
            self.context = ir.Context._CAPICreate(self.upstream_context._CAPIPtr)
        else:
            # Fallback: use upstream directly (may cause issues)
            self.context = ir.Context()
            self.context.load_all_available_dialects()
        
        # 3. Register Fly dialect on flydsl wrapper
        import flydsl
        flydsl.register_dialect(self.context)
        
        self.context.allow_unregistered_dialects = True
        
        # 4. Enter BOTH contexts (order matters - upstream last for current)
        self.context.__enter__()
        self.upstream_context.__enter__()
        
        # CRITICAL: Initialize location to avoid AttributeError in __del__
        self.location = ir.Location.unknown(self.context)
        self.location.__enter__()
        
        # Also set upstream location
        if hasattr(self.location, '_CAPIPtr'):
            self.upstream_location = upstream_ir.Location._CAPICreate(self.location._CAPIPtr)
        else:
            self.upstream_location = upstream_ir.Location.unknown()
        self.upstream_location.__enter__()
        
        self._cleaned_up = False

    def cleanup(self):
        """Explicitly cleanup resources. Call this before Python interpreter shutdown."""
        if self._cleaned_up:
            return
        self._cleaned_up = True
        
        # Exit in reverse order of entry
        try:
            if hasattr(self, 'upstream_location') and self.upstream_location is not None:
                self.upstream_location.__exit__(None, None, None)
                self.upstream_location = None
        except Exception:
            pass
            
        try:
            if hasattr(self, 'location') and self.location is not None:
                self.location.__exit__(None, None, None)
                self.location = None
        except Exception:
            pass
            
        try:
            if hasattr(self, 'upstream_context') and self.upstream_context is not None:
                self.upstream_context.__exit__(None, None, None)
                self.upstream_context = None
        except Exception:
            pass
            
        try:
            if hasattr(self, 'context') and self.context is not None:
                self.context.__exit__(None, None, None)
                self.context = None
        except Exception:
            pass

    def __del__(self):
        # Delegate to cleanup method
        self.cleanup()


class MlirModule:
    GPU_MODULE_NAME = "kernels"

    cls_kernel_fn = []
    cls_jit_fn = []
    cls_kernel_sym = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # CRITICAL: Use UPSTREAM ir for Module creation since GPU dialect is upstream
        # This ensures Module and GPU module use the same Python domain
        cls.module = upstream_ir.Module.create()
        cls.module.operation.attributes["gpu.container_module"] = upstream_ir.UnitAttr.get()

        with upstream_ir.InsertionPoint(cls.module.body):
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
        # Provide access to the global context for convenience
        self._context = _global_ctx.context
        self._upstream_context = _global_ctx.upstream_context
        
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
            # CRITICAL: Use upstream_ir since gpu_module is upstream
            with upstream_ir.InsertionPoint.at_block_begin(self.gpu_module.bodyRegion.blocks[0]):
                self.kernel_func_op[fn.__name__] = gpu_func(fn, emit=True)

        cls.cls_kernel_fn.append(wrapper)
        cls.cls_kernel_sym[fn.__name__] = fn.__name__
        return fn

    @classmethod
    def jit(cls, fn):
        def wrapper(self):
            # CRITICAL: Use upstream_ir since module is upstream
            with upstream_ir.InsertionPoint.at_block_begin(self.module.body):
                sig = inspect.signature(fn)
                input_types, return_types, _ = prep_func_types(sig, [])
                # CRITICAL: Convert flydsl types to upstream types for func dialect
                upstream_input_types = to_upstream_types(input_types)
                func.FuncOp.from_py_func(*upstream_input_types)(fn)

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
                    # Use upstream InsertionPoint for upstream gpu_module's Block
                    with upstream_ir.InsertionPoint.at_block_begin(
                        instance_self.gpu_module.bodyRegion.blocks[0]
                    ):
                        # Create a bound function that excludes 'self' from signature
                        # but binds instance_self for the actual call
                        sig = inspect.signature(fn)
                        params = list(sig.parameters.values())
                        
                        # Check if first param is 'self'
                        if params and params[0].name == 'self':
                            # Create a wrapper that binds self
                            def bound_fn(*args):
                                return fn(instance_self, *args)
                            bound_fn.__name__ = fn.__name__
                            
                            # Create new signature without 'self'
                            new_params = params[1:]
                            bound_fn.__signature__ = sig.replace(parameters=new_params)
                            
                            instance_self.kernel_func_op[fn.__name__] = gpu_func(
                                bound_fn, emit=True
                            )
                        else:
                            # No self parameter, use fn directly
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
                    # CRITICAL: Use upstream_ir since module is upstream
                    with upstream_ir.InsertionPoint.at_block_begin(instance_self.module.body):
                        sig = inspect.signature(fn)
                        # Get input types, skipping 'self' parameter
                        params = list(sig.parameters.values())
                        if params and params[0].name == 'self':
                            params = params[1:]  # Skip 'self'
                        
                        input_types = [
                            p.annotation
                            for p in params
                            if not p.annotation is inspect.Signature.empty
                        ]
                        
                        # CRITICAL: Convert flydsl types to upstream types for func dialect
                        upstream_input_types = to_upstream_types(input_types)
                        
                        # Create a bound function that correctly receives MLIR Value args
                        def bound_fn(*args):
                            return fn(instance_self, *args)
                        bound_fn.__name__ = fn.__name__
                        # Set up signature for from_py_func to use
                        bound_fn.__annotations__ = {
                            p.name: p.annotation
                            for p in params
                            if p.annotation is not inspect.Signature.empty
                        }
                        
                        func.FuncOp.from_py_func(*upstream_input_types, name=fn.__name__)(bound_fn)

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


# Create global context and register cleanup with atexit
import atexit

_global_ctx = GlobalRAIIMLIRContext()

def _cleanup_global_context():
    """Cleanup global context before Python interpreter shutdown."""
    global _global_ctx
    if _global_ctx is not None:
        _global_ctx.cleanup()
        _global_ctx = None

# Register cleanup to run BEFORE Python finalizes
atexit.register(_cleanup_global_context)
