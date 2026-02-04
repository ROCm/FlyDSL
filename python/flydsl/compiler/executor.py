import atexit
import ctypes
import os
import weakref
import torch

from mlir import ir as upstream_ir
from .._mlir.execution_engine import ExecutionEngine
from .._mlir.interop import to_upstream

# Track all executors for cleanup
_active_executors = []

def _cleanup_all_executors():
    """Cleanup all executors before Python shutdown."""
    global _active_executors
    for executor_ref in _active_executors:
        executor = executor_ref()
        if executor is not None:
            executor.cleanup()
    _active_executors.clear()

atexit.register(_cleanup_all_executors)


class Executor:
    def __init__(self, jit_module):
        # Get MLIR_PATH from environment variable
        mlir_path = os.environ.get('MLIR_PATH')
        if not mlir_path:
            raise RuntimeError(
                "Environment variable MLIR_PATH is not set!\n"
                "Please set MLIR_PATH before running the program, for example:\n"
                "  export MLIR_PATH=/path/to/llvm-project/buildmlir\n"
                "Or source the configuration script:\n"
                "  source pre_build.sh"
            )
        
        lib_dir = os.path.join(mlir_path, 'lib')
        
        # CRITICAL: Convert flydsl Module to upstream Module for ExecutionEngine
        upstream_module = to_upstream(jit_module)
        
        self.engine = ExecutionEngine(
            upstream_module,
            opt_level=3,
            shared_libs=[
                os.path.join(lib_dir, "libmlir_rocm_runtime.so"),
                os.path.join(lib_dir, "libmlir_runner_utils.so"),
            ],
        )
        self.engine.initialize()
        self._cleaned_up = False
        
        # Register this executor for cleanup
        _active_executors.append(weakref.ref(self))

    def cleanup(self):
        """Explicitly cleanup execution engine resources."""
        if getattr(self, '_cleaned_up', False):
            return
        self._cleaned_up = True
        if hasattr(self, 'engine') and self.engine is not None:
            # ExecutionEngine doesn't have explicit cleanup, but set to None
            # to release references
            self.engine = None

    def __del__(self):
        # Use getattr to avoid recursion in __getattr__
        if not getattr(self, '_cleaned_up', True):
            self.cleanup()

    def convert_args(self, args):
        if isinstance(args, torch.Tensor):
            # Return a ctypes pointer to the GPU pointer value
            # MLIR expects pointer-to-pointer for packed calling convention
            ptr_storage = ctypes.c_void_p(args.data_ptr())
            return ptr_storage
        else:
            raise TypeError(f"Unsupported argument type: {type(args)}")

    def __call__(self, *args):
        return self.__getattr__("__call__")(*args)

    def __getattr__(self, name: str):
        try:
            func_ptr = self.engine.raw_lookup(name)
        except KeyError:
            raise AttributeError(f"No such function: {name}") from None

        def wrapper(*args):
            # MLIR ExecutionEngine uses "packed" calling convention:
            # The JIT'd function expects a single void* argument that points to
            # an array of void* pointers. Each element points to the actual argument.
            # For void functions: [arg0_ptr, arg1_ptr, ...]
            # For functions with return value: [arg0_ptr, ..., ret_ptr]
            
            # Create storage for pointer values (must keep alive during call)
            ptr_storage = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    # Store GPU pointer value
                    ptr_val = ctypes.c_void_p(arg.data_ptr())
                    ptr_storage.append(ptr_val)
                else:
                    raise TypeError(f"Unsupported argument type: {type(arg)}")
            
            # Create array of pointers to the stored values
            args_array = (ctypes.c_void_p * len(args))()
            for i, ptr in enumerate(ptr_storage):
                args_array[i] = ctypes.addressof(ptr)
            
            # Call with packed convention: void func(void** args)
            func_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
            func_exe = func_type(func_ptr)
            func_exe(ctypes.cast(args_array, ctypes.c_void_p))

        return wrapper
