import ctypes
import os
import torch


from .._mlir.execution_engine import ExecutionEngine


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
        
        self.engine = ExecutionEngine(
            jit_module,
            opt_level=3,
            shared_libs=[
                os.path.join(lib_dir, "libmlir_rocm_runtime.so"),
                os.path.join(lib_dir, "libmlir_runner_utils.so"),
            ],
        )
        self.engine.initialize()

    def convert_args(self, args):
        if isinstance(args, torch.Tensor):
            return ctypes.cast(
                ctypes.pointer(ctypes.c_void_p(args.data_ptr())), ctypes.c_void_p
            )
        else:
            raise TypeError(f"Unsupported argument type: {type(args)}")

    def __call__(self, *args):
        return self.__getattr__("__call__")(*args)

    def __getattr__(self, name: str):
        try:
            func_ptr = self.engine.raw_lookup(name)
            func_exe = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(func_ptr)
        except KeyError:
            raise AttributeError(f"No such function: {name}") from None

        def wrapper(*args):
            addresses = [ctypes.c_void_p(0)]
            addresses += [self.convert_args(arg) for arg in args]
            c_args = (ctypes.c_void_p * len(addresses))(*addresses)
            return func_exe(c_args)

        return wrapper
