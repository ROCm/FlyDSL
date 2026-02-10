# FLIR IR Python Bindings
#
# This module loads the unified _flir_ir.so and sets up the necessary
# sys.modules aliases for MLIR Python bindings compatibility.
#
# The MLIR bindings are now under flydsl._mlir to avoid conflicts with
# other libraries that might use the _mlir namespace.

from typing import Sequence
import os
import sys

_this_dir = os.path.dirname(__file__)


def get_lib_dirs() -> Sequence[str]:
    return [_this_dir]


def get_include_dirs() -> Sequence[str]:
    return [os.path.join(_this_dir, "include")]


# Load unified module FIRST - before any other MLIR imports
from . import _flir_ir

# Register the module for direct imports
sys.modules['_flir_ir'] = _flir_ir

# Set up dialect search to find our dialect wrappers
# Without this, iterating operations won't return the correct OpView subclasses
_flir_ir._mlir.globals.append_dialect_search_prefix('flydsl._mlir.dialects')

# Register submodules for C++ code that imports _flir_ir.ir, etc.
# (required because MLIR_PYTHON_PACKAGE_PREFIX=_flir_ir. in the build)
sys.modules['_flir_ir.ir'] = _flir_ir._mlir.ir
sys.modules['_flir_ir.rewrite'] = _flir_ir._mlir.rewrite
sys.modules['_flir_ir.passmanager'] = _flir_ir._mlir.passmanager
sys.modules['_flir_ir.dialects.gpu'] = _flir_ir._mlirDialectsGPU
sys.modules['_flir_ir.dialects.llvm'] = _flir_ir._mlirDialectsLLVM
sys.modules['_flir_ir.execution_engine'] = _flir_ir._mlirExecutionEngine

# Store reference to the internal _mlir module
_mlir_internal = _flir_ir._mlir

# Register flydsl._mlir._mlir_libs.* aliases for LLVM Python wrappers compatibility
# These aliases allow LLVM's gpu/__init__.py to do:
#   from ..._mlir_libs._mlirDialectsGPU import *
# Now scoped to flydsl namespace - won't conflict with other libraries
sys.modules['flydsl._mlir._mlir_libs'] = sys.modules[__name__]
sys.modules['flydsl._mlir._mlir_libs._mlir'] = _flir_ir._mlir
sys.modules['flydsl._mlir._mlir_libs._mlirDialectsGPU'] = _flir_ir._mlirDialectsGPU
sys.modules['flydsl._mlir._mlir_libs._mlirDialectsLLVM'] = _flir_ir._mlirDialectsLLVM
sys.modules['flydsl._mlir._mlir_libs._mlirExecutionEngine'] = _flir_ir._mlirExecutionEngine


# Register submodules in this package's namespace
sys.modules[__name__ + '._mlir'] = _mlir_internal
sys.modules[__name__ + '._mlir.ir'] = _mlir_internal.ir
sys.modules[__name__ + '._mlir.rewrite'] = _mlir_internal.rewrite
sys.modules[__name__ + '._mlir.passmanager'] = _mlir_internal.passmanager
sys.modules[__name__ + '._mlirDialectsGPU'] = _flir_ir._mlirDialectsGPU
sys.modules[__name__ + '._mlirDialectsLLVM'] = _flir_ir._mlirDialectsLLVM
sys.modules[__name__ + '._mlirGPUPasses'] = _flir_ir._mlirGPUPasses
sys.modules[__name__ + '._mlirExecutionEngine'] = _flir_ir._mlirExecutionEngine
sys.modules[__name__ + '._mlirRegisterEverything'] = _flir_ir._mlirRegisterEverything
sys.modules[__name__ + '._flirPasses'] = _flir_ir._flirPasses

# Expose _mlir in module namespace
_mlir = _mlir_internal



# ---------------------------------------------------------------------------
# Dialect registry and Context initialization (from MLIR's _mlir_libs)
# ---------------------------------------------------------------------------

_dialect_registry = None
_load_on_create_dialects = None


def get_dialect_registry():
    global _dialect_registry
    if _dialect_registry is None:
        _dialect_registry = _mlir_internal.ir.DialectRegistry()
    return _dialect_registry


def append_load_on_create_dialect(dialect: str):
    global _load_on_create_dialects
    if _load_on_create_dialects is None:
        _load_on_create_dialects = [dialect]
    else:
        _load_on_create_dialects.append(dialect)


def get_load_on_create_dialects():
    global _load_on_create_dialects
    if _load_on_create_dialects is None:
        _load_on_create_dialects = []
    return _load_on_create_dialects


def _site_initialize():
    import importlib
    import itertools
    import logging

    ir = _mlir_internal.ir
    logger = logging.getLogger(__name__)
    post_init_hooks = []
    disable_multithreading = False
    disable_load_all_available_dialects = False

    def process_initializer_module(module_name):
        nonlocal disable_multithreading
        nonlocal disable_load_all_available_dialects
        try:
            m = importlib.import_module(f".{module_name}", __name__)
        except ModuleNotFoundError:
            return False
        except ImportError:
            logger.warning(
                f"Error importing mlir initializer {module_name}", exc_info=True
            )
            return False

        if hasattr(m, "register_dialects"):
            m.register_dialects(get_dialect_registry())
        if hasattr(m, "context_init_hook"):
            post_init_hooks.append(m.context_init_hook)
        if hasattr(m, "disable_multithreading"):
            if bool(m.disable_multithreading):
                disable_multithreading = True
        if hasattr(m, "disable_load_all_available_dialects"):
            disable_load_all_available_dialects = True
        return True

    init_module = None
    if process_initializer_module("_mlirRegisterEverything"):
        init_module = importlib.import_module("._mlirRegisterEverything", __name__)

    # Also process _flirPasses for FLIR dialect registration
    process_initializer_module("_flirPasses")

    for i in itertools.count():
        module_name = f"_site_initialize_{i}"
        if not process_initializer_module(module_name):
            break

    class Context(ir._BaseContext):
        def __init__(
            self, load_on_create_dialects=None, thread_pool=None, *args, **kwargs
        ):
            super().__init__(*args, **kwargs)
            self.append_dialect_registry(get_dialect_registry())
            for hook in post_init_hooks:
                hook(self)
            if disable_multithreading and thread_pool is not None:
                raise ValueError(
                    "Context constructor has given thread_pool argument, "
                    "but disable_multithreading flag is True."
                )
            if not disable_multithreading:
                if thread_pool is None:
                    self.enable_multithreading(True)
                else:
                    self.set_thread_pool(thread_pool)
            if load_on_create_dialects is not None:
                for dialect in load_on_create_dialects:
                    _ = self.dialects[dialect]
            else:
                if disable_load_all_available_dialects:
                    dialects = get_load_on_create_dialects()
                    for dialect in dialects:
                        _ = self.dialects[dialect]
                else:
                    self.load_all_available_dialects()
            if init_module:
                init_module.register_llvm_translations(self)

    ir.Context = Context

    class MLIRError(Exception):
        def __init__(self, message, error_diagnostics):
            self.message = message
            self.error_diagnostics = error_diagnostics
            super().__init__(message, error_diagnostics)

        def __str__(self):
            s = self.message
            if self.error_diagnostics:
                s += ":"
            for diag in self.error_diagnostics:
                s += (
                    "\nerror: "
                    + str(diag.location)[4:-1]
                    + ": "
                    + diag.message.replace("\n", "\n  ")
                )
                for note in diag.notes:
                    s += (
                        "\n note: "
                        + str(note.location)[4:-1]
                        + ": "
                        + note.message.replace("\n", "\n  ")
                    )
            return s

    ir.MLIRError = MLIRError


_site_initialize()
