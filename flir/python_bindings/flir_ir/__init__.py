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

# Register the module under its short name for direct imports.
# Also mirror nanobind's auto-registered submodules under the short prefix,
# because C++ capsule code uses MLIR_PYTHON_PACKAGE_PREFIX=_flir_ir._mlir.
# to call nb::module_::import_("_flir_ir._mlir.ir"), etc.
# Nanobind registers them under the full package path (flydsl._mlir._mlir_libs._flir_ir._mlir.ir),
# so we add short-name aliases automatically.
sys.modules['_flir_ir'] = _flir_ir
_full = _flir_ir.__name__  # e.g. flydsl._mlir._mlir_libs._flir_ir
for _key in list(sys.modules):
    if _key.startswith(_full + '.'):
        _short = '_flir_ir' + _key[len(_full):]
        if _short not in sys.modules:
            sys.modules[_short] = sys.modules[_key]
del _full

# Set up dialect search to find our dialect wrappers
# Without this, iterating operations won't return the correct OpView subclasses
_flir_ir._mlir.globals.append_dialect_search_prefix('flydsl._mlir.dialects')

# LLVM wrapper imports: proxy modules in _mlir_libs/ (_mlir/, _mlirDialectsGPU.py,
# etc.) re-export from _flir_ir, so LLVM's original wrapper files work unmodified.

_mlir_internal = _flir_ir._mlir



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
    import logging

    ir = _mlir_internal.ir
    logger = logging.getLogger(__name__)
    post_init_hooks = []
    disable_multithreading = False

    def process_initializer(m):
        """Process a _flir_ir submodule for dialect/pass registration."""
        nonlocal disable_multithreading
        if hasattr(m, "register_dialects"):
            m.register_dialects(get_dialect_registry())
        if hasattr(m, "context_init_hook"):
            post_init_hooks.append(m.context_init_hook)
        if hasattr(m, "disable_multithreading"):
            if bool(m.disable_multithreading):
                disable_multithreading = True

    # Access submodules directly from _flir_ir instead of importlib.import_module.
    # This avoids needing sys.modules aliases for package-relative imports.
    init_module = getattr(_flir_ir, '_mlirRegisterEverything', None)
    if init_module is not None:
        process_initializer(init_module)

    flir_passes = getattr(_flir_ir, '_flirPasses', None)
    if flir_passes is not None:
        process_initializer(flir_passes)

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
