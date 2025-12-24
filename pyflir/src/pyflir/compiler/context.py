import contextlib
import importlib
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from _mlir import ir

_PASSES_MODULE = None


def ensure_flir_python_extensions(context: ir.Context):
    """Ensure Flir passes and dialect are registered for the given context."""
    global _PASSES_MODULE
    if _PASSES_MODULE is None:
        # Prefer loading as a submodule of the active `_mlir` runtime.
        # If that is not available (common in dev builds), fall back to importing
        # the extension directly from our build tree without shadowing `_mlir`.
        try:
            from _mlir._mlir_libs import _flirPasses as _PASSES_MODULE
        except Exception:
            flir_root = Path(__file__).resolve().parents[3]
            ext_dir = flir_root / "build" / "python_packages" / "flir" / "_mlir" / "_mlir_libs"
            if ext_dir.exists():
                ext_dir_str = str(ext_dir)
                if ext_dir_str not in sys.path:
                    sys.path.insert(0, ext_dir_str)
            import _flirPasses as _PASSES_MODULE
    
    # Register dialects using the new nanobind interface
    from _mlir import ir as mlir_ir
    dialect_registry = mlir_ir.DialectRegistry()
    _PASSES_MODULE.register_dialects(dialect_registry._CAPIPtr)
    context.append_dialect_registry(dialect_registry)

    # Register LLVM IR translations (required for gpu-module-to-binary, etc).
    _PASSES_MODULE.register_llvm_translations(context._CAPIPtr)

    # Load all available dialects so op/type registration is available for
    # building IR (e.g. arith.constant) and parsing textual IR (e.g. func.func).
    # Without this, MLIR will report "operation was not registered".
    context.load_all_available_dialects()


@dataclass
class MLIRContext:
    context: ir.Context
    module: ir.Module

    def __str__(self):
        return str(self.module)


class RAIIMLIRContext:
    context: ir.Context
    location: ir.Location

    def __init__(self, location: Optional[ir.Location] = None, allow_unregistered_dialects=False):
        self.context = ir.Context()
        if allow_unregistered_dialects:
            self.context.allow_unregistered_dialects = True
        self.context.__enter__()
        ensure_flir_python_extensions(self.context)
        if location is None:
            location = ir.Location.unknown()
        self.location = location
        self.location.__enter__()

    def __del__(self):  # pragma: no cover
        # Avoid calling into MLIR python bindings during interpreter shutdown.
        # Some environments finalize without the GIL held, which can abort the process.
        try:
            if getattr(sys, "is_finalizing", lambda: False)():
                return
        except Exception:
            return

        try:
            self.location.__exit__(None, None, None)
        except Exception:
            pass
        try:
            self.context.__exit__(None, None, None)
        except Exception:
            pass
        if ir is not None:
            assert ir.Context is not self.context


class RAIIMLIRContextModule:
    context: ir.Context
    location: ir.Location
    insertion_point: ir.InsertionPoint
    module: ir.Module

    def __init__(self, location: Optional[ir.Location] = None, allow_unregistered_dialects=False):
        self.context = ir.Context()
        if allow_unregistered_dialects:
            self.context.allow_unregistered_dialects = True
        self.context.__enter__()
        ensure_flir_python_extensions(self.context)
        if location is None:
            location = ir.Location.unknown()
        self.location = location
        self.location.__enter__()
        self.module = ir.Module.create()
        self.insertion_point = ir.InsertionPoint(self.module.body)
        self.insertion_point.__enter__()

    def __del__(self):  # pragma: no cover
        # Avoid calling into MLIR python bindings during interpreter shutdown.
        # Some environments finalize without the GIL held, which can abort the process.
        try:
            if getattr(sys, "is_finalizing", lambda: False)():
                return
        except Exception:
            return

        try:
            self.insertion_point.__exit__(None, None, None)
        except Exception:
            pass
        try:
            self.location.__exit__(None, None, None)
        except Exception:
            pass
        try:
            self.context.__exit__(None, None, None)
        except Exception:
            pass
        if ir is not None:
            assert ir.Context is not self.context


class ExplicitlyManagedModule:
    module: ir.Module
    _ip: ir.InsertionPoint

    def __init__(self):
        self.module = ir.Module.create()
        self._ip = ir.InsertionPoint(self.module.body)
        self._ip.__enter__()

    def finish(self):
        self._ip.__exit__(None, None, None)
        return self.module

    def __str__(self):
        return str(self.module)


@contextlib.contextmanager
def enable_multithreading(context=None):
    if context is None:
        context = ir.Context.current
    context.enable_multithreading(True)
    yield
    context.enable_multithreading(False)


@contextlib.contextmanager
def disable_multithreading(context=None):
    if context is None:
        context = ir.Context.current

    context.enable_multithreading(False)
    yield
    context.enable_multithreading(True)


@contextlib.contextmanager
def enable_debug():
    ir._GlobalDebug.flag = True
    yield
    ir._GlobalDebug.flag = False
