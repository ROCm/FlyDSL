__version__ = "0.1.0"

import importlib as _importlib
import sys as _sys

# Register flydsl._mlir as the canonical ``_mlir`` so that code using
# ``from _mlir import ir`` (the old convention) gets the SAME C++ extension
# as ``from flydsl._mlir import ir`` (the new convention).  This prevents
# double-initialization of the MLIR PyGlobals singleton.
if "_mlir" not in _sys.modules:
    try:
        _pkg = _importlib.import_module("flydsl._mlir")
        _sys.modules["_mlir"] = _pkg
        # Eagerly import the core sub-modules so any later
        # ``from _mlir.ir import ...`` finds the already-loaded module
        # instead of triggering a second C-extension init.
        for _sub in ("ir", "passmanager", "execution_engine", "rewrite"):
            try:
                _m = _importlib.import_module(f"flydsl._mlir.{_sub}")
                _sys.modules[f"_mlir.{_sub}"] = _m
            except ImportError:
                pass
        # Alias the native extension libraries directory
        try:
            _libs = _importlib.import_module("flydsl._mlir._mlir_libs")
            _sys.modules["_mlir._mlir_libs"] = _libs
            # The critical C++ singleton lives here
            _core = _importlib.import_module("flydsl._mlir._mlir_libs._mlir")
            _sys.modules["_mlir._mlir_libs._mlir"] = _core
        except ImportError:
            pass
        # Alias dialects and extras
        for _sub in ("dialects", "extras"):
            try:
                _m = _importlib.import_module(f"flydsl._mlir.{_sub}")
                _sys.modules[f"_mlir.{_sub}"] = _m
            except ImportError:
                pass
    except ImportError:
        pass
