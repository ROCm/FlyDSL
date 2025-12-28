"""FLIR - ROCm Domain Specific Language for layout algebra"""

__version__ = "0.1.0"

# Setup Python path for embedded MLIR modules
import sys
import os
import importlib
from pathlib import Path
from pkgutil import extend_path

# Development convenience:
# In this repo we often have *two* copies of the `pyflir` package:
# - source tree:      `pyflir/src/pyflir`
# - build output:     `.flir/build/python_packages/pyflir/pyflir`
#
# Most test runs end up importing the build-output package because
# `.flir/build/python_packages/pyflir` is on `sys.path`. That makes edits to the
# source tree appear to "not work".
#
# To make `pyflir/src/...` edits take effect immediately (without rebuilding),
# we treat `pyflir` as a multi-path package and *prefer* the source tree for
# submodules when it exists.
__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_this = Path(__file__).resolve()
for _p in _this.parents:
    _src_pkg = _p / "pyflir" / "src" / "pyflir"
    if _src_pkg.is_dir():
        _src_pkg_str = str(_src_pkg)
        if _src_pkg_str not in __path__:  # type: ignore[operator]
            __path__.insert(0, _src_pkg_str)  # type: ignore[operator]
        break

# IMPORTANT:
# Do not blindly prepend build/python_packages/pyflir to sys.path.
# That directory contains an embedded `_mlir` package which can conflict with an
# external MLIR python runtime (mlir_core), leading to crashes like:
#   LLVM ERROR: Option 'basic' already exists!
#
# If you explicitly want to use the embedded MLIR runtime, set:
#   FLIR_USE_EMBEDDED_MLIR=1
# For backward compatibility (pre-rename), we also honor:
#   FLIR_USE_EMBEDDED_MLIR=1
_this_file = Path(__file__).resolve()
# Determine FLIR repo root robustly for both:
# - source tree import:    <root>/pyflir/src/pyflir/__init__.py
# - build output import:   <root>/.flir/build/python_packages/pyflir/pyflir/__init__.py
_flir_root = None
for _p in _this_file.parents:
    if (_p / "tests").is_dir() and (_p / "pyflir").is_dir():
        _flir_root = _p
        break
if _flir_root is None:
    # Fallback to previous heuristic.
    _flir_root = _this_file.parents[2]
_use_embedded = os.environ.get("FLIR_USE_EMBEDDED_MLIR")
if _use_embedded is None:
    _use_embedded = os.environ.get("FLIR_USE_EMBEDDED_MLIR", "0")

# Convenience for direct `python tests/...` runs inside the FLIR source tree:
# If an embedded build exists, default to using it to avoid importing the legacy
# in-tree `_mlir` shim under `<repo>/python/_mlir` (which does not ship `_flirPasses`).
if _use_embedded != "1" and os.environ.get("FLIR_AUTO_EMBEDDED", "1") != "0":
    try:
        _argv0 = Path(sys.argv[0]).resolve()
    except Exception:
        _argv0 = None
    _embedded_default = _flir_root / ".flir" / "build" / "python_packages" / "pyflir"
    if not _embedded_default.exists():
        _embedded_default = _flir_root / "build" / "python_packages" / "pyflir"
    if _embedded_default.exists() and _argv0 is not None:
        try:
            # Only auto-enable for scripts executed from within this repo (usually tests/).
            if _flir_root in _argv0.parents and "tests" in _argv0.parts:
                _use_embedded = "1"
        except Exception:
            pass

if _use_embedded == "1":
    # Default build layout: `.flir/build` (see build.sh/setup.py), with fallback
    # to legacy `build/`.
    _build_dir = os.environ.get("FLIR_BUILD_DIR") or os.environ.get("FLIR_BUILD_DIR")
    if _build_dir is None:
        _build_dir_path = _flir_root / ".flir" / "build"
        if not _build_dir_path.exists():
            _build_dir_path = _flir_root / "build"
    else:
        _build_dir_path = Path(_build_dir)
        if not _build_dir_path.is_absolute():
            _build_dir_path = _flir_root / _build_dir_path
    # New layout: python_packages/pyflir (and legacy python_packages/flir, python_packages/rocdsl).
    _python_packages_dir = _build_dir_path / "python_packages" / "pyflir"
    if not _python_packages_dir.exists():
        _python_packages_dir = _build_dir_path / "python_packages" / "flir"
    if not _python_packages_dir.exists():
        _python_packages_dir = _build_dir_path / "python_packages" / "rocdsl"
    if _python_packages_dir.exists():
        _python_packages_str = str(_python_packages_dir)
        if _python_packages_str not in sys.path:
            sys.path.insert(0, _python_packages_str)
        # If we're using embedded MLIR, ensure the legacy in-tree `python/` path
        # can't shadow the embedded `_mlir` runtime.
        _legacy_python = str((_flir_root / "python").resolve())
        if _legacy_python in sys.path:
            sys.path.remove(_legacy_python)
        # Also drop any other `_mlir` providers that may have been injected earlier.
        _cleaned = []
        for _p in sys.path:
            try:
                if (Path(_p) / "_mlir").exists() and _p != _python_packages_str:
                    continue
            except Exception:
                pass
            _cleaned.append(_p)
        sys.path[:] = _cleaned

# Lazy import dialects and passes to avoid requiring MLIR when only using runtime
def __getattr__(name):
    if name == "flir":
        return importlib.import_module(".dialects.ext.flir", __name__)
    elif name == "arith":
        return importlib.import_module(".dialects.ext.arith", __name__)
    elif name == "scf":
        return importlib.import_module(".dialects.ext.scf", __name__)
    elif name == "lang":
        return importlib.import_module(".lang", __name__)
    elif name in ["Pipeline", "run_pipeline", "lower_flir_to_standard"]:
        from . import passes
        return getattr(passes, name)
    elif name == "compile":
        from .compiler.compiler import compile
        return compile
    elif name == "Executor":
        from .compiler.executor import Executor
        return Executor

__all__ = [
    "flir",
    "arith",
    "scf",
    "lang",
    "Pipeline",
    "run_pipeline",
    "lower_flir_to_standard",
    "compile",
    "Executor",
]

# Export compiler modules (safe imports only).
from .compiler import Pipeline, run_pipeline
from .compiler.context import RAIIMLIRContextModule

__all__.extend(["Pipeline", "run_pipeline", "RAIIMLIRContextModule", "compile", "Executor"])
