"""FLIR - ROCm Domain Specific Language for layout algebra"""

# Base version - git commit is appended at build time by setup.py
# Format: "0.0.1.dev" or "0.0.1.dev+abc1234" (with git commit)
__version__ = "0.0.1.dev"

# Setup Python path for embedded MLIR modules
import sys
import os
import importlib
from pathlib import Path
from pkgutil import extend_path

# Development convenience:
# In this repo we often have *two* copies of the `flydsl` package:
# - source tree:      `flydsl/src/flydsl`
# - build output:     `.flir/build/python_packages/flydsl/flydsl`
#
# Most test runs end up importing the build-output package because
# `.flir/build/python_packages/flydsl` is on `sys.path`. That makes edits to the
# source tree appear to "not work".
#
# To make `flydsl/src/...` edits take effect immediately (without rebuilding),
# we treat `flydsl` as a multi-path package and *prefer* the source tree for
# submodules when it exists.
__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_this = Path(__file__).resolve()
_repo_root = None
for _p in _this.parents:
    _src_pkg = _p / "flydsl" / "src" / "flydsl"
    if _src_pkg.is_dir():
        _repo_root = _p
        _src_pkg_str = str(_src_pkg)
        if _src_pkg_str not in __path__:  # type: ignore[operator]
            __path__.insert(0, _src_pkg_str)  # type: ignore[operator]
        break
if _repo_root is None:
    # Fallback: best-effort guess (keeps installed/wheel layouts working).
    _repo_root = _this.parents[2]

_flir_root = _repo_root


def _ensure_build_output_on_path() -> None:
    """Add the build output directory to sys.path so flydsl._mlir can be imported.

    The build produces `flydsl/_mlir/_mlir_libs/_flir_ir.so` under
    `.flir/build/python_packages/flydsl/`. This function prepends that directory
    to sys.path so that the `flydsl._mlir` sub-package can be resolved.
    """
    _build_dir = os.environ.get("FLIR_BUILD_DIR")
    if _build_dir is None:
        _build_dir_path = _flir_root / ".flir" / "build"
        if not _build_dir_path.exists():
            _build_dir_path = _flir_root / "build"
    else:
        _build_dir_path = Path(_build_dir)
        if not _build_dir_path.is_absolute():
            _build_dir_path = _flir_root / _build_dir_path

    for _name in ("flydsl", "flir"):
        _p = _build_dir_path / "python_packages" / _name
        if _p.exists():
            _p_str = str(_p)
            if _p_str not in sys.path:
                sys.path.insert(0, _p_str)
            return


_ensure_build_output_on_path()

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
