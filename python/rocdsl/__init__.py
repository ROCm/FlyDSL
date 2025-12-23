"""RocDSL - ROCm Domain Specific Language for layout algebra"""

__version__ = "0.1.0"

# Setup Python path for embedded MLIR modules
import sys
import os
import importlib
from pathlib import Path

# IMPORTANT:
# Do not blindly prepend build/python_packages/rocdsl to sys.path.
# That directory contains an embedded `_mlir` package which can conflict with an
# external MLIR python runtime (mlir_core), leading to crashes like:
#   LLVM ERROR: Option 'basic' already exists!
#
# If you explicitly want to use the embedded MLIR runtime, set:
#   ROCDSL_USE_EMBEDDED_MLIR=1
_rocdsl_root = Path(__file__).resolve().parents[2]
if os.environ.get("ROCDSL_USE_EMBEDDED_MLIR", "0") == "1":
    # Default build layout: `.rocdsl/build` (see build.sh/setup.py), with fallback
    # to legacy `build/`.
    _build_dir = os.environ.get("ROCDSL_BUILD_DIR")
    if _build_dir is None:
        _build_dir_path = _rocdsl_root / ".rocdsl" / "build"
        if not _build_dir_path.exists():
            _build_dir_path = _rocdsl_root / "build"
    else:
        _build_dir_path = Path(_build_dir)
        if not _build_dir_path.is_absolute():
            _build_dir_path = _rocdsl_root / _build_dir_path
    _python_packages_dir = _build_dir_path / "python_packages" / "rocdsl"
    if _python_packages_dir.exists():
        _python_packages_str = str(_python_packages_dir)
        if _python_packages_str not in sys.path:
            sys.path.insert(0, _python_packages_str)

# Lazy import dialects and passes to avoid requiring MLIR when only using runtime
def __getattr__(name):
    if name == "rocir":
        return importlib.import_module(".dialects.ext.rocir", __name__)
    elif name == "arith":
        return importlib.import_module(".dialects.ext.arith", __name__)
    elif name == "scf":
        return importlib.import_module(".dialects.ext.scf", __name__)
    elif name == "lang":
        return importlib.import_module(".lang", __name__)
    elif name in ["Pipeline", "run_pipeline", "lower_rocir_to_standard"]:
        from . import passes
        return getattr(passes, name)
    elif name == "compile":
        from .compiler.compiler import compile
        return compile
    elif name == "Executor":
        from .compiler.executor import Executor
        return Executor

__all__ = [
    "rocir",
    "arith",
    "scf",
    "lang",
    "Pipeline",
    "run_pipeline",
    "lower_rocir_to_standard",
    "compile",
    "Executor",
]

# Export compiler modules (safe imports only).
from .compiler import Pipeline, run_pipeline
from .compiler.context import RAIIMLIRContextModule

__all__.extend(["Pipeline", "run_pipeline", "RAIIMLIRContextModule", "compile", "Executor"])
