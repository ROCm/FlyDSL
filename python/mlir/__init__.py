"""
Compatibility shim: expose the embedded MLIR Python bindings shipped under the
`_mlir` package as the conventional `mlir` package API.

This repo's embedded MLIR build installs bindings under `_mlir` (including
`_mlir.ir`, `_mlir.dialects`, ...). Many tests/import sites use `mlir.*`.
We bridge that here to avoid mixing two different MLIR runtimes in one process.
"""

from __future__ import annotations

import importlib
import sys


def _alias(name: str, target: str):
    mod = importlib.import_module(target)
    sys.modules[name] = mod
    return mod


# Core modules.
ir = _alias("mlir.ir", "_mlir.ir")
passmanager = _alias("mlir.passmanager", "_mlir.passmanager")
rewrite = _alias("mlir.rewrite", "_mlir.rewrite")

# Optional modules (may not be present in the embedded package layout).
try:
    execution_engine = _alias("mlir.execution_engine", "_mlir.execution_engine")
except ModuleNotFoundError:
    execution_engine = None

# Packages.
_mlir_libs = _alias("mlir._mlir_libs", "_mlir._mlir_libs")
dialects = _alias("mlir.dialects", "_mlir.dialects")
extras = _alias("mlir.extras", "_mlir.extras")
runtime = _alias("mlir.runtime", "_mlir.runtime")

# Eagerly alias common dialect modules to avoid importing generated dialect
# bindings twice under different module names (can crash with
# "Dialect namespace 'X' is already registered.").
for _d in ("arith", "scf", "gpu", "func", "memref", "math", "vector", "llvm", "rocdl", "rocir"):
    try:
        _alias(f"mlir.dialects.{_d}", f"_mlir.dialects.{_d}")
    except Exception:
        pass

# Common extras helpers.
try:
    _alias("mlir.extras.types", "_mlir.extras.types")
except Exception:
    pass


