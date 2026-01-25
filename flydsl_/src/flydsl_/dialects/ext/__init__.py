"""Extended dialect wrappers for more Pythonic MLIR programming.

This package is intentionally **lazy-imported** to keep import side effects small.
Several wrapper modules register helpers with the embedded MLIR runtime and some
environments are sensitive to heavy import-time registration.
"""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULES = {
    "arith",
    "scf",
    "flir",
    "gpu",
    "func",
    "rocdl",
    "memref",
    "vector",
    "math",
    "llvm",
    "buffer_ops",
    "block_reduce_ops",
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(name)


__all__ = sorted(list(_SUBMODULES))
