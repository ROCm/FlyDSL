# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Scoped LLVM cl::opt control via context manager."""

import warnings
from contextlib import contextmanager
from typing import Dict, Union

_fly_module = None
_UNKNOWN_OPTION_PREFIX = "Unknown LLVM option"


def _get_fly_module():
    global _fly_module
    if _fly_module is None:
        from .._mlir._mlir_libs import _mlirDialectsFly

        _fly_module = _mlirDialectsFly
    return _fly_module


def _warn_unknown(name: str, err: RuntimeError) -> bool:
    """Return True (and warn) if err means the LLVM option is not registered."""
    if str(err).startswith(_UNKNOWN_OPTION_PREFIX):
        warnings.warn(
            f"{err}; ignoring it. The option may have been moved to TableGen "
            f"(not a cl::opt) in the LLVM version in use.",
            RuntimeWarning,
            stacklevel=3,
        )
        return True
    return False


@contextmanager
def llvm_options(opts: Dict[str, Union[bool, int, str]]):
    """Temporarily set LLVM cl::opt values, restoring originals on exit.

    Options that are not registered cl::opts in the linked LLVM are skipped
    with a RuntimeWarning instead of raising.
    """
    _fly = _get_fly_module()
    saved: list = []
    try:
        for name, value in opts.items():
            if isinstance(value, bool):
                setter, kind = _fly.set_llvm_option_bool, "bool"
            elif isinstance(value, int):
                setter, kind = _fly.set_llvm_option_int, "int"
            elif isinstance(value, str):
                setter, kind = _fly.set_llvm_option_str, "str"
            else:
                raise TypeError(
                    f"Unsupported type {type(value).__name__} for LLVM option '{name}'; " "use bool, int, or str"
                )
            try:
                old = setter(name, value)
            except RuntimeError as err:
                if _warn_unknown(name, err):
                    continue
                raise
            saved.append((kind, name, old))
        yield
    finally:
        for kind, name, old in reversed(saved):
            if kind == "bool":
                _fly.set_llvm_option_bool(name, old)
            elif kind == "int":
                _fly.set_llvm_option_int(name, old)
            else:
                _fly.set_llvm_option_str(name, old)
