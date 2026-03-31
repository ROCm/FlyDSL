# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Scoped LLVM cl::opt control via context manager.

Usage::

    from flydsl.compiler.llvm_options import llvm_options

    with llvm_options({"enable-post-misched": False, "lsr-drop-solution": True}):
        # LLVM options active only within this block
        module = build_and_compile_kernel(...)
    # options automatically restored here
"""

from contextlib import contextmanager
from typing import Dict, Union


_fly_module = None


def _get_fly_module():
    global _fly_module
    if _fly_module is not None:
        return _fly_module
    try:
        from .._mlir._mlir_libs import _fly

        _fly_module = _fly
    except ImportError:
        from .._mlir._mlir_libs import _mlirDialectsFly

        _fly_module = _mlirDialectsFly
    return _fly_module


@contextmanager
def llvm_options(opts: Dict[str, Union[bool, int, str]]):
    """Temporarily set LLVM cl::opt values, restoring originals on exit.

    Args:
        opts: mapping from LLVM option name (without leading dashes) to value.
              Supported types: bool, int, str.
    """
    _fly = _get_fly_module()
    saved: list = []
    try:
        for name, value in opts.items():
            if isinstance(value, bool):
                old = _fly.set_llvm_option_bool(name, value)
                saved.append(("bool", name, old))
            elif isinstance(value, int):
                old = _fly.set_llvm_option_int(name, value)
                saved.append(("int", name, old))
            elif isinstance(value, str):
                old = _fly.set_llvm_option_str(name, value)
                saved.append(("str", name, old))
            else:
                raise TypeError(
                    f"Unsupported type {type(value).__name__} for LLVM option '{name}'; "
                    "use bool, int, or str"
                )
        yield
    finally:
        for kind, name, old in reversed(saved):
            if kind == "bool":
                _fly.set_llvm_option_bool(name, old)
            elif kind == "int":
                _fly.set_llvm_option_int(name, old)
            else:
                _fly.set_llvm_option_str(name, old)
