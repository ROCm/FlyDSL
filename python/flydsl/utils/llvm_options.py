"""Context manager for scoped LLVM cl::opt control.

Usage::

    from flydsl.utils.llvm_options import llvm_options

    with llvm_options({"enable-post-misched": False, "lsr-drop-solution": True}):
        # LLVM options active only within this block
        module = build_and_compile_kernel(...)
    # options automatically restored here
"""

from contextlib import contextmanager
from typing import Any, Dict, Union


def _get_fly_module():
    from .._mlir._mlir_libs import _fly
    return _fly


@contextmanager
def llvm_options(opts: Dict[str, Union[bool, int]]):
    """Temporarily set LLVM cl::opt values, restoring originals on exit.

    Args:
        opts: mapping from LLVM option name (without leading dashes) to value.
              Supported types: bool, int.

    Example::

        with llvm_options({
            "enable-post-misched": False,
            "lsr-drop-solution": True,
        }):
            exe(Q, K, V, O, B, S)
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
            else:
                raise TypeError(
                    f"Unsupported type {type(value).__name__} for LLVM option '{name}'; "
                    "use bool or int"
                )
        yield
    finally:
        for kind, name, old in reversed(saved):
            if kind == "bool":
                _fly.restore_llvm_option_bool(name, old)
            else:
                _fly.restore_llvm_option_int(name, old)
