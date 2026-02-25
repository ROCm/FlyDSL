"""Backward-compat shim — canonical location is now flydsl.expr.typing."""

from flydsl.expr.typing import Types, T, default_f8_type  # noqa: F401

memref = None
try:
    from flydsl._mlir.extras.types import memref  # noqa: F401
except ImportError:
    pass

__all__ = ["Types", "T", "default_f8_type", "memref"]
