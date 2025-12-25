"""Vector dialect helpers with better default debug locations.

This module exists so tests can import vector ops through `pyflir.dialects.ext`
instead of directly importing from `_mlir.dialects.*`.
"""

from __future__ import annotations

from _mlir.dialects import vector as _vector

from ._loc import maybe_default_loc

# Re-export everything from the upstream dialect module for convenience.
from _mlir.dialects.vector import *  # noqa: F401,F403,E402


def from_elements(*args, loc=None, ip=None, **kwargs):
    return _vector.from_elements(*args, loc=maybe_default_loc(loc), ip=ip, **kwargs)


def load(*args, loc=None, ip=None, **kwargs):
    return _vector.load(*args, loc=maybe_default_loc(loc), ip=ip, **kwargs)


def store(*args, loc=None, ip=None, **kwargs):
    return _vector.store(*args, loc=maybe_default_loc(loc), ip=ip, **kwargs)


