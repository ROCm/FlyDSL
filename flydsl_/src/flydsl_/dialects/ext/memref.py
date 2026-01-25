"""MemRef dialect helpers with better default debug locations.

When IR dumping is enabled, it is useful for common ops like `memref.load/store/view`
to carry the user call-site location instead of inheriting a single outer
`Location.current` for a whole block.
"""

from __future__ import annotations

from typing import Any, Sequence

from _mlir.dialects import memref as _memref

from ._loc import maybe_default_loc


# Re-export everything from the upstream dialect module for convenience.
from _mlir.dialects.memref import *  # noqa: F401,F403,E402


def load(memref: Any, indices: Sequence[Any], *, loc=None, ip=None):
    # Accept ArithValue / wrappers / python ints for indices.
    from . import arith as _arith_ext

    return _memref.load(
        _arith_ext.unwrap(memref),
        [_arith_ext.unwrap(i, index=True, loc=loc) for i in indices],
        loc=maybe_default_loc(loc),
        ip=ip,
    )


def store(value: Any, memref: Any, indices: Sequence[Any], *, loc=None, ip=None):
    # Accept ArithValue / wrappers / python ints for value and indices.
    from . import arith as _arith_ext

    return _memref.store(
        _arith_ext.unwrap(value),
        _arith_ext.unwrap(memref),
        [_arith_ext.unwrap(i, index=True, loc=loc) for i in indices],
        loc=maybe_default_loc(loc),
        ip=ip,
    )


def view(source: Any, byte_shift: Any, sizes: Sequence[Any], *, loc=None, ip=None):
    return _memref.view(source, byte_shift, sizes, loc=maybe_default_loc(loc), ip=ip)


def get_global(*args, loc=None, ip=None, **kwargs):
    # Signature differs across bindings; keep flexible.
    return _memref.get_global(*args, loc=maybe_default_loc(loc), ip=ip, **kwargs)


