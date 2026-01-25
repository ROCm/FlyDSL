"""Vector dialect helpers with better default debug locations.

This module exists so tests can import vector ops through `flydsl.dialects.ext`
instead of directly importing from `_mlir.dialects.*`.
"""

from __future__ import annotations

from _mlir.dialects import vector as _vector

from ._loc import maybe_default_loc

# Re-export everything from the upstream dialect module for convenience.
from _mlir.dialects.vector import *  # noqa: F401,F403,E402


def from_elements(*args, loc=None, ip=None, **kwargs):
    # The upstream `vector.from_elements` expects each element to be an `ir.Value`.
    # In our codebase, scalar values may be auto-wrapped as `arith.ArithValue`.
    # Unwrap them here so call sites don't need to sprinkle `arith.as_value(...)`.
    from . import arith as _arith_ext

    if len(args) >= 2:
        args = list(args)
        elems = args[1]
        if isinstance(elems, (list, tuple)):
            args[1] = [_arith_ext.unwrap(v) for v in elems]
        return _vector.from_elements(*args, loc=maybe_default_loc(loc), ip=ip, **kwargs)

    return _vector.from_elements(*args, loc=maybe_default_loc(loc), ip=ip, **kwargs)


def load(memref, indices, *, loc=None, ip=None, **kwargs):
    """Vector load wrapper that accepts ArithValue/wrappers for indices."""
    from . import arith as _arith_ext

    return _vector.load(
        _arith_ext.unwrap(memref),
        [_arith_ext.unwrap(i, index=True, loc=loc) for i in indices],
        loc=maybe_default_loc(loc),
        ip=ip,
        **kwargs,
    )


def store(value, memref, indices, *, loc=None, ip=None, **kwargs):
    """Vector store wrapper that accepts ArithValue/wrappers for value/indices."""
    from . import arith as _arith_ext

    return _vector.store(
        _arith_ext.unwrap(value),
        _arith_ext.unwrap(memref),
        [_arith_ext.unwrap(i, index=True, loc=loc) for i in indices],
        loc=maybe_default_loc(loc),
        ip=ip,
        **kwargs,
    )


# -----------------------------------------------------------------------------
# Thin wrappers for common op classes that otherwise require `.result` access.
# -----------------------------------------------------------------------------


def extract(vector, static_position=None, dynamic_position=None, *, loc=None, ip=None):
    """Wrapper around `vector.ExtractOp(...).result`."""
    from . import arith as _arith_ext

    if static_position is None:
        static_position = []
    if dynamic_position is None:
        dynamic_position = []
    dynamic_position = [_arith_ext.unwrap(i, index=True, loc=loc) for i in dynamic_position]
    return _vector.ExtractOp(
        _arith_ext.unwrap(vector, loc=loc),
        static_position=static_position,
        dynamic_position=dynamic_position,
        loc=maybe_default_loc(loc),
        ip=ip,
    ).result


def load_op(result_type, memref, indices, *, loc=None, ip=None):
    """Wrapper around `vector.LoadOp(...).result`."""
    from . import arith as _arith_ext

    return _vector.LoadOp(
        result_type,
        _arith_ext.unwrap(memref),
        [_arith_ext.unwrap(i, index=True, loc=loc) for i in indices],
        loc=maybe_default_loc(loc),
        ip=ip,
    ).result


def transfer_read(result_type, source, indices, permutation_map, padding, in_bounds, *, loc=None, ip=None):
    """Wrapper around `vector.TransferReadOp(...).result`."""
    from . import arith as _arith_ext

    return _vector.TransferReadOp(
        result_type,
        _arith_ext.unwrap(source),
        [_arith_ext.unwrap(i, index=True, loc=loc) for i in indices],
        permutation_map,
        _arith_ext.unwrap(padding),
        in_bounds,
        loc=maybe_default_loc(loc),
        ip=ip,
    ).result


def bitcast(result_type, source, *, loc=None, ip=None):
    """Wrapper around `vector.BitCastOp(...).result`."""
    from . import arith as _arith_ext

    return _vector.BitCastOp(
        result_type,
        _arith_ext.unwrap(source, loc=loc),
        loc=maybe_default_loc(loc),
        ip=ip,
    ).result


def shuffle(v1, v2, mask, *, loc=None, ip=None):
    """Wrapper around `vector.ShuffleOp(...).result`."""
    from . import arith as _arith_ext

    return _vector.ShuffleOp(
        _arith_ext.unwrap(v1, loc=loc),
        _arith_ext.unwrap(v2, loc=loc),
        mask,
        loc=maybe_default_loc(loc),
        ip=ip,
    ).result


def broadcast(result_type, source, *, loc=None, ip=None):
    """Wrapper around `vector.BroadcastOp(...).result`."""
    from . import arith as _arith_ext

    return _vector.BroadcastOp(
        result_type,
        _arith_ext.unwrap(source, loc=loc),
        loc=maybe_default_loc(loc),
        ip=ip,
    ).result


