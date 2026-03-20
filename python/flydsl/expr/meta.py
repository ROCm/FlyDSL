# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import inspect
from functools import wraps

from .._mlir import ir


def _to_raw_value(obj):
    if isinstance(obj, ir.Value):
        return obj
    if isinstance(obj, type):
        return obj
    if hasattr(obj, "__fly_values__"):
        values = obj.__fly_values__()
        if len(values) != 1:
            raise ValueError(f"Primitive function expects 1 value, got {len(values)}")
        return values[0]
    if isinstance(obj, tuple):
        return tuple(_to_raw_value(e) for e in obj)
    if isinstance(obj, list):
        return [_to_raw_value(e) for e in obj]
    return obj


def _flatten_args(args, kwargs):
    new_args = tuple(_to_raw_value(a) for a in args)
    new_kwargs = {k: _to_raw_value(v) if k not in ("loc", "ip") else v for k, v in kwargs.items()}
    return new_args, new_kwargs


def _caller_location(depth=1):
    """Build an MLIR Location from the Python call-site *depth* frames up."""
    frame = inspect.currentframe()
    for _ in range(depth + 1):
        if frame is not None:
            frame = frame.f_back
    if frame is None:
        return ir.Location.unknown()

    info = inspect.getframeinfo(frame)
    pos = getattr(info, "positions", None)
    line = pos.lineno if pos is not None else info.lineno
    col = (pos.col_offset or 0) if pos is not None else 0
    file_loc = ir.Location.file(info.filename, line, col)

    if info.code_context:
        label = " ".join(ln.strip() for ln in info.code_context)
    else:
        label = info.function
    return ir.Location.name(label, childLoc=file_loc)


def traced_op(op):
    @wraps(op)
    def wrapper(*args, **kwargs):
        loc = kwargs.pop("loc", None)
        if loc is None:
            loc = _caller_location(depth=1)
        args, kwargs = _flatten_args(args, kwargs)
        with loc:
            return op(*args, **kwargs)

    return wrapper
