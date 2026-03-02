import inspect
from functools import wraps

from .._mlir import ir


def _to_raw_value(obj):
    if isinstance(obj, ir.Value):
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


def traced_op(op):
    @wraps(op)
    def wrapper(*args, **kwargs):
        loc = kwargs.pop("loc", None)
        if loc is None:
            frame = inspect.currentframe().f_back
            frameInfo = inspect.getframeinfo(frame)
            if hasattr(frameInfo, "positions") and frameInfo.positions is not None:
                line = frameInfo.positions.lineno
                col = frameInfo.positions.col_offset or 0
            else:
                line = frameInfo.lineno
                col = 0
            file_loc = ir.Location.file(frameInfo.filename, line, col)

            loc = ir.Location.name(
                (
                    "".join([c.strip() for c in frameInfo.code_context])
                    if frameInfo.code_context
                    else frameInfo.function
                ),
                childLoc=file_loc,
            )
        args, kwargs = _flatten_args(args, kwargs)
        with loc:
            return op(*args, **kwargs)

    return wrapper
