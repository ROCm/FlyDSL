import inspect
from functools import wraps

from .._mlir import ir


def _unwrap_dsl_value(obj):
    if isinstance(obj, ir.Value):
        return obj
    if hasattr(obj, "__extract_ir_values__"):
        values = obj.__extract_ir_values__()
        if len(values) != 1:
            raise ValueError(f"Primitive function expects 1 value, got {len(values)}")
        return values[0]
    if isinstance(obj, tuple):
        return tuple(_unwrap_dsl_value(e) for e in obj)
    if isinstance(obj, list):
        return [_unwrap_dsl_value(e) for e in obj]
    return obj


def _unwrap_args(args, kwargs):
    new_args = tuple(_unwrap_dsl_value(a) for a in args)
    new_kwargs = {k: _unwrap_dsl_value(v) if k not in ("loc", "ip") else v for k, v in kwargs.items()}
    return new_args, new_kwargs


def dsl_api_wrapper(op):
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
        args, kwargs = _unwrap_args(args, kwargs)
        with loc:
            return op(*args, **kwargs)

    return wrapper
