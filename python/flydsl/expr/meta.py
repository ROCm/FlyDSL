import inspect
from functools import wraps

from .._mlir import ir


def dsl_api_wrapper(op):
    @wraps(op)
    def wrapper(*args, **kwargs):
        loc = kwargs.pop("loc", None)
        if loc is None:
            frame = inspect.currentframe().f_back
            frameInfo = inspect.getframeinfo(frame)
            file_loc = ir.Location.file(
                frameInfo.filename,
                frameInfo.positions.lineno,
                frameInfo.positions.col_offset,
            )
            loc = ir.Location.name(
                (
                    "".join([c.strip() for c in frameInfo.code_context])
                    if frameInfo.code_context
                    else frameInfo.function
                ),
                childLoc=file_loc,
            )
        with loc:
            return op(*args, **kwargs)

    return wrapper
