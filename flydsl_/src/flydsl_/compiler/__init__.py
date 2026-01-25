from .context import *
from .pipeline import *

# NOTE: `compile` / `Executor` are intentionally NOT imported at module import time:
# some environments do not ship `_mlir.execution_engine` bindings. We keep FLIR
# importable and surface a clear error only when the feature is used.

def compile(*args, **kwargs):
    from .compiler import compile as _compile
    return _compile(*args, **kwargs)

def Executor(*args, **kwargs):  # type: ignore
    from .executor import Executor as _Executor
    return _Executor(*args, **kwargs)

__all__ = [name for name in globals().keys() if not name.startswith("_")]
