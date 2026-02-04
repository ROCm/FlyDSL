"""
FlyDSL Func Dialect Wrapper

This module wraps the upstream mlir.dialects.func module to work with flydsl's Context.
"""

from ..._mlir import ir

# Import the upstream Func dialect
try:
    from mlir.dialects import func as _upstream_func
    
    # Re-export all func operations and utilities
    import sys
    _current_module = sys.modules[__name__]
    for name in dir(_upstream_func):
        if not name.startswith('_'):
            setattr(_current_module, name, getattr(_upstream_func, name))
    
    # Commonly used operations
    FuncOp = _upstream_func.FuncOp
    CallOp = _upstream_func.CallOp
    ReturnOp = _upstream_func.ReturnOp
    
except ImportError as e:
    raise ImportError(
        f"Failed to import upstream Func dialect. "
        f"Make sure mlir.dialects.func is available. Error: {e}"
    )

__all__ = ['FuncOp', 'CallOp', 'ReturnOp']
