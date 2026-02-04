"""
FlyDSL Arith Dialect Wrapper

This module wraps the upstream mlir.dialects.arith module to work with flydsl's Context.
Automatically converts flydsl types to upstream types when calling arith operations.
"""

from ..interop import to_upstream

# Import the upstream Arith dialect
try:
    from mlir.dialects import arith as _upstream_arith
    
    # Wrap the constant function to auto-convert types
    def constant(result_or_value, value=None, *, loc=None, ip=None):
        """
        Create an arith.constant operation.
        
        Automatically converts flydsl types to upstream mlir types.
        """
        if value is None:
            # Single argument form: constant(value) - infer type
            return _upstream_arith.constant(result_or_value, loc=loc, ip=ip)
        else:
            # Two argument form: constant(type, value)
            upstream_type = to_upstream(result_or_value)
            return _upstream_arith.constant(upstream_type, value, loc=loc, ip=ip)
    
    # Wrapper for index_cast
    def index_cast(result, input, *, loc=None, ip=None):
        """Create an arith.index_cast operation with auto type conversion."""
        upstream_result = to_upstream(result)
        upstream_input = to_upstream(input)
        return _upstream_arith.index_cast(upstream_result, upstream_input, loc=loc, ip=ip)
    
    # Wrapper for ceildivsi
    def ceildivsi(lhs, rhs, *, loc=None, ip=None):
        """Create an arith.ceildivsi operation."""
        upstream_lhs = to_upstream(lhs)
        upstream_rhs = to_upstream(rhs)
        return _upstream_arith.ceildivsi(upstream_lhs, upstream_rhs, loc=loc, ip=ip)
    
    # Re-export all arith operations and utilities
    import sys
    _current_module = sys.modules[__name__]
    for name in dir(_upstream_arith):
        if not name.startswith('_') and name not in ('constant', 'index_cast', 'ceildivsi'):
            setattr(_current_module, name, getattr(_upstream_arith, name))
    
    # Wrapper for IndexCastOp class
    class IndexCastOp:
        """Wrapper for arith.IndexCastOp with auto type conversion."""
        def __new__(cls, result_type, operand, *, loc=None, ip=None):
            upstream_result_type = to_upstream(result_type)
            upstream_operand = to_upstream(operand)
            return _upstream_arith.IndexCastOp(upstream_result_type, upstream_operand, loc=loc, ip=ip)
    
    # Commonly used operations (direct re-export)
    AddFOp = _upstream_arith.AddFOp
    AddIOp = _upstream_arith.AddIOp
    MulFOp = _upstream_arith.MulFOp
    MulIOp = _upstream_arith.MulIOp
    ConstantOp = _upstream_arith.ConstantOp
    _UpstreamIndexCastOp = _upstream_arith.IndexCastOp  # Keep original if needed
    
except ImportError as e:
    raise ImportError(
        f"Failed to import upstream Arith dialect. "
        f"Make sure mlir.dialects.arith is available. Error: {e}"
    )

__all__ = [
    'constant', 'index_cast', 'ceildivsi',
    'AddFOp', 'AddIOp', 'MulFOp', 'MulIOp', 'ConstantOp', 'IndexCastOp',
]
