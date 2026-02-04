"""
FlyDSL ODS Common utilities

This module provides ODS (Operation Definition Specification) utilities
for the FlyDSL dialect.

Uses upstream mlir's _mlir for base classes but registers fly-specific types.
"""

# Use upstream mlir._mlir_libs._mlir for base classes (OpView, etc.)
from mlir._mlir_libs import _mlir as _cext

# Re-export common utilities from upstream mlir.dialects._ods_common
from mlir.dialects._ods_common import (
    equally_sized_accessor,
    get_default_loc_context as _upstream_get_default_loc_context,
    get_op_results_or_values,
    segmented_accessor,
    get_op_result_or_op_results,
)

# Custom get_default_loc_context that uses both contexts
def get_default_loc_context(loc):
    """Get default context, supporting both upstream and flydsl contexts."""
    result = _upstream_get_default_loc_context(loc)
    # If upstream returns None, the InsertionPoint.current context will be used
    return result

__all__ = [
    '_cext',
    'equally_sized_accessor',
    'get_default_loc_context', 
    'get_op_results_or_values',
    'segmented_accessor',
    'get_op_result_or_op_results',
]
