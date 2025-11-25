"""CuTe Dialect for MLIR Python Bindings"""

from mlir import ir
from mlir.dialects._ods_common import _cext as _ods_cext

# Import generated ops
from ._cute_ops_gen import *
from ._cute_ops_gen import _Dialect

# Register the dialect and passes when imported
try:
    # Import the C++ extension module
    from . import _cuteDialect
    
    # Register CuTe passes
    _cuteDialect.cute.register_passes()
    
except Exception as e:
    import warnings
    warnings.warn(f"Failed to register CuTe passes: {e}")

__all__ = []
