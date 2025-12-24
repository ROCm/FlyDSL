"""Flir Dialect for MLIR Python Bindings"""

from _mlir import ir
from _mlir.dialects._ods_common import _cext as _ods_cext

# Import generated ops
from _flir_ops_gen import *
from _flir_ops_gen import _Dialect

# Import generated enums
try:
    from _flir_enum_gen import *
except ImportError:
    pass

# Import generated ROCm ops
try:
    from _flir_rocm_ops_gen import *
except ImportError:
    pass

# Register the dialect when imported
try:
    # The dialect is automatically registered via @_ods_cext.register_dialect decorator
    # in _flir_ops_gen.py, but we need to ensure it's loaded in the context
    pass
except Exception as e:
    import warnings
    warnings.warn(f"Failed to register Flir dialect: {e}")

__all__ = []
