"""RocDSL dialects"""

import sys
import os

# Add build directory to Python path for generated bindings
_build_dir = os.path.join(os.path.dirname(__file__), '../../../build/python_bindings')
if os.path.exists(_build_dir):
    sys.path.insert(0, os.path.abspath(_build_dir))

# Import the rocir dialect
try:
    from rocir import *
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import rocir dialect: {e}")
