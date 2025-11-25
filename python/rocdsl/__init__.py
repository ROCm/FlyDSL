"""RocDSL - ROCm Domain Specific Language for CuTe Layout Algebra"""

__version__ = "0.1.0"

# Register CuTe passes on import
import sys
import os
import warnings

def _register_cute_passes():
    """Register CuTe passes by importing the extension module."""
    try:
        # Find the Python bindings directory
        rocdsl_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        python_bindings_dir = os.path.join(rocdsl_root, "build/python_bindings")
        
        # Add to Python path if not already there
        if python_bindings_dir not in sys.path:
            sys.path.insert(0, python_bindings_dir)
        
        # Import the pass registration module and dialect module
        try:
            import _cutePassesExt
            import _cuteDialect  # Simple stub module for now
            
            # Register passes
            _cutePassesExt.register_passes()
        except ImportError as e:
            warnings.warn(f"CuTe passes extension module not found: {e}. Passes will not be available.")
            
    except Exception as e:
        warnings.warn(f"Failed to register CuTe passes: {e}")

# Register passes before importing other modules
_register_cute_passes()

from .dialects.ext import cute, arith, scf
from .passes import (
    Pipeline,
    run_pipeline,
    lower_cute_to_standard,
    lower_cute_to_nvgpu,
    optimize_layouts,
)

__all__ = [
    "cute",
    "arith",
    "scf",
    "Pipeline",
    "run_pipeline",
    "lower_cute_to_standard",
    "lower_cute_to_nvgpu",
    "optimize_layouts",
]
