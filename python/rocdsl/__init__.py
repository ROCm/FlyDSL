"""RocDSL - ROCm Domain Specific Language for layout algebra"""

__version__ = "0.1.0"

# Register Rocir passes on import
import sys
import os
import warnings

def _register_rocir_passes():
    """Register Rocir passes by importing the extension module."""
    try:
        # Find the Python bindings directory
        rocdsl_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        python_bindings_dir = os.path.join(rocdsl_root, "build/python_bindings")
        
        # Add to Python path if not already there
        if python_bindings_dir not in sys.path:
            sys.path.insert(0, python_bindings_dir)
        
        # Import the pass registration module and dialect module
        try:
            import _rocirPassesExt
            import _rocirDialect  # Simple stub module for now
            
            # Register passes
            _rocirPassesExt.register_passes()
        except ImportError as e:
            warnings.warn(f"Rocir passes extension module not found: {e}. Passes will not be available.")
            
    except Exception as e:
        warnings.warn(f"Failed to register Rocir passes: {e}")

# Register passes before importing other modules
# _register_rocir_passes()  # DISABLED: breaks MLIR context

# Lazy import dialects and passes to avoid requiring MLIR when only using runtime
def __getattr__(name):
    if name == "rocir":
        from .dialects.ext import rocir
        return rocir
    elif name == "arith":
        from .dialects.ext import arith
        return arith
    elif name == "scf":
        from .dialects.ext import scf
        return scf
    elif name in ["Pipeline", "run_pipeline", "lower_rocir_to_standard"]:
        from . import passes
        return getattr(passes, name)

__all__ = [
    "rocir",
    "arith",
    "scf",
    "Pipeline",
    "run_pipeline",
    "lower_rocir_to_standard",
]

# Export compiler modules
from .compiler import Pipeline, run_pipeline
from .compiler.context import RAIIMLIRContextModule

__all__.extend(["Pipeline", "run_pipeline", "RAIIMLIRContextModule"])
