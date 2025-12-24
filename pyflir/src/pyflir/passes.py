"""Pass management for FLIR compiler.

This module provides the main interface for defining and running
transformation passes on MLIR modules.
"""

# Re-export everything from compiler.pipeline
from .compiler.pipeline import (
    Pipeline,
    FLIRCompilerError,
    run_pipeline,
    lower_flir_to_standard,
    apply_flir_coord_lowering,
)

__all__ = [
    "Pipeline",
    "FLIRCompilerError",
    "run_pipeline",
    "lower_flir_to_standard",
    "apply_flir_coord_lowering",
]
