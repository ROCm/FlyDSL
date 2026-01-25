"""Helper utilities to run Flir lowering via the MLIR Python API.

This module is maintained for backward compatibility. New code should use
the Pipeline class from flydsl.compiler.pipeline instead.
"""

from _mlir.ir import Module as ir_Module

# Import from the new pipeline module
from flydsl.compiler.pipeline import Pipeline, FLIRCompilerError

# Legacy constant for backward compatibility (coord lowering is part of flir-to-standard now)
_FLIR_COORD_LOWERING_PIPELINE = "builtin.module(flir-to-standard)"


def apply_flir_coord_lowering(module: ir_Module) -> ir_Module:
    """Apply Flir lowering using the in-process PassManager.

    The pass mutates the provided module in-place and also returns it for
    convenience so callers can chain additional processing.

    Note: This function is maintained for backward compatibility. Coordinate
    lowering is part of `flir-to-standard` now, so new code should use
    Pipeline().flir_to_standard().run(module).

    Args:
        module: MLIR module containing Flir operations.

    Returns:
        The same module instance with Flir ops lowered to arithmetic.

    Raises:
        RuntimeError: If the pass pipeline fails.
    
    Example (new API):
        >>> from flydsl.compiler.pipeline import Pipeline
        >>> pipeline = Pipeline().flir_coord_lowering()
        >>> result = pipeline.run(module)
    """
    try:
        pipeline = Pipeline().flir_coord_lowering()
        return pipeline.run(module)
    except FLIRCompilerError as exc:
        # Re-raise as RuntimeError for backward compatibility
        raise RuntimeError(str(exc)) from exc
