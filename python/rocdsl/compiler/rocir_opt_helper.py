"""Helper utilities to run Rocir coordinate lowering via the MLIR Python API."""

from mlir.ir import Module as ir_Module
from mlir.passmanager import PassManager

from rocdsl.compiler.context import ensure_rocir_python_extensions

_ROCIR_COORD_LOWERING_PIPELINE = "builtin.module(rocir-coord-lowering)"


def apply_rocir_coord_lowering(module: ir_Module) -> ir_Module:
    """Apply the rocir-coord-lowering pass using the in-process PassManager.

    The pass mutates the provided module in-place and also returns it for
    convenience so callers can chain additional processing.

    Args:
        module: MLIR module containing Rocir coordinate operations.

    Returns:
        The same module instance with Rocir ops lowered to arithmetic.

    Raises:
        RuntimeError: If the pass pipeline fails.
    """
    ctx = module.context
    ensure_rocir_python_extensions(ctx)

    pm = PassManager.parse(_ROCIR_COORD_LOWERING_PIPELINE, context=ctx)
    try:
        with ctx:
            pm.run(module.operation)
    except RuntimeError as exc:
        raise RuntimeError(f"rocir-coord-lowering failed: {exc}") from exc

    return module
