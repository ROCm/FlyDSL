"""
FlyDSL ExecutionEngine Module

Re-exports ExecutionEngine from upstream mlir since it works with shared Context.
"""

# ExecutionEngine works with the shared underlying MlirContext*,
# so we can use upstream's ExecutionEngine directly
from mlir.execution_engine import ExecutionEngine

__all__ = ['ExecutionEngine']
