"""
FlyDSL PassManager Module

Re-exports PassManager from upstream mlir since it works with shared Context.
"""

# PassManager works with the shared underlying MlirContext*,
# so we can use upstream's PassManager directly
from mlir.passmanager import PassManager

__all__ = ['PassManager']
