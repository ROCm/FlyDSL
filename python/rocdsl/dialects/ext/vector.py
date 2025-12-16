"""Vector dialect re-export for RocDSL tests.

This module exists so tests can import vector ops through `rocdsl.dialects.ext`
instead of directly importing from `mlir.dialects.*`.
"""

from _mlir.dialects.vector import *  # noqa: F401,F403


