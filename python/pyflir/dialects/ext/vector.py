"""Vector dialect re-export for FLIR tests.

This module exists so tests can import vector ops through `flir.dialects.ext`
instead of directly importing from `mlir.dialects.*`.
"""

from _mlir.dialects.vector import *  # noqa: F401,F403


