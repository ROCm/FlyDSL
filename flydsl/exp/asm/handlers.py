# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Operation handlers for MLIR operations in the ASM backend.

Implementation has been split into mixins to keep files <1000 LOC.
"""

from .gather_to_shared import G2SHandler
from .handlers_arith_affine import _ArithAffineHandlers
from .handlers_control import _ControlFlowHandlers
from .handlers_memory import _MemoryHandlers


class OperationHandlers(_ArithAffineHandlers, _MemoryHandlers, _ControlFlowHandlers):
    """Handles MLIR operations for the ASM backend."""

    def __init__(self, walker):
        self.walker = walker
        self.g2s = G2SHandler(self)
