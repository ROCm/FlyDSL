# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared import/utility surface for split kernel pipeline modules."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple, TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from .kernel_ir import (
    KernelProgram,
    KernelBuilder,
    KInstr,
    KVReg,
    KSReg,
    KPhysVReg,
    KPhysSReg,
    KSpecialReg,
    KReg,
    KRegRange,
    KImm,
    KMemOffset,
    KernelABI,
    RegClass,
    M0,
    is_virtual,
)
from .kernel_liveness import compute_liveness, LivenessInfo
from .kernel_regalloc import (
    KernelRegAlloc,
    allocate_kernel,
    AllocationStats,
    AllocationError,
)
from .kernel_generator import KernelGenerator, PhysicalMapping, generate_program
from .unified_emitter import UnifiedEmitter, EmissionMode
from .instruction_categories import InstructionCategory, categorize_instruction
from .ticketing import Ticketing
from .instruction_registry import (
    InstructionRegistry,
    InstructionDef,
    OperandType,
    get_registry,
)


# Flag to enable algebraic simplification in kernel IR mode
# Enabled by default - the depth-tracking fix prevents O(n^2) behavior
_ENABLE_KERNEL_IR_SIMPLIFY = os.environ.get("WAVE_KERNEL_IR_SIMPLIFY", "1") == "1"


def _kernel_ticketing_enabled() -> bool:
    # Ticketing-driven waitcnt placement is enabled by default.
    # Set WAVE_KERNEL_TICKETING=0 to disable.
    return os.environ.get("WAVE_KERNEL_TICKETING", "1") == "1"


def _parse_waitcnt_threshold(op: Any) -> Optional[Tuple[str, int]]:
    """Parse 'vmcnt(N)' or 'lgkmcnt(N)' from an operand."""
    if not isinstance(op, str):
        return None
    if op.startswith("vmcnt(") and op.endswith(")"):
        try:
            return ("vmem", int(op[len("vmcnt(") : -1]))
        except ValueError:
            return None
    if op.startswith("lgkmcnt(") and op.endswith(")"):
        try:
            return ("lgkm", int(op[len("lgkmcnt(") : -1]))
        except ValueError:
            return None
    return None


def _iter_virtual_regs(operand: Any) -> List[KReg]:
    """Extract virtual regs referenced by an operand."""
    regs: List[KReg] = []
    if isinstance(operand, KRegRange):
        if is_virtual(operand.base_reg):
            # Base is enough if we also record components for defs.
            regs.append(operand.base_reg)
    elif isinstance(operand, (KVReg, KSReg)):
        if is_virtual(operand):
            regs.append(operand)
    return regs


# =============================================================================
# Operand type to register allocation info
# =============================================================================


def _get_def_info(operand_types: Tuple[OperandType, ...]) -> Tuple[str, int, int]:
    """
    Get destination register info from operand types.

    Returns: (class, count, alignment) where:
        - class: 'v' for VGPR, 's' for SGPR, None for no destination
        - count: number of registers (1, 2, 4, 16)
        - alignment: alignment requirement
    """
    for ot in operand_types:
        if ot == OperandType.VGPR:
            return ("v", 1, 1)
        elif ot == OperandType.VGPR_PAIR:
            return ("v", 2, 2)
        elif ot == OperandType.VGPR_QUAD:
            return ("v", 4, 4)
        elif ot == OperandType.VGPR_16:
            return ("v", 16, 4)
        elif ot == OperandType.SGPR:
            return ("s", 1, 1)
        elif ot == OperandType.SGPR_PAIR:
            return ("s", 2, 2)
        elif ot == OperandType.SGPR_QUAD:
            return ("s", 4, 4)
    return (None, 0, 1)


# Explicit export list - only public symbols
# This avoids namespace pollution when using `from ... import *`
__all__ = [
    # Core IR types
    "KernelProgram",
    "KernelBuilder",
    "KInstr",
    "KVReg",
    "KSReg",
    "KPhysVReg",
    "KPhysSReg",
    "KSpecialReg",
    "KReg",
    "KRegRange",
    "KImm",
    "KMemOffset",
    "KernelABI",
    "RegClass",
    "M0",
    "is_virtual",
    # Liveness and allocation
    "compute_liveness",
    "LivenessInfo",
    "KernelRegAlloc",
    "allocate_kernel",
    "AllocationStats",
    "AllocationError",
    # Code generation
    "KernelGenerator",
    "PhysicalMapping",
    "generate_program",
    "UnifiedEmitter",
    "EmissionMode",
    # Instruction info
    "InstructionCategory",
    "categorize_instruction",
    "InstructionRegistry",
    "InstructionDef",
    "OperandType",
    "get_registry",
    # Ticketing
    "Ticketing",
    # Helpers
    "_kernel_ticketing_enabled",
    "_get_def_info",
    "_ENABLE_KERNEL_IR_SIMPLIFY",
    "_parse_waitcnt_threshold",
    "_iter_virtual_regs",
]
