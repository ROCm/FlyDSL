# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Instruction categorization for AMDGCN assembly.

This is the SINGLE canonical definition of InstructionCategory for the
entire ASM backend. All modules should import from this file.

Categories are used for:
- Scoreboard tracking (VMEM/LGKM operations need tickets)
- Hazard detection
- Instruction scheduling
- YAML instruction definitions (instruction_registry.py)

Note: The YAML registry imports InstructionCategory from this module to
avoid duplicate enum definitions. The categorize_instruction() function
provides fast prefix-based categorization for runtime classification.
"""

from enum import Enum


class InstructionCategory(Enum):
    """
    Categories of instructions for scoreboarding, ticketing, and scheduling.

    String values are used for consistency with YAML instruction definitions.
    """

    # Memory operations
    VMEM = "vmem"  # Vector memory (buffer_load/store)
    SMEM = "smem"  # Scalar memory (s_load, s_store)
    LDS = "lds"  # Local data share (ds_read, ds_write)
    LGKM = "lgkm"  # Combined LDS/GDS/scalar memory/messages (for waitcnt)

    # ALU operations
    VALU = "valu"  # Vector ALU
    SALU = "salu"  # Scalar ALU
    MFMA = "mfma"  # Matrix fused multiply-add

    # Control flow
    CONTROL = "control"  # Branch, call, return, etc.

    # Special
    PSEUDO = "pseudo"  # Pseudo-instructions (_comment, _label, _raw_asm)
    OTHER = "other"  # Unknown/other (nop, wait, barrier, etc.)


def categorize_instruction(instruction: str) -> InstructionCategory:
    """
    Categorize an instruction by its mnemonic.

    Uses fast prefix-based rules to classify instructions into functional
    categories. This categorization is used for both ticketing (VMEM/LGKM
    operations need tickets) and hazard detection (scoreboard tracking).

    Args:
        instruction: Instruction mnemonic (e.g., "buffer_load_dwordx4", "ds_read_b64")

    Returns:
        InstructionCategory for the instruction

    Examples:
        >>> categorize_instruction("buffer_load_dwordx4")
        InstructionCategory.VMEM
        >>> categorize_instruction("ds_write_b32")
        InstructionCategory.LGKM
        >>> categorize_instruction("v_add_u32")
        InstructionCategory.VALU
        >>> categorize_instruction("s_mov_b32")
        InstructionCategory.SALU
        >>> categorize_instruction("v_mfma_f32_16x16x16_f16")
        InstructionCategory.MFMA
    """
    # Vector memory operations (buffer loads/stores)
    if instruction.startswith("buffer_"):
        return InstructionCategory.VMEM

    # LDS/GDS/scalar memory operations
    # Includes: ds_read, ds_write, s_load, messages
    elif instruction.startswith("ds_") or instruction.startswith("s_load"):
        return InstructionCategory.LGKM

    # Matrix FMA operations (must check before generic v_ prefix)
    elif instruction.startswith("v_mfma"):
        return InstructionCategory.MFMA

    # Vector ALU operations
    elif instruction.startswith("v_"):
        return InstructionCategory.VALU

    # Scalar ALU operations
    elif instruction.startswith("s_"):
        return InstructionCategory.SALU

    # Everything else (NOPs, waits, barriers, etc.)
    else:
        return InstructionCategory.OTHER
