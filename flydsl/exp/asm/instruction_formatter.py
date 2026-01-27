# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Instruction Formatter for AMDGCN assembly.

This module provides the SINGLE point for formatting physical instructions
to assembly text. All instruction rendering goes through this formatter,
ensuring consistent output regardless of the compilation path.

Architecture:
    InstructionRegistry (YAML)
            │
            ▼
    InstructionFormatter
            │
            ▼
    Assembly Text

Usage:
    from instruction_formatter import InstructionFormatter

    formatter = InstructionFormatter()
    line = formatter.format("v_add_u32", defs=["v0"], uses=["v1", "v2"])
    # -> "    v_add_u32 v0, v1, v2"
"""

import os
from typing import List, Any, Optional, Tuple

from .instruction_registry import (
    InstructionDef,
    OperandType,
    InstructionCategory,
    get_registry,
    Instruction,
)

# Enable strict operand validation (fail on missing/extra operands)
# Default: enabled for tests (via pytest) and when explicitly set
_STRICT_VALIDATION = os.environ.get("WAVE_STRICT_FORMATTER", "0") == "1"


class FormatterValidationError(Exception):
    """Raised when instruction formatting fails validation."""

    pass


# =============================================================================
# Operand Formatting
# =============================================================================


def format_operand(
    value: Any, operand_types: Tuple[OperandType, ...], is_use: bool = True
) -> str:
    """
    Format an operand value to its assembly string representation.

    Handles:
    - Physical register strings ("v0", "s[4:7]", etc.)
    - Register indices (int -> vN or sN based on operand_types)
    - Immediate values (hex for large values)
    - Register range tuples

    Args:
        value: The operand value (string, int, tuple, etc.)
        operand_types: Allowed types for this operand from InstructionDef
        is_use: True if this is a use (source), False if definition (dest)

    Returns:
        Assembly string representation
    """
    if value is None:
        return ""

    # String operand (register name, label, etc.) - pass through
    if isinstance(value, str):
        return value

    # Tuple - register range (e.g., (4, 7) -> "v[4:7]" or "s[4:7]")
    if (
        isinstance(value, tuple)
        and len(value) >= 2
        and all(isinstance(v, int) for v in value)
    ):
        if (
            OperandType.VGPR_PAIR in operand_types
            or OperandType.VGPR_QUAD in operand_types
            or OperandType.VGPR_16 in operand_types
        ):
            return f"v[{value[0]}:{value[-1]}]"
        elif (
            OperandType.SGPR_PAIR in operand_types
            or OperandType.SGPR_QUAD in operand_types
        ):
            return f"s[{value[0]}:{value[-1]}]"
        # Default to VGPR for unknown tuple
        return f"v[{value[0]}:{value[-1]}]"

    # Integer - could be register index or immediate
    if isinstance(value, int):
        # For definitions, always treat as register index
        if not is_use:
            if (
                OperandType.SGPR in operand_types
                or OperandType.SGPR_PAIR in operand_types
            ):
                return f"s{value}"
            else:
                return f"v{value}"

        # For uses, check if it should be a register or immediate
        has_reg_type = any(
            t in operand_types
            for t in (
                OperandType.VGPR,
                OperandType.SGPR,
                OperandType.VGPR_PAIR,
                OperandType.SGPR_PAIR,
                OperandType.VGPR_QUAD,
                OperandType.SGPR_QUAD,
            )
        )
        has_imm_type = (
            OperandType.IMM in operand_types or OperandType.SIMM in operand_types
        )

        # If only register types allowed, format as register
        if has_reg_type and not has_imm_type:
            if (
                OperandType.SGPR in operand_types
                or OperandType.SGPR_PAIR in operand_types
            ):
                return f"s{value}"
            else:
                return f"v{value}"

        # Format as immediate
        if abs(value) > 0xFFFF:
            return f"0x{value & 0xFFFFFFFF:x}"
        elif value < 0:
            return str(value)
        else:
            return str(value)

    return str(value)


# =============================================================================
# Instruction Formatter
# =============================================================================


class InstructionFormatter:
    """
    The single point for formatting physical instructions to assembly text.

    This class:
    1. Looks up instruction definitions from the registry
    2. Validates operand counts and types
    3. Formats the instruction line with proper syntax
    4. Handles instruction-specific formatting rules (buffer ops, waitcnt, etc.)

    All physical instruction generation goes through this class, ensuring
    consistent assembly output.
    """

    def __init__(self, architecture: str = "common", strict: Optional[bool] = None):
        """
        Initialize the formatter with an instruction registry.

        Args:
            architecture: Target architecture for instruction definitions
            strict: If True, raise errors for missing/invalid operands.
                   If None, use WAVE_STRICT_FORMATTER env var (default: False).
        """
        self._registry = get_registry(architecture)
        self._strict = strict if strict is not None else _STRICT_VALIDATION

    def format(
        self,
        name: str,
        defs: List[Any] = None,
        uses: List[Any] = None,
        comment: str = None,
    ) -> str:
        """
        Format an instruction to its assembly representation.

        Args:
            name: Instruction name (e.g., "v_add_u32", "s_mov_b32")
            defs: List of destination operands (physical registers)
            uses: List of source operands (physical registers or immediates)
            comment: Optional comment to append

        Returns:
            Formatted assembly line (e.g., "    v_add_u32 v0, v1, v2")
        """
        defs = defs or []
        uses = uses or []

        # Look up instruction definition
        instr_def = self._registry.get(name)

        if instr_def:
            return self._format_with_def(instr_def, defs, uses, comment)
        else:
            # Fallback for instructions not in registry
            return self._format_fallback(name, defs, uses, comment)

    def _format_with_def(
        self,
        instr_def: InstructionDef,
        defs: List[Any],
        uses: List[Any],
        comment: str,
    ) -> str:
        """Format instruction using its definition."""
        # Handle pseudo-ops
        if instr_def.category == InstructionCategory.PSEUDO:
            return self._format_pseudo(instr_def, defs, uses, comment)

        # Strict validation: check operand counts
        if self._strict:
            self._validate_operands(instr_def, defs, uses)

        # Build operand strings
        operands = []

        # Add destinations (is_use=False)
        for i, def_op in enumerate(instr_def.defs):
            if i < len(defs):
                operands.append(format_operand(defs[i], def_op.types, is_use=False))
            elif not def_op.optional and self._strict:
                raise FormatterValidationError(
                    f"{instr_def.name}: missing required destination operand {i} ({def_op.name})"
                )

        # Add sources (is_use=True)
        for i, use_op in enumerate(instr_def.uses):
            if i < len(uses):
                value = uses[i]

                # Handle offset specially
                if OperandType.OFFSET in use_op.types:
                    if value and value != 0:
                        operands.append(f"offset:{value}")
                    elif use_op.optional:
                        continue
                    else:
                        operands.append(
                            format_operand(value, use_op.types, is_use=True)
                        )
                else:
                    operands.append(format_operand(value, use_op.types, is_use=True))
            elif not use_op.optional and self._strict:
                raise FormatterValidationError(
                    f"{instr_def.name}: missing required source operand {i} ({use_op.name})"
                )

        # Build line
        mnemonic = instr_def.mnemonic

        if not operands:
            line = f"    {mnemonic}"
        else:
            line = f"    {mnemonic} {', '.join(operands)}"

        # Apply special formatting
        line = self._apply_special_formatting(instr_def, line, operands)

        # Add LDS modifier if needed
        if instr_def.lds_modifier:
            if "  //" in line:
                parts = line.split("  //")
                line = parts[0] + " lds  //" + parts[1]
            else:
                line = line + " lds"

        # Add comment
        if comment:
            line += f"  // {comment}"

        return line

    def _validate_operands(
        self,
        instr_def: InstructionDef,
        defs: List[Any],
        uses: List[Any],
    ) -> None:
        """
        Validate operand counts and types against instruction definition.

        Raises:
            FormatterValidationError: If validation fails
        """
        # Count required operands
        required_defs = sum(1 for d in instr_def.defs if not d.optional)
        required_uses = sum(1 for u in instr_def.uses if not u.optional)

        if len(defs) < required_defs:
            raise FormatterValidationError(
                f"{instr_def.name}: expected {required_defs} destination operands, "
                f"got {len(defs)}. Expected: {[d.name for d in instr_def.defs if not d.optional]}"
            )

        if len(uses) < required_uses:
            raise FormatterValidationError(
                f"{instr_def.name}: expected {required_uses} source operands, "
                f"got {len(uses)}. Expected: {[u.name for u in instr_def.uses if not u.optional]}"
            )

        # Check for extra operands (warning in strict mode)
        max_defs = len(instr_def.defs)
        max_uses = len(instr_def.uses)

        if len(defs) > max_defs:
            raise FormatterValidationError(
                f"{instr_def.name}: too many destination operands ({len(defs)} > {max_defs})"
            )

        if len(uses) > max_uses:
            raise FormatterValidationError(
                f"{instr_def.name}: too many source operands ({len(uses)} > {max_uses})"
            )

    def _format_pseudo(
        self,
        instr_def: InstructionDef,
        defs: List[Any],
        uses: List[Any],
        comment: str,
    ) -> str:
        """Format pseudo-op."""
        if instr_def.name == Instruction._COMMENT:
            return f"    // {comment or (uses[0] if uses else '')}"
        elif instr_def.name == Instruction._LABEL:
            label = uses[0] if uses else comment
            return f"{label}:"
        elif instr_def.name == Instruction._RAW_ASM:
            return uses[0] if uses else comment
        return ""

    def _format_fallback(
        self,
        name: str,
        defs: List[Any],
        uses: List[Any],
        comment: str,
    ) -> str:
        """Format instruction not in registry (fallback)."""
        # Use name as mnemonic
        mnemonic = name

        # Combine operands (defs first, then uses)
        operands = [str(d) for d in defs] + [str(u) for u in uses]

        if not operands:
            line = f"    {mnemonic}"
        else:
            line = f"    {mnemonic} {', '.join(operands)}"

        if comment:
            line += f"  // {comment}"

        return line

    def _apply_special_formatting(
        self,
        instr_def: InstructionDef,
        line: str,
        operands: List[str],
    ) -> str:
        """Apply instruction-specific formatting rules."""
        # DS instructions: offset is space-separated
        if instr_def.offset_format == "space_separated":
            if ", offset:" in line:
                parts = line.split(", offset:")
                if len(parts) == 2:
                    line = parts[0] + " offset:" + parts[1]

        # Buffer instructions: special formatting
        if instr_def.category == InstructionCategory.VMEM:
            line = self._format_buffer_instruction(instr_def, line)

        # s_waitcnt: format wait count nicely
        if instr_def.name == Instruction.S_WAITCNT:
            line = self._format_waitcnt(line, operands)

        return line

    def _format_buffer_instruction(self, instr_def: InstructionDef, line: str) -> str:
        """Format buffer instruction with offen modifier."""
        # Separate any comment
        comment_part = ""
        main_line = line
        if "  //" in line:
            main_line, comment_part = line.split("  //", 1)
            comment_part = "  //" + comment_part

        if " " in main_line:
            prefix, rest = main_line.split(" ", 1)
            parts = rest.split(", ")

            # LDS loads have 3 operands, regular have 4
            is_lds = instr_def.lds_modifier
            min_parts = 3 if is_lds else 4

            if len(parts) >= min_parts:
                base = ", ".join(parts[:min_parts])
                modifiers = ["offen"]
                for part in parts[min_parts:]:
                    if part.startswith("offset:"):
                        offset_val = part.split(":")[1].strip()
                        if offset_val != "0":
                            modifiers.append(part.strip())
                    else:
                        modifiers.append(part.strip())
                line = f"{prefix} {base} {' '.join(modifiers)}{comment_part}"

        return line

    def _format_waitcnt(self, line: str, operands: List[str]) -> str:
        """Format s_waitcnt instruction nicely."""
        for op in operands:
            if isinstance(op, str) and ("vmcnt" in op or "lgkmcnt" in op):
                # Already formatted as waitcnt string
                return f"    s_waitcnt {op}"
            try:
                # Parse the waitcnt immediate
                val = int(op.replace("0x", ""), 16 if "0x" in op else 10)
                vmcnt = val & 0x3F
                lgkmcnt = (val >> 8) & 0xF
                if lgkmcnt == 0:
                    return f"    s_waitcnt vmcnt({vmcnt})"
                else:
                    return f"    s_waitcnt vmcnt({vmcnt}) lgkmcnt({lgkmcnt})"
            except:
                pass
        return line

    # =========================================================================
    # Convenience Methods for Common Instruction Patterns
    # =========================================================================

    def format_mov(
        self, dst: str, src: Any, is_sgpr: bool = False, comment: str = None
    ) -> str:
        """Format a move instruction."""
        name = "s_mov_b32" if is_sgpr else "v_mov_b32"
        return self.format(name, defs=[dst], uses=[src], comment=comment)

    def format_load_dwordx2(
        self, dst: str, base: str, offset: int, comment: str = None
    ) -> str:
        """Format an s_load_dwordx2 instruction."""
        return self.format(
            "s_load_dwordx2", defs=[dst], uses=[base, offset], comment=comment
        )

    def format_waitcnt(
        self, vmcnt: int = None, lgkmcnt: int = None, comment: str = None
    ) -> str:
        """Format an s_waitcnt instruction."""
        parts = []
        if vmcnt is not None:
            parts.append(f"vmcnt({vmcnt})")
        if lgkmcnt is not None:
            parts.append(f"lgkmcnt({lgkmcnt})")
        waitcnt_str = " ".join(parts) if parts else "0"
        return self.format(Instruction.S_WAITCNT, uses=[waitcnt_str], comment=comment)

    def format_endpgm(self, comment: str = None) -> str:
        """Format an s_endpgm instruction."""
        return self.format(Instruction.S_ENDPGM, comment=comment)

    def format_label(self, label: str) -> str:
        """Format a label."""
        return f"{label}:"

    def format_comment(self, text: str) -> str:
        """Format a comment."""
        return f"    // {text}"


# =============================================================================
# Singleton Access
# =============================================================================

_formatter_cache: dict = {}


def get_formatter(architecture: str = "common") -> InstructionFormatter:
    """Get a cached formatter instance for the given architecture."""
    if architecture not in _formatter_cache:
        _formatter_cache[architecture] = InstructionFormatter(architecture)
    return _formatter_cache[architecture]
