# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Unified instruction registry for AMDGCN instructions.

This module provides a single source of truth for instruction definitions,
loaded from YAML files. It supports:
- Architecture-specific instructions
- Common instructions across architectures
- Dynamic emission method generation
- Latency and constraint information
- Dynamic Instruction enum generation for type-safe comparisons

Usage:
    registry = InstructionRegistry.load("gfx942")

    # Get instruction info
    info = registry.get("v_add_u32")
    print(info.mnemonic, info.latency)

    # Use with emitter
    emitter = InstructionEmitter(registry)
    emitter.v_add_u32(dst, src0, src1)

    # Type-safe instruction name comparisons using Instruction enum
    from .instruction_registry import Instruction
    if instr.name == Instruction.V_ADD_U32:
        ...
"""

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Import canonical InstructionCategory from instruction_categories module
from .instruction_categories import InstructionCategory


# ==============================================================================
# Operand Types
# ==============================================================================


class OperandType(Enum):
    """Types of operands an instruction can accept."""

    VGPR = auto()  # Single VGPR
    VGPR_PAIR = auto()  # Pair of consecutive VGPRs (v[n:n+1])
    VGPR_QUAD = auto()  # Quad of consecutive VGPRs (v[n:n+3])
    VGPR_16 = auto()  # 16 consecutive VGPRs
    SGPR = auto()  # Single SGPR
    SGPR_PAIR = auto()  # Pair of consecutive SGPRs (s[n:n+1])
    SGPR_QUAD = auto()  # Quad of consecutive SGPRs (s[n:n+3])
    IMM = auto()  # Immediate value
    IMM16 = auto()  # 16-bit immediate
    OFFSET = auto()  # Memory offset (offset:N)
    LABEL = auto()  # Branch label
    SPECIAL = auto()  # Special register (m0, exec, vcc)
    STRING = auto()  # String (for comments/raw asm)
    WAITCNT = auto()  # Wait count value
    OFF = auto()  # Literal "off"


# InstructionCategory is imported from instruction_categories.py to avoid
# duplicate enum definitions. See that module for the canonical definition.


# ==============================================================================
# Instruction Definition
# ==============================================================================


@dataclass(frozen=True)
class OperandDef:
    """Definition of an instruction operand."""

    name: str
    types: Tuple[OperandType, ...]  # Allowed types
    optional: bool = False
    alignment: int = 1  # For register ranges


@dataclass(frozen=True)
class InstructionDef:
    """
    Complete definition of an instruction.

    This contains all the information needed to:
    - Emit the instruction to assembly
    - Allocate registers for it
    - Schedule it with latency awareness
    """

    name: str  # Internal name (e.g., "v_add_u32")
    mnemonic: str  # Assembly mnemonic
    category: InstructionCategory  # Instruction category
    format: str  # Encoding format
    defs: Tuple[OperandDef, ...]  # Destination operands
    uses: Tuple[OperandDef, ...]  # Source operands
    latency: int  # Approximate latency in cycles

    # Special properties
    is_memory: bool = False  # Memory access
    is_sync: bool = False  # Synchronization
    is_barrier: bool = False  # Barrier
    is_branch: bool = False  # Branch instruction
    is_conditional: bool = False  # Conditional branch
    is_terminator: bool = False  # Block terminator
    sets_scc: bool = False  # Sets SCC
    reads_scc: bool = False  # Reads SCC
    lds_modifier: bool = False  # Has LDS modifier
    offset_format: str = "comma"  # "comma" or "space_separated"
    is_raw: bool = False  # Raw assembly
    is_label_def: bool = False  # Label definition
    is_accumulator: bool = False  # MFMA accumulator

    # Architecture support
    architectures: Tuple[str, ...] = ()  # Empty = all architectures


# ==============================================================================
# Type Parsing
# ==============================================================================


def _parse_operand_type(type_str: str) -> Tuple[OperandType, ...]:
    """Parse operand type string to tuple of OperandType."""
    type_map = {
        "vgpr": OperandType.VGPR,
        "vgpr_pair": OperandType.VGPR_PAIR,
        "vgpr_quad": OperandType.VGPR_QUAD,
        "vgpr_16": OperandType.VGPR_16,
        "sgpr": OperandType.SGPR,
        "sgpr_pair": OperandType.SGPR_PAIR,
        "sgpr_quad": OperandType.SGPR_QUAD,
        "imm": OperandType.IMM,
        "imm16": OperandType.IMM16,
        "offset": OperandType.OFFSET,
        "label": OperandType.LABEL,
        "special": OperandType.SPECIAL,
        "string": OperandType.STRING,
        "waitcnt": OperandType.WAITCNT,
        "off": OperandType.OFF,
    }

    types = []
    for t in type_str.split("|"):
        t = t.strip().lower()
        if t in type_map:
            types.append(type_map[t])

    return tuple(types) if types else (OperandType.IMM,)


def _parse_category(category_str: str) -> InstructionCategory:
    """Parse category string to InstructionCategory."""
    # Map YAML category strings to InstructionCategory enum values
    # InstructionCategory uses string values, so we can match directly
    category_str_lower = category_str.lower()
    try:
        return InstructionCategory(category_str_lower)
    except ValueError:
        # Fallback for unmapped categories
        return InstructionCategory.OTHER


def _parse_operand_def(op_dict: Dict[str, Any]) -> OperandDef:
    """Parse operand definition dictionary."""
    return OperandDef(
        name=op_dict.get("name", ""),
        types=_parse_operand_type(op_dict.get("type", "imm")),
        optional=op_dict.get("optional", False),
        alignment=op_dict.get("alignment", 1),
    )


def _parse_instruction_def(name: str, data: Dict[str, Any]) -> InstructionDef:
    """Parse instruction definition from YAML data."""
    # Parse operand definitions
    defs_data = data.get("defs", [])
    uses_data = data.get("uses", [])
    defs = tuple(_parse_operand_def(d) for d in defs_data)
    uses = tuple(_parse_operand_def(u) for u in uses_data)

    # Parse special properties
    special = data.get("special", {})

    return InstructionDef(
        name=name,
        mnemonic=data.get("mnemonic", name),
        category=_parse_category(data.get("category", "valu")),
        format=data.get("format", "unknown"),
        defs=defs,
        uses=uses,
        latency=data.get("latency", 1),
        is_memory=special.get("memory", False),
        is_sync=special.get("sync", False),
        is_barrier=special.get("barrier", False),
        is_branch=special.get("branch", False),
        is_conditional=special.get("conditional", False),
        is_terminator=special.get("terminator", False),
        sets_scc=special.get("sets_scc", False),
        reads_scc=special.get("reads_scc", False),
        lds_modifier=special.get("lds_modifier", False),
        offset_format=special.get("offset_format", "comma"),
        is_raw=special.get("raw", False),
        is_label_def=special.get("label_def", False),
        is_accumulator=special.get("accumulator", False),
        architectures=tuple(data.get("architectures", [])),
    )


# ==============================================================================
# Instruction Registry
# ==============================================================================


class InstructionRegistry:
    """
    Registry of instruction definitions loaded from YAML files.

    Provides lookup by instruction name and supports architecture-specific
    instruction sets.
    """

    def __init__(self, architecture: str = "common"):
        self.architecture = architecture
        self._instructions: Dict[str, InstructionDef] = {}
        self._by_category: Dict[InstructionCategory, List[InstructionDef]] = {}
        self._by_mnemonic: Dict[str, InstructionDef] = {}

    @classmethod
    def load(cls, architecture: str = "common") -> "InstructionRegistry":
        """
        Load instruction registry for the given architecture.

        Loads common.yaml first, then overlays architecture-specific definitions.
        """
        registry = cls(architecture)

        # Find the instruction_defs directory
        module_dir = Path(__file__).parent
        defs_dir = module_dir / "instruction_defs"

        # Load common definitions
        common_path = defs_dir / "common.yaml"
        if common_path.exists():
            registry._load_yaml(common_path)

        # Load architecture-specific definitions (overrides common)
        if architecture != "common":
            arch_path = defs_dir / f"{architecture}.yaml"
            if arch_path.exists():
                registry._load_yaml(arch_path)

        return registry

    def _load_yaml(self, path: Path) -> None:
        """Load instructions from a YAML file."""
        try:
            import yaml
        except ImportError:
            # Fallback: try to use ruamel.yaml
            try:
                from ruamel.yaml import YAML

                yaml_loader = YAML()
                with open(path, "r") as f:
                    data = yaml_loader.load(f)
            except ImportError:
                raise ImportError(
                    "Either 'pyyaml' or 'ruamel.yaml' is required for instruction loading. "
                    "Install with: pip install pyyaml"
                )
        else:
            with open(path, "r") as f:
                data = yaml.safe_load(f)

        if not data or "instructions" not in data:
            return

        for name, instr_data in data["instructions"].items():
            instr_def = _parse_instruction_def(name, instr_data)
            self._register(instr_def)

    def _register(self, instr: InstructionDef) -> None:
        """Register an instruction definition."""
        self._instructions[instr.name] = instr
        self._by_mnemonic[instr.mnemonic] = instr

        if instr.category not in self._by_category:
            self._by_category[instr.category] = []
        self._by_category[instr.category].append(instr)

    def get(self, name: str) -> Optional[InstructionDef]:
        """Get instruction definition by name."""
        return self._instructions.get(name)

    def get_by_mnemonic(self, mnemonic: str) -> Optional[InstructionDef]:
        """Get instruction definition by mnemonic."""
        return self._by_mnemonic.get(mnemonic)

    def get_category(self, category: InstructionCategory) -> List[InstructionDef]:
        """Get all instructions in a category."""
        return self._by_category.get(category, [])

    def __contains__(self, name: str) -> bool:
        return name in self._instructions

    def __iter__(self):
        return iter(self._instructions.values())

    def __len__(self) -> int:
        return len(self._instructions)


# ==============================================================================
# Instruction Enum Generation
# ==============================================================================


def _generate_instruction_enum(instructions: Dict[str, InstructionDef]) -> type:
    """
    Dynamically generate an Instruction enum from loaded instruction names.

    The generated enum uses str as a mixin, allowing direct comparison with
    strings without needing .value:

        if instr.name == Instruction.V_ADD_U32:  # Works directly!

    Args:
        instructions: Dictionary of instruction name -> InstructionDef

    Returns:
        A dynamically generated Instruction enum class
    """
    # Build enum members: convert instruction names to UPPER_SNAKE_CASE
    # e.g., "v_add_u32" -> "V_ADD_U32"
    members = {}
    for name in instructions.keys():
        # Convert to uppercase for enum member name
        enum_name = name.upper()
        members[enum_name] = name

    # Create a string enum (inherits from str) for easy comparison
    # This allows: instr.name == Instruction.V_ADD_U32 (no .value needed)
    return Enum("Instruction", members, type=str)


# ==============================================================================
# Global Registry Instance
# ==============================================================================

_global_registry: Optional[InstructionRegistry] = None
_global_instruction_enum: Optional[type] = None


def get_registry(architecture: str = "common") -> InstructionRegistry:
    """
    Get the global instruction registry, loading if necessary.

    Uses lazy loading to avoid loading YAML at import time.
    """
    global _global_registry, _global_instruction_enum
    if _global_registry is None or _global_registry.architecture != architecture:
        _global_registry = InstructionRegistry.load(architecture)
        # Generate instruction enum when registry is loaded
        _global_instruction_enum = _generate_instruction_enum(
            _global_registry._instructions
        )
    return _global_registry


def get_instruction_enum(architecture: str = "common") -> type:
    """
    Get the Instruction enum for type-safe instruction name comparisons.

    The enum is generated from the loaded instruction definitions, so it
    includes all instructions from the YAML files.

    Usage:
        Instruction = get_instruction_enum("gfx942")
        if instr.name == Instruction.V_ADD_U32:
            ...

    Args:
        architecture: Target architecture (default: "common")

    Returns:
        Dynamically generated Instruction enum class
    """
    global _global_instruction_enum
    # Ensure registry is loaded (which generates the enum)
    get_registry(architecture)
    return _global_instruction_enum


def get_instruction(
    name: str, architecture: str = "common"
) -> Optional[InstructionDef]:
    """Convenience function to get an instruction definition."""
    return get_registry(architecture).get(name)


# ==============================================================================
# Instruction Property Helpers
# ==============================================================================


def get_instruction_category(
    name: str, architecture: str = "common"
) -> InstructionCategory:
    """
    Get the category of an instruction by name.

    Args:
        name: Instruction name (e.g., "v_add_u32")
        architecture: Target architecture (default "common")

    Returns:
        InstructionCategory for the instruction

    Raises:
        ValueError: If instruction is not found in registry
    """
    registry = get_registry(architecture)
    instr_def = registry.get(name)
    if instr_def is None:
        raise ValueError(
            f"Unknown instruction '{name}' not found in registry for "
            f"architecture '{architecture}'. Please add it to the YAML definitions."
        )
    return instr_def.category


def is_conditional_branch(name: str, architecture: str = "common") -> bool:
    """
    Check if instruction is a conditional branch.

    Args:
        name: Instruction name (e.g., "s_cbranch_scc0")
        architecture: Target architecture (default "common")

    Returns:
        True if instruction is a conditional branch

    Raises:
        ValueError: If instruction is not found in registry
    """
    registry = get_registry(architecture)
    instr_def = registry.get(name)
    if instr_def is None:
        raise ValueError(
            f"Unknown instruction '{name}' not found in registry for "
            f"architecture '{architecture}'. Please add it to the YAML definitions."
        )
    return instr_def.is_conditional


def is_branch_instruction(name: str, architecture: str = "common") -> bool:
    """
    Check if instruction is a branch (conditional or unconditional).

    Args:
        name: Instruction name (e.g., "s_branch", "s_cbranch_scc0")
        architecture: Target architecture (default "common")

    Returns:
        True if instruction is a branch

    Raises:
        ValueError: If instruction is not found in registry
    """
    registry = get_registry(architecture)
    instr_def = registry.get(name)
    if instr_def is None:
        raise ValueError(
            f"Unknown instruction '{name}' not found in registry for "
            f"architecture '{architecture}'. Please add it to the YAML definitions."
        )
    return instr_def.is_branch


# ==============================================================================
# Convenience: Lazy-loaded Instruction enum
# ==============================================================================
# This provides a module-level Instruction that loads on first access.
# Import with: from .instruction_registry import Instruction


class _InstructionProxy:
    """
    Proxy class that lazily loads the Instruction enum on first attribute access.

    This allows importing Instruction at module level without triggering YAML
    loading until the enum is actually used.
    """

    _enum: Optional[type] = None

    def __getattr__(self, name: str):
        if self._enum is None:
            # Default to gfx942 for common use case
            self._enum = get_instruction_enum("gfx942")
        return getattr(self._enum, name)

    def __repr__(self):
        return "<Instruction (lazy-loaded)>"


# Module-level proxy for convenient imports
Instruction = _InstructionProxy()
