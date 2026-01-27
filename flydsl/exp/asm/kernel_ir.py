# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Kernel-level IR for whole-program register allocation.

This module defines a structured intermediate representation for entire AMDGCN
kernels, enabling a single linear-scan register allocator to manage all VGPRs
and SGPRs. The key insight is that MLIR provides SSA values, so we can build
a complete picture of register lifetimes before allocation.

Architecture:
    1. Operation handlers emit KInstr to KernelProgram using virtual regs
    2. Liveness analysis computes live ranges for all virtual regs
    3. Linear scan allocator assigns physical registers
    4. Renderer emits final assembly with physical register numbers

Virtual register types:
    - KVReg: Virtual VGPR (placeholder for physical vN)
    - KSReg: Virtual SGPR (placeholder for physical sN)
    - KPhysVReg/KPhysSReg: Precolored physical regs (ABI-mandated)

Operand types:
    - KImm: Immediate constant value
    - KRegRange: Contiguous register range (for pairs, quads, MFMA blocks)

Instruction representation:
    - KInstr: Single instruction with name, defs, uses, and constraints
    - KernelProgram: Ordered sequence of instructions + metadata

Note: Instructions are identified by name strings (e.g., "v_add_u32") rather
than enum opcodes. This allows the YAML instruction registry to be the single
source of truth for instruction definitions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Union
from enum import Enum, auto


# =============================================================================
# Register Classes
# =============================================================================


class RegClass(Enum):
    """Register class for allocation."""

    VGPR = auto()  # Vector general-purpose register
    SGPR = auto()  # Scalar general-purpose register
    AGPR = auto()  # Accumulator register (MFMA)


# =============================================================================
# Virtual Registers
# =============================================================================


@dataclass(frozen=True)
class KVReg:
    """Virtual VGPR - placeholder for physical vN."""

    id: int

    def __repr__(self) -> str:
        return f"kv{self.id}"


@dataclass(frozen=True)
class KSReg:
    """Virtual SGPR - placeholder for physical sN."""

    id: int

    def __repr__(self) -> str:
        return f"ks{self.id}"


@dataclass(frozen=True)
class KPhysVReg:
    """Precolored physical VGPR - for ABI-mandated registers."""

    index: int

    def __repr__(self) -> str:
        return f"v{self.index}"


@dataclass(frozen=True)
class KPhysSReg:
    """Precolored physical SGPR - for ABI-mandated registers."""

    index: int

    def __repr__(self) -> str:
        return f"s{self.index}"


@dataclass(frozen=True)
class KSpecialReg:
    """
    Special non-allocatable register (e.g., m0, exec, vcc).

    These registers are not part of the VGPR/SGPR pools and are
    never allocated by the register allocator. They are rendered
    verbatim in the final assembly.
    """

    name: str

    def __repr__(self) -> str:
        return self.name


# Well-known special registers
M0 = KSpecialReg("m0")
EXEC = KSpecialReg("exec")
VCC = KSpecialReg("vcc")


# Type aliases
KVirtualReg = Union[KVReg, KSReg]
KPhysicalReg = Union[KPhysVReg, KPhysSReg]
KReg = Union[KVirtualReg, KPhysicalReg, KSpecialReg]


def is_virtual(reg: KReg) -> bool:
    """Check if register is virtual (needs allocation)."""
    return isinstance(reg, (KVReg, KSReg))


def is_special(reg: KReg) -> bool:
    """Check if register is a special non-allocatable register."""
    return isinstance(reg, KSpecialReg)


def is_vgpr(reg: KReg) -> bool:
    """Check if register is a VGPR (virtual or physical)."""
    return isinstance(reg, (KVReg, KPhysVReg))


def is_sgpr(reg: KReg) -> bool:
    """Check if register is an SGPR (virtual or physical)."""
    return isinstance(reg, (KSReg, KPhysSReg))


def get_reg_class(reg: KReg) -> RegClass:
    """Get the register class for a register."""
    if isinstance(reg, (KVReg, KPhysVReg)):
        return RegClass.VGPR
    elif isinstance(reg, (KSReg, KPhysSReg)):
        return RegClass.SGPR
    raise ValueError(f"Unknown register type: {type(reg)}")


# =============================================================================
# Register Ranges (for pairs, quads, MFMA accumulators)
# =============================================================================


@dataclass(frozen=True)
class KRegRange:
    """
    A contiguous range of registers.

    Used for:
    - Register pairs (64-bit values, addresses)
    - Register quads (128-bit values, MFMA inputs)
    - MFMA accumulators (4-16 registers)

    The base_reg is the first register in the range. The allocator
    ensures consecutive physical registers are assigned.
    """

    base_reg: KReg
    count: int
    alignment: int = 1  # Physical register alignment (1, 2, 4, ...)

    def __repr__(self) -> str:
        base = repr(self.base_reg)
        if self.count == 1:
            return base
        return f"{base}..+{self.count}"

    def __getitem__(self, idx: int) -> "KReg":
        """Get individual register from range by index."""
        if idx < 0 or idx >= self.count:
            raise IndexError(
                f"Index {idx} out of range for KRegRange of count {self.count}"
            )

        base = self.base_reg
        if isinstance(base, KVReg):
            return KVReg(base.id + idx)
        elif isinstance(base, KSReg):
            return KSReg(base.id + idx)
        elif isinstance(base, KPhysVReg):
            return KPhysVReg(base.index + idx)
        elif isinstance(base, KPhysSReg):
            return KPhysSReg(base.index + idx)
        else:
            raise TypeError(f"Cannot index into KRegRange with base type {type(base)}")


# =============================================================================
# Immediate Values
# =============================================================================


@dataclass(frozen=True)
class KImm:
    """Immediate constant value."""

    value: int

    def __repr__(self) -> str:
        if -16 <= self.value <= 64:
            return str(self.value)
        return f"0x{self.value:x}"


@dataclass(frozen=True)
class KMemOffset:
    """Memory offset for load/store instructions."""

    bytes: int

    def __repr__(self) -> str:
        return f"offset:{self.bytes}"


# Type alias for all operand types
KOperand = Union[KReg, KRegRange, KImm, KMemOffset]


# =============================================================================
# Instruction Constraints
# =============================================================================


@dataclass
class KInstrConstraints:
    """
    Constraints on instruction operands for register allocation.

    These constraints inform the allocator about:
    - Register class requirements (VGPR vs SGPR)
    - Alignment requirements (pairs, quads)
    - Precoloring (must use specific physical reg)
    """

    # Destination constraints
    dst_class: Optional[RegClass] = None  # Required class for destination
    dst_alignment: int = 1  # Required alignment (1, 2, 4, ...)
    dst_size: int = 1  # Number of consecutive regs needed

    # Source constraints (per-operand)
    src_classes: Tuple[Optional[RegClass], ...] = ()  # Required class per source

    # Special constraints
    same_as_src: Optional[int] = None  # dst must be same as src[N]
    tied_operands: Tuple[
        Tuple[int, int], ...
    ] = ()  # (dst_idx, src_idx) pairs that must match

    # Memory constraints
    has_offset_field: bool = False  # Can encode immediate offset

    # Side effect flags
    m0_clobber: bool = False  # Clobbers m0
    vcc_clobber: bool = False  # Clobbers vcc


# =============================================================================
# Instructions
# =============================================================================


@dataclass
class KInstr:
    """
    A single kernel IR instruction.

    Each instruction has:
    - name: Instruction name (e.g., "v_add_u32") - matches registry
    - defs: Registers/ranges defined (written) by this instruction
    - uses: Operands read by this instruction
    - constraints: Allocation constraints
    - comment: Optional comment for debugging
    - target: Branch target label (for branch instructions)

    Note: Instructions are identified by name strings rather than enum opcodes.
    This allows the YAML instruction registry to be the single source of truth.
    """

    name: str  # Instruction name (e.g., "v_add_u32")
    defs: Tuple[Union[KReg, KRegRange], ...]  # Defined registers
    uses: Tuple[KOperand, ...]  # Used operands
    constraints: KInstrConstraints = field(default_factory=KInstrConstraints)
    comment: Optional[str] = None
    target: Optional[str] = None  # Branch target label (cleaner than using comment)

    def get_virtual_defs(self) -> List[KVirtualReg]:
        """Get all virtual registers defined by this instruction."""
        result = []
        for d in self.defs:
            if isinstance(d, KRegRange):
                if is_virtual(d.base_reg):
                    result.append(d.base_reg)
            elif is_virtual(d):
                result.append(d)
        return result

    def get_virtual_uses(self) -> List[KVirtualReg]:
        """Get all virtual registers used by this instruction."""
        result = []
        for u in self.uses:
            if isinstance(u, KRegRange):
                if is_virtual(u.base_reg):
                    result.append(u.base_reg)
            elif isinstance(u, (KVReg, KSReg)):
                result.append(u)
        return result

    def __repr__(self) -> str:
        defs_str = ", ".join(repr(d) for d in self.defs) if self.defs else ""
        uses_str = ", ".join(repr(u) for u in self.uses)
        if defs_str:
            return f"{self.name} {defs_str} = {uses_str}"
        return f"{self.name} {uses_str}"

    # Pseudo-instruction helpers
    @property
    def is_comment(self) -> bool:
        return self.name == "_comment"

    @property
    def is_label(self) -> bool:
        return self.name == "_label"

    @property
    def is_raw_asm(self) -> bool:
        return self.name == "_raw_asm"

    # Category-based helpers (using registry lookup)
    @property
    def category(self) -> "InstructionCategory":
        """Get the instruction category from the registry."""
        from .instruction_registry import get_instruction_category

        return get_instruction_category(self.name)

    @property
    def is_valu(self) -> bool:
        """Check if this is a VALU instruction."""
        from .instruction_categories import InstructionCategory

        return self.category == InstructionCategory.VALU

    @property
    def is_mfma(self) -> bool:
        """Check if this is an MFMA instruction."""
        from .instruction_categories import InstructionCategory

        return self.category == InstructionCategory.MFMA

    @property
    def is_vmem(self) -> bool:
        """Check if this is a VMEM (buffer) instruction."""
        from .instruction_categories import InstructionCategory

        return self.category == InstructionCategory.VMEM

    @property
    def is_branch(self) -> bool:
        """Check if this is a branch instruction."""
        # _preload_branch is a pseudo that renders as s_branch - treat as branch
        # so CFG builder correctly handles control flow. The branch target is
        # stored in instr.comment to enable CFG edge resolution.
        if self.name == "_preload_branch":
            return True
        if self.name.startswith("_"):
            # Other pseudo-instructions are not branches
            return False
        from .instruction_registry import is_branch_instruction

        return is_branch_instruction(self.name)

    @property
    def is_conditional_branch(self) -> bool:
        """Check if this is a conditional branch."""
        # Handle pseudo-instructions - they are not conditional branches
        if self.name.startswith("_"):
            return False
        from .instruction_registry import is_conditional_branch

        return is_conditional_branch(self.name)


# =============================================================================
# Kernel Program
# =============================================================================


@dataclass
class KernelABI:
    """
    ABI bindings for precolored registers.

    These registers are fixed by the GPU ABI and cannot be allocated
    to other values.
    """

    # Flat thread ID in v0 (contains packed tid_x, tid_y, tid_z)
    flat_tid_vreg: KPhysVReg = field(default_factory=lambda: KPhysVReg(0))

    # Kernarg pointer in s[0:1]
    kernarg_ptr_sreg_lo: KPhysSReg = field(default_factory=lambda: KPhysSReg(0))
    kernarg_ptr_sreg_hi: KPhysSReg = field(default_factory=lambda: KPhysSReg(1))

    # Workgroup ID in s[2], s[3], s[4] (optional, depends on kernel)
    workgroup_id_x_sreg: Optional[KPhysSReg] = None
    workgroup_id_y_sreg: Optional[KPhysSReg] = None
    workgroup_id_z_sreg: Optional[KPhysSReg] = None

    # Number of SGPRs preloaded by hardware (for kernel argument preloading)
    # When > 0, s[2:2+preload_count-1] are reserved for preloaded args
    preload_sgpr_count: int = 0

    def get_reserved_vgprs(self) -> Set[int]:
        """Get set of reserved VGPR indices."""
        return {self.flat_tid_vreg.index}

    def get_reserved_sgprs(self) -> Set[int]:
        """Get set of reserved SGPR indices."""
        reserved = {self.kernarg_ptr_sreg_lo.index, self.kernarg_ptr_sreg_hi.index}
        if self.workgroup_id_x_sreg:
            reserved.add(self.workgroup_id_x_sreg.index)
        if self.workgroup_id_y_sreg:
            reserved.add(self.workgroup_id_y_sreg.index)
        if self.workgroup_id_z_sreg:
            reserved.add(self.workgroup_id_z_sreg.index)
        # Reserve SGPRs for hardware-preloaded kernel arguments
        # Preloaded args start at s[2] (after kernarg_ptr at s[0:1])
        if self.preload_sgpr_count > 0:
            for i in range(self.preload_sgpr_count):
                reserved.add(2 + i)
        return reserved


@dataclass
class KernelProgram:
    """
    A complete kernel program in IR form.

    Contains:
    - instructions: Ordered list of kernel instructions
    - abi: ABI bindings for precolored registers
    - Additional metadata for allocation limits
    """

    instructions: List[KInstr] = field(default_factory=list)
    abi: KernelABI = field(default_factory=KernelABI)

    # Allocation limits
    max_vgprs: int = 256
    max_sgprs: int = 104

    # Virtual register counters
    _next_vreg_id: int = field(default=0, repr=False)
    _next_sreg_id: int = field(default=0, repr=False)

    # Loop control registers - these are exempt from SSA validation because
    # loop counters are naturally re-defined in the latch. The allocator
    # still assigns physical registers normally.
    _loop_control_sregs: Set[int] = field(default_factory=set, repr=False)

    # Accumulator VGPRs (MFMA accumulators) are updated in-place by MFMA.
    # We model this as a read-modify-write at the KernelProgram level, which
    # requires permitting re-definition of these VGPRs for SSA validation.
    _accumulator_vregs: Set[int] = field(default_factory=set, repr=False)

    def alloc_vreg(self) -> KVReg:
        """Allocate a new virtual VGPR."""
        vreg = KVReg(self._next_vreg_id)
        self._next_vreg_id += 1
        return vreg

    def alloc_vreg_range(self, count: int, alignment: int = 1) -> KRegRange:
        """Allocate a range of virtual VGPRs."""
        base = self.alloc_vreg()
        # Reserve additional IDs for the range
        for _ in range(count - 1):
            self.alloc_vreg()
        return KRegRange(base, count, alignment)

    def alloc_sreg(self) -> KSReg:
        """Allocate a new virtual SGPR."""
        sreg = KSReg(self._next_sreg_id)
        self._next_sreg_id += 1
        return sreg

    def alloc_sreg_range(self, count: int, alignment: int = 1) -> KRegRange:
        """Allocate a range of virtual SGPRs."""
        base = self.alloc_sreg()
        # Reserve additional IDs for the range
        for _ in range(count - 1):
            self.alloc_sreg()
        return KRegRange(base, count, alignment)

    def emit(self, instr: KInstr):
        """Add an instruction to the program."""
        self.instructions.append(instr)

    def emit_comment(self, text: str):
        """Emit a comment."""
        from .instruction_registry import Instruction

        self.emit(KInstr(Instruction._COMMENT, (), (KImm(0),), comment=text))

    def register_loop_control_sreg(self, sreg: KSReg) -> None:
        """
        Mark an SGPR as a loop control register.

        Loop control registers (counter, step, bound) are exempt from SSA
        validation because they are naturally re-defined in the latch.
        They still go through normal register allocation.
        """
        self._loop_control_sregs.add(sreg.id)

    def is_loop_control_sreg(self, sreg: KSReg) -> bool:
        """Check if an SGPR is a loop control register."""
        return sreg.id in self._loop_control_sregs

    def register_accumulator_vreg(self, vreg: KVReg) -> None:
        """
        Mark a VGPR as an accumulator register (MFMA accumulator).

        Accumulator registers are updated in-place and may be defined multiple
        times (read-modify-write semantics). They are therefore exempt from the
        "defined exactly once" SSA validation rule at the KernelProgram level.
        """
        self._accumulator_vregs.add(vreg.id)

    def register_accumulator_vreg_range(self, vreg_range: KRegRange) -> None:
        """Mark a VGPR range as accumulator registers (all components)."""
        base = vreg_range.base_reg
        if not isinstance(base, KVReg):
            return
        for i in range(vreg_range.count):
            self._accumulator_vregs.add(base.id + i)

    def is_accumulator_vreg(self, vreg: KVReg) -> bool:
        """Check if a VGPR is an accumulator register."""
        return vreg.id in self._accumulator_vregs

    def __len__(self) -> int:
        return len(self.instructions)

    def __iter__(self):
        return iter(self.instructions)


# =============================================================================
# Builder Helpers
# =============================================================================


class KernelBuilder:
    """
    Helper class for building KernelProgram.

    Provides convenient methods for emitting common instruction patterns
    and managing virtual registers.

    Note: Most methods are generated dynamically via __getattr__ using
    the instruction registry. Only essential methods are defined here.
    """

    def __init__(self, program: Optional[KernelProgram] = None):
        self.program = program if program is not None else KernelProgram()
        # Lazy-loaded registry
        self._registry = None

    def _get_registry(self):
        """Lazy load instruction registry."""
        if self._registry is None:
            from .instruction_registry import get_registry

            self._registry = get_registry("common")
        return self._registry

    # Virtual register allocation
    def vreg(self) -> KVReg:
        return self.program.alloc_vreg()

    def vreg_pair(self) -> KRegRange:
        return self.program.alloc_vreg_range(2, alignment=2)

    def vreg_quad(self) -> KRegRange:
        return self.program.alloc_vreg_range(4, alignment=4)

    def sreg(self) -> KSReg:
        return self.program.alloc_sreg()

    def sreg_pair(self) -> KRegRange:
        return self.program.alloc_sreg_range(2, alignment=2)

    def comment(self, text: str):
        """Emit a comment."""
        self.program.emit_comment(text)
