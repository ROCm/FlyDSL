# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Generator for kernel IR to AMDGCN assembly.

This module converts a KernelProgram (with virtual registers allocated to
physical registers) into final assembly text.

The generator:
1. Takes a KernelProgram and a mapping from virtual to physical registers
2. Substitutes physical register numbers for virtual registers
3. Uses InstructionFormatter for ALL instruction rendering
4. Formats instructions according to AMDGCN assembly syntax

Note: All instruction formatting goes through InstructionFormatter, which
uses the YAML registry as the single source of truth.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from .kernel_ir import (
    KernelProgram,
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
    KOperand,
)
from .instruction_formatter import get_formatter
from .instruction_registry import Instruction


@dataclass
class PhysicalMapping:
    """
    Mapping from virtual registers to physical register indices.

    Supports both single registers and range components.
    """

    vreg_map: Dict[int, int]  # KVReg.id -> physical VGPR index
    sreg_map: Dict[int, int]  # KSReg.id -> physical SGPR index
    # Range membership: component_id -> (base_id, size) for resolving range components
    vreg_ranges: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    sreg_ranges: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    def get_phys_vreg(self, vreg: KVReg) -> int:
        """Get physical VGPR index for a virtual VGPR."""
        # Direct lookup
        if vreg.id in self.vreg_map:
            return self.vreg_map[vreg.id]

        # Check if this is a component of a range
        if vreg.id in self.vreg_ranges:
            base_id, size = self.vreg_ranges[vreg.id]
            offset = vreg.id - base_id
            if base_id in self.vreg_map:
                return self.vreg_map[base_id] + offset

        raise ValueError(f"Virtual VGPR {vreg} not allocated")

    def get_phys_sreg(self, sreg: KSReg) -> int:
        """Get physical SGPR index for a virtual SGPR."""
        # Direct lookup
        if sreg.id in self.sreg_map:
            return self.sreg_map[sreg.id]

        # Check if this is a component of a range
        if sreg.id in self.sreg_ranges:
            base_id, size = self.sreg_ranges[sreg.id]
            offset = sreg.id - base_id
            if base_id in self.sreg_map:
                return self.sreg_map[base_id] + offset

        raise ValueError(f"Virtual SGPR {sreg} not allocated")


class KernelGenerator:
    """
    Generates AMDGCN assembly from KernelProgram.

    Uses InstructionFormatter as the SINGLE point for all instruction
    formatting, ensuring consistent assembly output.
    """

    def __init__(
        self,
        program: KernelProgram,
        mapping: PhysicalMapping,
        architecture: str = "common",
    ):
        self.program = program
        self.mapping = mapping
        self._formatter = get_formatter(architecture)

    def generate(self) -> List[str]:
        """Generate the entire program as assembly lines."""
        lines = []
        for instr in self.program.instructions:
            result = self._generate_instr(instr)
            if result is not None:
                # Handle multi-line results (from pseudo-instructions)
                if isinstance(result, list):
                    lines.extend(result)
                else:
                    lines.append(result)
        return lines

    def generate_to_string(self) -> str:
        """Generate the entire program as a single string."""
        return "\n".join(self.generate())

    # =========================================================================
    # Register Resolution
    # =========================================================================

    def _resolve_reg(self, reg: KReg) -> str:
        """Resolve a register to its physical string representation."""
        if isinstance(reg, KVReg):
            phys = self.mapping.get_phys_vreg(reg)
            return f"v{phys}"
        elif isinstance(reg, KSReg):
            phys = self.mapping.get_phys_sreg(reg)
            return f"s{phys}"
        elif isinstance(reg, KPhysVReg):
            return f"v{reg.index}"
        elif isinstance(reg, KPhysSReg):
            return f"s{reg.index}"
        elif isinstance(reg, KSpecialReg):
            return reg.name
        raise ValueError(f"Unknown register type: {type(reg)}")

    def _resolve_reg_range(self, range_: KRegRange) -> str:
        """Resolve a register range to its physical string representation."""
        base_reg = range_.base_reg
        count = range_.count

        if isinstance(base_reg, KVReg):
            start = self.mapping.get_phys_vreg(base_reg)
            return f"v[{start}:{start + count - 1}]"
        elif isinstance(base_reg, KSReg):
            start = self.mapping.get_phys_sreg(base_reg)
            return f"s[{start}:{start + count - 1}]"
        elif isinstance(base_reg, KPhysVReg):
            start = base_reg.index
            return f"v[{start}:{start + count - 1}]"
        elif isinstance(base_reg, KPhysSReg):
            start = base_reg.index
            return f"s[{start}:{start + count - 1}]"
        raise ValueError(f"Unknown base register type: {type(base_reg)}")

    def _resolve_operand(self, op: KOperand) -> str:
        """Resolve an operand to its string representation."""
        if isinstance(op, KRegRange):
            return self._resolve_reg_range(op)
        elif isinstance(op, (KVReg, KSReg, KPhysVReg, KPhysSReg, KSpecialReg)):
            return self._resolve_reg(op)
        elif isinstance(op, KImm):
            # Format immediate appropriately
            if -16 <= op.value <= 64:
                return str(op.value)
            return f"0x{op.value & 0xffffffff:x}"
        elif isinstance(op, int):
            # Handle raw integer immediates
            if -16 <= op <= 64:
                return str(op)
            return f"0x{op & 0xffffffff:x}"
        elif isinstance(op, KMemOffset):
            # Return the offset value directly - InstructionFormatter will add "offset:" prefix
            return op.bytes
        elif isinstance(op, str):
            # Handle string operands (e.g., waitcnt values)
            return op
        raise ValueError(f"Unknown operand type: {type(op)}")

    # =========================================================================
    # Instruction Generation
    # =========================================================================

    def _generate_instr(self, instr: KInstr) -> Optional[Union[str, List[str]]]:
        """Generate a single instruction to assembly."""
        name = instr.name

        # Handle built-in pseudo-ops
        if instr.is_comment:
            if instr.comment:
                return self._formatter.format_comment(instr.comment)
            return None

        if instr.is_label:
            if instr.comment:
                return self._formatter.format_label(instr.comment)
            return None

        if instr.is_raw_asm:
            return instr.comment if instr.comment else None

        # Handle custom pseudo-instructions
        handler = self._get_pseudo_handler(name)
        if handler:
            return handler(instr)

        # Standard instruction: resolve operands and format
        defs = [self._resolve_def(d) for d in instr.defs]
        uses = [self._resolve_operand(u) for u in instr.uses]

        return self._formatter.format(name, defs=defs, uses=uses, comment=instr.comment)

    def _resolve_def(self, d: KOperand) -> str:
        """Resolve a definition operand."""
        if isinstance(d, KRegRange):
            return self._resolve_reg_range(d)
        else:
            return self._resolve_reg(d)

    # =========================================================================
    # Pseudo-Instruction Handlers
    # =========================================================================

    def _get_pseudo_handler(self, name: str):
        """Get handler for pseudo-instruction, or None if not a pseudo."""
        handlers = {
            "_srd_define": self._handle_srd_define,
            "_srd_copy_define": self._handle_srd_copy_define,
            "_g2s_srd_copy": self._handle_g2s_srd_copy,
            "_srd_load_base": self._handle_srd_load_base,
            "_srd_copy_base": self._handle_srd_copy_base,
            "_srd_fill_size": self._handle_srd_fill_size,
            "_srd_fill_stride": self._handle_srd_fill_stride,
            "_label": self._handle_label,
            "s_cbranch_scc1": self._handle_branch,
            "s_branch": self._handle_branch,
            "s_cmp_lt_u32": self._handle_compare,
            "_loop_inc": self._handle_loop_inc,
            "_loop_begin": self._handle_noop,
            "_loop_end": self._handle_noop,
            "_mfma_acc": self._handle_mfma_acc,
            "_mfma_acc_16x16x32": self._handle_mfma_acc_16x16x32,
            "buffer_load_dword_lds": self._handle_buffer_load_lds,
            "buffer_load_dwordx4_lds": self._handle_buffer_load_lds,
            "_init_acc_quad": self._handle_init_acc_quad,
            # Preload entry point pattern: s_branch to aligned kernel entry
            # Uses standard _label (handled by _handle_label) and _raw_asm for alignment
            "_preload_branch": self._handle_preload_branch,
        }
        return handlers.get(name)

    def _handle_srd_define(self, instr: KInstr) -> Optional[str]:
        """SRD define - no output, just for liveness tracking."""
        return None

    def _handle_srd_copy_define(self, instr: KInstr) -> Optional[str]:
        """SRD copy define - no output, just for liveness tracking."""
        return None

    def _handle_g2s_srd_copy(self, instr: KInstr) -> List[str]:
        """G2S SRD copy: emit 4 s_mov/s_and/s_or instructions."""
        new_range = instr.defs[0]
        orig_range = instr.uses[0]
        cache_swizzle = instr.uses[1].value if isinstance(instr.uses[1], KImm) else 0

        new_base = self._resolve_reg_range(new_range)
        orig_base = self._resolve_reg_range(orig_range)

        # Parse s[N:M] format to get start index
        new_start = int(new_base.split("[")[1].split(":")[0])
        orig_start = int(orig_base.split("[")[1].split(":")[0])

        lines = []
        # Word0: copy base address low
        lines.append(
            self._formatter.format(
                "s_mov_b32",
                defs=[f"s{new_start}"],
                uses=[f"s{orig_start}"],
                comment="SRD word0",
            )
        )

        # Word1: base address high (with optional cache swizzle)
        lines.append(
            self._formatter.format(
                "s_and_b32",
                defs=[f"s{new_start + 1}"],
                uses=[f"s{orig_start + 1}", "0xffff"],
            )
        )
        if cache_swizzle != 0:
            lines.append(
                self._formatter.format(
                    "s_or_b32",
                    defs=[f"s{new_start + 1}"],
                    uses=[f"s{new_start + 1}", hex(cache_swizzle)],
                    comment="cache swizzle",
                )
            )

        # Word2: max buffer size
        lines.append(
            self._formatter.format(
                "s_mov_b32",
                defs=[f"s{new_start + 2}"],
                uses=["0x7ffffffd"],
                comment="SRD word2",
            )
        )

        # Word3: LDS format
        lines.append(
            self._formatter.format(
                "s_mov_b32",
                defs=[f"s{new_start + 3}"],
                uses=["0x27000"],
                comment="SRD word3",
            )
        )

        return lines

    def _handle_srd_load_base(self, instr: KInstr) -> str:
        """Load base address into SRD[0:1] from kernarg segment."""
        srd_range = instr.uses[0]
        kernarg_pair = instr.uses[1]
        offset_imm = instr.uses[2]

        srd_phys = self._resolve_reg_range(srd_range)
        kernarg_phys = (
            self._resolve_reg_range(kernarg_pair)
            if isinstance(kernarg_pair, KRegRange)
            else self._resolve_operand(kernarg_pair)
        )
        offset = offset_imm.value if isinstance(offset_imm, KImm) else offset_imm

        # Extract base from "s[N:M]" format
        base = int(srd_phys.split("[")[1].split(":")[0])

        return self._formatter.format(
            "s_load_dwordx2",
            defs=[f"s[{base}:{base+1}]"],
            uses=[kernarg_phys, offset],
            comment=instr.comment,
        )

    def _handle_srd_copy_base(self, instr: KInstr) -> str:
        """Copy preloaded base address into SRD[0:1] using s_mov_b64.

        Used with kernel argument preloading where args are already in SGPRs.

        Instruction format:
            defs=(srd_range,)  - SRD range being written to
            uses=(preloaded_pair,) - Preloaded arg registers being read from
        """
        # SRD range is in defs (we're writing to it)
        srd_range = instr.defs[0]
        # Preloaded pair is in uses (we're reading from it)
        preloaded_pair = instr.uses[0]

        srd_phys = self._resolve_reg_range(srd_range)
        preloaded_phys = self._resolve_reg_range(preloaded_pair)

        # Extract base from "s[N:M]" format
        srd_base = int(srd_phys.split("[")[1].split(":")[0])

        return self._formatter.format(
            "s_mov_b64",
            defs=[f"s[{srd_base}:{srd_base+1}]"],
            uses=[preloaded_phys],
            comment=instr.comment,
        )

    def _handle_srd_fill_size(self, instr: KInstr) -> str:
        """Fill SRD[2] with size value."""
        srd_range = instr.uses[0]
        size_val = (
            instr.uses[1].value if isinstance(instr.uses[1], KImm) else instr.uses[1]
        )

        srd_phys = self._resolve_reg_range(srd_range)
        base = int(srd_phys.split("[")[1].split(":")[0])

        return self._formatter.format(
            "s_mov_b32",
            defs=[f"s{base + 2}"],
            uses=[hex(size_val)],
            comment=instr.comment,
        )

    def _handle_srd_fill_stride(self, instr: KInstr) -> str:
        """Fill SRD[3] with stride descriptor."""
        srd_range = instr.uses[0]
        stride_val = (
            instr.uses[1].value if isinstance(instr.uses[1], KImm) else instr.uses[1]
        )

        srd_phys = self._resolve_reg_range(srd_range)
        base = int(srd_phys.split("[")[1].split(":")[0])

        return self._formatter.format(
            "s_mov_b32",
            defs=[f"s{base + 3}"],
            uses=[hex(stride_val)],
            comment=instr.comment,
        )

    def _handle_preload_branch(self, instr: KInstr) -> str:
        """Emit s_branch to aligned kernel entry point (for preloading)."""
        if not instr.target:
            raise ValueError("_preload_branch requires target field")
        return self._formatter.format("s_branch", uses=[instr.target])

    def _handle_label(self, instr: KInstr) -> str:
        """Handle label pseudo-instruction."""
        label = instr.comment if instr.comment else "label"
        return self._formatter.format_label(label)

    def _handle_branch(self, instr: KInstr) -> str:
        """Handle branch instructions."""
        if not instr.target:
            raise ValueError(f"{instr.name} requires target field")
        return self._formatter.format(instr.name, uses=[instr.target])

    def _handle_compare(self, instr: KInstr) -> str:
        """Handle comparison instruction."""
        if len(instr.uses) >= 2:
            op0 = self._resolve_operand(instr.uses[0])
            op1 = self._resolve_operand(instr.uses[1])
            return self._formatter.format(
                "s_cmp_lt_u32", uses=[op0, op1], comment=instr.comment
            )
        return None

    def _handle_loop_inc(self, instr: KInstr) -> str:
        """Handle loop increment (counter = counter + step)."""
        # The counter is both defined and used - use the def for the destination
        if len(instr.defs) >= 1 and len(instr.uses) >= 2:
            counter = self._resolve_operand(instr.defs[0])
            counter_use = self._resolve_operand(instr.uses[0])
            step = self._resolve_operand(instr.uses[1])
            return self._formatter.format(
                "s_add_u32",
                defs=[counter],
                uses=[counter_use, step],
                comment=instr.comment,
            )
        # Fallback for legacy format (no def, just uses)
        elif len(instr.uses) >= 2:
            counter = self._resolve_operand(instr.uses[0])
            step = self._resolve_operand(instr.uses[1])
            return self._formatter.format(
                "s_add_u32", defs=[counter], uses=[counter, step], comment=instr.comment
            )
        return None

    def _handle_mfma_acc(self, instr: KInstr) -> str:
        """Handle MFMA 16x16x16 with accumulator (in-place update)."""
        if len(instr.uses) >= 3:
            acc = self._resolve_operand(instr.uses[0])
            a = self._resolve_operand(instr.uses[1])
            b = self._resolve_operand(instr.uses[2])
            return self._formatter.format(
                Instruction.V_MFMA_F32_16X16X16_F16,
                defs=[acc],
                uses=[a, b, acc],
                comment=instr.comment,
            )
        return None

    def _handle_mfma_acc_16x16x32(self, instr: KInstr) -> str:
        """Handle MFMA 16x16x32 with accumulator (in-place update)."""
        if len(instr.uses) >= 3:
            acc = self._resolve_operand(instr.uses[0])
            a = self._resolve_operand(instr.uses[1])
            b = self._resolve_operand(instr.uses[2])
            return self._formatter.format(
                Instruction.V_MFMA_F32_16X16X32_F16,
                defs=[acc],
                uses=[a, b, acc],
                comment=instr.comment,
            )
        return None

    def _handle_buffer_load_lds(self, instr: KInstr) -> str:
        """Handle buffer_load with LDS destination."""
        if len(instr.uses) >= 3:
            vaddr = self._resolve_operand(instr.uses[0])
            srd = (
                self._resolve_reg_range(instr.uses[1])
                if hasattr(instr.uses[1], "base_reg")
                else self._resolve_operand(instr.uses[1])
            )
            soffset = self._resolve_operand(instr.uses[2])

            # Determine mnemonic based on instruction name
            mnemonic = (
                "buffer_load_dword"
                if "dword_lds" in instr.name
                else "buffer_load_dwordx4"
            )

            return self._formatter.format(
                mnemonic + "_lds", uses=[vaddr, srd, soffset], comment=instr.comment
            )
        return None

    def _handle_init_acc_quad(self, instr: KInstr) -> List[str]:
        """Initialize accumulator quad to zero."""
        if len(instr.defs) >= 1:
            dst_range = instr.defs[0]
            base = self._resolve_operand(dst_range)

            # Parse the base register index
            if "[" in base:
                base_idx = int(base.split("[")[1].split(":")[0])
            else:
                base_idx = int(base[1:])

            lines = []
            for i in range(4):
                comment = instr.comment if i == 3 else None
                lines.append(
                    self._formatter.format(
                        "v_mov_b32",
                        defs=[f"v{base_idx + i}"],
                        uses=[0],
                        comment=comment,
                    )
                )
            return lines
        return None

    def _handle_noop(self, instr: KInstr) -> Optional[str]:
        """No-op pseudo instruction (emits no assembly)."""
        return None


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_program(program: KernelProgram, mapping: PhysicalMapping) -> str:
    """Generate assembly string from a kernel program."""
    return KernelGenerator(program, mapping).generate_to_string()


# =============================================================================
# Backwards Compatibility Aliases
# =============================================================================

KernelRenderer = KernelGenerator
render_program = lambda program, mapping: KernelGenerator(
    program, mapping
).generate_to_string()
