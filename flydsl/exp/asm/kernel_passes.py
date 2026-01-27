# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from typing import Dict, List, Tuple

from .kernel_pipeline_shared import (
    KVReg,
    KPhysVReg,
    KInstr,
    KImm,
    KRegRange,
    UnifiedEmitter,
    allocate_kernel,
    AllocationStats,
    KernelGenerator,
)
from .instruction_registry import Instruction


class _CompilationPasses:
    @property
    def unified(self) -> UnifiedEmitter:
        """
        Get the unified emitter for this context.

        This provides a consistent API with the legacy unified-emission surface, allowing
        callers to use kernel_ctx.unified.v_add_u32(...) syntax.

        When using the unified emitter:
        - Methods that exist on KernelCompilationContext are called directly
        - Methods that don't exist fall back to emit_raw()
        - Virtual registers are returned for instructions with destinations

        Example:
            result = kernel_ctx.unified.v_add_u32(src0, src1, comment="add")
        """
        return self._unified

    # =========================================================================
    # Finalization
    # =========================================================================

    def finalize(self) -> Tuple[List[str], AllocationStats]:
        """
        Finalize the kernel program and generate assembly.

        This:
        1. Emits s_endpgm at the end
        2. Emits SRD prologue (all SRD setup at program start)
        3. Applies peephole optimizations
        4. Applies accumulator-init optimization (IR-to-IR)
        5. Inserts/coalesces waitcnt via ticketing (IR-to-IR)
        6. Applies hazard mitigation (inserts s_nop where needed)
        7. Computes liveness for all virtual registers
        8. Runs linear scan allocation
        9. Renders to assembly

        Returns:
            Tuple of (assembly lines, allocation statistics)
        """
        # Emit s_endpgm at the end of the program (if not already there)
        if (
            not self.program.instructions
            or self.program.instructions[-1].name != Instruction.S_ENDPGM
        ):
            self.program.emit(KInstr(Instruction.S_ENDPGM, defs=(), uses=()))

        # Emit SRD prologue - moves all SRD setup to program start
        self._emit_srd_prologue()

        # Apply peephole optimizations (fuse lshl+add, lshl+or, etc.)
        self._apply_peephole_optimizations()

        # Apply accumulator-init optimization (explicit pass; runs before ticketing
        # so downstream analyses see the final, semantically accurate stream).
        self._optimize_accumulator_init()

        # Optionally insert/coalesce waitcnt using ticketing.
        # Must run before hazard mitigation/regalloc so liveness sees the final stream.
        self._apply_ticketing_waitcnt_placement()

        # Apply hazard mitigation pass
        self._apply_hazard_mitigation()

        # Get reserved registers from ABI
        reserved_vgprs = self.program.abi.get_reserved_vgprs()
        reserved_sgprs = self.program.abi.get_reserved_sgprs()

        # Note: Loop control registers are now virtual SGPRs that go through
        # normal register allocation. They are marked as loop control registers
        # in the program, which exempts them from SSA validation (allowing the
        # redefinition in the loop latch). No manual reservation needed.

        # Allocate
        mapping, stats = allocate_kernel(
            self.program,
            reserved_vgprs=reserved_vgprs,
            reserved_sgprs=reserved_sgprs,
            precolored_sregs=self._precolored_sregs if self._precolored_sregs else None,
        )

        # Render
        generator = KernelGenerator(self.program, mapping)
        asm_lines = generator.generate()

        return asm_lines, stats

    def _apply_hazard_mitigation(self):
        """
        Apply precise hazard mitigation to the program.

        On gfx940+ (CDNA3/4), there's a hazard when v_readfirstlane_b32
        immediately reads a VGPR that was just written by a VALU instruction.
        This requires 1 wait state (s_nop 0) between them.

        This implementation is precise: it only inserts s_nop when:
        1. Current instruction is a VALU that writes a VGPR
        2. Next instruction is v_readfirstlane_b32
        3. v_readfirstlane reads the VGPR that the VALU just wrote
        """
        instructions = self.program.instructions
        if not instructions:
            return

        # Find positions where we need to insert s_nop
        insertions = []

        for i in range(len(instructions) - 1):
            instr = instructions[i]
            next_instr = instructions[i + 1]

            # Check if next instruction is v_readfirstlane_b32
            if next_instr.name != Instruction.V_READFIRSTLANE_B32:
                continue

            # Check if current instruction is a VALU that writes a VGPR
            if not self._is_valu_vgpr_write(instr):
                continue

            # Check if the VALU writes to a register that readfirstlane reads
            if self._writes_to_readfirstlane_src(instr, next_instr):
                insertions.append(i + 1)

        # Insert s_nop instructions in reverse order to preserve indices
        for idx in reversed(insertions):
            instructions.insert(
                idx,
                KInstr(Instruction.S_NOP, (), (KImm(0),), comment="hazard mitigation"),
            )

    def _is_valu_vgpr_write(self, instr: KInstr) -> bool:
        """Check if instruction is a VALU that writes a VGPR."""
        # Must be a VALU instruction (not MFMA which also writes VGPRs)
        if not instr.is_valu:
            return False
        # Must have at least one def (destination)
        if not instr.defs:
            return False
        # Exclude v_readfirstlane (reads VGPR, writes SGPR)
        if instr.name == Instruction.V_READFIRSTLANE_B32:
            return False
        return True

    def _writes_to_readfirstlane_src(
        self, valu_instr: KInstr, readfirstlane_instr: KInstr
    ) -> bool:
        """Check if VALU writes to a VGPR that v_readfirstlane reads."""
        if not valu_instr.defs or not readfirstlane_instr.uses:
            return False

        # Get the VGPR(s) written by the VALU
        written_regs = set()
        for def_reg in valu_instr.defs:
            if isinstance(def_reg, KVReg):
                written_regs.add(def_reg.id)
            elif isinstance(def_reg, KRegRange) and isinstance(def_reg.base_reg, KVReg):
                for i in range(def_reg.count):
                    written_regs.add(def_reg.base_reg.id + i)

        # Get the VGPR read by v_readfirstlane (first use operand)
        for use_op in readfirstlane_instr.uses:
            if isinstance(use_op, KVReg):
                if use_op.id in written_regs:
                    return True
            elif isinstance(use_op, KPhysVReg):
                # Physical VGPR - would need physical mapping to check
                # Conservative: return True if any VGPR was written
                if written_regs:
                    return True

        return False

    def _apply_peephole_optimizations(self):
        """
        Apply peephole optimizations to fuse instruction sequences.

        Fuses patterns like:
        - v_lshlrev_b32 + v_add_u32 -> v_lshl_add_u32
        - v_lshlrev_b32 + v_or_b32 -> v_lshl_or_b32

        These fused instructions are supported on gfx9+ and save VALU cycles.
        """
        instructions = self.program.instructions
        if not instructions:
            return

        # Track which registers are written by which instruction index
        # This helps us find the producer of a register
        reg_writers: Dict[int, int] = {}  # vreg_id -> instruction_index

        # First pass: build def map
        for i, instr in enumerate(instructions):
            for def_reg in instr.defs:
                if isinstance(def_reg, KVReg):
                    reg_writers[def_reg.id] = i

        # Second pass: find fusion opportunities
        # We'll mark instructions to delete and create replacements
        to_delete = set()
        replacements = []  # (index, new_instr)

        for i, instr in enumerate(instructions):
            if i in to_delete:
                continue

            # Pattern: v_add_u32 vD, vA, vB where vA was produced by v_lshlrev_b32
            # Fuse to: v_lshl_add_u32 vD, src, shift, vB
            if (
                instr.name == Instruction.V_ADD_U32
                and len(instr.uses) == 2
                and len(instr.defs) == 1
            ):
                dst = instr.defs[0]
                src_a, src_b = instr.uses

                # Check if src_a is a VGPR produced by a v_lshlrev_b32
                if isinstance(src_a, KVReg) and src_a.id in reg_writers:
                    shift_idx = reg_writers[src_a.id]
                    shift_instr = instructions[shift_idx]

                    if (
                        shift_instr.name == Instruction.V_LSHLREV_B32
                        and len(shift_instr.uses) == 2
                        and isinstance(shift_instr.uses[0], KImm)
                        and shift_idx not in to_delete
                    ):

                        shift_amt = shift_instr.uses[0]
                        shift_src = shift_instr.uses[1]

                        # Check that the shift result isn't used elsewhere
                        # (for simplicity, we only fuse if the shift result is used once)
                        shift_result = shift_instr.defs[0]
                        uses_of_shift = sum(
                            1
                            for j, other in enumerate(instructions)
                            if j != i and j not in to_delete
                            for u in other.uses
                            if isinstance(u, KVReg) and u.id == shift_result.id
                        )

                        if uses_of_shift == 0:
                            # Can fuse!
                            # v_lshl_add_u32 vD, src, shift, addend
                            fused = KInstr(
                                Instruction.V_LSHL_ADD_U32,
                                (dst,),
                                (shift_src, shift_amt, src_b),
                                comment=f"fused: ({shift_src} << {shift_amt.value}) + {src_b}",
                            )
                            to_delete.add(shift_idx)
                            replacements.append((i, fused))
                            continue

                # Check if src_b is a VGPR produced by a v_lshlrev_b32 (commutative)
                if isinstance(src_b, KVReg) and src_b.id in reg_writers:
                    shift_idx = reg_writers[src_b.id]
                    shift_instr = instructions[shift_idx]

                    if (
                        shift_instr.name == Instruction.V_LSHLREV_B32
                        and len(shift_instr.uses) == 2
                        and isinstance(shift_instr.uses[0], KImm)
                        and shift_idx not in to_delete
                    ):

                        shift_amt = shift_instr.uses[0]
                        shift_src = shift_instr.uses[1]

                        shift_result = shift_instr.defs[0]
                        uses_of_shift = sum(
                            1
                            for j, other in enumerate(instructions)
                            if j != i and j not in to_delete
                            for u in other.uses
                            if isinstance(u, KVReg) and u.id == shift_result.id
                        )

                        if uses_of_shift == 0:
                            # Can fuse!
                            fused = KInstr(
                                Instruction.V_LSHL_ADD_U32,
                                (dst,),
                                (shift_src, shift_amt, src_a),
                                comment=f"fused: ({shift_src} << {shift_amt.value}) + {src_a}",
                            )
                            to_delete.add(shift_idx)
                            replacements.append((i, fused))
                            continue

            # Pattern: v_or_b32 vD, vA, vB where vA was produced by v_lshlrev_b32
            # Fuse to: v_lshl_or_b32 vD, src, shift, vB
            if (
                instr.name == Instruction.V_OR_B32
                and len(instr.uses) == 2
                and len(instr.defs) == 1
            ):
                dst = instr.defs[0]
                src_a, src_b = instr.uses

                # Check if src_a is a VGPR produced by a v_lshlrev_b32
                if isinstance(src_a, KVReg) and src_a.id in reg_writers:
                    shift_idx = reg_writers[src_a.id]
                    shift_instr = instructions[shift_idx]

                    if (
                        shift_instr.name == Instruction.V_LSHLREV_B32
                        and len(shift_instr.uses) == 2
                        and isinstance(shift_instr.uses[0], KImm)
                        and shift_idx not in to_delete
                    ):

                        shift_amt = shift_instr.uses[0]
                        shift_src = shift_instr.uses[1]

                        shift_result = shift_instr.defs[0]
                        uses_of_shift = sum(
                            1
                            for j, other in enumerate(instructions)
                            if j != i and j not in to_delete
                            for u in other.uses
                            if isinstance(u, KVReg) and u.id == shift_result.id
                        )

                        if uses_of_shift == 0:
                            fused = KInstr(
                                Instruction.V_LSHL_OR_B32,
                                (dst,),
                                (shift_src, shift_amt, src_b),
                                comment=f"fused: ({shift_src} << {shift_amt.value}) | {src_b}",
                            )
                            to_delete.add(shift_idx)
                            replacements.append((i, fused))
                            continue

        # Apply replacements and deletions
        if replacements or to_delete:
            # Build new instruction list
            new_instructions = []
            replace_map = {idx: instr for idx, instr in replacements}

            for i, instr in enumerate(instructions):
                if i in to_delete:
                    continue
                if i in replace_map:
                    new_instructions.append(replace_map[i])
                else:
                    new_instructions.append(instr)

            self.program.instructions = new_instructions

        # Note: accumulator-init optimization is run as a dedicated pass from finalize().

    def _optimize_accumulator_init(self):
        """
        Optimize accumulator initialization by using immediate 0 in the first MFMA.

        Pattern:
        - _init_acc_quad defines a quad (kv0..kv3)
        - _mfma_acc or _mfma_acc_16x16x32 uses that quad as accumulator

        Optimization:
        - Delete _init_acc_quad (saves 4 v_mov_b32 instructions)
        - Replace first _mfma_acc with _mfma_zero_acc (uses immediate 0)

        This optimization is similar to what LLVM does - it uses immediate 0 for
        the accumulator in the first MFMA instruction instead of explicitly
        initializing registers to zero.

        IMPORTANT: This optimization is ONLY safe for MFMA instructions that are
        NOT inside a loop. If an MFMA is inside a loop, it will execute multiple
        times, and only the first execution should use 0 as accumulator; subsequent
        executions must use the accumulated result from previous iterations.
        """
        instructions = self.program.instructions
        if not instructions:
            return

        # First, identify which instruction indices are inside loops using
        # structured loop markers (avoid parsing label comment strings).
        in_loop_depth = 0
        inside_loop: Dict[int, bool] = {}  # instruction_index -> is_inside_loop
        saw_loop_marker = False

        for i, instr in enumerate(instructions):
            if instr.name == "_loop_begin":
                saw_loop_marker = True
                in_loop_depth += 1
            elif instr.name == "_loop_end":
                saw_loop_marker = True
                if in_loop_depth <= 0:
                    raise ValueError(
                        "Malformed loop markers: encountered _loop_end with depth 0"
                    )
                in_loop_depth -= 1

            # Mark current instruction
            inside_loop[i] = in_loop_depth > 0

        if saw_loop_marker and in_loop_depth != 0:
            raise ValueError(
                f"Malformed loop markers: finished scan with depth={in_loop_depth} (expected 0)"
            )

        # Track accumulator quads defined by _init_acc_quad
        # Maps base_vreg_id -> instruction_index
        acc_init_map: Dict[int, int] = {}

        # Track first MFMA use of each accumulator (only if NOT inside a loop)
        # Maps base_vreg_id -> instruction_index
        first_mfma_use: Dict[int, int] = {}

        # First pass: find _init_acc_quad instructions
        for i, instr in enumerate(instructions):
            if instr.name == "_init_acc_quad" and len(instr.defs) >= 1:
                acc_range = instr.defs[0]
                if hasattr(acc_range, "base_reg") and isinstance(
                    acc_range.base_reg, KVReg
                ):
                    base_id = acc_range.base_reg.id
                    acc_init_map[base_id] = i

        if not acc_init_map:
            return  # No accumulators to optimize

        # Second pass: find first MFMA that uses each accumulator
        # ONLY optimize if the MFMA is NOT inside a loop!
        for i, instr in enumerate(instructions):
            if instr.name in ("_mfma_acc", "_mfma_acc_16x16x32"):
                if len(instr.uses) >= 1:
                    acc_range = instr.uses[0]
                    if hasattr(acc_range, "base_reg") and isinstance(
                        acc_range.base_reg, KVReg
                    ):
                        base_id = acc_range.base_reg.id
                        is_in_loop = inside_loop.get(i, False)
                        # Only track if this is an initialized accumulator, first use, AND NOT inside a loop
                        if (
                            base_id in acc_init_map
                            and base_id not in first_mfma_use
                            and not is_in_loop
                        ):
                            first_mfma_use[base_id] = i

        if not first_mfma_use:
            return  # No MFMA uses found (or all are inside loops)

        # Replace:
        #   _init_acc_quad (explicit v_mov zeroing) + first _mfma_acc (in-place pseudo)
        # With:
        #   first v_mfma_* that DEFINES the accumulator registers and uses KImm(0)
        #   as the accumulator operand (semantic transparency at IR level).
        #
        # We delete the init (removes the v_mov sequence) and replace the first MFMA
        # pseudo with a real MFMA instruction that defines the accumulator range.
        to_delete = set()
        replacements = {}  # index -> new_instr

        for base_id, mfma_idx in first_mfma_use.items():
            init_idx = acc_init_map[base_id]
            mfma_instr = instructions[mfma_idx]

            to_delete.add(init_idx)

            if len(mfma_instr.uses) < 3:
                raise ValueError(
                    f"Malformed MFMA instruction: expected 3 uses, got {len(mfma_instr.uses)}"
                )

            acc_range = mfma_instr.uses[0]
            a_range = mfma_instr.uses[1]
            b_range = mfma_instr.uses[2]

            if mfma_instr.name == "_mfma_acc":
                isa_name = Instruction.V_MFMA_F32_16X16X16_F16
            else:
                isa_name = Instruction.V_MFMA_F32_16X16X32_F16

            new_mfma = KInstr(
                isa_name,
                (acc_range,),  # define accumulator range here (dominates later uses)
                (a_range, b_range, KImm(0)),
                comment="MFMA first use (acc=0, optimized init)",
            )
            replacements[mfma_idx] = new_mfma

        # Apply changes
        if to_delete or replacements:
            new_instructions = []
            for i, instr in enumerate(instructions):
                if i in to_delete:
                    continue
                repl = replacements.get(i)
                new_instructions.append(repl if repl is not None else instr)
            self.program.instructions = new_instructions

    def finalize_to_string(self) -> str:
        """Finalize and return assembly as a single string."""
        lines, _ = self.finalize()
        return "\n".join(lines)

    # =========================================================================
    # Statistics
    # =========================================================================

    @property
    def num_instructions(self) -> int:
        return len(self.program)

    @property
    def num_virtual_vregs(self) -> int:
        return self.program._next_vreg_id

    @property
    def num_virtual_sregs(self) -> int:
        return self.program._next_sreg_id

    @property
    def cse_hit_count(self) -> int:
        return self._cse_hits
