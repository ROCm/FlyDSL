# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from typing import Dict, List, Tuple

from .kernel_pipeline_shared import (
    KVReg,
    KSReg,
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

        # Apply rematerialization pass to reduce register pressure
        self._apply_rematerialization()

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

        Additionally, for CDNA3-class targets there are "trans" vector ops
        (exp/log/rcp/rsq/sqrt/sin/cos etc.) that may require an extra wait state
        when followed immediately by a non-trans consumer. While hardware often
        forwards results, ISA docs list these as software-inserted wait states.
        We conservatively insert `s_nop 0` after a trans op if the next
        instruction is a non-trans vector op.

        This implementation is precise: it only inserts s_nop when:
        1. Current instruction is a VALU that writes a VGPR
        2. Next instruction is v_readfirstlane_b32
        3. v_readfirstlane reads the VGPR that the VALU just wrote
        """
        instructions = self.program.instructions
        if not instructions:
            return

        # Helper: does an instruction read the special VCC register?
        def _reads_vcc(instr: KInstr) -> bool:
            from .kernel_ir import KSpecialReg

            for u in instr.uses:
                if isinstance(u, KSpecialReg) and u.name == "vcc":
                    return True
            return False

        # Trans ops that may require an extra wait before a non-trans consumer.
        # Keep this conservative and string-based to match instruction naming.
        trans_ops = {
            "v_exp_f32",
            "v_log_f32",
            "v_rcp_f32",
            "v_rcp_iflag_f32",
            "v_rsq_f32",
            "v_rcp_f64",
            "v_rsq_f64",
            "v_sqrt_f32",
            "v_sqrt_f64",
            "v_sin_f32",
            "v_cos_f32",
            "v_rcp_f16",
            "v_sqrt_f16",
            "v_rsq_f16",
            "v_log_f16",
            "v_exp_f16",
            "v_sin_f16",
            "v_cos_f16",
            "v_exp_legacy_f32",
            "v_log_legacy_f32",
        }

        # Find positions where we need to insert s_nop
        insertions: List[Tuple[int, int]] = []

        def _written_vgprs(instr: KInstr) -> set[int]:
            """Best-effort set of virtual VGPR ids written by this instruction."""
            written = set()
            for d in instr.defs:
                if isinstance(d, KVReg):
                    written.add(d.id)
                elif isinstance(d, KRegRange) and isinstance(d.base_reg, KVReg):
                    for k in range(d.count):
                        written.add(d.base_reg.id + k)
            return written

        def _reads_any_vgpr(instr: KInstr, vgprs: set[int]) -> bool:
            """Return True if instr uses any of the given virtual VGPR ids."""
            if not vgprs:
                return False
            for u in instr.uses:
                if isinstance(u, KVReg) and u.id in vgprs:
                    return True
                if isinstance(u, KRegRange) and isinstance(u.base_reg, KVReg):
                    for k in range(u.count):
                        if (u.base_reg.id + k) in vgprs:
                            return True
            return False

        def _writes_any_vgpr(instr: KInstr, vgprs: set[int]) -> bool:
            """Return True if instr defines any of the given virtual VGPR ids."""
            if not vgprs:
                return False
            for d in instr.defs:
                if isinstance(d, KVReg) and d.id in vgprs:
                    return True
                if isinstance(d, KRegRange) and isinstance(d.base_reg, KVReg):
                    for k in range(d.count):
                        if (d.base_reg.id + k) in vgprs:
                            return True
            return False

        # MFMA result hazard (non-local):
        # LLVM often inserts `s_nop 6` between an MFMA chain and the first consumer
        # that reads the accumulator VGPRs (e.g. a store). This consumer is not
        # necessarily the immediate next instruction (exec-mask control flow,
        # epilogue packing, etc. can be in between).
        #
        # We conservatively scan forward from each MFMA to the first non-MFMA
        # consumer of its written VGPRs, and insert `s_nop 6` right before it.
        # This is still "precise enough" for our IR because accumulator defs are
        # explicit and most kernels consume them shortly after the MFMA chain.
        mfma_insertions: List[Tuple[int, int]] = []
        for i, instr in enumerate(instructions):
            if not instr.is_mfma:
                continue
            written = _written_vgprs(instr)
            if not written:
                continue
            # Find first subsequent non-MFMA read of these defs.
            for j in range(i + 1, len(instructions)):
                nxt = instructions[j]
                # Skip no-op pseudos.
                if nxt.is_comment or nxt.is_label:
                    continue
                # Stop if accumulator is overwritten before any read (unlikely).
                if _writes_any_vgpr(nxt, written) and not _reads_any_vgpr(nxt, written):
                    break
                # Insert the wait right before the first consumer, including an MFMA
                # that consumes the accumulator written by the previous MFMA (RAW).
                if _reads_any_vgpr(nxt, written):
                    mfma_insertions.append((j, 6))
                    break

        if mfma_insertions:
            # De-dupe (same position may be discovered from multiple MFMA in a chain).
            seen = set()
            for pos, n in mfma_insertions:
                if (pos, n) in seen:
                    continue
                seen.add((pos, n))
                insertions.append((pos, n))

        for i in range(len(instructions) - 1):
            instr = instructions[i]
            next_instr = instructions[i + 1]

            # MFMA->MFMA accumulator RAW hazard:
            # Some targets require a software wait state between an MFMA that writes
            # an accumulator quad and the next MFMA that immediately consumes it.
            # LLVM sometimes relies on the backend/hardware scheduling; for our
            # assembler backend we insert a conservative wait.
            if instr.is_mfma and next_instr.is_mfma:
                written = _written_vgprs(instr)
                if _reads_any_vgpr(next_instr, written):
                    insertions.append((i + 1, 6))
                    continue

            # VALU that clobbers VCC -> immediate consumer that reads VCC.
            # On CDNA3+, ISA docs require software-inserted wait states here.
            # Match LLVM/hipcc practice: insert s_nop 1 (2 cycles).
            if instr.is_valu and instr.constraints.vcc_clobber and _reads_vcc(next_instr):
                insertions.append((i + 1, 1))  # (pos, snop_count)
                continue

            # Non-DLops VALU Write VGPR -> V_MFMA* read VGPR hazard.
            # Per AMD ISA Table 37, requires 2 wait states (s_nop 1).
            # This applies when a VALU instruction (like v_mov_b32) writes
            # a VGPR that is immediately read by an MFMA instruction.
            # Also applies to _pack_vgpr_pair pseudo which generates v_mov instructions.
            is_valu_like = instr.is_valu or instr.name in ("_pack_vgpr_pair", "_pack_vgpr_quad")
            if is_valu_like and next_instr.is_mfma:
                written = _written_vgprs(instr)
                if written and _reads_any_vgpr(next_instr, written):
                    insertions.append((i + 1, 1))  # s_nop 1 = 2 wait states
                    continue

            # Trans op -> non-trans VALU: only insert NOP if there's a real dependency.
            # Per Table 12: valu_trans_to_non_trans (1 wait state)
            # Optimized: only when non-trans VALU reads the result of trans op.
            if (
                isinstance(instr.name, str)
                and instr.name in trans_ops
                and next_instr.is_valu
                and not (isinstance(next_instr.name, str) and next_instr.name in trans_ops)
            ):
                written = _written_vgprs(instr)
                if written and _reads_any_vgpr(next_instr, written):
                    insertions.append((i + 1, 0))
                    continue

            instr_name_str = instr.name if isinstance(instr.name, str) else str(instr.name)
            next_name_str = next_instr.name if isinstance(next_instr.name, str) else str(next_instr.name)

            # V_CMPX* writes EXEC -> V_MFMA* requires 4 wait states.
            # Per Table 37: No exec mask forwarding for XDL/DGEMM.
            # This is always required since MFMA execution is masked by EXEC.
            if instr_name_str.startswith("v_cmpx") and next_instr.is_mfma:
                insertions.append((i + 1, 3))  # s_nop 3 = 4 wait states
                continue

            # VALU writes VGPR -> VALU DPP reads that VGPR requires 2 wait states.
            # Per Table 11: valu_vgpr_to_dpp
            # Optimized: only when DPP reads the VGPR that VALU just wrote.
            if instr.is_valu and "_dpp" in next_name_str.lower():
                written = _written_vgprs(instr)
                if written and _reads_any_vgpr(next_instr, written):
                    insertions.append((i + 1, 1))  # s_nop 1 = 2 wait states
                    continue

            # VALU writes SGPR/VCC -> V_{READ,WRITE}LANE using that as lane select.
            # Per Table 11: valu_sgpr_vcc_to_lane_select (4 wait states)
            # Optimized: only when readlane/writelane uses the SGPR that was just written.
            if instr.is_valu and getattr(instr.constraints, 'writes_sgpr', False):
                if next_name_str in ("v_readlane_b32", "v_writelane_b32"):
                    # Check if next instruction uses an SGPR that instr wrote
                    written_sgprs = self._get_written_sgprs(instr)
                    read_sgprs = self._get_read_sgprs(next_instr)
                    if written_sgprs & read_sgprs:
                        insertions.append((i + 1, 3))  # s_nop 3 = 4 wait states
                        continue

            # VALU writes VGPRn -> v_readlane vsrc0 reads VGPRn requires 1 wait state.
            # Per Table 11: valu_vgpr_to_readlane_vsrc0
            # Already optimized: checks register dependency.
            if instr.is_valu and next_name_str == "v_readlane_b32":
                written = _written_vgprs(instr)
                if written and _reads_any_vgpr(next_instr, written):
                    insertions.append((i + 1, 0))  # s_nop 0 = 1 wait state
                    continue

            # Check if next instruction is v_readfirstlane_b32
            if next_instr.name != Instruction.V_READFIRSTLANE_B32:
                continue

            # Check if current instruction is a VALU that writes a VGPR
            if not self._is_valu_vgpr_write(instr):
                continue

            # Check if the VALU writes to a register that readfirstlane reads
            if self._writes_to_readfirstlane_src(instr, next_instr):
                insertions.append((i + 1, 0))

        # Insert s_nop instructions in reverse order to preserve indices
        for idx, snop_count in reversed(insertions):
            instructions.insert(
                idx,
                KInstr(
                    Instruction.S_NOP,
                    (),
                    (KImm(snop_count),),
                    comment="hazard mitigation",
                ),
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

    def _get_written_sgprs(self, instr: KInstr) -> set:
        """Get the set of SGPR ids written by an instruction."""
        from .kernel_ir import KSReg

        written = set()
        for d in instr.defs:
            if isinstance(d, KSReg):
                written.add(d.id)
            elif isinstance(d, KRegRange) and isinstance(d.base_reg, KSReg):
                for k in range(d.count):
                    written.add(d.base_reg.id + k)
        return written

    def _get_read_sgprs(self, instr: KInstr) -> set:
        """Get the set of SGPR ids read by an instruction."""
        from .kernel_ir import KSReg

        read = set()
        for u in instr.uses:
            if isinstance(u, KSReg):
                read.add(u.id)
            elif isinstance(u, KRegRange) and isinstance(u.base_reg, KSReg):
                for k in range(u.count):
                    read.add(u.base_reg.id + k)
        return read

    def _apply_rematerialization(self):
        """
        Apply rematerialization to reduce register pressure.

        For registers defined by cheap instructions (MOV, ADD, MUL with immediates)
        that have very long live ranges, we can recompute them near their use points
        instead of keeping them alive across the entire function.

        This is critical for reducing VGPR pressure in GEMM kernels where:
        - Address computation constants are loaded at the start
        - Used only in the epilogue (after the main loop)
        - Creating 2000+ instruction live ranges

        Strategy:
        1. Identify "rematerializable" instructions (cheap ops with immediate/SGPR operands)
        2. Find registers with very long live ranges (>1000 instructions)
        3. For uses that occur much later than the def, insert a rematerialization
           before the use and update the use to reference the new register
        """
        from .kernel_liveness import compute_liveness, build_cfg, has_loops

        instructions = self.program.instructions
        if not instructions:
            return

        # Compute initial liveness
        liveness = compute_liveness(self.program, use_cfg=True)

        # Build CFG to identify loop regions
        cfg = build_cfg(self.program)
        loop_blocks = set()
        if has_loops(cfg):
            for block in cfg.blocks:
                for succ in block.successors:
                    if succ.id <= block.id:
                        # Back-edge: mark all blocks in loop
                        for b in cfg.blocks:
                            if succ.id <= b.id <= block.id:
                                loop_blocks.add(b.id)

        # Map instruction index to block
        idx_to_block = {}
        for block in cfg.blocks:
            for idx in range(block.start_idx, block.end_idx + 1):
                idx_to_block[idx] = block

        # Identify rematerializable instructions
        # These are cheap ops that can be recomputed without side effects
        REMAT_OPS = {
            'Instruction.V_MOV_B32', 'v_mov_b32',
            'v_add_u32', 'v_sub_u32',
            'v_mul_lo_u32', 'v_mul_hi_u32',
            'v_lshlrev_b32', 'v_lshrrev_b32', 'v_ashrrev_i32',
            'v_and_b32', 'v_or_b32', 'v_xor_b32',
        }

        def is_rematerializable(instr):
            """Check if instruction can be rematerialized."""
            if instr.name not in REMAT_OPS:
                return False
            # All operands must be immediates or SGPRs (not VGPRs that might change)
            for u in instr.uses:
                if isinstance(u, KVReg):
                    return False  # Depends on VGPR, can't remat safely
                if isinstance(u, KRegRange) and isinstance(u.base_reg, KVReg):
                    return False
            return True

        def get_single_def(instr):
            """Get the single VGPR defined by an instruction, or None."""
            if len(instr.defs) != 1:
                return None
            d = instr.defs[0]
            if isinstance(d, KVReg):
                return d
            if isinstance(d, KRegRange) and d.count == 1 and isinstance(d.base_reg, KVReg):
                return d.base_reg
            return None

        # Find candidates for rematerialization
        # Strategy: For registers defined before the loop with uses after the loop,
        # rematerialize ALL uses that are in the epilogue (outside the loop).
        # This maximally reduces the live range of the original register.
        REMAT_THRESHOLD = 500  # Minimum live range length to consider
        remat_candidates = {}  # vreg -> (def_idx, instruction, uses_outside_loop)

        # Find the end of the loop (max end_idx of loop blocks)
        loop_end_idx = 0
        for block in cfg.blocks:
            if block.id in loop_blocks:
                loop_end_idx = max(loop_end_idx, block.end_idx)

        for lr in liveness.vreg_ranges:
            if lr.size != 1:
                continue  # Only single VGPRs for now
            if (lr.end - lr.start) < REMAT_THRESHOLD:
                continue

            def_idx = lr.start
            if def_idx >= len(instructions):
                continue

            instr = instructions[def_idx]
            if not is_rematerializable(instr):
                continue

            vreg = get_single_def(instr)
            if vreg is None or vreg.id != lr.reg.id:
                continue

            # Check if def is before loop
            def_block = idx_to_block.get(def_idx)
            if def_block is None:
                continue

            def_in_loop = def_block.id in loop_blocks
            if def_in_loop:
                continue  # Only remat values defined outside the loop

            # Find ALL uses that are in the epilogue (outside the loop, after loop_end)
            uses = liveness.use_points.get(lr.reg, [])
            uses_to_remat = []
            for use_idx in uses:
                use_block = idx_to_block.get(use_idx)
                if use_block and use_block.id not in loop_blocks and use_idx > loop_end_idx:
                    uses_to_remat.append(use_idx)

            if uses_to_remat:
                remat_candidates[vreg] = (def_idx, instr, sorted(uses_to_remat))

        if not remat_candidates:
            return

        # Apply rematerialization: insert copies of cheap instructions before late uses
        # We need to be careful to update uses properly

        # Group uses by their earliest occurrence for batch insertion
        insertions = []  # (insert_before_idx, new_instr, old_vreg, new_vreg)

        for old_vreg, (def_idx, orig_instr, use_indices) in remat_candidates.items():
            # Create one rematerialization point for all late uses
            # Insert just before the first late use
            insert_idx = min(use_indices)

            # Allocate a new virtual register for the rematerialized value
            new_vreg = self.program.alloc_vreg()

            # Create a copy of the instruction with the new destination
            new_defs = (new_vreg,)
            new_instr = KInstr(
                orig_instr.name,
                new_defs,
                orig_instr.uses,
                comment=f"remat from {old_vreg}",
            )

            insertions.append((insert_idx, new_instr, old_vreg, new_vreg, use_indices))

        if not insertions:
            return

        # Sort by insertion point (descending) so we can insert without affecting
        # the indices of later insertions
        insertions.sort(key=lambda x: -x[0])

        # Apply insertions and update uses
        # Since we process in descending order, each insertion doesn't affect
        # earlier (higher) insertion points
        for insert_idx, new_instr, old_vreg, new_vreg, use_indices in insertions:
            # Insert the rematerialization instruction
            self.program.instructions.insert(insert_idx, new_instr)

            # Update ALL uses of old_vreg from insert_idx+1 onwards to new_vreg
            # This is safe because:
            # 1. The remat instruction is at insert_idx
            # 2. All uses after it should use the rematerialized value
            # 3. Uses before insert_idx (in the loop or prologue) keep using old_vreg
            for i in range(insert_idx + 1, len(self.program.instructions)):
                instr = self.program.instructions[i]

                # Check if this instruction uses old_vreg
                has_old_vreg = False
                for u in instr.uses:
                    if u == old_vreg:
                        has_old_vreg = True
                        break
                    if isinstance(u, KRegRange) and u.base_reg == old_vreg:
                        has_old_vreg = True
                        break

                if not has_old_vreg:
                    continue

                # Replace uses of old_vreg with new_vreg
                new_uses = []
                for u in instr.uses:
                    if u == old_vreg:
                        new_uses.append(new_vreg)
                    elif isinstance(u, KRegRange) and u.base_reg == old_vreg:
                        new_uses.append(KRegRange(new_vreg, u.count, u.alignment))
                    else:
                        new_uses.append(u)

                # Create updated instruction
                self.program.instructions[i] = KInstr(
                    instr.name,
                    instr.defs,
                    tuple(new_uses),
                    comment=instr.comment,
                )

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
