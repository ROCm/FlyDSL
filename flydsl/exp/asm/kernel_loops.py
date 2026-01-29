# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from typing import List, Union

from .kernel_pipeline_shared import KRegRange, KInstr, KImm
from .kernel_ir import KSReg
from .instruction_registry import Instruction


class _LoopSupport:
    # =========================================================================
    # Loop Support
    # =========================================================================

    def begin_loop(
        self,
        lower_bound: Union[int, KSReg],
        upper_bound: Union[int, KSReg],
        step: Union[int, KSReg],
    ) -> dict:
        """
        Begin a loop structure with virtual registers for loop control.

        Loop control registers (counter, step, upper_bound) use virtual SGPRs
        that are marked as loop control registers. This allows them to:
        - Be allocated by the register allocator (no hardcoded register numbers)
        - Be exempt from SSA validation (loop counters are re-defined in latch)

        Args:
            lower_bound: Loop start value (int constant or KSReg for dynamic)
            upper_bound: Loop end value (int constant or KSReg for dynamic)
            step: Loop step value (int constant or KSReg for dynamic)

        Returns a loop context dict for use with emit_loop_header/latch/end.
        """
        loop_id = self._loop_counter
        self._loop_counter += 1

        # Use VIRTUAL SGPRs for loop control - allocator picks physical regs
        # Each loop needs 3 SGPRs: counter, step, upper_bound
        counter_vreg = self.sreg()
        step_vreg = self.sreg()
        upper_bound_vreg = self.sreg()

        # Mark these as loop control registers (exempt from SSA validation)
        self.program.register_loop_control_sreg(counter_vreg)
        self.program.register_loop_control_sreg(step_vreg)
        self.program.register_loop_control_sreg(upper_bound_vreg)

        # Initialize loop counter and bounds using virtual registers
        self.comment(f"Initialize loop {loop_id}")

        # Helper to emit mov from either immediate or register
        from .kernel_ir import KVReg
        def emit_mov(dst, src, name, val_str):
            if isinstance(src, KSReg):
                # SGPR to SGPR copy
                self.program.emit(
                    KInstr(
                        "s_mov_b32",
                        (dst,),
                        (src,),
                        comment=f"loop {loop_id} {name} = dynamic (sgpr)",
                    )
                )
            elif isinstance(src, KVReg):
                # VGPR to SGPR: use v_readfirstlane_b32
                self.program.emit(
                    KInstr(
                        "v_readfirstlane_b32",
                        (dst,),
                        (src,),
                        comment=f"loop {loop_id} {name} = dynamic (vgpr)",
                    )
                )
            elif isinstance(src, int):
                # Static integer immediate
                self.program.emit(
                    KInstr(
                        "s_mov_b32",
                        (dst,),
                        (KImm(src),),
                        comment=f"loop {loop_id} {name} = {val_str}",
                    )
                )
            else:
                raise ValueError(f"Unsupported loop bound type: {type(src)}")

        emit_mov(counter_vreg, lower_bound, "counter", str(lower_bound))
        emit_mov(step_vreg, step, "step", str(step))
        emit_mov(upper_bound_vreg, upper_bound, "upper", str(upper_bound))

        # Store numeric values for static bounds, None for dynamic
        lower_val = lower_bound if isinstance(lower_bound, int) else None
        upper_val = upper_bound if isinstance(upper_bound, int) else None
        step_val = step if isinstance(step, int) else None

        loop_ctx = {
            "loop_id": loop_id,
            "counter_sreg": counter_vreg,
            "step_sreg": step_vreg,
            "upper_bound_sreg": upper_bound_vreg,
            "lower_bound": lower_val,
            "upper_bound": upper_val,
            "step": step_val,
            "is_dynamic": not all(isinstance(x, int) for x in [lower_bound, upper_bound, step]),
        }

        self._loop_stack.append(loop_ctx)
        return loop_ctx

    def emit_loop_header(self, loop_ctx: dict):
        """Emit loop header with comparison and conditional branch."""
        loop_id = loop_ctx["loop_id"]
        counter = loop_ctx["counter_sreg"]
        upper = loop_ctx["upper_bound_sreg"]

        # Emit label
        self.emit_label(f"loop_{loop_id}_header")

        # Compare counter < upper (sets SCC)
        self.program.emit(
            KInstr(
                "s_cmp_lt_u32",
                (),
                (counter, upper),
                comment=f"compare loop {loop_id} counter < upper",
            )
        )

        # Branch to body if SCC=1
        self.program.emit(
            KInstr(Instruction.S_CBRANCH_SCC1, (), (), target=f"loop_{loop_id}_body")
        )

        # Branch to exit if not taken
        self.program.emit(
            KInstr(Instruction.S_BRANCH, (), (), target=f"loop_{loop_id}_exit")
        )

        # Body label
        self.emit_label(f"loop_{loop_id}_body")
        # Structured loop marker (used by backend passes; emits no assembly).
        # Avoid relying on label comment parsing in later transformations.
        self.program.emit(
            KInstr("_loop_begin", defs=(), uses=(), comment=f"loop_{loop_id}")
        )

    def emit_loop_latch(self, loop_ctx: dict):
        """Emit loop latch (increment counter and branch back)."""
        loop_id = loop_ctx["loop_id"]
        counter = loop_ctx["counter_sreg"]
        step = loop_ctx["step_sreg"]

        # Latch label
        self.emit_label(f"loop_{loop_id}_latch")

        # Increment counter - counter is a loop control register so SSA
        # validation will allow this redefinition
        # s_add_u32: counter = counter + step
        self.program.emit(
            KInstr(
                Instruction._LOOP_INC,
                (counter,),  # def: counter (redefinition allowed for loop regs)
                (counter, step),  # uses: counter, step
                comment=f"loop {loop_id} counter += step",
            )
        )

        # Branch back to header
        self.program.emit(
            KInstr(Instruction.S_BRANCH, (), (), target=f"loop_{loop_id}_header")
        )

    def end_loop(self):
        """End loop and emit exit label."""
        loop_ctx = self._loop_stack.pop()
        loop_id = loop_ctx["loop_id"]
        # Structured loop marker (used by backend passes; emits no assembly).
        self.program.emit(
            KInstr("_loop_end", defs=(), uses=(), comment=f"loop_{loop_id}")
        )
        self.emit_label(f"loop_{loop_id}_exit")

    def alloc_accumulators(self, count: int) -> List[KRegRange]:
        """
        Allocate accumulator VGPR quads for loop iter_args.

        Returns list of KRegRange quads, each initialized to 0.
        """
        accumulators = []
        for i in range(count):
            quad = self.vreg_quad()
            accumulators.append(quad)
            # Mark as accumulator regs: MFMA updates these in-place (RMW semantics),
            # so they are exempt from strict SSA validation.
            self.program.register_accumulator_vreg_range(quad)

            # Initialize to 0 using a pseudo-instruction that defines the whole range
            # This ensures liveness analysis sees the quad with proper alignment
            self.program.emit(
                KInstr(
                    "_init_acc_quad",
                    (quad,),
                    (KImm(0),),
                    comment=f"Initialize accumulator {i} to 0.0",
                )
            )

        return accumulators
