# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from typing import Optional, Tuple

from .kernel_pipeline_shared import KReg, KVReg, KRegRange, KInstr, KImm
from .instruction_registry import Instruction


class _MFMASupport:
    # =========================================================================
    # MFMA Support
    # =========================================================================

    def emit_mfma_f32_16x16x16_f16(
        self,
        a_regs: Tuple[KReg, ...],
        b_regs: Tuple[KReg, ...],
        acc_regs: Optional[Tuple[KReg, ...]] = None,
    ) -> Tuple[KReg, ...]:
        """
        Emit MFMA instruction with virtual register tracking.

        Args:
            a_regs: Tuple of 2 VGPRs for A operand (f16x2)
            b_regs: Tuple of 2 VGPRs for B operand (f16x2)
            acc_regs: Optional tuple of 4 VGPRs for accumulator (f32x4)
                      If None, allocates new result registers

        Returns:
            Tuple of 4 VGPRs containing the result
        """
        # NOTE: Wait for LDS reads is now handled by the ticketing pass
        # (_apply_ticketing_waitcnt_placement) based on register dependencies.
        # This avoids redundant consecutive lgkmcnt(0) instructions.

        # Build operand ranges
        a_range = KRegRange(a_regs[0], 2, alignment=2) if len(a_regs) >= 2 else None
        b_range = KRegRange(b_regs[0], 2, alignment=2) if len(b_regs) >= 2 else None

        # Determine result/accumulator
        if acc_regs is not None and len(acc_regs) == 4:
            # Use provided accumulator as both input and output
            result_regs = acc_regs
            acc_range = KRegRange(acc_regs[0], 4, alignment=4)
            # Ensure accumulator regs are marked as such for SSA validation.
            if isinstance(acc_range.base_reg, KVReg):
                self.program.register_accumulator_vreg_range(acc_range)

            # MFMA with accumulator: v_mfma dst, a, b, acc
            # Note: MFMA updates the accumulator in-place (read-modify-write).
            # We model this by defining the accumulator range as a def.
            self.program.emit(
                KInstr(
                    "_mfma_acc",  # Pseudo: uses accumulator, doesn't define new regs
                    (acc_range,),  # def: accumulator range (RMW)
                    (acc_range, a_range, b_range),
                    comment="MFMA with accumulator (in-place)",
                )
            )
        else:
            # Allocate new quad for result, use 0 as accumulator
            result_range = self.vreg_quad()
            self.program.register_accumulator_vreg_range(result_range)
            result_regs = tuple(KVReg(result_range.base_reg.id + i) for i in range(4))

            # MFMA with zero accumulator: v_mfma dst, a, b, 0
            self.program.emit(
                KInstr(
                    Instruction.V_MFMA_F32_16X16X16_F16,
                    (result_range,),
                    (a_range, b_range, KImm(0)),
                    comment="MFMA with zero accumulator",
                )
            )

        return result_regs

    def emit_mfma_f32_16x16x32_f16(
        self,
        a_regs: Tuple[KReg, ...],
        b_regs: Tuple[KReg, ...],
        acc_regs: Optional[Tuple[KReg, ...]] = None,
    ) -> Tuple[KReg, ...]:
        """
        Emit MFMA 16x16x32 instruction with virtual register tracking.

        For 16x16x32: A needs 8 x f16 (4 VGPRs), B needs 8 x f16 (4 VGPRs),
        result is 4 x f32 (4 VGPRs).

        Args:
            a_regs: Tuple of 4 VGPRs for A operand (f16x8)
            b_regs: Tuple of 4 VGPRs for B operand (f16x8)
            acc_regs: Optional tuple of 4 VGPRs for accumulator (f32x4)
                      If None, allocates new result registers

        Returns:
            Tuple of 4 VGPRs containing the result
        """
        # Build operand ranges - 16x16x32 needs 4 VGPRs for A and B
        a_range = KRegRange(a_regs[0], 4, alignment=4) if len(a_regs) >= 4 else None
        b_range = KRegRange(b_regs[0], 4, alignment=4) if len(b_regs) >= 4 else None

        # Determine result/accumulator
        if acc_regs is not None and len(acc_regs) == 4:
            # Use provided accumulator as both input and output
            result_regs = acc_regs
            acc_range = KRegRange(acc_regs[0], 4, alignment=4)
            if isinstance(acc_range.base_reg, KVReg):
                self.program.register_accumulator_vreg_range(acc_range)

            # MFMA with accumulator: v_mfma dst, a, b, acc
            self.program.emit(
                KInstr(
                    "_mfma_acc_16x16x32",  # Pseudo: in-place accumulator update
                    (acc_range,),  # def: accumulator range (RMW)
                    (acc_range, a_range, b_range),
                    comment="MFMA 16x16x32 with accumulator (in-place)",
                )
            )
        else:
            # Allocate new quad for result, use 0 as accumulator
            result_range = self.vreg_quad()
            self.program.register_accumulator_vreg_range(result_range)
            result_regs = tuple(KVReg(result_range.base_reg.id + i) for i in range(4))

            # MFMA with zero accumulator: v_mfma dst, a, b, 0
            self.program.emit(
                KInstr(
                    Instruction.V_MFMA_F32_16X16X32_F16,
                    (result_range,),
                    (a_range, b_range, KImm(0)),
                    comment="MFMA 16x16x32 with zero accumulator",
                )
            )

        return result_regs
