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

    def emit_mfma_f32_16x16x32_fp8_fp8(
        self,
        a_regs: Tuple[KReg, ...],
        b_regs: Tuple[KReg, ...],
        acc_regs: Optional[Tuple[KReg, ...]] = None,
    ) -> Tuple[KReg, ...]:
        """
        Emit MFMA FP8xFP8 -> FP32 (16x16x32) with virtual register tracking.

        Per ISA defs (gfx942+): a = vgpr_pair, b = vgpr_pair, c/dst = vgpr_quad.
        """
        # Operand ranges: fp8 path uses 2 VGPRs per operand.
        # Hardware requires 64-bit aligned vgpr_pair operands, so pack into
        # aligned temporaries to satisfy assembler constraints.
        if len(a_regs) < 2 or len(b_regs) < 2:
            raise ValueError("FP8 MFMA expects 2 VGPRs per operand (vgpr_pair).")

        a_tmp = self.vreg_pair()
        b_tmp = self.vreg_pair()
        self.program.emit(
            KInstr(
                "_pack_vgpr_pair",
                (a_tmp,),
                (a_regs[0], a_regs[1]),
                comment="pack mfma a",
            )
        )
        self.program.emit(
            KInstr(
                "_pack_vgpr_pair",
                (b_tmp,),
                (b_regs[0], b_regs[1]),
                comment="pack mfma b",
            )
        )
        a_range = a_tmp
        b_range = b_tmp

        # Model MFMA as SSA-correct by default:
        # - The result is a NEW quad (defs)
        # - The accumulator operand (c) is a separate quad (uses)
        #
        # This matches LLVM's common lowering for gfx94x (dst != c), and avoids
        # incorrect behavior when the incoming accumulator value is still needed
        # after the MFMA (which is common in debugging and some schedules).
        #
        # We still treat MFMA result quads as "accumulator vregs" for SSA validation
        # purposes (they may be redefined across a chain), but we do not force in-place.
        
        acc_range = None
        if (
            acc_regs is not None
            and len(acc_regs) == 4
            and all(isinstance(r, KVReg) for r in acc_regs)
        ):
            base_id = acc_regs[0].id
            check1 = self.program._vreg_range_bases.get(base_id) == 4
            check2 = all(self.program.is_accumulator_vreg(r) for r in acc_regs)
            check3 = all(acc_regs[i].id == base_id + i for i in range(4))
            if check1 and check2 and check3:
                acc_range = KRegRange(acc_regs[0], 4, alignment=4)
                self.program.register_accumulator_vreg_range(acc_range)
        if acc_range is None and acc_regs is not None and len(acc_regs) == 4:
            # Pack any 4-lane accumulator (e.g. constant vectors) into a fresh quad.
            acc_range = self.vreg_quad()
            self.program.register_accumulator_vreg_range(acc_range)
            self.program.emit(
                KInstr(
                    "_pack_vgpr_quad",
                    (acc_range,),
                    (acc_regs[0], acc_regs[1], acc_regs[2], acc_regs[3]),
                    comment="pack mfma acc",
                )
            )

        # Emit the MFMA.
        #
        # MFMA is a read-modify-write operation that updates the accumulator in place.
        # When we have a valid accumulator (acc_range), we should REUSE it as the result
        # to avoid allocating new registers for each MFMA in a chain.
        #
        # This is critical for register pressure: a 64x256 tile has 64 MFMA tiles,
        # each needing 4 VGPRs. Without in-place accumulation, we'd need 256 VGPRs
        # just for accumulators in ONE k-iteration, let alone pipelined stages.

        if acc_regs is None:
            # First MFMA in chain: allocate new accumulator, use c=0
            from .kernel_ir import KImm

            result_range = self.vreg_quad()
            self.program.register_accumulator_vreg_range(result_range)
            result_regs = tuple(KVReg(result_range.base_reg.id + i) for i in range(4))

            self.program.emit(
                KInstr(
                    "v_mfma_f32_16x16x32_fp8_fp8",
                    (result_range,),
                    (a_range, b_range, KImm(0)),
                    comment="MFMA FP8 16x16x32 (c=0)",
                )
            )
        elif acc_range is not None:
            # Accumulating MFMA: REUSE the accumulator as both input and output.
            # This is the correct hardware behavior - MFMA updates accumulator in-place.
            result_range = acc_range
            result_regs = tuple(KVReg(acc_range.base_reg.id + i) for i in range(4))

            self.program.emit(
                KInstr(
                    "v_mfma_f32_16x16x32_fp8_fp8",
                    (result_range,),  # def: same as accumulator
                    (a_range, b_range, acc_range),  # use: accumulator
                    comment="MFMA FP8 16x16x32 (in-place)",
                )
            )
        else:
            # Accumulator was provided but wasn't a proper range - allocate new
            result_range = self.vreg_quad()
            self.program.register_accumulator_vreg_range(result_range)
            result_regs = tuple(KVReg(result_range.base_reg.id + i) for i in range(4))

            # Need to copy acc_regs to result_range first
            self.program.emit(
                KInstr(
                    "_pack_vgpr_quad",
                    (result_range,),
                    (acc_regs[0], acc_regs[1], acc_regs[2], acc_regs[3]),
                    comment="pack mfma acc (unaligned input)",
                )
            )
            self.program.emit(
                KInstr(
                    "v_mfma_f32_16x16x32_fp8_fp8",
                    (result_range,),
                    (a_range, b_range, result_range),
                    comment="MFMA FP8 16x16x32",
                )
            )

        return result_regs

    def emit_mfma_f32_16x16x16_bf16(
        self,
        a_regs: Tuple[KReg, ...],
        b_regs: Tuple[KReg, ...],
        acc_regs: Optional[Tuple[KReg, ...]] = None,
    ) -> Tuple[KReg, ...]:
        """
        Emit MFMA BF16xBF16 -> FP32 (16x16x16) with virtual register tracking.

        Per ISA defs (gfx942+): a = vgpr_pair (4 bf16s), b = vgpr_pair, c/dst = vgpr_quad.
        """
        if len(a_regs) < 2 or len(b_regs) < 2:
            raise ValueError("BF16 MFMA 16x16x16 expects 2 VGPRs per operand (vgpr_pair, 4 bf16s).")

        # Pack operands into aligned pairs
        a_tmp = self.vreg_pair()
        b_tmp = self.vreg_pair()
        self.program.emit(
            KInstr(
                "_pack_vgpr_pair",
                (a_tmp,),
                (a_regs[0], a_regs[1]),
                comment="pack mfma a",
            )
        )
        self.program.emit(
            KInstr(
                "_pack_vgpr_pair",
                (b_tmp,),
                (b_regs[0], b_regs[1]),
                comment="pack mfma b",
            )
        )
        a_range = a_tmp
        b_range = b_tmp

        acc_range = None
        if (
            acc_regs is not None
            and len(acc_regs) == 4
            and all(isinstance(r, KVReg) for r in acc_regs)
        ):
            base_id = acc_regs[0].id
            check1 = self.program._vreg_range_bases.get(base_id) == 4
            check2 = all(self.program.is_accumulator_vreg(r) for r in acc_regs)
            check3 = all(acc_regs[i].id == base_id + i for i in range(4))
            if check1 and check2 and check3:
                acc_range = KRegRange(acc_regs[0], 4, alignment=4)
                self.program.register_accumulator_vreg_range(acc_range)
        if acc_range is None and acc_regs is not None and len(acc_regs) == 4:
            acc_range = self.vreg_quad()
            self.program.register_accumulator_vreg_range(acc_range)
            self.program.emit(
                KInstr(
                    "_pack_vgpr_quad",
                    (acc_range,),
                    (acc_regs[0], acc_regs[1], acc_regs[2], acc_regs[3]),
                    comment="pack mfma acc",
                )
            )

        # MFMA updates accumulator in-place. Reuse acc_range when available.
        if acc_regs is None:
            # First MFMA in chain: allocate new accumulator, use c=0
            from .kernel_ir import KImm

            result_range = self.vreg_quad()
            self.program.register_accumulator_vreg_range(result_range)
            result_regs = tuple(KVReg(result_range.base_reg.id + i) for i in range(4))

            self.program.emit(
                KInstr(
                    "v_mfma_f32_16x16x16_bf16",
                    (result_range,),
                    (a_range, b_range, KImm(0)),
                    comment="MFMA BF16 16x16x16 (c=0)",
                )
            )
        elif acc_range is not None:
            # Accumulating MFMA: reuse accumulator in-place
            result_range = acc_range
            result_regs = tuple(KVReg(acc_range.base_reg.id + i) for i in range(4))

            self.program.emit(
                KInstr(
                    "v_mfma_f32_16x16x16_bf16",
                    (result_range,),
                    (a_range, b_range, acc_range),
                    comment="MFMA BF16 16x16x16 (in-place)",
                )
            )
        else:
            # Accumulator provided but not a proper range
            result_range = self.vreg_quad()
            self.program.register_accumulator_vreg_range(result_range)
            result_regs = tuple(KVReg(result_range.base_reg.id + i) for i in range(4))

            self.program.emit(
                KInstr(
                    "_pack_vgpr_quad",
                    (result_range,),
                    (acc_regs[0], acc_regs[1], acc_regs[2], acc_regs[3]),
                    comment="pack mfma acc (unaligned input)",
                )
            )
            self.program.emit(
                KInstr(
                    "v_mfma_f32_16x16x16_bf16",
                    (result_range,),
                    (a_range, b_range, result_range),
                    comment="MFMA BF16 16x16x16",
                )
            )

        return result_regs

    def emit_mfma_i32_16x16x32_i8(
        self,
        a_regs: Tuple[KReg, ...],
        b_regs: Tuple[KReg, ...],
        acc_regs: Optional[Tuple[KReg, ...]] = None,
    ) -> Tuple[KReg, ...]:
        """
        Emit MFMA INT8xINT8 -> INT32 (16x16x32) with virtual register tracking.

        Per ISA defs (gfx942+): a = vgpr_pair, b = vgpr_pair, c/dst = vgpr_quad.
        """
        if len(a_regs) < 2 or len(b_regs) < 2:
            raise ValueError("INT8 MFMA 16x16x32 expects 2 VGPRs per operand (vgpr_pair).")

        # Pack operands into aligned pairs
        a_tmp = self.vreg_pair()
        b_tmp = self.vreg_pair()
        self.program.emit(
            KInstr(
                "_pack_vgpr_pair",
                (a_tmp,),
                (a_regs[0], a_regs[1]),
                comment="pack mfma a",
            )
        )
        self.program.emit(
            KInstr(
                "_pack_vgpr_pair",
                (b_tmp,),
                (b_regs[0], b_regs[1]),
                comment="pack mfma b",
            )
        )
        a_range = a_tmp
        b_range = b_tmp

        acc_range = None
        if (
            acc_regs is not None
            and len(acc_regs) == 4
            and all(isinstance(r, KVReg) for r in acc_regs)
        ):
            base_id = acc_regs[0].id
            check1 = self.program._vreg_range_bases.get(base_id) == 4
            check2 = all(self.program.is_accumulator_vreg(r) for r in acc_regs)
            check3 = all(acc_regs[i].id == base_id + i for i in range(4))
            if check1 and check2 and check3:
                acc_range = KRegRange(acc_regs[0], 4, alignment=4)
                self.program.register_accumulator_vreg_range(acc_range)
        if acc_range is None and acc_regs is not None and len(acc_regs) == 4:
            acc_range = self.vreg_quad()
            self.program.register_accumulator_vreg_range(acc_range)
            self.program.emit(
                KInstr(
                    "_pack_vgpr_quad",
                    (acc_range,),
                    (acc_regs[0], acc_regs[1], acc_regs[2], acc_regs[3]),
                    comment="pack mfma acc",
                )
            )

        # MFMA updates accumulator in-place. Reuse acc_range when available.
        if acc_regs is None:
            # First MFMA in chain: allocate new accumulator, use c=0
            from .kernel_ir import KImm

            result_range = self.vreg_quad()
            self.program.register_accumulator_vreg_range(result_range)
            result_regs = tuple(KVReg(result_range.base_reg.id + i) for i in range(4))

            self.program.emit(
                KInstr(
                    "v_mfma_i32_16x16x32_i8",
                    (result_range,),
                    (a_range, b_range, KImm(0)),
                    comment="MFMA INT8 16x16x32 (c=0)",
                )
            )
        elif acc_range is not None:
            # Accumulating MFMA: reuse accumulator in-place
            result_range = acc_range
            result_regs = tuple(KVReg(acc_range.base_reg.id + i) for i in range(4))

            self.program.emit(
                KInstr(
                    "v_mfma_i32_16x16x32_i8",
                    (result_range,),
                    (a_range, b_range, acc_range),
                    comment="MFMA INT8 16x16x32 (in-place)",
                )
            )
        else:
            # Accumulator provided but not a proper range
            result_range = self.vreg_quad()
            self.program.register_accumulator_vreg_range(result_range)
            result_regs = tuple(KVReg(result_range.base_reg.id + i) for i in range(4))

            self.program.emit(
                KInstr(
                    "_pack_vgpr_quad",
                    (result_range,),
                    (acc_regs[0], acc_regs[1], acc_regs[2], acc_regs[3]),
                    comment="pack mfma acc (unaligned input)",
                )
            )
            self.program.emit(
                KInstr(
                    "v_mfma_i32_16x16x32_i8",
                    (result_range,),
                    (a_range, b_range, result_range),
                    comment="MFMA INT8 16x16x32",
                )
            )

        return result_regs
