# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Provisional cross-lane VALU instructions (CDNA, wave64): lane reads, lane counts, permlane swaps."""

from ......_mlir import ir
from ......_mlir.dialects import _rocdl_ops_gen as _rocdl
from ......_mlir.dialects import llvm, vector
from .._core import Arg, Imm, Instruction, IrInstruction, mlir_type

V_READFIRSTLANE_B32 = Instruction(
    "V_READFIRSTLANE_B32",
    "rocdl.readfirstlane",
    args=(Arg("src", "i32"),),
    result="=src",
    doc="``src`` of the first active lane, as a wave-uniform value.",
)
V_READLANE_B32 = Instruction(
    "V_READLANE_B32",
    "rocdl.readlane",
    args=(Arg("src", "i32", ods="src0"), Arg("lane", "i32", ods="src1")),
    result="=src",
    doc="``src`` of lane ``lane`` (wave-uniform), as a wave-uniform value.",
)
V_MBCNT_LO_U32_B32 = Instruction(
    "V_MBCNT_LO_U32_B32",
    "rocdl.mbcnt.lo",
    args=(Arg("mask", "i32", ods="in0"), Arg("base", "i32", ods="in1")),
    result="i32",
    doc="``base`` + the set bits of ``mask`` below this lane among lanes 0-31.",
)
V_MBCNT_HI_U32_B32 = Instruction(
    "V_MBCNT_HI_U32_B32",
    "rocdl.mbcnt.hi",
    args=(Arg("mask", "i32", ods="in0"), Arg("base", "i32", ods="in1")),
    result="i32",
    doc="``base`` + the set bits of ``mask`` below this lane among lanes 32-63. "
    "``V_MBCNT_HI_U32_B32(-1, V_MBCNT_LO_U32_B32(-1, 0))`` is the lane id.",
)


def _permlane_swap(rows, op_cls):
    def build(ods):
        i32 = mlir_type("i32")
        pair = op_cls(ir.Type.parse("!llvm.struct<(i32, i32)>"), ods["old"], ods["src"], ods["fi"], ods["boundControl"])
        halves = [llvm.extractvalue(i32, pair.result, [k]) for k in (0, 1)]
        return vector.from_elements(ods["res"], halves)

    return IrInstruction(
        f"V_PERMLANE{rows}_SWAP_B32",
        f"rocdl.permlane{rows}.swap",
        args=(Arg("old", "i32"), Arg("src", "i32")),
        imms=(Imm("fi", "bool", False), Imm("bound_control", "bool", False)),
        result="2xi32",
        doc=f"Swap {rows}-lane rows between ``old`` and ``src`` (gfx950). Returns ``[new old, new src]``.",
        build=build,
    )


V_PERMLANE16_SWAP_B32 = _permlane_swap(16, _rocdl.Permlane16SwapOp)  # gfx950
V_PERMLANE32_SWAP_B32 = _permlane_swap(32, _rocdl.Permlane32SwapOp)  # gfx950
