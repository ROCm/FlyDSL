# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Provisional vector ALU instructions (CDNA): byte permute, median of three, packed dot products."""

from ......_mlir.dialects import llvm
from .._core import Arg, Imm, Instruction, IrInstruction

V_PERM_B32 = IrInstruction(
    "V_PERM_B32",
    "llvm.amdgcn.perm",
    args=(Arg("src0", "i32"), Arg("src1", "i32"), Arg("sel", "i32")),
    result="i32",
    doc="Byte permute: byte ``k`` of the result is byte ``sel[k]`` (0-7) of the 8 bytes ``{src0, src1}`` (``src0`` the "
    "high word); selector values 8 and above give constants or sign fills.",
    build=lambda ods: llvm.call_intrinsic(
        ods["res"], "llvm.amdgcn.perm", [ods["src0"], ods["src1"], ods["sel"]], [], []
    ),
)


def _med3(ty):
    return Instruction(
        f"V_MED3_{ty.upper()}",
        "rocdl.fmed3",
        args=(Arg("src0", ty), Arg("src1", ty), Arg("src2", ty)),
        result="=src0",
        doc="Median of three (``clamp(src0, src1, src2)`` when ``src1 <= src2``).",
    )


V_MED3_F32 = _med3("f32")
V_MED3_F16 = _med3("f16")


def _dot(name, op, ab, c, doc):
    return Instruction(
        name,
        op,
        args=(Arg("a", ab), Arg("b", ab), Arg("c", c)),
        imms=(Imm("clamp", "bool", False),),
        result=c,
        doc=doc + " ``clamp`` saturates the result.",
    )


V_DOT2_F32_F16 = _dot("V_DOT2_F32_F16", "rocdl.fdot2", "2xf16", "f32", "``a[0] * b[0] + a[1] * b[1] + c``.")
V_DOT2_F32_BF16 = _dot(  # gfx950
    "V_DOT2_F32_BF16", "rocdl.fdot2.f32.bf16", "2xbf16", "f32", "``a[0] * b[0] + a[1] * b[1] + c``."
)
V_DOT2_I32_I16 = _dot(
    "V_DOT2_I32_I16", "rocdl.sdot2", "2xi16", "i32", "Signed dot product of two i16 pairs plus ``c``."
)
V_DOT2_U32_U16 = _dot(
    "V_DOT2_U32_U16", "rocdl.udot2", "2xi16", "i32", "Unsigned dot product of two u16 pairs plus ``c``."
)
V_DOT4_I32_I8 = _dot("V_DOT4_I32_I8", "rocdl.sdot4", "i32", "i32", "Signed dot product of four packed i8 plus ``c``.")
V_DOT4_U32_U8 = _dot("V_DOT4_U32_U8", "rocdl.udot4", "i32", "i32", "Unsigned dot product of four packed u8 plus ``c``.")
V_DOT8_I32_I4 = _dot("V_DOT8_I32_I4", "rocdl.sdot8", "i32", "i32", "Signed dot product of eight packed i4 plus ``c``.")
V_DOT8_U32_U4 = _dot(
    "V_DOT8_U32_U4", "rocdl.udot8", "i32", "i32", "Unsigned dot product of eight packed u4 plus ``c``."
)
