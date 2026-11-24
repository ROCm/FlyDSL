# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Matrix instructions: MFMA and MX-scaled MFMA (CDNA).

Operands are declared in the types the mnemonic names (``vector<4xbf16>`` for ``..._BF16``, eight fp8 values for
``..._FP8_FP8``); where the LLVM intrinsic takes them as integer bits (clang's builtin prototypes) the wrapper bitcasts.
All take ``fx.Tensor`` fragments as operands and an optional ``out`` tensor (:class:`._core.MatrixInstruction`).
Instructions with a FlyROCDL atom also carry its register layouts (:class:`._core.MmaInstruction`): the single-block
16x16 and 32x32 shapes the CDNA3 MFMA atom lowers to, and the CDNA4 scaled MFMA.

gfx942 reads fp8 / bf8 data in the FNUZ formats and gfx950 in the OCP ones, so each architecture builds its own fp8 /
bf8 MFMAs (:func:`fp8_mfmas`).
"""

import itertools

from ....._mlir import ir
from ....._mlir.dialects.fly_rocdl import MmaOpCDNA3_MFMAType, MmaOpCDNA4_MFMAScaleType
from ....typing import as_ir_value
from ....utils.arith import unwrap
from ._core import Arg, Imm, MatrixInstruction, MmaInstruction

_F16 = ir.F16Type.get
_BF16 = ir.BF16Type.get
_F32 = ir.F32Type.get
# The atom identifies fp8/bf8 operands by these types; the layouts do not depend on the format.
_FP8 = ir.Float8E4M3FNType.get
_BF8 = ir.Float8E5M2FNUZType.get


def _I8():
    return ir.IntegerType.get_signless(8)


def _I32():
    return ir.IntegerType.get_signless(32)


def _cdna3_mfma(mn, k, elem_a, elem_b, elem_acc):
    return MmaOpCDNA3_MFMAType.get(mn, mn, k, elem_a(), elem_b(), elem_acc())


_MFMA_DOC = """Matrix fused multiply-add ``D = A * B + C`` over one wave; returns D with the type of ``c``.

``cbsz`` / ``abid`` broadcast one A block to the other blocks of a multi-block instruction and ``blgp`` permutes the B
lanes (for the f64 instructions ``blgp`` negates operands instead); 0 disables them."""

_SCALE_DOC = """MX-scaled MFMA ``D = (scale_a * A) * (scale_b * B) + C`` over fp8 / fp6 / fp4 data (gfx950).

``a`` / ``b``: 32 values per lane, each in its own format: fp8 e4m3 / e5m2, fp6 e2m3 / e3m2 or fp4 e2m1
(``fx.Vector[fx.Float8E4M3FN, 32]`` ...); their element types set the instruction's ``cbsz`` / ``blgp`` format fields.
``scale_a`` / ``scale_b``: four E8M0 scales (``fx.Vector[fx.Float8E8M0FNU, 4]``), of which element ``opsel_a`` /
``opsel_b`` is used. Layouts count elements of the selected format."""

# The matrix_format (cbsz / blgp) of each scaled MFMA data type.
_FORMATS = {"f8E4M3FN": 0, "f8E5M2": 1, "f6E2M3FN": 2, "f6E3M2FN": 3, "f4E2M1FN": 4}


def _format_of(name):
    """The format of operand ``name``, whose type is already checked to be 32 elements of a ``_FORMATS`` type."""
    return lambda values: _FORMATS[str(unwrap(as_ir_value(values[name])).type.element_type)]


def _scale_mfma(name, op, mn, k, acc):
    data = tuple(f"32x{f}" for f in _FORMATS)
    return MmaInstruction(
        name,
        op,
        args=(
            Arg("a", data, op_type=("8xi32", "6xi32", "4xi32")),
            Arg("b", data, op_type=("8xi32", "6xi32", "4xi32")),
            Arg("c", acc),
            Arg("scale_a", "4xf8E8M0FNU", op_type="i32"),
            Arg("scale_b", "4xf8E8M0FNU", op_type="i32"),
        ),
        imms=(
            Imm("cbsz", "matrix_format", fixed=_format_of("a")),
            Imm("blgp", "matrix_format", fixed=_format_of("b")),
            Imm("opsel_a", "i32", 0, values=range(4)),
            Imm("opsel_b", "i32", 0, values=range(4)),
        ),
        result="=c",
        doc=_SCALE_DOC,
        mma_op=lambda: MmaOpCDNA4_MFMAScaleType.get(mn, mn, k, _FP8(), _FP8(), _F32()),
    )


V_MFMA_SCALE_F32_16X16X128_F8F6F4 = _scale_mfma(
    "V_MFMA_SCALE_F32_16X16X128_F8F6F4", "rocdl.mfma.scale.f32.16x16x128.f8f6f4", 16, 128, "4xf32"
)
V_MFMA_SCALE_F32_32X32X64_F8F6F4 = _scale_mfma(
    "V_MFMA_SCALE_F32_32X32X64_F8F6F4", "rocdl.mfma.scale.f32.32x32x64.f8f6f4", 32, 64, "16xf32"
)


# The fp8 / bf8 formats each architecture's MFMAs read.
FP8_FORMATS = {"gfx942": {"FP8": "f8E4M3FNUZ", "BF8": "f8E5M2FNUZ"}, "gfx950": {"FP8": "f8E4M3FN", "BF8": "f8E5M2"}}


def fp8_mfmas(arch):
    """The fp8 / bf8 MFMAs of ``arch`` by mnemonic, ``a`` / ``b`` in the formats ``arch`` reads (``FP8_FORMATS``)."""
    fmt, atom = FP8_FORMATS[arch], {"FP8": _FP8, "BF8": _BF8}
    instrs = {}
    for mn, k, acc in ((16, 32, "4xf32"), (32, 16, "16xf32")):
        for ta, tb in itertools.product(("BF8", "FP8"), repeat=2):
            name = f"V_MFMA_F32_{mn}X{mn}X{k}_{ta}_{tb}"
            instrs[name] = MmaInstruction(
                name,
                f"rocdl.mfma.f32.{mn}x{mn}x{k}.{ta.lower()}.{tb.lower()}",
                args=(Arg("a", f"8x{fmt[ta]}", op_type="i64"), Arg("b", f"8x{fmt[tb]}", op_type="i64"), Arg("c", acc)),
                imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
                result="=c",
                doc=_MFMA_DOC,
                mma_op=lambda mn=mn, k=k, a=atom[ta], b=atom[tb]: _cdna3_mfma(mn, k, a, b, _F32),
            )
    return instrs


V_MFMA_F32_16X16X16_BF16 = MmaInstruction(
    "V_MFMA_F32_16X16X16_BF16",
    "rocdl.mfma.f32.16x16x16bf16.1k",
    args=(Arg("a", "4xbf16", op_type="4xi16"), Arg("b", "4xbf16", op_type="4xi16"), Arg("c", "4xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
    mma_op=lambda: _cdna3_mfma(16, 16, _BF16, _BF16, _F32),
)
V_MFMA_F32_16X16X16_F16 = MmaInstruction(
    "V_MFMA_F32_16X16X16_F16",
    "rocdl.mfma.f32.16x16x16f16",
    args=(Arg("a", "4xf16"), Arg("b", "4xf16"), Arg("c", "4xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
    mma_op=lambda: _cdna3_mfma(16, 16, _F16, _F16, _F32),
)
V_MFMA_F32_16X16X1_4B_F32 = MatrixInstruction(
    "V_MFMA_F32_16X16X1_4B_F32",
    "rocdl.mfma.f32.16x16x1f32",
    args=(Arg("a", "f32"), Arg("b", "f32"), Arg("c", "16xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
)
V_MFMA_F32_16X16X32_BF16 = MmaInstruction(
    "V_MFMA_F32_16X16X32_BF16",
    "rocdl.mfma.f32.16x16x32.bf16",
    args=(Arg("a", "8xbf16"), Arg("b", "8xbf16"), Arg("c", "4xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
    mma_op=lambda: _cdna3_mfma(16, 32, _BF16, _BF16, _F32),
)  # gfx950
V_MFMA_F32_16X16X32_F16 = MmaInstruction(
    "V_MFMA_F32_16X16X32_F16",
    "rocdl.mfma.f32.16x16x32.f16",
    args=(Arg("a", "8xf16"), Arg("b", "8xf16"), Arg("c", "4xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
    mma_op=lambda: _cdna3_mfma(16, 32, _F16, _F16, _F32),
)  # gfx950
V_MFMA_F32_16X16X4_4B_BF16 = MatrixInstruction(
    "V_MFMA_F32_16X16X4_4B_BF16",
    "rocdl.mfma.f32.16x16x4bf16.1k",
    args=(Arg("a", "4xbf16", op_type="4xi16"), Arg("b", "4xbf16", op_type="4xi16"), Arg("c", "16xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
)
V_MFMA_F32_16X16X4_4B_F16 = MatrixInstruction(
    "V_MFMA_F32_16X16X4_4B_F16",
    "rocdl.mfma.f32.16x16x4f16",
    args=(Arg("a", "4xf16"), Arg("b", "4xf16"), Arg("c", "16xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
)
V_MFMA_F32_16X16X4_F32 = MmaInstruction(
    "V_MFMA_F32_16X16X4_F32",
    "rocdl.mfma.f32.16x16x4f32",
    args=(Arg("a", "f32"), Arg("b", "f32"), Arg("c", "4xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
    mma_op=lambda: _cdna3_mfma(16, 4, _F32, _F32, _F32),
)
V_MFMA_F32_16X16X8_XF32 = MatrixInstruction(
    "V_MFMA_F32_16X16X8_XF32",
    "rocdl.mfma.f32.16x16x8.xf32",
    args=(Arg("a", "2xf32"), Arg("b", "2xf32"), Arg("c", "4xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
)  # gfx942 only
V_MFMA_F32_32X32X16_BF16 = MmaInstruction(
    "V_MFMA_F32_32X32X16_BF16",
    "rocdl.mfma.f32.32x32x16.bf16",
    args=(Arg("a", "8xbf16"), Arg("b", "8xbf16"), Arg("c", "16xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
    mma_op=lambda: _cdna3_mfma(32, 16, _BF16, _BF16, _F32),
)  # gfx950
V_MFMA_F32_32X32X16_F16 = MmaInstruction(
    "V_MFMA_F32_32X32X16_F16",
    "rocdl.mfma.f32.32x32x16.f16",
    args=(Arg("a", "8xf16"), Arg("b", "8xf16"), Arg("c", "16xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
    mma_op=lambda: _cdna3_mfma(32, 16, _F16, _F16, _F32),
)  # gfx950
V_MFMA_F32_32X32X1_2B_F32 = MatrixInstruction(
    "V_MFMA_F32_32X32X1_2B_F32",
    "rocdl.mfma.f32.32x32x1f32",
    args=(Arg("a", "f32"), Arg("b", "f32"), Arg("c", "32xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
)
V_MFMA_F32_32X32X2_F32 = MmaInstruction(
    "V_MFMA_F32_32X32X2_F32",
    "rocdl.mfma.f32.32x32x2f32",
    args=(Arg("a", "f32"), Arg("b", "f32"), Arg("c", "16xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
    mma_op=lambda: _cdna3_mfma(32, 2, _F32, _F32, _F32),
)
V_MFMA_F32_32X32X4_XF32 = MatrixInstruction(
    "V_MFMA_F32_32X32X4_XF32",
    "rocdl.mfma.f32.32x32x4.xf32",
    args=(Arg("a", "2xf32"), Arg("b", "2xf32"), Arg("c", "16xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
)  # gfx942 only
V_MFMA_F32_32X32X4_2B_BF16 = MatrixInstruction(
    "V_MFMA_F32_32X32X4_2B_BF16",
    "rocdl.mfma.f32.32x32x4bf16.1k",
    args=(Arg("a", "4xbf16", op_type="4xi16"), Arg("b", "4xbf16", op_type="4xi16"), Arg("c", "32xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
)
V_MFMA_F32_32X32X4_2B_F16 = MatrixInstruction(
    "V_MFMA_F32_32X32X4_2B_F16",
    "rocdl.mfma.f32.32x32x4f16",
    args=(Arg("a", "4xf16"), Arg("b", "4xf16"), Arg("c", "32xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
)
V_MFMA_F32_32X32X8_BF16 = MmaInstruction(
    "V_MFMA_F32_32X32X8_BF16",
    "rocdl.mfma.f32.32x32x8bf16.1k",
    args=(Arg("a", "4xbf16", op_type="4xi16"), Arg("b", "4xbf16", op_type="4xi16"), Arg("c", "16xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
    mma_op=lambda: _cdna3_mfma(32, 8, _BF16, _BF16, _F32),
)
V_MFMA_F32_32X32X8_F16 = MmaInstruction(
    "V_MFMA_F32_32X32X8_F16",
    "rocdl.mfma.f32.32x32x8f16",
    args=(Arg("a", "4xf16"), Arg("b", "4xf16"), Arg("c", "16xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
    mma_op=lambda: _cdna3_mfma(32, 8, _F16, _F16, _F32),
)
V_MFMA_F32_4X4X1_16B_F32 = MatrixInstruction(
    "V_MFMA_F32_4X4X1_16B_F32",
    "rocdl.mfma.f32.4x4x1f32",
    args=(Arg("a", "f32"), Arg("b", "f32"), Arg("c", "4xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
)
V_MFMA_F32_4X4X4_16B_BF16 = MatrixInstruction(
    "V_MFMA_F32_4X4X4_16B_BF16",
    "rocdl.mfma.f32.4x4x4bf16.1k",
    args=(Arg("a", "4xbf16", op_type="4xi16"), Arg("b", "4xbf16", op_type="4xi16"), Arg("c", "4xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
)
V_MFMA_F32_4X4X4_16B_F16 = MatrixInstruction(
    "V_MFMA_F32_4X4X4_16B_F16",
    "rocdl.mfma.f32.4x4x4f16",
    args=(Arg("a", "4xf16"), Arg("b", "4xf16"), Arg("c", "4xf32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
)
V_MFMA_F64_16X16X4_F64 = MatrixInstruction(
    "V_MFMA_F64_16X16X4_F64",
    "rocdl.mfma.f64.16x16x4f64",
    args=(Arg("a", "f64"), Arg("b", "f64"), Arg("c", "4xf64")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "neg_mod", 0)),
    result="=c",
    doc=_MFMA_DOC,
)
V_MFMA_F64_4X4X4_4B_F64 = MatrixInstruction(
    "V_MFMA_F64_4X4X4_4B_F64",
    "rocdl.mfma.f64.4x4x4f64",
    args=(Arg("a", "f64"), Arg("b", "f64"), Arg("c", "f64")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "neg_mod", 0)),
    result="=c",
    doc=_MFMA_DOC,
)
V_MFMA_I32_16X16X32_I8 = MmaInstruction(
    "V_MFMA_I32_16X16X32_I8",
    "rocdl.mfma.i32.16x16x32.i8",
    args=(Arg("a", "8xi8", op_type="i64"), Arg("b", "8xi8", op_type="i64"), Arg("c", "4xi32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
    mma_op=lambda: _cdna3_mfma(16, 32, _I8, _I8, _I32),
)
V_MFMA_I32_16X16X4_4B_I8 = MatrixInstruction(
    "V_MFMA_I32_16X16X4_4B_I8",
    "rocdl.mfma.i32.16x16x4i8",
    args=(Arg("a", "4xi8", op_type="i32"), Arg("b", "4xi8", op_type="i32"), Arg("c", "16xi32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
)
V_MFMA_I32_16X16X64_I8 = MatrixInstruction(
    "V_MFMA_I32_16X16X64_I8",
    "rocdl.mfma.i32.16x16x64.i8",
    args=(Arg("a", "16xi8", op_type="4xi32"), Arg("b", "16xi8", op_type="4xi32"), Arg("c", "4xi32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
)  # gfx950
V_MFMA_I32_32X32X16_I8 = MmaInstruction(
    "V_MFMA_I32_32X32X16_I8",
    "rocdl.mfma.i32.32x32x16.i8",
    args=(Arg("a", "8xi8", op_type="i64"), Arg("b", "8xi8", op_type="i64"), Arg("c", "16xi32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
    mma_op=lambda: _cdna3_mfma(32, 16, _I8, _I8, _I32),
)
V_MFMA_I32_32X32X32_I8 = MatrixInstruction(
    "V_MFMA_I32_32X32X32_I8",
    "rocdl.mfma.i32.32x32x32.i8",
    args=(Arg("a", "16xi8", op_type="4xi32"), Arg("b", "16xi8", op_type="4xi32"), Arg("c", "16xi32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
)  # gfx950
V_MFMA_I32_32X32X4_2B_I8 = MatrixInstruction(
    "V_MFMA_I32_32X32X4_2B_I8",
    "rocdl.mfma.i32.32x32x4i8",
    args=(Arg("a", "4xi8", op_type="i32"), Arg("b", "4xi8", op_type="i32"), Arg("c", "32xi32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
)
V_MFMA_I32_4X4X4_16B_I8 = MatrixInstruction(
    "V_MFMA_I32_4X4X4_16B_I8",
    "rocdl.mfma.i32.4x4x4i8",
    args=(Arg("a", "4xi8", op_type="i32"), Arg("b", "4xi8", op_type="i32"), Arg("c", "4xi32")),
    imms=(Imm("cbsz", "i32", 0), Imm("abid", "i32", 0), Imm("blgp", "perm_b", 0)),
    result="=c",
    doc=_MFMA_DOC,
)
