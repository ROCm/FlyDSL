# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Provisional conversion instructions (CDNA): fp8 / bf8 conversions and the gfx950 MX-scaled ``V_CVT_SCALEF32_*``.

- ``fp8`` is E4M3 and ``bf8`` is E5M2 in the target's native flavour (FNUZ on gfx942, OCP on gfx950). Packed narrow
  data travels in i32 (``vector<6xi32>`` for 32 fp6 / bf6 values); a DSL value of the same width is bitcast.
- ``*_sel`` / ``*_sel_index`` immediates pick the byte, half or slot of the packed i32 read or written.
- Instructions that write part of a register take its previous value (``old`` / ``old_vdst``) and return the merged
  register; chain them to fill every slot.
- ``SCALEF32`` conversions multiply by the f32 ``scale``; ``SR`` variants round stochastically with ``seed``.
"""

from .._core import Arg, Imm, Instruction

V_CVT_F32_BF8 = Instruction(
    "V_CVT_F32_BF8",
    "rocdl.cvt.f32.bf8",
    args=(Arg("src_a", "i32"),),
    imms=(Imm("byte_sel", "i32", values=range(0, 4)),),
    result="f32",
)
V_CVT_F32_FP8 = Instruction(
    "V_CVT_F32_FP8",
    "rocdl.cvt.f32.fp8",
    args=(Arg("src_a", "i32"),),
    imms=(Imm("byte_sel", "i32", values=range(0, 4)),),
    result="f32",
)
V_CVT_PK_BF8_F32 = Instruction(
    "V_CVT_PK_BF8_F32",
    "rocdl.cvt.pk.bf8.f32",
    args=(Arg("src_a", "f32"), Arg("src_b", "f32"), Arg("old", "i32")),
    imms=(Imm("word_sel", "bool"),),
    result="i32",
)
V_CVT_PK_F32_BF8 = Instruction(
    "V_CVT_PK_F32_BF8",
    "rocdl.cvt.pk.f32.bf8",
    args=(Arg("src", "i32"),),
    imms=(Imm("word_sel", "bool"),),
    result="2xf32",
)
V_CVT_PK_F32_FP8 = Instruction(
    "V_CVT_PK_F32_FP8",
    "rocdl.cvt.pk.f32.fp8",
    args=(Arg("src", "i32"),),
    imms=(Imm("word_sel", "bool"),),
    result="2xf32",
)
V_CVT_PK_FP8_F32 = Instruction(
    "V_CVT_PK_FP8_F32",
    "rocdl.cvt.pk.fp8.f32",
    args=(Arg("src_a", "f32"), Arg("src_b", "f32"), Arg("old", "i32")),
    imms=(Imm("word_sel", "bool"),),
    result="i32",
)
V_CVT_PKRTZ_F16_F32 = Instruction(
    "V_CVT_PKRTZ_F16_F32",
    "rocdl.cvt.pkrtz",
    args=(Arg("src_a", "f32"), Arg("src_b", "f32")),
    result="2xf16",
)
V_CVT_SCALEF32_2XPK16_BF6_F32 = Instruction(
    "V_CVT_SCALEF32_2XPK16_BF6_F32",
    "rocdl.cvt.scalef32.2xpk16.bf6.f32",
    args=(Arg("src0", "16xf32"), Arg("src1", "16xf32"), Arg("scale", "f32")),
    result="6xi32",
)  # gfx950
V_CVT_SCALEF32_2XPK16_FP6_F32 = Instruction(
    "V_CVT_SCALEF32_2XPK16_FP6_F32",
    "rocdl.cvt.scalef32.2xpk16.fp6.f32",
    args=(Arg("src0", "16xf32"), Arg("src1", "16xf32"), Arg("scale", "f32")),
    result="6xi32",
)  # gfx950
V_CVT_SCALEF32_F16_BF8 = Instruction(
    "V_CVT_SCALEF32_F16_BF8",
    "rocdl.cvt.scalef32.f16.bf8",
    args=(Arg("old_vdst", "2xf16"), Arg("src", "i32"), Arg("scale", "f32")),
    imms=(Imm("src_sel_index", "i32", values=range(0, 4)), Imm("dst_lo_hi_sel", "bool")),
    result="2xf16",
)  # gfx950
V_CVT_SCALEF32_F16_FP8 = Instruction(
    "V_CVT_SCALEF32_F16_FP8",
    "rocdl.cvt.scalef32.f16.fp8",
    args=(Arg("old_vdst", "2xf16"), Arg("src", "i32"), Arg("scale", "f32")),
    imms=(Imm("src_sel_index", "i32", values=range(0, 4)), Imm("dst_lo_hi_sel", "bool")),
    result="2xf16",
)  # gfx950
V_CVT_SCALEF32_F32_BF8 = Instruction(
    "V_CVT_SCALEF32_F32_BF8",
    "rocdl.cvt.scalef32.f32.bf8",
    args=(Arg("src", "i32"), Arg("scale", "f32")),
    imms=(Imm("src_sel_index", "i32", values=range(0, 4)),),
    result="f32",
)  # gfx950
V_CVT_SCALEF32_F32_FP8 = Instruction(
    "V_CVT_SCALEF32_F32_FP8",
    "rocdl.cvt.scalef32.f32.fp8",
    args=(Arg("src", "i32"), Arg("scale", "f32")),
    imms=(Imm("src_sel_index", "i32", values=range(0, 4)),),
    result="f32",
)  # gfx950
V_CVT_SCALEF32_PK_BF16_BF8 = Instruction(
    "V_CVT_SCALEF32_PK_BF16_BF8",
    "rocdl.cvt.scalef32.pk.bf16.bf8",
    args=(Arg("src", "i32"), Arg("scale", "f32")),
    imms=(Imm("src_lo_hi_sel", "bool"),),
    result="2xbf16",
)  # gfx950
V_CVT_SCALEF32_PK_BF16_FP4 = Instruction(
    "V_CVT_SCALEF32_PK_BF16_FP4",
    "rocdl.cvt.scalef32.pk.bf16.fp4",
    args=(Arg("src", "i32"), Arg("scale", "f32")),
    imms=(Imm("src_sel_index", "i32", values=range(0, 4)),),
    result="2xbf16",
)  # gfx950
V_CVT_SCALEF32_PK_BF16_FP8 = Instruction(
    "V_CVT_SCALEF32_PK_BF16_FP8",
    "rocdl.cvt.scalef32.pk.bf16.fp8",
    args=(Arg("src", "i32"), Arg("scale", "f32")),
    imms=(Imm("src_lo_hi_sel", "bool"),),
    result="2xbf16",
)  # gfx950
V_CVT_SCALEF32_PK_BF8_BF16 = Instruction(
    "V_CVT_SCALEF32_PK_BF8_BF16",
    "rocdl.cvt.scalef32.pk.bf8.bf16",
    args=(Arg("old_vdst", "2xi16"), Arg("src0", "2xbf16"), Arg("scale", "f32")),
    imms=(Imm("dst_lo_hi_sel", "bool"),),
    result="2xi16",
)  # gfx950
V_CVT_SCALEF32_PK_BF8_F16 = Instruction(
    "V_CVT_SCALEF32_PK_BF8_F16",
    "rocdl.cvt.scalef32.pk.bf8.f16",
    args=(Arg("old_vdst", "2xi16"), Arg("src0", "2xf16"), Arg("scale", "f32")),
    imms=(Imm("dst_lo_hi_sel", "bool"),),
    result="2xi16",
)  # gfx950
V_CVT_SCALEF32_PK_BF8_F32 = Instruction(
    "V_CVT_SCALEF32_PK_BF8_F32",
    "rocdl.cvt.scalef32.pk.bf8.f32",
    args=(Arg("old_vdst", "2xi16"), Arg("src0", "f32"), Arg("src1", "f32"), Arg("scale", "f32")),
    imms=(Imm("dst_lo_hi_sel", "bool"),),
    result="2xi16",
)  # gfx950
V_CVT_SCALEF32_PK_F16_BF8 = Instruction(
    "V_CVT_SCALEF32_PK_F16_BF8",
    "rocdl.cvt.scalef32.pk.f16.bf8",
    args=(Arg("src", "i32"), Arg("scale", "f32")),
    imms=(Imm("src_lo_hi_sel", "bool"),),
    result="2xf16",
)  # gfx950
V_CVT_SCALEF32_PK_F16_FP4 = Instruction(
    "V_CVT_SCALEF32_PK_F16_FP4",
    "rocdl.cvt.scalef32.pk.f16.fp4",
    args=(Arg("src", "i32"), Arg("scale", "f32")),
    imms=(Imm("src_sel_index", "i32", values=range(0, 4)),),
    result="2xf16",
)  # gfx950
V_CVT_SCALEF32_PK_F16_FP8 = Instruction(
    "V_CVT_SCALEF32_PK_F16_FP8",
    "rocdl.cvt.scalef32.pk.f16.fp8",
    args=(Arg("src", "i32"), Arg("scale", "f32")),
    imms=(Imm("src_lo_hi_sel", "bool"),),
    result="2xf16",
)  # gfx950
V_CVT_SCALEF32_PK_F32_BF8 = Instruction(
    "V_CVT_SCALEF32_PK_F32_BF8",
    "rocdl.cvt.scalef32.pk.f32.bf8",
    args=(Arg("src", "i32"), Arg("scale", "f32")),
    imms=(Imm("src_lo_hi_sel", "bool"),),
    result="2xf32",
)  # gfx950
V_CVT_SCALEF32_PK_F32_FP4 = Instruction(
    "V_CVT_SCALEF32_PK_F32_FP4",
    "rocdl.cvt.scalef32.pk.f32.fp4",
    args=(Arg("src", "i32"), Arg("scale", "f32")),
    imms=(Imm("src_sel_index", "i32", values=range(0, 4)),),
    result="2xf32",
)  # gfx950
V_CVT_SCALEF32_PK_F32_FP8 = Instruction(
    "V_CVT_SCALEF32_PK_F32_FP8",
    "rocdl.cvt.scalef32.pk.f32.fp8",
    args=(Arg("src", "i32"), Arg("scale", "f32")),
    imms=(Imm("src_lo_hi_sel", "bool"),),
    result="2xf32",
)  # gfx950
V_CVT_SCALEF32_PK_FP4_BF16 = Instruction(
    "V_CVT_SCALEF32_PK_FP4_BF16",
    "rocdl.cvt.scalef32.pk.fp4.bf16",
    args=(Arg("old_vdst", "i32"), Arg("src", "2xbf16"), Arg("scale", "f32")),
    imms=(Imm("dst_sel_index", "i32", values=range(0, 4)),),
    result="i32",
)  # gfx950
V_CVT_SCALEF32_PK_FP4_F16 = Instruction(
    "V_CVT_SCALEF32_PK_FP4_F16",
    "rocdl.cvt.scalef32.pk.fp4.f16",
    args=(Arg("old_vdst", "i32"), Arg("src", "2xf16"), Arg("scale", "f32")),
    imms=(Imm("dst_sel_index", "i32", values=range(0, 4)),),
    result="i32",
)  # gfx950
V_CVT_SCALEF32_PK_FP4_F32 = Instruction(
    "V_CVT_SCALEF32_PK_FP4_F32",
    "rocdl.cvt.scalef32.pk.fp4.f32",
    args=(Arg("old_vdst", "i32"), Arg("src0", "f32"), Arg("src1", "f32"), Arg("scale", "f32")),
    imms=(Imm("dst_sel_index", "i32", values=range(0, 4)),),
    result="i32",
)  # gfx950
V_CVT_SCALEF32_PK_FP8_BF16 = Instruction(
    "V_CVT_SCALEF32_PK_FP8_BF16",
    "rocdl.cvt.scalef32.pk.fp8.bf16",
    args=(Arg("old_vdst", "2xi16"), Arg("src0", "2xbf16"), Arg("scale", "f32")),
    imms=(Imm("dst_lo_hi_sel", "bool"),),
    result="2xi16",
)  # gfx950
V_CVT_SCALEF32_PK_FP8_F16 = Instruction(
    "V_CVT_SCALEF32_PK_FP8_F16",
    "rocdl.cvt.scalef32.pk.fp8.f16",
    args=(Arg("old_vdst", "2xi16"), Arg("src0", "2xf16"), Arg("scale", "f32")),
    imms=(Imm("dst_lo_hi_sel", "bool"),),
    result="2xi16",
)  # gfx950
V_CVT_SCALEF32_PK_FP8_F32 = Instruction(
    "V_CVT_SCALEF32_PK_FP8_F32",
    "rocdl.cvt.scalef32.pk.fp8.f32",
    args=(Arg("old_vdst", "2xi16"), Arg("src0", "f32"), Arg("src1", "f32"), Arg("scale", "f32")),
    imms=(Imm("dst_lo_hi_sel", "bool"),),
    result="2xi16",
)  # gfx950
V_CVT_SCALEF32_PK32_BF16_BF6 = Instruction(
    "V_CVT_SCALEF32_PK32_BF16_BF6",
    "rocdl.cvt.scalef32.pk32.bf16.bf6",
    args=(Arg("src", "6xi32"), Arg("scale", "f32")),
    result="32xbf16",
)  # gfx950
V_CVT_SCALEF32_PK32_BF16_FP6 = Instruction(
    "V_CVT_SCALEF32_PK32_BF16_FP6",
    "rocdl.cvt.scalef32.pk32.bf16.fp6",
    args=(Arg("src", "6xi32"), Arg("scale", "f32")),
    result="32xbf16",
)  # gfx950
V_CVT_SCALEF32_PK32_BF6_BF16 = Instruction(
    "V_CVT_SCALEF32_PK32_BF6_BF16",
    "rocdl.cvt.scalef32.pk32.bf6.bf16",
    args=(Arg("src", "32xbf16"), Arg("scale", "f32")),
    result="6xi32",
)  # gfx950
V_CVT_SCALEF32_PK32_BF6_F16 = Instruction(
    "V_CVT_SCALEF32_PK32_BF6_F16",
    "rocdl.cvt.scalef32.pk32.bf6.f16",
    args=(Arg("src", "32xf16"), Arg("scale", "f32")),
    result="6xi32",
)  # gfx950
V_CVT_SCALEF32_PK32_F16_BF6 = Instruction(
    "V_CVT_SCALEF32_PK32_F16_BF6",
    "rocdl.cvt.scalef32.pk32.f16.bf6",
    args=(Arg("src", "6xi32"), Arg("scale", "f32")),
    result="32xf16",
)  # gfx950
V_CVT_SCALEF32_PK32_F16_FP6 = Instruction(
    "V_CVT_SCALEF32_PK32_F16_FP6",
    "rocdl.cvt.scalef32.pk32.f16.fp6",
    args=(Arg("src", "6xi32"), Arg("scale", "f32")),
    result="32xf16",
)  # gfx950
V_CVT_SCALEF32_PK32_F32_BF6 = Instruction(
    "V_CVT_SCALEF32_PK32_F32_BF6",
    "rocdl.cvt.scalef32.pk32.f32.bf6",
    args=(Arg("src", "6xi32"), Arg("scale", "f32")),
    result="32xf32",
)  # gfx950
V_CVT_SCALEF32_PK32_F32_FP6 = Instruction(
    "V_CVT_SCALEF32_PK32_F32_FP6",
    "rocdl.cvt.scalef32.pk32.f32.fp6",
    args=(Arg("src", "6xi32"), Arg("scale", "f32")),
    result="32xf32",
)  # gfx950
V_CVT_SCALEF32_PK32_FP6_BF16 = Instruction(
    "V_CVT_SCALEF32_PK32_FP6_BF16",
    "rocdl.cvt.scalef32.pk32.fp6.bf16",
    args=(Arg("src", "32xbf16"), Arg("scale", "f32")),
    result="6xi32",
)  # gfx950
V_CVT_SCALEF32_PK32_FP6_F16 = Instruction(
    "V_CVT_SCALEF32_PK32_FP6_F16",
    "rocdl.cvt.scalef32.pk32.fp6.f16",
    args=(Arg("src", "32xf16"), Arg("scale", "f32")),
    result="6xi32",
)  # gfx950
V_CVT_SCALEF32_SR_BF8_BF16 = Instruction(
    "V_CVT_SCALEF32_SR_BF8_BF16",
    "rocdl.cvt.scalef32.sr.bf8.bf16",
    args=(Arg("old_vdst", "i32"), Arg("src0", "bf16"), Arg("seed", "i32"), Arg("scale", "f32")),
    imms=(Imm("dst_sel_index", "i32", values=range(0, 4)),),
    result="i32",
)  # gfx950
V_CVT_SCALEF32_SR_BF8_F16 = Instruction(
    "V_CVT_SCALEF32_SR_BF8_F16",
    "rocdl.cvt.scalef32.sr.bf8.f16",
    args=(Arg("old_vdst", "i32"), Arg("src0", "f16"), Arg("seed", "i32"), Arg("scale", "f32")),
    imms=(Imm("dst_sel_index", "i32", values=range(0, 4)),),
    result="i32",
)  # gfx950
V_CVT_SCALEF32_SR_BF8_F32 = Instruction(
    "V_CVT_SCALEF32_SR_BF8_F32",
    "rocdl.cvt.scalef32.sr.bf8.f32",
    args=(Arg("old_vdst", "i32"), Arg("src0", "f32"), Arg("seed", "i32"), Arg("scale", "f32")),
    imms=(Imm("dst_sel_index", "i32", values=range(0, 4)),),
    result="i32",
)  # gfx950
V_CVT_SCALEF32_SR_FP8_BF16 = Instruction(
    "V_CVT_SCALEF32_SR_FP8_BF16",
    "rocdl.cvt.scalef32.sr.fp8.bf16",
    args=(Arg("old_vdst", "i32"), Arg("src0", "bf16"), Arg("seed", "i32"), Arg("scale", "f32")),
    imms=(Imm("dst_sel_index", "i32", values=range(0, 4)),),
    result="i32",
)  # gfx950
V_CVT_SCALEF32_SR_FP8_F16 = Instruction(
    "V_CVT_SCALEF32_SR_FP8_F16",
    "rocdl.cvt.scalef32.sr.fp8.f16",
    args=(Arg("old_vdst", "i32"), Arg("src0", "f16"), Arg("seed", "i32"), Arg("scale", "f32")),
    imms=(Imm("dst_sel_index", "i32", values=range(0, 4)),),
    result="i32",
)  # gfx950
V_CVT_SCALEF32_SR_FP8_F32 = Instruction(
    "V_CVT_SCALEF32_SR_FP8_F32",
    "rocdl.cvt.scalef32.sr.fp8.f32",
    args=(Arg("old_vdst", "i32"), Arg("src0", "f32"), Arg("seed", "i32"), Arg("scale", "f32")),
    imms=(Imm("dst_sel_index", "i32", values=range(0, 4)),),
    result="i32",
)  # gfx950
V_CVT_SCALEF32_SR_PK_FP4_BF16 = Instruction(
    "V_CVT_SCALEF32_SR_PK_FP4_BF16",
    "rocdl.cvt.scalef32.sr.pk.fp4.bf16",
    args=(Arg("old_vdst", "i32"), Arg("src", "2xbf16"), Arg("seed", "i32"), Arg("scale", "f32")),
    imms=(Imm("dst_sel_index", "i32", values=range(0, 4)),),
    result="i32",
)  # gfx950
V_CVT_SCALEF32_SR_PK_FP4_F16 = Instruction(
    "V_CVT_SCALEF32_SR_PK_FP4_F16",
    "rocdl.cvt.scalef32.sr.pk.fp4.f16",
    args=(Arg("old_vdst", "i32"), Arg("src", "2xf16"), Arg("seed", "i32"), Arg("scale", "f32")),
    imms=(Imm("dst_sel_index", "i32", values=range(0, 4)),),
    result="i32",
)  # gfx950
V_CVT_SCALEF32_SR_PK_FP4_F32 = Instruction(
    "V_CVT_SCALEF32_SR_PK_FP4_F32",
    "rocdl.cvt.scalef32.sr.pk.fp4.f32",
    args=(Arg("old_vdst", "i32"), Arg("src", "2xf32"), Arg("seed", "i32"), Arg("scale", "f32")),
    imms=(Imm("dst_sel_index", "i32", values=range(0, 4)),),
    result="i32",
)  # gfx950
V_CVT_SCALEF32_SR_PK32_BF6_BF16 = Instruction(
    "V_CVT_SCALEF32_SR_PK32_BF6_BF16",
    "rocdl.cvt.scalef32.sr.pk32.bf6.bf16",
    args=(Arg("src", "32xbf16"), Arg("seed", "i32"), Arg("scale", "f32")),
    result="6xi32",
)  # gfx950
V_CVT_SCALEF32_SR_PK32_BF6_F16 = Instruction(
    "V_CVT_SCALEF32_SR_PK32_BF6_F16",
    "rocdl.cvt.scalef32.sr.pk32.bf6.f16",
    args=(Arg("src", "32xf16"), Arg("seed", "i32"), Arg("scale", "f32")),
    result="6xi32",
)  # gfx950
V_CVT_SCALEF32_SR_PK32_BF6_F32 = Instruction(
    "V_CVT_SCALEF32_SR_PK32_BF6_F32",
    "rocdl.cvt.scalef32.sr.pk32.bf6.f32",
    args=(Arg("src", "32xf32"), Arg("seed", "i32"), Arg("scale", "f32")),
    result="6xi32",
)  # gfx950
V_CVT_SCALEF32_SR_PK32_FP6_BF16 = Instruction(
    "V_CVT_SCALEF32_SR_PK32_FP6_BF16",
    "rocdl.cvt.scalef32.sr.pk32.fp6.bf16",
    args=(Arg("src", "32xbf16"), Arg("seed", "i32"), Arg("scale", "f32")),
    result="6xi32",
)  # gfx950
V_CVT_SCALEF32_SR_PK32_FP6_F16 = Instruction(
    "V_CVT_SCALEF32_SR_PK32_FP6_F16",
    "rocdl.cvt.scalef32.sr.pk32.fp6.f16",
    args=(Arg("src", "32xf16"), Arg("seed", "i32"), Arg("scale", "f32")),
    result="6xi32",
)  # gfx950
V_CVT_SCALEF32_SR_PK32_FP6_F32 = Instruction(
    "V_CVT_SCALEF32_SR_PK32_FP6_F32",
    "rocdl.cvt.scalef32.sr.pk32.fp6.f32",
    args=(Arg("src", "32xf32"), Arg("seed", "i32"), Arg("scale", "f32")),
    result="6xi32",
)  # gfx950
V_CVT_SR_BF8_F32 = Instruction(
    "V_CVT_SR_BF8_F32",
    "rocdl.cvt.sr.bf8.f32",
    args=(Arg("src_a", "f32"), Arg("src_b", "i32"), Arg("old", "i32")),
    imms=(Imm("byte_sel", "i32", values=range(0, 4)),),
    result="i32",
)
V_CVT_SR_FP8_F32 = Instruction(
    "V_CVT_SR_FP8_F32",
    "rocdl.cvt.sr.fp8.f32",
    args=(Arg("src_a", "f32"), Arg("src_b", "i32"), Arg("old", "i32")),
    imms=(Imm("byte_sel", "i32", values=range(0, 4)),),
    result="i32",
)

#: The gfx950-only MX-scaled conversions.
SCALEF32 = (
    V_CVT_SCALEF32_2XPK16_BF6_F32,
    V_CVT_SCALEF32_2XPK16_FP6_F32,
    V_CVT_SCALEF32_F16_BF8,
    V_CVT_SCALEF32_F16_FP8,
    V_CVT_SCALEF32_F32_BF8,
    V_CVT_SCALEF32_F32_FP8,
    V_CVT_SCALEF32_PK_BF16_BF8,
    V_CVT_SCALEF32_PK_BF16_FP4,
    V_CVT_SCALEF32_PK_BF16_FP8,
    V_CVT_SCALEF32_PK_BF8_BF16,
    V_CVT_SCALEF32_PK_BF8_F16,
    V_CVT_SCALEF32_PK_BF8_F32,
    V_CVT_SCALEF32_PK_F16_BF8,
    V_CVT_SCALEF32_PK_F16_FP4,
    V_CVT_SCALEF32_PK_F16_FP8,
    V_CVT_SCALEF32_PK_F32_BF8,
    V_CVT_SCALEF32_PK_F32_FP4,
    V_CVT_SCALEF32_PK_F32_FP8,
    V_CVT_SCALEF32_PK_FP4_BF16,
    V_CVT_SCALEF32_PK_FP4_F16,
    V_CVT_SCALEF32_PK_FP4_F32,
    V_CVT_SCALEF32_PK_FP8_BF16,
    V_CVT_SCALEF32_PK_FP8_F16,
    V_CVT_SCALEF32_PK_FP8_F32,
    V_CVT_SCALEF32_PK32_BF16_BF6,
    V_CVT_SCALEF32_PK32_BF16_FP6,
    V_CVT_SCALEF32_PK32_BF6_BF16,
    V_CVT_SCALEF32_PK32_BF6_F16,
    V_CVT_SCALEF32_PK32_F16_BF6,
    V_CVT_SCALEF32_PK32_F16_FP6,
    V_CVT_SCALEF32_PK32_F32_BF6,
    V_CVT_SCALEF32_PK32_F32_FP6,
    V_CVT_SCALEF32_PK32_FP6_BF16,
    V_CVT_SCALEF32_PK32_FP6_F16,
    V_CVT_SCALEF32_SR_BF8_BF16,
    V_CVT_SCALEF32_SR_BF8_F16,
    V_CVT_SCALEF32_SR_BF8_F32,
    V_CVT_SCALEF32_SR_FP8_BF16,
    V_CVT_SCALEF32_SR_FP8_F16,
    V_CVT_SCALEF32_SR_FP8_F32,
    V_CVT_SCALEF32_SR_PK_FP4_BF16,
    V_CVT_SCALEF32_SR_PK_FP4_F16,
    V_CVT_SCALEF32_SR_PK_FP4_F32,
    V_CVT_SCALEF32_SR_PK32_BF6_BF16,
    V_CVT_SCALEF32_SR_PK32_BF6_F16,
    V_CVT_SCALEF32_SR_PK32_BF6_F32,
    V_CVT_SCALEF32_SR_PK32_FP6_BF16,
    V_CVT_SCALEF32_SR_PK32_FP6_F16,
    V_CVT_SCALEF32_SR_PK32_FP6_F32,
)
