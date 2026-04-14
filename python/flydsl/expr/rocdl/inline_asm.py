# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GCN/CDNA inline assembly wrappers for ROCm GPU instructions.

These emit LLVM inline asm ops for instructions that have no corresponding
MLIR ROCDL dialect op yet.  The underlying ISA instructions are defined in
LLVM's AMDGPU backend (VOP1Instructions.td / VOP3Instructions.td) but the
MLIR ROCDLOps.td tablegen does not surface them.

TODO: Remove these inline asm wrappers once upstream MLIR adds proper ROCDL
dialect ops for v_cvt_off_f32_i4 and v_cvt_pk_bf16_f32.
"""


def _to_ir(v):
    """Coerce DSL Numeric to ir.Value if needed."""
    from ..._mlir import ir as _ir
    if not isinstance(v, _ir.Value) and hasattr(v, 'ir_value'):
        return v.ir_value()
    return v


def cvt_off_f32_i4(src_i32, byte_sel=None):
    """gfx9xx: v_cvt_off_f32_i4 — convert low nibble (bits[3:0]) to f32.

    With byte_sel=0..3, uses SDWA to select the byte before conversion,
    avoiding an explicit shift.  byte_sel=None uses the plain VOP1 form.
    """
    from ..._mlir.dialects import llvm as _llvm
    from ..._mlir import ir

    if byte_sel is not None:
        sel = ["BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3"][int(byte_sel)]
        return _llvm.inline_asm(
            ir.F32Type.get(),
            [_to_ir(src_i32)],
            f"v_cvt_off_f32_i4_sdwa $0, $1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:{sel}",
            "=v,v",
            has_side_effects=False,
        )
    return _llvm.inline_asm(
        ir.F32Type.get(),
        [_to_ir(src_i32)],
        "v_cvt_off_f32_i4 $0, $1",
        "=v,v",
        has_side_effects=False,
    )


def cvt_pk_bf16_f32(src_a_f32, src_b_f32):
    """gfx950: v_cvt_pk_bf16_f32 vdst, vsrc0, vsrc1.

    Pack two f32 values into 2xbf16 in i32.
    dst[15:0] = bf16(src_a), dst[31:16] = bf16(src_b).
    """
    from ..._mlir.dialects import llvm as _llvm
    from ..._mlir import ir
    return _llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [_to_ir(src_a_f32), _to_ir(src_b_f32)],
        "v_cvt_pk_bf16_f32 $0, $1, $2",
        "=v,v,v",
        has_side_effects=False,
    )
