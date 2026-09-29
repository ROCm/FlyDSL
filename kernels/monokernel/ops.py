# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Common AMD expression primitives for fused model-layer kernels."""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value
from kernels.common import buffer_ops as bo
from kernels.common.dpp_utils import update_dpp_i32

_UMAX_DPP_ASM = "\n".join(
    [
        "s_nop 1\nv_max_u32_dpp $0, $0, $0 " + control
        for control in (
            "row_shr:1 bound_ctrl:0",
            "row_shr:2 bound_ctrl:0",
            "row_shr:4 bound_ctrl:0",
            "row_shr:8 bound_ctrl:0",
            "row_bcast:15 row_mask:0xa",
            "row_bcast:31 row_mask:0xc",
        )
    ]
)


def rsrc(addr):
    return bo.create_buffer_resource_from_addr(addr)


def uniform(value):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(value).ir_value()))


def uniform_f32(value):
    return uniform(fx.Float32(value).bitcast(fx.Int32)).bitcast(fx.Float32)


def spin_pause() -> None:
    """Keep tagged-mailbox polling loads inside the retry loop."""

    llvm.InlineAsmOp(None, [], "s_nop 0", "", has_side_effects=True)


def read_lane_i32(value, lane):
    """Read one i32 from ``lane`` while keeping IR conversion local."""

    return fx.Int32(rocdl.readlane(T.i32, fx.Int32(value), fx.Int32(lane)))


def write_lane_i32(value, lane, vector):
    """Write one i32 into ``lane`` of a wave-distributed value."""

    return fx.Int32(
        llvm.call_intrinsic(
            T.i32,
            "llvm.amdgcn.writelane.i32",
            [as_ir_value(fx.Int32(item)) for item in (value, lane, vector)],
            [],
            [],
        )
    )


def bpermute_i32(byte_offset, value):
    """Read an i32 VGPR value from the lane selected by a byte offset."""

    return fx.Int32(rocdl.ds_bpermute(T.i32, fx.Int32(byte_offset), fx.Int32(value)))


def mem_realtime():
    """Read the device-wide 64-bit realtime counter."""

    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def _hardware_f32(name, value):
    return fx.Float32(llvm.call_intrinsic(T.f32, name, [fx.Float32(value).ir_value()], [], []))


def rsq(value):
    return _hardware_f32("llvm.amdgcn.rsq.f32", value)


def rcp(value):
    return _hardware_f32("llvm.amdgcn.rcp.f32", value)


def exp(value):
    return _hardware_f32("llvm.amdgcn.exp2.f32", fx.Float32(value) * 1.4426950408889634)


def xshfl(value, offset):
    """Return the value from ``lane ^ offset`` using VALU/DPP operations."""

    if offset >= 16:
        return value.shuffle_xor(offset, 64)
    is_float = isinstance(value, fx.Float32)
    source = value.bitcast(fx.Int32) if is_float else fx.Int32(value)
    if offset == 8:
        result = fx.Int32(update_dpp_i32(source, source, 0x118, 0xF, 0xC, False))
        result = fx.Int32(update_dpp_i32(result, source, 0x108, 0xF, 0x3, False))
    elif offset == 4:
        result = fx.Int32(update_dpp_i32(source, source, 0x114, 0xF, 0xA, False))
        result = fx.Int32(update_dpp_i32(result, source, 0x104, 0xF, 0x5, False))
    elif offset == 2:
        result = fx.Int32(update_dpp_i32(source, source, 0x4E, 0xF, 0xF, False))
    else:
        result = fx.Int32(update_dpp_i32(source, source, 0xB1, 0xF, 0xF, False))
    return result.bitcast(fx.Float32) if is_float else result


def wave_umax(value):
    return fx.Int32(fx.coop.warp_reduce(fx.Uint32(value), fx.ReductionOp.MAX, width=64))


def wave_umax_dpp(value):
    """Return a wave maximum through the tuned GLM/TileRT DPP schedule."""

    result = llvm.InlineAsmOp(T.i32, [as_ir_value(fx.Int32(value))], _UMAX_DPP_ASM, "=v,0").result
    return read_lane_i32(result, 63)


def xred(value, offset, op):
    """Combine a value with ``lane ^ offset`` using a symmetric operation."""

    if offset < 16:
        return op(value, xshfl(value, offset))
    is_float = isinstance(value, fx.Float32)
    source = as_ir_value(value.bitcast(fx.Int32) if is_float else fx.Int32(value))
    swap = rocdl.permlane32_swap if offset == 32 else rocdl.permlane16_swap
    pair = swap(llvm.StructType.get_literal([T.i32, T.i32]), source, source, False, False)
    lhs, rhs = (fx.Int32(llvm.extractvalue(T.i32, pair, [index])) for index in range(2))
    if is_float:
        return op(lhs.bitcast(fx.Float32), rhs.bitcast(fx.Float32))
    return op(type(value)(lhs), type(value)(rhs))


def fp8_roundtrip(lhs, rhs):
    """Round an f32 pair through E4M3FN and return the f32 pair."""

    word = rocdl.cvt_pk_fp8_f32(T.i32, lhs, rhs, fx.Int32(0), False)
    pair_type = fx.Vector.make_type(2, fx.Float32)
    pair = fx.Vector(rocdl.cvt_pk_f32_fp8(res=pair_type, src=word, word_sel=False))
    return pair[0], pair[1]


def f8_word(k):
    """Map an FP8 activation index to the packed LDS word order."""

    return (k // 64) * 16 + ((k % 32) // 8) * 4 + ((k % 64) // 32) * 2 + (k % 8) // 4


def fp8_to_bf16x8(word0, word1):
    """Convert two dwords of eight FP8 values to a BF16 vector."""

    one = as_ir_value(fx.Float32(1.0))
    parts = []
    for word in (word0, word1):
        for half in range_constexpr(2):
            pair = fx.Vector(rocdl.cvt_scalef32_pk_bf16_fp8(T.vec(2, T.bf16), as_ir_value(word), one, bool(half)))
            parts += [pair[0], pair[1]]
    return fx.Vector.from_elements(parts, fx.BFloat16)


def mxfp8_to_bf16x8(word0, word1, scale):
    """Convert two dwords of eight FP8 values with one E8M0 block scale to BF16."""

    scale = as_ir_value(scale)
    parts = []
    for word in (word0, word1):
        for half in range_constexpr(2):
            pair = fx.Vector(rocdl.cvt_scalef32_pk_bf16_fp8(T.vec(2, T.bf16), as_ir_value(word), scale, bool(half)))
            parts += [pair[0], pair[1]]
    return fx.Vector.from_elements(parts, fx.BFloat16)


def steady_counter():
    """Read the 100 MHz constant-rate counter (``s_memrealtime`` does not exist on gfx11+)."""

    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.readsteadycounter", [], [], []))


def wave32_umax(value):
    """Return the unsigned wave32 maximum in every lane."""

    return fx.Int32(fx.coop.warp_reduce(fx.Uint32(value), fx.ReductionOp.MAX, width=32))


# Four E8M0 bytes equal to 1.0: any lane or byte a gfx1250 scaled convert reads is exact.
_E8M0_ONES = 0x7F7F7F7F


def fp8x8_to_bf16_gfx1250(word0, word1):
    """Widen eight E4M3 values (two dwords) to BF16 exactly on gfx1250."""

    src = fx.Vector.from_elements([fx.Int32(word0), fx.Int32(word1)], fx.Int32)
    return fx.Vector(
        rocdl.cvt_scale_pk8_bf16_fp8(T.vec(8, T.bf16), as_ir_value(src), as_ir_value(fx.Int32(_E8M0_ONES)), 0)
    )


def fp4x8_to_bf16_gfx1250(word, scale, scale_sel):
    """Decode eight E2M1 values (one dword) to BF16 times one E8M0 scale on gfx1250.

    The instruction reads its scale from another lane: lane ``l`` uses lane
    ``l % 16`` (even ``scale_sel``) or ``16 + l % 16`` (odd), byte
    ``(0, 1)``, ``(2, 3)``, ``(0, 2)`` or ``(1, 3)`` for lanes below / above 16
    as ``(scale_sel >> 1) & 3`` is 0, 1, 2 or 3 (measured on gfx1250).
    """

    return fx.Vector(
        rocdl.cvt_scale_pk8_bf16_fp4(
            T.vec(8, T.bf16), as_ir_value(fx.Int32(word)), as_ir_value(fx.Int32(scale)), scale_sel
        )
    )


def wmma_bf16_gfx1250(a, b, c):
    """``c += a @ b`` for one wave32 16x16x32 BF16 tile with an FP32 accumulator."""

    return fx.Vector(rocdl.wmma_f32_16x16x32_bf16_(T.vec(8, T.f32), as_ir_value(a), as_ir_value(b), as_ir_value(c)))


def mxfp4_to_bf16x8(word, scale):
    """Convert one packed dword of eight scaled E2M1 values to BF16."""

    parts = []
    for select in range_constexpr(4):
        pair = fx.Vector(
            rocdl.cvt_scalef32_pk_bf16_fp4(
                T.vec(2, T.bf16),
                as_ir_value(word),
                as_ir_value(scale),
                select,
            )
        )
        parts += [pair[0], pair[1]]
    return fx.Vector.from_elements(parts, fx.BFloat16)
