# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Common AMD expression primitives for fused MLA + MoE kernels."""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value
from kernels.common import buffer_ops as bo
from kernels.common.dpp_utils import update_dpp_i32


def rsrc(addr):
    return bo.create_buffer_resource_from_addr(addr)


def uniform(value):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(value).ir_value()))


def uniform_f32(value):
    return uniform(fx.Float32(value).bitcast(fx.Int32)).bitcast(fx.Float32)


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
