# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Building blocks shared by the gfx11 (RDNA3) WMMA kernels."""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import rocdl


def wmma_f32_16x16x16(a_vec, b_vec, acc, *, bf16: bool):
    """``acc += a @ b`` on one 16x16x16 tile: gfx11 WMMA takes v16 operands and a v8 accumulator, and spells
    bf16 as i16."""
    if bf16:
        return rocdl.wmma_f32_16x16x16_bf16(acc.type, a_vec.bitcast(fx.Int16), b_vec.bitcast(fx.Int16), acc).result
    return rocdl.wmma_f32_16x16x16_f16(acc.type, a_vec, b_vec, acc).result


def lds_load_v8(lds_ptr, elem_dtype, v8_idx):
    """Eight ``elem_dtype`` values at element ``8 * v8_idx`` of an LDS array."""
    ptr = fx.add_offset(lds_ptr, fx.make_int_tuple(fx.Int32(v8_idx * 8)))
    return fx.make_view(fx.recast_iter(elem_dtype, ptr), fx.make_layout(8, 1)).load()


def barrier() -> None:
    """Workgroup barrier after the LDS traffic drains: the split signal/wait and s_wait_dscnt are gfx12+."""
    _llvm.inline_asm(
        res=None, operands_=[], asm_string="s_waitcnt lgkmcnt(0)\ns_barrier", constraints="", has_side_effects=True
    )
