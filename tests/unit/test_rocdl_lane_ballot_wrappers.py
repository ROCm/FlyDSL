# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Unit tests for ROCDL lane-shuffle and ballot-mask DSL wrappers in ``flydsl.expr.rocdl``.

Validates that ``ds_swizzle``, ``permlanex16``, ``mbcnt_lo``, and ``mbcnt_hi`` coerce
``fx.Int32``, ``fx.Uint32``, Python ``int`` literals (including unsigned 32-bit hex masks),
and raw ``ir.Value`` operands cleanly and emit verified ROCDL MLIR ops without GPU hardware.
"""

from __future__ import annotations

import pytest

import flydsl  # noqa: F401  preload comgr
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import func
from flydsl.expr.typing import T

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


def _build_module(build_fn, arg_types, result_types):
    with ir.Context() as ctx, ir.Location.unknown(ctx):
        in_tys = [t() for t in arg_types]
        out_tys = [t() for t in result_types]
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            f = func.FuncOp("test_rocdl_lane_ballot_wrappers", ir.FunctionType.get(in_tys, out_tys))
            with ir.InsertionPoint(f.add_entry_block()):
                results = build_fn(*list(f.entry_block.arguments))
                if not isinstance(results, (list, tuple)):
                    results = [results]
                func.ReturnOp(results)
        assert module.operation.verify()
        return str(module)


def test_ds_swizzle_and_permlanex16_accept_dsl_values_and_literals():
    def build(src_raw, old_raw):
        src = fx.Int32(src_raw)
        old = fx.Int32(old_raw)

        swz_lit = fx.rocdl.ds_swizzle(T.i32, src, (16 << 10) | 0x1F)
        swz_dsl = fx.rocdl.ds_swizzle(T.i32, fx.Uint32(src_raw), fx.Int32((1 << 10) | 0x1F))
        perm_bcast = fx.rocdl.permlanex16(T.i32, old, src, 0xFFFFFFFF, 0xFFFFFFFF, False, False)
        perm_xor = fx.rocdl.permlanex16(T.i32, src_raw, src, 0x76543210, 0xFEDCBA98)
        return swz_lit, swz_dsl, perm_bcast, perm_xor

    ir_text = _build_module(
        build,
        arg_types=[lambda: T.i32, lambda: T.i32],
        result_types=[lambda: T.i32, lambda: T.i32, lambda: T.i32, lambda: T.i32],
    )
    assert "rocdl.ds_swizzle" in ir_text
    assert "rocdl.permlanex16" in ir_text


def test_mbcnt_lo_and_mbcnt_hi_accept_dsl_uint32_int32_and_literals():
    def build(lo_raw, hi_raw):
        lo_mask = fx.Uint32(lo_raw)
        hi_mask = fx.Uint32(hi_raw)

        lo_cnt = fx.Int32(fx.rocdl.mbcnt_lo(T.i32, lo_mask, 0))
        hi_cnt = fx.rocdl.mbcnt_hi(T.i32, hi_mask, lo_cnt)
        lit_cnt = fx.rocdl.mbcnt_lo(T.i32, 0xFFFFFFFF, 0)
        return hi_cnt, lit_cnt

    ir_text = _build_module(
        build,
        arg_types=[lambda: T.i32, lambda: T.i32],
        result_types=[lambda: T.i32, lambda: T.i32],
    )
    assert "rocdl.mbcnt.lo" in ir_text
    assert "rocdl.mbcnt.hi" in ir_text


def test_lane_and_ballot_wrappers_remain_omitted_from_stable_all():
    for name in ("ds_swizzle", "permlanex16", "mbcnt_lo", "mbcnt_hi"):
        assert hasattr(fx.rocdl, name)
        assert name not in fx.rocdl.__all__
