# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Unit tests for ROCDL lane-shuffle DSL wrappers (``ds_swizzle`` and ``permlanex16``).

Validates that ``ds_swizzle`` and ``permlanex16`` coerce ``fx.Int32``, ``fx.Uint32``,
Python ``int`` literals (including unsigned 32-bit hex lane-select masks), and raw
``ir.Value`` operands cleanly and emit verified ROCDL MLIR ops without GPU hardware.
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
            f = func.FuncOp("test_rocdl_hadamard_lane_wrappers", ir.FunctionType.get(in_tys, out_tys))
            with ir.InsertionPoint(f.add_entry_block()):
                results = build_fn(*list(f.entry_block.arguments))
                if not isinstance(results, (list, tuple)):
                    results = [results]
                func.ReturnOp(results)
        assert module.operation.verify()
        return str(module)


def test_ds_swizzle_and_permlanex16_accept_dsl_values_and_literals():
    def build(src_raw):
        src = fx.Int32(src_raw)
        swz_xor4 = fx.rocdl.ds_swizzle(T.i32, src, (4 << 10) | 0x1F)
        swz_xor16 = fx.rocdl.ds_swizzle(T.i32, fx.Uint32(src_raw), fx.Int32((16 << 10) | 0x1F))
        perm_xor16 = fx.rocdl.permlanex16(T.i32, src, src, 0x76543210, 0xFEDCBA98, False, False)
        return swz_xor4, swz_xor16, perm_xor16

    ir_text = _build_module(
        build,
        arg_types=[lambda: T.i32],
        result_types=[lambda: T.i32, lambda: T.i32, lambda: T.i32],
    )
    assert "rocdl.ds_swizzle" in ir_text
    assert "rocdl.permlanex16" in ir_text


def test_hadamard_lane_wrappers_remain_omitted_from_stable_all():
    for name in ("ds_swizzle", "permlanex16"):
        assert hasattr(fx.rocdl, name)
        assert name not in fx.rocdl.__all__
