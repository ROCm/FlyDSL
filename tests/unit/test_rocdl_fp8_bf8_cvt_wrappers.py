# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Unit tests for ROCDL FP8/BF8/F16 conversion DSL wrappers in ``flydsl.expr.rocdl``.

Validates that ``fx.Float32``, ``fx.Int32``, Python ``float``/``int`` literals,
and raw ``ir.Value`` operands are coerced cleanly by ``_to_ir`` and emit valid
verified ROCDL conversion ops without GPU hardware.
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
            f = func.FuncOp("test_cvt", ir.FunctionType.get(in_tys, out_tys))
            with ir.InsertionPoint(f.add_entry_block()):
                results = build_fn(*list(f.entry_block.arguments))
                if not isinstance(results, (list, tuple)):
                    results = [results]
                func.ReturnOp(results)
        assert module.operation.verify()
        return str(module)


def test_pack_fp8_bf8_accept_dsl_values_and_literals():
    def build(a_raw, b_raw, seed_raw):
        a = fx.Float32(a_raw)
        b = fx.Float32(b_raw)
        seed = fx.Int32(seed_raw)

        fp8_lo = fx.rocdl.cvt_pk_fp8_f32(T.i32, a, b, 0, False)
        fp8_all = fx.rocdl.cvt_pk_fp8_f32(T.i32, a, 0.0, fx.Int32(fp8_lo), True)

        bf8_lo = fx.rocdl.cvt_pk_bf8_f32(T.i32, a, b, 0, False)
        bf8_all = fx.rocdl.cvt_pk_bf8_f32(T.i32, a, 0.0, fx.Int32(bf8_lo), True)

        sr_fp8 = fx.rocdl.cvt_sr_fp8_f32(T.i32, a, seed, fx.Int32(0), 2)
        sr_bf8 = fx.rocdl.cvt_sr_bf8_f32(T.i32, b, 12345, fx.Int32(sr_fp8), 3)

        pk_f16 = fx.rocdl.cvt_pkrtz(T.f16x2, a, b)
        return fp8_all, bf8_all, sr_bf8, pk_f16

    ir_text = _build_module(
        build,
        arg_types=[lambda: T.f32, lambda: T.f32, lambda: T.i32],
        result_types=[lambda: T.i32, lambda: T.i32, lambda: T.i32, lambda: T.f16x2],
    )
    assert "rocdl.cvt.pk.fp8.f32" in ir_text
    assert "rocdl.cvt.pk.bf8.f32" in ir_text
    assert "rocdl.cvt.sr.fp8.f32" in ir_text
    assert "rocdl.cvt.sr.bf8.f32" in ir_text
    assert "rocdl.cvt.pkrtz" in ir_text


def test_scalar_unpack_fp8_bf8_accept_dsl_int32_and_literals():
    def build(packed_raw):
        packed = fx.Int32(packed_raw)

        fp8_b1 = fx.rocdl.cvt_f32_fp8(T.f32, packed, 1)
        bf8_b3 = fx.rocdl.cvt_f32_bf8(T.f32, packed, 3)
        fp8_lit = fx.rocdl.cvt_f32_fp8(T.f32, 0x3C003C00, 0)
        bf8_lit = fx.rocdl.cvt_f32_bf8(T.f32, 0x3C003C00, 2)
        return fp8_b1, bf8_b3, fp8_lit, bf8_lit

    ir_text = _build_module(
        build,
        arg_types=[lambda: T.i32],
        result_types=[
            lambda: T.f32,
            lambda: T.f32,
            lambda: T.f32,
            lambda: T.f32,
        ],
    )
    assert "rocdl.cvt.f32.fp8" in ir_text
    assert "rocdl.cvt.f32.bf8" in ir_text


@pytest.mark.skip(
    reason=(
        "cvt_pk_f32_bf8 DSL wrapper is commented out until upstream "
        "llvm/llvm-project#231721 lands: unmodified LLVM VOP1Instructions.td "
        "maps HasCvtFP8ByteSel v_cvt_pk_f32_{fp8,bf8} only to _fake16 pseudos."
    )
)
def test_packed_unpack_bf8_disabled_until_llvm_true16_fix():
    """Uncomment alongside ``cvt_pk_f32_bf8`` once llvm/llvm-project#231721 lands in FlyDSL's LLVM commit."""
    # def build(packed_raw):
    #     packed = fx.Int32(packed_raw)
    #     bf8_lo = fx.rocdl.cvt_pk_f32_bf8(T.f32x2, packed, False)
    #     bf8_hi = fx.rocdl.cvt_pk_f32_bf8(T.f32x2, packed, True)
    #     return bf8_lo, bf8_hi
    #
    # ir_text = _build_module(
    #     build,
    #     arg_types=[lambda: T.i32],
    #     result_types=[lambda: T.f32x2, lambda: T.f32x2],
    # )
    # assert "rocdl.cvt.pk.f32.bf8" in ir_text


def test_cvt_wrappers_remain_omitted_from_stable_all():
    for name in (
        "cvt_pk_fp8_f32",
        "cvt_pk_bf8_f32",
        "cvt_f32_fp8",
        "cvt_f32_bf8",
        "cvt_sr_fp8_f32",
        "cvt_sr_bf8_f32",
        "cvt_pkrtz",
    ):
        assert hasattr(fx.rocdl, name)
        assert name not in fx.rocdl.__all__
