# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Unit tests for ROCDL conversion, intra-lane dot, DPP, and math DSL wrappers in ``flydsl.expr.rocdl``.

Validates that ``fx.Float32``, ``fx.Int32``, ``fx.BFloat16``, ``fx.Float16``, ``fx.Vector``,
Python ``float``/``int`` literals, and raw ``ir.Value`` operands are coerced cleanly and emit
valid verified ROCDL MLIR ops without GPU hardware.
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
            f = func.FuncOp("test_rocdl_wrappers", ir.FunctionType.get(in_tys, out_tys))
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


def test_scalar_and_packed_unpack_accept_dsl_int32_and_literals():
    def build(packed_raw):
        packed = fx.Int32(packed_raw)

        fp8_b1 = fx.rocdl.cvt_f32_fp8(T.f32, packed, 1)
        bf8_b3 = fx.rocdl.cvt_f32_bf8(T.f32, packed, 3)
        fp8_lit = fx.rocdl.cvt_f32_fp8(T.f32, 0x3C003C00, 0)
        bf8_lit = fx.rocdl.cvt_f32_bf8(T.f32, 0x3C003C00, 2)
        pk_fp8_lo = fx.rocdl.cvt_pk_f32_fp8(T.f32x2, packed, False)
        perm = fx.rocdl.perm_b32(packed, 0x3F800000, 0x07060302)
        return fp8_b1, bf8_b3, fp8_lit, bf8_lit, pk_fp8_lo, perm

    ir_text = _build_module(
        build,
        arg_types=[lambda: T.i32],
        result_types=[
            lambda: T.f32,
            lambda: T.f32,
            lambda: T.f32,
            lambda: T.f32,
            lambda: T.f32x2,
            lambda: T.i32,
        ],
    )
    assert "rocdl.cvt.f32.fp8" in ir_text
    assert "rocdl.cvt.f32.bf8" in ir_text
    assert "rocdl.cvt.pk.f32.fp8" in ir_text
    assert "llvm.amdgcn.perm" in ir_text


def test_fdot2_wrappers_accept_vectors_and_packed_i32():
    def build(v_f16_raw, v_bf16_raw, pk_i32_raw, acc_f32_raw):
        pk_i32 = fx.Int32(pk_i32_raw)
        acc_f32 = fx.Float32(acc_f32_raw)

        d_f16_vec = fx.rocdl.fdot2(T.f32, v_f16_raw, v_f16_raw, acc_f32, clamp=False)
        d_f16_i32 = fx.rocdl.fdot2(T.f32, pk_i32, 0x3C003C00, 0.0, True)

        d_bf16_vec = fx.rocdl.fdot2_f32_bf16(T.f32, v_bf16_raw, v_bf16_raw, acc_f32, clamp=False)
        d_bf16_i32 = fx.rocdl.fdot2_f32_bf16(T.f32, pk_i32, pk_i32, 0.0)

        d_bf16_acc = fx.rocdl.fdot2_bf16_bf16(T.bf16, pk_i32, v_bf16_raw, 0.0)
        d_f16_acc = fx.rocdl.fdot2_f16_f16(T.f16, pk_i32, v_f16_raw, 0.0)
        return d_f16_vec, d_f16_i32, d_bf16_vec, d_bf16_i32, d_bf16_acc, d_f16_acc

    ir_text = _build_module(
        build,
        arg_types=[
            lambda: T.f16x2,
            lambda: ir.VectorType.get([2], T.bf16),
            lambda: T.i32,
            lambda: T.f32,
        ],
        result_types=[
            lambda: T.f32,
            lambda: T.f32,
            lambda: T.f32,
            lambda: T.f32,
            lambda: T.bf16,
            lambda: T.f16,
        ],
    )
    assert "rocdl.fdot2 " in ir_text
    assert "rocdl.fdot2.f32.bf16" in ir_text
    assert "rocdl.fdot2.bf16.bf16" in ir_text
    assert "rocdl.fdot2.f16.f16" in ir_text


def test_int_dot_math_and_dpp_wrappers_accept_dsl_values():
    def build(a_raw, b_raw, x_raw):
        a = fx.Int32(a_raw)
        b = fx.Int32(b_raw)
        x = fx.Float32(x_raw)

        sd4 = fx.rocdl.sdot4(T.i32, a, b, 0, clamp=False)
        ud4 = fx.rocdl.udot4(T.i32, a, 0x01010101, fx.Int32(sd4))
        sud4 = fx.rocdl.sudot4(T.i32, a, b, fx.Int32(ud4), sign_a=True, sign_b=False, clamp=False)

        sd8 = fx.rocdl.sdot8(T.i32, a, b, fx.Int32(sud4))
        ud8 = fx.rocdl.udot8(T.i32, a, 0x11111111, fx.Int32(sd8))
        sud8 = fx.rocdl.sudot8(T.i32, a, b, fx.Int32(ud8), sign_a=True, sign_b=False, clamp=False)

        ex = fx.rocdl.exp2(T.f32, x)
        rq = fx.rocdl.rsq(T.f32, fx.Float32(ex))
        dpp = fx.rocdl.update_dpp(T.f32, fx.Float32(rq), x, 0x101, 0xF, 0xF, True)
        return sud8, dpp

    ir_text = _build_module(
        build,
        arg_types=[lambda: T.i32, lambda: T.i32, lambda: T.f32],
        result_types=[lambda: T.i32, lambda: T.f32],
    )
    assert "rocdl.sdot4" in ir_text
    assert "rocdl.udot4" in ir_text
    assert "rocdl.sudot4" in ir_text
    assert "rocdl.sdot8" in ir_text
    assert "rocdl.udot8" in ir_text
    assert "rocdl.sudot8" in ir_text
    assert "rocdl.exp2" in ir_text
    assert "rocdl.rsq" in ir_text
    assert "rocdl.update.dpp" in ir_text


@pytest.mark.skip(
    reason=(
        "cvt_pk_f32_bf8, fp8x4_to_bf16x2_pair, and bf8x4_to_bf16x2_pair are commented "
        "out until upstream llvm/llvm-project#231721 lands (ROCm/FlyDSL#1376): unmodified "
        "LLVM VOP1Instructions.td maps HasCvtFP8ByteSel v_cvt_pk_f32_{fp8,bf8} only to "
        "_fake16 pseudos."
    )
)
def test_packed_unpack_bf8_and_bf16x2_pairs_disabled_until_llvm_true16_fix():
    """Uncomment alongside True16 unpack wrappers once llvm/llvm-project#231721 lands (ROCm/FlyDSL#1376)."""
    # def build(packed_raw):
    #     packed = fx.Int32(packed_raw)
    #     bf8_lo = fx.rocdl.cvt_pk_f32_bf8(T.f32x2, packed, False)
    #     bf8_hi = fx.rocdl.cvt_pk_f32_bf8(T.f32x2, packed, True)
    #     fp8_b0, fp8_b1 = fx.rocdl.fp8x4_to_bf16x2_pair(packed)
    #     bf8_b0, bf8_b1 = fx.rocdl.bf8x4_to_bf16x2_pair(packed)
    #     return bf8_lo, bf8_hi, fp8_b0, fp8_b1, bf8_b0, bf8_b1
    #
    # ir_text = _build_module(
    #     build,
    #     arg_types=[lambda: T.i32],
    #     result_types=[
    #         lambda: T.f32x2,
    #         lambda: T.f32x2,
    #         lambda: T.i32,
    #         lambda: T.i32,
    #         lambda: T.i32,
    #         lambda: T.i32,
    #     ],
    # )
    # assert "rocdl.cvt.pk.f32.bf8" in ir_text
    # assert "rocdl.cvt.pk.f32.fp8" in ir_text
    # assert "llvm.amdgcn.perm" in ir_text


def test_cvt_and_dot_wrappers_remain_omitted_from_stable_all():
    for name in (
        "cvt_pk_fp8_f32",
        "cvt_pk_bf8_f32",
        "cvt_sr_fp8_f32",
        "cvt_sr_bf8_f32",
        "cvt_pkrtz",
        "cvt_f32_fp8",
        "cvt_f32_bf8",
        "cvt_pk_f32_fp8",
        "rcp",
        "exp2",
        "rsq",
        "perm_b32",
        "fdot2",
        "fdot2_f32_bf16",
        "fdot2_f16_f16",
        "fdot2_bf16_bf16",
        "sdot4",
        "udot4",
        "sudot4",
        "sdot8",
        "udot8",
        "sudot8",
        "update_dpp",
    ):
        assert hasattr(fx.rocdl, name)
        assert name not in fx.rocdl.__all__
