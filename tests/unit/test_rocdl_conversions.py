# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import pytest

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import func

pytestmark = pytest.mark.l0_backend_agnostic


def test_cvt_f32_fp8_is_public_and_returns_flydsl_value():
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with ir.Location.unknown(ctx):
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                i32 = ir.IntegerType.get_signless(32)
                f32 = ir.F32Type.get()
                function = func.FuncOp("test", ir.FunctionType.get([i32], [f32]))
                with ir.InsertionPoint(function.add_entry_block()):
                    value = fx.rocdl.cvt_f32_fp8(
                        fx.Int32(function.entry_block.arguments[0]),
                        byte_sel=2,
                    )
                    assert isinstance(value, fx.Float32)
                    func.ReturnOp([value.ir_value()])
            module.operation.verify()

    assert "cvt_f32_fp8" in fx.rocdl.__all__
    assert "rocdl.cvt.f32.fp8" in str(module)
