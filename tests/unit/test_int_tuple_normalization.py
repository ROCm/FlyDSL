# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import pytest

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith


@pytest.mark.parametrize("builder", [fx.make_int_tuple, fx.make_shape, fx.make_stride, fx.make_coord])
@pytest.mark.parametrize("nested", [False, True])
def test_wrapped_index_is_normalized(builder, nested):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            index_type = ir.IndexType.get()
            raw = arith.ConstantOp(index_type, ir.IntegerAttr.get(index_type, 7)).result
            raw_input = (2, [raw, 3]) if nested else raw
            wrapped_input = (2, [fx.Index(raw), 3]) if nested else fx.Index(raw)

            control = builder(raw_input)
            result = builder(wrapped_input)

            assert result.type == control.type
            casts = [op for op in module.body.operations if op.name == "arith.index_cast"]
            assert len(casts) == 2
            assert all(op.result.type == ir.IntegerType.get_signless(64) for op in casts)

        assert module.operation.verify()


@pytest.mark.parametrize("dtype", [fx.Boolean, fx.Int8, fx.Uint8, fx.Int16, fx.Uint16, fx.Int32, fx.Int64])
def test_wrapped_integer_width_is_preserved_or_widened(dtype):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            raw = arith.ConstantOp(dtype.ir_type, ir.IntegerAttr.get(dtype.ir_type, 1)).result
            result = fx.make_int_tuple(dtype(raw))
            assert result.owner.operands[0].type == ir.IntegerType.get_signless(max(32, dtype.width))

            static = fx.make_int_tuple(dtype(1))
            assert len(static.owner.operands) == 0

        assert module.operation.verify()
