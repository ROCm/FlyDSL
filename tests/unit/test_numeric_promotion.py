#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""C++-style usual-arithmetic-conversion promotion for DSL Numeric types.

We deliberately skip the C++ "integer promotion to int" step: ``int8 + int8``
must stay ``int8``, ``uint16 + uint16`` stays ``uint16``. Cross-width and
cross-sign promotion follows usual arithmetic conversions (unsigned wins at
equal width; wider wins among same-sign; signed-can-represent rule for
mixed-sign mixed-width).
"""

import pytest

import flydsl.expr as fx
from flydsl._mlir.ir import Context, InsertionPoint, Location, Module

pytestmark = [pytest.mark.l1b_target_dialect]


def _binop(lhs_ty, rhs_ty, op):
    """Build two block-arg values of the requested DSL types and apply `op`.

    Returns the resulting Numeric. We use block args so the operands are
    genuinely dynamic ir.Values (not Python literals), which is the path
    most kernel code hits.
    """
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            from flydsl._mlir.dialects import func
            from flydsl._mlir.ir import FunctionType

            with InsertionPoint(module.body):
                f = func.FuncOp("k", FunctionType.get([lhs_ty.ir_type, rhs_ty.ir_type], []))
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    a = lhs_ty(entry.arguments[0])
                    b = rhs_ty(entry.arguments[1])
                    result = op(a, b)
                    func.ReturnOp([])
            assert module.operation.verify()
            return result


# Same-sign / same-width: must stay narrow (no auto-int32 promotion).
@pytest.mark.parametrize(
    "ty",
    [fx.Int8, fx.Int16, fx.Uint8, fx.Uint16, fx.Int32, fx.Int64, fx.Uint32, fx.Uint64, fx.Int128, fx.Uint128],
)
def test_same_type_stays_narrow(ty):
    assert _binop(ty, ty, lambda a, b: a + b).dtype is ty
    assert _binop(ty, ty, lambda a, b: a * b).dtype is ty


# Same-sign cross-width: wider wins.
@pytest.mark.parametrize(
    "a,b,expected",
    [
        (fx.Int8, fx.Int16, fx.Int16),
        (fx.Int8, fx.Int32, fx.Int32),
        (fx.Int16, fx.Int64, fx.Int64),
        (fx.Uint8, fx.Uint16, fx.Uint16),
        (fx.Uint16, fx.Uint64, fx.Uint64),
        (fx.Int32, fx.Int128, fx.Int128),
        (fx.Int64, fx.Int128, fx.Int128),
        (fx.Uint32, fx.Uint128, fx.Uint128),
    ],
)
def test_same_sign_wider_wins(a, b, expected):
    assert _binop(a, b, lambda x, y: x + y).dtype is expected
    assert _binop(b, a, lambda x, y: x + y).dtype is expected  # commutative


# Mixed sign: unsigned wins iff u.width >= s.width, else signed.
@pytest.mark.parametrize(
    "a,b,expected",
    [
        (fx.Int32, fx.Uint32, fx.Uint32),  # equal width → unsigned wins
        (fx.Int32, fx.Uint64, fx.Uint64),  # u wider → unsigned wins
        (fx.Int64, fx.Uint32, fx.Int64),  # s wider → signed (signed-can-represent)
        (fx.Int8, fx.Uint16, fx.Uint16),  # u wider → unsigned
        (fx.Int16, fx.Uint8, fx.Int16),  # s wider → signed
        (fx.Int128, fx.Uint128, fx.Uint128),  # equal width → unsigned
        (fx.Int128, fx.Uint64, fx.Int128),  # s wider → signed
        (fx.Int128, fx.Uint32, fx.Int128),  # s wider → signed
        (fx.Uint128, fx.Int32, fx.Uint128),  # u wider → unsigned
        (fx.Uint128, fx.Int64, fx.Uint128),  # u wider → unsigned
    ],
)
def test_mixed_sign(a, b, expected):
    assert _binop(a, b, lambda x, y: x + y).dtype is expected
    assert _binop(b, a, lambda x, y: x + y).dtype is expected


# Python literal: as_numeric promotes int→Int32 (C++ `int` literal default),
# then C++ promotion runs.
def test_python_int_literal_promotes_via_int32():
    # Int8(arg) + 5 → Int8 + Int32 → Int32 (wider wins)
    with Context() as ctx, Location.unknown(ctx):
        ctx.allow_unregistered_dialects = True
        module = Module.create()
        from flydsl._mlir.dialects import func
        from flydsl._mlir.ir import FunctionType

        with InsertionPoint(module.body):
            f = func.FuncOp("k", FunctionType.get([fx.Int8.ir_type], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                a = fx.Int8(entry.arguments[0])
                r = a + 5
                func.ReturnOp([])
        assert module.operation.verify()
        assert r.dtype is fx.Int32


# Int + Float: promote to the float side.
@pytest.mark.parametrize(
    "itype,ftype",
    [
        (fx.Int8, fx.Float16),
        (fx.Int32, fx.Float32),
        (fx.Int64, fx.Float64),
        (fx.Int128, fx.Float64),  # no Float128; precision loss is expected and OK
    ],
)
def test_int_plus_float(itype, ftype):
    assert _binop(itype, ftype, lambda x, y: x + y).dtype is ftype
    assert _binop(ftype, itype, lambda x, y: x + y).dtype is ftype


# Float + Float: wider wins.
@pytest.mark.parametrize(
    "a,b,expected",
    [
        (fx.Float16, fx.Float32, fx.Float32),
        (fx.Float32, fx.Float64, fx.Float64),
        (fx.Float16, fx.Float64, fx.Float64),
    ],
)
def test_float_wider_wins(a, b, expected):
    assert _binop(a, b, lambda x, y: x + y).dtype is expected
    assert _binop(b, a, lambda x, y: x + y).dtype is expected


# Boolean arithmetic: bool + bool → Int32 (matches C++ "bool participates as int").
def test_bool_plus_bool_widens_to_int32():
    with Context() as ctx, Location.unknown(ctx):
        ctx.allow_unregistered_dialects = True
        module = Module.create()
        from flydsl._mlir.dialects import func
        from flydsl._mlir.ir import FunctionType

        with InsertionPoint(module.body):
            f = func.FuncOp("k", FunctionType.get([fx.Boolean.ir_type, fx.Boolean.ir_type], []))
            entry = f.add_entry_block()
            with InsertionPoint(entry):
                a = fx.Boolean(entry.arguments[0])
                b = fx.Boolean(entry.arguments[1])
                r = a + b
                func.ReturnOp([])
        assert module.operation.verify()
        assert r.dtype is fx.Int32


# True division on integers: Python `/` lifts int/int to float.
@pytest.mark.parametrize(
    "ty,expected",
    [
        (fx.Int8, fx.Float32),
        (fx.Int32, fx.Float32),
        (fx.Int64, fx.Float64),
        (fx.Int128, fx.Float64),
    ],
)
def test_truediv_int_lifts_to_float(ty, expected):
    assert _binop(ty, ty, lambda x, y: x / y).dtype is expected


# Floor division on integers: stays integer (Python `//` semantics).
@pytest.mark.parametrize("ty", [fx.Int8, fx.Int32, fx.Int64, fx.Uint32, fx.Int128])
def test_floordiv_int_stays_int(ty):
    assert _binop(ty, ty, lambda x, y: x // y).dtype is ty
