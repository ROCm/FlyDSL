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

import operator

import pytest

import flydsl.expr as fx
from flydsl._mlir.ir import Context, InsertionPoint, Location, Module

pytestmark = [pytest.mark.l1b_target_dialect]


def _binop(lhs_ty, rhs_ty, op):

    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            from flydsl._mlir.dialects import func
            from flydsl._mlir.ir import FunctionType, VectorType

            def _vec(t):
                return VectorType.get([4], t.ir_type)

            with InsertionPoint(module.body):
                ftype = FunctionType.get([lhs_ty.ir_type, rhs_ty.ir_type, _vec(lhs_ty), _vec(rhs_ty)], [])
                f = func.FuncOp("k", ftype)
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    a = lhs_ty(entry.arguments[0])
                    b = rhs_ty(entry.arguments[1])
                    va = fx.Vector(entry.arguments[2], 4, lhs_ty)
                    vb = fx.Vector(entry.arguments[3], 4, rhs_ty)
                    scalar = op(a, b)
                    vector = op(va, vb)
                    func.ReturnOp([])
            assert module.operation.verify()
    assert vector.dtype is scalar.dtype, (
        f"vector/scalar dtype drift for {lhs_ty.__name__} {op.__name__} {rhs_ty.__name__}: "
        f"vector -> {vector.dtype.__name__}, scalar -> {scalar.dtype.__name__}"
    )
    return scalar


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
    assert _binop(fx.Boolean, fx.Boolean, operator.add).dtype is fx.Int32


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


# ---------------------------------------------------------------------------
# Broader operator coverage. Every case runs through `_binop`, which builds a
# scalar pair and a vector pair and asserts they promote identically — so these
# double as the Vector/Numeric result-type consistency checks.
# ---------------------------------------------------------------------------

# Representative pairs: same-sign, mixed-sign, and cross-kind mixing.
_MIXED_PAIRS = [
    (fx.Int8, fx.Int16),
    (fx.Uint8, fx.Uint16),
    (fx.Int32, fx.Uint32),
    (fx.Int16, fx.Uint8),
    (fx.Int64, fx.Uint32),
    (fx.Int32, fx.Float32),
    (fx.Float16, fx.Float32),
]

_MIXED_INT_PAIRS = [
    (fx.Int8, fx.Int16),
    (fx.Uint8, fx.Uint16),
    (fx.Uint16, fx.Uint64),
    (fx.Int32, fx.Uint32),
    (fx.Int16, fx.Uint8),
    (fx.Int64, fx.Uint32),
]


# Subtraction / multiplication promote exactly like addition (no override).
@pytest.mark.parametrize("op", [operator.sub, operator.mul])
@pytest.mark.parametrize("a,b", _MIXED_PAIRS)
def test_sub_mul_promote_like_add(op, a, b):
    assert _binop(a, b, op).dtype is _binop(a, b, operator.add).dtype


# Integer mod / bitwise keep the usual integer promotion (no override).
@pytest.mark.parametrize("op", [operator.mod, operator.and_, operator.or_, operator.xor])
@pytest.mark.parametrize("a,b", _MIXED_INT_PAIRS)
def test_int_mod_bitwise_promote_usual(op, a, b):
    assert _binop(a, b, op).dtype is _binop(a, b, operator.add).dtype


# Comparisons yield Boolean regardless of operand types.
@pytest.mark.parametrize("op", [operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne])
@pytest.mark.parametrize("a,b", _MIXED_PAIRS)
def test_comparison_yields_boolean(op, a, b):
    assert _binop(a, b, op).dtype is fx.Boolean
