#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Type interoperability spec, enforced.

Executable counterpart of the "Type interoperability" section in
``docs/arithmetic_types.md``: the common-type matrix and per-operation result rules
below mirror that doc; keep the two in sync when either changes.

Verifies the common-type rule holds identically across scalar ``Numeric`` and
``Vector`` operators (operand-kind independence), and locks in the scalar/vector
alignment for bool arithmetic and integer-only bitwise/shift ops.
"""

import operator
from contextlib import contextmanager

import pytest

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import func

pytestmark = pytest.mark.l0_backend_agnostic

# ── Core dtype set (docs/arithmetic_types.md) ────────────────────────────
CORE = [
    fx.Boolean,
    fx.Int8,
    fx.Int16,
    fx.Int32,
    fx.Int64,
    fx.Uint32,
    fx.Float16,
    fx.BFloat16,
    fx.Float32,
    fx.Float64,
]

# ── Table A — common-type lattice (intermediate type) ─────────────────────
# Rows/cols follow CORE order. Mirrors docs/arithmetic_types.md Table A.
COMMON_TYPE_EXPECTED = {
    fx.Boolean: [
        fx.Boolean,
        fx.Int8,
        fx.Int16,
        fx.Int32,
        fx.Int64,
        fx.Uint32,
        fx.Float16,
        fx.BFloat16,
        fx.Float32,
        fx.Float64,
    ],
    fx.Int8: [
        fx.Int8,
        fx.Int8,
        fx.Int16,
        fx.Int32,
        fx.Int64,
        fx.Uint32,
        fx.Float16,
        fx.BFloat16,
        fx.Float32,
        fx.Float64,
    ],
    fx.Int16: [
        fx.Int16,
        fx.Int16,
        fx.Int16,
        fx.Int32,
        fx.Int64,
        fx.Uint32,
        fx.Float16,
        fx.BFloat16,
        fx.Float32,
        fx.Float64,
    ],
    fx.Int32: [
        fx.Int32,
        fx.Int32,
        fx.Int32,
        fx.Int32,
        fx.Int64,
        fx.Uint32,
        fx.Float32,
        fx.Float32,
        fx.Float32,
        fx.Float64,
    ],
    fx.Int64: [
        fx.Int64,
        fx.Int64,
        fx.Int64,
        fx.Int64,
        fx.Int64,
        fx.Int64,
        fx.Float64,
        fx.Float64,
        fx.Float64,
        fx.Float64,
    ],
    fx.Uint32: [
        fx.Uint32,
        fx.Uint32,
        fx.Uint32,
        fx.Uint32,
        fx.Int64,
        fx.Uint32,
        fx.Float32,
        fx.Float32,
        fx.Float32,
        fx.Float64,
    ],
    fx.Float16: [
        fx.Float16,
        fx.Float16,
        fx.Float16,
        fx.Float32,
        fx.Float64,
        fx.Float32,
        fx.Float16,
        fx.Float32,
        fx.Float32,
        fx.Float64,
    ],
    fx.BFloat16: [
        fx.BFloat16,
        fx.BFloat16,
        fx.BFloat16,
        fx.Float32,
        fx.Float64,
        fx.Float32,
        fx.Float32,
        fx.BFloat16,
        fx.Float32,
        fx.Float64,
    ],
    fx.Float32: [
        fx.Float32,
        fx.Float32,
        fx.Float32,
        fx.Float32,
        fx.Float64,
        fx.Float32,
        fx.Float32,
        fx.Float32,
        fx.Float32,
        fx.Float64,
    ],
    fx.Float64: [
        fx.Float64,
        fx.Float64,
        fx.Float64,
        fx.Float64,
        fx.Float64,
        fx.Float64,
        fx.Float64,
        fx.Float64,
        fx.Float64,
        fx.Float64,
    ],
}


def _expected_common_type(a, b):
    return COMMON_TYPE_EXPECTED[a][CORE.index(b)]


def _widen_bool(t):
    """Bool widens to Int32 for arithmetic ops (Table A note)."""
    return fx.Int32 if t is fx.Boolean else t


# ── MLIR context helpers ──────────────────────────────────────────────────


@contextmanager
def _in_func():
    """Open a module + empty function so ops have a valid insertion point."""
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with ir.Location.unknown(ctx):
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                f = func.FuncOp("t", ir.FunctionType.get([], []))
                with ir.InsertionPoint(f.add_entry_block()):
                    yield


def _dtype_of(value):
    """Result dtype for a scalar Numeric or a Vector."""
    return value.dtype if isinstance(value, fx.Vector) else type(value)


def _scalar(dtype):
    return dtype(1)


def _vector(dtype):
    return fx.Vector.filled(4, 1, dtype)


def _dynamic_binop(lhs_ty, rhs_ty, op):
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with ir.Location.unknown(ctx):
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                ftype = ir.FunctionType.get([lhs_ty.ir_type, rhs_ty.ir_type], [])
                f = func.FuncOp("k", ftype)
                with ir.InsertionPoint(f.add_entry_block()):
                    a = lhs_ty(f.entry_block.arguments[0])
                    b = rhs_ty(f.entry_block.arguments[1])
                    result = op(a, b)
                    func.ReturnOp([])
            assert module.operation.verify()
            return result


def _dynamic_literal_binop(arg_ty, literal, op):
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with ir.Location.unknown(ctx):
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                f = func.FuncOp("k", ir.FunctionType.get([arg_ty.ir_type], []))
                with ir.InsertionPoint(f.add_entry_block()):
                    arg = arg_ty(f.entry_block.arguments[0])
                    result = op(arg, literal)
                    func.ReturnOp([])
            assert module.operation.verify()
            return result


_SAME_KIND_OPERANDS = [
    ("scalar", lambda a, b: (_scalar(a), _scalar(b))),
    ("vector", lambda a, b: (_vector(a), _vector(b))),
]
_ALL_KIND_OPERANDS = [
    *_SAME_KIND_OPERANDS,
    ("vector-scalar", lambda a, b: (_vector(a), _scalar(b))),
    ("scalar-vector", lambda a, b: (_scalar(a), _vector(b))),
]


def _assert_result_types(op, a, b, expected, *, operand_kinds=_SAME_KIND_OPERANDS):
    for name, make_operands in operand_kinds:
        assert _dtype_of(op(*make_operands(a, b))) is expected, f"{name} {a.__name__} {op.__name__} {b.__name__}"


def _assert_raises(op, a, b, *, operand_kinds=_SAME_KIND_OPERANDS):
    for _name, make_operands in operand_kinds:
        with pytest.raises(TypeError):
            op(*make_operands(a, b))


# ── Table A: common-type lattice ──────────────────────────────────────────


class TestCommonTypeLattice:
    @pytest.mark.parametrize("a", CORE, ids=lambda t: t.__name__)
    def test_arith_result_matches_spec_scalar_and_vector(self, a):
        """`+` result equals Table A with bool pre-widened to i32, for BOTH the
        scalar and vector operator paths (operand-kind independence + the single
        shared lattice)."""
        with _in_func():
            for b in CORE:
                expected = _expected_common_type(_widen_bool(a), _widen_bool(b))
                _assert_result_types(operator.add, a, b, expected)


class TestDynamicScalarInterop:
    """Dynamic scalar operands use the same type interop rules as the spec table.

    These cases keep coverage for widths outside ``docs/arithmetic_types.md``'s
    compact core table without maintaining a second type-rule test file.
    """

    @pytest.mark.parametrize(
        "ty",
        [
            fx.Int8,
            fx.Int16,
            fx.Uint8,
            fx.Uint16,
            fx.Int32,
            fx.Int64,
            fx.Uint32,
            fx.Uint64,
            fx.Int128,
            fx.Uint128,
        ],
        ids=lambda t: t.__name__,
    )
    def test_same_type_stays_narrow(self, ty):
        assert _dynamic_binop(ty, ty, operator.add).dtype is ty
        assert _dynamic_binop(ty, ty, operator.mul).dtype is ty

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
        ids=lambda t: t.__name__,
    )
    def test_same_sign_wider_wins(self, a, b, expected):
        assert _dynamic_binop(a, b, operator.add).dtype is expected
        assert _dynamic_binop(b, a, operator.add).dtype is expected

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (fx.Int32, fx.Uint32, fx.Uint32),
            (fx.Int32, fx.Uint64, fx.Uint64),
            (fx.Int64, fx.Uint32, fx.Int64),
            (fx.Int8, fx.Uint16, fx.Uint16),
            (fx.Int16, fx.Uint8, fx.Int16),
            (fx.Int128, fx.Uint128, fx.Uint128),
            (fx.Int128, fx.Uint64, fx.Int128),
            (fx.Int128, fx.Uint32, fx.Int128),
            (fx.Uint128, fx.Int32, fx.Uint128),
            (fx.Uint128, fx.Int64, fx.Uint128),
        ],
        ids=lambda t: t.__name__,
    )
    def test_mixed_sign(self, a, b, expected):
        assert _dynamic_binop(a, b, operator.add).dtype is expected
        assert _dynamic_binop(b, a, operator.add).dtype is expected

    def test_python_int_literal_enters_as_int32(self):
        assert _dynamic_literal_binop(fx.Int8, 5, operator.add).dtype is fx.Int32

    def test_int128_plus_float64_uses_float64(self):
        assert _dynamic_binop(fx.Int128, fx.Float64, operator.add).dtype is fx.Float64
        assert _dynamic_binop(fx.Float64, fx.Int128, operator.add).dtype is fx.Float64

    def test_int128_truediv_lifts_to_float64(self):
        assert _dynamic_binop(fx.Int128, fx.Int128, operator.truediv).dtype is fx.Float64

    def test_int128_floordiv_stays_integer(self):
        assert _dynamic_binop(fx.Int128, fx.Int128, operator.floordiv).dtype is fx.Int128


# ── Table B: operation → result type ──────────────────────────────────────

# Same-dtype operands; expected result dtype per op. (bool handled separately.)
_ARITH_SAME = [
    fx.Int8,
    fx.Int16,
    fx.Int32,
    fx.Int64,
    fx.Uint32,
    fx.Float16,
    fx.BFloat16,
    fx.Float32,
    fx.Float64,
]
_TRUEDIV_EXPECTED = {
    fx.Int8: fx.Float32,
    fx.Int16: fx.Float32,
    fx.Int32: fx.Float32,
    fx.Int64: fx.Float64,
    fx.Uint32: fx.Float32,
    fx.Float16: fx.Float16,
    fx.BFloat16: fx.BFloat16,
    fx.Float32: fx.Float32,
    fx.Float64: fx.Float64,
}
_INT_TYPES = [fx.Int8, fx.Int16, fx.Int32, fx.Int64, fx.Uint32]
_FLOAT_TYPES = [fx.Float16, fx.BFloat16, fx.Float32, fx.Float64]
_INT_FLOAT_PAIRS = [(fx.Int32, fx.Float32), (fx.Float32, fx.Int32)]


class TestOperationResultType:
    @pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul, operator.floordiv, operator.mod])
    @pytest.mark.parametrize("ty", _ARITH_SAME, ids=lambda t: t.__name__)
    def test_arith_keeps_common_type(self, op, ty):
        with _in_func():
            _assert_result_types(op, ty, ty, ty)

    @pytest.mark.parametrize("ty", _ARITH_SAME, ids=lambda t: t.__name__)
    def test_truediv_result(self, ty):
        with _in_func():
            _assert_result_types(operator.truediv, ty, ty, _TRUEDIV_EXPECTED[ty])

    @pytest.mark.parametrize("op", [operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne])
    @pytest.mark.parametrize("ty", [fx.Int32, fx.Uint32, fx.Float32], ids=lambda t: t.__name__)
    def test_comparison_returns_boolean(self, op, ty):
        with _in_func():
            _assert_result_types(op, ty, ty, fx.Boolean)

    @pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor, operator.lshift, operator.rshift])
    @pytest.mark.parametrize("ty", _INT_TYPES, ids=lambda t: t.__name__)
    def test_bitwise_shift_keeps_int_type(self, op, ty):
        with _in_func():
            _assert_result_types(op, ty, ty, ty)

    @pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor, operator.lshift, operator.rshift])
    @pytest.mark.parametrize("ty", _FLOAT_TYPES, ids=lambda t: t.__name__)
    def test_bitwise_shift_on_float_raises(self, op, ty):
        """Bitwise/shift on a float common type raises TypeError for BOTH paths
        (previously the vector path emitted invalid IR)."""
        with _in_func():
            _assert_raises(op, ty, ty)

    @pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor, operator.lshift, operator.rshift])
    @pytest.mark.parametrize(
        "a,b",
        _INT_FLOAT_PAIRS,
        ids=[f"{a.__name__}-{b.__name__}" for a, b in _INT_FLOAT_PAIRS],
    )
    def test_bitwise_shift_on_int_float_common_type_raises(self, op, a, b):
        with _in_func():
            _assert_raises(op, a, b, operand_kinds=_ALL_KIND_OPERANDS)


# ── Bool arithmetic alignment (Table A note) ──────────────────────────────


class TestBoolArithmetic:
    @pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul])
    def test_bool_arith_widens_to_int32(self, op):
        with _in_func():
            _assert_result_types(op, fx.Boolean, fx.Boolean, fx.Int32)

    def test_bool_comparison_stays_boolean(self):
        with _in_func():
            _assert_result_types(operator.lt, fx.Boolean, fx.Boolean, fx.Boolean)

    @pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor])
    def test_bool_bitwise_stays_boolean(self, op):
        with _in_func():
            _assert_result_types(op, fx.Boolean, fx.Boolean, fx.Boolean)


# ── Table C: operand-kind independence ────────────────────────────────────


class TestOperandKindIndependence:
    """Same op on the same dtypes yields the same result dtype whether operands
    are scalar, vector, or Python literals, in either order."""

    CASES = [
        (operator.add, fx.Int32, fx.Float32, fx.Float32),
        (operator.add, fx.Float16, fx.Int64, fx.Float64),
        (operator.mul, fx.Uint32, fx.Int32, fx.Uint32),
        (operator.truediv, fx.Int32, fx.Int32, fx.Float32),
        (operator.lt, fx.Int32, fx.Float32, fx.Boolean),
    ]

    @pytest.mark.parametrize("op,a,b,expected", CASES)
    def test_all_kinds_agree(self, op, a, b, expected):
        with _in_func():
            _assert_result_types(op, a, b, expected, operand_kinds=_ALL_KIND_OPERANDS)

    def test_python_literal_operands(self):
        """Literals enter the lattice by DSL type: int→Int32, float→Float32."""
        with _in_func():
            # Int32 vector + python float → Float32
            assert _dtype_of(_vector(fx.Int32) + 1.0) is fx.Float32
            # Float32 vector + python int → Float32
            assert _dtype_of(_vector(fx.Float32) + 1) is fx.Float32
            # scalar Int32 + python float → Float32
            assert _dtype_of(_scalar(fx.Int32) + 1.0) is fx.Float32


# ── Intermediate type (observable via emitted casts) ──────────────────────


class TestIntermediateType:
    def test_comparison_uses_common_type_before_compare(self):
        """`Int32 < Float32`: intermediate is f32 (common type), result is bool.

        The integer operand is cast (sitofp) to f32 and the comparison is cmpf.
        """
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            with ir.Location.unknown(ctx):
                module = ir.Module.create()
                with ir.InsertionPoint(module.body):
                    ftype = ir.FunctionType.get([ir.IntegerType.get_signless(32), ir.F32Type.get()], [])
                    f = func.FuncOp("c", ftype)
                    with ir.InsertionPoint(f.add_entry_block()):
                        a = fx.Int32(f.entry_block.arguments[0])
                        b = fx.Float32(f.entry_block.arguments[1])
                        _ = a < b
                        func.ReturnOp([])
                assert module.operation.verify()
                text = str(module)
        assert "arith.sitofp" in text  # int → f32 intermediate cast
        assert "arith.cmpf" in text  # compared as f32
