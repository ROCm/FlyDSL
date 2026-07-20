#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Conformance tests for ``docs/language/arithmetic_types.md``.

Same scope as that spec: the ``Numeric`` / ``Vector`` type tower and its public
methods, how a value holds a compile-time or run-time payload, and the
type-interoperability lattice + per-operation result rules for scalar / vector /
mixed operands. Keep the two in sync when either changes.

    Part 1  →  ## The type tower           (type tower, shared public methods, Vector)
    Part 2  →  ## Compile-time and run-time values
    Part 3  →  ## Type interoperability     (operand normalization, common type, result type)
"""

import operator

import pytest
from lang_utils import dtype_of, dynamic_binop, dynamic_literal_binop, run, source_ir, vec

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith
from flydsl.expr import math as fmath
from flydsl.expr.numeric import (
    BFloat16,
    Boolean,
    Float16,
    Float32,
    Int8,
    Int16,
    Int32,
    Int64,
    Numeric,
    Uint32,
    as_numeric,
)
from flydsl.expr.typing import (
    Float32x4,
    ReductionOp,
    Vector,
    full,
    full_like,
    zeros_like,
)

pytestmark = pytest.mark.l1a_compile_no_target_dialect


# ###########################################################################
# Shared fixtures & helpers
#   (docs/language/arithmetic_types.md → operand builders / expected-type tables)
# ###########################################################################


def _assert_alias_ir_type(alias, shape, dtype):
    vty = ir.VectorType(alias.ir_type)
    assert tuple(vty.shape) == shape
    assert Numeric.from_ir_type(vty.element_type) is dtype


# ── Core dtype set (docs/language/arithmetic_types.md) ────────────────────
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
# Rows/cols follow CORE order. Mirrors docs/language/arithmetic_types.md Table A.
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


# ── Operand builders (constant operands, built inside a trace) ─────────────
def _scalar(dtype):
    return dtype(1)


def _vector(dtype):
    return fx.Vector.filled(4, 1, dtype)


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
        assert dtype_of(op(*make_operands(a, b))) is expected, f"{name} {a.__name__} {op.__name__} {b.__name__}"


def _assert_raises(op, a, b, *, operand_kinds=_SAME_KIND_OPERANDS):
    for _name, make_operands in operand_kinds:
        with pytest.raises(TypeError):
            op(*make_operands(a, b))


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


# ###########################################################################
# Part 1 — The type tower
#   (docs/language/arithmetic_types.md → ## The type tower)
# ###########################################################################


# ── Numeric scalars & the type tower ────────────────────────────────────────


class TestTypeTower:
    """Every type named in the spec's type tower is importable from ``fx``."""

    NAMES = [
        # integers
        "Int4", "Int8", "Int16", "Int32", "Int64", "Int128",
        "Uint8", "Uint16", "Uint32", "Uint64", "Uint128",
        # floats
        "Float16", "BFloat16", "Float32", "Float64",
        # narrow floats
        "Float8E5M2", "Float8E4M3FN", "Float8E4M3FNUZ", "Float8E4M3B11FNUZ",
        "Float8E4M3", "Float6E2M3FN", "Float6E3M2FN", "Float8E8M0FNU", "Float4E2M1FN",
        # predicate
        "Boolean",
    ]  # fmt: skip

    @pytest.mark.parametrize("name", NAMES)
    def test_type_exported(self, name):
        assert hasattr(fx, name), f"fx.{name} missing"
        assert issubclass(getattr(fx, name), Numeric)

    def test_boolean_is_width_one(self):
        assert Boolean.width == 1


# ── Public methods (shared by Numeric and Vector) ───────────────────────────


class TestOperators:

    def test_add_two_tensors(self):
        def body():
            _ = vec(Float32) + vec(Float32)

        assert "arith.addf" in source_ir(body)

    def test_mul_scalar_broadcast(self):
        def body():
            _ = vec(Float32) * 2.0

        # Scalar 2.0 is splatted into a vector constant via arith_const
        assert "arith.mulf" in source_ir(body)

    def test_sub_reverse(self):
        def body():
            _ = 1.0 - vec(Float32)

        # Scalar 1.0 is splatted into a vector constant via arith_const
        assert "arith.subf" in source_ir(body)

    def test_int_add(self):
        def body():
            _ = vec(Int32) + vec(Int32)

        assert "arith.addi" in source_ir(body)

    def test_comparison_returns_boolean(self):
        def body():
            result = vec(Float32) < vec(Float32)
            assert isinstance(result, Vector)
            assert result.dtype is Boolean

        run(body)

    def test_bitwise_and_or_xor(self):
        def body():
            ta = vec(Uint32)
            tb = vec(Uint32)
            _ = ta & tb
            _ = ta | tb
            _ = ta ^ tb

        ir_text = source_ir(body)
        assert "arith.andi" in ir_text
        assert "arith.ori" in ir_text
        assert "arith.xori" in ir_text

    def test_shift_ops(self):
        def body():
            ta = vec(Uint32)
            _ = ta >> 16
            _ = ta << 8

        ir_text = source_ir(body)
        assert "arith.shrui" in ir_text
        assert "arith.shli" in ir_text

    def test_unsigned_shift_uses_shrui(self):
        """Uint32 Vector >> must use shrui, not shrsi."""

        def body():
            _ = vec(Uint32) >> 16

        ir_text = source_ir(body)
        assert "arith.shrui" in ir_text
        assert "arith.shrsi" not in ir_text

    def test_signed_shift_uses_shrsi(self):
        """Int32 Vector >> must use shrsi."""

        def body():
            _ = vec(Int32) >> 16

        assert "arith.shrsi" in source_ir(body)

    def test_neg(self):
        def body():
            _ = -vec(Float32)

        assert "arith.negf" in source_ir(body)

    def test_truediv(self):
        def body():
            _ = vec(Float32) / vec(Float32)

        assert "arith.divf" in source_ir(body)

    def test_pow(self):
        def body():
            _ = vec(Float32) ** vec(Float32)

        assert "math.powf" in source_ir(body)

    def test_floordiv_int(self):
        def body():
            _ = vec(Int32) // vec(Int32)

        assert "arith.floordivsi" in source_ir(body)

    def test_mod_int(self):
        def body():
            _ = vec(Int32) % vec(Int32)

        assert "arith.remsi" in source_ir(body)

    def test_neg_int(self):
        """Negating an integer Vector should produce arith.subi (0 - x)."""

        def body():
            result = -vec(Int32)
            assert isinstance(result, Vector)

        assert "arith.subi" in source_ir(body)

    def test_neg_preserves_unsigned_dtype(self):
        """Unary `-` keeps the element dtype, including unsignedness."""

        def body():
            assert (-vec(Uint32)).dtype is Uint32
            assert (-vec(Int32)).dtype is Int32

        assert "arith.subi" in source_ir(body)

    def test_invert(self):
        """`~` on an integer Vector emits xori and keeps the (signed) dtype."""

        def body():
            result = ~vec(Int32)
            assert isinstance(result, Vector)
            assert result.dtype is Int32

        assert "arith.xori" in source_ir(body)

    def test_invert_preserves_unsigned_dtype(self):
        """`~vec(Uint32)` must stay Uint32 (not decay to Int32)."""

        def body():
            assert (~vec(Uint32)).dtype is Uint32

        assert "arith.xori" in source_ir(body)

    def test_invert_float_raises(self):
        """`~` on a float Vector raises TypeError (bitwise is integer-only)."""

        def body():
            with pytest.raises(TypeError):
                ~vec(Float32)

        run(body)

    def test_abs_preserves_dtype(self):
        """`abs()` keeps the element dtype for signed int, unsigned int, and float."""

        def body():
            assert abs(vec(Int32)).dtype is Int32
            assert abs(vec(Uint32)).dtype is Uint32
            assert abs(vec(Float32)).dtype is Float32

        ir_text = source_ir(body)
        assert "math.absi" in ir_text  # signed int
        assert "math.absf" in ir_text  # float

    def test_divmod(self):
        """`divmod(v, w)` returns the (floordiv, mod) pair, both keeping dtype."""

        def body():
            q, r = divmod(vec(Int32), vec(Int32))
            assert q.dtype is Int32
            assert r.dtype is Int32

        ir_text = source_ir(body)
        assert "arith.floordivsi" in ir_text
        assert "arith.remsi" in ir_text

    def test_pow_int(self):
        """`**` on an integer Vector keeps the integer dtype."""

        def body():
            assert (vec(Int32) ** vec(Int32)).dtype is Int32

        run(body)

    def test_reverse_bitwise(self):
        def body():
            ta = vec(Uint32)
            tb = vec(Uint32)
            # reverse ops: rhs.__rand__ etc.
            r1 = tb & ta
            r2 = tb | ta
            r3 = tb ^ ta
            assert isinstance(r1, Vector)
            assert isinstance(r2, Vector)
            assert isinstance(r3, Vector)

        ir_text = source_ir(body)
        assert "arith.andi" in ir_text
        assert "arith.ori" in ir_text
        assert "arith.xori" in ir_text


class TestScalarUnaryOps:
    """Scalar unary operators mirror the shared-method table and preserve the
    compile-time property just like the binary operators."""

    def test_abs_folds_compile_time(self):
        def body():
            r = abs(Int32(-5))
            assert r.is_static()
            assert int(r) == 5

        run(body)

    def test_abs_boolean_raises(self):
        """abs is undefined for a boolean, consistently for static and run-time
        values (the IR layer already rejects it)."""

        def body(x: fx.Boolean):
            with pytest.raises(TypeError):
                abs(Boolean(True))  # compile-time
            with pytest.raises(TypeError):
                abs(x)  # run-time

        run(body, True)

    def test_neg_folds_compile_time(self):
        def body():
            r = -Int32(3)
            assert r.is_static()
            assert int(r) == -3

        run(body)

    def test_invert_folds_compile_time(self):
        """`~x` on a compile-time integer folds (no MLIR), like every other
        bitwise operator — regression guard for the compile-time property."""

        def body():
            r = ~Int32(0)
            assert r.is_static()
            assert int(r) == -1

        run(body)

    def test_invert_preserves_unsigned(self):
        def body():
            r = ~Uint32(0)
            assert r.is_static()
            assert dtype_of(r) is Uint32
            assert int(r) == 0xFFFFFFFF

        run(body)

    def test_invert_boolean_is_logical_not(self):
        def body():
            assert dtype_of(~Boolean(True)) is Boolean
            assert bool(~Boolean(True)) is False
            assert bool(~Boolean(False)) is True

        run(body)

    def test_invert_runtime_emits_ir(self):
        def body(n: fx.Int32):
            assert not (~n).is_static()

        assert "arith.xori" in source_ir(body, 5)

    def test_divmod_scalar(self):
        def body():
            q, r = divmod(Int32(7), Int32(3))
            assert int(q) == 2
            assert int(r) == 1
            assert dtype_of(q) is Int32

        run(body)


class TestVectorCommonTypeHandling:
    """Mixed-dtype vector operators convert both operands to a common type
    using the same rule as scalar Numeric arithmetic
    """

    def test_int_plus_float_uses_common_type(self):
        """Int32 Vector + Float32 Vector → float; must emit addf, not addi."""

        def body():
            result = vec(Int32) + vec(Float32)
            assert result.dtype is Float32

        ir_text = source_ir(body)
        assert "arith.addf" in ir_text
        assert "arith.addi" not in ir_text

    def test_float_plus_int_uses_common_type(self):
        """Float32 Vector + Int32 Vector → float (LHS already float)."""

        def body():
            result = vec(Float32) + vec(Int32)
            assert result.dtype is Float32

        assert "arith.addf" in source_ir(body)

    def test_int_widening_uses_common_type(self):
        """Int16 Vector + Int32 Vector widens to the wider integer type."""

        def body():
            result = vec(Int16) + vec(Int32)
            assert result.dtype is Int32

        assert "arith.addi" in source_ir(body)

    def test_f16_plus_f32_extends_narrow_operand(self):
        """Float16 + Float32 uses Float32; the narrower operand is extended."""

        def body():
            result = vec(Float16) + vec(Float32)
            assert result.dtype is Float32

        ir_text = source_ir(body)
        assert "arith.extf" in ir_text
        assert "arith.addf" in ir_text

    def test_scalar_numeric_operand_uses_common_type(self):
        """Int32 Vector + Float32 scalar converts the vector to float."""

        def body():
            result = vec(Int32) + Float32(2.0)
            assert result.dtype is Float32

        assert "arith.addf" in source_ir(body)

    def test_python_float_operand_uses_common_type(self):
        """BFloat16 Vector + Python float uses f32 (no explicit .to())."""

        def body():
            result = vec(BFloat16) + 1.0
            assert result.dtype is Float32

        ir_text = source_ir(body)
        assert "arith.extf" in ir_text
        assert "arith.addf" in ir_text


class TestToConversion:

    def test_same_type_noop(self):
        def body():
            ta = vec(Float32)
            assert ta.to(Float32) is ta

        run(body)

    def test_float_to_float_truncf(self):
        def body():
            _ = vec(Float32).to(BFloat16)

        assert "arith.truncf" in source_ir(body)

    def test_float_to_float_extf(self):
        def body():
            _ = vec(Float16).to(Float32)

        assert "arith.extf" in source_ir(body)

    def test_float_to_int(self):
        def body():
            _ = vec(Float32).to(Int32)

        assert "arith.fptosi" in source_ir(body)

    def test_int_to_float(self):
        def body():
            _ = vec(Int32).to(Float32)

        assert "arith.sitofp" in source_ir(body)

    def test_uint_to_float(self):
        """Uint32 → Float32 should use uitofp, not sitofp."""

        def body():
            result = vec(Uint32).to(Float32)
            assert result.dtype is Float32

        ir_text = source_ir(body)
        assert "arith.uitofp" in ir_text
        assert "arith.sitofp" not in ir_text

    def test_float_to_uint(self):
        """Float32 → Uint32 should use fptoui, not fptosi."""

        def body():
            result = vec(Float32).to(Uint32)
            assert result.dtype is Uint32

        ir_text = source_ir(body)
        assert "arith.fptoui" in ir_text
        assert "arith.fptosi" not in ir_text

    def test_int16_to_int32(self):
        """Int16 → Int32 should use extsi."""

        def body():
            result = vec(Int16).to(Int32)
            assert result.dtype is Int32
            assert result.shape == (8,)

        assert "arith.extsi" in source_ir(body)

    def test_to_ir_value_returns_self(self):
        """to(ir.Value) should return self unchanged."""

        def body():
            ta = vec(Float32)
            assert ta.to(ir.Value) is ta

        run(body)

    def test_to_preserves_shape(self):
        def body():
            result = vec(Float32).to(BFloat16)
            assert result.shape == (8,)
            assert result.dtype is BFloat16

        run(body)


class TestNumericIntrospection:
    """``Numeric``-only class/instance methods from the spec's method table."""

    @pytest.mark.parametrize(
        "ty,width,log_width",
        [
            (Int8, 8, 3),
            (Int16, 16, 4),
            (Int32, 32, 5),
            (Int64, 64, 6),
            (Float16, 16, 4),
            (Float32, 32, 5),
        ],
        ids=lambda t: getattr(t, "__name__", t),
    )
    def test_width_and_log_width(self, ty, width, log_width):
        assert ty.width == width
        assert ty.log_width == log_width

    def test_is_static_compile_time_true(self):
        def body():
            assert Int32(5).is_static() is True

        run(body)

    def test_is_static_available_on_all_numeric(self):
        """``is_static`` lives on the ``Numeric`` base, so floats have it too
        (regression guard: it used to be Integer-only)."""

        def body():
            assert Float32(1.0).is_static() is True
            assert BFloat16(1.0).is_static() is True
            assert Boolean(True).is_static() is True

        run(body)

    def test_is_static_runtime_false(self):
        def body(n: fx.Int32):
            assert n.is_static() is False

        run(body, 5)

    def test_as_numeric_is_publicly_exported(self):
        """The spec lists ``as_numeric`` as a public helper, so it must be
        reachable from the ``flydsl.expr`` namespace (regression guard for the
        ``typing.__all__`` export)."""
        assert hasattr(fx, "as_numeric")
        assert fx.as_numeric is as_numeric

    def test_as_numeric_and_from_python_value(self):
        def body():
            # exercise via the public ``fx`` path the spec documents
            assert dtype_of(fx.as_numeric(5)) is Int32
            assert dtype_of(fx.as_numeric(1.0)) is Float32
            assert dtype_of(Numeric.from_python_value(5)) is Int32
            assert dtype_of(Numeric.from_python_value(True)) is Boolean

        run(body)

    def test_from_python_value_big_int_is_int64(self):
        def body():
            assert dtype_of(Numeric.from_python_value(5_000_000_000)) is Int64

        run(body)

    def test_as_numeric_passthrough(self):
        def body():
            x = Int32(7)
            assert as_numeric(x) is x

        run(body)

    def test_from_ir_type(self):
        def body():
            from flydsl._mlir.extras import types as T

            assert Numeric.from_ir_type(T.f32()) is Float32
            assert Numeric.from_ir_type(T.i32()) is Int32

        run(body)

    def test_scalar_to_value_preserving(self):
        """``x.to(dtype)`` is a value-preserving scalar conversion (spec method
        table); a compile-time value stays compile-time."""

        def body():
            r = Int32(5).to(Float32)
            assert dtype_of(r) is Float32
            assert r.is_static() and float(r.value) == 5.0
            assert Float32(5.0).to(Int32).value == 5
            x = Int32(5)
            assert x.to(Int32) is x  # same-type conversion is a no-op

        run(body)


class TestVectorOps:

    def test_bitcast(self):
        def body():
            result = vec(Float32).bitcast(Uint32)
            assert result.shape == (8,)
            assert result.dtype is Uint32

        assert "vector.bitcast" in source_ir(body)

    def test_bitcast_width_change(self):
        """f32 → f16 bitcast: 8 elements * 32 bits = 256 bits → 16 elements * 16 bits."""

        def body():
            result = vec(Float32).bitcast(Float16)
            assert result.shape == (16,)
            assert result.dtype is Float16

        assert "vector.bitcast" in source_ir(body)

    def test_shuffle(self):
        def body():
            result = vec(Float32).shuffle(vec(Float32), [0, 2, 4, 6])
            assert result.shape == (4,)
            assert result.dtype is Float32

        assert "vector.shuffle" in source_ir(body)

    def test_bitcast_widen_element_shrinks_lanes(self):
        """Widening the element type shrinks the lane count (total bits fixed):
        Float16 x8 (128 bits) → Float32 x4."""

        def body():
            result = Vector.filled(8, 1.0, Float16).bitcast(Float32)
            assert result.dtype is Float32
            assert result.shape == (4,)

        assert "vector.bitcast" in source_ir(body)


class TestScalarBitcast:
    """``Numeric.bitcast`` reinterprets the bits at equal width."""

    def test_float_to_int_roundtrip(self):
        def body():
            i = Float32(1.0).bitcast(Int32)
            assert dtype_of(i) is Int32
            back = i.bitcast(Float32)
            assert dtype_of(back) is Float32

        assert "arith.bitcast" in source_ir(body)

    def test_bad_dtype_raises(self):
        def body():
            with pytest.raises(TypeError):
                Float32(1.0).bitcast(123)

        run(body)


class TestElementAccess:

    def test_getitem_int(self):
        def body():
            elem = vec(Float32)[0]
            assert isinstance(elem, Float32)

        assert "vector.extract" in source_ir(body)

    def test_getitem_invalid_type(self):
        def body():
            with pytest.raises(TypeError):
                vec(Float32)["bad"]

        run(body)


class TestReduction:

    def test_reduce_add(self):
        def body():
            _ = vec(Float32).reduce(ReductionOp.ADD)

        assert "vector.reduction <add>" in source_ir(body)

    def test_reduce_max(self):
        def body():
            _ = vec(Float32).reduce(ReductionOp.MAX)

        assert "vector.reduction <maxnumf>" in source_ir(body)

    def test_reduce_min(self):
        def body():
            _ = vec(Float32).reduce(ReductionOp.MIN)

        assert "vector.reduction <minimumf>" in source_ir(body)

    def test_reduce_with_fastmath(self):
        def body():
            _ = vec(Float32).reduce(ReductionOp.ADD, fastmath=arith.FastMathFlags.fast)

        ir_text = source_ir(body)
        assert "fastmath" in ir_text.lower() or "fast" in ir_text

    def test_reduce_returns_numeric(self):
        """reduce() should return Numeric, not raw ir.Value."""

        def body():
            assert isinstance(vec(Float32).reduce(ReductionOp.ADD), Float32)

        run(body)

    def test_int_reduce_add(self):
        def body():
            assert isinstance(vec(Int32).reduce(ReductionOp.ADD), Int32)

        assert "vector.reduction <add>" in source_ir(body)

    def test_int_reduce_max_signed(self):
        """Int32 MAX should use maxsi, not maxnumf."""

        def body():
            assert isinstance(vec(Int32).reduce(ReductionOp.MAX), Int32)

        assert "vector.reduction <maxsi>" in source_ir(body)

    def test_int_reduce_max_unsigned(self):
        """Uint32 MAX should use maxui."""

        def body():
            assert isinstance(vec(Uint32).reduce(ReductionOp.MAX), Uint32)

        assert "vector.reduction <maxui>" in source_ir(body)

    def test_int_reduce_min_signed(self):
        """Int32 MIN should use minsi."""

        def body():
            assert isinstance(vec(Int32).reduce(ReductionOp.MIN), Int32)

        assert "vector.reduction <minsi>" in source_ir(body)

    def test_int_reduce_min_unsigned(self):
        """Uint32 MIN should use minui."""

        def body():
            assert isinstance(vec(Uint32).reduce(ReductionOp.MIN), Uint32)

        assert "vector.reduction <minui>" in source_ir(body)

    def test_reduce_with_init_val(self):
        """reduce() with init_val should pass acc to vector.reduction."""

        def body():
            result = vec(Float32).reduce(ReductionOp.ADD, init_val=Float32(0.0))
            assert isinstance(result, Float32)

        assert "vector.reduction <add>" in source_ir(body)

    def test_reduce_string_add(self):
        """reduce() accepts plain string 'add'."""

        def body():
            assert isinstance(vec(Float32).reduce("add"), Float32)

        assert "vector.reduction <add>" in source_ir(body)

    def test_reduce_string_max(self):
        """reduce() accepts plain string 'max'."""

        def body():
            _ = vec(Float32).reduce("max")

        assert "vector.reduction <maxnumf>" in source_ir(body)

    def test_reduce_combining_kind_direct(self):
        """reduce() accepts raw CombiningKind."""
        from flydsl._mlir.dialects.vector import CombiningKind

        def body():
            assert isinstance(vec(Float32).reduce(CombiningKind.ADD), Float32)

        assert "vector.reduction <add>" in source_ir(body)

    def test_reduce_bad_op_raises(self):
        """reduce() raises on invalid op type."""

        def body():
            with pytest.raises(TypeError):
                vec(Float32).reduce(42)

        run(body)

    def test_reduce_bad_string_raises(self):
        """reduce() raises on unknown string."""

        def body():
            with pytest.raises(ValueError, match="unknown"):
                vec(Float32).reduce("foobar")

        run(body)


class TestFactories:

    def test_full(self):
        def body():
            t = full(8, 1.0, Float32)
            assert t.shape == (8,)
            assert t.dtype is Float32

        assert "vector.broadcast" in source_ir(body)

    def test_full_like(self):
        def body():
            ta = vec(Float32)
            t = full_like(ta, 0.0)
            assert t.shape == ta.shape
            assert t.dtype == ta.dtype

        assert "vector.broadcast" in source_ir(body)

    def test_zeros_like(self):
        def body():
            ta = vec(Float32)
            t = zeros_like(ta)
            assert t.shape == ta.shape
            assert t.dtype == ta.dtype

        assert "vector.broadcast" in source_ir(body)

    def test_full_with_numeric_fill(self):
        """full() with Numeric fill_value should work."""

        def body():
            t = full(8, Float32(2.5), Float32)
            assert t.shape == (8,)
            assert t.dtype is Float32

        assert "vector.broadcast" in source_ir(body)

    def test_classmethod_filled(self):
        def body():
            t = Vector.filled(8, 1.0, Float32)
            assert t.shape == (8,)
            assert t.dtype is Float32

        assert "vector.broadcast" in source_ir(body)

    def test_classmethod_filled_like(self):
        def body():
            ta = vec(Float32)
            t = Vector.filled_like(ta, 2.0)
            assert t.shape == ta.shape
            assert t.dtype is Float32

        assert "vector.broadcast" in source_ir(body)

    def test_classmethod_zeros_like(self):
        def body():
            ta = vec(Float32)
            t = Vector.zeros_like(ta)
            assert t.shape == ta.shape
            assert t.dtype is Float32

        assert "vector.broadcast" in source_ir(body)


class TestFmath:

    def test_exp2_tensor(self):
        def body():
            result = fmath.exp2(vec(Float32))
            assert isinstance(result, Vector)
            assert result.dtype is Float32
            assert result.shape == (8,)

        assert "math.exp2" in source_ir(body)

    def test_rsqrt_tensor(self):
        def body():
            _ = fmath.rsqrt(vec(Float32))

        assert "math.rsqrt" in source_ir(body)

    def test_fastmath_flag(self):
        from flydsl.expr.arith import FastMathFlags

        def body():
            _ = fmath.exp2(vec(Float32), fastmath=FastMathFlags.fast)

        assert "fast" in source_ir(body)

    def test_scalar_float(self):
        """math on scalar Float32 returns Float32 Numeric."""

        def body():
            result = fmath.sqrt(Float32(1.0))
            assert not isinstance(result, Vector)
            assert isinstance(result, Float32)

        run(body)

    def test_vector_scalar_atan2(self):
        """atan2 with Vector and scalar broadcasts the scalar."""

        def body():
            ta = vec(Float32)
            # scalar is broadcast to match vector type via _coerce_other
            _ = fmath.atan2(ta, ta)

        run(body)

    def test_new_functions_exist(self):
        """Verify all newly added functions are accessible."""
        for name in ["erf", "acos", "asin", "atan", "atan2", "tan", "log10"]:
            assert hasattr(fmath, name), f"fmath.{name} missing"

    def test_erf_tensor(self):
        def body():
            _ = fmath.erf(vec(Float32))

        assert "math.erf" in source_ir(body)

    def test_atan2_tensor(self):
        def body():
            result = fmath.atan2(vec(Float32), vec(Float32))
            assert isinstance(result, Vector)

        assert "math.atan2" in source_ir(body)


class TestSelect:
    """`cond.select` promotes its two branches to their common type with the same
    Type-interoperability rules as the operators (spec: shared method table)."""

    def test_scalar_select_promotes_to_common_type(self):
        def body(x: fx.Int32):
            r = (x < fx.Int32(2)).select(fx.Float32(1.0), x)
            assert dtype_of(r) is fx.Float32

        run(body, 5)

    def test_scalar_select_same_type_stays(self):
        def body(x: fx.Int32):
            r = (x < fx.Int32(2)).select(fx.Int32(1), fx.Int32(2))
            assert dtype_of(r) is fx.Int32

        run(body, 5)

    def test_scalar_select_literal_branch_normalized_by_value(self):
        """A Python-literal branch enters by DSL type (float→Float32) before the
        common type is taken, so `select(1.0, int32)` promotes to Float32."""

        def body(x: fx.Int32):
            r = (x < fx.Int32(2)).select(1.0, x)
            assert dtype_of(r) is fx.Float32

        run(body, 5)

    def test_scalar_select_emits_promotion_cast(self):
        def body(x: fx.Int32):
            (x < fx.Int32(2)).select(fx.Float32(1.0), x)

        ir_text = source_ir(body, 5)
        assert "arith.select" in ir_text
        assert "arith.sitofp" in ir_text  # the Int32 branch is promoted to f32

    def test_vector_select_promotes_element_type(self):
        def body():
            cond = _vector(fx.Int32) < _vector(fx.Int32)  # Boolean vector
            r = cond.select(_vector(fx.Float32), _vector(fx.Int32))
            assert isinstance(r, fx.Vector)
            assert r.dtype is fx.Float32
            assert r.shape == (4,)

        run(body)

    def test_vector_select_scalar_branch_broadcast(self):
        def body():
            cond = _vector(fx.Int32) < _vector(fx.Int32)
            r = cond.select(fx.Float32(1.0), _vector(fx.Int32))
            assert r.dtype is fx.Float32
            assert r.shape == (4,)

        run(body)

    def test_vector_select_mismatched_lane_count_raises(self):
        def body():
            cond = fx.Vector.filled(4, 1, fx.Int32) < fx.Vector.filled(4, 1, fx.Int32)
            with pytest.raises(ValueError):
                cond.select(fx.Vector.filled(3, 1, fx.Float32), fx.Vector.filled(3, 1, fx.Float32))

        run(body)

    def test_vector_non_boolean_condition_converted_by_truthiness(self):
        """A non-``Boolean`` condition vector converts lane-wise by ``!= 0``,
        matching scalar select (previously produced invalid IR)."""

        def body():
            cond = _vector(fx.Int32)  # i32 vector, not i1
            r = cond.select(_vector(fx.Float32), _vector(fx.Float32))
            assert r.dtype is fx.Float32
            assert r.shape == (4,)

        ir_text = source_ir(body)
        assert "arith.cmpi" in ir_text  # lane-wise != 0
        assert "arith.select" in ir_text

    def test_vector_boolean_condition_no_extra_conversion(self):
        """A genuine Boolean condition vector is used as-is (no spurious != 0)."""

        def body():
            cond = _vector(fx.Int32) < _vector(fx.Int32)  # already Boolean vector
            cond.select(_vector(fx.Float32), _vector(fx.Float32))

        # exactly one cmpi: the comparison itself, no added truthiness test
        assert source_ir(body).count("arith.cmpi") == 1

    def test_non_boolean_condition_converted_by_truthiness(self):
        """A non-``Boolean`` condition is accepted and converted by a nonzero
        test (``!= 0``), not rejected as an invalid select operand."""

        def body(x: fx.Int32):
            r = x.select(fx.Int32(10), fx.Int32(20))
            assert dtype_of(r) is fx.Int32

        ir_text = source_ir(body, 5)
        assert "arith.cmpi" in ir_text  # x != 0 truthiness test
        assert "arith.select" in ir_text

    def test_non_boolean_float_condition(self):
        """A float condition also converts by truthiness (cmpf != 0)."""

        def body(x: fx.Float32):
            r = x.select(fx.Int32(1), fx.Int32(2))
            assert dtype_of(r) is fx.Int32

        assert "arith.cmpf" in source_ir(body, 1.0)

    def test_static_boolean_condition_folds(self):
        """A compile-time ``Boolean`` condition folds at trace time: the chosen
        branch is returned directly, staying compile-time (no ``arith.select``)."""

        def body():
            r_true = Boolean(True).select(Int32(10), Int32(20))
            assert r_true.is_static() and int(r_true) == 10
            r_false = Boolean(False).select(Int32(10), Int32(20))
            assert r_false.is_static() and int(r_false) == 20

        assert "arith.select" not in source_ir(body)

    def test_static_condition_folds_after_promotion(self):
        """Folding still yields the promoted common type."""

        def body():
            r = Boolean(True).select(Int32(1), Float32(2.0))
            assert r.is_static()
            assert dtype_of(r) is Float32
            assert float(r.value) == 1.0

        run(body)

    def test_static_non_boolean_condition_folds_by_truthiness(self):
        """A compile-time non-``Boolean`` condition converts by truthiness and
        then folds, preserving the compile-time category end-to-end."""

        def body():
            assert int(Int32(5).select(Int32(10), Int32(20))) == 10  # nonzero → true
            assert int(Int32(0).select(Int32(10), Int32(20))) == 20  # zero → false
            assert int(Float32(2.5).select(Int32(1), Int32(2))) == 1

        run(body)

    def test_static_condition_runtime_branch_stays_runtime(self):
        """Folding preserves category: a static condition selecting a run-time
        branch yields a run-time value."""

        def body(x: fx.Int32):
            r = Boolean(True).select(x, Int32(0))
            assert not r.is_static()

        run(body, 7)

    def test_runtime_condition_not_folded(self):
        """A run-time condition must NOT fold — it still emits ``arith.select``."""

        def body(x: fx.Int32):
            r = (x < Int32(3)).select(Int32(10), Int32(20))
            assert not r.is_static()

        assert "arith.select" in source_ir(body, 5)

    def test_scalar_boolean_select_with_vector_branches_falls_back(self):
        """A scalar ``Boolean`` condition with non-scalar (Vector) branches takes
        the raw-select fallback (no scalar promotion) and still produces a
        correctly-typed Vector."""

        def body():
            r = (Int32(1) < Int32(2)).select(_vector(fx.Float32), _vector(fx.Float32))
            assert isinstance(r, fx.Vector)
            assert r.dtype is fx.Float32
            assert r.shape == (4,)

        run(body)


# ── Vector type specifics ───────────────────────────────────────────────────


class TestConstruction:

    def test_init_from_vector(self):
        def body():
            raw = vec(Float32).ir_value()
            t = Vector(raw, 8, Float32)
            assert t.shape == (8,)
            assert t.dtype is Float32
            assert t.element_type is Float32
            assert t.numel == 8

        run(body)

    def test_init_shape_int_vs_tuple(self):
        def body():
            raw = vec(Float32).ir_value()
            t1 = Vector(raw, 8, Float32)
            t2 = Vector(raw, (8,), Float32)
            assert t1.shape == t2.shape == (8,)

        run(body)

    def test_signed_false_for_float(self):
        def body():
            assert vec(Float32).signed is False

        run(body)

    def test_signed_true_for_int32(self):
        def body():
            assert vec(Int32).signed is True

        run(body)

    def test_signed_false_for_uint32(self):
        def body():
            assert vec(Uint32).signed is False

        run(body)

    def test_str_repr(self):
        def body():
            s = str(vec(Float32))
            assert "Vector" in s
            assert "Float32" in s

        run(body)


class TestVectorAliases:
    def test_alias_is_specialized_subclass(self):
        from flydsl.expr.typing import Float32x4

        assert issubclass(Float32x4, Vector)

        def body():
            _assert_alias_ir_type(Float32x4, (4,), Float32)

        run(body)

    def test_alias_exported_from_package(self):
        import flydsl.expr as fx

        assert fx.Float32x4 is Float32x4

        def body():
            _assert_alias_ir_type(fx.BFloat16x8, (8,), BFloat16)
            _assert_alias_ir_type(fx.Int8x16, (16,), Int8)

        run(body)

    def test_make_type_matches_plain_vector(self):
        from flydsl.expr.typing import Float16x8

        def body():
            assert Float32x4.make_type(4, Float32) == Vector.make_type(4, Float32)
            assert Float16x8.make_type(8, Float16) == Vector.make_type(8, Float16)

        run(body)

    def test_construct_enforces_fixed_shape_and_dtype(self):
        from flydsl.expr.typing import Float32x2

        def body():
            v = Float32x4.filled(4, 1.0, Float32)
            assert isinstance(v, Float32x4)
            assert v.dtype is Float32
            assert v.shape == (4,)
            with pytest.raises(ValueError):
                # value has 4 elements but the alias fixes the shape to (2,)
                Float32x2(v.ir_value())
            with pytest.raises(ValueError):
                Float32x4(v.ir_value(), dtype=Float16)

        run(body)

    def test_construct_accepts_matching_dtype_keyword(self):
        def body():
            v = Float32x4(Vector.filled(4, 1.0, Float32).ir_value(), dtype=Float32)
            assert isinstance(v, Float32x4)
            assert v.dtype is Float32
            assert v.shape == (4,)

        run(body)

    def test_construct_from_scalar_splat(self):
        from flydsl.expr.typing import Int8x16

        def body():
            f32_vec = Float32x4(1.0)
            assert isinstance(f32_vec, Float32x4)
            assert f32_vec.dtype is Float32
            assert f32_vec.shape == (4,)

            i8_vec = Int8x16(7)
            assert isinstance(i8_vec, Int8x16)
            assert i8_vec.dtype is Int8
            assert i8_vec.shape == (16,)

        run(body)

    def test_construct_from_typed_list_elements(self):
        from flydsl.expr.typing import Int32x4

        def body():
            f32_vec = Float32x4([Float32(0.0), Float32(1.0), Float32(2.0), Float32(3.0)])
            assert isinstance(f32_vec, Float32x4)
            assert f32_vec.dtype is Float32
            assert f32_vec.shape == (4,)

            i32_vec = Int32x4([i for i in range(4)])
            assert isinstance(i32_vec, Int32x4)
            assert i32_vec.dtype is Int32
            assert i32_vec.shape == (4,)

        run(body)

    def test_construct_from_literal_list_uses_alias_dtype(self):
        from flydsl.expr.typing import Int8x16

        def body():
            f32_vec = Float32x4([0, 1, 2, 3])
            assert isinstance(f32_vec, Float32x4)
            assert f32_vec.dtype is Float32
            assert f32_vec.shape == (4,)

            i8_vec = Int8x16([i for i in range(16)])
            assert isinstance(i8_vec, Int8x16)
            assert i8_vec.dtype is Int8
            assert i8_vec.shape == (16,)

        run(body)

    def test_construct_from_typed_tuple_elements(self):
        from flydsl.expr.typing import Float16x8

        def body():
            v = Float16x8(tuple(Float16(i) for i in range(8)))
            assert isinstance(v, Float16x8)
            assert v.dtype is Float16
            assert v.shape == (8,)

        run(body)


class TestProtocol:
    def test_extract_to_ir_values_roundtrip(self):
        def body():
            values = vec(Float32).__extract_to_ir_values__()
            assert len(values) == 1
            reconstructed = Vector.__construct_from_ir_values__(values)
            assert isinstance(reconstructed, ir.Value)

        run(body)

    def test_hash(self):
        """Vector must be hashable since __eq__ is overridden."""

        def body():
            h = hash(vec(Float32))
            assert isinstance(h, int)

        run(body)


# ###########################################################################
# Part 2 — Compile-time and run-time values
#   (docs/language/arithmetic_types.md → ## Compile-time and run-time values)
# ###########################################################################


class TestCompileTimeVsRuntime:
    """Arithmetic preserves the compile-time property (spec: "Compile-time and
    run-time values"): compile-time ⊕ compile-time folds to a compile-time
    value; contact with a run-time value yields a run-time value."""

    def test_const_op_const_stays_compile_time(self):
        def body():
            r = fx.Int32(3) + fx.Int32(4)
            assert r.is_static()
            assert int(r) == 7

        run(body)

    def test_const_op_runtime_becomes_runtime(self):
        def body(n: fx.Int32):
            r = fx.Int32(3) + n
            assert not r.is_static()

        run(body, 4)

    def test_runtime_op_runtime_stays_runtime(self):
        def body(a: fx.Int32, b: fx.Int32):
            assert not (a + b).is_static()

        run(body, 3, 4)


class TestCompileTimeAsPython:
    """A compile-time ``Numeric`` is usable wherever Python expects a value; a
    run-time one raises when forced to Python."""

    def test_int_and_bool_and_if(self):
        def body():
            assert int(Int32(5)) == 5
            assert bool(Boolean(True)) is True
            if Int32(1):
                pass
            else:
                raise AssertionError("compile-time truthiness failed")

        run(body)

    def test_runtime_forced_to_python_raises(self):
        def body(n: fx.Int32):
            with pytest.raises(RuntimeError):
                int(n)

        run(body, 5)


# ###########################################################################
# Part 3 — Type interoperability
#   (docs/language/arithmetic_types.md → ## Type interoperability)
# ###########################################################################


# ── Operand normalization ───────────────────────────────────────────────────


class TestNormalizationEdges:
    """Operand-normalization corners from the spec that the core lattice test
    (compact table) does not reach."""

    def test_big_int_literal_enters_as_int64(self):
        """A Python int outside the Int32 range normalizes to Int64."""

        def body():
            assert dtype_of(Int32(3) + 5_000_000_000) is Int64

        run(body)

    def test_small_int_literal_enters_as_int32(self):
        def body():
            assert dtype_of(Int32(3) + 2) is Int32

        run(body)

    def test_bool_shift_stays_boolean(self):
        """Per the spec, a Boolean in a shift stays Boolean."""

        def body():
            r = Boolean(True) << Boolean(True)
            assert dtype_of(r) is Boolean

        run(body)


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
        def body():
            _assert_result_types(op, a, b, expected, operand_kinds=_ALL_KIND_OPERANDS)

        run(body)

    def test_python_literal_operands(self):
        """Literals enter the lattice by DSL type: int→Int32, float→Float32."""

        def body():
            # Int32 vector + python float → Float32
            assert dtype_of(_vector(fx.Int32) + 1.0) is fx.Float32
            # Float32 vector + python int → Float32
            assert dtype_of(_vector(fx.Float32) + 1) is fx.Float32
            # scalar Int32 + python float → Float32
            assert dtype_of(_scalar(fx.Int32) + 1.0) is fx.Float32

        run(body)


# ── Common type ─────────────────────────────────────────────────────────────


class TestCommonTypeLattice:
    @pytest.mark.parametrize("a", CORE, ids=lambda t: t.__name__)
    def test_arith_result_matches_spec_scalar_and_vector(self, a):
        """`+` result equals Table A with bool pre-widened to i32, for BOTH the
        scalar and vector operator paths (operand-kind independence + the single
        shared lattice)."""

        def body():
            for b in CORE:
                expected = _expected_common_type(_widen_bool(a), _widen_bool(b))
                _assert_result_types(operator.add, a, b, expected)

        run(body)


class TestDynamicScalarInterop:
    """Dynamic scalar operands use the same type interop rules as the spec table.

    These cases keep coverage for widths outside the compact core table without
    maintaining a second type-rule test file, and exercise the promotion path
    with run-time operands (jit parameters).
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
        assert dynamic_binop(ty, ty, operator.add)[0] is ty
        assert dynamic_binop(ty, ty, operator.mul)[0] is ty

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
        assert dynamic_binop(a, b, operator.add)[0] is expected
        assert dynamic_binop(b, a, operator.add)[0] is expected

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
        assert dynamic_binop(a, b, operator.add)[0] is expected
        assert dynamic_binop(b, a, operator.add)[0] is expected

    def test_python_int_literal_enters_as_int32(self):
        assert dynamic_literal_binop(fx.Int8, 5, operator.add) is fx.Int32

    def test_int128_plus_float64_uses_float64(self):
        assert dynamic_binop(fx.Int128, fx.Float64, operator.add)[0] is fx.Float64
        assert dynamic_binop(fx.Float64, fx.Int128, operator.add)[0] is fx.Float64

    def test_int128_truediv_lifts_to_float64(self):
        assert dynamic_binop(fx.Int128, fx.Int128, operator.truediv)[0] is fx.Float64

    def test_int128_floordiv_stays_integer(self):
        assert dynamic_binop(fx.Int128, fx.Int128, operator.floordiv)[0] is fx.Int128


# ── Result type ─────────────────────────────────────────────────────────────


class TestOperationResultType:
    @pytest.mark.parametrize(
        "op", [operator.add, operator.sub, operator.mul, operator.floordiv, operator.mod, operator.pow]
    )
    @pytest.mark.parametrize("ty", _ARITH_SAME, ids=lambda t: t.__name__)
    def test_arith_keeps_common_type(self, op, ty):
        """`+ - * // % **` all keep the common type `C` (spec result-type table)."""

        def body():
            _assert_result_types(op, ty, ty, ty)

        run(body)

    @pytest.mark.parametrize("ty", _ARITH_SAME, ids=lambda t: t.__name__)
    def test_truediv_result(self, ty):
        def body():
            _assert_result_types(operator.truediv, ty, ty, _TRUEDIV_EXPECTED[ty])

        run(body)

    @pytest.mark.parametrize("op", [operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne])
    @pytest.mark.parametrize("ty", [fx.Int32, fx.Uint32, fx.Float32], ids=lambda t: t.__name__)
    def test_comparison_returns_boolean(self, op, ty):
        def body():
            _assert_result_types(op, ty, ty, fx.Boolean)

        run(body)

    @pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor, operator.lshift, operator.rshift])
    @pytest.mark.parametrize("ty", _INT_TYPES, ids=lambda t: t.__name__)
    def test_bitwise_shift_keeps_int_type(self, op, ty):
        def body():
            _assert_result_types(op, ty, ty, ty)

        run(body)

    @pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor, operator.lshift, operator.rshift])
    @pytest.mark.parametrize("ty", _FLOAT_TYPES, ids=lambda t: t.__name__)
    def test_bitwise_shift_on_float_raises(self, op, ty):
        """Bitwise/shift on a float common type raises TypeError for BOTH paths
        (previously the vector path emitted invalid IR)."""

        def body():
            _assert_raises(op, ty, ty)

        run(body)

    @pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor, operator.lshift, operator.rshift])
    @pytest.mark.parametrize(
        "a,b",
        _INT_FLOAT_PAIRS,
        ids=[f"{a.__name__}-{b.__name__}" for a, b in _INT_FLOAT_PAIRS],
    )
    def test_bitwise_shift_on_int_float_common_type_raises(self, op, a, b):
        def body():
            _assert_raises(op, a, b, operand_kinds=_ALL_KIND_OPERANDS)

        run(body)


class TestBoolArithmetic:
    @pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul])
    def test_bool_arith_widens_to_int32(self, op):
        def body():
            _assert_result_types(op, fx.Boolean, fx.Boolean, fx.Int32)

        run(body)

    def test_bool_comparison_stays_boolean(self):
        def body():
            _assert_result_types(operator.lt, fx.Boolean, fx.Boolean, fx.Boolean)

        run(body)

    @pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor])
    def test_bool_bitwise_stays_boolean(self, op):
        def body():
            _assert_result_types(op, fx.Boolean, fx.Boolean, fx.Boolean)

        run(body)


class TestIntermediateType:
    def test_comparison_uses_common_type_before_compare(self):
        """`Int32 < Float32`: intermediate is f32 (common type), result is bool.

        The integer operand is cast (sitofp) to f32 and the comparison is cmpf.
        """
        result_dtype, ir_text = dynamic_binop(fx.Int32, fx.Float32, operator.lt)
        assert result_dtype is fx.Boolean
        assert "arith.sitofp" in ir_text  # int → f32 intermediate cast
        assert "arith.cmpf" in ir_text  # compared as f32
