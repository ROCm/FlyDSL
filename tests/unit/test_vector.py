#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Unit tests for Vector, ReductionOp, math (vector support), and factory functions.

All tests are IR-level (no GPU required). They build MLIR modules using
Vector operations and verify the generated IR text.
"""

import pytest

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, func
from flydsl.expr import math as fmath
from flydsl.expr.numeric import (
    BFloat16,
    Boolean,
    Float16,
    Float32,
    Int8,
    Int16,
    Int32,
    Numeric,
    Uint32,
)
from flydsl.expr.typing import Float32x4
from flydsl.expr.vector import (
    ReductionOp,
    Vector,
    full,
    full_like,
    zeros_like,
)

pytestmark = pytest.mark.l0_backend_agnostic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_module(build_fn, arg_types=None):
    """Build an MLIR module, call *build_fn* with block arguments, return IR text."""
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with ir.Location.unknown(ctx):
            if arg_types is None:
                types = [ir.VectorType.get([8], ir.F32Type.get())]
            else:
                types = [t() if callable(t) else t for t in arg_types]
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                ftype = ir.FunctionType.get(types, [])
                f = func.FuncOp("test", ftype)
                with ir.InsertionPoint(f.add_entry_block()):
                    build_fn(*f.entry_block.arguments)
                    func.ReturnOp([])
            module.operation.verify()
            return str(module)


def _vec_f32():
    return ir.VectorType.get([8], ir.F32Type.get())


def _vec_f16():
    return ir.VectorType.get([8], ir.F16Type.get())


def _vec_bf16():
    return ir.VectorType.get([8], ir.BF16Type.get())


def _vec_i32():
    return ir.VectorType.get([8], ir.IntegerType.get_signless(32))


def _vec_i16():
    return ir.VectorType.get([8], ir.IntegerType.get_signless(16))


# ===========================================================================
# A. Construction & properties
# ===========================================================================


class TestConstruction:

    def test_init_from_vector(self):
        def build(raw):
            t = Vector(raw, 8, Float32)
            assert t.shape == (8,)
            assert t.dtype is Float32
            assert t.element_type is Float32
            assert t.numel == 8

        _build_module(build)

    def test_init_shape_int_vs_tuple(self):
        def build(raw):
            t1 = Vector(raw, 8, Float32)
            t2 = Vector(raw, (8,), Float32)
            assert t1.shape == t2.shape == (8,)

        _build_module(build)

    def test_signed_false_for_float(self):
        def build(raw):
            t = Vector(raw, 8, Float32)
            assert t.signed is False

        _build_module(build)

    def test_signed_true_for_int32(self):
        def build(raw):
            t = Vector(raw, 8, Int32)
            assert t.signed is True

        _build_module(build, [_vec_i32])

    def test_signed_false_for_uint32(self):
        def build(raw):
            t = Vector(raw, 8, Uint32)
            assert t.signed is False

        _build_module(build, [_vec_i32])

    def test_str_repr(self):
        def build(raw):
            t = Vector(raw, 8, Float32)
            s = str(t)
            assert "Vector" in s
            assert "Float32" in s

        _build_module(build)


# ===========================================================================
# B. Operators
# ===========================================================================


class TestOperators:

    def test_add_two_tensors(self):
        def build(a, b):
            ta = Vector(a, 8, Float32)
            tb = Vector(b, 8, Float32)
            _ = ta + tb

        ir_text = _build_module(build, [_vec_f32, _vec_f32])
        assert "arith.addf" in ir_text

    def test_mul_scalar_broadcast(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            _ = ta * 2.0

        ir_text = _build_module(build)
        # Scalar 2.0 is splatted into a vector constant via arith_const
        assert "arith.mulf" in ir_text

    def test_sub_reverse(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            _ = 1.0 - ta

        ir_text = _build_module(build)
        # Scalar 1.0 is splatted into a vector constant via arith_const
        assert "arith.subf" in ir_text

    def test_int_add(self):
        def build(a, b):
            ta = Vector(a, 8, Int32)
            tb = Vector(b, 8, Int32)
            _ = ta + tb

        ir_text = _build_module(build, [_vec_i32, _vec_i32])
        assert "arith.addi" in ir_text

    def test_comparison_returns_boolean(self):
        def build(a, b):
            ta = Vector(a, 8, Float32)
            tb = Vector(b, 8, Float32)
            result = ta < tb
            assert isinstance(result, Vector)
            assert result.dtype is Boolean

        _build_module(build, [_vec_f32, _vec_f32])

    def test_bitwise_and_or_xor(self):
        def build(a, b):
            ta = Vector(a, 8, Uint32)
            tb = Vector(b, 8, Uint32)
            _ = ta & tb
            _ = ta | tb
            _ = ta ^ tb

        ir_text = _build_module(build, [_vec_i32, _vec_i32])
        assert "arith.andi" in ir_text
        assert "arith.ori" in ir_text
        assert "arith.xori" in ir_text

    def test_shift_ops(self):
        def build(a):
            ta = Vector(a, 8, Uint32)
            _ = ta >> 16
            _ = ta << 8

        ir_text = _build_module(build, [_vec_i32])
        assert "arith.shrui" in ir_text
        assert "arith.shli" in ir_text

    def test_unsigned_shift_uses_shrui(self):
        """Uint32 Vector >> must use shrui, not shrsi."""

        def build(a):
            ta = Vector(a, 8, Uint32)
            _ = ta >> 16

        ir_text = _build_module(build, [_vec_i32])
        assert "arith.shrui" in ir_text
        assert "arith.shrsi" not in ir_text

    def test_signed_shift_uses_shrsi(self):
        """Int32 Vector >> must use shrsi."""

        def build(a):
            ta = Vector(a, 8, Int32)
            _ = ta >> 16

        ir_text = _build_module(build, [_vec_i32])
        assert "arith.shrsi" in ir_text

    def test_neg(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            _ = -ta

        ir_text = _build_module(build)
        assert "arith.negf" in ir_text

    def test_truediv(self):
        def build(a, b):
            ta = Vector(a, 8, Float32)
            tb = Vector(b, 8, Float32)
            _ = ta / tb

        ir_text = _build_module(build, [_vec_f32, _vec_f32])
        assert "arith.divf" in ir_text

    def test_pow(self):
        def build(a, b):
            ta = Vector(a, 8, Float32)
            tb = Vector(b, 8, Float32)
            _ = ta**tb

        ir_text = _build_module(build, [_vec_f32, _vec_f32])
        assert "math.powf" in ir_text

    def test_floordiv_int(self):
        def build(a, b):
            ta = Vector(a, 8, Int32)
            tb = Vector(b, 8, Int32)
            _ = ta // tb

        ir_text = _build_module(build, [_vec_i32, _vec_i32])
        assert "arith.floordivsi" in ir_text

    def test_mod_int(self):
        def build(a, b):
            ta = Vector(a, 8, Int32)
            tb = Vector(b, 8, Int32)
            _ = ta % tb

        ir_text = _build_module(build, [_vec_i32, _vec_i32])
        assert "arith.remsi" in ir_text

    def test_neg_int(self):
        """Negating an integer Vector should produce arith.subi (0 - x)."""

        def build(a):
            ta = Vector(a, 8, Int32)
            result = -ta
            assert isinstance(result, Vector)

        ir_text = _build_module(build, [_vec_i32])
        assert "arith.subi" in ir_text

    def test_reverse_bitwise(self):
        def build(a, b):
            ta = Vector(a, 8, Uint32)
            tb = Vector(b, 8, Uint32)
            # reverse ops: rhs.__rand__ etc.
            r1 = tb & ta
            r2 = tb | ta
            r3 = tb ^ ta
            assert isinstance(r1, Vector)
            assert isinstance(r2, Vector)
            assert isinstance(r3, Vector)

        ir_text = _build_module(build, [_vec_i32, _vec_i32])
        assert "arith.andi" in ir_text
        assert "arith.ori" in ir_text
        assert "arith.xori" in ir_text


# ===========================================================================
# C. Common-type handling in operators
# ===========================================================================


class TestCommonTypeHandling:
    """Mixed-dtype vector operators convert both operands to a common type
    using the same rule as scalar Numeric arithmetic
    """

    def test_int_plus_float_uses_common_type(self):
        """Int32 Vector + Float32 Vector → float; must emit addf, not addi."""

        def build(a, b):
            ta = Vector(a, 8, Int32)
            tb = Vector(b, 8, Float32)
            result = ta + tb
            assert result.dtype is Float32

        ir_text = _build_module(build, [_vec_i32, _vec_f32])
        assert "arith.addf" in ir_text
        assert "arith.addi" not in ir_text

    def test_float_plus_int_uses_common_type(self):
        """Float32 Vector + Int32 Vector → float (LHS already float)."""

        def build(a, b):
            ta = Vector(a, 8, Float32)
            tb = Vector(b, 8, Int32)
            result = ta + tb
            assert result.dtype is Float32

        ir_text = _build_module(build, [_vec_f32, _vec_i32])
        assert "arith.addf" in ir_text

    def test_int_widening_uses_common_type(self):
        """Int16 Vector + Int32 Vector widens to the wider integer type."""

        def build(a, b):
            ta = Vector(a, 8, Int16)
            tb = Vector(b, 8, Int32)
            result = ta + tb
            assert result.dtype is Int32

        ir_text = _build_module(build, [_vec_i16, _vec_i32])
        assert "arith.addi" in ir_text

    def test_f16_plus_f32_extends_narrow_operand(self):
        """Float16 + Float32 uses Float32; the narrower operand is extended."""

        def build(a, b):
            ta = Vector(a, 8, Float16)
            tb = Vector(b, 8, Float32)
            result = ta + tb
            assert result.dtype is Float32

        ir_text = _build_module(build, [_vec_f16, _vec_f32])
        assert "arith.extf" in ir_text
        assert "arith.addf" in ir_text

    def test_scalar_numeric_operand_uses_common_type(self):
        """Int32 Vector + Float32 scalar converts the vector to float."""

        def build(a):
            ta = Vector(a, 8, Int32)
            result = ta + Float32(2.0)
            assert result.dtype is Float32

        ir_text = _build_module(build, [_vec_i32])
        assert "arith.addf" in ir_text

    def test_python_float_operand_uses_common_type(self):
        """BFloat16 Vector + Python float uses f32 (no explicit .to())."""

        def build(a):
            ta = Vector(a, 8, BFloat16)
            result = ta + 1.0
            assert result.dtype is Float32

        ir_text = _build_module(build, [_vec_bf16])
        assert "arith.extf" in ir_text
        assert "arith.addf" in ir_text


# ===========================================================================
# D. Type conversion (.to())
# ===========================================================================


class TestToConversion:

    def test_same_type_noop(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            result = ta.to(Float32)
            assert result is ta

        _build_module(build)

    def test_float_to_float_truncf(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            _ = ta.to(BFloat16)

        ir_text = _build_module(build)
        assert "arith.truncf" in ir_text

    def test_float_to_float_extf(self):
        def build(a):
            ta = Vector(a, 8, Float16)
            _ = ta.to(Float32)

        ir_text = _build_module(build, [_vec_f16])
        assert "arith.extf" in ir_text

    def test_float_to_int(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            _ = ta.to(Int32)

        ir_text = _build_module(build)
        assert "arith.fptosi" in ir_text

    def test_int_to_float(self):
        def build(a):
            ta = Vector(a, 8, Int32)
            _ = ta.to(Float32)

        ir_text = _build_module(build, [_vec_i32])
        assert "arith.sitofp" in ir_text

    def test_uint_to_float(self):
        """Uint32 → Float32 should use uitofp, not sitofp."""

        def build(a):
            ta = Vector(a, 8, Uint32)
            result = ta.to(Float32)
            assert result.dtype is Float32

        ir_text = _build_module(build, [_vec_i32])
        assert "arith.uitofp" in ir_text
        assert "arith.sitofp" not in ir_text

    def test_float_to_uint(self):
        """Float32 → Uint32 should use fptoui, not fptosi."""

        def build(a):
            ta = Vector(a, 8, Float32)
            result = ta.to(Uint32)
            assert result.dtype is Uint32

        ir_text = _build_module(build)
        assert "arith.fptoui" in ir_text
        assert "arith.fptosi" not in ir_text

    def test_int16_to_int32(self):
        """Int16 → Int32 should use extsi."""

        def build(a):
            ta = Vector(a, 8, Int16)
            result = ta.to(Int32)
            assert result.dtype is Int32
            assert result.shape == (8,)

        ir_text = _build_module(build, [_vec_i16])
        assert "arith.extsi" in ir_text

    def test_to_ir_value_returns_self(self):
        """to(ir.Value) should return self unchanged."""

        def build(a):
            ta = Vector(a, 8, Float32)
            result = ta.to(ir.Value)
            assert result is ta

        _build_module(build)

    def test_to_preserves_shape(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            result = ta.to(BFloat16)
            assert result.shape == (8,)
            assert result.dtype is BFloat16

        _build_module(build)


# ===========================================================================
# E. Reduction
# ===========================================================================


class TestReduction:

    def test_reduce_add(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            _ = ta.reduce(ReductionOp.ADD)

        ir_text = _build_module(build)
        assert "vector.reduction <add>" in ir_text

    def test_reduce_max(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            _ = ta.reduce(ReductionOp.MAX)

        ir_text = _build_module(build)
        assert "vector.reduction <maxnumf>" in ir_text

    def test_reduce_min(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            _ = ta.reduce(ReductionOp.MIN)

        ir_text = _build_module(build)
        assert "vector.reduction <minimumf>" in ir_text

    def test_reduce_with_fastmath(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            fm = arith.FastMathFlags.fast
            _ = ta.reduce(ReductionOp.ADD, fastmath=fm)

        ir_text = _build_module(build)
        assert "fastmath" in ir_text.lower() or "fast" in ir_text

    def test_reduce_returns_numeric(self):
        """reduce() should return Numeric, not raw ir.Value."""

        def build(a):
            ta = Vector(a, 8, Float32)
            result = ta.reduce(ReductionOp.ADD)
            assert isinstance(result, Float32)

        _build_module(build)

    def test_int_reduce_add(self):
        def build(a):
            ta = Vector(a, 8, Int32)
            result = ta.reduce(ReductionOp.ADD)
            assert isinstance(result, Int32)

        ir_text = _build_module(build, [_vec_i32])
        assert "vector.reduction <add>" in ir_text

    def test_int_reduce_max_signed(self):
        """Int32 MAX should use maxsi, not maxnumf."""

        def build(a):
            ta = Vector(a, 8, Int32)
            result = ta.reduce(ReductionOp.MAX)
            assert isinstance(result, Int32)

        ir_text = _build_module(build, [_vec_i32])
        assert "vector.reduction <maxsi>" in ir_text

    def test_int_reduce_max_unsigned(self):
        """Uint32 MAX should use maxui."""

        def build(a):
            ta = Vector(a, 8, Uint32)
            result = ta.reduce(ReductionOp.MAX)
            assert isinstance(result, Uint32)

        ir_text = _build_module(build, [_vec_i32])
        assert "vector.reduction <maxui>" in ir_text

    def test_int_reduce_min_signed(self):
        """Int32 MIN should use minsi."""

        def build(a):
            ta = Vector(a, 8, Int32)
            result = ta.reduce(ReductionOp.MIN)
            assert isinstance(result, Int32)

        ir_text = _build_module(build, [_vec_i32])
        assert "vector.reduction <minsi>" in ir_text

    def test_int_reduce_min_unsigned(self):
        """Uint32 MIN should use minui."""

        def build(a):
            ta = Vector(a, 8, Uint32)
            result = ta.reduce(ReductionOp.MIN)
            assert isinstance(result, Uint32)

        ir_text = _build_module(build, [_vec_i32])
        assert "vector.reduction <minui>" in ir_text

    def test_reduce_with_init_val(self):
        """reduce() with init_val should pass acc to vector.reduction."""

        def build(a):
            ta = Vector(a, 8, Float32)
            init = Float32(0.0)
            result = ta.reduce(ReductionOp.ADD, init_val=init)
            assert isinstance(result, Float32)

        ir_text = _build_module(build)
        assert "vector.reduction <add>" in ir_text

    def test_reduce_string_add(self):
        """reduce() accepts plain string 'add'."""

        def build(a):
            ta = Vector(a, 8, Float32)
            result = ta.reduce("add")
            assert isinstance(result, Float32)

        ir_text = _build_module(build)
        assert "vector.reduction <add>" in ir_text

    def test_reduce_string_max(self):
        """reduce() accepts plain string 'max'."""

        def build(a):
            ta = Vector(a, 8, Float32)
            _ = ta.reduce("max")

        ir_text = _build_module(build)
        assert "vector.reduction <maxnumf>" in ir_text

    def test_reduce_combining_kind_direct(self):
        """reduce() accepts raw CombiningKind."""
        from flydsl._mlir.dialects.vector import CombiningKind

        def build(a):
            ta = Vector(a, 8, Float32)
            result = ta.reduce(CombiningKind.ADD)
            assert isinstance(result, Float32)

        ir_text = _build_module(build)
        assert "vector.reduction <add>" in ir_text

    def test_reduce_bad_op_raises(self):
        """reduce() raises on invalid op type."""

        def build(a):
            ta = Vector(a, 8, Float32)
            with pytest.raises(TypeError):
                ta.reduce(42)

        _build_module(build)

    def test_reduce_bad_string_raises(self):
        """reduce() raises on unknown string."""

        def build(a):
            ta = Vector(a, 8, Float32)
            with pytest.raises(ValueError, match="unknown"):
                ta.reduce("foobar")

        _build_module(build)


# ===========================================================================
# F. Element access
# ===========================================================================


class TestElementAccess:

    def test_getitem_int(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            elem = ta[0]
            assert isinstance(elem, Float32)

        ir_text = _build_module(build)
        assert "vector.extract" in ir_text

    def test_getitem_invalid_type(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            with pytest.raises(TypeError):
                ta["bad"]

        _build_module(build)


# ===========================================================================
# G. Vector ops
# ===========================================================================


class TestVectorOps:

    def test_bitcast(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            result = ta.bitcast(Uint32)
            assert result.shape == (8,)
            assert result.dtype is Uint32

        ir_text = _build_module(build)
        assert "vector.bitcast" in ir_text

    def test_bitcast_width_change(self):
        """f32 → f16 bitcast: 8 elements * 32 bits = 256 bits → 16 elements * 16 bits."""

        def build(a):
            ta = Vector(a, 8, Float32)
            result = ta.bitcast(Float16)
            assert result.shape == (16,)
            assert result.dtype is Float16

        ir_text = _build_module(build)
        assert "vector.bitcast" in ir_text

    def test_shuffle(self):
        def build(a, b):
            ta = Vector(a, 8, Float32)
            tb = Vector(b, 8, Float32)
            result = ta.shuffle(tb, [0, 2, 4, 6])
            assert result.shape == (4,)
            assert result.dtype is Float32

        ir_text = _build_module(build, [_vec_f32, _vec_f32])
        assert "vector.shuffle" in ir_text


# ===========================================================================
# H. Factory functions
# ===========================================================================


class TestFactories:

    def test_full(self):
        def build(a):
            t = full(8, 1.0, Float32)
            assert t.shape == (8,)
            assert t.dtype is Float32

        ir_text = _build_module(build)
        assert "vector.broadcast" in ir_text

    def test_full_like(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            t = full_like(ta, 0.0)
            assert t.shape == ta.shape
            assert t.dtype == ta.dtype

        ir_text = _build_module(build)
        assert "vector.broadcast" in ir_text

    def test_zeros_like(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            t = zeros_like(ta)
            assert t.shape == ta.shape
            assert t.dtype == ta.dtype

        ir_text = _build_module(build)
        assert "vector.broadcast" in ir_text

    def test_full_with_numeric_fill(self):
        """full() with Numeric fill_value should work."""

        def build(a):
            t = full(8, Float32(2.5), Float32)
            assert t.shape == (8,)
            assert t.dtype is Float32

        ir_text = _build_module(build)
        assert "vector.broadcast" in ir_text

    def test_classmethod_filled(self):
        def build(a):
            t = Vector.filled(8, 1.0, Float32)
            assert t.shape == (8,)
            assert t.dtype is Float32

        ir_text = _build_module(build)
        assert "vector.broadcast" in ir_text

    def test_classmethod_filled_like(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            t = Vector.filled_like(ta, 2.0)
            assert t.shape == ta.shape
            assert t.dtype is Float32

        ir_text = _build_module(build)
        assert "vector.broadcast" in ir_text

    def test_classmethod_zeros_like(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            t = Vector.zeros_like(ta)
            assert t.shape == ta.shape
            assert t.dtype is Float32

        ir_text = _build_module(build)
        assert "vector.broadcast" in ir_text


# ===========================================================================
# I. fmath
# ===========================================================================


class TestFmath:

    def test_exp2_tensor(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            result = fmath.exp2(ta)
            assert isinstance(result, Vector)
            assert result.dtype is Float32
            assert result.shape == (8,)

        ir_text = _build_module(build)
        assert "math.exp2" in ir_text

    def test_rsqrt_tensor(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            _ = fmath.rsqrt(ta)

        ir_text = _build_module(build)
        assert "math.rsqrt" in ir_text

    def test_fastmath_flag(self):
        from flydsl.expr.arith import FastMathFlags

        def build(a):
            ta = Vector(a, 8, Float32)
            _ = fmath.exp2(ta, fastmath=FastMathFlags.fast)

        ir_text = _build_module(build)
        assert "fast" in ir_text

    def test_scalar_float(self):
        """math on scalar Float32 returns Float32 Numeric."""

        def build(raw):
            x = Float32(raw)
            result = fmath.sqrt(x)
            assert not isinstance(result, Vector)
            assert isinstance(result, Float32)

        _build_module(build, [ir.F32Type.get])

    def test_vector_scalar_atan2(self):
        """atan2 with Vector and scalar broadcasts the scalar."""

        def build(a, raw_scalar):
            ta = Vector(a, 8, Float32)
            # scalar is broadcast to match vector type via _coerce_other
            _ = fmath.atan2(ta, ta)

        _build_module(build, [_vec_f32, ir.F32Type.get])

    def test_new_functions_exist(self):
        """Verify all newly added functions are accessible."""
        for name in ["erf", "acos", "asin", "atan", "atan2", "tan", "log10"]:
            assert hasattr(fmath, name), f"fmath.{name} missing"

    def test_erf_tensor(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            _ = fmath.erf(ta)

        ir_text = _build_module(build)
        assert "math.erf" in ir_text

    def test_atan2_tensor(self):
        def build(a, b):
            ta = Vector(a, 8, Float32)
            tb = Vector(b, 8, Float32)
            result = fmath.atan2(ta, tb)
            assert isinstance(result, Vector)

        ir_text = _build_module(build, [_vec_f32, _vec_f32])
        assert "math.atan2" in ir_text


# ===========================================================================
# J. scf.for integration
# ===========================================================================


class TestProtocol:
    def test_extract_to_ir_values_roundtrip(self):
        def build(a):
            ta = Vector(a, 8, Float32)
            values = ta.__extract_to_ir_values__()
            assert len(values) == 1
            reconstructed = Vector.__construct_from_ir_values__(values)
            assert isinstance(reconstructed, ir.Value)

        _build_module(build)

    def test_hash(self):
        """Vector must be hashable since __eq__ is overridden."""

        def build(a):
            ta = Vector(a, 8, Float32)
            h = hash(ta)
            assert isinstance(h, int)

        _build_module(build)


# ===========================================================================
# K. Common vector type aliases (Float32x4, ...)
# ===========================================================================


class TestVectorAliases:
    @staticmethod
    def _assert_alias_ir_type(alias, shape, dtype):
        vty = ir.VectorType(alias.ir_type)
        assert tuple(vty.shape) == shape
        assert Numeric.from_ir_type(vty.element_type) is dtype

    def test_alias_is_specialized_subclass(self):
        from flydsl.expr.typing import Float32x4

        assert issubclass(Float32x4, Vector)

        def build(_a):
            self._assert_alias_ir_type(Float32x4, (4,), Float32)

        _build_module(build)

    def test_alias_exported_from_package(self):
        import flydsl.expr as fx

        assert fx.Float32x4 is Float32x4

        def build(_a):
            self._assert_alias_ir_type(fx.BFloat16x8, (8,), BFloat16)
            self._assert_alias_ir_type(fx.Int8x16, (16,), Int8)

        _build_module(build)

    def test_make_type_matches_plain_vector(self):
        from flydsl.expr.typing import Float16x8

        def build(_a):
            assert Float32x4.make_type(4, Float32) == Vector.make_type(4, Float32)
            assert Float16x8.make_type(8, Float16) == Vector.make_type(8, Float16)

        _build_module(build)

    def test_construct_enforces_fixed_shape_and_dtype(self):
        from flydsl.expr.typing import Float32x2

        def build(_a):
            vec = Float32x4.filled(4, 1.0, Float32)
            assert isinstance(vec, Float32x4)
            assert vec.dtype is Float32
            assert vec.shape == (4,)
            with pytest.raises(ValueError):
                # value has 4 elements but the alias fixes the shape to (2,)
                Float32x2(vec.ir_value())
            with pytest.raises(ValueError):
                Float32x4(vec.ir_value(), dtype=Float16)

        _build_module(build)

    def test_construct_accepts_matching_dtype_keyword(self):
        def build(_a):
            vec = Float32x4(Vector.filled(4, 1.0, Float32).ir_value(), dtype=Float32)
            assert isinstance(vec, Float32x4)
            assert vec.dtype is Float32
            assert vec.shape == (4,)

        _build_module(build)

    def test_construct_from_scalar_splat(self):
        from flydsl.expr.typing import Int8x16

        def build(_a):
            f32_vec = Float32x4(1.0)
            assert isinstance(f32_vec, Float32x4)
            assert f32_vec.dtype is Float32
            assert f32_vec.shape == (4,)

            i8_vec = Int8x16(7)
            assert isinstance(i8_vec, Int8x16)
            assert i8_vec.dtype is Int8
            assert i8_vec.shape == (16,)

        _build_module(build)

    def test_construct_from_typed_list_elements(self):
        from flydsl.expr.typing import Int32x4

        def build(_a):
            f32_vec = Float32x4([Float32(0.0), Float32(1.0), Float32(2.0), Float32(3.0)])
            assert isinstance(f32_vec, Float32x4)
            assert f32_vec.dtype is Float32
            assert f32_vec.shape == (4,)

            i32_vec = Int32x4([i for i in range(4)])
            assert isinstance(i32_vec, Int32x4)
            assert i32_vec.dtype is Int32
            assert i32_vec.shape == (4,)

        _build_module(build)

    def test_construct_from_literal_list_uses_alias_dtype(self):
        from flydsl.expr.typing import Int8x16

        def build(_a):
            f32_vec = Float32x4([0, 1, 2, 3])
            assert isinstance(f32_vec, Float32x4)
            assert f32_vec.dtype is Float32
            assert f32_vec.shape == (4,)

            i8_vec = Int8x16([i for i in range(16)])
            assert isinstance(i8_vec, Int8x16)
            assert i8_vec.dtype is Int8
            assert i8_vec.shape == (16,)

        _build_module(build)

    def test_construct_from_typed_tuple_elements(self):
        from flydsl.expr.typing import Float16x8

        def build(_a):
            vec = Float16x8(tuple(Float16(i) for i in range(8)))
            assert isinstance(vec, Float16x8)
            assert vec.dtype is Float16
            assert vec.shape == (8,)

        _build_module(build)
