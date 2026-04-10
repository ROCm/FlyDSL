#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Unit tests for TensorSSA, ReductionOp, fmath, and factory functions.

All tests are IR-level (no GPU required). They build MLIR modules using
TensorSSA operations and verify the generated IR text.
"""

import pytest

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, func

from flydsl.expr.tensor_ssa import (
    TensorSSA,
    ReductionOp,
    full,
    full_like,
    zeros_like,
    _resolve_promote_type,
)
from flydsl.expr.numeric import (
    Float32,
    Float16,
    BFloat16,
    Float64,
    Int32,
    Uint32,
    Boolean,
    Numeric,
)
from flydsl.expr import fmath

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


# ===========================================================================
# A. Construction & properties
# ===========================================================================

class TestConstruction:

    def test_init_from_vector(self):
        def build(raw):
            t = TensorSSA(raw, 8, Float32)
            assert t.shape == (8,)
            assert t.dtype is Float32
            assert t.element_type is Float32
            assert t.numel == 8
        _build_module(build)

    def test_init_shape_int_vs_tuple(self):
        def build(raw):
            t1 = TensorSSA(raw, 8, Float32)
            t2 = TensorSSA(raw, (8,), Float32)
            assert t1.shape == t2.shape == (8,)
        _build_module(build)

    def test_signed_false_for_float(self):
        def build(raw):
            t = TensorSSA(raw, 8, Float32)
            assert t.signed is False
        _build_module(build)

    def test_signed_true_for_int32(self):
        def build(raw):
            t = TensorSSA(raw, 8, Int32)
            assert t.signed is True
        _build_module(build, [_vec_i32])

    def test_signed_false_for_uint32(self):
        def build(raw):
            t = TensorSSA(raw, 8, Uint32)
            assert t.signed is False
        _build_module(build, [_vec_i32])

    def test_str_repr(self):
        def build(raw):
            t = TensorSSA(raw, 8, Float32)
            s = str(t)
            assert "TensorSSA" in s
            assert "Float32" in s
        _build_module(build)


# ===========================================================================
# B. Operators
# ===========================================================================

class TestOperators:

    def test_add_two_tensors(self):
        def build(a, b):
            ta = TensorSSA(a, 8, Float32)
            tb = TensorSSA(b, 8, Float32)
            _ = ta + tb
        ir_text = _build_module(build, [_vec_f32, _vec_f32])
        assert "arith.addf" in ir_text

    def test_mul_scalar_broadcast(self):
        def build(a):
            ta = TensorSSA(a, 8, Float32)
            _ = ta * 2.0
        ir_text = _build_module(build)
        assert "vector.broadcast" in ir_text
        assert "arith.mulf" in ir_text

    def test_sub_reverse(self):
        def build(a):
            ta = TensorSSA(a, 8, Float32)
            _ = 1.0 - ta
        ir_text = _build_module(build)
        assert "vector.broadcast" in ir_text
        assert "arith.subf" in ir_text

    def test_int_add(self):
        def build(a, b):
            ta = TensorSSA(a, 8, Int32)
            tb = TensorSSA(b, 8, Int32)
            _ = ta + tb
        ir_text = _build_module(build, [_vec_i32, _vec_i32])
        assert "arith.addi" in ir_text

    def test_comparison_returns_boolean(self):
        def build(a, b):
            ta = TensorSSA(a, 8, Float32)
            tb = TensorSSA(b, 8, Float32)
            result = ta < tb
            assert isinstance(result, TensorSSA)
            assert result.dtype is Boolean
        _build_module(build, [_vec_f32, _vec_f32])

    def test_bitwise_and_or_xor(self):
        def build(a, b):
            ta = TensorSSA(a, 8, Uint32)
            tb = TensorSSA(b, 8, Uint32)
            _ = ta & tb
            _ = ta | tb
            _ = ta ^ tb
        ir_text = _build_module(build, [_vec_i32, _vec_i32])
        assert "arith.andi" in ir_text
        assert "arith.ori" in ir_text
        assert "arith.xori" in ir_text

    def test_shift_ops(self):
        def build(a):
            ta = TensorSSA(a, 8, Uint32)
            _ = ta >> 16
            _ = ta << 8
        ir_text = _build_module(build, [_vec_i32])
        assert "arith.shrui" in ir_text
        assert "arith.shli" in ir_text

    def test_unsigned_shift_uses_shrui(self):
        """Uint32 TensorSSA >> must use shrui, not shrsi."""
        def build(a):
            ta = TensorSSA(a, 8, Uint32)
            _ = ta >> 16
        ir_text = _build_module(build, [_vec_i32])
        assert "arith.shrui" in ir_text
        assert "arith.shrsi" not in ir_text

    def test_signed_shift_uses_shrsi(self):
        """Int32 TensorSSA >> must use shrsi."""
        def build(a):
            ta = TensorSSA(a, 8, Int32)
            _ = ta >> 16
        ir_text = _build_module(build, [_vec_i32])
        assert "arith.shrsi" in ir_text

    def test_neg(self):
        def build(a):
            ta = TensorSSA(a, 8, Float32)
            _ = -ta
        ir_text = _build_module(build)
        assert "arith.subf" in ir_text

    def test_truediv(self):
        def build(a, b):
            ta = TensorSSA(a, 8, Float32)
            tb = TensorSSA(b, 8, Float32)
            _ = ta / tb
        ir_text = _build_module(build, [_vec_f32, _vec_f32])
        assert "arith.divf" in ir_text


# ===========================================================================
# C. Type promotion
# ===========================================================================

class TestTypePromotion:

    def test_same_type(self):
        assert _resolve_promote_type(Float32, Float32) is Float32

    def test_f16_f32(self):
        assert _resolve_promote_type(Float16, Float32) is Float32

    def test_bf16_f32(self):
        assert _resolve_promote_type(BFloat16, Float32) is Float32

    def test_int_float(self):
        """Int32 + Float32 → Float32."""
        assert _resolve_promote_type(Int32, Float32) is Float32

    def test_int_wider_than_float(self):
        """Float16 + Int32 → Float32 (int width 32 > float width 16)."""
        assert _resolve_promote_type(Float16, Int32) is Float32

    def test_int_same_width_as_float(self):
        """Float32 + Int32 → Float32 (same width, float wins)."""
        assert _resolve_promote_type(Float32, Int32) is Float32

    def test_int_narrower_than_float(self):
        """Float32 + Int16 → Float32 (int is narrower)."""
        from flydsl.expr.numeric import Int16
        assert _resolve_promote_type(Float32, Int16) is Float32

    def test_int64_with_float32(self):
        """Float32 + Int64 → Float64 (int width 64 > float width 32)."""
        from flydsl.expr.numeric import Int64
        assert _resolve_promote_type(Float32, Int64) is Float64

    def test_f16_f64(self):
        assert _resolve_promote_type(Float16, Float64) is Float64

    def test_promote_in_operator(self):
        """Float16 tensor + Float32 tensor → arith.extf + arith.addf."""
        def build(a, b):
            ta = TensorSSA(a, 8, Float16)
            tb = TensorSSA(b, 8, Float32)
            result = ta + tb
            assert result.dtype is Float32
        ir_text = _build_module(build, [_vec_f16, _vec_f32])
        assert "arith.extf" in ir_text
        assert "arith.addf" in ir_text

    def test_promote_bf16_scalar(self):
        """BFloat16 tensor + Float32 scalar → broadcast + arith.addf in f32."""
        def build(a):
            ta = TensorSSA(a, 8, BFloat16)
            result = ta + Float32(1.0)
            assert result.dtype is Float32
        ir_text = _build_module(build, [_vec_bf16])
        assert "arith.extf" in ir_text
        assert "vector.broadcast" in ir_text
        assert "arith.addf" in ir_text


# ===========================================================================
# D. Type conversion (.to())
# ===========================================================================

class TestToConversion:

    def test_same_type_noop(self):
        def build(a):
            ta = TensorSSA(a, 8, Float32)
            result = ta.to(Float32)
            assert result is ta
        _build_module(build)

    def test_float_to_float_truncf(self):
        def build(a):
            ta = TensorSSA(a, 8, Float32)
            _ = ta.to(BFloat16)
        ir_text = _build_module(build)
        assert "arith.truncf" in ir_text

    def test_float_to_float_extf(self):
        def build(a):
            ta = TensorSSA(a, 8, Float16)
            _ = ta.to(Float32)
        ir_text = _build_module(build, [_vec_f16])
        assert "arith.extf" in ir_text

    def test_float_to_int(self):
        def build(a):
            ta = TensorSSA(a, 8, Float32)
            _ = ta.to(Int32)
        ir_text = _build_module(build)
        assert "arith.fptosi" in ir_text

    def test_int_to_float(self):
        def build(a):
            ta = TensorSSA(a, 8, Int32)
            _ = ta.to(Float32)
        ir_text = _build_module(build, [_vec_i32])
        assert "arith.sitofp" in ir_text

    def test_to_preserves_shape(self):
        def build(a):
            ta = TensorSSA(a, 8, Float32)
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
            ta = TensorSSA(a, 8, Float32)
            _ = ta.reduce(ReductionOp.ADD)
        ir_text = _build_module(build)
        assert "vector.reduction <add>" in ir_text

    def test_reduce_max(self):
        def build(a):
            ta = TensorSSA(a, 8, Float32)
            _ = ta.reduce(ReductionOp.MAX)
        ir_text = _build_module(build)
        assert "vector.reduction <maxnumf>" in ir_text

    def test_reduce_min(self):
        def build(a):
            ta = TensorSSA(a, 8, Float32)
            _ = ta.reduce(ReductionOp.MIN)
        ir_text = _build_module(build)
        assert "vector.reduction <minimumf>" in ir_text

    def test_reduce_with_fastmath(self):
        def build(a):
            ta = TensorSSA(a, 8, Float32)
            fm = arith.FastMathFlags.fast
            _ = ta.reduce(ReductionOp.ADD, fastmath=fm)
        ir_text = _build_module(build)
        assert "fastmath" in ir_text.lower() or "fast" in ir_text


# ===========================================================================
# F. Element access
# ===========================================================================

class TestElementAccess:

    def test_getitem_int(self):
        def build(a):
            ta = TensorSSA(a, 8, Float32)
            elem = ta[0]
            assert isinstance(elem, Float32)
        ir_text = _build_module(build)
        assert "vector.extract" in ir_text

    def test_getitem_invalid_type(self):
        def build(a):
            ta = TensorSSA(a, 8, Float32)
            with pytest.raises(TypeError):
                ta["bad"]
        _build_module(build)


# ===========================================================================
# G. Vector ops
# ===========================================================================

class TestVectorOps:

    def test_bitcast(self):
        def build(a):
            ta = TensorSSA(a, 8, Float32)
            result = ta.bitcast(Uint32)
            assert result.shape == (8,)
            assert result.dtype is Uint32
        ir_text = _build_module(build)
        assert "vector.bitcast" in ir_text

    def test_bitcast_width_change(self):
        """f32 → f16 bitcast: 8 elements * 32 bits = 256 bits → 16 elements * 16 bits."""
        def build(a):
            ta = TensorSSA(a, 8, Float32)
            result = ta.bitcast(Float16)
            assert result.shape == (16,)
            assert result.dtype is Float16
        ir_text = _build_module(build)
        assert "vector.bitcast" in ir_text

    def test_shuffle(self):
        def build(a, b):
            ta = TensorSSA(a, 8, Float32)
            tb = TensorSSA(b, 8, Float32)
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
            ta = TensorSSA(a, 8, Float32)
            t = full_like(ta, 0.0)
            assert t.shape == ta.shape
            assert t.dtype == ta.dtype
        ir_text = _build_module(build)
        assert "vector.broadcast" in ir_text

    def test_zeros_like(self):
        def build(a):
            ta = TensorSSA(a, 8, Float32)
            t = zeros_like(ta)
            assert t.shape == ta.shape
            assert t.dtype == ta.dtype
        ir_text = _build_module(build)
        assert "vector.broadcast" in ir_text


# ===========================================================================
# I. fmath
# ===========================================================================

class TestFmath:

    def test_exp2_tensor(self):
        def build(a):
            ta = TensorSSA(a, 8, Float32)
            result = fmath.exp2(ta)
            assert isinstance(result, TensorSSA)
            assert result.dtype is Float32
            assert result.shape == (8,)
        ir_text = _build_module(build)
        assert "math.exp2" in ir_text

    def test_rsqrt_tensor(self):
        def build(a):
            ta = TensorSSA(a, 8, Float32)
            _ = fmath.rsqrt(ta)
        ir_text = _build_module(build)
        assert "math.rsqrt" in ir_text

    def test_fastmath_flag(self):
        def build(a):
            ta = TensorSSA(a, 8, Float32)
            _ = fmath.exp2(ta, fastmath=True)
        ir_text = _build_module(build)
        assert "fast" in ir_text

    def test_scalar_float(self):
        """fmath on scalar Float32 returns raw ir.Value."""
        def build(raw):
            x = Float32(raw)
            result = fmath.sqrt(x)
            assert not isinstance(result, TensorSSA)
        _build_module(build, [ir.F32Type.get])

    def test_int_scalar_raises(self):
        """fmath on integer Numeric should raise TypeError."""
        def build(raw):
            x = Int32(raw)
            with pytest.raises(TypeError, match="float"):
                fmath.exp2(x)
        _build_module(build, [lambda: ir.IntegerType.get_signless(32)])

    def test_mixed_types_raises(self):
        """Mixing TensorSSA and Numeric in atan2 should raise TypeError."""
        def build(a, raw_scalar):
            ta = TensorSSA(a, 8, Float32)
            x = Float32(raw_scalar)
            with pytest.raises(TypeError):
                fmath.atan2(ta, x)
        _build_module(build, [_vec_f32, ir.F32Type.get])

    def test_new_functions_exist(self):
        """Verify all newly added functions are accessible."""
        for name in ["erf", "acos", "asin", "atan", "atan2", "tan", "log10"]:
            assert hasattr(fmath, name), f"fmath.{name} missing"

    def test_erf_tensor(self):
        def build(a):
            ta = TensorSSA(a, 8, Float32)
            _ = fmath.erf(ta)
        ir_text = _build_module(build)
        assert "math.erf" in ir_text

    def test_atan2_tensor(self):
        def build(a, b):
            ta = TensorSSA(a, 8, Float32)
            tb = TensorSSA(b, 8, Float32)
            result = fmath.atan2(ta, tb)
            assert isinstance(result, TensorSSA)
        ir_text = _build_module(build, [_vec_f32, _vec_f32])
        assert "math.atan2" in ir_text


# ===========================================================================
# J. scf.for integration
# ===========================================================================

class TestScfIntegration:

    def test_extract_and_reconstruct(self):
        def build(a):
            ta = TensorSSA(a, 8, Float32)
            values = ta.__extract_mlir_values__()
            assert len(values) == 1
            reconstructed = ta.__new_from_mlir_values__(values)
            assert isinstance(reconstructed, TensorSSA)
            assert reconstructed.shape == ta.shape
            assert reconstructed.dtype is ta.dtype
        _build_module(build)

    def test_hash(self):
        """TensorSSA must be hashable since __eq__ is overridden."""
        def build(a):
            ta = TensorSSA(a, 8, Float32)
            h = hash(ta)
            assert isinstance(h, int)
        _build_module(build)
