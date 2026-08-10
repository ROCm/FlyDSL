#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import math

import pytest

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import func


def _build_module(build_fn, arg_types=()):
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with ir.Location.unknown(ctx):
            types = [ty() if callable(ty) else ty for ty in arg_types]
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                ftype = ir.FunctionType.get(types, [])
                f = func.FuncOp("test", ftype)
                with ir.InsertionPoint(f.add_entry_block()):
                    build_fn(*f.entry_block.arguments)
                    func.ReturnOp([])
            module.operation.verify()
            return str(module)


def _static(call):
    with ir.Context(), ir.Location.unknown():
        return call()


def _vector_type(size, element_type):
    return lambda: ir.VectorType.get([size], element_type())


@pytest.mark.l0_backend_agnostic
def test_typed_arithmetic_exports():
    assert fx.max is fx.arith.max
    assert fx.min is fx.arith.min
    assert fx.ceildiv is fx.arith.ceildiv
    assert fx.ceildiv is not fx.ceil_div


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize(
    ("fn", "op_name"),
    [
        (fx.max, "arith.maximumf"),
        (fx.min, "arith.minimumf"),
    ],
)
def test_float_extreme_dispatch(fn, op_name):
    def build(lhs, rhs):
        result = fn(fx.Float32(lhs), fx.Float32(rhs))
        assert isinstance(result, fx.Float32)

    text = _build_module(build, [ir.F32Type.get, ir.F32Type.get])
    assert op_name in text


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize(
    ("fn", "dtype", "op_name"),
    [
        (fx.max, fx.Int32, "arith.maxsi"),
        (fx.min, fx.Int32, "arith.minsi"),
        (fx.max, fx.Uint32, "arith.maxui"),
        (fx.min, fx.Uint32, "arith.minui"),
    ],
)
def test_integer_extreme_dispatch(fn, dtype, op_name):
    def build(lhs, rhs):
        result = fn(dtype(lhs), dtype(rhs))
        assert isinstance(result, dtype)

    text = _build_module(
        build,
        [
            lambda: ir.IntegerType.get_signless(32),
            lambda: ir.IntegerType.get_signless(32),
        ],
    )
    assert op_name in text


@pytest.mark.l0_backend_agnostic
def test_extreme_resolves_common_type_before_reduction():
    def build(a, b, c):
        result = fx.max(fx.Int32(a), fx.Int64(b), fx.Int32(c))
        assert isinstance(result, fx.Int64)

    text = _build_module(
        build,
        [
            lambda: ir.IntegerType.get_signless(32),
            lambda: ir.IntegerType.get_signless(64),
            lambda: ir.IntegerType.get_signless(32),
        ],
    )
    assert text.count("arith.maxsi") == 2
    assert "i64" in text


@pytest.mark.l0_backend_agnostic
def test_extreme_flattens_nested_lists_and_tuples():
    def build(a, b, c):
        result = fx.min([fx.Float32(a), (fx.Float32(b), [fx.Float32(c)])])
        assert isinstance(result, fx.Float32)

    text = _build_module(build, [ir.F32Type.get, ir.F32Type.get, ir.F32Type.get])
    assert text.count("arith.minimumf") == 2


@pytest.mark.l0_backend_agnostic
def test_extreme_single_operand_returns_promoted_dsl_value():
    result = _static(lambda: fx.max(True))
    assert isinstance(result, fx.Int32)
    assert int(result) == 1


@pytest.mark.l0_backend_agnostic
def test_extreme_empty_input_rejected():
    with pytest.raises(ValueError, match="at least one operand"):
        _static(fx.max)


@pytest.mark.l0_backend_agnostic
def test_extreme_static_nan_propagates_in_both_positions():
    lhs_nan = _static(lambda: fx.max(fx.Float32(float("nan")), fx.Float32(1.0)))
    rhs_nan = _static(lambda: fx.max(fx.Float32(1.0), fx.Float32(float("nan"))))
    assert math.isnan(lhs_nan.value)
    assert math.isnan(rhs_nan.value)


@pytest.mark.l0_backend_agnostic
def test_extreme_static_signed_zero_matches_maximum_minimum():
    maximum = _static(lambda: fx.max(fx.Float32(-0.0), fx.Float32(0.0)))
    minimum = _static(lambda: fx.min(fx.Float32(-0.0), fx.Float32(0.0)))
    assert math.copysign(1.0, maximum.value) == 1.0
    assert math.copysign(1.0, minimum.value) == -1.0


@pytest.mark.l0_backend_agnostic
def test_extreme_rejects_index_and_narrow_storage_float():
    with pytest.raises(TypeError, match="does not support Index"):
        _static(lambda: fx.max(fx.Index(1), fx.Index(2)))
    with pytest.raises(TypeError, match="narrow storage float"):
        _static(lambda: fx.max(fx.Float8E5M2(1.0), fx.Float8E5M2(2.0)))


@pytest.mark.l0_backend_agnostic
def test_extreme_vector_broadcast_and_type_promotion():
    def build(raw_vector):
        vector = fx.Vector(raw_vector, (2, 2), fx.Int32)
        result = fx.max(vector, fx.Int64(7))
        assert isinstance(result, fx.Vector)
        assert result.shape == (2, 2)
        assert result.dtype is fx.Int64

    text = _build_module(build, [_vector_type(4, lambda: ir.IntegerType.get_signless(32))])
    assert "arith.maxsi" in text
    assert "vector<4xi64>" in text


@pytest.mark.l0_backend_agnostic
def test_extreme_explicit_fastmath_is_preserved():
    def build(lhs, rhs):
        fx.max(fx.Float32(lhs), fx.Float32(rhs), fastmath="contract")

    text = _build_module(build, [ir.F32Type.get, ir.F32Type.get])
    line = next(line for line in text.splitlines() if "arith.maximumf" in line)
    assert "fastmath<contract>" in line


@pytest.mark.l0_backend_agnostic
def test_numeric_minimumf_compatibility_proxy():
    def build(lhs, rhs):
        result = fx.Float32(lhs).minimumf(fx.Float32(rhs))
        assert isinstance(result, fx.Float32)

    text = _build_module(build, [ir.F32Type.get, ir.F32Type.get])
    assert "arith.minimumf" in text


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize(
    ("dtype", "op_name"),
    [
        (fx.Int32, "arith.ceildivsi"),
        (fx.Uint32, "arith.ceildivui"),
    ],
)
def test_ceildiv_dispatch(dtype, op_name):
    def build(lhs, rhs):
        result = fx.ceildiv(dtype(lhs), dtype(rhs))
        assert isinstance(result, dtype)

    text = _build_module(
        build,
        [
            lambda: ir.IntegerType.get_signless(32),
            lambda: ir.IntegerType.get_signless(32),
        ],
    )
    assert op_name in text
    assert "arith.addi" not in text
    assert "arith.subi" not in text


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize(
    ("lhs", "rhs", "expected"),
    [
        (7, 2, 4),
        (7, -2, -3),
        (-7, 2, -3),
        (-7, -2, 4),
        (0, 3, 0),
    ],
)
def test_signed_ceildiv_static(lhs, rhs, expected):
    result = _static(lambda: fx.ceildiv(fx.Int32(lhs), fx.Int32(rhs)))
    assert isinstance(result, fx.Int32)
    assert int(result) == expected


@pytest.mark.l0_backend_agnostic
def test_unsigned_ceildiv_static_is_overflow_safe():
    result = _static(lambda: fx.ceildiv(fx.Uint32(0xFFFFFFFF), fx.Uint32(2)))
    assert isinstance(result, fx.Uint32)
    assert int(result) == 0x80000000


@pytest.mark.l0_backend_agnostic
def test_ceildiv_static_undefined_cases_raise():
    with pytest.raises(ZeroDivisionError):
        _static(lambda: fx.ceildiv(fx.Uint32(1), fx.Uint32(0)))
    with pytest.raises(OverflowError):
        _static(lambda: fx.ceildiv(fx.Int32(-(1 << 31)), fx.Int32(-1)))


@pytest.mark.l0_backend_agnostic
def test_ceildiv_vector_broadcast_preserves_shape_and_unsignedness():
    def build(raw_vector, divisor):
        vector = fx.Vector(raw_vector, (2, 2), fx.Uint32)
        result = fx.ceildiv(vector, fx.Uint32(divisor))
        assert isinstance(result, fx.Vector)
        assert result.shape == (2, 2)
        assert result.dtype is fx.Uint32

    text = _build_module(
        build,
        [
            _vector_type(4, lambda: ir.IntegerType.get_signless(32)),
            lambda: ir.IntegerType.get_signless(32),
        ],
    )
    assert "arith.ceildivui" in text


@pytest.mark.l0_backend_agnostic
def test_ceildiv_rejects_non_integer_boolean_and_index_inputs():
    with pytest.raises(TypeError, match="requires integer operands"):
        _static(lambda: fx.ceildiv(fx.Float32(3.0), fx.Float32(2.0)))
    with pytest.raises(TypeError, match="does not support Boolean"):
        _static(lambda: fx.ceildiv(fx.Boolean(True), fx.Boolean(True)))
    with pytest.raises(TypeError, match="does not support Index"):
        _static(lambda: fx.ceildiv(fx.Index(3), fx.Index(2)))
