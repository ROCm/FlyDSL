"""Test operator overloading for elegant Pythonic syntax (Flyx style)."""

import pytest
from _mlir.ir import Context, Location, Module, InsertionPoint

try:
    from rocdsl.dialects.ext import arith, rocir
except ImportError:
    pytest.skip("RocDSL dialect not available", allow_module_level=True)


class _ArithOps(rocir.MlirModule):
    @rocir.jit
    def test_arithmetic(
        self: rocir.T.i64,
    ):
        a = arith.index(10)
        b = arith.index(3)
        c = a + b
        return c.value

    @rocir.jit
    def test_mixed(
        self: rocir.T.i64,
    ):
        a = arith.index(100)
        b = a + 50
        return b.value

    @rocir.jit
    def test_comparison(
        self: rocir.T.i64,
    ):
        a = arith.index(10)
        b = arith.index(5)
        lt = a < b
        return lt.value

    @rocir.jit
    def test_float_ops(
        self: rocir.T.i64,
    ):
        a = arith.f32(3.14)
        b = arith.f32(2.0)
        c = a + b
        return c.value

    @rocir.jit
    def test_chained(
        self: rocir.T.i64,
    ):
        a = arith.index(10)
        b = arith.index(5)
        c = arith.index(2)
        result = (a + b) * c - 3
        return result.value


def test_arithmetic_operators(ctx):
    m = _ArithOps()
    s = str(m.module)
    assert "func.func" in s

    


def test_mixed_operators(ctx):
    m = _ArithOps()
    s = str(m.module)
    assert "arith.addi" in s or "arith.addf" in s

    


def test_comparison_operators(ctx):
    m = _ArithOps()
    s = str(m.module)
    assert "arith.cmpi" in s or "arith.cmpf" in s

    


def test_float_operators(ctx):
    m = _ArithOps()
    s = str(m.module)
    assert "arith.addf" in s

    


def test_chained_operations(ctx):
    m = _ArithOps()
    s = str(m.module)
    assert "arith.muli" in s or "arith.mulf" in s

    


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
