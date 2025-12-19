#!/usr/bin/env python3
"""
Test rocir.print and rocir.printf functionality.
Following the layout print notebook example.

This test demonstrates the difference between static (compile-time) and
dynamic (runtime) printing.
"""

import rocdsl
from rocdsl.dialects.ext import rocir
from rocdsl.dialects.ext.arith import Index


def test_print_basic():
    """Test that rocir.print is available and works as Python's print."""
    # rocir.print should just be Python's built-in print
    assert rocdsl.rocir.print is print


class _PrintfModule(rocir.MlirModule):
    @rocir.jit
    def printf_one(self: rocir.T.i64):
        x = Index(42)
        rocir.printf(">?? {}", x)
        return []

    @rocir.jit
    def printf_two(self: rocir.T.i64):
        a = Index(10)
        b = Index(20)
        rocir.printf("a: {}, b: {}", a, b)
        return []

    @rocir.jit
    def print_vs_printf(self: rocir.T.i64):
        a = Index(8)
        rocir.printf(">?? {}", a)
        c2 = Index(2)
        rocir.printf(">?? {}", c2)
        return []

    @rocir.jit
    def printf_layout(self: rocir.T.i64):
        dim0 = Index(9)
        dim1 = Index(4)
        dim2 = Index(8)
        shape = rocir.make_shape(dim0, (dim1, dim2))
        rocir.printf("Shape dims: {} x ({} x {})", dim0, dim1, dim2)
        return []


def test_printf_ir_generation():
    """Test that rocir.printf generates the correct MLIR operations."""
    ir_str = str(_PrintfModule().module)
    assert "gpu.printf" in ir_str


def test_printf_with_multiple_args():
    """Test printf with multiple arguments."""
    ir_str = str(_PrintfModule().module)
    assert "gpu.printf" in ir_str
    # Check that format string is correct
    assert "a: {}, b: {}" in ir_str or "a: %lld, b: %lld" in ir_str


def test_print_vs_printf_concept():
    """Conceptual test showing the difference between print and printf.
    
    This demonstrates the key difference highlighted in the reference notebook:
    - rocir.print: Shows static/compile-time values
    - rocir.printf: Shows dynamic/runtime values
    """
    ir_str = str(_PrintfModule().module)
    assert "gpu.printf" in ir_str


def test_printf_with_layout_types():
    """Test printf with rocir layout types."""
    ir_str = str(_PrintfModule().module)
    assert "rocir.make_shape" in ir_str
    assert "gpu.printf" in ir_str
