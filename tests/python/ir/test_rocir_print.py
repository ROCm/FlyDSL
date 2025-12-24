#!/usr/bin/env python3
"""
Test flir.print and flir.printf functionality.
Following the layout print notebook example.

This test demonstrates the difference between static (compile-time) and
dynamic (runtime) printing.
"""

import pyflir
from pyflir.dialects.ext import flir
from pyflir.dialects.ext.arith import Index


def test_print_basic():
    """Test that flir.print is available and works as Python's print."""
    # flir.print should just be Python's built-in print
    assert flir.flir.print is print


class _PrintfModule(flir.MlirModule):
    @flir.jit
    def printf_one(self: flir.T.i64):
        x = Index(42)
        flir.printf(">?? {}", x)
        return []

    @flir.jit
    def printf_two(self: flir.T.i64):
        a = Index(10)
        b = Index(20)
        flir.printf("a: {}, b: {}", a, b)
        return []

    @flir.jit
    def print_vs_printf(self: flir.T.i64):
        a = Index(8)
        flir.printf(">?? {}", a)
        c2 = Index(2)
        flir.printf(">?? {}", c2)
        return []

    @flir.jit
    def printf_layout(self: flir.T.i64):
        dim0 = Index(9)
        dim1 = Index(4)
        dim2 = Index(8)
        shape = flir.make_shape(dim0, (dim1, dim2))
        flir.printf("Shape dims: {} x ({} x {})", dim0, dim1, dim2)
        return []


def test_printf_ir_generation():
    """Test that flir.printf generates the correct MLIR operations."""
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
    - flir.print: Shows static/compile-time values
    - flir.printf: Shows dynamic/runtime values
    """
    ir_str = str(_PrintfModule().module)
    assert "gpu.printf" in ir_str


def test_printf_with_layout_types():
    """Test printf with flir layout types."""
    ir_str = str(_PrintfModule().module)
    assert "flir.make_shape" in ir_str
    assert "gpu.printf" in ir_str
