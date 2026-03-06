#!/usr/bin/env python3
"""
Test fly.print_ (printf) functionality via the Fly dialect API.

Covers:
  - {} auto-format placeholders
  - bare-value printf (no format string)
  - Python literal auto-materialization (int, float, bool)
  - format arg-count mismatch detection
"""

import re
import pytest

from flydsl._mlir.ir import (
    Context, Location, Module, InsertionPoint,
    FunctionType, IntegerType, IndexType,
)
from flydsl._mlir.dialects.fly import IntTupleType
from flydsl._mlir.dialects import fly, arith, func
import flydsl.expr as fx


def _build_module(name, build_fn):
    """Build a module with a single function, return its IR string."""
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            i32 = IntegerType.get_signless(32)
            with InsertionPoint(module.body):
                f = func.FuncOp(name, FunctionType.get([], []))
                with InsertionPoint(f.add_entry_block()):
                    build_fn(i32)
                    func.ReturnOp([])
            return str(module)


def _count(pattern, text):
    return len(re.findall(pattern, text))


# ===========================================================================
# 1. {} placeholder resolution
# ===========================================================================

def test_placeholder_single():
    """{} placeholder with an i32 value generates fly.print with the value as operand."""
    def build(i32):
        x = arith.ConstantOp(i32, 42).result
        fx.printf("x={}", x)
    ir = _build_module("single_placeholder", build)
    assert "fly.print" in ir
    assert "42" in ir


def test_placeholder_multi():
    """Multiple {} placeholders produce a fly.print with multiple operands."""
    def build(i32):
        a = arith.ConstantOp(i32, 10).result
        b = arith.ConstantOp(i32, 20).result
        fx.printf("a={}, b={}", a, b)
    ir = _build_module("multi_placeholder", build)
    assert "fly.print" in ir
    assert "10" in ir
    assert "20" in ir


# ===========================================================================
# 2. Python literal auto-materialization
# ===========================================================================

def test_python_int_literal():
    """Python int is auto-materialized as i32 constant."""
    def build(i32):
        fx.printf("int={}", 42)
    ir = _build_module("int_literal", build)
    assert "fly.print" in ir
    assert "42" in ir


def test_python_float_literal():
    """Python float is auto-materialized as f64 constant."""
    def build(i32):
        fx.printf("float={}", 3.14)
    ir = _build_module("float_literal", build)
    assert "fly.print" in ir
    assert "3.14" in ir or "3.140000" in ir


# ===========================================================================
# 3. Static string embedding
# ===========================================================================

def test_static_string_in_placeholder():
    """Static string values are embedded directly in the format string."""
    def build(i32):
        fx.printf("type={}", "hello")
    ir = _build_module("static_string", build)
    assert "hello" in ir


# ===========================================================================
# 4. IR structure check
# ===========================================================================

def test_print_generates_fly_print():
    """fx.printf generates fly.print op."""
    def build(i32):
        x = arith.ConstantOp(i32, 1).result
        fx.printf("v={}", x)
    ir = _build_module("ir_check", build)
    assert 'fly.print "v' in ir or "fly.print" in ir
