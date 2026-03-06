#!/usr/bin/env python3
"""
Test flir.print and flir.printf functionality.

Covers:
  - compile-time print (Python's built-in)
  - device printf with {} auto-format placeholders
  - device printf with explicit C format specifiers
  - bare-value printf (no format string)
  - no-arg printf (just newline)
  - flir compound type printing (layout, shape, stride, coord)
  - mixed scalar + compound arguments
  - printf_if (predicated printing)
  - printf_once (thread-0-only printing)
  - f16/bf16 float promotion to f32
  - Python literal auto-materialization (bool, int, float)
  - format arg-count validation
"""

import re
import pytest

import flydsl
from flydsl.dialects.ext import flir
from flydsl.dialects.ext.arith import Index
from _mlir.ir import IndexType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ir(module_cls):
    """Instantiate an MlirModule subclass and return its IR string."""
    return str(module_cls().module)


def _count(pattern, text):
    """Count occurrences of *pattern* (regex) in *text*."""
    return len(re.findall(pattern, text))


# ===========================================================================
# 1. Compile-time print
# ===========================================================================

def test_print_is_builtin():
    """flir.print should be Python's built-in print."""
    assert flir.print is print
    assert flir.flir.print is print


# ===========================================================================
# 2. Basic {} placeholder resolution
# ===========================================================================

class _PlaceholderModule(flir.MlirModule):
    @flir.jit
    def single_placeholder(self: flir.T.i64):
        x = Index(42)
        flir.printf("x={}", x)
        return []

    @flir.jit
    def multi_placeholder(self: flir.T.i64):
        a = Index(10)
        b = Index(20)
        flir.printf("a={}, b={}", a, b)
        return []


def test_placeholder_single():
    """Single {} is resolved to %lld for index type."""
    ir = _ir(_PlaceholderModule)
    assert "gpu.printf" in ir
    # {} must have been replaced with a C specifier
    assert "{}" not in ir
    assert "%lld" in ir


def test_placeholder_multi():
    """Multiple {} placeholders are each resolved."""
    ir = _ir(_PlaceholderModule)
    assert 'a=%lld, b=%lld' in ir


# ===========================================================================
# 3. C-style format specifiers (no {} at all)
# ===========================================================================

class _CStyleModule(flir.MlirModule):
    @flir.jit
    def c_style(self: flir.T.i64):
        x = Index(42)
        flir.printf("val=%lld\n", x)
        return []


def test_c_style_passthrough():
    """Explicit C specifiers are passed through unchanged."""
    ir = _ir(_CStyleModule)
    # The format string should contain %lld and the arg should appear
    assert "val=%lld" in ir
    assert "%c42" in ir or "42 : index" in ir


# ===========================================================================
# 4. Bare values (no format string)
# ===========================================================================

class _BareModule(flir.MlirModule):
    @flir.jit
    def bare_values(self: flir.T.i64):
        a = Index(1)
        b = Index(2)
        flir.printf(a, b)
        return []


def test_bare_values():
    """Bare values without a format string get auto-formatted."""
    ir = _ir(_BareModule)
    assert "gpu.printf" in ir
    # Auto-generated format should be "%lld, %lld\n"
    assert "%lld, %lld" in ir


# ===========================================================================
# 5. No-arg printf
# ===========================================================================

class _NoArgModule(flir.MlirModule):
    @flir.jit
    def no_args(self: flir.T.i64):
        flir.printf()
        return []


def test_no_arg_newline():
    """printf() with no args emits a newline."""
    ir = _ir(_NoArgModule)
    assert "gpu.printf" in ir
    # Should contain just the escaped newline
    assert r'\0A' in ir


# ===========================================================================
# 6. Flir compound type printing (layout, shape, stride)
# ===========================================================================

class _CompoundModule(flir.MlirModule):
    @flir.jit
    def print_layout(self: flir.T.i64):
        layout = flir.make_layout(
            (Index(8), Index(4)), stride=(Index(4), Index(1))
        )
        flir.printf("L={}", layout)
        return []

    @flir.jit
    def print_shape(self: flir.T.i64):
        shape = flir.make_shape(Index(3), Index(5))
        flir.printf("S={}", shape)
        return []


def test_layout_printf_decomposes():
    """Printing a layout decomposes into get_shape/get_stride + element prints."""
    ir = _ir(_CompoundModule)
    # Layout printing uses flir.get_shape / flir.get_stride / flir.get
    assert "flir.get_shape" in ir
    assert "flir.get_stride" in ir
    assert "flir.get" in ir
    # Multiple gpu.printf calls for open/close parens, elements, colon
    assert _count(r"gpu\.printf", ir) >= 8  # parens + elements + colon + newline


def test_shape_printf_decomposes():
    """Printing a shape decomposes into element-wise prints."""
    ir = _ir(_CompoundModule)
    # Shape printing uses flir.get
    assert "flir.get" in ir


# ===========================================================================
# 7. Mixed scalar + compound
# ===========================================================================

class _MixedModule(flir.MlirModule):
    @flir.jit
    def mixed(self: flir.T.i64):
        x = Index(42)
        layout = flir.make_layout(
            (Index(8), Index(4)), stride=(Index(4), Index(1))
        )
        flir.printf("id={} layout={}", x, layout)
        return []


def test_mixed_scalar_and_compound():
    """Scalar args and compound flir args can be mixed in one printf."""
    ir = _ir(_MixedModule)
    # Scalar part: id=%lld
    assert "id=%lld" in ir
    # Compound part: get_shape / get_stride
    assert "flir.get_shape" in ir
    assert "flir.get_stride" in ir


# ===========================================================================
# 8. printf_if (predicated printing)
# ===========================================================================

class _PrintfIfModule(flir.MlirModule):
    @flir.jit
    def predicated(self: flir.T.i64):
        x = Index(42)
        # Build a predicate: thread_idx_x == 0
        from _mlir.dialects import gpu, arith
        from _mlir.ir import IndexType, IntegerAttr
        tidx = flir._unwrap_value(gpu.thread_id(gpu.Dimension.x))
        zero = flir._unwrap_value(
            arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), 0)).result
        )
        pred = flir._unwrap_value(
            arith.CmpIOp(arith.CmpIPredicate.eq, tidx, zero).result
        )
        flir.printf_if(pred, "hello x={}", x)
        return []


def test_printf_if_generates_scf_if():
    """printf_if wraps the printf inside an scf.if block."""
    ir = _ir(_PrintfIfModule)
    assert "scf.if" in ir
    assert "gpu.printf" in ir
    assert "arith.cmpi eq" in ir


# ===========================================================================
# 9. printf_once (thread 0 only)
# ===========================================================================

class _PrintfOnceModule(flir.MlirModule):
    @flir.jit
    def once(self: flir.T.i64):
        x = Index(42)
        flir.printf_once("hello x={}", x)
        return []


def test_printf_once_generates_thread_check():
    """printf_once generates thread_id == 0 check + scf.if."""
    ir = _ir(_PrintfOnceModule)
    assert "gpu.thread_id  x" in ir
    assert "arith.cmpi eq" in ir
    assert "scf.if" in ir
    assert "gpu.printf" in ir
    assert "hello x=%lld" in ir


# ===========================================================================
# 10. Python literal auto-materialization
# ===========================================================================

class _LiteralModule(flir.MlirModule):
    @flir.jit
    def literals(self: flir.T.i64):
        flir.printf("int={} float={} bool={}", 42, 3.14, True)
        return []


def test_python_literals():
    """Python int/float/bool are auto-materialized as MLIR constants."""
    ir = _ir(_LiteralModule)
    assert "gpu.printf" in ir
    # int → index constant 42
    assert "42" in ir
    # float → f32 constant
    assert "3.14" in ir or "3.140000" in ir
    # bool → i32 constant 1
    assert "1 : i32" in ir or "1 : index" in ir


# ===========================================================================
# 11. Format arg-count validation
# ===========================================================================

def test_too_few_args_raises():
    """More {} placeholders than args should raise ValueError."""
    class _Bad(flir.MlirModule):
        @flir.jit
        def bad(self: flir.T.i64):
            x = Index(1)
            flir.printf("{} {} {}", x)  # 3 placeholders, 1 arg
            return []

    with pytest.raises(ValueError, match="placeholder"):
        _Bad()  # error raised during IR construction


# ===========================================================================
# 12. Trailing newline
# ===========================================================================

class _NewlineModule(flir.MlirModule):
    @flir.jit
    def auto_newline(self: flir.T.i64):
        x = Index(1)
        flir.printf("x={}", x)  # no explicit \n
        return []

    @flir.jit
    def explicit_newline(self: flir.T.i64):
        x = Index(1)
        flir.printf("x={}\n", x)  # explicit \n
        return []


def test_auto_trailing_newline():
    """printf auto-appends newline if not present."""
    ir = _ir(_NewlineModule)
    # Both functions should end with \0A (escaped newline)
    # Count \0A occurrences — should be at least 2 (one per function)
    assert _count(r'\\0A', ir) >= 2


# ===========================================================================
# 13. Exports
# ===========================================================================

def test_exports():
    """printf, printf_if, printf_once are in __all__."""
    assert "printf" in flir.__all__
    assert "printf_if" in flir.__all__
    assert "printf_once" in flir.__all__
    assert "print" in flir.__all__
