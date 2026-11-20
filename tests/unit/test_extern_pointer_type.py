# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""ffi can declare LLVM pointer arguments and inttoptr an integer address."""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler import jit_function
from flydsl.expr.extern import ffi

pytestmark = [pytest.mark.l1a_compile_no_target_dialect]

_read_ptr = ffi("extern_read_ptr", ["ptr", "int32"], "ptr")
_write_i32 = ffi("extern_write_i32", ["ptr<1>", "int32", "int32"], "void")


@pytest.fixture
def capture_frontend(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-flydsl-key")

    captured = {}

    def compile_capture(cls, module, **_kwargs):
        captured["asm"] = module.operation.get_asm(enable_debug_info=False)
        return module

    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(compile_capture))
    return captured


def test_ffi_pointer_types_declare_and_inttoptr(capture_frontend):
    @flyc.kernel
    def extern_ptr_call(addr: fx.Int64, value: fx.Int32, index: fx.Int32):
        remote = _read_ptr(addr, index)
        _read_ptr(remote, index)
        _write_i32(addr, value, index)

    @flyc.jit
    def launch(addr: fx.Int64, value: fx.Int32, index: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        extern_ptr_call(addr, value, index).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    launch(0, 1, 0)
    asm = capture_frontend["asm"]
    assert 'llvm.func @extern_read_ptr(!llvm.ptr, i32) -> !llvm.ptr attributes {sym_visibility = "private"}' in asm
    assert 'llvm.func @extern_write_i32(!llvm.ptr<1>, i32, i32) attributes {sym_visibility = "private"}' in asm
    assert asm.count("llvm.inttoptr") == 2
    assert "llvm.call @extern_read_ptr" in asm
    assert "llvm.call @extern_write_i32" in asm


def test_ffi_rejects_pointer_address_space_mismatch(capture_frontend):
    @flyc.kernel
    def extern_ptr_mismatch(addr: fx.Int64, value: fx.Int32, index: fx.Int32):
        remote = _read_ptr(addr, index)
        _write_i32(remote, value, index)

    @flyc.jit
    def launch(addr: fx.Int64, value: fx.Int32, index: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        extern_ptr_mismatch(addr, value, index).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    with pytest.raises(TypeError, match=r"ptr<1>"):
        launch(0, 1, 0)


def test_ffi_rejects_python_int_pointer(capture_frontend):
    @flyc.kernel
    def extern_ptr_const(value: fx.Int32, index: fx.Int32):
        _write_i32(0, value, index)

    @flyc.jit
    def launch(value: fx.Int32, index: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        extern_ptr_const(value, index).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    with pytest.raises(TypeError, match="Python int"):
        launch(1, 0)


def test_ffi_rejects_pointer_value_for_scalar_argument(capture_frontend):
    take_i64 = ffi("extern_take_i64", ["int64"], "void")

    @flyc.kernel
    def extern_ptr_as_int(addr: fx.Int64, index: fx.Int32):
        remote = _read_ptr(addr, index)
        take_i64(remote)

    @flyc.jit
    def launch(addr: fx.Int64, index: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        extern_ptr_as_int(addr, index).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    with pytest.raises(TypeError, match=r"expects i64, got !llvm.ptr"):
        launch(0, 0)


def test_ffi_rejects_non_i64_address(capture_frontend):
    @flyc.kernel
    def extern_ptr_i32(addr: fx.Int32, index: fx.Int32):
        _read_ptr(addr, index)

    @flyc.jit
    def launch(addr: fx.Int32, index: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        extern_ptr_i32(addr, index).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    with pytest.raises(TypeError, match="i64"):
        launch(0, 0)


def test_ffi_rejects_redeclare_with_different_type(capture_frontend):
    other = ffi("extern_read_ptr", ["ptr<1>", "int32"], "ptr<1>")

    @flyc.kernel
    def extern_ptr_redeclare(addr: fx.Int64, index: fx.Int32):
        _read_ptr(addr, index)
        other(addr, index)

    @flyc.jit
    def launch(addr: fx.Int64, index: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        extern_ptr_redeclare(addr, index).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    with pytest.raises(TypeError, match="already declared"):
        launch(0, 0)


def test_ffi_rejects_bad_pointer_spelling(capture_frontend):
    bad = ffi("extern_bad_ptr_symbol", ["ptr<>"], "void")

    @flyc.kernel
    def extern_bad_ptr(addr: fx.Int64):
        bad(addr)

    @flyc.jit
    def launch(addr: fx.Int64, stream: fx.Stream = fx.Stream(None)):
        extern_bad_ptr(addr).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    with pytest.raises(ValueError, match="address space"):
        launch(0)
