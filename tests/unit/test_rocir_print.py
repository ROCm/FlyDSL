#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Test fx.printf: IR generation, GPU lowering, CPU lowering."""

from flydsl._mlir.ir import (
    Context, Location, Module, InsertionPoint,
    FunctionType, IntegerType, F32Type,
)
from flydsl._mlir.dialects import arith, func, gpu
from flydsl._mlir.passmanager import PassManager
from flydsl.compiler.kernel_function import create_gpu_module, get_gpu_module_body, create_gpu_func
import flydsl.expr as fx
import pytest

pytestmark = [pytest.mark.l1a_compile_no_target_dialect]


def _build_cpu_module(build_fn):
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            i32 = IntegerType.get_signless(32)
            with InsertionPoint(module.body):
                f = func.FuncOp("test", FunctionType.get([], []))
                with InsertionPoint(f.add_entry_block()):
                    build_fn(i32)
                    func.ReturnOp([])
            return str(module)


def _build_gpu_module(build_fn):
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            with InsertionPoint(module.body):
                gpu_mod = create_gpu_module("mod")
                with InsertionPoint(get_gpu_module_body(gpu_mod)):
                    kern = create_gpu_func("kern", FunctionType.get([], []))
                    with InsertionPoint(kern.add_entry_block()):
                        build_fn()
                        gpu.ReturnOp([])
            return str(module)


def _lower(ir_text):
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.parse(ir_text)
            PassManager.parse("builtin.module(fly-layout-lowering)").run(module.operation)
            return str(module)


# -- IR generation --

def test_printf_generates_fly_print():
    def build(i32):
        x = arith.ConstantOp(i32, 42).result
        fx.printf("val={}", x)
    ir = _build_cpu_module(build)
    assert "fly.print" in ir


def test_printf_multi_placeholder():
    def build(i32):
        a = arith.ConstantOp(i32, 10).result
        b = arith.ConstantOp(i32, 20).result
        fx.printf("a={}, b={}", a, b)
    ir = _build_cpu_module(build)
    assert "fly.print" in ir
    assert "10" in ir and "20" in ir


def test_printf_python_literals():
    def build(i32):
        fx.printf("int={}, float={}, str={}", 42, 3.14, "hello")
    ir = _build_cpu_module(build)
    assert "fly.print" in ir
    assert "42" in ir
    assert "hello" in ir


def test_printf_bare_value():
    def build(i32):
        x = arith.ConstantOp(i32, 99).result
        fx.printf(x)
    ir = _build_cpu_module(build)
    assert "fly.print" in ir


# -- GPU lowering: fly.print → gpu.printf --

def test_gpu_printf_lowering():
    def build():
        a = arith.ConstantOp(IntegerType.get_signless(32), 1).result
        b = arith.ConstantOp(F32Type.get(), 2.0).result
        fx.printf("a={}, b={}", a, b)
    ir = _lower(_build_gpu_module(build))
    assert "gpu.printf" in ir
    assert "fly.print" not in ir
    assert "%d" in ir and "%f" in ir


# -- CPU lowering: fly.print → vector.print --

def test_cpu_printf_lowering():
    def build(i32):
        x = arith.ConstantOp(i32, 42).result
        fx.printf("val={}", x)
    ir = _lower(_build_cpu_module(build))
    assert "vector.print" in ir
    assert "fly.print" not in ir
