# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Regression tests for @flyc.jit/@flyc.kernel defined as class methods."""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler import jit_function

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


class ClassBoundProgram:
    @flyc.kernel
    def kernel(self, value: fx.Int32, scale: fx.Constexpr[int]):
        fx.printf("class-bound value={} scale={}", value, scale)

    @flyc.jit
    def run(self, value: fx.Int32, scale: fx.Constexpr[int], stream: fx.Stream = fx.Stream(None)):
        self.kernel(value, scale).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    @flyc.jit
    def __call__(self, value: fx.Int32, scale: fx.Constexpr[int], stream: fx.Stream = fx.Stream(None)):
        self.kernel(value, scale).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


def reset_jit(jit_fn):
    jit_fn._call_state_cache.clear()
    jit_fn._mem_cache.clear()
    jit_fn._last_compiled = None
    jit_fn.manager_key = None
    jit_fn.cache_manager = None
    jit_fn._target = None

    jit_fn.sig = None
    jit_fn.has_self_param = False


@pytest.fixture(autouse=True)
def frontend_only_compile(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-flydsl-key")

    def compile_noop(cls, module, *, arch: str = "", func_name: str = ""):
        return module

    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(compile_noop))
    reset_jit(ClassBoundProgram.run)
    reset_jit(ClassBoundProgram.__call__)


def last_compile(jit_fn):
    last_compiled = jit_fn._last_compiled
    assert last_compiled is not None
    return last_compiled


def test_class_defined_jit_method_binds_self_and_launches_kernel_method():
    program = ClassBoundProgram()

    program.run(7, 3)

    cache_key, artifact = last_compile(ClassBoundProgram.run)
    assert cache_key[0] == ("_self_type_", ClassBoundProgram)
    assert "func.func @run" in artifact.source_ir
    assert "gpu.func @kernel_0" in artifact.source_ir
    assert "gpu.launch_func" in artifact.source_ir
    assert "@kernels::@kernel_0" in artifact.source_ir


def test_class_defined_jit_call_special_method_binds_self():
    program = ClassBoundProgram()

    program(11, 5)

    cache_key, artifact = last_compile(ClassBoundProgram.__call__)
    assert cache_key[0] == ("_self_type_", ClassBoundProgram)
    assert "func.func @__call__" in artifact.source_ir
    assert "gpu.func @kernel_0" in artifact.source_ir
    assert "gpu.launch_func" in artifact.source_ir
    assert "@kernels::@kernel_0" in artifact.source_ir
