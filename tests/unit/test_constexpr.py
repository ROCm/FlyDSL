# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from __future__ import annotations

from typing import Callable, Tuple

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler import jit_function


@pytest.fixture
def frontend_only_jit(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-flydsl-key")

    def compile_noop(cls, module, **_kwargs):
        return module

    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(compile_noop))


def test_jit_accepts_lambda_constexpr_parameter(frontend_only_jit):
    @flyc.jit
    def build(value: fx.Int32, transform: fx.Constexpr[Callable], expected: fx.Constexpr[int]):
        assert transform(5) == expected
        result = transform(value)
        assert isinstance(result, fx.Int32)

    build(7, lambda x: x * 3, 15)


def test_jit_accepts_scalar_and_tuple_constexpr_parameters(frontend_only_jit):
    @flyc.jit
    def build(flag: fx.Constexpr[bool], scale: fx.Constexpr[float], shape: fx.Constexpr[Tuple[int, int]]):
        assert flag
        assert scale == 1.5
        assert shape == (16, 32)

    build(True, 1.5, (16, 32))


def test_jit_accepts_nested_tuple_constexpr_parameter(frontend_only_jit):
    @flyc.jit
    def build(shape: fx.Constexpr[Tuple[int, Tuple[bool, float]]]):
        assert shape == (16, (True, 2.5))

    build((16, (True, 2.5)))
