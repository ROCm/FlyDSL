# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from unittest import mock

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler import jit_function


@pytest.fixture
def compile_only_frontend(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-flydsl-key")

    compile_calls = []

    def compile_noop(cls, module, **_kwargs):
        compile_calls.append(module)
        return module

    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(compile_noop))

    materialize = mock.Mock(side_effect=AssertionError("compile-only materialized the execution engine"))
    monkeypatch.setattr(jit_function.CompiledArtifact, "_get_func_exe", materialize)
    return compile_calls, materialize


@pytest.mark.parametrize("compile_only", ("1", "true", "yes", "on"))
def test_compile_only_persists_artifact_without_execution_engine(
    monkeypatch,
    compile_only_frontend,
    compile_only,
):
    monkeypatch.setenv("COMPILE_ONLY", compile_only)
    compile_calls, materialize = compile_only_frontend

    @flyc.jit
    def launch(value: fx.Int32):
        pass

    assert flyc.compile(launch, 7) is None
    assert flyc.compile(launch, 7) is None

    assert len(compile_calls) == 1
    assert launch._last_compiled is not None
    assert launch.cache_info().disk_size == 1
    materialize.assert_not_called()
