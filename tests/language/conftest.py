# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared pytest configuration for the language conformance suite (tests/language).

Language semantics are exercised through the real DSL frontend via ``@flyc.jit``
in frontend-only mode: ``COMPILE_ONLY`` is set (no device execution) and the
MLIR compile step is replaced by a no-op, so tracing runs without a GPU and
without lowering to any target dialect. Kernel and jit tracing share the same
type/arithmetic semantics, so testing at the jit level is sufficient here.

The fixture is ``autouse`` so every test in this directory runs under the
frontend-only harness without having to request it explicitly.
"""

import pytest

from flydsl.compiler import jit_function


@pytest.fixture(autouse=True)
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
