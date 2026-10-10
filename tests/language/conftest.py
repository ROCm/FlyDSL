# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared pytest configuration for the language conformance suite (tests/language).

Tests are grouped by their language documentation. L1a cases use the real DSL
frontend via ``@flyc.jit`` with backend compilation replaced by a no-op.
Cases marked ``l1b_target_dialect`` or ``l2_device`` use the full compiler and
runtime, so one documentation topic can include all three verification tiers.
"""

import pytest

from flydsl.compiler import jit_function


@pytest.fixture(autouse=True)
def frontend_only_jit(request, monkeypatch):
    if request.node.get_closest_marker("l1b_target_dialect") or request.node.get_closest_marker("l2_device"):
        return

    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-flydsl-key")

    def compile_noop(cls, module, **_kwargs):
        return module

    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(compile_noop))
