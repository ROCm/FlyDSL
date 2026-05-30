# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Regression tests for JIT dependency cache invalidation."""

import sys

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler import jit_function

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


def _helper():
    pass


def _sentinel_helper():
    raise RuntimeError("sentinel helper observed during retrace")


@flyc.kernel
def _dependency_kernel():
    _helper()


@flyc.jit
def _dependency_launch(stream: fx.Stream = fx.Stream(None)):
    _dependency_kernel().launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


def _reset_jit(jit_fn):
    jit_fn._clear_inprocess_caches()
    jit_fn.manager_key = None
    jit_fn._manager_owner_cls = None
    jit_fn.cache_manager = None
    jit_fn._target = None
    jit_fn._sig = None
    jit_fn._has_self_param = False


@pytest.fixture(autouse=True)
def compile_only(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-flydsl-key")

    def compile_noop(cls, module, *, arch: str = "", func_name: str = "", link_libs=None):
        return module

    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(compile_noop))
    _reset_jit(_dependency_launch)


def test_cache_disabled_revalidates_helper_dependency_source(monkeypatch):
    _dependency_launch()
    assert len(_dependency_launch._mem_cache) == 1

    monkeypatch.setattr(sys.modules[__name__], "_helper", _sentinel_helper)

    with pytest.raises(RuntimeError, match="sentinel helper observed"):
        _dependency_launch()
