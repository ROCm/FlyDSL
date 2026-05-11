# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Smoke coverage for PyTorch FakeTensor metadata in FlyDSL AOT paths."""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.jit_argument import TensorAdaptor

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


@flyc.kernel
def _fake_aot_kernel(Out: fx.Tensor):
    pass


@flyc.jit
def _fake_aot_launch(Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _fake_aot_kernel(Out).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


def _fake_tensor_mode():
    fake_tensor = pytest.importorskip("torch._subclasses.fake_tensor")
    return fake_tensor.FakeTensorMode()


def _reset_jit(jit_fn):
    jit_fn._call_state_cache.clear()
    jit_fn._mem_cache.clear()
    jit_fn._last_compiled = None
    jit_fn.manager_key = None
    jit_fn.cache_manager = None
    jit_fn._target = None
    jit_fn._sig = None
    jit_fn._has_self_param = False


def test_fake_tensor_dlpack_constructs_tensor_adaptor():
    """FakeTensorMode should provide enough DLPack metadata for compile-only AOT."""
    with _fake_tensor_mode():
        fake = torch.empty((16, 8), dtype=torch.float16, device="cuda").transpose(0, 1)
        adaptor = flyc.from_dlpack(fake)

    assert tuple(adaptor.tensor_adaptor.shape) == (8, 16)
    assert tuple(adaptor.tensor_adaptor.stride) == (1, 8)


def test_tensor_cache_signature_matches_fake_and_real_layout_metadata():
    """A fake AOT artifact should match the same real tensor but not a different layout."""
    with _fake_tensor_mode():
        fake = torch.empty_strided((8, 8), (16, 2), dtype=torch.float16, device="cuda")

    matching_real = torch.empty_strided((8, 8), (16, 2), dtype=torch.float16)
    different_real = torch.empty((8, 8), dtype=torch.float16)

    assert TensorAdaptor.raw_cache_signature(fake) == TensorAdaptor.raw_cache_signature(matching_real)
    assert TensorAdaptor.raw_cache_signature(fake) != TensorAdaptor.raw_cache_signature(different_real)


def test_fake_tensor_compile_only_aot_cache_is_reused_by_real_tensor(monkeypatch, tmp_path):
    """Fake compile-only AOT should write a disk artifact that a real tensor can load."""
    from flydsl.compiler import jit_function

    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "0")
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-fake-tensor-aot-key")

    compile_calls = []

    def compile_noop(cls, module, *, arch: str = "", func_name: str = "", link_libs=None):
        compile_calls.append(func_name)
        return module

    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(compile_noop))

    _reset_jit(_fake_aot_launch)
    with _fake_tensor_mode():
        fake = torch.empty((8, 8), dtype=torch.float16, device="cuda")
        _fake_aot_launch(fake)

    assert compile_calls == ["_fake_aot_launch"]
    assert list(tmp_path.rglob("*.pkl"))

    _reset_jit(_fake_aot_launch)
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "1")
    real = torch.empty((8, 8), dtype=torch.float16)
    _fake_aot_launch(real)

    assert compile_calls == ["_fake_aot_launch"]
