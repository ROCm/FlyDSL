# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""device runtime registry and compile-backend pairing."""

import pytest

import flydsl.runtime.device_runtime as dr


@pytest.fixture(autouse=True)
def _reset_device_runtime_singleton():
    """Each test starts without a cached DeviceRuntime instance."""
    dr._instance = None
    dr._runtime_cls_override = None
    dr._EXTRA_MAPPINGS.clear()
    yield
    dr._instance = None
    dr._runtime_cls_override = None
    dr._EXTRA_MAPPINGS.clear()


class _FakeCudaRuntime(dr.DeviceRuntime):
    kind = "cuda"

    def device_count(self) -> int:
        return 1


def test_rocm_runtime_kind_matches_compile_backend(monkeypatch):
    monkeypatch.delenv("FLYDSL_RUNTIME_KIND", raising=False)
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    rt = dr.get_device_runtime()
    assert rt.kind == "rocm"
    dr.ensure_compile_runtime_compatible("rocm", runtime=rt)


def test_ensure_mismatch_raises():
    bad = _FakeCudaRuntime()
    with pytest.raises(RuntimeError, match="requires device runtime kind"):
        dr.ensure_compile_runtime_compatible("rocm", runtime=bad)


def test_unknown_runtime_kind_env(monkeypatch):
    """Invalid FLYDSL_RUNTIME_KIND fails at compile/runtime pairing first."""
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "not_a_real_kind")
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    with pytest.raises(RuntimeError, match="requires device runtime kind"):
        dr.get_device_runtime()


def test_unknown_runtime_kind_after_pairing_passes(monkeypatch):
    """When env strings agree with mapping, unknown kind fails in class lookup."""
    dr.register_compile_runtime_mapping("custom", "weird_kind")
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "custom")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "weird_kind")
    try:
        with pytest.raises(ValueError, match="Unknown FLYDSL_RUNTIME_KIND"):
            dr.get_device_runtime()
    finally:
        dr._EXTRA_MAPPINGS.pop("custom", None)


def test_register_compile_runtime_mapping():
    dr.register_compile_runtime_mapping("foo", "rocm")
    try:
        dr.ensure_compile_runtime_compatible("foo", runtime=dr.RocmDeviceRuntime())
    finally:
        dr._EXTRA_MAPPINGS.pop("foo", None)


def test_pairing_from_env_no_singleton(monkeypatch):
    """Pairing check used on compile path must not create DeviceRuntime."""
    monkeypatch.delenv("FLYDSL_RUNTIME_KIND", raising=False)
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    dr.ensure_compile_runtime_pairing_from_env("rocm")
    assert dr._instance is None


def test_pairing_from_env_mismatch_raises(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "not_a_registered_kind")
    with pytest.raises(RuntimeError, match="requires device runtime kind"):
        dr.ensure_compile_runtime_pairing_from_env("rocm")
