# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Behavioral tests for GPU architecture detection and fallback."""

import pytest

from flydsl.runtime import device


@pytest.fixture(autouse=True)
def _clear_arch_cache():
    """Avoid stale lru_cache hits when monkeypatching hardware detection."""
    device._arch_from_hardware.cache_clear()
    yield
    device._arch_from_hardware.cache_clear()


def test_get_rocm_arch_uses_explicit_env(monkeypatch):
    """Explicit user override should always win."""
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx950")
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "7.0.0")
    monkeypatch.delenv("FLYDSL_GPU_ARCH_FALLBACK", raising=False)
    monkeypatch.setattr(device, "_arch_from_hardware", lambda: None)
    assert device.get_rocm_arch() == "gfx950"


def test_get_rocm_arch_raises_without_hardware_and_no_fallback(monkeypatch):
    """Fail loudly when architecture cannot be detected."""
    monkeypatch.delenv("FLYDSL_GPU_ARCH", raising=False)
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    monkeypatch.delenv("FLYDSL_GPU_ARCH_FALLBACK", raising=False)
    monkeypatch.setattr(device, "_arch_from_hardware", lambda: None)
    with pytest.raises(
        RuntimeError, match="Unable to detect ROCm GPU architecture"
    ):
        device.get_rocm_arch()


def test_get_rocm_arch_fallback_opt_in(monkeypatch):
    """Opt-in fallback should restore legacy gfx942 behavior."""
    monkeypatch.delenv("FLYDSL_GPU_ARCH", raising=False)
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    monkeypatch.setenv("FLYDSL_GPU_ARCH_FALLBACK", "1")
    monkeypatch.setattr(device, "_arch_from_hardware", lambda: None)
    assert device.get_rocm_arch() == "gfx942"


def test_get_rocm_arch_fallback_env_true_false(monkeypatch):
    """Fallback flag accepts common truthy and falsy values."""
    monkeypatch.delenv("FLYDSL_GPU_ARCH", raising=False)
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    monkeypatch.setenv("FLYDSL_GPU_ARCH_FALLBACK", "false")
    monkeypatch.setattr(device, "_arch_from_hardware", lambda: None)
    with pytest.raises(RuntimeError, match="Unable to detect ROCm GPU architecture"):
        device.get_rocm_arch()

    monkeypatch.setenv("FLYDSL_GPU_ARCH_FALLBACK", "on")
    assert device.get_rocm_arch() == "gfx942"


def test_get_rocm_arch_raises_for_invalid_override(monkeypatch):
    """Invalid explicit env override should fail with clear diagnostics."""
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "bad-arch")
    monkeypatch.setattr(device, "_arch_from_hardware", lambda: None)
    with pytest.raises(RuntimeError, match="invalid"):
        device.get_rocm_arch()


def test_parse_hsa_override(monkeypatch):
    """HSA-style major.minor.patch override should convert to gfxNNN."""
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "9.4.2")
    monkeypatch.delenv("FLYDSL_GPU_ARCH", raising=False)
    monkeypatch.delenv("FLYDSL_GPU_ARCH_FALLBACK", raising=False)
    monkeypatch.setattr(device, "_arch_from_hardware", lambda: None)
    assert device.get_rocm_arch() == "gfx942"


def test_parse_hsa_override_multi_digit_parts(monkeypatch):
    """Multi-digit dotted override values should be accepted."""
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    monkeypatch.delenv("FLYDSL_GPU_ARCH", raising=False)
    monkeypatch.delenv("FLYDSL_GPU_ARCH_FALLBACK", raising=False)
    monkeypatch.setattr(device, "_arch_from_hardware", lambda: None)
    assert device.get_rocm_arch() == "gfx1100"
