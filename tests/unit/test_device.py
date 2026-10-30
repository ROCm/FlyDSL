# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import os
import subprocess
import sys
from pathlib import Path

import pytest

from flydsl.runtime import device

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clear_arch_caches(monkeypatch):
    for name in ("FLYDSL_GPU_ARCH", "HSA_OVERRIDE_GFX_VERSION", "ARCH"):
        monkeypatch.delenv(name, raising=False)
    # These functions may be monkeypatched until after this fixture tears down.
    clear_arch = device._arch_from_hardware.cache_clear
    clear_count = device.get_rocm_device_count.cache_clear
    clear_arch()
    clear_count()
    yield
    clear_arch()
    clear_count()


def test_arch_detection_uses_visible_device_query(monkeypatch):
    calls = []

    def check_output(command, **kwargs):
        calls.append((command, kwargs))
        return "gfx000\ngfx950:sramecc+:xnack-\n"

    monkeypatch.setattr(subprocess, "check_output", check_output)

    assert device._arch_from_rocm_agent_enumerator() == "gfx950:sramecc+:xnack-"
    assert calls[0][0] == ["rocm_agent_enumerator", "-name"]


def test_arch_detection_ignores_hidden_heterogeneous_gpu(monkeypatch):
    """KFD counting order must not choose the architecture of a hidden GPU."""

    def check_output(command, **_kwargs):
        if command == ["rocm_agent_enumerator"]:
            return "gfx1100\ngfx950\n"
        assert command == ["rocm_agent_enumerator", "-name"]
        return "gfx950:sramecc+:xnack-\ngfx9-4-generic:sramecc+:xnack-\n"

    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(subprocess, "check_output", check_output)

    assert device._arch_from_hardware() == "gfx950"


@pytest.mark.parametrize("failure", ["missing", "timeout", "exit", "empty", "cpu-only"])
def test_hardware_detection_falls_back_to_rocminfo(monkeypatch, failure):
    def check_output(command, **_kwargs):
        if command == ["rocm_agent_enumerator", "-name"]:
            if failure == "missing":
                raise FileNotFoundError(command[0])
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, 1)
            if failure == "exit":
                raise subprocess.CalledProcessError(1, command)
            return "gfx000\n" if failure == "cpu-only" else ""
        assert command == ["rocminfo"]
        return "  Name:                    gfx000\n  Name:                    gfx950:sramecc+:xnack-\n"

    monkeypatch.setattr(subprocess, "check_output", check_output)

    assert device._arch_from_hardware() == "gfx950"


@pytest.mark.parametrize("failure", ["missing", "timeout", "exit", "empty"])
def test_hardware_detection_does_not_default_to_gfx942(monkeypatch, failure):
    calls = []

    def check_output(command, **_kwargs):
        calls.append(command[0])
        if failure == "missing":
            raise FileNotFoundError(command[0])
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        if failure == "exit":
            raise subprocess.CalledProcessError(1, command)
        return ""

    monkeypatch.setattr(subprocess, "check_output", check_output)

    for _ in range(3):
        with pytest.raises(RuntimeError, match="FLYDSL_GPU_ARCH"):
            device.get_rocm_arch()

    assert calls == ["rocm_agent_enumerator", "rocminfo"]


@pytest.mark.parametrize("override", ["FLYDSL_GPU_ARCH", "HSA_OVERRIDE_GFX_VERSION", "ARCH"])
def test_explicit_target_overrides_bypass_hardware_detection(monkeypatch, override):
    monkeypatch.setenv(override, "9.5.0" if override == "HSA_OVERRIDE_GFX_VERSION" else "GFX950")
    monkeypatch.setattr(
        device,
        "_arch_from_hardware",
        lambda: pytest.fail("hardware detection should not run"),
    )

    assert device.get_rocm_arch() == "gfx950"
    assert device.get_warp_size() == 64
    assert device.is_rdna_arch() is False


@pytest.mark.parametrize("name", ["FLYDSL_GPU_ARCH", "HSA_OVERRIDE_GFX_VERSION"])
def test_arch_overrides_take_precedence_over_compile_arch(monkeypatch, name):
    """``ARCH`` is a compile-time hint; the dedicated overrides win when both are set."""
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv(name, "gfx950")
    monkeypatch.setattr(device, "_arch_from_hardware", lambda: pytest.fail("hardware detection should not run"))

    assert device.get_rocm_arch() == "gfx950"


def test_device_count_counts_gpus_not_supported_isas(monkeypatch):
    """Two identical GPUs must count as two, even when each exposes two ISAs."""

    def check_output(command, **_kwargs):
        if command == ["rocm_agent_enumerator", "-name"]:
            return "gfx950:sramecc+:xnack-\ngfx9-4-generic:sramecc+:xnack-\n" * 2
        assert command == ["rocm_agent_enumerator"]
        return "gfx000\ngfx950\ngfx950\n"

    monkeypatch.setattr(subprocess, "check_output", check_output)

    assert device.get_rocm_device_count() == 2


@pytest.mark.parametrize("failure", ["missing", "timeout"])
def test_device_count_remains_best_effort(monkeypatch, failure):
    def check_output(command, **_kwargs):
        if failure == "missing":
            raise FileNotFoundError(command[0])
        raise subprocess.TimeoutExpired(command, 5)

    monkeypatch.setattr(subprocess, "check_output", check_output)

    assert device.get_rocm_device_count() == 0


@pytest.mark.parametrize(
    ("module", "expected_test"),
    [
        ("test_device.py", "test_arch_detection_uses_visible_device_query"),
        ("test_softmax_autotune.py", "test_candidate_order_is_deterministic"),
        ("test_ktrace_emit.py", "test_ranges_pair_by_start_slot_not_by_name"),
    ],
)
def test_backend_agnostic_collection_needs_no_gpu(tmp_path, module, expected_test):
    """Collecting L0 tests must work with no detection tools and no arch override."""
    if module != "test_device.py":
        pytest.importorskip("torch")
        pytest.importorskip("flydsl._mlir.ir")
    probe_env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("FLYDSL_GPU_ARCH", "HSA_OVERRIDE_GFX_VERSION", "ARCH")
    }
    # Empty PATH: rocm_agent_enumerator and rocminfo are not resolvable.
    probe_env["PATH"] = ""
    # Preserve the package/build paths used by the parent test session.
    probe_env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_REPO_ROOT), *sys.path]))
    probe_env["PYTEST_ADDOPTS"] = ""
    probe_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    probe_env["FLYDSL_AUTOTUNE_CACHE_DIR"] = str(tmp_path / "autotune")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "l0_backend_agnostic",
            "-p",
            "no:cacheprovider",
            str(_REPO_ROOT / "tests" / "unit" / module),
        ],
        cwd=_REPO_ROOT,
        env=probe_env,
        text=True,
        capture_output=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert expected_test in result.stdout, result.stdout + result.stderr
