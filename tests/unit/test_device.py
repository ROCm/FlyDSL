# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import os
import subprocess
import sys
from pathlib import Path

import pytest

from flydsl.runtime import device
from flydsl.runtime.device import get_llc_cache_bytes

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clear_arch_caches(monkeypatch):
    for name in ("FLYDSL_GPU_ARCH", "HSA_OVERRIDE_GFX_VERSION"):
        monkeypatch.delenv(name, raising=False)
    # These functions may be monkeypatched until after this fixture tears down.
    clear_arch = device._arch_from_hardware.cache_clear
    clear_count = device.get_rocm_device_count.cache_clear
    clear_arch()
    clear_count()
    yield
    clear_arch()
    clear_count()


@pytest.fixture
def linux_detection(monkeypatch):
    """Take the rocm_agent_enumerator/rocminfo path these tests mock; Windows asks torch and HIP instead."""
    monkeypatch.setattr(sys, "platform", "linux")


def test_arch_detection_uses_visible_device_query(monkeypatch):
    calls = []

    def check_output(command, **kwargs):
        calls.append((command, kwargs))
        return "gfx000\ngfx950:sramecc+:xnack-\n"

    monkeypatch.setattr(subprocess, "check_output", check_output)

    assert device._arch_from_rocm_agent_enumerator() == "gfx950:sramecc+:xnack-"
    assert calls[0][0] == ["rocm_agent_enumerator", "-name"]


@pytest.mark.usefixtures("linux_detection")
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


@pytest.mark.usefixtures("linux_detection")
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


@pytest.mark.usefixtures("linux_detection")
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


@pytest.mark.parametrize(
    ("platform", "tried"), [("linux", "rocm_agent_enumerator or rocminfo"), ("win32", "torch.cuda")]
)
def test_detection_failure_names_what_was_tried(monkeypatch, platform, tried):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(subprocess, "check_output", lambda command, **_kwargs: "")
    monkeypatch.setattr(device, "_arch_from_torch", lambda: "")

    with pytest.raises(RuntimeError, match=f"with {tried}\\. Set FLYDSL_GPU_ARCH"):
        device.get_rocm_arch()


@pytest.mark.parametrize("override", ["FLYDSL_GPU_ARCH", "HSA_OVERRIDE_GFX_VERSION"])
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


@pytest.mark.usefixtures("linux_detection")
def test_device_count_counts_gpus_not_supported_isas(monkeypatch):
    """Two identical GPUs must count as two, even when each exposes two ISAs."""

    def check_output(command, **_kwargs):
        if command == ["rocm_agent_enumerator", "-name"]:
            return "gfx950:sramecc+:xnack-\ngfx9-4-generic:sramecc+:xnack-\n" * 2
        assert command == ["rocm_agent_enumerator"]
        return "gfx000\ngfx950\ngfx950\n"

    monkeypatch.setattr(subprocess, "check_output", check_output)

    assert device.get_rocm_device_count() == 2


@pytest.mark.usefixtures("linux_detection")
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
        key: value for key, value in os.environ.items() if key not in ("FLYDSL_GPU_ARCH", "HSA_OVERRIDE_GFX_VERSION")
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


# --- LLC (last-level cache) detection using synthetic KFD sysfs trees ---
#
# All values below are made up; nothing depends on the GPUs of the host running
# the tests. Each node is (pci_domain, pci_bus, [(cache_level, size_kib), ...]),
# one cache tuple per KFD caches/*/properties entry.

_MIB = 1024 * 1024

_CPU_NODE = (0, 0, [(3, 32768)])  # location_id 0; must never be matched as a GPU
_GPU_NODES = {
    "l3_small": (0, 0x10, [(1, 32), (3, 64 * 1024)]),
    "l3_large": (0, 0x20, [(3, 96 * 1024)]),
    "l2_only": (0, 0x30, [(1, 32), (2, 48 * 1024)]),  # no L3: L2 is the last level
    "l3_below_bigger_l2": (0, 0x40, [(2, 16 * 1024), (3, 8 * 1024)]),
    "l3_split": (0, 0x60, [(2, 4 * 1024), (2, 4 * 1024), (3, 32 * 1024), (3, 32 * 1024)]),
    "other_domain_same_bus": (1, 0x20, [(3, 32 * 1024)]),
}
_SINGLE_GPU_NODE = (0, 0x50, [(3, 128 * 1024)])


def _write_properties(path, **kv):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for key, value in kv.items():
            f.write(f"{key} {value}\n")


def _write_kfd_tree(root, nodes):
    for idx, (domain, bus, caches) in enumerate(nodes):
        node = os.path.join(root, str(idx))
        _write_properties(os.path.join(node, "properties"), domain=domain, location_id=bus << 8)
        for cache_idx, (level, size_kib) in enumerate(caches):
            _write_properties(os.path.join(node, "caches", str(cache_idx), "properties"), level=level, size=size_kib)


def _loc(name):
    domain, bus, _ = _GPU_NODES[name]
    return domain, bus << 8


@pytest.fixture
def _llc_env(monkeypatch):
    monkeypatch.delenv("FLYDSL_RUNTIME_LLC_CACHE_MB", raising=False)
    device._kfd_llc_bytes.cache_clear()
    yield
    device._kfd_llc_bytes.cache_clear()


@pytest.fixture
def fake_kfd_root(tmp_path, monkeypatch, _llc_env):
    """CPU node plus every GPU node in ``_GPU_NODES`` (multi-GPU host)."""
    root = tmp_path / "nodes"
    _write_kfd_tree(str(root), [_CPU_NODE, *_GPU_NODES.values()])
    monkeypatch.setattr(device, "_KFD_TOPOLOGY_ROOT", str(root))
    return root


@pytest.fixture
def fake_kfd_root_single_gpu(tmp_path, monkeypatch, _llc_env):
    """CPU node plus one GPU node (single-GPU host)."""
    root = tmp_path / "nodes"
    _write_kfd_tree(str(root), [_CPU_NODE, _SINGLE_GPU_NODE])
    monkeypatch.setattr(device, "_KFD_TOPOLOGY_ROOT", str(root))
    return root


@pytest.mark.parametrize("name,expected_mib", [("l3_small", 64), ("l3_large", 96)])
def test_location_selects_matching_node_l3(fake_kfd_root, name, expected_mib):
    assert device._kfd_llc_bytes(_loc(name)) == expected_mib * _MIB


def test_l2_only_node_reports_l2_as_llc(fake_kfd_root):
    """Without an L3 entry the large L2 is the last-level cache."""
    assert device._kfd_llc_bytes(_loc("l2_only")) == 48 * _MIB


def test_highest_level_wins_over_larger_lower_level(fake_kfd_root):
    """L3 is reported even when the L2 entry is bigger; L1 is never considered."""
    assert device._kfd_llc_bytes(_loc("l3_below_bigger_l2")) == 8 * _MIB


def test_entries_at_same_level_are_summed(fake_kfd_root):
    """Several instances of the last-level cache add up to the device total."""
    assert device._kfd_llc_bytes(_loc("l3_split")) == 64 * _MIB


def test_domain_distinguishes_nodes_on_the_same_bus(fake_kfd_root):
    assert device._kfd_llc_bytes(_loc("l3_large")) == 96 * _MIB
    assert device._kfd_llc_bytes(_loc("other_domain_same_bus")) == 32 * _MIB


def test_unknown_location_returns_zero(fake_kfd_root):
    assert device._kfd_llc_bytes((0, 0xFF << 8)) == 0
    domain, location_id = _loc("l3_large")
    assert device._kfd_llc_bytes((domain, location_id | 1)) == 0  # same bus, other function


def test_cpu_node_location_id_zero_is_never_matched(fake_kfd_root):
    assert device._kfd_llc_bytes((0, 0)) == 0


def test_no_location_with_single_gpu_returns_that_nodes_llc(fake_kfd_root_single_gpu):
    assert device._kfd_llc_bytes(None) == 128 * _MIB


def test_no_location_with_multiple_gpus_returns_zero(fake_kfd_root):
    assert device._kfd_llc_bytes(None) == 0


def test_env_override_wins_and_needs_no_sysfs(monkeypatch, _llc_env):
    monkeypatch.setattr(device, "_KFD_TOPOLOGY_ROOT", "/nonexistent/path/for/this/test")
    monkeypatch.setenv("FLYDSL_RUNTIME_LLC_CACHE_MB", "100")
    assert get_llc_cache_bytes() == 100 * _MIB


def test_missing_sysfs_root_returns_zero_not_raise(monkeypatch, _llc_env):
    monkeypatch.setattr(device, "_KFD_TOPOLOGY_ROOT", "/nonexistent/path/for/this/test")
    assert device._kfd_llc_bytes(_loc("l3_small")) == 0


# --- current HIP device -> PCI address -> LLC, with a fake HIP library ---


class _FakeHip:
    """Stand-in for the ctypes ``libamdhip64`` handle: the current device sits at a fixed BDF."""

    def __init__(self, bdf="0000:c1:00.0", status=0):
        self._bdf = bdf.encode()
        self._status = status

    def hipDeviceGetPCIBusId(self, buf, length, dev):
        buf.value = self._bdf
        return self._status


@pytest.fixture
def fake_hip(monkeypatch):
    from flydsl.runtime.device_runtime import rocm

    def install(lib):
        monkeypatch.setattr(rocm, "_HIP_LIB", lib)
        monkeypatch.setattr(rocm, "_hip_get_device", lambda: 0)
        return rocm

    return install


def test_hip_location_parses_bdf_string(fake_hip):
    rocm = fake_hip(_FakeHip("0001:c1:02.3"))
    assert rocm._hip_current_pci_location() == (1, 0xC1 << 8 | 2 << 3 | 3)


def test_hip_location_none_when_hip_call_fails(fake_hip):
    rocm = fake_hip(_FakeHip(status=1))
    assert rocm._hip_current_pci_location() is None


def test_hip_location_none_without_hip_library(fake_hip):
    rocm = fake_hip(None)
    assert rocm._hip_current_pci_location() is None


def test_uses_current_hip_device_location_on_multi_gpu(fake_kfd_root, fake_hip):
    """Several GPU nodes: the public API resolves the node via the current HIP device."""
    domain, bus, _ = _GPU_NODES["l3_large"]
    fake_hip(_FakeHip(f"{domain:04x}:{bus:02x}:00.0"))
    assert get_llc_cache_bytes() == 96 * _MIB


def test_without_hip_falls_back_to_single_gpu_rule(fake_kfd_root, fake_hip):
    fake_hip(None)
    assert get_llc_cache_bytes() == 0
