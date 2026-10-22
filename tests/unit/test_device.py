# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""GPU-independent tests for LLC detection using fake KFD sysfs trees."""

import os

import pytest

import flydsl.runtime.device as device_module
from flydsl.runtime.device import get_llc_cache_bytes

pytestmark = [pytest.mark.l0_backend_agnostic]


def _write_properties(path, **kv):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for key, value in kv.items():
            f.write(f"{key} {value}\n")


def _make_fake_kfd_tree(root):
    """node 0 = CPU (location_id 0); node 1 = bus 131, 64 MiB LLC;
    node 2 = bus 3, 96 MiB LLC."""
    _write_properties(os.path.join(root, "0", "properties"), location_id=0)
    _write_properties(os.path.join(root, "0", "caches", "0", "properties"), level=3, size=32768)
    _write_properties(os.path.join(root, "1", "properties"), location_id=33536)
    _write_properties(os.path.join(root, "1", "caches", "0", "properties"), level=1, size=32)
    _write_properties(os.path.join(root, "1", "caches", "1", "properties"), level=3, size=65536)
    _write_properties(os.path.join(root, "2", "properties"), location_id=768)
    _write_properties(os.path.join(root, "2", "caches", "0", "properties"), level=3, size=98304)


def _make_fake_kfd_tree_single_gpu(root):
    """node 0 = CPU (location_id 0); node 1 = bus 5, 128 MiB LLC (single GPU case)."""
    _write_properties(os.path.join(root, "0", "properties"), location_id=0)
    _write_properties(os.path.join(root, "1", "properties"), location_id=1280)
    _write_properties(os.path.join(root, "1", "caches", "0", "properties"), level=3, size=131072)


@pytest.fixture
def fake_kfd_root(tmp_path, monkeypatch):
    root = tmp_path / "nodes"
    _make_fake_kfd_tree(str(root))
    monkeypatch.setattr(device_module, "_KFD_TOPOLOGY_ROOT", str(root))
    monkeypatch.delenv("FLYDSL_RUNTIME_LLC_CACHE_MB", raising=False)
    get_llc_cache_bytes.cache_clear()
    yield root
    get_llc_cache_bytes.cache_clear()


@pytest.fixture
def fake_kfd_root_single_gpu(tmp_path, monkeypatch):
    """Single GPU case: one CPU node + one GPU node."""
    root = tmp_path / "nodes"
    _make_fake_kfd_tree_single_gpu(str(root))
    monkeypatch.setattr(device_module, "_KFD_TOPOLOGY_ROOT", str(root))
    monkeypatch.delenv("FLYDSL_RUNTIME_LLC_CACHE_MB", raising=False)
    get_llc_cache_bytes.cache_clear()
    yield root
    get_llc_cache_bytes.cache_clear()


def test_matches_bus_131_to_64_mib(fake_kfd_root):
    assert get_llc_cache_bytes(pci_bus_id=131) == 64 * 1024 * 1024


def test_matches_bus_3_to_96_mib(fake_kfd_root):
    assert get_llc_cache_bytes(pci_bus_id=3) == 96 * 1024 * 1024


def test_unknown_bus_returns_zero(fake_kfd_root):
    assert get_llc_cache_bytes(pci_bus_id=999) == 0


def test_cpu_node_location_id_zero_is_never_matched(fake_kfd_root):
    assert get_llc_cache_bytes(pci_bus_id=0) == 0


def test_pci_bus_id_none_with_single_gpu_returns_that_nodes_llc(fake_kfd_root_single_gpu):
    """With only one non-CPU GPU node, pci_bus_id=None resolves to that node."""
    assert get_llc_cache_bytes(pci_bus_id=None) == 128 * 1024 * 1024


def test_pci_bus_id_none_with_multiple_gpus_returns_zero(fake_kfd_root):
    """With multiple non-CPU GPU nodes, pci_bus_id=None returns 0 (ambiguous)."""
    assert get_llc_cache_bytes(pci_bus_id=None) == 0


def test_env_override_wins_and_needs_no_sysfs(monkeypatch):
    monkeypatch.setattr(device_module, "_KFD_TOPOLOGY_ROOT", "/nonexistent/path/for/this/test")
    monkeypatch.setenv("FLYDSL_RUNTIME_LLC_CACHE_MB", "100")
    get_llc_cache_bytes.cache_clear()
    try:
        assert get_llc_cache_bytes(pci_bus_id=131) == 100 * 1024 * 1024
    finally:
        monkeypatch.delenv("FLYDSL_RUNTIME_LLC_CACHE_MB", raising=False)
        get_llc_cache_bytes.cache_clear()


def test_missing_sysfs_root_returns_zero_not_raise(monkeypatch):
    monkeypatch.setattr(device_module, "_KFD_TOPOLOGY_ROOT", "/nonexistent/path/for/this/test")
    monkeypatch.delenv("FLYDSL_RUNTIME_LLC_CACHE_MB", raising=False)
    get_llc_cache_bytes.cache_clear()
    assert get_llc_cache_bytes(pci_bus_id=131) == 0
    get_llc_cache_bytes.cache_clear()
