# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import functools
import os
import subprocess
from typing import Optional

from ..utils import env

_ROCM_AGENT_TIMEOUT_S = int(os.environ.get("FLYDSL_ROCM_AGENT_TIMEOUT", "300"))


def _arch_from_rocm_agent_enumerator() -> Optional[str]:
    """Query rocm_agent_enumerator (standard ROCm tool) for the first GPU arch."""
    try:
        out = subprocess.check_output(
            ["rocm_agent_enumerator", "-name"],
            text=True,
            timeout=_ROCM_AGENT_TIMEOUT_S,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            name = line.strip()
            if name.startswith("gfx") and name != "gfx000":
                return name
    except Exception:
        pass
    return None


@functools.lru_cache(maxsize=None)
def _arch_from_hardware() -> str:
    """Cached hardware detection (rocm_agent_enumerator is slow)."""
    arch = _arch_from_rocm_agent_enumerator()
    if arch:
        return arch.split(":", 1)[0]
    return "gfx942"


def get_rocm_arch() -> str:
    """Best-effort ROCm GPU arch string, always lower-cased (e.g. 'gfx942').

    Lower-casing happens here so every caller can compare against lower-case
    literals without normalising first; ROCm itself only ever emits lower-case
    names, so this only affects hand-set environment overrides.
    """
    env = os.environ.get("FLYDSL_GPU_ARCH") or os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if env:
        env = env.lower()
        if env.startswith("gfx"):
            return env
        if env.count(".") == 2:
            parts = env.split(".")
            return f"gfx{parts[0]}{parts[1]}{parts[2]}"

    return _arch_from_hardware().lower()


@functools.lru_cache(maxsize=None)
def get_rocm_device_count() -> int:
    """Best-effort ROCm visible GPU count via ``rocm_agent_enumerator`` (standard ROCm tool).

    Uses the same invocation as :func:`_arch_from_rocm_agent_enumerator`. Returns 0
    when the tool is unavailable or no discrete GPU agents are reported.
    """
    try:
        out = subprocess.check_output(
            ["rocm_agent_enumerator", "-name"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        n = 0
        for line in out.splitlines():
            name = line.strip()
            if name.startswith("gfx") and name != "gfx000":
                n += 1
        return n
    except Exception:
        return 0


def is_rdna_arch(arch: Optional[str] = None) -> bool:
    """Check if architecture is RDNA-based (gfx10, gfx11, or gfx120).

    This is the single source of truth for CDNA vs RDNA classification.
    RDNA architectures use wave32 and have different buffer descriptor flags.
    CDNA5 gfx1250 is also wave32, but is intentionally not classified as RDNA.

    If arch is None, the current GPU arch is auto-detected.
    """
    if arch is None:
        arch = get_rocm_arch()
    if not arch:
        return False
    arch = arch.lower()
    if arch.startswith("gfx10") or arch.startswith("gfx11"):
        return True
    if arch.startswith("gfx120"):
        return True
    return False


def get_warp_size(arch: Optional[str] = None) -> int:
    """Lanes per warp for an architecture.

    If arch is None, the current GPU arch is auto-detected.
    """
    if arch is None:
        arch = get_rocm_arch()
    if not arch:
        return 64
    arch = arch.lower()
    if arch.startswith("gfx10") or arch.startswith("gfx11") or arch.startswith("gfx12"):
        return 32
    return 64


# LDS bytes available per CU / WGP, keyed by architecture.
SMEM_CAPACITY_MAP = {
    "gfx942": 65536,
    "gfx950": 163840,
    "gfx1100": 65536,
    "gfx1151": 65536,
    "gfx1201": 65536,
    "gfx1250": 327680,
}


def check_smem_capacity(allocated_bytes: int, arch: str = None):
    """Raise if the requested shared-memory bytes exceed the device LDS capacity.

    The check is skipped when ``arch`` is None or not in the capacity map.
    """
    if arch is None:
        return
    limit = SMEM_CAPACITY_MAP.get(arch)
    if limit is not None and allocated_bytes > limit:
        raise RuntimeError(
            f"Shared Memory Overflow: Requested {allocated_bytes} bytes, " f"but device {arch} limit is {limit} bytes."
        )


_KFD_TOPOLOGY_ROOT = "/sys/class/kfd/kfd/topology/nodes"


def _parse_kfd_properties(path: str) -> dict:
    """Parse a KFD ``properties`` file into ``{key: first_value_token}``."""
    result: dict = {}
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    result[parts[0]] = parts[1]
    except OSError:
        pass
    return result


def _llc_bytes_from_kfd_node(node_dir: str) -> int:
    """Max level-3 cache size (bytes) under one KFD topology node directory."""
    caches_dir = os.path.join(node_dir, "caches")
    try:
        entries = os.listdir(caches_dir)
    except OSError:
        return 0
    best = 0
    for entry in entries:
        props = _parse_kfd_properties(os.path.join(caches_dir, entry, "properties"))
        if props.get("level") != "3":
            continue
        try:
            size_kb = int(props["size"])
        except (KeyError, ValueError):
            continue
        best = max(best, size_kb * 1024)
    return best


@functools.lru_cache(maxsize=None)
def get_llc_cache_bytes(pci_bus_id: Optional[int] = None) -> int:
    """Last-level (Infinity) cache size in bytes for one GPU, 0 if unknown.

    Resolution order: ``FLYDSL_RUNTIME_LLC_CACHE_MB`` override, then KFD
    topology sysfs matched by PCI bus id, then 0 ("unknown, do not assume a
    size"). There is deliberately no arch-keyed fallback table: the same
    ``gfx1100`` name covers both an 80 MB and a 96 MB LLC part (7900 XT vs
    XTX), so a hardcoded guess is more dangerous than admitting the size is
    unknown.

    Cached via ``functools.lru_cache``. Pass a fresh ``pci_bus_id`` or call
    ``get_llc_cache_bytes.cache_clear()`` after changing the environment
    override or monkeypatching KFD sysfs in tests.
    """
    override_mb = env.runtime.llc_cache_mb
    if override_mb:
        return override_mb * 1024 * 1024

    try:
        node_names = os.listdir(_KFD_TOPOLOGY_ROOT)
    except OSError:
        return 0

    candidates = []
    for name in node_names:
        node_dir = os.path.join(_KFD_TOPOLOGY_ROOT, name)
        props = _parse_kfd_properties(os.path.join(node_dir, "properties"))
        loc = props.get("location_id")
        if loc is None:
            continue
        try:
            loc_int = int(loc)
        except ValueError:
            continue
        if loc_int == 0:
            continue  # the CPU node
        candidates.append((loc_int >> 8, node_dir))

    if pci_bus_id is None:
        if len(candidates) != 1:
            return 0
        node_dir = candidates[0][1]
    else:
        matches = [d for bus, d in candidates if bus == pci_bus_id]
        if len(matches) != 1:
            return 0
        node_dir = matches[0]

    return _llc_bytes_from_kfd_node(node_dir)
