# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import functools
import os
import subprocess
import sys
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


def _arch_from_rocminfo() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["rocminfo"],
            text=True,
            timeout=_ROCM_AGENT_TIMEOUT_S,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            _, _, name = line.partition("Name:")
            name = name.strip()
            if name.startswith("gfx") and name != "gfx000":
                return name
    except (OSError, subprocess.SubprocessError, UnicodeError):
        pass
    return None


def _arch_from_torch() -> str:
    """Device 0's arch from a ROCm torch; "" when there is no torch or no GPU."""
    try:
        import torch

        return torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    except Exception:
        return ""


@functools.lru_cache(maxsize=None)
def _arch_from_hardware() -> str:
    """Cache hardware detection, including failures, to avoid repeated slow probes."""
    if sys.platform == "win32":
        # Windows ROCm ships no rocm_agent_enumerator, and guessing gfx942 there
        # would compile for the wrong GPU.
        return _arch_from_torch()
    arch = _arch_from_rocm_agent_enumerator() or _arch_from_rocminfo()
    if arch:
        return arch.split(":", 1)[0]
    return ""


def get_rocm_arch() -> str:
    """ROCm GPU arch string, always lower-cased (e.g. 'gfx942').

    Lower-casing happens here so every caller can compare against lower-case
    literals without normalising first; ROCm itself only ever emits lower-case
    names, so this only affects hand-set environment overrides.

    Raises RuntimeError if neither an explicit override nor hardware supplies a target.
    """
    env = os.environ.get("FLYDSL_GPU_ARCH") or os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if env:
        env = env.lower()
        if env.startswith("gfx"):
            return env
        if env.count(".") == 2:
            parts = env.split(".")
            return f"gfx{parts[0]}{parts[1]}{parts[2]}"

    arch = _arch_from_hardware()
    if not arch:
        tried = "torch.cuda" if sys.platform == "win32" else "rocm_agent_enumerator or rocminfo"
        raise RuntimeError(
            f"Unable to detect a ROCm GPU architecture with {tried}. "
            "Set FLYDSL_GPU_ARCH to the intended gfx target when hardware detection is unavailable."
        )
    return arch.lower()


@functools.lru_cache(maxsize=None)
def get_rocm_device_count() -> int:
    """Best-effort physical GPU count via ``rocm_agent_enumerator`` (standard ROCm tool).

    The default query avoids duplicate ISAs but may ignore runtime visibility filters.
    Returns 0 when the tool is unavailable or no discrete GPU agents are reported.
    Windows has no such tool: there HIP's ``hipGetDeviceCount`` answers.
    """
    if sys.platform == "win32":
        from .device_runtime.rocm import _hip_device_count

        return _hip_device_count()
    try:
        out = subprocess.check_output(
            ["rocm_agent_enumerator"],
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
    "gfx1250-strict": 327680,
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
    result = {}
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
    """Total size (bytes) of the highest cache level >= 2 under one KFD node.

    Picks L3 where present (CDNA3/4, RDNA), otherwise L2. KFD may list one entry
    per cache instance; entries at a level are summed, as ROCr does for
    ``HSA_AGENT_INFO_CACHE_SIZE``.
    """
    caches_dir = os.path.join(node_dir, "caches")
    try:
        entries = os.listdir(caches_dir)
    except OSError:
        return 0
    totals: dict = {}
    for entry in entries:
        props = _parse_kfd_properties(os.path.join(caches_dir, entry, "properties"))
        try:
            level, size_kb = int(props["level"]), int(props["size"])
        except (KeyError, ValueError):
            continue
        if level >= 2:
            totals[level] = totals.get(level, 0) + size_kb * 1024
    return totals[max(totals)] if totals else 0


@functools.lru_cache(maxsize=None)
def _kfd_llc_bytes(pci_location: Optional[tuple]) -> int:
    """LLC bytes of the KFD GPU node at ``(pci_domain, location_id)``, 0 if unknown.

    With ``pci_location`` None, uses the GPU node only if it is the sole one.
    """
    try:
        node_names = os.listdir(_KFD_TOPOLOGY_ROOT)
    except OSError:
        return 0

    candidates = []
    for name in node_names:
        node_dir = os.path.join(_KFD_TOPOLOGY_ROOT, name)
        props = _parse_kfd_properties(os.path.join(node_dir, "properties"))
        try:
            location = int(props["domain"]), int(props["location_id"])
        except (KeyError, ValueError):
            continue
        if location[1] == 0:
            continue  # the CPU node
        candidates.append((location, node_dir))

    if pci_location is not None:
        candidates = [c for c in candidates if c[0] == pci_location]
    if len(candidates) != 1:
        return 0
    return _llc_bytes_from_kfd_node(candidates[0][1])


def get_llc_cache_bytes() -> int:
    """Last-level cache size in bytes of the current HIP device, 0 if unknown.

    ``FLYDSL_RUNTIME_LLC_CACHE_MB`` overrides detection. Otherwise the size is
    read from KFD topology sysfs for the node at the current device's PCI address,
    which follows ``hipSetDevice`` and HIP/ROCR visibility; without HIP only a
    single-GPU host resolves. There is no arch-keyed table on purpose:
    ``gfx1100`` alone covers 80 MB and 96 MB parts, so guessing is worse than
    reporting unknown.
    """
    override_mb = env.runtime.llc_cache_mb
    if override_mb:
        return override_mb * 1024 * 1024

    from .device_runtime.rocm import _hip_current_pci_location

    return _kfd_llc_bytes(_hip_current_pci_location())
