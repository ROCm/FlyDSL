# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import functools
import os
import subprocess
from typing import Optional

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
# gfx120x family (RDNA4: gfx1200/gfx1201/gfx1202/…) shares 64 KiB LDS.
_SMEM_GFX120X = 65536
SMEM_CAPACITY_MAP = {
    "gfx942": 65536,
    "gfx950": 163840,
    "gfx1100": 65536,
    "gfx1151": 65536,
    "gfx1200": _SMEM_GFX120X,
    "gfx1201": _SMEM_GFX120X,
    "gfx1202": _SMEM_GFX120X,
    "gfx120x": _SMEM_GFX120X,
    "gfx1250": 327680,
}


def _smem_capacity_for_arch(arch: str):
    """Return LDS capacity for ``arch``, or None if unknown.

    An exact ``SMEM_CAPACITY_MAP`` key wins, including for gfx942, gfx950,
    gfx11, and gfx1250. Strings that are not exact keys stay unknown, matching
    main, unless they are a gfx120x chip (``gfx1203``, ``gfx1201:xnack-``).
    Only that family uses the shared 64 KiB value.
    """
    if not arch:
        return None
    if arch in SMEM_CAPACITY_MAP:
        return SMEM_CAPACITY_MAP[arch]
    key = arch.lower().split(":")[0]
    if key.startswith("gfx120"):
        return SMEM_CAPACITY_MAP.get(key, _SMEM_GFX120X)
    return None


def check_smem_capacity(allocated_bytes: int, arch: str = None):
    """Raise if the requested shared-memory bytes exceed the device LDS capacity.

    The check is skipped when ``arch`` is None or not in the capacity map /
    gfx120x family fallback.
    """
    if arch is None:
        return
    limit = _smem_capacity_for_arch(arch)
    if limit is not None and allocated_bytes > limit:
        raise RuntimeError(
            f"Shared Memory Overflow: Requested {allocated_bytes} bytes, " f"but device {arch} limit is {limit} bytes."
        )


def get_gcn_arch(device=None) -> str:
    """Return the GCN arch of ``device`` (for example ``gfx1201``), or ``""``.

    Never raises. This is the tensor-device check. ``get_rocm_arch`` is the
    process arch and is a different function.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001
        return ""
    if device is None:
        if not torch.cuda.is_available():
            return ""
        device = torch.device("cuda")
    try:
        arch = torch.cuda.get_device_properties(device).gcnArchName or ""
    except Exception:  # noqa: BLE001
        return ""
    return str(arch).lower().split(":")[0]


def is_gfx120x_arch(arch: str | None) -> bool:
    """True when ``arch`` starts with ``gfx120``. Never raises.

    gfx1250 does not match. ``is_rdna_arch`` is the wider RDNA check.
    """
    if not arch:
        return False
    return str(arch).lower().split(":")[0].startswith("gfx120")


def is_gfx120x(device=None) -> bool:
    """True when ``device`` is gfx120x. Never raises."""
    return is_gfx120x_arch(get_gcn_arch(device))


def require_gfx120x(device=None, *, what: str = "this gfx120x kernel") -> None:
    """Raise ``ValueError`` unless ``device`` is gfx120x.

    Shared routers must use ``is_gfx120x`` so other arches do not error.
    """
    if is_gfx120x(device):
        return
    arch = get_gcn_arch(device) or "<unknown>"
    raise ValueError(f"{what} requires gfx120x (RDNA4), got arch={arch!r}")
