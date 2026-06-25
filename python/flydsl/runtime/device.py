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
def _arch_from_hardware() -> Optional[str]:
    """Cached hardware detection (rocm_agent_enumerator is slow)."""
    arch = _arch_from_rocm_agent_enumerator()
    if arch:
        return arch.split(":", 1)[0]
    return None


def _parse_arch_override(value: Optional[str]) -> Optional[str]:
    """Accept either gfx-format or dotted HSA override values."""
    if not value:
        return None
    value = value.strip()
    if value.startswith("gfx"):
        if len(value) == 3 or not value[3:].isdigit():
            raise ValueError(f"Invalid FLYDSL_GPU_ARCH value: {value}")
        return value
    if value.count(".") == 2:
        parts = value.split(".")
        if all(part.isdigit() for part in parts):
            return f"gfx{''.join(parts)}"
        raise ValueError(f"Invalid HSA_OVERRIDE_GFX_VERSION value: {value}")
    return None


def _bool_env_var(name: str) -> bool:
    """Boolean env var parser compatible with project conventions."""
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def get_rocm_arch() -> str:
    """Resolved ROCm GPU architecture string (e.g. 'gfx942')."""
    env = os.environ.get("FLYDSL_GPU_ARCH") or os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if env is not None:
        arch = _parse_arch_override(env)
        if arch:
            return arch
        raise RuntimeError(
            f"FLYDSL_GPU_ARCH/HSA_OVERRIDE_GFX_VERSION is set to an invalid value: {env!r}. "
            "Expected values like 'gfx942' or '9.4.2'."
        )

    hardware_arch = _arch_from_hardware()
    if hardware_arch:
        return hardware_arch

    if _bool_env_var("FLYDSL_GPU_ARCH_FALLBACK"):
        return "gfx942"

    raise RuntimeError(
        "Unable to detect ROCm GPU architecture. "
        "Set FLYDSL_GPU_ARCH (or HSA_OVERRIDE_GFX_VERSION), "
        "or set FLYDSL_GPU_ARCH_FALLBACK=1 to use the legacy fallback."
    )


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
    """Check if architecture is RDNA-based (gfx10/11/12, wave32).

    This is the single source of truth for CDNA vs RDNA classification.
    RDNA architectures use wave32 and have different buffer descriptor flags.

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
