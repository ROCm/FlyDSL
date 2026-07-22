# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import functools
import os
import subprocess
from typing import Optional


def normalize_rocm_arch(value: str, *, source: str = "ROCm architecture") -> str:
    """Normalize ``gfxNNN`` or ``major.minor.step`` into a base gfx name."""
    value = value.strip().lower().split(":", 1)[0]
    if value.startswith("gfx") and len(value) > 3 and all(ch.isalnum() or ch == "-" for ch in value[3:]):
        return value
    parts = value.split(".")
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        return f"gfx{''.join(parts)}"
    raise ValueError(f"Invalid {source} {value!r}; expected gfxNNN or major.minor.step")


def get_rocm_arch_override() -> Optional[str]:
    """Return the explicit ROCm target using the canonical precedence."""
    for name in ("ARCH", "FLYDSL_GPU_ARCH", "HSA_OVERRIDE_GFX_VERSION"):
        value = os.environ.get(name)
        if value:
            return normalize_rocm_arch(value, source=name)
    return None


def get_rocm_arch(device_id: Optional[int] = None) -> str:
    """Resolve one logical HIP device's architecture without a fallback ISA."""
    override = get_rocm_arch_override()
    if override:
        return override

    from .device_runtime import get_device_runtime

    runtime = get_device_runtime()
    if device_id is None:
        device_id = runtime.current_device_id()
    return runtime.device_arch(device_id)


@functools.lru_cache(maxsize=None)
def get_rocm_device_count() -> int:
    """Best-effort ROCm visible GPU count via ``rocm_agent_enumerator`` (standard ROCm tool).

    Returns 0 when the tool is unavailable or no discrete GPU agents are
    reported. Runtime target resolution uses HIP instead of this helper.
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
