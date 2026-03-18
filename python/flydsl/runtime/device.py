import functools
import os
import subprocess
from typing import Optional


def _arch_from_rocm_agent_enumerator(timeout_s: int = 5) -> Optional[str]:
    """Query rocm_agent_enumerator (standard ROCm tool) for the first GPU arch."""
    try:
        out = subprocess.check_output(
            ["rocm_agent_enumerator", "-name"],
            text=True,
            timeout=timeout_s,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            name = line.strip()
            if name.startswith("gfx") and name != "gfx000":
                return name
    except Exception:
        pass
    return None


_cached_rocm_arch: Optional[str] = None


def get_rocm_arch(timeout_s: int = 5) -> str:
    """Best-effort ROCm GPU arch string (e.g. 'gfx942')."""
    global _cached_rocm_arch
    if _cached_rocm_arch is not None:
        return _cached_rocm_arch

    env = os.environ.get("FLYDSL_GPU_ARCH") or os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if env:
        if env.startswith("gfx"):
            _cached_rocm_arch = env
            return _cached_rocm_arch
        if env.count(".") == 2:
            parts = env.split(".")
            _cached_rocm_arch = f"gfx{parts[0]}{parts[1]}{parts[2]}"
            return _cached_rocm_arch

    arch = _arch_from_rocm_agent_enumerator(timeout_s=timeout_s)
    if arch:
        _cached_rocm_arch = arch.split(":", 1)[0]
        return _cached_rocm_arch

    _cached_rocm_arch = "gfx942"
    return _cached_rocm_arch


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
    return arch.startswith("gfx10") or arch.startswith("gfx11") or arch.startswith("gfx12")
