import functools
import os
import subprocess
from typing import Optional


def _arch_from_rocm_agent_enumerator() -> Optional[str]:
    """Query rocm_agent_enumerator (standard ROCm tool) for the first GPU arch."""
    try:
        out = subprocess.check_output(
            ["rocm_agent_enumerator", "-name"],
            text=True,
            timeout=5,
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
def get_rocm_arch() -> str:
    """Best-effort ROCm GPU arch string (e.g. 'gfx942') without torch."""
    env = (os.environ.get("FLYDSL_GPU_ARCH")
        or os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    )
    if env:
        if env.startswith("gfx"):
            return env
        if env.count(".") == 2:
            parts = env.split(".")
            return f"gfx{parts[0]}{parts[1]}{parts[2]}"

    arch = _arch_from_rocm_agent_enumerator()
    if arch:
        return arch.split(":", 1)[0]

    return "gfx942"
