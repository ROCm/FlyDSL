# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import functools
import os
import subprocess
from pathlib import Path
from typing import Optional

_ROCM_AGENT_TIMEOUT_S = int(os.environ.get("FLYDSL_ROCM_AGENT_TIMEOUT", "300"))


@functools.lru_cache(maxsize=None)
def get_rocm_toolkit_path() -> Optional[str]:
    """Return a directory MLIR's ROCDL backend recognizes as a toolkit.

    MLIR's gpu-module-to-binary expects ``<toolkit>/llvm/bin/ld.lld`` for
    linking and ``<toolkit>/amdgcn/bitcode`` for device libraries. The
    rocm-sdk Python wheels (``_rocm_sdk_core``) ship both, but at
    ``<sdk>/lib/llvm/bin/ld.lld`` and ``<sdk>/lib/llvm/amdgcn/bitcode``, so
    the layout doesn't directly match. We synthesize a tiny symlink-based
    shim under ``~/.flydsl/toolkit`` and return its path.

    Order of preference:
      1. ``FLYDSL_ROCM_TOOLKIT_PATH`` env var (explicit override)
      2. ``ROCM_PATH`` env var
      3. ``/opt/rocm`` if present and well-formed
      4. Synthesized shim pointing at the rocm-sdk Python wheel.
    Returns ``None`` if no toolkit can be located.
    """

    def _well_formed(root: Path) -> bool:
        return (root / "llvm" / "bin" / "ld.lld").exists() and (root / "amdgcn" / "bitcode").is_dir()

    for env_var in ("FLYDSL_ROCM_TOOLKIT_PATH", "ROCM_PATH"):
        val = os.environ.get(env_var, "").strip()
        if val and _well_formed(Path(val)):
            return val

    opt_rocm = Path("/opt/rocm")
    if _well_formed(opt_rocm):
        return str(opt_rocm)

    try:
        import _rocm_sdk_core  # type: ignore[import-not-found]
    except ImportError:
        return None

    sdk_root = Path(_rocm_sdk_core.__file__).parent
    llvm_dir = sdk_root / "lib" / "llvm"
    if not (llvm_dir / "bin" / "ld.lld").exists() or not (llvm_dir / "amdgcn" / "bitcode").is_dir():
        return None

    shim_root = Path(os.environ.get("FLYDSL_ROCM_TOOLKIT_SHIM_DIR") or (Path.home() / ".flydsl" / "toolkit"))
    shim_root.mkdir(parents=True, exist_ok=True)
    (shim_root / "llvm" / "bin").mkdir(parents=True, exist_ok=True)
    amdgcn_link = shim_root / "amdgcn"
    if not amdgcn_link.exists():
        amdgcn_link.symlink_to(llvm_dir / "amdgcn")
    # ``ld.lld`` in the rocm-sdk wheel is a tiny stub that needs to resolve
    # its own argv[0] to load companion libraries. Copying it elsewhere
    # breaks that lookup, so we drop a thin exec wrapper instead.
    wrapper = shim_root / "llvm" / "bin" / "ld.lld"
    wrapper_text = f'#!/bin/bash\nexec "{llvm_dir}/bin/ld.lld" "$@"\n'
    if not wrapper.exists() or wrapper.read_text() != wrapper_text:
        wrapper.write_text(wrapper_text)
        wrapper.chmod(0o755)
    return str(shim_root)


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
    """Best-effort ROCm GPU arch string (e.g. 'gfx942')."""
    env = os.environ.get("FLYDSL_GPU_ARCH") or os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if env:
        if env.startswith("gfx"):
            return env
        if env.count(".") == 2:
            parts = env.split(".")
            return f"gfx{parts[0]}{parts[1]}{parts[2]}"

    return _arch_from_hardware()


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
