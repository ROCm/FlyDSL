# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Cross-platform helpers for Windows / Linux compatibility."""

import os
import sys

IS_WINDOWS = sys.platform == "win32"


def _ensure_junction(link: str, target: str) -> bool:
    """Create a Windows directory junction from ``link`` to ``target``. Junctions
    don't require admin (unlike symlinks). Returns True if link exists and points
    at target."""
    import subprocess

    if os.path.exists(link):
        try:
            if os.path.samefile(link, target):
                return True
        except OSError:
            pass
        # Stale or unrelated — leave it alone and assume usable
        return os.path.isdir(link)
    os.makedirs(os.path.dirname(link), exist_ok=True)
    try:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", link, target],
            check=True,
            capture_output=True,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def rocm_toolkit_path() -> str:
    """Path MLIR's gpu-module-to-binary should use as ``toolkit``.

    MLIR's ROCDL target appends ``llvm/bin/ld.lld`` (for linking) and
    ``amdgcn/bitcode`` (for OCML/OCKL bitcode) to this path.

    - Linux ROCm at ``/opt/rocm`` has both at the standard relative locations →
      empty/default toolkit works.
    - Windows TheRock SDK has ``ld.lld.exe`` at ``<ROCM>/lib/llvm/bin/`` and
      bitcode at ``<ROCM>/lib/llvm/amdgcn/bitcode/``. No single toolkit path
      satisfies both MLIR-expected sub-paths, so we stage a directory under the
      user's LocalAppData with directory junctions that unify the layout:
          <staging>/llvm    → <ROCM>/lib/llvm
          <staging>/amdgcn  → <ROCM>/lib/llvm/amdgcn

    Returns an empty string if no override is needed or staging fails.
    """
    if not IS_WINDOWS:
        return ""
    rocm = os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH")
    if not rocm:
        return ""
    llvm_dir = os.path.join(rocm, "lib", "llvm")
    ld_lld = os.path.join(llvm_dir, "bin", "ld.lld.exe")
    amdgcn = os.path.join(llvm_dir, "amdgcn")
    if not (os.path.isfile(ld_lld) and os.path.isdir(amdgcn)):
        return ""

    cache_root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    staging = os.path.join(cache_root, "flydsl", "rocm_toolkit")
    ok_llvm = _ensure_junction(os.path.join(staging, "llvm"), llvm_dir)
    ok_amdgcn = _ensure_junction(os.path.join(staging, "amdgcn"), amdgcn)
    if ok_llvm and ok_amdgcn:
        # MLIR pass-option parser is brace/colon sensitive — use forward slashes.
        return staging.replace("\\", "/")

    # Fallback: lld still works from <ROCM>/lib (OCML won't resolve).
    candidate = os.path.join(rocm, "lib")
    if os.path.isfile(os.path.join(candidate, "llvm", "bin", "ld.lld.exe")):
        return candidate.replace("\\", "/")
    return ""


def shared_lib_ext() -> str:
    """Return the native shared-library file extension for the current platform."""
    if IS_WINDOWS:
        return ".dll"
    return ".so"


def shared_lib_name(basename: str) -> str:
    """Convert a Linux-style shared library name to the platform equivalent.

    Examples (Linux → Windows):
        libfoo.so        → foo.dll
        libfoo.so.3      → foo.dll
        _bar*.so         → _bar*.pyd   (Python extension)
    """
    if not IS_WINDOWS:
        return basename

    # Python extension modules: _name*.so → _name*.pyd
    if basename.startswith("_") and basename.endswith(".so"):
        return basename[:-3] + ".pyd"

    # Versioned sonames: libfoo.so.3 → foo.dll
    name = basename
    if ".so." in name:
        name = name[: name.index(".so.")]
    elif name.endswith(".so"):
        name = name[:-3]

    # Drop lib prefix (Windows convention)
    if name.startswith("lib"):
        name = name[3:]

    return name + ".dll"


def shared_lib_glob(pattern: str) -> str:
    """Convert a Linux glob pattern for shared libraries to the platform equivalent.

    Examples (Linux → Windows):
        _mlirDialectsFly*.so   → _mlirDialectsFly*.pyd
        libFly*.so             → Fly*.dll
        libfoo.so              → foo.dll
    """
    if not IS_WINDOWS:
        return pattern

    # Python extension globs: _name*.so → _name*.pyd
    if pattern.startswith("_") and pattern.endswith(".so"):
        return pattern[:-3] + ".pyd"

    # lib*.so globs
    name = pattern
    if name.endswith(".so"):
        name = name[:-3]
    if name.startswith("lib"):
        name = name[3:]
    return name + ".dll"
