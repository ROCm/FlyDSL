# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""ROCm / HIP device runtime (default FlyDSL GPU stack)."""

from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import shutil
import threading
from pathlib import Path
from typing import ClassVar

from ..device import get_rocm_device_count
from .base import DeviceRuntime

# Cached HIP runtime handle (``libamdhip64``); cached once.
_HIP_LIB = None
_HIP_LIB_TRIED = False


def _hip_library_names() -> tuple:
    if os.name != "nt":
        return ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5")
    # No sonames on Windows: open the HIP DLL the JIT runtime binds to.
    try:
        return (_load_hip_runtime(),)
    except OSError:
        return ()


def _hip_get_device() -> int:
    """Active HIP device index via ``hipGetDevice``."""
    global _HIP_LIB, _HIP_LIB_TRIED
    if not _HIP_LIB_TRIED:
        _HIP_LIB_TRIED = True
        for soname in _hip_library_names():
            try:
                lib = ctypes.CDLL(soname)
                lib.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
                lib.hipGetDevice.restype = ctypes.c_int
                _HIP_LIB = lib
                break
            except OSError:
                continue
    if _HIP_LIB is None:
        return 0
    dev = ctypes.c_int(0)
    try:
        if _HIP_LIB.hipGetDevice(ctypes.byref(dev)) == 0:
            return int(dev.value)
    except Exception:
        pass
    return 0


def _hip_device_count() -> int:
    """Visible HIP devices via ``hipGetDeviceCount``; 0 without HIP or a device.

    HIP applies HIP_VISIBLE_DEVICES. The library is the one ``_hip_get_device`` loads.
    """
    _hip_get_device()
    if _HIP_LIB is None:
        return 0
    count = ctypes.c_int(0)
    try:
        if _HIP_LIB.hipGetDeviceCount(ctypes.byref(count)) == 0:
            return int(count.value)
    except Exception:
        pass
    return 0


class RocmDeviceRuntime(DeviceRuntime):
    """HIP-based runtime; matches compile backend ``rocm``.

    ``device_count()`` delegates to ``get_rocm_device_count`` in ``device.py``
    (``rocm_agent_enumerator``; ``hipGetDeviceCount`` on Windows). On Linux this is
    the *physical* agent count, which ignores runtime visibility filters;
    ``current_device_id()`` queries HIP ``hipGetDevice`` directly.
    """

    kind: ClassVar[str] = "rocm"

    def device_count(self) -> int:
        return get_rocm_device_count()

    def current_device_id(self) -> int:
        return _hip_get_device()


# ---------------------------------------------------------------------------
# Windows: where the HIP runtime and the ROCDL link toolkit come from. Every
# lookup scans the same roots, in this order: ROCM_PATH, the ROCm SDK installed
# in this interpreter (TheRock's rocm[libraries]), HIP_PATH. AMD's HIP SDK
# installer sets HIP_PATH machine-wide, so it comes last and cannot shadow the
# SDK installed next to FlyDSL.
# ---------------------------------------------------------------------------

# The name fly_jit_runtime.dll imports, fixed by the SDK's amdhip64.lib.
_HIP_DLL = "amdhip64_7.dll"
_LLD = "ld.lld.exe"
_INSTALL_HINT = (
    'install the ROCm SDK into this Python environment (pip install "rocm[libraries]" from the '
    "TheRock wheel index), or set ROCM_PATH to a ROCm installation"
)
_loaded = []


def _therock_root() -> Path | None:
    try:
        import rocm_sdk

        return rocm_sdk.find_libraries("amdhip64")[0].parents[1]
    except Exception:
        return None


def _rocm_roots() -> list[Path]:
    roots = [os.environ.get("ROCM_PATH"), _therock_root(), os.environ.get("HIP_PATH")]
    return [Path(r) for r in roots if r]


@functools.lru_cache(maxsize=1)
def _load_hip_runtime() -> str:
    """Make sure one HIP runtime is loaded; return the name or path that opens it.

    A HIP DLL already in the process (torch loads one) is the one to use: loading
    another copy by path would leave two runtimes, and FlyDSL would query the wrong
    one. Otherwise load the SDK's copy by full path. The display driver puts its own
    amdhip64_7.dll in System32, where a load by bare name would find it, and a
    loaded module is what every later by-name import binds to.
    """
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetModuleHandleW.restype = ctypes.c_void_p
    if kernel32.GetModuleHandleW(_HIP_DLL):
        return _HIP_DLL
    for root in _rocm_roots():
        hip = root / "bin" / _HIP_DLL
        if hip.is_file():
            _loaded.append(ctypes.CDLL(str(hip)))
            return str(hip)
    raise OSError(f"{_HIP_DLL} not found: {_INSTALL_HINT}")


def _preload_runtime_libs(paths) -> None:
    """Load the ExecutionEngine's shared libraries before it does.

    The engine loads them by full path with the default search order, which skips a
    DLL's own folder (mlir_c_runner_utils.dll imports mlir_float16_utils.dll from
    there) and would take System32's HIP for fly_jit_runtime.dll. ctypes searches
    the DLL's folder, and the engine then gets the modules already loaded.
    """
    _load_hip_runtime()
    _loaded.extend(ctypes.CDLL(str(p)) for p in paths)


@functools.lru_cache(maxsize=1)
def _rocm_toolkit_root() -> Path:
    """A root with the layout ROCDL links against: llvm/bin/ld.lld and amdgcn/bitcode.

    TheRock keeps both under lib/llvm, so its files are staged into that layout
    under ~/.flydsl, once per SDK.
    """
    for root in _rocm_roots():
        if (root / "llvm" / "bin" / _LLD).is_file() and (root / "amdgcn" / "bitcode").is_dir():
            return root
        llvm = root / "lib" / "llvm"
        if (llvm / "bin" / _LLD).is_file() and (llvm / "amdgcn" / "bitcode").is_dir():
            key = hashlib.sha256(os.path.normcase(os.path.abspath(root)).encode()).hexdigest()[:16]
            return _stage_toolkit(llvm, Path.home() / ".flydsl" / "rocm-toolkit" / key)
    raise RuntimeError(f"ROCDL needs ld.lld and the AMDGPU device bitcode; none found: {_INSTALL_HINT}")


def _stage_toolkit(llvm: Path, dest: Path) -> Path:
    """Lay out ``llvm``'s ld.lld and device bitcode as dest/llvm/bin and dest/amdgcn/bitcode.

    Real files only (hard links, or copies across volumes): processes started from an
    OpenSSH session refuse to traverse user-created junctions (WinError 448), and
    symlinks need a privilege.
    """
    _place(llvm / "bin" / _LLD, dest / "llvm" / "bin" / _LLD)
    for bitcode in sorted((llvm / "amdgcn" / "bitcode").glob("*.bc")):
        _place(bitcode, dest / "amdgcn" / "bitcode" / bitcode.name)
    return dest


def _place(src: Path, dst: Path) -> None:
    want = src.stat()
    try:
        have = dst.stat()
        # Reinstalling or upgrading the SDK writes new files, so size or mtime moves.
        if (have.st_size, have.st_mtime_ns) == (want.st_size, want.st_mtime_ns):
            return
    except FileNotFoundError:
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        _publish(src, tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def _publish(src: Path, tmp: Path, dst: Path) -> None:
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copy2(src, tmp)
    # Another process may stage the same file: each one swaps in a complete file.
    # On POSIX, renaming over another link to the same file does nothing and leaves
    # tmp behind, which the caller removes.
    os.replace(tmp, dst)
