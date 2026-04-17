# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Runtime compatibility shims loaded at import time.

Kept separate from ``__init__.py`` so the workaround logic is isolated and
easy to find / disable.
"""

import ctypes
import os
import sys


def _default_comgr_path() -> str:
    """Return the default path to the system ``libamd_comgr`` library."""
    if sys.platform == "win32":
        rocm = os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH", "")
        if rocm:
            return os.path.join(rocm, "bin", "amd_comgr.dll")
        return "amd_comgr.dll"
    return "/opt/rocm/lib/libamd_comgr.so.3"


def _comgr_sim_name() -> str:
    """Return the simulator-side comgr library name."""
    if sys.platform == "win32":
        return "amd_comgr.dll"
    return "libamd_comgr.so.3"


def _maybe_preload_system_comgr() -> None:
    """Pre-load system ``libamd_comgr`` to avoid duplicate-option LLVM errors.

    The FFM simulator ships its own ``libamd_comgr`` that registers the same
    LLVM command-line options as the system copy.  If both are loaded the
    process aborts with *"Option 'greedy' already exists!"*.  Loading the
    system copy first (with ``RTLD_GLOBAL``) makes the simulator copy a
    harmless no-op.

    This function is a no-op outside FFM simulator sessions.
    """
    disable = os.environ.get("FLYDSL_DISABLE_COMGR_PRELOAD", "").strip().lower()
    if disable in {"1", "true", "yes", "on"}:
        return

    model_path = os.environ.get("GFX1250_MODEL_PATH", "")
    hsa_model_lib = os.environ.get("HSA_MODEL_LIB", "")
    in_ffm_session = ("ffm-lite" in hsa_model_lib) or ("ffmlite" in model_path)
    if not in_ffm_session:
        return

    system_comgr = os.environ.get("FLYDSL_COMGR_PRELOAD_PATH", _default_comgr_path())
    sim_comgr = os.path.join(model_path, "rocm", _comgr_sim_name())
    if not (os.path.exists(system_comgr) and os.path.exists(sim_comgr)):
        return

    mode = getattr(os, "RTLD_NOW", 0) | getattr(os, "RTLD_GLOBAL", 0)
    try:
        ctypes.CDLL(system_comgr, mode=mode)
    except OSError:
        # Keep import robust if the host ROCm stack differs.
        pass
