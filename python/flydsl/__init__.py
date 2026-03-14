import ctypes
import os

_BASE_VERSION = "0.1.0"


# Workaround: resolve FFM simulator "LLVM ERROR: Option 'greedy' already exists!"
def _maybe_preload_system_comgr() -> None:
    disable = os.environ.get("FLYDSL_DISABLE_COMGR_PRELOAD", "").strip().lower()
    if disable in {"1", "true", "yes", "on"}:
        return

    model_path = os.environ.get("GFX1250_MODEL_PATH", "")
    hsa_model_lib = os.environ.get("HSA_MODEL_LIB", "")
    in_ffm_session = ("ffm-lite" in hsa_model_lib) or ("ffmlite" in model_path)
    if not in_ffm_session:
        return

    system_comgr = os.environ.get(
        "FLYDSL_COMGR_PRELOAD_PATH", "/opt/rocm/lib/libamd_comgr.so.3"
    )
    sim_comgr = os.path.join(model_path, "rocm", "libamd_comgr.so.3")
    if not (os.path.exists(system_comgr) and os.path.exists(sim_comgr)):
        return

    mode = getattr(os, "RTLD_NOW", 0) | getattr(os, "RTLD_GLOBAL", 0)
    try:
        ctypes.CDLL(system_comgr, mode=mode)
    except OSError:
        # Keep import robust if the host ROCm stack differs.
        pass


_maybe_preload_system_comgr()

try:
    from ._version import __version__
except ImportError:
    __version__ = _BASE_VERSION
