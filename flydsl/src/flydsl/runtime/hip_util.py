"""
HIP runtime helpers used by kernels/tests.

This module must be importable even when the optional `hip` Python package is
not installed, because some tests import it before conditionally skipping.
"""

from __future__ import annotations

from typing import Any

from .device import get_rocm_arch


def get_hip_arch() -> str:
    """Return the current ROCm GPU arch string (e.g. 'gfx942')."""

    return get_rocm_arch()


def hip_check(ret: Any, msg: str | None = None) -> Any:
    """Check a HIP API return value.

    Accepts either:
    - an int error code (0 means success), or
    - a (err, value) tuple as commonly returned by hip-python.

    Returns the underlying value for tuple returns, or the original `ret` for
    non-tuple returns.
    """

    err = ret
    value = None
    if isinstance(ret, tuple) and ret:
        err = ret[0]
        value = ret[1] if len(ret) > 1 else None

    # hipSuccess is 0.
    if int(err) != 0:
        details = f"HIP call failed with error code {int(err)}"
        # Best-effort: try to translate error code to string if hip is available.
        try:
            from hip import hip as _hip  # type: ignore

            try:
                # Some hip-python versions expose hipGetErrorString
                s = _hip.hipGetErrorString(int(err))
                if isinstance(s, tuple):
                    # (err, cstr)
                    s = s[1]
                if s:
                    details += f" ({s})"
            except Exception:
                pass
        except Exception:
            pass

        if msg:
            details = f"{msg}: {details}"
        raise RuntimeError(details)

    return value if isinstance(ret, tuple) else ret


__all__ = ["hip_check", "get_hip_arch"]


