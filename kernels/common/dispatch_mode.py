# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Host dispatch mode override for gfx120x size-gated / ``*_auto`` entry points.

``FLYDSL_DISPATCH_MODE`` (default ``auto``):

* ``auto`` — use measured size-dispatch breakpoints (R9700 / HIP 7.17 stamp;
  constants in ``kernels/common/gfx120x_autotune_tables.py``).
* ``force_flydsl`` — always take the FlyDSL kernel path (skip HIP
  fallbacks; among FlyDSL alternatives prefer the native/WMMA path).
* ``force_hip`` — route to a pure-PyTorch reference path (no FlyDSL WMMA;
  avoids editing breakpoint constants on other RDNA4/ROCm stacks).

Invalid values raise ``ValueError`` at first read.
"""

import os
from typing import Literal

DispatchMode = Literal["auto", "force_flydsl", "force_hip"]

_VALID = ("auto", "force_flydsl", "force_hip")
_ENV = "FLYDSL_DISPATCH_MODE"


def get_dispatch_mode() -> DispatchMode:
    """Return normalized ``FLYDSL_DISPATCH_MODE`` (default ``auto``)."""
    raw = os.environ.get(_ENV, "auto")
    if raw is None or raw == "":
        return "auto"
    mode = str(raw).strip().lower()
    if mode not in _VALID:
        raise ValueError(f"{_ENV}={raw!r} invalid; expected one of {_VALID}")
    return mode  # type: ignore[return-value]


def dispatch_mode_doc() -> dict:
    """Structured description of dispatch-mode env override."""
    return {
        "env": _ENV,
        "default": "auto",
        "values": list(_VALID),
        "measured_hw": "gfx1201 R9700",
        "measured_hip_stamp": "7.17.26374",
        "note": (
            "Thresholds are R9700/HIP 7.17 measured; use force_flydsl|force_hip "
            "to bypass size gates without editing constants on other RDNA4/ROCm."
        ),
    }


__all__ = ["DispatchMode", "get_dispatch_mode", "dispatch_mode_doc", "_ENV"]
