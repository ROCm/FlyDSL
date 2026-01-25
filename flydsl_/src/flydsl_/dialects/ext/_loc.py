"""Shared location utilities for FLIR/Python wrappers.

We want IR dumps to be debuggable, but also avoid paying stack-inspection costs
in normal (non-dump) runs. This module centralizes the "default loc" policy so
individual dialect wrappers don't duplicate it.
"""

from __future__ import annotations

import os
from typing import Optional

from _mlir.ir import Location


def maybe_default_loc(loc: Optional[Location]) -> Optional[Location]:
    """Return a better default `loc` when IR dumping is enabled.

    Behavior:
    - If `loc` is provided and not `unknown`, return it.
    - If dumping is disabled, return `loc` unchanged (usually None).
    - If dumping is enabled, prefer user call-site (`get_user_code_loc()`),
      otherwise fall back to `Location.current`, otherwise keep None.
    """

    # Treat explicit unknown the same as missing.
    try:
        if loc is not None and str(loc) == "loc(unknown)":
            loc = None
    except Exception:
        pass

    if loc is not None:
        return loc

    # Only do stack inspection when debugging/dumping.
    dump_enabled = os.environ.get("FLIR_DUMP_IR", "0") == "1" or os.environ.get("FLIR_DUMP_IR", "0") == "1"
    auto_loc = os.environ.get("FLIR_AUTO_LOC", "0") == "1" or os.environ.get("FLIR_AUTO_LOC", "0") == "1"
    if not dump_enabled and not auto_loc:
        return None

    # Prefer a file/line location pointing at user code.
    try:
        from .func import get_user_code_loc

        loc = get_user_code_loc()
    except Exception:
        loc = None

    try:
        if loc is not None and str(loc) == "loc(unknown)":
            loc = None
    except Exception:
        pass

    if loc is None:
        try:
            loc = Location.current
        except Exception:
            loc = None

    return loc


