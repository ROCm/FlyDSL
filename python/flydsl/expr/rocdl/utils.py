# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

from ..._mlir import ir
from ..numeric import Numeric


def normalize_s_waitcnt_field(name, value, maximum):
    """Coerce a wait-counter argument to a static Python ``int``.

    ``None`` means "do not wait on this counter" and maps to ``maximum``,
    which is the encoding the hardware reads as "already satisfied".

    Wait counters are encoded into an instruction's immediate field, so the
    value has to be known at compile time; a run-time ``Integer`` is rejected
    rather than silently materialised as a constant.

    An ``ir.IntegerAttr`` is also accepted: the ODS-generated ROCDL builders
    these wrappers shadow take ``Union[int, IntegerAttr]``, so callers written
    against the raw builders keep working.
    """
    if value is None:
        return maximum

    if isinstance(value, ir.IntegerAttr):
        value = value.value

    if isinstance(value, Numeric):
        if not value.is_static():
            raise TypeError(f"{name} must be a static Python int or Integer, got a run-time value")
        value = value.value

    if not isinstance(value, int):
        raise TypeError(f"{name} must be a static Python int or Integer, got {type(value).__name__}")

    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must be in [0, {maximum}] on this target, got {value}")

    return int(value)
