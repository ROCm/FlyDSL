# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

from ..._mlir import ir
from ...runtime.device import get_rocm_arch
from ..numeric import Numeric


def _check_numeric_type(what, ty, supported):
    """Check that ``ty`` is one of the DSL numeric types in ``supported``."""
    if ty not in supported:
        names = ", ".join(t.__name__ for t in supported) or "none"
        raise ValueError(f"{what} has no instruction for {ty.__name__}; supported: {names}")


def _check_mfma_shape(what, shapes, m, n, k, elem_ty_a, elem_ty_b):
    """Reject an MFMA atom with no instruction; ``shapes`` maps an (A, B) DSL type pair to its ``(M, N, K)``."""
    types = (elem_ty_a, elem_ty_b)
    name = f"({elem_ty_a.__name__}, {elem_ty_b.__name__})"
    if types not in shapes:
        supported = ", ".join(f"({a.__name__}, {b.__name__})" for a, b in shapes)
        raise ValueError(f"{what} has no instruction for {name}; supported: {supported}")
    if (m, n, k) not in shapes[types]:
        supported = ", ".join("x".join(map(str, mnk)) for mnk in shapes[types])
        raise ValueError(f"{what} has no {m}x{n}x{k} instruction for {name}; supported: {supported}")


def require_lds_dma_support(what):
    """Reject global -> LDS direct loads (``buffer_load_* ... lds``) on gfx11.

    RDNA3 / RDNA3.5 dropped the VMEM -> LDS hardware path that gfx9 and gfx10
    expose. Emitting the intrinsic anyway passes MLIR verification and then
    aborts the whole process in LLVM instruction selection with "Do not know
    how to expand this operator's operand!", so reject it while a Python
    traceback still points at the kernel line that asked for it.
    """
    # get_rocm_arch() is the compile target (FLYDSL_GPU_ARCH when set), not necessarily the
    # installed device.
    arch = get_rocm_arch()
    if not arch or not arch.startswith("gfx11"):
        return
    raise ValueError(
        f"{what} is not supported on target arch {arch!r}: gfx11 (RDNA3 / RDNA3.5) has no "
        "global -> LDS direct-load hardware. Use a buffer load into registers followed by "
        "ds_write instead."
    )


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
