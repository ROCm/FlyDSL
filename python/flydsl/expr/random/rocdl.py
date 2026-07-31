# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""ROCDL-specific overrides for random algorithms."""

from ..._mlir.dialects import llvm
from ...compiler.backends import get_backend
from ..numeric import Boolean, Float32, Uint32
from ..typing import BFloat16x2
from . import universal

__all__ = [
    "cvt_f32_to_bf16_sr",
]


def cvt_f32_to_bf16_sr(x, rand):
    """Convert with one ``v_cvt_sr_bf16_f32`` where the architecture has it.

    Only gfx950 does this.
    """
    if get_backend().target.arch not in ("gfx950",):
        return universal.cvt_f32_to_bf16_sr(x, rand)

    # The result is a tied 2xbf16: word_sel picks the half that gets written and
    # the other half keeps the old value, undefined here because only half 0 is
    # read back. An undef old value also keeps the tie free of an extra v_mov.
    return BFloat16x2(
        llvm.call_intrinsic(
            BFloat16x2.ir_type,
            "llvm.amdgcn.cvt.sr.bf16.f32",
            [
                llvm.mlir_undef(BFloat16x2.ir_type),
                Float32(x).ir_value(),
                Uint32(rand).ir_value(),
                Boolean(0).ir_value(),  # word_sel
            ],
            [],
            [],
        )
    )[0]
