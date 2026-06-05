# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Layout/atom-API helpers for the gfx950 dual-wave flash-attention kernel.

These wrap FlyDSL atoms in the style of ``kernels/fp8_gemm_utils.py`` so the
hand-built ``flash_attn_gfx950.py`` pipeline can express its primitives through
the layout/atom API instead of raw ROCDL intrinsics, while keeping the generated
ISA bit-identical.
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import fly as fly_dialect


class FaMfma32x32x16:
    """``mfma_f32_32x32x16_{bf16,f16}`` expressed via the layout MMA atom.

    Bit-identical to a raw ``rocdl.mfma_f32_32x32x16_{bf16,f16}`` call: the
    non-scaled CDNA3 MFMA atom lowers to the same intrinsic. Inputs ``a`` / ``b``
    are the packed v8 operands and ``c`` the v16f32 accumulator; the result is a
    new v16f32 accumulator.
    """

    def __init__(self, elem_dtype, acc_type):
        self.atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, elem_dtype))
        self.acc_type = acc_type

    def acc(self, a, b, c):
        return fly_dialect.mma_atom_call_ssa([self.acc_type], self.atom, a, b, c)
