# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from ..._mlir.dialects.fly_rocdl import CopyOpCDNA4LdsReadTransposeType, MmaOpCDNA4_MFMAScaleType
from ..._mlir.extras import types as T


def LDSReadTrans(trans_granularity, bit_size):
    """Create a GFX950 LDS read-transpose copy atom (ds_read_tr series)."""
    return CopyOpCDNA4LdsReadTransposeType.get(trans_granularity, bit_size)


LDSReadTrans4_64b = lambda: CopyOpCDNA4LdsReadTransposeType.get(4, 64)
LDSReadTrans8_64b = lambda: CopyOpCDNA4LdsReadTransposeType.get(8, 64)
LDSReadTrans6_96b = lambda: CopyOpCDNA4LdsReadTransposeType.get(6, 96)
LDSReadTrans16_64b = lambda: CopyOpCDNA4LdsReadTransposeType.get(16, 64)


def MFMA_Scale(m, n, k, elem_ty_a, elem_ty_b=None, elem_ty_acc=None, *, opsel_a=0, opsel_b=0):
    """Create a CDNA4 scaled MFMA atom (mfma.scale.f32.*.f8f6f4).

    Current atom state:
    - `scale_a` (`i32`), default zero
    - `scale_b` (`i32`), default zero
    """
    ty_a = elem_ty_a.ir_type if hasattr(elem_ty_a, "ir_type") else elem_ty_a
    if elem_ty_b is None:
        ty_b = ty_a
    else:
        ty_b = elem_ty_b.ir_type if hasattr(elem_ty_b, "ir_type") else elem_ty_b
    if elem_ty_acc is None:
        ty_acc = T.f32()
    else:
        ty_acc = elem_ty_acc.ir_type if hasattr(elem_ty_acc, "ir_type") else elem_ty_acc
    return MmaOpCDNA4_MFMAScaleType.get(m, n, k, ty_a, ty_b, ty_acc, opsel_a=opsel_a, opsel_b=opsel_b)
