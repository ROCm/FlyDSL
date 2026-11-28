# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

from ..._mlir.dialects.fly_nvvm import (
    CopyOpSM75_LdMatrixType,
    CopyOpSM80_CpAsyncType,
    MmaOpSM80_MmaSyncType,
)
from ..._mlir.extras import types as T


def MmaSync(m, n, k, elem_ty_ab, elem_ty_acc=None):
    """Create an SM80 ``mma.sync.aligned`` MMA op type.

    Maps to PTX ``mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32``. Only the
    16x8x16 f16->f32 shape is currently supported.
    """
    ty_ab = elem_ty_ab.ir_type if hasattr(elem_ty_ab, "ir_type") else elem_ty_ab
    if elem_ty_acc is None:
        ty_acc = T.f32()
    else:
        ty_acc = elem_ty_acc.ir_type if hasattr(elem_ty_acc, "ir_type") else elem_ty_acc
    return MmaOpSM80_MmaSyncType.get(m, n, k, ty_ab, ty_ab, ty_acc)


def CpAsync(bit_size):
    """Create an SM80 cp.async.shared.global copy atom (global -> shared, async)."""
    return CopyOpSM80_CpAsyncType.get(bit_size)


CpAsync32b = lambda: CopyOpSM80_CpAsyncType.get(32)
CpAsync64b = lambda: CopyOpSM80_CpAsyncType.get(64)
CpAsync128b = lambda: CopyOpSM80_CpAsyncType.get(128)


def LdMatrix(num, trans=False):
    """Create an SM75+ ldmatrix copy atom (shared -> register), 8x8 b16 tiles."""
    return CopyOpSM75_LdMatrixType.get(num, trans)


LdMatrixX1 = lambda trans=False: CopyOpSM75_LdMatrixType.get(1, trans)
LdMatrixX2 = lambda trans=False: CopyOpSM75_LdMatrixType.get(2, trans)
LdMatrixX4 = lambda trans=False: CopyOpSM75_LdMatrixType.get(4, trans)
