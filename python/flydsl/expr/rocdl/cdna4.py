# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from ..._mlir.dialects.fly_rocdl import CopyOpCDNA4LdsReadTransposeType


def LDSReadTrans(trans_granularity, bit_size):
    """Create a GFX950 LDS read-transpose copy atom (ds_read_tr series)."""
    return CopyOpCDNA4LdsReadTransposeType.get(trans_granularity, bit_size)


LDSReadTrans4_64b = lambda: CopyOpCDNA4LdsReadTransposeType.get(4, 64)
LDSReadTrans8_64b = lambda: CopyOpCDNA4LdsReadTransposeType.get(8, 64)
LDSReadTrans6_96b = lambda: CopyOpCDNA4LdsReadTransposeType.get(6, 96)
LDSReadTrans16_64b = lambda: CopyOpCDNA4LdsReadTransposeType.get(16, 64)
