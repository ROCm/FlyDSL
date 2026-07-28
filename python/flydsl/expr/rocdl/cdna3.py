# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

from ..._mlir.dialects import rocdl as mlir_rocdl
from .utils import normalize_s_waitcnt_field


def s_waitcnt(vmcnt=None, lgkmcnt=None, expcnt=None):
    """
    Emit a CDNA3/GFX942-encoded s_waitcnt operation.

    vmcnt: split across [3:0] and [15:14]
    expcnt: [6:4]
    lgkmcnt: [11:8]
    """
    vmcnt = normalize_s_waitcnt_field("vmcnt", vmcnt, 63)
    expcnt = normalize_s_waitcnt_field("expcnt", expcnt, 7)
    lgkmcnt = normalize_s_waitcnt_field("lgkmcnt", lgkmcnt, 15)
    encoded_vmcnt = (vmcnt & 0xF) | ((vmcnt & 0x30) << 10)
    mlir_rocdl.s_waitcnt(encoded_vmcnt | (lgkmcnt << 8) | (expcnt << 4))
