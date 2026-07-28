# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

from ..._mlir.dialects import rocdl as mlir_rocdl
from .utils import normalize_s_waitcnt_field


def s_waitcnt(vmcnt=None, lgkmcnt=None, expcnt=None):
    """Emit a RDNA4/GFX120-encoded s_waitcnt operation.

    expcnt: [2:0]
    lgkmcnt: [9:4]
    vmcnt: [15:10]
    """
    vmcnt = normalize_s_waitcnt_field("vmcnt", vmcnt, 63)
    lgkmcnt = normalize_s_waitcnt_field("lgkmcnt", lgkmcnt, 63)
    expcnt = normalize_s_waitcnt_field("expcnt", expcnt, 7)
    mlir_rocdl.s_waitcnt(vmcnt << 10 | lgkmcnt << 4 | expcnt)
