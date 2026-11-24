# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Provisional members of the architecture modules: VALU conversions, dot products and lane instructions.

Their scheduling class is ``"valu"``, which is not exclusive, so :class:`ScheduleRegion` does not order them: it keeps
them between the ordered instructions they are written between. Each architecture module lists them apart
(``PROVISIONAL``) and appends them to its ``__all__``.
"""

from .alu import (
    V_DOT2_F32_BF16,
    V_DOT2_F32_F16,
    V_DOT2_I32_I16,
    V_DOT2_U32_U16,
    V_DOT4_I32_I8,
    V_DOT4_U32_U8,
    V_DOT8_I32_I4,
    V_DOT8_U32_U4,
    V_MED3_F16,
    V_MED3_F32,
    V_PERM_B32,
)
from .cvt import (
    SCALEF32,
    V_CVT_F32_BF8,
    V_CVT_F32_FP8,
    V_CVT_PK_BF8_F32,
    V_CVT_PK_F32_BF8,
    V_CVT_PK_F32_FP8,
    V_CVT_PK_FP8_F32,
    V_CVT_PKRTZ_F16_F32,
    V_CVT_SR_BF8_F32,
    V_CVT_SR_FP8_F32,
)
from .wave import (
    V_MBCNT_HI_U32_B32,
    V_MBCNT_LO_U32_B32,
    V_PERMLANE16_SWAP_B32,
    V_PERMLANE32_SWAP_B32,
    V_READFIRSTLANE_B32,
    V_READLANE_B32,
)

#: gfx942 and gfx950.
CDNA3 = (
    # alu
    V_PERM_B32,
    V_MED3_F32,
    V_MED3_F16,
    V_DOT2_F32_F16,
    V_DOT2_I32_I16,
    V_DOT2_U32_U16,
    V_DOT4_I32_I8,
    V_DOT4_U32_U8,
    V_DOT8_I32_I4,
    V_DOT8_U32_U4,
    # cvt
    V_CVT_F32_FP8,
    V_CVT_F32_BF8,
    V_CVT_PK_FP8_F32,
    V_CVT_PK_BF8_F32,
    V_CVT_PK_F32_FP8,
    V_CVT_PK_F32_BF8,
    V_CVT_SR_FP8_F32,
    V_CVT_SR_BF8_F32,
    V_CVT_PKRTZ_F16_F32,
    # wave
    V_READFIRSTLANE_B32,
    V_READLANE_B32,
    V_MBCNT_LO_U32_B32,
    V_MBCNT_HI_U32_B32,
)
#: gfx950: the gfx942 set plus the gfx950-only instructions.
CDNA4 = CDNA3 + (V_DOT2_F32_BF16, V_PERMLANE16_SWAP_B32, V_PERMLANE32_SWAP_B32, *SCALEF32)

for _instr in CDNA4:  # every provisional instruction is a VALU instruction
    _instr.sched = "valu"
