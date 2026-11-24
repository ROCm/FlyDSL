# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""One wrapper per ISA instruction, named by its mnemonic, in one module per architecture.

ROCDL only. Reached as ``fx.rocdl.experimental.inst.cdna3`` (gfx942) and ``fx.rocdl.experimental.inst.cdna4``
(gfx950); an architecture module exports exactly the instructions its ISA supports::

    from flydsl.expr.rocdl.experimental.inst import cdna3

    acc = cdna3.V_MFMA_F32_32X32X8_BF16(a, b, acc)
    cdna3.V_MFMA_F32_32X32X8_BF16(frag_a, frag_b, frag_acc, out=frag_acc)  # fx.Tensor fragments, result stored
    v = cdna3.BUFFER_LOAD_DWORDX4(rsrc, voffset, soffset, dtype=fx.Vector[fx.BFloat16, 8], aux="nt")
    cdna3.BUFFER_LOAD_DWORDX4(rsrc, voffset, out=frag)  # loaded as frag's element type, stored to the fragment

``ScheduleRegion()`` keeps the instructions called in a block in the order they are written::

    with fx.rocdl.experimental.inst.ScheduleRegion():
        x = cdna3.BUFFER_LOAD_DWORDX4(rsrc, voffset)
        acc = cdna3.V_MFMA_F32_32X32X8_BF16(a, b, acc)  # stays after the load in the ISA
"""

from . import cdna3 as cdna3
from . import cdna4 as cdna4
from ._sched import ScheduleRegion as ScheduleRegion

__all__ = [
    "cdna3",
    "cdna4",
    "ScheduleRegion",
]
