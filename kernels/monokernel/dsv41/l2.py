# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""GL2 prefetch on gfx1250 (``GLOBAL_PREFETCH_B8``, ISA 4.9.6): a lane names one
byte, the line holding it comes into the GL2 (SCOPE DEV: not the WGP cache), no
data returns and LOADcnt does not move -- nothing to wait for, so a CTA issues
the weights it reads later and goes on.

Rules (kernel-wiki mi400-async-multicast-paths 4.1, corrections ledger):
  * an address is translated like a demand load and a failed translation is
    reported to the host: only lines inside the tensor (``nbytes`` exact);
  * the GL2 is 96 channels an AID shared by its four XCDs, not one uniform
    device cache: a CTA prefetches what it reads itself, never another's share;
  * a line a lane: ``LINE`` 128 B covers the GL2's 256 B lines twice over and
    the 128 B WGP/GL1 lines once (cheap: a CTA's 512 lanes are 64 KB an issue).
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, rocdl
from flydsl.expr import const_expr, range_constexpr

from kernels.monokernel.execution_gfx1250 import THREADS
from kernels.monokernel.sync_gfx1250 import SCOPE_DEV, traced

LINE = 128


def prefetch(addr):
    """This lane's line at the i64 global address ``addr`` -> GL2."""
    ptr = llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), fx.Int64(addr).ir_value())
    rocdl.global_prefetch(ptr, cache_policy=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), SCOPE_DEV))


@traced
def block(base, nbytes: int, idx, lanes: int = THREADS):
    """Lines of [base, base + nbytes) (``nbytes`` a whole number of lines, the
    region inside its tensor): lane ``idx`` of ``lanes`` takes lines idx,
    idx + lanes, ..."""
    assert nbytes % LINE == 0, nbytes
    lines = nbytes // LINE
    for q in range_constexpr(-(-lines // lanes)):
        i = idx + q * lanes
        if const_expr((q + 1) * lanes <= lines):
            prefetch(base + fx.Int64(i) * LINE)
        else:
            if i < lines:
                prefetch(base + fx.Int64(i) * LINE)


@traced
def rows(base, n_rows: int, row_bytes: int, stride: int, idx, lanes: int = THREADS):
    """Lines of ``n_rows`` rows of ``row_bytes`` (whole lines), ``stride`` bytes
    apart from ``base``: lane ``idx`` of ``lanes`` takes the flat (row, line)s
    idx, idx + lanes, ..."""
    assert row_bytes % LINE == 0, row_bytes
    per = row_bytes // LINE
    total = n_rows * per
    for q in range_constexpr(-(-total // lanes)):
        i = idx + q * lanes
        at = base + fx.Int64(i // per) * stride + fx.Int64(i % per) * LINE
        if const_expr((q + 1) * lanes <= total):
            prefetch(at)
        else:
            if i < total:
                prefetch(at)
