# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""``step_begin`` on gfx1250: a decode step's start, after its mailboxes (scratch
and this rank's peer buffer) were zeroed on the same stream. The step's kernels
reuse every tag a step, so a rank must not push into a peer before that peer has
cleared: one thread bumps this rank's epoch (a device word that persists across
steps and graph replays), writes it into every rank's barrier slot ``rank``
(system scope) and waits until every slot of its own barrier holds it.

The barrier slots sit past the cleared peer regions (``BARRIER_BYTES`` at
``barrier_offset``): the step's clear must not touch them.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from kernels.common import buffer_ops as bo
from kernels.monokernel.dsv41.layout import peer_bytes
from kernels.monokernel.dsv41.sources import cache_tag, kernel_name
from kernels.monokernel.ops_gfx1250 import load_ptr64
from kernels.monokernel.sync_gfx1250 import SCOPE_SYS, rsrc, traced

BARRIER_BYTES = 256  # a word a rank (up to 64)
_CURRENT_STREAM = fx.Stream(None)


def barrier_offset(tokens: int, tp: int) -> int:
    return -(-peer_bytes(tokens, tp) // 256) * 256


def symmetric_bytes(tokens: int, tp: int) -> int:
    """The peer buffer a rank allocates: the regions, then the barrier slots."""
    return barrier_offset(tokens, tp) + BARRIER_BYTES


@traced
def _barrier(epoch, sym, peers, bar, rank, tp):
    e = fx.Int32(bo.buffer_load(rsrc(epoch), 0, vec_width=1, dtype=T.i32)) + 1
    bo.buffer_store(e, rsrc(epoch), 0)
    for p in range_constexpr(tp):
        bo.buffer_store(e, rsrc(load_ptr64(peers, p) + fx.Int64(bar)), rank, cache_modifier=SCOPE_SYS)
    mine = rsrc(sym + fx.Int64(bar))
    low = fx.Int32(0)
    for p in range_constexpr(tp):
        v = fx.Int32(bo.buffer_load(mine, p, vec_width=1, dtype=T.i32, cache_modifier=SCOPE_SYS))
        low = (v < e).select(low + 1, low)
    while low != 0:
        rocdl.s_sleep(1)
        low = fx.Int32(0)
        for p in range_constexpr(tp):
            v = fx.Int32(bo.buffer_load(mine, p, vec_width=1, dtype=T.i32, cache_modifier=SCOPE_SYS))
            low = (v < e).select(low + 1, low)


def build_step_begin(tp: int):
    name = kernel_name("dsv41_step_begin", tp=tp)

    @flyc.kernel(name=name, known_block_size=[32, 1, 1])
    def step_begin(epoch: Int64, sym: Int64, peers: Int64, bar: Int32, rank: Int32):
        if fx.thread_idx.x == 0:
            _barrier(epoch, sym, peers, bar, rank, tp)

    tag = cache_tag(name, tp)

    @flyc.jit
    def launch(epoch: Int64, sym: Int64, peers: Int64, bar: Int32, rank: Int32, stream: fx.Stream = _CURRENT_STREAM):
        _ = tag  # the build in the disk cache's key (``sources.cache_tag``)
        step_begin(epoch, sym, peers, bar, rank).launch(grid=(1,), block=(32,), stream=stream)

    return launch
