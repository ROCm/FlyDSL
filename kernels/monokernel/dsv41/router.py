# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""V4.1's router GEMV tile on gfx1250: bf16 x times the gate weight [E, 5120],
16 experts a tile, K in ``ROUTER_PARTS`` contiguous parts. MI350's order kept
where it is defined: in part p wave slot w (0..7) takes the part's 32-column K
steps w, w + 8, .. in that order, the slots' partials are summed in slot order,
the parts' sums in part order (K2b's route), rounded once to bf16. The
16x16x32 bf16 WMMA's own K order is the hardware's (as the MFMA's was), so the
logits agree with MI350's to rounding, not bit for bit. Waves 8..15 idle: the
stage is a few microseconds and its order fixed by the slots.
"""

import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from kernels.common import buffer_ops as bo
from kernels.monokernel.dsv41.dims import HIDDEN, ROWS
from kernels.monokernel.dsv41.moe_shape import ROUTER_PARTS
from kernels.monokernel.sync_gfx1250 import rsrc, traced
from kernels.monokernel.wmma_gfx1250 import acc_zero, red_index, wmma_bf16

SLOTS = 8  # wave slots of a part: MI350's eight waves
STEPS = HIDDEN // 32 // SLOTS // ROUTER_PARTS  # 5 K steps a slot
assert STEPS * 32 * SLOTS * ROUTER_PARTS == HIDDEN


def _k0s(part, lane, slot):
    """This lane's K offsets in part ``part``: a step's [8 h, + 8) and
    [16 + 8 h, + 8) of its 32 (the bf16 WMMA's half h = lane // 16)."""
    h = lane // 16
    return [
        ((part * STEPS * SLOTS + slot + SLOTS * i) * 32 + 8 * h, (part * STEPS * SLOTS + slot + SLOTS * i) * 32 + 16 + 8 * h)
        for i in range(STEPS)
    ]


def _operands(ptr, row, part, lane, slot, cm):
    return [
        [
            fx.Vector(bo.buffer_load(rsrc(ptr), (row * HIDDEN + k) // 2, vec_width=4, dtype=T.i32, cache_modifier=cm))
            for k in pair
        ]
        for pair in _k0s(part, lane, fx.min(slot, SLOTS - 1))
    ]


def router_weight_loads(w, tile, part, lane, wave, cm=0):
    """The gate weight operands of rows 16 ``tile`` .. (A: row lane % 16)."""
    return _operands(w, tile * ROWS + lane % 16, part, lane, wave, cm)


def router_x_loads(x, part, n, lane, wave, cm=0, t0=0):
    """x's operands for tokens t0 .. t0 + n (B: column lane % 16; past n the
    last token's)."""
    tok = fx.min(lane % 16, n - 1)
    if not (isinstance(t0, int) and t0 == 0):
        tok = tok + t0
    return _operands(x, tok, part, lane, wave, cm)


def router_x_polls(poll, xg, part, n, lane, wave, t0=0):
    """``router_x_loads``' operands polled from XG's tagged rows (token parallel:
    every rank's norm pushes its rows there; the data its own flag). A 16-byte
    chunk of 8 bf16 is 4 pairs: two 16 B specs."""
    tok = fx.min(lane % 16, n - 1)
    if not (isinstance(t0, int) and t0 == 0):
        tok = tok + t0
    ks = _k0s(part, lane, fx.min(wave, SLOTS - 1))
    specs = []
    for pair in ks:
        for k in pair:
            base = (tok * HIDDEN + k) // 2
            specs += [(xg, base, 2), (xg, base + 2, 2)]
    got = poll(specs)
    out, i = [], 0
    for pair in ks:
        row = []
        for _ in pair:
            row.append(fx.Vector.from_elements(got[i] + got[i + 1], fx.Int32))
            i += 2
        out.append(row)
    return out


def _bf16x16(pair):
    return fx.Vector.from_elements([pair[q][e] for q in range(2) for e in range(4)], fx.Int32).bitcast(fx.BFloat16)


@traced
def router_tile_wmma(wvs, xvs, lane, wave, red):
    """A slot's WMMAs, K steps in order -> red[wave][lane] (8 f32: rows
    8 (lane / 16) .., token lane % 16). Waves past the slots store nothing."""
    acc = acc_zero()
    for i in range_constexpr(len(wvs)):
        acc = wmma_bf16(_bf16x16(wvs[i]), _bf16x16(xvs[i]), acc)
    if wave < SLOTS:
        fx.ptr_store(acc, red + red_index(wave, lane, 0))


def router_row_sum(red, r, t):
    """Row r, token t: the slots' partials in slot order."""
    lane = 16 * (r // 8) + t
    tot = fx.Float32(0.0)
    for w in range_constexpr(SLOTS):
        tot = tot + fx.ptr_load(red + red_index(w, lane, r % 8))
    return tot
