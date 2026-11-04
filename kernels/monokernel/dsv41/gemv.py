# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The FP8 GEMV every dense V4.1 projection runs (wqkv_a, wq_b, the indexer's
wq_b, wo_a, wo_b, the shared expert): 16 weight rows of a group32 FP8 weight
(32 x 32 E8M0 block scales) against a tile of <= 16 tokens' MXFP8 rows in LDS,
one scaled 16 x 16 x 128 WMMA a K step.

Weights: ``aiter.ops.shuffle.shuffle_weight_gfx1250`` layout, (N / 16, K * 16)
bytes; row group ``rg``'s K chunk c (16 bytes) of row r at byte
``rg * 16 K + 256 c + 16 r``. A lane of half h reads chunks 8 st + 2 j + h of
step st (``wmma_gfx1250``): four 16-byte loads, the wave's 2 KiB contiguous.

Activations: a token's MXFP8 words in an LDS row of ``lds_row(k / 4)`` words, its
E8M0 codes one i32 a group in a row of ``lds_row(k / 32)``, as the MI350 kernels
lay them out (``load_x8``).

K split: wave w takes steps w, w + WAVES, ...; ``row_sum`` adds the waves'
partials in wave order. The WMMA's own K accumulation differs from the MI350
MFMA's, so a projection agrees with MI350 to rounding, not bit for bit.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from kernels.common import buffer_ops as bo
from kernels.monokernel.dsv41.mx import UNIT_SCALE
from kernels.monokernel.execution_gfx1250 import WAVES
from kernels.monokernel.sync_gfx1250 import TH_NT, rsrc
from kernels.monokernel.wmma_gfx1250 import ACC, FP8, acc_zero, red_index, wmma_scaled

STEP = 128
LDS_PAD = 4  # words a token's LDS row is padded by (bank spread)


def lds_row(words):
    return words + LDS_PAD


def k_steps(k):
    return -(-k // STEP)


def gemv_steps(wave, k, split=None, tiled=False):
    """This wave's K steps [(step, live)]: steps w, w + WAVES, ... (``live`` a
    traced flag on a ragged last round), or ``split`` waves one contiguous range
    each (``tiled``: wave w takes part w % split of row group w / split)."""
    steps = k_steps(k)
    if const_expr(split is None):
        per = -(-steps // WAVES)
        ragged = steps % WAVES != 0
        return [(wave + WAVES * i, (wave + WAVES * i < steps) if ragged else None) for i in range(per)]
    assert split <= WAVES and steps % split == 0, (split, steps)
    per = steps // split
    part = wave % split if tiled else fx.min(wave, split - 1)
    return [(part * per + i, None) for i in range(per)]


def _scale_dword(r_ws, k, rg, st):
    """The four E8M0 codes of row group ``rg``'s step ``st`` from the 32 x 32 block
    scales (N / 32, K / 32): bytes (rg // 2) * kg + 4 st .. +3, a group past K
    the unit code."""
    kg = k // 32
    base = (rg // 2) * kg + 4 * st
    if const_expr(kg % 4 == 0):
        return fx.Int32(bo.buffer_load(r_ws, base // 4, vec_width=1, dtype=T.i32))
    word = fx.Int32(UNIT_SCALE * 0x01010101)
    for g in range_constexpr(4):
        at = (rg // 2) * kg + fx.min(4 * st + g, kg - 1)
        w = fx.Int32(bo.buffer_load(r_ws, at // 4, vec_width=1, dtype=T.i32))
        byte = (w >> ((at % 4) * 8)) & 0xFF
        byte = (4 * st + g < kg).select(byte, fx.Int32(UNIT_SCALE))
        word = (word & fx.Int32(~(0xFF << (8 * g)))) | (byte << (8 * g))
    return word


def _step_load(lane, r_w, r_ws, k, rg, st):
    """Step st's weight operand (v16i32) and scale dword for this lane. A chunk
    past K is clamped into the row (selected out later)."""
    h, r = lane // 16, lane % 16
    kg = k // 32
    blk = rg * (4 * k)  # the row group's first dword
    parts = []
    for j in range_constexpr(4):
        g = fx.min(4 * st + j, kg - 1)  # chunks 2 g + h of this lane
        at = blk + (2 * g + h) * 64 + r * 4
        parts.append(fx.Vector(bo.buffer_load(r_w, at, vec_width=4, dtype=T.i32, cache_modifier=TH_NT)))
    return parts, _scale_dword(r_ws, k, rg, st)


def gemv_fp8_loads(lane, wave, w, ws, k, rg, split=None, tiled=False):
    """The weight operands of rows 16 rg .. +16 for this wave's steps, all in
    flight at once: loads only, so a stage can issue them before the wait for
    its input."""
    steps = k_steps(k)
    r_w, r_ws = rsrc(w), rsrc(ws)
    if const_expr(tiled):
        assert WAVES % split == 0
        rg = rg + wave // split
    return [
        _step_load(lane, r_w, r_ws, k, rg, st if live is None else fx.min(st, steps - 1))
        for st, live in gemv_steps(wave, k, split, tiled)
    ]


def gemv_scale_dwords(wave, ws, k, rg, split=None, tiled=False):
    """This wave's scale dwords of row group ``rg`` (``_scale_dword``), a step
    each: loads only, so a stage issues them early (``gemv_fp8_lds_ops``)."""
    steps = k_steps(k)
    r_ws = rsrc(ws)
    return [
        _scale_dword(r_ws, k, rg, st if live is None else fx.min(st, steps - 1))
        for st, live in gemv_steps(wave, k, split, tiled)
    ]


def gemv_fp8_lds_ops(lane, wave, wl, ws, k, rg, split=None, tiled=False, scales=None):
    """``gemv_fp8_loads``' operands with the weight read from LDS: ``wl`` (Int32)
    holds the 16-row group's ``k * 16`` shuffled bytes (one ``tdm.load_block``
    of the group's contiguous block, dword ``rg * 4 k`` of the weight); the
    scale dwords from global, or ``scales`` (``gemv_scale_dwords``, issued
    earlier). The same dwords per lane as ``_step_load``."""
    steps = k_steps(k)
    if const_expr(scales is None):
        scales = gemv_scale_dwords(wave, ws, k, rg, split, tiled)
    h, r = lane // 16, lane % 16
    kg = k // 32
    out = []
    for i, (st, live) in enumerate(gemv_steps(wave, k, split, tiled)):
        st = st if live is None else fx.min(st, steps - 1)
        parts = []
        for j in range_constexpr(4):
            g = fx.min(4 * st + j, kg - 1)
            parts.append(fx.Vector(fx.ptr_load(wl + ((2 * g + h) * 64 + r * 4), result_type=fx.Vector.make_type(4, fx.Int32))))
        out.append((parts, scales[i]))
    return out


def group_bytes(k: int) -> int:
    """A 16-row group's shuffled bytes (``shuffle_gfx1250``: contiguous)."""
    return 16 * k


def _step_wmma(lane, k, xl, xsl, rows, st, op, acc, live):
    """``acc`` through step st: this wave's weight operand against the tokens'
    MXFP8 rows in LDS. A group past K (or a dead step) is zero on both sides at
    unit scale: a NaN code past the row must not reach the WMMA."""
    parts, sa = op
    h, t = lane // 16, fx.min(lane % 16, rows - 1)
    kg = k // 32
    x_row, s_row = lds_row(k // 4), lds_row(kg)
    if const_expr(live is not None):
        st = fx.min(st, k_steps(k) - 1)
    a, b = [], []
    for j in range_constexpr(4):
        g = 4 * st + j
        ok = g < kg
        if const_expr(live is not None):
            ok = ok & live
        xw = fx.Vector(
            fx.ptr_load(
                xl + (t * x_row + fx.min(g, kg - 1) * 8 + 4 * h),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )
        a += [ok.select(parts[j][d], fx.Int32(0)) for d in range(4)]
        b += [ok.select(xw[d], fx.Int32(0)) for d in range(4)]
    if const_expr(kg % 4 == 0):  # the row 16-byte aligned: one load
        cv = fx.Vector(fx.ptr_load(xsl + (t * s_row + 4 * st), result_type=fx.Vector.make_type(4, fx.Int32)))
        codes = [cv[g] for g in range(4)]
    else:
        codes = [fx.ptr_load(xsl + (t * s_row + fx.min(4 * st + g, kg - 1))) for g in range(4)]
    sb = fx.Int32(0)
    for g in range_constexpr(4):
        ok = 4 * st + g < kg
        if const_expr(live is not None):
            ok = ok & live
        code = ok.select(codes[g] & 0xFF, fx.Int32(UNIT_SCALE))
        sb = sb | (code << (8 * g))
    if const_expr(live is not None):
        sa = live.select(sa, fx.Int32(UNIT_SCALE * 0x01010101))
    av = fx.Vector.from_elements(a, fx.Int32)
    bv = fx.Vector.from_elements(b, fx.Int32)
    return wmma_scaled(av, bv, acc, sa, sb, FP8, FP8)


def gemv_fp8_wmmas(lane, wave, k, xl, xsl, red, ops, rows, split=None, tiled=False):
    """``gemv_fp8_loads``' operands against tokens 0 .. rows - 1 in LDS ->
    this wave's partial accumulator in ``red`` (``wmma_gfx1250.red_index``)."""
    acc = acc_zero()
    for i, (st, live) in enumerate(gemv_steps(wave, k, split, tiled)):
        acc = _step_wmma(lane, k, xl, xsl, rows, st, ops[i], acc, live)
    if const_expr(split is not None and not tiled):
        acc = fx.Vector.from_elements([(wave < split).select(acc[e], fx.Float32(0.0)) for e in range(ACC)], fx.Float32)
    fx.ptr_store(acc, red + red_index(wave, lane, 0))
