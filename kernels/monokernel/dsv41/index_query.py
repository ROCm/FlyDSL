# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""K1's indexer projections (FULL / REINDEX layers) on gfx1250: the index query
and the head weights the scorer reads, bit for bit the original chain
(``Indexer.project``, then ``quantize_query_rows`` / ``scale_indexer_weights``).
Every rank computes all 32 heads, as the original does.

    iw      (32 tasks, a row of weights_proj [32, 5120] each): aiter
            ``wv_splitk_small``'s order -- a 64-lane wave's 8-element chunks at
            k = k1 + 512 k2 + 8 lane64, then its v_add_f32 DPP chain (row_shr 8,
            4, 2, wave_shr 1, row_bcast 15, row_bcast 31, lane 63). gfx12 has no
            wave64, wave_shr or row_bcast: a pair of wave32 waves (2 v, 2 v + 1)
            is the 64-lane wave v, its per-lane sums meet in LDS, and one wave
            replays the chain on (low, high) halves, lane 63's value exactly.
    iquant  (S x 32 tasks, a (token, head) row of 128): the FP8 plane's
            ``dynamic_per_token_scaled_quant`` or the FP4 plane's ``quantize_fp4``
            (group-32 E8M0 E2M1), and the head weight. A max is exact in any
            order, so these use plain wave32 trees.

The FP4 query is written in the natural layout the gfx1250 scorer reads: a
(token, head) row's 64 packed bytes (even dim in the low nibble) at
``row * 64``, its four E8M0 codes at ``row * 4`` (``iqs_out``).
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from kernels.common import buffer_ops as bo
from kernels.monokernel.dsv41.mx import FP8_MAX, clamp_fp8
from kernels.monokernel.execution_gfx1250 import WAVE, WAVES
from kernels.monokernel.ops_gfx1250 import (
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    fp8_pack4,
    lane_gather,
)
from kernels.monokernel.sync_gfx1250 import TH_NT, lds_barrier, rsrc, traced

HEADS = 32  # index_n_heads
DIM = 128  # index_head_dim
IDX_ROWS = HEADS * DIM  # 4096
IDX_ROPE = 64
IDX_WEIGHTS_SCALE = DIM**-0.5 * HEADS**-0.5  # 1/64: exact
IW_TASKS = HEADS
IW_ROWS = 1  # iw rows a task (``stage_iw``; 2: index layers +0.6 us, 2026-10-30 -- a CTA's serial work, not the reads)
AMAX_SEED = 1e-10
VWAVES = WAVES // 2  # 64-lane virtual waves (wave pairs)
FP4_MIN_AMAX = 6.0 * 2.0**-126


def scratch_pairs(tokens: int, hidden: int) -> dict:
    return {
        "nb": tokens * hidden // 2,  # the normed row, bf16 pairs
        "iq": tokens * IDX_ROWS // 2,  # the rotated index query, bf16 pairs
        "iw": tokens * HEADS,  # weights_proj out, f32 of bf16
    }


def _shr(v, lane, k):
    """DPP row_shr:k on a wave32 half: lane - k of the same 16-lane row, else 0."""
    moved = lane_gather(v.bitcast(fx.Int32), fx.max(lane - k, 0)).bitcast(fx.Float32)
    return fx.Float32((lane % 16 >= k).select(moved, fx.Float32(0.0)))


def wv_reduce64(lo, hi, lane):
    """wv_splitk's cross-lane chain on a 64-lane value held as (lo = lanes 0..31,
    hi = lanes 32..63) by one wave32; returns lane 63's result (wave-uniform after
    the final gather)."""
    for k in (8, 4, 2):
        lo = _shr(lo, lane, k) + lo
        hi = _shr(hi, lane, k) + hi
    # wave_shr:1 -- lane l += lane l - 1 across the whole 64 lanes
    lo_m1 = lane_gather(lo.bitcast(fx.Int32), fx.max(lane - 1, 0)).bitcast(fx.Float32)
    hi_m1 = lane_gather(hi.bitcast(fx.Int32), fx.max(lane - 1, 0)).bitcast(fx.Float32)
    lo31 = lane_gather(lo.bitcast(fx.Int32), 31).bitcast(fx.Float32)
    lo_new = fx.Float32((lane >= 1).select(lo_m1 + lo, lo))
    hi_new = fx.Float32((lane >= 1).select(hi_m1 + hi, lo31 + hi))
    lo, hi = lo_new, hi_new
    # row_bcast:15 -- rows 1 and 3 add the last lane of rows 0 and 2
    lo15 = lane_gather(lo.bitcast(fx.Int32), 15).bitcast(fx.Float32)
    hi15 = lane_gather(hi.bitcast(fx.Int32), 15).bitcast(fx.Float32)
    lo = fx.Float32((lane >= 16).select(lo15 + lo, lo))
    hi = fx.Float32((lane >= 16).select(hi15 + hi, hi))
    # row_bcast:31 -- rows 2 and 3 add lane 31
    lo31 = lane_gather(lo.bitcast(fx.Int32), 31).bitcast(fx.Float32)
    hi = lo31 + hi
    return lane_gather(hi.bitcast(fx.Int32), 31).bitcast(fx.Float32)


@traced
@traced
def stage_iw(c, task, hidden, nbl, load_group):
    """Rows IW_ROWS (task // groups) .. of weights_proj against token group ``task % groups``
    (VWAVES tokens, ``iw_groups``): virtual wave v (waves 2 v, 2 v + 1) takes the
    group's token v, the group's normed rows in LDS ``nbl`` (bf16 pairs;
    ``load_group(first token, tokens)``) -> IW. A (row, group) a task: a row a
    task, its groups in turn, ran 6 groups back to back on 32 CTAs (S = 48: the
    index stage 50 us, 2026-10-30); a token's sum is the same either way."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    a = c["args"]
    xbuf = c["xbuf"]
    iw_dsts = c.get("iw_dsts")  # token parallel: every rank's IWG (the quantization reads them)
    vw = wave // 2
    vlane = (wave % 2) * WAVE + lane
    groups = iw_groups(s)
    g = task % groups * VWAVES
    n = fx.min(fx.Int32(VWAVES), s - g)
    ks = [k1 + k2 * 512 for k1 in range(0, hidden, 2048) for k2 in range(4) if k1 + k2 * 512 < hidden]

    def weights(m):
        return [
            fx.Vector(
                bo.buffer_load(
                    rsrc(a["iw_w"]), (m * hidden + fx.min(k0 + vlane * 8, hidden - 8)) // 2,
                    vec_width=4, dtype=T.i32, cache_modifier=TH_NT,
                )
            )
            for k0 in ks
        ]  # fmt: skip

    # IW_ROWS rows a task over one copy of the group's rows (a row a task re-read the
    # group from NBG for each of the 32: 22 MB of system-scope reads at S = 48), the
    # next row's weights loading while one is summed
    m0 = task // groups * IW_ROWS
    wvs = weights(m0)
    load_group(g, n)
    tl = fx.min(vw, n - 1)
    t = tl + g
    for r in range_constexpr(IW_ROWS):
        m = m0 + r
        if const_expr(r + 1 < IW_ROWS):
            nxt = weights(m + 1)
        total = fx.Float32(0.0)
        for i in range_constexpr(len(ks)):
            k = ks[i] + vlane * 8
            live = k < hidden
            kk = fx.min(k, hidden - 8)
            xv = fx.Vector(fx.ptr_load(nbl + ((tl * hidden + kk) // 2), result_type=fx.Vector.make_type(4, fx.Int32)))
            wv = wvs[i]
            ax = [bf_lo(xv[j]) for j in range(4)]
            ay = [bf_hi(xv[j]) for j in range(4)]
            bx = [bf_lo(wv[j]) for j in range(4)]
            by = [bf_hi(wv[j]) for j in range(4)]
            accx = ax[0] * bx[0]
            accy = ay[0] * by[0]
            for j in range_constexpr(1, 4):
                accx = fx.Float32(fmath.fma(ax[j], bx[j], accx))
                accy = fx.Float32(fmath.fma(ay[j], by[j], accy))
            total = fx.Float32(live.select(total + (accx + accy), total))
        # the 64 lane sums of virtual wave vw meet in LDS
        fx.ptr_store(total, xbuf + (vw * 64 + vlane))
        lds_barrier()
        if wave % 2 == 0:
            lo = fx.ptr_load(xbuf + (vw * 64 + lane))
            hi = fx.ptr_load(xbuf + (vw * 64 + WAVE + lane))
            w = wv_reduce64(lo, hi, lane)
            if (lane == 0) & (vw < n):
                if const_expr(iw_dsts is not None):
                    for p in range_constexpr(len(iw_dsts)):
                        c["put"](iw_dsts[p], (c["t0"] + t) * HEADS + m, bf16_round(w))
                else:
                    c["put"](c["iw"], t * HEADS + m, bf16_round(w))
        lds_barrier()
        if const_expr(r + 1 < IW_ROWS):
            wvs = nxt


def iw_groups(s) -> int:
    """Token groups of VWAVES an iw row splits into (``stage_iw``'s tasks)."""
    return -(-s // VWAVES)


def iw_tasks(s) -> int:
    return HEADS // IW_ROWS * iw_groups(s)


def poll_query(c, t, h, lane):
    """Token t, head h: the lane's two bf16 pairs of the index query (dims 4 lane
    .. 4 lane + 3) and the head's weight, one poll batch."""
    base = (t * IDX_ROWS + h * DIM) // 2 + 2 * lane
    return c["poll"]([(c["iq"], base, 1), (c["iq"], base + 1, 1), (c["iw"], t * HEADS + h, 1)])


@traced
def stage_iquant(c, task, who=0):
    """FP8 plane: token task // 32, head task % 32, wave 0, 4 dims a lane."""
    lane, wave = c["lane"], c["wave"]
    a = c["args"]
    t = task // HEADS
    h = task % HEADS
    if wave == who:
        got = poll_query(c, t, h, lane)
        x = [bf_lo(got[0][0]), bf_hi(got[0][0]), bf_lo(got[1][0]), bf_hi(got[1][0])]
        m = fx.Float32(AMAX_SEED)
        for v in x:
            m = fx.max(m, abs(v))
        amax = butterfly(m, (1, 2, 4, 8, 16), fx.max)
        scale = amax * fx.Float32(1.0 / FP8_MAX)
        inv = fx.Float32(1.0) / scale
        q = fp8_pack4(*[clamp_fp8(v * inv) for v in x])
        row = t * HEADS + h
        bo.buffer_store(q, rsrc(a["iq_out"]), row * (DIM // 4) + lane)
        if lane == 0:
            w = got[2][0].bitcast(fx.Float32)
            bo.buffer_store(scale, rsrc(a["iqs_out"]), row)
            bo.buffer_store((w * scale) * fx.Float32(IDX_WEIGHTS_SCALE), rsrc(a["iw_out"]), row)


def _e2m1(x):
    """The E2M1 nibble of ``x`` (a value over its scale): nearest-even with
    ``quantize_fp4``'s asymmetric midpoints, the sign bit x's."""
    m = abs(x)
    code = fx.Int32(0)
    for edge, value, inclusive in (
        (0.25, 1, False), (0.75, 2, True), (1.25, 3, False), (1.75, 4, True),
        (2.5, 5, False), (3.5, 6, True), (5.0, 7, False),
    ):  # fmt: skip
        past = (m >= fx.Float32(edge)) if inclusive else (m > fx.Float32(edge))
        code = fx.Int32(past.select(fx.Int32(value), code))
    sign = (x.bitcast(fx.Int32) >> 31) & 1
    return code | (sign << 3)


@traced
def stage_iquant_fp4(c, task, who=0):
    """FP4 plane: ``quantize_fp4`` to the bit, wave 0, 4 dims a lane, a 32-dim
    group 8 lanes: code = ceil(log2(max(amax, FP4_MIN_AMAX) / 6)), the value
    times 2^(127 - code) to E2M1."""
    lane, wave = c["lane"], c["wave"]
    a = c["args"]
    t = task // HEADS
    h = task % HEADS
    if wave == who:
        got = poll_query(c, t, h, lane)
        x = [bf_lo(got[0][0]), bf_hi(got[0][0]), bf_lo(got[1][0]), bf_hi(got[1][0])]
        m = fx.max(fx.max(abs(x[0]), abs(x[1])), fx.max(abs(x[2]), abs(x[3])))
        amax = butterfly(m, (1, 2, 4), fx.max)
        bits = (fx.max(amax, fx.Float32(FP4_MIN_AMAX)) * fx.Float32(1.0 / 6.0)).bitcast(fx.Int32)
        code = (bits >> 23) + fx.Int32(((bits & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0)))
        inv = ((fx.Int32(254) - code) << 23).bitcast(fx.Float32)
        nib = [_e2m1(v * inv) for v in x]
        half = (nib[0] | (nib[1] << 4)) | ((nib[2] | (nib[3] << 4)) << 8)
        row = t * HEADS + h
        bo.buffer_store(fx.Int16(half), rsrc(a["iq_out"]), row * (DIM // 4) + lane)
        if lane % 8 == 0:
            bo.buffer_store(fx.Int8(code), rsrc(a["iqs_out"]), row * 4 + lane // 8)
        if lane == 0:
            w = got[2][0].bitcast(fx.Float32)
            bo.buffer_store(w * fx.Float32(IDX_WEIGHTS_SCALE), rsrc(a["iw_out"]), row)


