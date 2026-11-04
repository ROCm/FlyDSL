# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""K2b ``moe`` on gfx1250: one V4.1 layer's FFN from ``normed`` (K2a's MoE input)
to the TP-reduced output, ported stage for stage from the MI350 mono kernel.

    xq      (S): the row as MXFP8 for the routed experts (amax seeded 1e-10)
            and for the shared one (floored 1e-4) -> X8R, X8 (plain words at
            device scope, then the token's flag X8RDY)
    router  (E / 16 x 4): bf16 WMMA GEMV, a 16-row tile and a quarter of K
            (``router``) -> the part's f32 sum -> LOGIT
    route   (S): the parts summed in order -> bf16 logits; aiter
            ``topk_gating_kernel_opt`` (sqrt softplus + bias, top-k, renormalized
            x 1.5), one wave32 a token (12 candidates a lane, ties broken as the
            wave64 kernel does) -> ROUTE (8 copies)
    shared  (sh_inter / 16): FP8 GEMV gate and up -> bf16 -> silu x up -> bf16 ->
            SMID; an odd task quantizes its group of 32 -> SMQ / SMC
    ug      (|U| x parts): MXFP4 x MXFP8 GEMV of an expert's gate / up columns,
            clamp, silu x up, bf16, MXFP8 -> MID, the task's flag UGF
    down    (|U| x 320 units, a balanced contiguous range a CTA): the MXFP4 down
            GEMV of a unit's 16 w2 rows (TDM-streamed, a ring a wave) against
            its expert's picks' MID (staged in LDS) x the routing weight -> bf16
            CONTRIB tagged pairs; then 20 rows a CTA: the picks' CONTRIB (polled)
            summed in top-k order in bf16, + the shared expert's FP8 down
            -> bf16 -> every rank's FFN peer region
    reduce  (160): the TP partials in rank order -> bf16 -> ``out``
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from kernels.common import buffer_ops as bo
from kernels.monokernel.dsv41 import l2, tdm
from kernels.monokernel.dsv41.attn_pre import TILE, load_x8, plus, tile_rows, token_tiles
from kernels.monokernel.dsv41.attn_post import sum_partials
from kernels.monokernel.dsv41.dims import HIDDEN, ROWS, Dims
from kernels.monokernel.dsv41.gemv import (
    gemv_fp8_lds_ops,
    gemv_fp8_loads,
    gemv_fp8_wmmas,
    gemv_scale_dwords,
    group_bytes,
    k_steps,
    lds_row,
)
from kernels.monokernel.dsv41.layout import MOE_TAG0, mailbox, pair_layout, peer_offsets
from kernels.monokernel.dsv41.moe_shape import (
    COMBINE_ROWS,
    DOWN_SPAN,
    DOWN_TILES,
    MASK_WORDS,
    ROUTER_PARTS,
    MoeBuild,
    combine_tiles,
    down_range,
    n_union,
    pick_at,
    route_shape,
    route_w,
    slot_first,
    tile_first,
    tile_slot,
    ug_range,
    union_expert,
)
from kernels.monokernel.dsv41.mx import FP8_MAX, clamp_fp8, code_ceil, rcp_pow2
from kernels.monokernel.dsv41.router import (
    router_row_sum, router_tile_wmma, router_weight_loads, router_x_loads, router_x_polls,
)  # fmt: skip
from kernels.monokernel.dsv41.sources import cache_tag, kernel_name
from kernels.monokernel.dsv41.token_par import own_count, own_view
from kernels.monokernel.execution_gfx1250 import BLOCKS, LDS_MAX, THREADS, WAVE, WAVES, first_task
from kernels.monokernel.ops_gfx1250 import (
    _hw_f32,
    ballot,
    bf_hi,
    bf_lo,
    bf16_round,
    butterfly,
    fp8_pack4,
    fresh,
    hw_exp2,
    hw_rcp,
    lane_gather,
    lanes_below,
    load_ptr64,
    popcount,
    readlane,
    uniform,
)
from kernels.monokernel.sync_gfx1250 import SCOPE_DEV, TH_NT, TH_RT, lds_barrier, preg, publish, rsrc, sreg, to_bf16, traced, wait_store
from kernels.monokernel.wmma_gfx1250 import ACC, FP4, FP8, red_index, row_sum, wmma_scaled
from kernels.monokernel.wmma_gfx1250 import acc_zero as acc8

SCALE = 1.5
LIMIT = 10.0  # swiglu_limit
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
RED_COLS = 32
RED_TASKS = HIDDEN // RED_COLS  # 160
ROUTE_COPIES = 8  # CTA b polls route copy b % 8
X8_WORDS = HIDDEN // 4
X8_BATCH = 5  # load_x8_rows' 16-byte loads in flight at once (20 VGPRs)
XQ0 = DOWN0 = RED0 = 0
_CURRENT_STREAM = fx.Stream(None)

ARGS = (
    "x", "gate_w", "bias", "w13", "w13_s", "w2", "w2_s", "sgu", "sgu_s", "sw2", "sw2_s", "out",
    "scratch", "sym", "peers", "rank", "layer",
)  # fmt: skip


def scratch_layout(key: MoeBuild) -> dict:
    s, rs = key.tokens, route_shape(key)
    d = rs.dims
    pairs = {
        "x8r": s * X8_WORDS // 2,
        "x8rs": s * HIDDEN // 32 // 2,
        "x8": s * X8_WORDS // 2,
        "x8s": s * HIDDEN // 32 // 2,
        "x8rdy": s,
        "logit": s * rs.experts * ROUTER_PARTS,
        "route": ROUTE_COPIES * s * 2 * (rs.topk + int(key.herd)),
        "smid": s * d.sh_inter,
        "smq": s * d.sh_inter // 4,
        "smc": s * d.sh_inter // 32,
        "mid": rs.picks * d.mid_words // 2,
        "mids": rs.picks * d.inter // 32 // 2,
        "ugf": rs.ug_flags,
        "contrib": rs.picks * HIDDEN // 2,
    }
    return pair_layout(pairs.items())


def scratch_bytes(key: MoeBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def ffn_region_pair(src, t, s, col):
    """Pair of (source rank, token, column) in the FFN peer region, laid out by
    the sender: [src][combine CTA][token][COMBINE_ROWS columns], so a CTA's push
    to a rank is one contiguous run (S x 80 B) written 16 B a lane."""
    return ((src * BLOCKS + col // COMBINE_ROWS) * s + t) * (COMBINE_ROWS // 2) + col % COMBINE_ROWS // 2


def st_dev(v, region, i):
    """Word i of scratch ``region`` (plain words, not tagged pairs) at device
    scope: another CTA reads it after the flag."""
    bo.buffer_store(v, rsrc(region.value), i, cache_modifier=SCOPE_DEV)


def ld_dev4(region, i):
    return fx.Vector(bo.buffer_load(rsrc(region.value), i, vec_width=4, dtype=T.i32, cache_modifier=SCOPE_DEV))


def ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


# ---------------------------------------------------------------- xq
def put_mxfp8_group(c, words_mb, codes_mb, t, g, v, amax):
    """Token t's group g of 32 values ``v`` as MXFP8 (the ceil code of amax / 448,
    x / scale clamped, RNE) -> plain words (the token's X8RDY announces them)."""
    code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
    inv = rcp_pow2(code)
    words = [fp8_pack4(*[clamp_fp8(v[4 * w + i] * inv) for i in range(4)]) for w in range(8)]
    for h in range_constexpr(2):
        st_dev(fx.Vector.from_elements(words[4 * h : 4 * h + 4], fx.Int32), c[words_mb], t * X8_WORDS + g * 8 + 4 * h)
    st_dev(fx.Int32(code), c[codes_mb], t * (HIDDEN // 32) + g)


def xg_words(c, t, w0, nwords):
    """Token parallel: words w0 .. w0 + nwords (bf16 pairs) of token t's normed row
    straight from this rank's XG (tagged pairs, every rank's norm pushes there):
    the data its own flag, no XRDY hop. One batched poll, 16 B a spec."""
    base = t * (HIDDEN // 2) + w0
    got = c["poll"]([(c["x_xg"], base + 2 * k, 2) for k in range(nwords // 2)])
    return [w for pair in got for w in pair]


@traced
def await_x(c, tokens):
    """In a launch that also writes x (``layer_post``): the tokens' rows ready
    (XRDY), polled by the first threads, then the CTA's barrier. Token parallel:
    nothing (the readers poll XG's tagged rows themselves)."""
    await_x_range(c, fx.Int32(tokens[0]), len(tokens), len(tokens))


@traced
def await_x_range(c, t0, n, nmax):
    """``await_x`` of tokens t0 .. t0 + n (n traced, <= nmax threads polling)."""
    if const_expr(c["x_ready"] and c.get("x_xg") is None):
        tid = c["tid"]
        if tid < nmax:
            c["poll"]([(c["xrdy"], t0 + fx.min(tid, n - 1), 1)])
        lds_barrier()


@traced
def stage_xq(c, t):
    """Token t, a group of 32 a thread (threads past 160 mirror 159)."""
    tid, a, cm = c["tid"], c["args"], c["x_cm"]
    g = fx.min(tid, HIDDEN // 32 - 1)
    mine = tid < HIDDEN // 32
    if const_expr(c.get("x_xg") is not None):
        # token parallel: the row from XG; it lands in ``normed`` here too (the
        # launch's output: every token, the gather stage gone)
        wl = xg_words(c, t, g * 16, 16)
        words = [fx.Vector.from_elements(wl[4 * q : 4 * q + 4], fx.Int32) for q in range(4)]
        if mine:
            for q in range_constexpr(4):
                bo.buffer_store(words[q], rsrc(a["x"]), (t * HIDDEN + g * 32) // 2 + 4 * q, cache_modifier=cm)
    else:
        words = [
            fx.Vector(bo.buffer_load(rsrc(a["x"]), (t * HIDDEN + g * 32) // 2 + 4 * q, vec_width=4, dtype=T.i32,
                                     cache_modifier=cm))  # fmt: skip
            for q in range(4)
        ]
    v = []
    for i in range_constexpr(32):
        w = words[i // 8][i % 8 // 2]
        v.append((w << 16).bitcast(fx.Float32) if i % 2 == 0 else (w & fx.Int32(-65536)).bitcast(fx.Float32))
    amax = fx.Float32(1e-10)
    for y in v:
        amax = fx.max(amax, abs(y))
    if mine:
        put_mxfp8_group(c, "x8r", "x8rs", t, g, v, amax)
        put_mxfp8_group(c, "x8", "x8s", t, g, v, fx.max(amax, fx.Float32(1e-4)))


# ---------------------------------------------------------------- router
def router_weights(c, task):
    rs = c["rs"]
    tile = task % (rs.experts // ROWS * ROUTER_PARTS) // ROUTER_PARTS
    return router_weight_loads(c["args"]["gate_w"], tile, task % ROUTER_PARTS, c["lane"], c["wave"], TH_NT)


@traced
def stage_router(c, task, wvs):
    """Router task (token group, 16-row tile, K part) -> the part's f32 sums."""
    s, tid, red, lane, wave = c["S"], c["tid"], c["red"], c["lane"], c["wave"]
    rs = c["rs"]
    groups, per = rs.router_groups
    per_group = rs.experts // ROWS * ROUTER_PARTS
    tile, part = task % per_group // ROUTER_PARTS, task % ROUTER_PARTS
    g0 = 0 if groups == 1 else task // per_group * per
    tiles = [
        (plus(g0, j0), min(TILE, s - j0) if groups == 1 else fx.min(fx.Int32(min(TILE, per - j0)), s - plus(g0, j0)))
        for j0 in range(0, per, TILE)
    ]
    if const_expr(c.get("x_xg") is not None):
        # token parallel: the rows from XG's tagged pairs (no XRDY wait). First the wait on
        # one two-pair unit a thread, CTAs and threads spread over the group's rows (their
        # rows' every tag still checked by the polls after): spinning in the polls re-read
        # every router CTA's rows a round (wo_a's input the same: K2 -1.7 us, 2026-10-31)
        nt = s if groups == 1 else per
        spread = tid + c["bid"] * THREADS
        tok = fx.min(plus(g0, spread // (HIDDEN // 4) % nt), s - 1)
        c["poll"]([(c["x_xg"], tok * (HIDDEN // 2) + spread % (HIDDEN // 4) * 2, 2)])
        xs = [router_x_polls(c["poll"], c["x_xg"], part, n, lane, wave, t0) for t0, n in tiles]
    else:
        xs = [router_x_loads(c["args"]["x"], part, n, lane, wave, c["x_cm"], t0) for t0, n in tiles]
    for (t0, n), xvs in zip(tiles, xs):
        router_tile_wmma(wvs, xvs, lane, wave, red)
        lds_barrier()
        if tid < ROWS * n:
            r, tl = tid % ROWS, tid // ROWS
            c["put"](c["logit"], (plus(t0, tl) * rs.experts + tile * ROWS + r) * ROUTER_PARTS + part, router_row_sum(red, r, tl))
        lds_barrier()


# ---------------------------------------------------------------- route
def _score(x):
    """sqrt(log2(1 + exp2(x log2e)) ln2), x itself past 20 (topk_gating)."""
    soft = _hw_f32("llvm.amdgcn.log.f32", fx.Float32(1.0) + hw_exp2(x * fx.Float32(LOG2E))) * fx.Float32(LN2)
    sp = fx.Float32((x > fx.Float32(20.0)).select(x, soft))
    return fx.Float32(fmath.sqrt(sp))


def _rank(e):
    """The wave64 kernel's tie order of expert e: lane e % 64, then e / 64."""
    return (e % 64) * 8 + e // 64


_NEG_INF = float("-inf")
ROUTE_CHUNK = 8  # stage_route: candidates a loop trip


def _ahead(vj, rj, val, rank):
    """Whether (vj, rj) precedes (val, rank) in the picks' total order."""
    return (vj > val) | ((vj == val) & (rj < rank))


FROUTE_LAYERS = 64  # a recorded routing's layers (``_recorded_pick``: the layer modulo this)


def froute_bytes(key) -> int:
    """Scratch ``froute``: FROUTE_LAYERS x S x top-k (id, weight) word pairs."""
    return FROUTE_LAYERS * key.tokens * route_shape(key).topk * 8


def _recorded_pick(c, t, k, topk):
    """Pick k of token t in a recorded routing (scratch ``froute``, a bench's
    ``--route-npz``: ids and final weights of this layer, modulo FROUTE_LAYERS):
    (id, weight)."""
    layer = (c["tag"] - (MOE_TAG0 + 1)) % FROUTE_LAYERS
    at = ((layer * c["S"] + t) * topk + k) * 2
    fw = fx.Vector(bo.buffer_load(rsrc(c["froute"].value), at, vec_width=2, dtype=T.i32))
    return fw[0], fw[1].bitcast(fx.Float32)


@traced
def stage_route(c, t, tl=None):
    """aiter topk_gating for token t, a thread an expert: its score, then its place
    in the total order (value, then ``_rank``: the wave64 kernel's picks). A
    wave's experts first rank among themselves (``readlane``, no LDS): only a
    wave's top-k can be in the top-k (an expert ahead of a top-k one is one
    too), so the top-k ranks exactly among those E / 32 x k candidates (LDS
    ``red``). The top-k write (id, weight) at their place and wave 0's first k
    lanes sum the weights in pick order and publish. Picking one by one (a
    wave's argmax, 3 butterflies a pick, then a merge) took ~10 us at S = 48;
    counting against all E experts in LDS 7.5 us (2026-10-29 stamps). ``herd``:
    the top-(k + 1) published with their scores as they are (``_herd_route``
    drops one and renormalizes)."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    rs = c["rs"]
    herd = c["herd"]
    topk, experts = rs.topk + int(herd), rs.experts
    forced = c.get("froute") is not None
    assert not (forced and herd), "a recorded routing is the model's: no HERD"
    assert experts <= THREADS and experts % WAVE == 0 and topk <= WAVE
    n_c = experts // WAVE * topk  # candidates
    a = c["args"]
    # a thread past the experts repeats the last one's (its loads, no candidate)
    e = fx.min(tid, fx.Int32(experts - 1))
    mine = tid < experts
    bias = ld_f32(a["bias"], e)
    got = c["poll"]([(c["logit"], (t * experts + e) * ROUTER_PARTS + h, 2) for h in (0, 2)])
    if const_expr(tl is not None):
        tl("route.poll")
    parts = got[0] + got[1]
    tot = fx.Float32(0.0)
    for p in range_constexpr(ROUTER_PARTS):
        tot = tot + parts[p].bitcast(fx.Float32)
    sc = _score(bf16_round(tot))
    orig = fx.Float32(fx.isnan(sc).select(fx.Float32(0.0), sc))
    val = fx.Float32(fx.max(sc + bias, fx.Float32(_NEG_INF)))
    rank = _rank(e)
    # the wave's own order: lane j's (value, rank) broadcast
    in_wave = fx.Int32(0)
    for j in range_constexpr(WAVE):
        first = _ahead(readlane(val, j), readlane(rank, j), val, rank)
        in_wave = in_wave + first.select(fx.Int32(1), fx.Int32(0))
    # candidate wave * k + its order in the wave: (value, rank, weight)
    if mine & (in_wave < topk):
        at = red + 3 * (wave * topk + in_wave)
        fx.ptr_store(val, at)
        fx.ptr_store(rank.bitcast(fx.Float32), at + 1)
        fx.ptr_store(orig, at + 2)
    lds_barrier()
    if const_expr(tl is not None):
        tl("route.rank")
    # thread i < n_c: candidate i's place among them (a runtime loop of
    # ROUTE_CHUNK a trip: unrolled, the loads were hoisted and spilled)
    me = red + 3 * fx.min(tid, fx.Int32(n_c - 1))
    cv, cr = fx.ptr_load(me), fx.ptr_load(me + 1).bitcast(fx.Int32)
    ahead = fx.Int32(0)
    full = n_c // ROUTE_CHUNK * ROUTE_CHUNK
    # an opaque trip count: a loop, not 72 unrolled candidates (straight-line code
    # is fetched cold every layer: ~7 us at S = 48 under the HBM stream)
    for j0 in range(0, uniform(fresh(full)), ROUTE_CHUNK):
        for j in range_constexpr(ROUTE_CHUNK):
            at = red + (3 * j0 + 3 * j)
            first = _ahead(fx.ptr_load(at), fx.ptr_load(at + 1).bitcast(fx.Int32), cv, cr)
            ahead = ahead + first.select(fx.Int32(1), fx.Int32(0))
    for j in range_constexpr(full, n_c):
        at = red + 3 * j
        first = _ahead(fx.ptr_load(at), fx.ptr_load(at + 1).bitcast(fx.Int32), cv, cr)
        ahead = ahead + first.select(fx.Int32(1), fx.Int32(0))
    picks = red + 3 * n_c
    if (tid < n_c) & (ahead < topk):
        ce = (cr % 8) * 64 + cr // 8  # ``_rank``'s inverse
        fx.ptr_store(ce.bitcast(fx.Float32), picks + 2 * ahead)
        fx.ptr_store(fx.ptr_load(me + 2), picks + (2 * ahead + 1))
    lds_barrier()
    if const_expr(tl is not None):
        tl("route.pick")
    if wave == 0:
        total = fx.Float32(0.0)
        for k in range_constexpr(topk):
            total = total + fx.ptr_load(picks + (2 * k + 1))
        k = fx.min(lane, fx.Int32(topk - 1))
        my_id = fx.ptr_load(picks + 2 * k).bitcast(fx.Int32)
        my_w = fx.Float32(fx.ptr_load(picks + (2 * k + 1)))
        scale = fx.Float32(SCALE) / fx.max(total, fx.Float32(1e-20))
        # a recorded routing (a bench's ``--route-npz``) published in place of the picks,
        # the logits still awaited and ranked: the served path's time (publishing it
        # without them put the route ~8 us early: K2 97.1 vs 105.2 us, 2026-11-03)
        if const_expr(herd):
            pub_id, pub_w = my_id, my_w
        else:
            pub_id, pub_w = _recorded_pick(c, t, k, topk) if forced else (my_id, my_w * scale)
        if lane < topk:
            for r in range_constexpr(ROUTE_COPIES):
                slot = (r * c["S"] + t) * 2 * topk + lane
                c["put"](c["route"], slot, pub_id)
                c["put"](c["route"], slot + topk, pub_w)
    lds_barrier()


def lane_prefix(v, lane):
    """The wave's inclusive prefix sum of v, lanes ascending."""
    incl = v
    for d in (1, 2, 4, 8, 16):
        moved = lane_gather(incl, fx.max(lane - d, 0))
        incl = incl + (lane >= d).select(moved, fx.Int32(0))
    return incl


@traced
def _herd_route(c, tab):
    """``MoeBuild.herd`` (aiter ``herd_topk``'s finalize): every token's top-(k + 1)
    candidates (ids, scores) from ROUTE into RED; each expert's popularity among the
    step's candidates (counted in the expert bitmaps' words, before ``load_route``
    clears them); each token drops its least popular candidate (ties: the lower
    score, then the lower id) and keeps the other k in candidate order, renormalized
    to SCALE -> ``tab``'s (ids, weights) rows, as ``_poll_route`` leaves them."""
    s, tid = c["S"], c["tid"]
    rs = c["rs"]
    topk, experts = rs.topk, rs.experts
    kc = topk + 1
    m = s * kc
    cand, pop = c["red"], tab + rs.mask0
    assert 2 * m <= WAVES * WAVE * 8 and m <= THREADS
    idxs = [plus(pas, tid) for pas in range(0, 2 * m, THREADS)]
    copy_at = c["bid"] % ROUTE_COPIES * (2 * m)
    got = c["poll"]([(c["route"], copy_at + fx.min(idx, 2 * m - 1), 1) for idx in idxs])
    for i in range_constexpr(len(idxs)):
        if idxs[i] < 2 * m:
            fx.ptr_store(got[i][0].bitcast(fx.Float32), cand + idxs[i])
    for pas in range_constexpr(0, experts, THREADS):
        idx = plus(pas, tid)
        if idx < experts:
            fx.ptr_store(fx.Int32(0), pop + idx)
    lds_barrier()
    if tid < m:
        eid = fx.ptr_load(cand + (tid // kc * 2 * kc + tid % kc)).bitcast(fx.Int32)
        fx.llvm.atomic_add(pop + eid, fx.Int32(1), syncscope="workgroup")
    lds_barrier()
    if tid < s:
        row = cand + tid * 2 * kc
        ids = [fx.ptr_load(row + j).bitcast(fx.Int32) for j in range_constexpr(kc)]
        vals = [fx.Float32(fx.ptr_load(row + (kc + j))) for j in range_constexpr(kc)]
        pops = [fx.ptr_load(pop + ids[j]) for j in range_constexpr(kc)]
        drop, d_id, d_val, d_pop = fx.Int32(0), ids[0], vals[0], pops[0]
        for j in range_constexpr(1, kc):
            lower = (vals[j] < d_val) | ((vals[j] == d_val) & (ids[j] < d_id))
            take = (pops[j] < d_pop) | ((pops[j] == d_pop) & lower)
            drop = take.select(fx.Int32(j), drop)
            d_id = take.select(ids[j], d_id)
            d_val = fx.Float32(take.select(vals[j], d_val))
            d_pop = take.select(pops[j], d_pop)
        total = fx.Float32(0.0)
        for j in range_constexpr(kc):
            total = total + fx.Float32((drop != j).select(vals[j], fx.Float32(0.0)))
        scale = fx.Float32(SCALE) / fx.max(total, fx.Float32(1e-20))
        out = tab + tid * 2 * topk
        for j in range_constexpr(kc):
            at = j - (drop < j).select(fx.Int32(1), fx.Int32(0))
            w = fx.Float32(vals[j] * scale)
            if drop != j:
                fx.ptr_store(ids[j], out + at)
                fx.ptr_store(w.bitcast(fx.Int32), out + (topk + at))
    lds_barrier()


@traced
def _poll_route(c, tab):
    """Every token's (ids, weights) from ROUTE into ``tab``'s rows."""
    s, tid = c["S"], c["tid"]
    n = s * c["rs"].topk
    idxs = [plus(pas, tid) for pas in range(0, 2 * n, THREADS)]
    copy_at = c["bid"] % ROUTE_COPIES * (2 * n)
    got = c["poll"]([(c["route"], copy_at + fx.min(idx, 2 * n - 1), 1) for idx in idxs])
    for i in range_constexpr(len(idxs)):
        if idxs[i] < 2 * n:
            fx.ptr_store(got[i][0], tab + idxs[i])


@traced
def load_route(c, tab):
    """Every token's (ids, weights) into LDS ``tab``, then the union U of the ids
    (ascending), each (t, k)'s slot, each expert's slot, the picks in slot order
    and each slot's first position (``RouteShape``). A pick a thread ORs its
    token's bit into its expert's bitmap; a chunk of 32 experts a wave counts
    and prefixes them. Every CTA derives the same table."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rs = c["rs"]
    topk, max_u, experts = rs.topk, rs.max_u, rs.experts
    n = s * topk
    assert s <= 32 * MASK_WORDS
    if const_expr(c["herd"]):
        _herd_route(c, tab)
    else:
        _poll_route(c, tab)
    for pas in range_constexpr(0, MASK_WORDS * experts, THREADS):
        idx = plus(pas, tid)
        if idx < MASK_WORDS * experts:
            fx.ptr_store(fx.Int32(0), tab + (rs.mask0 + idx))
    lds_barrier()
    if tid < n:
        t = tid // topk
        eid = fx.ptr_load(tab + (t * 2 * topk + tid % topk))
        fx.llvm.atomic_or(tab + (rs.mask0 + eid * MASK_WORDS + t // 32), fx.Int32(1) << (t % 32), syncscope="workgroup")
    lds_barrier()
    base = 2 * n
    chunks = experts // WAVE
    assert chunks <= WAVES
    sums = c["red"]
    e = lane + WAVE * fx.min(wave, chunks - 1)
    count = fx.Int32(0)
    for w in range_constexpr(MASK_WORDS):
        count = count + popcount(fx.ptr_load(tab + (rs.mask0 + e * MASK_WORDS + w)))
    ball = ballot(count != 0)
    incl = lane_prefix(count, lane)
    tiles = (count + TILE - 1) // TILE
    tincl = lane_prefix(tiles, lane) if rs.multi else tiles
    totals = [popcount(ball), lane_gather(incl, WAVE - 1), lane_gather(tincl, WAVE - 1)]
    if (wave < chunks) & (lane == 0):
        for k in range_constexpr(3):
            fx.ptr_store(totals[k].bitcast(fx.Float32), sums + (wave * 3 + k))
    lds_barrier()
    below = [fx.Int32(0), fx.Int32(0), fx.Int32(0)]  # n_u, first, tile0
    for j in range_constexpr(chunks - 1):
        for k in range_constexpr(3):
            v = fx.ptr_load(sums + (j * 3 + k)).bitcast(fx.Int32)
            below[k] = below[k] + (j < wave).select(v, fx.Int32(0))
    n_u, first, tile0 = below
    slot = n_u + lanes_below(ball, lane)
    e_at = fresh(e)
    if wave < chunks:
        fx.ptr_store(slot, tab + (rs.expert_slot0 + e_at))
        if count != 0:
            fx.ptr_store(fx.Int32(e), tab + (base + 1 + slot))
            fx.ptr_store(first + incl - count, tab + (rs.first0 + slot))
        if const_expr(rs.multi):
            tf = tile0 + tincl - tiles
            if count != 0:
                fx.ptr_store(tf, tab + (rs.tile_first0 + slot))
            for j in range_constexpr(rs.slot_tiles):
                if j < tiles:
                    fx.ptr_store(slot, tab + (rs.tile_slot0 + tf + j))
    if (wave == chunks - 1) & (lane == 0):
        n_all = n_u + totals[0]
        fx.ptr_store(n_all, tab + base)
        fx.ptr_store(fx.Int32(n), tab + (rs.first0 + n_all))
        if const_expr(rs.multi):
            fx.ptr_store(tile0 + totals[2], tab + (rs.tile_first0 + n_all))
    lds_barrier()
    if tid < n:
        t = tid // topk
        eid = fx.ptr_load(tab + (t * 2 * topk + tid % topk))
        slot = fx.ptr_load(tab + (rs.expert_slot0 + eid))
        fx.ptr_store(slot, tab + (base + 1 + max_u + tid))
        rank = fx.Int32(0)
        for w in range_constexpr(MASK_WORDS):
            word = fx.ptr_load(tab + (rs.mask0 + eid * MASK_WORDS + w))
            lo = t - 32 * w
            keep = (lo >= 32).select(fx.Int32(-1), (lo > 0).select((fx.Int32(1) << fx.max(lo, 0)) - 1, fx.Int32(0)))
            rank = rank + popcount(word & keep)
        pos = fx.ptr_load(tab + (rs.first0 + slot)) + rank
        fx.ptr_store(tid, tab + (rs.order0 + pos))
    lds_barrier()


# ---------------------------------------------------------------- MXFP8 rows -> LDS
@traced
def load_x8_rows(c, words_mb, codes_mb, xl, xsl, token_of, live, pack=False):
    """MXFP8 rows ``token_of(j)``, j < ``live``, of scratch ``words_mb`` /
    ``codes_mb`` into LDS rows j: the first ``live`` lanes wait on their token's
    X8RDY, then plain 16-byte loads at device scope. ``pack``: the codes four a
    word (``pack_codes``; rows ``lds_row(groups / 4)`` words)."""
    tid = c["tid"]
    if tid < TILE:
        c["poll"]([(c["x8rdy"], token_of(fx.min(tid, live - 1)), 1)])
    lds_barrier()
    for mb, width, dst in ((words_mb, X8_WORDS, xl), (codes_mb, HIDDEN // 32, xsl)):
        quads = width // 4
        packed = pack and dst is xsl
        n_it = -(-TILE * quads // THREADS)
        # X8_BATCH loads in flight at once (rows clamped in bounds; only the
        # stores conditional): a load inside the store's branch waited out
        # one round trip each, 10 in a row (ATT 2026-10-28)
        for b0 in range_constexpr(0, n_it, X8_BATCH):
            its = list(range(b0, min(b0 + X8_BATCH, n_it)))
            vs = []
            for i in its:
                q = tid + THREADS * i
                vs.append(ld_dev4(c[mb], token_of(fx.min(q // quads, live - 1)) * width + q % quads * 4))
            for i, v in zip(its, vs):
                q = tid + THREADS * i
                row = q // quads
                if (q < TILE * quads) & (row < live):
                    if const_expr(packed):
                        fx.ptr_store(pack_codes(v), dst + (row * lds_row(quads) + q % quads))
                    else:
                        fx.ptr_store(v, dst + (row * lds_row(width) + q % quads * 4))
    lds_barrier()


# ---------------------------------------------------------------- shared expert
def _silu_mul(g, u):
    """aiter silu_and_mul with a limit: gate min limit (bf16), up clamped;
    x rcp(1 + exp(-x)) (ocml exp)."""
    g = bf16_round(fx.min(g, fx.Float32(LIMIT)))
    u = fx.max(fx.min(u, fx.Float32(LIMIT)), fx.Float32(-LIMIT))
    act = g * hw_rcp(fx.Float32(1.0) + fx.Float32(fmath.exp(-g)))
    return act * u


@traced
def shared_prefetch(c, task, live):
    """K2b's start: the shared task's gate and up 16-row groups (16 x 5120 FP8,
    each contiguous: ``group_bytes``) -> LDS ``shl``, a tensor op each on waves
    0 and 1, landing through xq, the router and the route."""
    wave = c["wave"]
    nb = group_bytes(HIDDEN)
    if live:
        for g in range_constexpr(2):
            if wave == g:
                rg = task + g * c["d"].shared_tasks
                tdm.load_block(c["shl"] + g * (nb // 4), c["args"]["sgu"] + fx.Int64(rg) * fx.Int64(nb), nb, TH_RT)


@traced
def shared_ready(c):
    """The preloaded groups landed: the issuing waves drain, then the CTA."""
    if c["wave"] < 2:
        tdm.wait(0)
    lds_barrier()


@traced
def stage_shared(c, task, t0, n, scales):
    """Gate rows 16 task .., up rows sh_inter + 16 task .. against tokens t0 ..
    t0 + n -> bf16 -> silu x up -> SMID; an odd task then quantizes its group.
    The weights from LDS (``shared_prefetch``), ``scales`` their dwords (gate, up)."""
    tid, red, d, lane, wave = c["tid"], c["red"], c["d"], c["lane"], c["wave"]
    a = c["args"]
    nb = group_bytes(HIDDEN) // 4
    gops = gemv_fp8_lds_ops(lane, wave, c["shl"], a["sgu_s"], HIDDEN, task, scales=scales[0])
    gemv_fp8_wmmas(lane, wave, HIDDEN, c["xl"], c["xsl"], red, gops, n)
    lds_barrier()
    gate = fx.Float32(0.0)
    if tid < ROWS * n:
        gate = row_sum(red, tid % ROWS, tid // ROWS, range(WAVES))
    lds_barrier()
    uops = gemv_fp8_lds_ops(lane, wave, c["shl"] + nb, a["sgu_s"], HIDDEN, d.shared_tasks + task, scales=scales[1])
    gemv_fp8_wmmas(lane, wave, HIDDEN, c["xl"], c["xsl"], red, uops, n)
    lds_barrier()
    if tid < ROWS * n:
        r = tid % ROWS
        t = plus(t0, tid // ROWS)
        up = row_sum(red, r, tid // ROWS, range(WAVES))
        v = bf16_round(_silu_mul(bf16_round(gate), bf16_round(up)))
        c["put"](c["smid"], t * d.sh_inter + task * ROWS + r, v)
        fx.ptr_store(v, c["pair"] + tid)
    lds_barrier()
    if task % 2 == 1:
        smid_quant(c, task, t0, n)
    lds_barrier()


@traced
def smid_quant(c, task, t0, n):
    """Odd shared task: its 16 columns (LDS ``pair``) and the even task's (SMID)
    are a quantize_fp8 group of 32 (floor 1e-4) -> SMQ / SMC, a thread a token."""
    tid, sh = c["tid"], c["d"].sh_inter
    assert c["d"].shared_tasks % 2 == 0
    if tid < n:
        t = plus(t0, tid)
        got = c["poll"]([(c["smid"], t * sh + (task - 1) * ROWS + r, 1) for r in range(ROWS)])
        vals = [got[r][0].bitcast(fx.Float32) for r in range(ROWS)] + [
            fx.ptr_load(c["pair"] + (tid * ROWS + r)) for r in range(ROWS)
        ]
        amax = fx.Float32(1e-4)
        for y in vals:
            amax = fx.max(amax, abs(y))
        code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
        inv = rcp_pow2(code)
        g = task // 2
        for w in range_constexpr(8):
            word = fp8_pack4(*[clamp_fp8(vals[4 * w + i] * inv) for i in range(4)])
            c["put_words"](c["smq"], t * (sh // 4) + g * 8 + w, [word])
        c["put_words"](c["smc"], t * (sh // 32) + g, [code])


# ---------------------------------------------------------------- MXFP4 operands
FP4_TILE_DWORDS = 4  # a lane's 16 bytes, twice a step


def fp4_expert(w, e, n, k):
    """Expert e's buffer of a packed MoE FP4 weight [E, n, k] (``pack_moe_fp4``).
    ``e`` (wave-uniform: a unit's expert, read from the LDS route table into a
    VGPR) through ``uniform``: a descriptor in VGPRs made every load of it a
    waterfall loop waited out in full (the w2 scales and register tiles,
    s_wait_loadcnt 0 a load: 2026-10-29 ISA)."""
    return rsrc(w + fx.Int64(uniform(e)) * fx.Int64(n * k // 2))


def fp4_scales_expert(ws, e, n, k):
    """Expert e's E8M0 scales of a packed MoE FP4 weight (``fp4_expert``)."""
    return rsrc(ws + fx.Int64(uniform(e)) * fx.Int64(n * k // 32))


def fp4_step_load(r_w, k, rg, st, lane):
    """16-row tile rg's FP4 A operand of 128-K step st (aiter's shuffle 16 x 16:
    a lane's 16 bytes at step 1024 + lane 16 and + 512) -> v8i32."""
    at = rg * (2 * k) + st * 256 + lane * 4  # dwords
    lo = fx.Vector(bo.buffer_load(r_w, at, vec_width=4, dtype=T.i32, cache_modifier=TH_NT))
    hi = fx.Vector(bo.buffer_load(r_w, at + 128, vec_width=4, dtype=T.i32, cache_modifier=TH_NT))
    return fx.Vector.from_elements([lo[q] for q in range(4)] + [hi[q] for q in range(4)], fx.Int32)


def fp4_scale_load(r_ws, k, rg, st, lane):
    """Tile rg's n32k4 scale dword of step st for ``opsel`` 0: lane l < 16 loads
    row 16 (rg % 2) + l of the 32-row block (aiter shares a block's dword across
    two tiles by opsel; an immediate, where rg is traced here)."""
    at = (rg // 2) * (k // 4) + st * 32 + (rg % 2) * 16 + lane % 16
    return fx.Int32(bo.buffer_load(r_ws, at, vec_width=1, dtype=T.i32, cache_modifier=TH_NT))


def pack_codes(cv):
    """Four E8M0 codes (an i32 each, low byte) -> one word, the WMMA's scale
    operand of a 128-K step."""
    return (cv[0] & 0xFF) | ((cv[1] & 0xFF) << 8) | ((cv[2] & 0xFF) << 16) | ((cv[3] & 0xFF) << 24)


def lds_fp8_operand(xl, xsl, k, row, st, lane, packed=False):
    """MXFP8 row ``row`` of an LDS tile (``lds_row`` stride) as the B operand of
    step st (bytes [32 j + 16 h, + 16), h = lane / 16) and its scale dword;
    ``packed``: the codes stored four a word (``pack_codes``), a step's one load."""
    h = lane // 16
    x_row = lds_row(k // 4)
    words = []
    for j in range_constexpr(4):
        v = fx.Vector(fx.ptr_load(xl + (row * x_row + (4 * st + j) * 8 + 4 * h), result_type=fx.Vector.make_type(4, fx.Int32)))
        words += [v[q] for q in range(4)]
    if const_expr(packed):
        sb = fx.Int32(fx.ptr_load(xsl + (row * lds_row(k // 128) + st)))
    else:
        cv = fx.Vector(fx.ptr_load(xsl + (row * lds_row(k // 32) + 4 * st), result_type=fx.Vector.make_type(4, fx.Int32)))
        sb = pack_codes(cv)
    return fx.Vector.from_elements(words, fx.Int32), sb


# ---------------------------------------------------------------- routed ug
# A CTA's units (``moe_shape.ug_range``: contiguous, tile-major): a wave's slice of a
# unit (tile j, K slice ks: 20 contiguous 1 KB K steps of w13) by 16-byte buffer loads
# into VGPRs, UG_CHUNK steps a chunk, two chunks in flight; the unit's scales preloaded
# into LDS a unit ahead (one tensor op). (A TDM ring of LDS slots instead: slower,
# 2026-10-27.)
def ug_steps(rs) -> int:
    """A wave's K steps of a unit: K over the unit's slices (20 at ug_groups 2, 10 at 1)."""
    return HIDDEN // 128 // rs.ug_k_slices


def ug_wave(rs, wave):
    """An ug wave's (tile of the task's 4 ug_groups tiles, K slice)."""
    tiles = rs.ug_tiles
    return wave % tiles, wave // tiles


def ug_red(rs, j, ks):
    """RED's 32-lane slot of tile j's K slice ks partial."""
    return ks * rs.ug_tiles + j


def ug_units(rs, tab):
    """The launch's ug units: |U| (one token tile) or its pick tiles, x parts."""
    n = tile_first(rs, tab, n_union(rs, tab)) if rs.multi else n_union(rs, tab)
    return n * rs.ug_parts


def ug_unit(c, i):
    """Unit i: (expert, first FP4 group, first pick, picks, a lane's LDS x row,
    its token tile: the slot past one token tile, else the pick tile)."""
    rs, tab, lane = c["rs"], c["tab"], c["lane"]
    tile = i // rs.ug_parts
    part0 = i % rs.ug_parts * rs.ug_groups
    if const_expr(rs.multi):
        u = tile_slot(rs, tab, tile)
        first = slot_first(rs, tab, u) + (tile - tile_first(rs, tab, u)) * TILE
        count = slot_first(rs, tab, u + 1) - first
        col = fx.min(lane % 16, count - 1)
    else:
        u = tile
        first = slot_first(rs, tab, u)
        count = slot_first(rs, tab, u + 1) - first
        col = pick_at(rs, tab, first + fx.min(lane % 16, count - 1)) // rs.topk
    return union_expert(rs, tab, u), part0, first, count, col, tile


def ug_slice(c):
    """This wave's 16-row w13 tile of a unit at its first FP4 group (rg = 4 part0
    + j) and its first K step."""
    j, ks = ug_wave(c["rs"], c["wave"])
    return j, ks * ug_steps(c["rs"])


def ug_scales(c, i):
    """Unit i's w13 scale dwords of this wave's slice, a K step each."""
    d = c["d"]
    e, part0, _, _, _, _ = ug_unit(c, i)
    j, st0 = ug_slice(c)
    r_ws = fp4_scales_expert(c["args"]["w13_s"], e, 2 * d.inter, HIDDEN)
    return [fp4_scale_load(r_ws, HIDDEN, 4 * part0 + j, st0 + s, c["lane"]) for s in range(ug_steps(c["rs"]))]


@traced
def ug_prologue(c, lo, hi):
    """The CTA's first unit's scales (``ug_scale_issue``)."""
    if (c["wave"] == 0) & (lo < hi):
        ug_scale_issue(c, lo, 0)


UG_CHUNK = 4  # K steps of a wave's weights a chunk, two chunks in flight


UG_SCALE_WORDS = 4 * (HIDDEN // 4)  # a slot: a unit's 2 ug_groups n32k4 32-row blocks of K bytes, ug_groups <= 2 (20 KB)


def ug_scale_bytes(rs) -> int:
    assert rs.ug_groups <= 2
    return 2 * rs.ug_groups * HIDDEN


def ug_scale_slot(c, slot):
    """LDS slot ``slot`` (of 2) of the register path's preloaded unit scales: in
    the pool past the x tile (the shared stage's region, free by ug)."""
    return c["shl"] + slot * UG_SCALE_WORDS


def ug_scale_issue(c, i, slot):
    """Unit i's w13 scales (its 2 ug_groups contiguous n32k4 blocks) -> ``ug_scale_slot``:
    one tensor op (wave 0)."""
    d = c["d"]
    e, part0, _, _, _, _ = ug_unit(c, i)
    n, k = 2 * d.inter, HIDDEN
    at = c["args"]["w13_s"] + fx.Int64(e) * fx.Int64(n * k // 32) + fx.Int64(2 * part0) * fx.Int64(k)
    tdm.load_block(ug_scale_slot(c, slot), at, ug_scale_bytes(c["rs"]), TH_RT)


def ug_chunk_loads(c, e, part0, i0, scl=None):
    """Steps i0 .. i0 + UG_CHUNK of this wave's slice of a unit (expert e, first
    group part0): the w13 operands (16-byte buffer loads) and scale dwords (from
    the unit's preloaded slot ``scl``, else buffer loads)."""
    lane, d = c["lane"], c["d"]
    j, st0 = ug_slice(c)
    n, k = 2 * d.inter, HIDDEN
    rg = 4 * part0 + j
    r_w, r_ws = fp4_expert(c["args"]["w13"], e, n, k), fp4_scales_expert(c["args"]["w13_s"], e, n, k)

    def scale(st):
        if const_expr(scl is None):
            return fp4_scale_load(r_ws, k, rg, st, lane)
        # ``fp4_scale_load``'s dword, block rg / 2 - 2 part0 = j / 2 of the slot
        return fx.Int32(fx.ptr_load(scl + ((j // 2) * (k // 4) + st * 32 + (j % 2) * 16 + lane % 16)))

    return [(fp4_step_load(r_w, k, rg, st0 + i, lane), scale(st0 + i))
            for i in range(i0, min(i0 + UG_CHUNK, ug_steps(c["rs"])))]  # fmt: skip


def ug_regs(c, e, part0, col, ops, scl=None):
    """This wave's slice of a unit, its first chunk ``ops`` already in flight;
    UG_CHUNK steps computing while the next chunk loads; a step's B operand read
    from LDS one step ahead of its WMMA. ``scl``: the unit's preloaded scales.
    Past one token tile the B operand is read at its step (the 17 VGPRs ahead spill
    S = 48). (The next unit's first chunk loaded under this unit's last: no gain
    and 36 more live VGPRs, 2026-10-27.)"""
    lane = c["lane"]
    _, st0 = ug_slice(c)

    def operand(i):
        return lds_fp8_operand(c["xl"], c["xsl"], HIDDEN, col, st0 + i, lane, packed=True)

    acc = acc8()
    ahead = not c["rs"].multi
    bnext = operand(0) if ahead else None
    steps = ug_steps(c["rs"])
    for i0 in range_constexpr(0, steps, UG_CHUNK):
        nxt = ug_chunk_loads(c, e, part0, i0 + UG_CHUNK, scl) if i0 + UG_CHUNK < steps else None
        for i in range_constexpr(len(ops)):
            bv, sb = bnext if ahead else operand(i0 + i)
            if const_expr(ahead and i0 + i + 1 < steps):
                bnext = operand(i0 + i + 1)
            acc = wmma_scaled(ops[i][0], bv, acc, ops[i][1], sb, FP4, FP8)
        ops = nxt
    return acc


@traced
def ug_epilogue(c, part0, idx, first, count):
    """Thread ``idx`` = (group p, pick t, column i): silu x up of intermediate
    column 32 (part0 + p) + i of the slot's pick t, its group's MXFP8 (a wave:
    one token's 32 columns) -> MID row ``first`` + t."""
    lane, red, d, rs = c["lane"], c["red"], c["d"], c["rs"]
    cols = rs.tile_picks
    p = idx // (cols * 32)
    t = fx.min(idx % (cols * 32) // 32, cols - 1)
    i = idx % 32
    if (idx < rs.ug_groups * cols * 32) & (t < count):
        # column i of group p: tile 4 p + i / 8, its row pair 2 (i % 8)
        j = 4 * p + i // 8
        rows = [2 * (i % 8), 2 * (i % 8) + 1]
        gu = []
        for row in rows:
            at_lane = 16 * (row // 8) + t
            tot = fx.Float32(0.0)
            for ks in range_constexpr(rs.ug_k_slices):
                tot = tot + fx.ptr_load(red + red_index(ug_red(rs, j, ks), at_lane, row % 8))
            gu.append(tot)
        gc = fx.min(gu[0], fx.Float32(LIMIT))
        uc = fx.max(fx.min(gu[1], fx.Float32(LIMIT)), fx.Float32(-LIMIT))
        sig = hw_rcp(fx.Float32(1.0) + hw_exp2(gc * fx.Float32(-LOG2E)))
        v = bf16_round(gc * sig * uc)
        amax = fx.max(butterfly(abs(v), (1, 2, 4, 8, 16), fx.max), fx.Float32(1e-10))
        code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
        q8 = clamp_fp8(v * rcp_pow2(code))
        nb = [lane_gather(q8.bitcast(fx.Int32), fx.min(lane + q, WAVE - 1)).bitcast(fx.Float32) for q in range(1, 4)]
        word = fp8_pack4(q8, *nb)
        row = first + t
        part = part0 + p
        if i % 4 == 0:
            st_dev(word, c["mid"], row * d.mid_words + part * 8 + i // 4)
        if i == 0:
            st_dev(code, c["mids"], row * (d.inter // 32) + part)




@traced
def ug_finish(c, i, part0, first, count, acc, prev, drain=False):
    """Unit i's partials -> RED, the epilogue (silu x up, MXFP8 -> MID). Its
    UGF flag goes out with the next unit's (``prev``: the unit before, -1 none):
    by then every thread's MID stores of ``prev`` drained, so no unit waits on
    its own stores (``ug_flush`` flags the last). ``drain``: wave 0's tensor ops
    (the next unit's scales) land before the barrier."""
    rs, lane, red = c["rs"], c["lane"], c["red"]
    j, ks = ug_wave(rs, c["wave"])
    fx.ptr_store(acc, red + red_index(ug_red(rs, j, ks), lane, 0))
    wait_store(0)
    if const_expr(drain):
        if c["wave"] == 0:
            tdm.wait(0)
    lds_barrier()
    if (c["tid"] == 0) & (prev >= 0):
        c["put"](c["ugf"], prev, 1)
    for pas in range_constexpr(0, rs.ug_groups * rs.tile_picks * 32, THREADS):
        ug_epilogue(c, part0, pas + c["tid"], first, count)
    # RED read before the next unit's partials land
    lds_barrier()


@traced
def ug_flush(c, prev):
    """The CTA's last unit's flag, its MID stores drained."""
    wait_store(0)
    lds_barrier()
    if (c["tid"] == 0) & (prev >= 0):
        c["put"](c["ugf"], prev, 1)


@traced
def ug_tile_x(c, first, count):
    """Past one token tile: the pick tile's token rows (picks first ..) -> LDS x."""
    rs, tab = c["rs"], c["tab"]

    def token(j):
        return pick_at(rs, tab, first + fx.min(j, count - 1)) // rs.topk

    load_x8_rows(c, "x8r", "x8rs", c["xl"], c["xsl"], token, fx.min(count, TILE), pack=True)


@traced
def stage_ug(c, lo, hi):
    """The CTA's ug units [lo, hi) (``ug_regs``): unit i's scales preloaded
    (``ug_scale_issue``, two slots: unit i + 1's lands while i computes;
    ``ug_prologue`` issued the first). (Units claimed from a device-scope pool past a
    static 3/4: +2.7 us at S = 48, 2026-10-31.)"""
    rs, wave = c["rs"], c["wave"]
    if wave == 0:
        tdm.wait(0)
    lds_barrier()
    prev_tile = fx.Int32(-1)
    done = fx.Int32(-1)
    for i in range(lo, hi, 1):
        e, part0, first, count, col, tile = ug_unit(c, i)
        if const_expr(rs.multi):
            if tile != prev_tile:
                ug_tile_x(c, first, count)
        prev_tile = tile
        if (wave == 0) & (i + 1 < hi):
            ug_scale_issue(c, i + 1, (i + 1 - lo) % 2)
        scl = ug_scale_slot(c, (i - lo) % 2)
        acc = ug_regs(c, e, part0, col, ug_chunk_loads(c, e, part0, 0, scl), scl)
        ug_finish(c, i, part0, first, count, acc, done, drain=True)
        done = i
    ug_flush(c, done)


# ---------------------------------------------------------------- down
DOWN_UNIT_COST = 14  # a down unit's w2 tile in pick-row units (fit 2026-10-29: 0.084 vs 0.006 us)


def _down_cost_at(rs, tab, u):
    """The cost of the units before slot u's first: DOWN_UNIT_COST a w2 tile plus
    one a pick row (``slot_first``: the slots' running pick counts)."""
    return DOWN_UNIT_COST * DOWN_TILES * u + DOWN_TILES * slot_first(rs, tab, u)


def _down_split(c, n_u, goal):
    """The first unit whose preceding cost reaches ``goal``: the slot by a lane
    scan (a wave's lanes test candidate slots at once, a ballot counts those
    starting at or before the goal: no serial search), then within the slot."""
    rs, tab, lane = c["rs"], c["tab"], c["lane"]
    below = fx.Int32(0)
    for k in range_constexpr(-(-(rs.max_u + 1) // WAVE)):
        u = fx.min(lane + WAVE * k, n_u)
        ok = (lane + WAVE * k <= n_u) & (_down_cost_at(rs, tab, u) <= goal)
        below = below + popcount(ballot(ok))
    u = fx.min(below - 1, n_u - 1)
    per = DOWN_UNIT_COST + slot_first(rs, tab, u + 1) - slot_first(rs, tab, u)
    r = (goal - _down_cost_at(rs, tab, u) + per - 1) // per
    return u * DOWN_TILES + fx.min(fx.max(r, fx.Int32(0)), fx.Int32(DOWN_TILES))


def down_range_cost(c, bid, n_u):
    """CTA ``bid``'s units [lo, hi) of the |U| x DOWN_TILES, balanced by cost: a
    unit streams its w2 tile and multiplies it by its slot's every pick (MID
    operands, the epilogue, a CONTRIB row each), so a count split left the CTAs
    over busy experts the down's tail (S = 48: a range's picks 216 .. 1971, a
    CTA's down time correlated 0.6 with them, the last CTA 9 us past the median)."""
    rs, tab = c["rs"], c["tab"]
    total = _down_cost_at(rs, tab, n_u)
    lo = (bid == 0).select(fx.Int32(0), _down_split(c, n_u, bid * total // BLOCKS))
    hi = (bid == BLOCKS - 1).select(n_u * DOWN_TILES, _down_split(c, n_u, (bid + 1) * total // BLOCKS))
    return lo, hi


# Expert-major and balanced (``down_range_cost``): CTA b streams units [lo, hi) of
# the |U| x 320 (slot u, 16-row tile r), a slot at a time (its ug flags, its MID rows
# staged in LDS), a wave every WAVES-th unit with its w2 tile straight into VGPRs
# (``down_slots_regs``) -> CONTRIB; then every CTA combines 20 rows. (A TDM ring of
# w2 tiles a wave instead: capped ~9.2 TB/s, K2 +6 us at S = 48, 2026-10-29; its LDS
# plan stays -- ``down_plan``: the region holds the shared down's staging and fixes
# whether the sw2 groups preload.)
DOWN_RINGS = ((16, 3), (16, 2), (8, 3), (8, 2), (4, 2), (2, 2))  # (waves, slots), the first that fits


def unit_bytes(d) -> int:
    """A w2 16-row tile's FP4 bytes (shuffled 16 x 16: one contiguous block)."""
    return ROWS * d.inter // 2


def pad_half(d) -> bool:
    """Whether the last K step's second 64 columns are all padding (inter_real <=
    inter - 64: TP4 576 of 640, TP8 288 of 384). They are the tile's last 512 B
    (a step: K 128 st .. + 64 in its first 512 B, the next 64 in the second)."""
    return d.inter - d.inter_real >= 64


def unit_load_bytes(d) -> int:
    """The bytes of a w2 tile down loads: past the padding half when there is one."""
    return unit_bytes(d) - (512 if pad_half(d) else 0)


def down_flags(c, u):
    """Slot u's ug flags, a lane each (a slot's units x parts; past them the last)."""
    rs, tab, lane = c["rs"], c["tab"], c["lane"]
    if const_expr(not rs.multi):
        n = rs.ug_parts
        return [u * rs.ug_parts + fx.min(lane + WAVE * q, n - 1) for q in range(-(-n // WAVE))]
    per = rs.slot_tiles * rs.ug_parts
    tile0 = tile_first(rs, tab, u)
    tiles = tile_first(rs, tab, u + 1) - tile0
    out = []
    for q in range(-(-per // WAVE)):
        f = fx.min(lane + WAVE * q, per - 1)
        out.append((tile0 + fx.min(f // rs.ug_parts, tiles - 1)) * rs.ug_parts + f % rs.ug_parts)
    return out


def down_scales(c, j):
    """Unit j's w2 scale dwords, a K step each (``fp4_scale_load``)."""
    rs, tab, d = c["rs"], c["tab"], c["d"]
    k = d.inter
    e = union_expert(rs, tab, j // DOWN_TILES)
    r_ws = fp4_scales_expert(c["args"]["w2_s"], e, HIDDEN, k)
    return [fp4_scale_load(r_ws, k, j % DOWN_TILES, st, c["lane"]) for st in range(k // 128)]


def down_issue_sw2(c, t0):
    """The combine's two shared-down 16-row groups (rows 16 t0 .. + 32: one
    contiguous block) -> LDS ``sw2l``: one tensor op, issued by wave 0 at down's start."""
    nb = group_bytes(c["d"].sh_inter)
    tdm.load_block(c["sw2l"], c["args"]["sw2"] + fx.Int64(t0) * fx.Int64(nb), 2 * nb, TH_RT)


def wave_units(c, lo, hi):
    """This wave's first unit and count: lo + wave, + dwaves, .. < hi."""
    nw = c["dwaves"]
    first = lo + c["wave"]
    return first, fx.max(hi - first + (nw - 1), fx.Int32(0)) // nw


@traced
def down_mid_slot(c, u, p0):
    """Slot u's MID rows (its ug flags waited on) -> LDS ``midl`` / ``midsl`` at
    their place from the range's first pick position ``p0`` (a slot's rows never
    overlap another's: no barrier between a slot's units and the next's staging)."""
    tid, wave, tab, rs, d = c["tid"], c["wave"], c["tab"], c["rs"], c["d"]
    pa = slot_first(rs, tab, u)
    rows = slot_first(rs, tab, u + 1) - pa
    if wave == 0:
        c["poll"]([(c["ugf"], f, 1) for f in down_flags(c, u)])
    lds_barrier()
    for mb, width, dst, packed in (("mid", d.mid_words, c["midl"], False), ("mids", d.inter // 32, c["midsl"], True)):
        quads = width // 4
        n_it = -(-c["S"] * quads // THREADS)  # a slot holds at most S picks
        for b0 in range_constexpr(0, n_it, X8_BATCH):
            its = list(range(b0, min(b0 + X8_BATCH, n_it)))
            vs = [ld_dev4(c[mb], (pa + fx.min((tid + THREADS * it) // quads, rows - 1)) * width
                          + (tid + THREADS * it) % quads * 4) for it in its]  # fmt: skip
            for it, v in zip(its, vs):
                q = tid + THREADS * it
                row = q // quads
                if row < rows:
                    at = pa - p0 + row
                    if const_expr(packed):
                        fx.ptr_store(pack_codes(v), dst + (at * lds_row(quads) + q % quads))
                    else:
                        fx.ptr_store(v, dst + (at * lds_row(width) + q % quads * 4))
    lds_barrier()


@traced
def down_slots_seg(c, lo, hi, tl):
    """The register path a slot at a time: each of the range's (<= DOWN_SPAN)
    slots waits on its own ug flags and stages its own MID rows, then its units
    stream (``down_slots_regs``), the next slot's first tiles issued before its
    wait. Waiting on every slot of the range at once held a CTA behind the
    range's last ug unit (S = 48: ug's ends spread ~11 us, down's starts with
    them); MI355's down polls a round's slots only."""
    tab, rs = c["tab"], c["rs"]
    u0, u1 = lo // DOWN_TILES, (hi - 1) // DOWN_TILES
    p0 = slot_first(rs, tab, u0)
    for k in range(u1 - u0 + 1):
        u = u0 + k
        a = fx.max(lo, u * DOWN_TILES)
        b = fx.min(hi, (u + 1) * DOWN_TILES)
        pre = down_regs_first(c, a, b)
        down_mid_slot(c, u, p0)
        if const_expr(tl is not None):
            if k == 0:
                tl("down.prep")
        down_slots_regs(c, a, b, p0, pre)


def down_unit_loads(c, j):
    """Unit j's w2 tile A operands straight into VGPRs: a K step's
    two 16-byte buffer loads (``fp4_step_load``), the padding half not loaded."""
    rs, tab, d, lane = c["rs"], c["tab"], c["d"], c["lane"]
    k = d.inter
    e = union_expert(rs, tab, j // DOWN_TILES)
    r_w = fp4_expert(c["args"]["w2"], e, HIDDEN, k)
    rg = j % DOWN_TILES
    ops = []
    for st in range_constexpr(k // 128):
        if const_expr(st == k // 128 - 1 and pad_half(d)):
            at = rg * (2 * k) + st * 256 + lane * 4
            lo = fx.Vector(bo.buffer_load(r_w, at, vec_width=4, dtype=T.i32, cache_modifier=TH_NT))
            ops.append(fx.Vector.from_elements([lo[q] for q in range(4)] + [fx.Int32(0)] * 4, fx.Int32))
        else:
            ops.append(fp4_step_load(r_w, k, rg, st, lane))
    return ops


@traced
def down_unit(c, j, ops, sc, p0):
    """Unit j = (slot u, tile r) with its w2 tile's A operands ``ops`` (``down_unit_loads``):
    the tile times each of u's pick tiles (MID rows in LDS from pick position p0)
    -> the picks' CONTRIB rows 16 r .., at device scope."""
    lane, tab, d, rs = c["lane"], c["tab"], c["d"], c["rs"]
    k = d.inter
    u, r = j // DOWN_TILES, j % DOWN_TILES
    first = slot_first(rs, tab, u)
    count = slot_first(rs, tab, u + 1) - first
    for j0 in range_constexpr(0, rs.tile_picks if not rs.multi else c["S"], TILE):
        if j0 < count:
            jj = fx.min(lane % 16 + j0, count - 1)
            acc = acc8()
            # a step's B operand read one step ahead of its WMMA
            bnext = lds_fp8_operand(c["midl"], c["midsl"], k, first - p0 + jj, 0, lane, packed=True)
            for st in range_constexpr(k // 128):
                bv, sb = bnext
                if const_expr(st + 1 < k // 128):
                    bnext = lds_fp8_operand(c["midl"], c["midsl"], k, first - p0 + jj, st + 1, lane, packed=True)
                acc = wmma_scaled(ops[st], bv, acc, sc[st], sb, FP4, FP8)
            if lane % 16 + j0 < count:
                # the pick's routing weight applied here (the combine's bf16(v w)),
                # out as tagged bf16 pairs: the combine polls the data itself, no
                # store drain + DDF flag between (a lane's 8 rows, two 16 B stores)
                p = pick_at(rs, tab, first + jj)
                wk = route_w(rs, tab, p // rs.topk, p % rs.topk)
                ys = [bf16_round(acc[q] * wk) for q in range(ACC)]
                at = p * HIDDEN + r * ROWS + 8 * (lane // 16)  # element = word index of its pair
                for h in range_constexpr(2):
                    c["put_bf"](c["contrib"], at + 4 * h, ys[4 * h : 4 * h + 4])


DOWN_PHASE = 97  # stride of a CTA's start rotation within a range (``down_unit_at``)


def down_unit_at(c, lo, hi, pos):
    """The unit at position ``pos`` of [lo, hi) for this CTA: positions rotated by
    a per-CTA phase (a bijection: every unit once, the results unchanged). The
    CTAs' ranges are equal and contiguous, so unrotated their waves read
    ``base_b + k`` tiles in lockstep, the same offsets into equally spaced
    regions (S = 48: a CTA's down time kept 0.6 of its rank across GPUs, not
    explained by its pick tiles or slots)."""
    span = fx.max(hi - lo, fx.Int32(1))
    rot = (c["bid"] * DOWN_PHASE) % span
    return lo + (fx.min(pos, span - 1) + rot) % span


def down_regs_first(c, lo, hi):
    """This wave's first unit's w2 tile and scales (``down_slots_regs``' ``pre``)."""
    j = down_unit_at(c, lo, hi, c["wave"])
    return down_unit_loads(c, j), down_scales(c, j)


@traced
def down_slots_regs(c, lo, hi, p0, pre):
    """This wave's units (positions wave, + dwaves, .. of [lo, hi), ``down_unit_at``)
    with their w2 tiles straight into VGPRs, two sets ping-ponged
    two units an iteration: unit i + 1's set loads while unit i computes, unit
    i + 2's refills unit i's set as soon as it is consumed (``pre``: the first's,
    issued before the MID's wait). One set carried a unit ahead became a copy at
    the back edge waited out in full (s_wait_loadcnt 0 every unit: 4.3 TB/s,
    2026-10-29); the TDM ring tops out at ~9.2 TB/s whatever the op size
    (``probe_tdm --sizes``). Three sets (two units in flight) spilled TP2 S = 17
    (104 VGPRs)."""
    nw = c["dwaves"]
    _, n = wave_units(c, lo, hi)
    pos = c["wave"]
    ops_a, sc_a = pre
    for i in range(0, n, 2):
        p = pos + nw * i
        jb = down_unit_at(c, lo, hi, p + nw)
        ops_b, sc_b = down_unit_loads(c, jb), down_scales(c, jb)
        down_unit(c, down_unit_at(c, lo, hi, p), ops_a, sc_a, p0)
        ja = down_unit_at(c, lo, hi, p + 2 * nw)
        ops_a, sc_a = down_unit_loads(c, ja), down_scales(c, ja)
        if i + 1 < n:
            down_unit(c, jb, ops_b, sc_b, p0)


@traced
def down_shared(c, tasks, tl=None):
    """Each of ``tasks``' rows (consecutive row groups) of the shared expert's FP8
    down projection of its MXFP8 intermediate (SMQ / SMC) for every token, bf16
    -> LDS ``sdl``. Every token's rows staged at once in the down ring (free past
    the slots), then the (token tile, task) GEMVs a round at a time: a GEMV's
    K steps one a wave, as many GEMVs a round as the waves hold. K = sh_inter is
    5 steps: one GEMV at a time left 11 of 16 waves idle, and S = 48's 3 tiles x
    2 tasks ran 6 rounds behind 3 polls (~11 us). The weights from LDS ``sw2l``
    (``down_issue_sw2``) when ``sw2_early``, else loaded into VGPRs here."""
    sw = c["sw2_early"]
    s, tid, red, d, lane, wave = c["S"], c["tid"], c["red"], c["d"], c["lane"], c["wave"]
    a = c["args"]
    k = d.sh_inter
    steps = k_steps(k)
    per = WAVES // steps  # GEMVs a round
    assert per >= 1
    nb = group_bytes(k) // 4
    tiles = token_tiles(s)
    jobs = [(i, q) for i in range(len(tiles)) for q in range(len(tasks))]
    rounds = -(-len(jobs) // per)
    # this wave's GEMV of a round and its K step (a wave past per x steps idle)
    g = fx.min(wave // steps, per - 1)
    st = fx.min(wave - g * steps, steps - 1)
    busy = wave < per * steps

    def job_of(r):
        """Round r's job of this wave: (live, token tile's first row, rows, task)."""
        j = r * per + g
        live = busy & (j < len(jobs))
        jc = fx.min(j, len(jobs) - 1)
        t0 = jc // len(tasks) * TILE
        return live, t0, fx.min(fx.Int32(TILE), s - t0), jc % len(tasks)

    def ops_of(r):
        _, _, _, q = job_of(r)
        if const_expr(sw):
            return gemv_fp8_lds_ops(lane, st, c["sw2l"] + q * nb, a["sw2_s"], k, tasks[0] + q)
        return gemv_fp8_loads(lane, st, a["sw2"], a["sw2_s"], k, tasks[0] + q)

    # global weights in flight through the rows' polls (LDS ones read at use)
    pre = []
    if const_expr(not sw):
        pre = [ops_of(r) for r in range(rounds)]
    xr, sr = lds_row(k // 4), lds_row(k // 32)
    xl = c["ring"]
    xsl = xl + _align32(s * xr)
    load_x8(c, c["smq"], c["smc"], k // 4, k // 32, xl, xsl, 0, s)
    if const_expr(tl is not None):
        tl("dsh.x0")
    for r in range_constexpr(rounds):
        live, t0, nn, _ = job_of(r)
        if live:
            if const_expr(sw):
                ops = ops_of(r)
            else:
                ops = pre[r]
            gemv_fp8_wmmas(lane, st, k, xl + t0 * xr, xsl + t0 * sr, red + g * steps * WAVE * ACC, ops, nn)
        lds_barrier()
        for gg in range_constexpr(per):
            if const_expr(r * per + gg < len(jobs)):
                i, q = jobs[r * per + gg]
                t0c, nc = tiles[i]
                if tid < ROWS * nc:
                    v = row_sum(red + gg * steps * WAVE * ACC, tid % ROWS, tid // ROWS, range(steps))
                    fx.ptr_store(bf16_round(v), c["sdl"] + ((q * s + t0c + tid // ROWS) * ROWS + tid % ROWS))
        lds_barrier()



@traced
def down_combine(c, bid, n_u):
    """Rows [20 bid, + 20) of every token: each pick's weighted contribution
    (``down_unit``'s tagged bf16 pairs in CONTRIB, polled as they land), summed in
    the token's top-k order, + the shared expert's -> bf16 pairs to every rank's
    FFN region."""
    s, tid, rs = c["S"], c["tid"], c["rs"]
    d = c["d"]
    t0, t1 = combine_tiles(bid)
    # token parallel: decided here (a Python local first set inside a traced branch
    # would become one of its results)
    tok_par = bool(c["tok_par"])
    own_n = s
    if const_expr(tok_par):
        own_n = own_count(s, d.tp)
    # four rows (two pairs, 16 B) a thread: the CTA's rows land as one contiguous
    # run of every rank's region (``ffn_region_pair``), whole lines a store
    quad = COMBINE_ROWS // 4
    for pas in range_constexpr(0, s * quad, THREADS):
        idx = plus(pas, tid)
        if idx < s * quad:
            t = idx // quad
            row = COMBINE_ROWS * bid + 4 * (idx % quad)
            got = c["poll"]([(c["contrib"], ((t * rs.topk + k) * HIDDEN + row) // 2, 2) for k in range(rs.topk)])
            routed = [fx.Float32(0.0)] * 4
            for k in range_constexpr(rs.topk):
                vals = [bf_lo(got[k][0]), bf_hi(got[k][0]), bf_lo(got[k][1]), bf_hi(got[k][1])]
                routed = [bf16_round(routed[h] + vals[h]) for h in range(4)]
            vs = []
            for h in range_constexpr(4):
                rr = row + h
                shared = fx.ptr_load(c["sdl"] + (((rr // ROWS - t0) * s + t) * ROWS + rr % ROWS))
                vs.append(bf16_round(routed[h] + shared))
            if const_expr(tok_par):
                # token parallel: to the token's owner alone, at its own index there
                for p in range_constexpr(d.tp):
                    if t // own_n == p:
                        dst = preg(c["peer_addr"](p), c["peer"]["ffn"], "ffn")
                        c["put_bf"](dst, 2 * ffn_region_pair(c["rank"], t - p * own_n, own_n, row), vs)
            else:
                for p in range_constexpr(d.tp):
                    dst = preg(c["peer_addr"](p), c["peer"]["ffn"], "ffn")
                    c["put_bf"](dst, 2 * ffn_region_pair(c["rank"], t, s, row), vs)


@traced
def stage_down(c, bid):
    """The CTA's down range (``down_range_cost``): the shared expert's two down groups
    in flight, the range streamed a slot at a time (``down_slots_seg``: tagged CONTRIB
    pairs), the shared expert's down of the combine's two tiles, then the combine of
    rows [20 bid, + 20). The shared down placed first (overlapping ug's flags) or
    before the route table's wait sat on the path either way (ug ends level across the
    CTAs; the route is ready as the shared stage ends): measured 2026-10-28, kept here.
    (The shared down on the shared CTAs during ug, polled here: ug +4 us under it, K2
    +2.4 us, 2026-10-31.)"""
    tid, wave, tab, rs, tl = c["tid"], c["wave"], c["tab"], c["rs"], c["tl"]
    n_u = n_union(rs, tab)
    lo, hi = down_range_cost(c, bid, n_u)
    t0 = combine_tiles(bid)[0]
    sw2 = c["sw2_early"]
    if const_expr(sw2):
        if wave == 0:
            down_issue_sw2(c, t0)
    down_slots_seg(c, lo, hi, tl)
    # wave 0's sw2 group landed (a wave with no units waited on none of its ops);
    # the CONTRIB pairs carry their tag: no drain, no flag
    if const_expr(sw2):
        if wave == 0:
            tdm.wait(0)
    lds_barrier()
    if const_expr(tl is not None):
        tl("down.slots")
    down_shared(c, [t0, t0 + 1], tl)
    if const_expr(tl is not None):
        tl("down.shared")
    down_combine(c, bid, n_u)


# ---------------------------------------------------------------- reduce
@traced
def reduce_pair(c, task, idx):
    s, tp = c["S"], c["d"].tp
    if idx < s * RED_COLS // 4:
        # four columns (two pairs, one 16 B load a rank) a thread
        t = idx // (RED_COLS // 4)
        col = task * RED_COLS + 4 * (idx % (RED_COLS // 4))
        own = preg(c["sym"], c["peer"]["ffn"], "ffn")
        accs = sum_partials(c["poll"], own, lambda src: ffn_region_pair(src, t, s, col), tp, npairs=2)
        out = rsrc(c["args"]["out"])
        for j in range_constexpr(4):
            bo.buffer_store(to_bf16(accs[j]), out, t * HIDDEN + col + j)


@traced
def stage_reduce(c, task):
    for pas in range_constexpr(0, c["S"] * RED_COLS // 4, THREADS):
        reduce_pair(c, task, plus(pas, c["tid"]))


# ---------------------------------------------------------------- GL2 prefetch
L2PF_DENSE, L2PF_W2 = 1, 2  # ``l2pf`` levels: K2b's dense weights at K2's start; + the down range's w2


@traced
def l2_prefetch_dense(c, key, bid):
    """K2's start (the attention ahead, HBM idle): the GL2 lines of this CTA's
    K2b dense weights, read after the route chain -- its router tasks' gate rows
    (16 rows x a K part), its shared task's gate and up groups and its
    combine's two shared-down groups (``l2`` rules: a CTA's own share only)."""
    rs, d, tid, a = c["rs"], c["d"], c["tid"], c["args"]
    sch = schedule(key)
    part = HIDDEN // ROUTER_PARTS * 2  # a K part of a bf16 gate row
    for task in range(first_task(bid, sch["router"][0]), sch["router"][1], BLOCKS):
        tile = task % (rs.experts // ROWS * ROUTER_PARTS) // ROUTER_PARTS
        base = a["gate_w"] + fx.Int64(tile * ROWS) * (HIDDEN * 2) + fx.Int64(task % ROUTER_PARTS) * part
        l2.rows(base, ROWS, part, HIDDEN * 2, tid)
    unit = first_task(bid, sch["shared"][0])
    if unit < rs.shared_units:
        nb = group_bytes(HIDDEN)
        for g in range_constexpr(2):
            l2.block(a["sgu"] + fx.Int64(unit % d.shared_tasks + g * d.shared_tasks) * nb, nb, tid)
    nb2 = group_bytes(d.sh_inter)
    l2.block(a["sw2"] + fx.Int64(combine_tiles(bid)[0]) * nb2, 2 * nb2, tid)


@traced
def l2_prefetch_down(c, bid):
    """The route table in: the GL2 lines of this CTA's down range, read after
    its ug -- each unit's w2 tile (``unit_load_bytes``) and its 32-row n32k4
    scale block (K bytes), flat (unit, line) a lane."""
    rs, tab, d, tid, a = c["rs"], c["tab"], c["d"], c["tid"], c["args"]
    lo, hi = down_range(bid, n_union(rs, tab))
    per, sper = unit_load_bytes(d) // l2.LINE, d.inter // l2.LINE
    assert unit_load_bytes(d) % l2.LINE == 0 and d.inter % l2.LINE == 0
    total = (hi - lo) * (per + sper)
    for q in range(0, total, THREADS):
        i = q + tid
        if i < total:
            j, r = lo + i // (per + sper), i % (per + sper)
            e = fx.Int64(union_expert(rs, tab, j // DOWN_TILES))
            tile = j % DOWN_TILES
            if r < per:
                l2.prefetch(a["w2"] + e * (HIDDEN * d.inter // 2) + fx.Int64(tile) * unit_bytes(d) + fx.Int64(r) * l2.LINE)
            else:
                l2.prefetch(a["w2_s"] + e * (HIDDEN * d.inter // 32) + fx.Int64(tile // 2) * d.inter
                            + fx.Int64(r - per) * l2.LINE)  # fmt: skip



# ---------------------------------------------------------------- the kernel
def schedule(key: MoeBuild) -> dict:
    """The launch's placed stages, name -> (base CTA, tasks, kind): the table the
    kernel and the host preflight both read. ug and down give each CTA a
    contiguous range of units (``ug_range``, ``down_range``) from their base."""
    s, rs = key.tokens, route_shape(key)
    return {
        "xq": (XQ0, s, "stride"),
        "router": (rs.router0, rs.router_tasks, "stride"),
        "route": (rs.route0, s, "stride"),
        "shared": (rs.shared0, rs.shared_units, "one"),
        # ug: a contiguous range of units a CTA from this base (``ug_range``)
        "ug": (rs.ug0, BLOCKS, "one"),
        "down": (DOWN0, BLOCKS, "stride"),
        "reduce": (RED0, RED_TASKS, "stride"),
    }


@traced
def run_moe(c, key, bid, tl=None, after_shared=None):
    """K2b's stages; ``tl(k)``: a timeline build's stamp of stage point k;
    ``after_shared()``: a merged launch's work nothing here waits on (K2a's FFN
    gates), run past the shared stage, before the route table's wait."""
    s, tid, rs = key.tokens, c["tid"], c["rs"]
    sch = schedule(key)
    # the shared task's gate / up groups in flight from the start (LDS ``shl``),
    # their scale dwords in VGPRs: they land through xq, the router and the route
    shared_tasks = c["d"].shared_tasks
    unit = first_task(bid, sch["shared"][0])
    stask = fx.min(unit, rs.shared_units - 1) % shared_tasks
    shared_prefetch(c, stask, unit < rs.shared_units)
    sgu_s = c["args"]["sgu_s"]
    ssc = (gemv_scale_dwords(c["wave"], sgu_s, HIDDEN, stask), gemv_scale_dwords(c["wave"], sgu_s, HIDDEN, shared_tasks + stask))
    for t in range(first_task(bid, sch["xq"][0]), s, BLOCKS):
        await_x(c, [t])
        stage_xq(c, t)
        publish(c["put"], c["x8rdy"], t, 1, tid == 0)
    if const_expr(tl is not None):
        tl("xq")
    groups, per = rs.router_groups
    per_group = rs.experts // ROWS * ROUTER_PARTS
    for task in range(first_task(bid, sch["router"][0]), sch["router"][1], BLOCKS):
        # the task's weights in flight through x's wait; its token group's rows
        # only (one group: every token), not every token of the step
        wvs = router_weights(c, task)
        if const_expr(groups == 1):
            await_x(c, list(range(s)))
        else:
            g0 = task // per_group * per
            await_x_range(c, g0, fx.min(fx.Int32(per), s - g0), per)
        stage_router(c, task, wvs)
    if const_expr(tl is not None):
        tl("router")
    if const_expr(tl is not None):
        # every CTA's sub-points; a route CTA's own overwrite them
        tl("route.poll")
        tl("route.rank")
        tl("route.pick")
    # the top-k of the logits (a bench's ``--route-npz``: its recorded picks published)
    for t in range(first_task(bid, sch["route"][0]), s, BLOCKS):
        stage_route(c, t, tl)
    if const_expr(tl is not None):
        tl("route")
    assert rs.shared_units <= BLOCKS
    if unit < rs.shared_units:
        if const_expr(not rs.multi):
            load_x8_rows(c, "x8", "x8s", c["xl"], c["xsl"], lambda j: j, s)
            shared_ready(c)
            stage_shared(c, unit, 0, s, ssc)
        else:
            t0 = unit // shared_tasks * TILE
            n = fx.min(fx.Int32(TILE), s - t0)
            load_x8_rows(c, "x8", "x8s", c["xl"], c["xsl"], lambda j: t0 + j, n)
            shared_ready(c)
            stage_shared(c, unit % shared_tasks, t0, n, ssc)
    if const_expr(tl is not None):
        tl("shared")
    if const_expr(after_shared is not None):
        after_shared()
    # every CTA's route table (ug and down), then its ug units' first w13 chunks
    # in flight through the x rows' wait
    load_route(c, c["tab"])
    if const_expr(tl is not None):
        tl("tab")
    n_ug = ug_units(rs, c["tab"])
    # the units over the CTAs without a shared task only (they follow the shared stage's
    # from the ug base): a shared CTA reaches ug ~6 us late (S = 48), and a CTA's ug time
    # hardly grows with its units (latency bound): K2 105.6 -> 104.8 us, 2026-10-31
    lo, hi = ug_range(first_task(bid, sch["ug"][0]), n_ug, BLOCKS - rs.shared_units)
    ug_prologue(c, lo, hi)
    if const_expr(c["l2pf"] >= L2PF_W2):
        l2_prefetch_down(c, bid)
    if const_expr(not rs.multi):
        if lo < hi:
            load_x8_rows(c, "x8r", "x8rs", c["xl"], c["xsl"], lambda j: j, s, pack=True)
    stage_ug(c, lo, hi)
    if const_expr(tl is not None):
        tl("ug")
    c["tl"] = tl  # stage_down's sub-points
    stage_down(c, bid)
    if const_expr(tl is not None):
        tl("down")
    # token parallel: this rank's tokens' sums only (``out`` from its first row)
    cr = c
    if const_expr(c["tok_par"]):  # a statement: an if-expression would be traced as a select
        cr = own_view(c, ("out",))
    for task in range(first_task(bid, sch["reduce"][0]), RED_TASKS, BLOCKS):
        stage_reduce(cr, task)
    if const_expr(tl is not None):
        tl("reduce")


def _lds_fixed(key: MoeBuild) -> dict:
    """K2b's LDS apart from the pool (32-bit words): RED, the route table, the
    shared stage's pair rows, the shared expert's MXFP8 intermediate tile and
    its down rows (two 16-row tiles)."""
    s, rs = key.tokens, route_shape(key)
    d, rows = rs.dims, tile_rows(s)
    return {
        "red": WAVES * WAVE * 8,
        "tab": rs.tab_words,
        "pair": ROWS * s,
        "sxl": rows * lds_row(d.sh_inter // 4),
        "sxsl": rows * lds_row(d.sh_inter // 32),
        "sdl": 2 * s * ROWS,
    }


def _align32(words: int) -> int:
    return -(-words // 32) * 32  # a 128 B line


def shared_slot_words() -> int:
    """The shared expert's gate and up 16-row groups (``shared_prefetch``)."""
    return 2 * group_bytes(HIDDEN) // 4


def sw2_slot_words(d) -> int:
    """The shared expert's two down 16-row groups of a combine (``down_issue_sw2``)."""
    return 2 * group_bytes(d.sh_inter) // 4


def _ug_words(key: MoeBuild) -> tuple:
    """The pool up to ug (words): the MXFP8 x tile (xl, xsl, its end padded to a
    128 B line), then a region holding the shared expert's preloaded gate / up
    groups (shared stage) and, after them in time, ug's two scale slots
    (``ug_scale_slot``)."""
    s = key.tokens
    rows = tile_rows(s)
    xl, xsl = rows * lds_row(X8_WORDS), rows * lds_row(HIDDEN // 32)
    return xl, _align32(xl + xsl) - xl, shared_slot_words()


def _down_words(key: MoeBuild, waves: int, slots: int, sw2: bool = True) -> tuple:
    """The pool in down (words): the w2 rings (``waves`` x ``slots`` tiles), the
    staged MID rows and their codes (padded to a 128 B line), then the shared
    expert's two preloaded down groups (``sw2``: when ``sw2_early``)."""
    rs = route_shape(key)
    d = rs.dims
    ring = waves * slots * unit_bytes(d) // 4
    mid, mids = rs.mid_rows * lds_row(d.mid_words), rs.mid_rows * lds_row(d.inter // 32)
    return ring, mid, _align32(ring + mid + mids) - ring - mid, sw2_slot_words(d) if sw2 else 0


LDS_SLACK = 1024  # bytes: the fields' alignment padding


def _fits(key, pool_words) -> bool:
    return 4 * (sum(_lds_fixed(key).values()) + pool_words) + LDS_SLACK <= LDS_MAX


def down_plan(key: MoeBuild) -> tuple:
    """(waves, slots, sw2) of down: the first of ``DOWN_RINGS`` (a ring's slots
    past 2 given up for the sw2 preload, its waves never) that fits. A third
    slot adds no stream bandwidth (``probe_bw``: 2 x 5 KB a wave holds 160 KB in
    flight a CTA) where the preload saves the shared down ~1 us."""
    for waves, slots in DOWN_RINGS:
        for w, sl, sw2 in ((waves, slots, True), (waves, 2, True), (waves, slots, False)):
            if sl <= slots and _fits(key, sum(_down_words(key, w, sl, sw2=sw2))):
                return w, sl, sw2
    raise ValueError(f"K2b {key}: no down ring fits {LDS_MAX} B of LDS")


def down_ring(key: MoeBuild) -> tuple:
    """(waves, slots) of down's w2 rings (``down_plan``)."""
    return down_plan(key)[:2]


def sw2_early(key: MoeBuild) -> bool:
    """Whether the shared expert's two down groups preload into LDS at down's
    start (``down_plan``; else they load into VGPRs at the shared down)."""
    return down_plan(key)[2]


def lds_words(key: MoeBuild) -> dict:
    words = _lds_fixed(key)
    words["pool"] = max(sum(_ug_words(key)), sum(_down_words(key, *down_ring(key), sw2=sw2_early(key))))
    if not _fits(key, words["pool"]):
        raise ValueError(f"K2b {key}: pool {4 * words['pool']} B past {LDS_MAX} B of LDS")
    return words


def moe_lds(key: MoeBuild):
    i32 = {"tab", "pool", "sxl", "sxsl"}
    fields = {
        k: fx.Array[fx.Int32 if k in i32 else fx.Float32, n, 128 if k == "pool" else 16]
        for k, n in lds_words(key).items()
    }
    return fx.struct(type("K2bLds", (), {"__annotations__": fields, "__module__": __name__}))


def moe_args(x, gate_w, bias, w13, w13_s, w2, w2_s, sgu, sgu_s, sw2, sw2_s, out):
    return {
        "x": x, "gate_w": gate_w, "bias": bias, "w13": w13, "w13_s": w13_s, "w2": w2, "w2_s": w2_s,
        "sgu": sgu, "sgu_s": sgu_s, "sw2": sw2, "sw2_s": sw2_s, "out": out,
    }  # fmt: skip


def moe_context(key, lds, mb, bases, layout, scratch, args, rank, sym, x_ready=False):
    """K2b's stage context over LDS ``lds`` (a ``moe_lds`` struct, peeked).
    ``x_ready``: a launch that also writes x (``layer_post``): xq and the router
    wait on its XRDY flags and read it at device scope."""
    # in the merged K2 the lane math starts from a fresh thread id: K2a's is held
    # from its stages on (standalone, the opaque move cost K2b 28 spilled VGPRs)
    tid = fresh(fx.thread_idx.x) if x_ready else fx.thread_idx.x
    s, rs = key.tokens, route_shape(key)
    # down's LDS plan (``down_plan``): its ring region holds the shared down's staging
    waves, slots = down_ring(key)
    xw, xpad, _ = _ug_words(key)
    ring, midw, midsw, _ = _down_words(key, waves, slots, sw2=False)
    pool = lds.pool.ptr
    c = {
        "S": s, "tid": tid, "bid": fx.block_idx.x, "lane": tid % WAVE, "wave": tid // WAVE, "d": Dims(key.tp),
        "rs": rs, "red": lds.red.ptr, "tab": lds.tab.ptr, "pair": lds.pair.ptr, "sdl": lds.sdl.ptr,
        "xl": pool, "xsl": pool + xw, "ring": pool, "midl": pool + ring,
        "midsl": pool + (ring + midw), "sxl": lds.sxl.ptr, "sxsl": lds.sxsl.ptr,
        "dwaves": WAVES,  # every wave streams its down units (``down_slots_regs``)
        "shl": pool + (xw + xpad), "sw2l": pool + (ring + midw + midsw), "sw2_early": sw2_early(key),
        "put": mb.put, "put_bf": mb.put_bf, "put_words": mb.put_words, "poll": mb.poll, "tag": mb.tag,
        "peer_addr": lambda p: bases[p], "rank": rank, "sym": sym, "peer": peer_offsets(s, key.tp),
        "args": args, "x_ready": x_ready, "x_cm": SCOPE_DEV if x_ready else 0, "tl": None, "l2pf": 0,
        "tok_par": False, "x_xg": None, "herd": key.herd,
    }  # fmt: skip
    for region, (off, _) in layout.items():
        c[region] = sreg(scratch, off, region)
    return c


def build_moe(key: MoeBuild):
    s = key.tokens
    assert 1 <= s <= 48
    if key.timeline:
        raise NotImplementedError("gfx1250 K2b: the timeline build is not ported yet")
    rs = route_shape(key)
    layout = scratch_layout(key)
    Lds = moe_lds(key)
    name = kernel_name("dsv41_moe", s=s, e=rs.experts, k=rs.topk, tp=key.tp, g=key.ug_groups, hd=int(key.herd))

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def moe(
        x: Int64, gate_w: Int64, bias: Int64, w13: Int64, w13_s: Int64, w2: Int64, w2_s: Int64, sgu: Int64,
        sgu_s: Int64, sw2: Int64, sw2_s: Int64, out: Int64, scratch: Int64, sym: Int64, peers: Int64,
        rank: Int32, layer: Int32,
    ):  # fmt: skip
        lds = fx.SharedAllocator().allocate(Lds).peek()
        mb = mailbox(layer, scratch, key.diag_off, layout, tag0=MOE_TAG0)
        bases = [load_ptr64(peers, p) for p in range(key.tp)]
        args = moe_args(x, gate_w, bias, w13, w13_s, w2, w2_s, sgu, sgu_s, sw2, sw2_s, out)
        c = moe_context(key, lds, mb, bases, layout, scratch, args, rank, sym)
        run_moe(c, key, c["bid"])

    tag = cache_tag(name, key)

    @flyc.jit
    def launch(
        x: Int64, gate_w: Int64, bias: Int64, w13: Int64, w13_s: Int64, w2: Int64, w2_s: Int64, sgu: Int64,
        sgu_s: Int64, sw2: Int64, sw2_s: Int64, out: Int64, scratch: Int64, sym: Int64, peers: Int64,
        rank: Int32, layer: Int32, stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = tag  # the build in the disk cache's key (``sources.cache_tag``)
        moe(
            x, gate_w, bias, w13, w13_s, w2, w2_s, sgu, sgu_s, sw2, sw2_s, out, scratch, sym, peers, rank, layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
