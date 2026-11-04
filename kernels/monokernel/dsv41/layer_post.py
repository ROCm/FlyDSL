# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""K2 ``layer_post`` on gfx1250: K2a (``attn_post``) and K2b (``moe``) of one layer
in one launch -- the attention output to the MoE's reduced output.

A CTA runs K2a's stages, then K2b's. K2a's norm stores each token's ``normed``
row at device scope; the CTA then publishes the token's flag in XRDY (its stores
drained first), and K2b's xq and router wait on it and read the row at device
scope -- the buffer is every layer's, so a plain load could hit this XCD's L2
copy of the previous layer's. Each half keeps its own mailbox tag and scratch
regions (K2b's past K2a's: K2a's gate may still poll while another CTA is in
K2b) and one debug record area; their LDS is one union, a barrier between.

A selecting layer's indexer (``index_bound_max`` > 0) runs first in K2a, its
arguments a table at ``iargs`` (``index_score.INDEX_ARGS``, MI355's names; 0 in a
build without it). Not yet: the indexer's FP4 plane and token-parallel build.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32, Int64

from kernels.common import buffer_ops as bo

from kernels.monokernel.dsv41 import l2
from kernels.monokernel.dsv41 import attn_pre as k1
from kernels.monokernel.dsv41 import attn_post as k2a
from kernels.monokernel.dsv41 import index_score as ix
from kernels.monokernel.dsv41 import moe as k2b
from kernels.monokernel.dsv41.layout import MOE_TAG0, POST_TAG0, mailbox
from kernels.monokernel.dsv41.moe_shape import MoeBuild, route_shape
from kernels.monokernel.dsv41.sources import cache_tag, kernel_name
from kernels.monokernel.dsv41.dims import HIDDEN, Q_RANK, ROWS, Dims
from kernels.monokernel.dsv41.token_par import own_count
from kernels.monokernel.execution_gfx1250 import BLOCKS, THREADS, first_task
from kernels.monokernel.ops_gfx1250 import load_ptr64
from kernels.monokernel.sync_gfx1250 import SCOPE_DEV, lds_barrier, preg, publish, realtime, rsrc, traced

_CURRENT_STREAM = fx.Stream(None)
ALIGN = 256
# a timeline build's stamps: point 0 the CTA's start, then the end of each stage
# (its task loop) on that CTA, the REALTIME counter's low 32 bits (100 MHz)
TL_STAGES = (
    "start", "iscore", "isel0.in", "isel0.keys", "isel0.extra", "isel0.passes", "isel0.emit", "isel0.topk", "isel0.pub", "isel",
    "index", "score", "irq.stat", "irq.pv", "irq", "woa.x", "woa", "wob.x8b", "wob.task1", "wob", "seam.red", "reduce+seam", "norm", "gate",
    "xq", "router", "route.poll", "route.rank", "route.pick", "route", "shared", "tab", "ug", "down.prep", "down.slots", "dsh.x0", "down.shared", "down", "reduce",
)  # fmt: skip
TL = {name: k for k, name in enumerate(TL_STAGES)}
TL_POINTS = len(TL_STAGES)

ARGS = (
    *k2a.ARGS[: k2a.ARGS.index("scratch")],
    *[n for n in k2b.ARGS[: k2b.ARGS.index("scratch")] if n != "x"],
    "scratch", "sym", "peers", "rank", "layer", "tl", "iargs", "iact", "sel", "sbound", "nxt",
)  # fmt: skip
# ``nxt``: the next layer's weights (``pf_next``), a table of their pointers in this order
# (0 where that layer has none: ``iq_w`` past a layer without the indexer)
NEXT_W = ("wqkv", "wqb", "iq_w", "woa", "wob", "gate_w", "sgu", "sw2")
PF_NEXT_K1, PF_NEXT_K2A, PF_NEXT_K2B = 1, 2, 3


@dataclass(frozen=True)
class LayerPostBuild:
    tokens: int
    tp: int = 4
    diag_off: int = -1
    ug_groups: int = 2
    # GL2 prefetch (the A/B; K2 +2.5 us at S = 48, 2026-10-31: off): 1 K2a's and K2b's dense
    # weights at K2's start, 2 + the down range's w2 once routed (``moe.L2PF_*``)
    l2pf: int = 0
    tok_par: bool = False  # token parallel (S a multiple of TP): ``token_par``
    # a bench's recorded routing (``--route-npz``): its picks read from scratch ``froute`` and
    # published by ``moe.stage_route`` past its ranking of the logits (the served path's time)
    forced_route: bool = False
    # the indexer (``index_score``), as MI355's: its largest column bound (max_model_len) sizes its
    # mailbox regions and merge levels, 0 builds none; a layer runs it when its ``iact`` (``iargs``) is set
    index_bound_max: int = 0
    index_fp4: bool = False  # the FP4 index plane's scorer: not ported (a build error)
    router_groups: int = 0  # the router's token groups at most (``MoeBuild.router_groups``; 0 auto)
    # the next layer's weights into the GL2 as each CTA ends (``prefetch_next``): 1 the next
    # K1's GEMV weights, 2 + K2a's wo_a / wo_b, 3 + K2b's dense (router, shared expert)
    pf_next: int = 0
    timeline: bool = False

    def halves(self):
        """(K2a's build key, K2b's): one debug area (absolute offsets) for both."""
        return (
            k2a.AttnPostBuild(tokens=self.tokens, tp=self.tp, diag_off=self.diag_off, tok_par=self.tok_par,
                              index_bound_max=self.index_bound_max, index_fp4=self.index_fp4),
            MoeBuild(tokens=self.tokens, tp=self.tp, diag_off=self.diag_off, ug_groups=self.ug_groups,
                     router_groups=self.router_groups),
        )


def moe_base(key: LayerPostBuild) -> int:
    key_a, _ = key.halves()
    return -(-k2a.scratch_bytes(key_a) // ALIGN) * ALIGN


def scratch_layout(key: LayerPostBuild) -> dict:
    """K2a's regions, then K2b's, then XRDY (a token's ``normed`` row written)."""
    key_a, key_b = key.halves()
    out = dict(k2a.scratch_layout(key_a))
    base = moe_base(key)
    moe = k2b.scratch_layout(key_b)
    for name, (off, n) in moe.items():
        out[name] = (base + off, n)
    end = base + max(o + n for o, n in moe.values())
    out["xrdy"] = (end, key.tokens * 8)
    if key.forced_route:
        # a recorded routing (``moe._recorded_pick``): [layer % FROUTE_LAYERS][token][k] (id, weight)
        out["froute"] = (end + key.tokens * 8, k2b.froute_bytes(key_b))
    return out


def scratch_bytes(key: LayerPostBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


@traced
def stamp(tl, k):
    """Thread 0 of the CTA: the counter at point k -> tl[bid][k]."""
    if fx.thread_idx.x == 0:
        bo.buffer_store(fx.Int32(realtime() & 0xFFFFFFFF), rsrc(tl), fx.block_idx.x * TL_POINTS + k)


@traced
def publish_x_tok_par(ca, cb, t, part=0, parts=1):
    """Token parallel: own token t (the step's row ``rank n + t``) normed: the row,
    staged in LDS by the norm (``attn_pre.stage_norm``, ``XG_LDS0``), pushed to every
    rank's XG, this one's too, with K2b's tag: xq and the router poll the tagged
    rows there (``moe.xg_words``, ``router_x_polls``), no XRDY flag, no gather
    stage."""
    tid, rank, tp = ca["tid"], ca["rank"], ca["d"].tp
    n = own_count(ca["S"], tp)
    tg = rank * n + t  # the step's row
    assert k1.XG_LDS0 + k1.XG_LDS_WORDS <= k2a.RED_WORDS
    dsts = [preg(cb["peer_addr"](p), cb["peer"]["xg"], "xg") for p in range(tp)]
    put_words = cb["put_words"]
    lds = fx.recast_iter(fx.Int32, ca["red"]) + k1.XG_LDS0
    tt = fx.min(tid, 255)
    v4 = fx.Vector.make_type(4, fx.Int32)
    for ch in range_constexpr(3):
        col = tt * 8 + 2048 * ch
        w = fx.Vector(fx.ptr_load(lds + (tt * 3 + ch) * 4, result_type=v4))
        live = (tid < 256) & (col < HIDDEN)
        if const_expr(parts > 1):
            # columns [part, part + 1) HIDDEN / parts alone (``stage_norm``'s part)
            live = live & (col >= part * (HIDDEN // parts)) & (col < (part + 1) * (HIDDEN // parts))
        if live:
            for p in range_constexpr(tp):
                # every rank, this one too: pairs (tg H + col) / 2 .. + 3, two 16 B stores
                put_words(dsts[p], (tg * HIDDEN + col) // 2, [w[0], w[1]])
                put_words(dsts[p], (tg * HIDDEN + col) // 2 + 2, [w[2], w[3]])


@traced
def prefetch_next(ca, cb, key, bid, nxt):
    """As this CTA ends: the GL2 lines the same CTA reads of the next layer's weights
    (``NEXT_W`` at ``nxt``) -- K1's wqkv / wq_b / iq tasks (its placement:
    ``attn_pre``'s WQKV0 / WQB0 / IQ0, token parallel as this launch), then by level
    K2a's (``attn_post.l2_prefetch``) and K2b's dense (``moe.l2_prefetch_dense``) on
    the next layer's pointers. A CTA prefetches only what it reads itself (``l2``:
    the GL2 is an AID's); every line is inside its tensor (a task < its count)."""
    tid, s, tp = ca["tid"], key.tokens, key.tp
    w = {n: load_ptr64(nxt, i) for i, n in enumerate(NEXT_W)}
    own = s // tp if key.tok_par else s
    ntiles = len(k1.token_tiles(own))
    if const_expr(ntiles == 1):
        tasks = [first_task(bid, k1.WQKV0)]
    else:
        tasks = k1.tile_tasks(bid, k1.WQKV0, ntiles, k1.WQKV_TASKS)[1]
    for t in tasks:
        if t < k1.WQKV_TASKS:
            l2.block(w["wqkv"] + fx.Int64(t) * (ROWS * HIDDEN), ROWS * HIDDEN, tid)
    wqb0 = first_task(bid, k1.WQB0)
    for i in range_constexpr(Dims(tp).wqb_tasks // BLOCKS):
        l2.block(w["wqb"] + fx.Int64(k1.nth_task(wqb0, i)) * (ROWS * Q_RANK), ROWS * Q_RANK, tid)
    if w["iq_w"] != 0:
        l2.block(w["iq_w"] + fx.Int64(first_task(bid, k1.IQ0)) * (ROWS * Q_RANK), ROWS * Q_RANK, tid)
    key_a, key_b = key.halves()
    if const_expr(key.pf_next >= PF_NEXT_K2A):
        k2a.l2_prefetch({**ca, "args": {**ca["args"], "woa": w["woa"], "wob": w["wob"]}}, key_a, bid)
    if const_expr(key.pf_next >= PF_NEXT_K2B):
        nb = {n: w[n] for n in ("gate_w", "sgu", "sw2")}
        k2b.l2_prefetch_dense({**cb, "args": {**cb["args"], **nb}}, key_b, bid)


def build_layer_post(key: LayerPostBuild):
    s = key.tokens
    assert not key.tok_par or s % key.tp == 0, f"token parallel: {s} rows over TP {key.tp}"
    if key.index_bound_max:
        ix.check_build(key.index_bound_max, key.index_fp4, s, key.tok_par)
    key_a, key_b = key.halves()
    rs = route_shape(key_b)
    layout = scratch_layout(key)
    layout_a = {n: layout[n] for n in k2a.scratch_layout(key_a)}
    layout_b = {n: layout[n] for n in (*k2b.scratch_layout(key_b), "xrdy", "froute") if n in layout}
    LdsA, offsets_a = k2a.post_lds(key_a)
    LdsB = k2b.moe_lds(key_b)
    Lds = fx.union(type("K2Lds", (), {"__annotations__": {"a": LdsA, "b": LdsB}, "__module__": __name__}))
    name = kernel_name("dsv41_layer_post", s=s, e=rs.experts, k=rs.topk, tp=key.tp, g=key.ug_groups, tl=int(key.timeline),
                       pf=key.l2pf, tpar=int(key.tok_par), fr=int(key.forced_route), ib=key.index_bound_max,
                       rg=key.router_groups, pn=key.pf_next)  # fmt: skip

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def layer_post(
        q: Int64, pool: Int64, kmeta: Int64, ring_off: Int32, ring_slots: Int32, sink: Int64,
        qk_scale: Int32, pos: Int64, cos: Int64, sin: Int64, woa: Int64, woa_s: Int64,
        wob: Int64, wob_s: Int64, res_in: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64,
        hc_fn: Int64, hc_scale: Int64, hc_base: Int64, ffn_w: Int64, res_out: Int64,
        post_out: Int64, comb_out: Int64, pre_out: Int64, normed: Int64,
        gate_w: Int64, bias: Int64, w13: Int64, w13_s: Int64, w2: Int64, w2_s: Int64, sgu: Int64,
        sgu_s: Int64, sw2: Int64, sw2_s: Int64, out: Int64,
        scratch: Int64, sym: Int64, peers: Int64, rank: Int32, layer: Int32, tl: Int64, iargs: Int64,
        iact: Int32, sel: Int64, sbound: Int64, nxt: Int64,
    ):  # fmt: skip
        halves = fx.SharedAllocator().allocate(Lds)
        bases = [load_ptr64(peers, p) for p in range(key.tp)]
        args_a = k2a.post_args(
            q, pool, kmeta, ring_off, ring_slots, sink, qk_scale, pos, cos, sin, woa, woa_s, wob, wob_s,
            res_in, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, ffn_w, res_out, post_out,
            comb_out, pre_out, normed,
        )  # fmt: skip
        args_b = k2b.moe_args(normed, gate_w, bias, w13, w13_s, w2, w2_s, sgu, sgu_s, sw2, sw2_s, out)
        mb_a = mailbox(layer, scratch, key.diag_off, layout, tag0=POST_TAG0)
        mb_b = mailbox(layer, scratch, key.diag_off, layout, tag0=MOE_TAG0)
        ca = k2a.post_context(key_a, halves.a.peek(), offsets_a, mb_a, bases, layout_a, scratch, args_a, rank, sym)
        ca["normed_cm"] = SCOPE_DEV
        if const_expr(key.index_bound_max > 0):
            # what every layer reads -- whether it selects (``iact``) and the selection the
            # attention reads (``sel``, ``sbound``) -- are kernel arguments, as MI355's; the
            # indexer's own from the ``iargs`` table where each is read (a cold scalar load
            # ahead of every layer's attention cost a reusing layer ~1.5 us, 2026-10-30)
            ca["args"] = ix.IndexArgs({**ca["args"], "iact": iact, "sel": sel, "sbound": sbound}, iargs)
        cb = k2b.moe_context(key_b, halves.b.peek(), mb_b, bases, layout_b, scratch, args_b, rank, sym, x_ready=True)
        cb["l2pf"] = key.l2pf
        k2a.stamp_start(key_a, scratch, ca["bid"], ca["tid"])
        if const_expr(key.l2pf >= k2b.L2PF_DENSE):
            # this CTA's weights past the attention into the GL2 while HBM idles
            k2a.l2_prefetch(ca, key_a, ca["bid"])
            k2b.l2_prefetch_dense(cb, key_b, cb["bid"])
        tlf = (lambda name: stamp(tl, TL[name])) if key.timeline else None
        if const_expr(key.timeline and key.index_bound_max > 0):
            # the indexer's sub-points (``index_score._tl``): its score tasks and its selection
            ca["itl"] = lambda name: stamp(tl, TL[name]) if name in TL else None
        if const_expr(key.timeline):
            stamp(tl, TL["start"])
        if const_expr(key.tok_par):
            # this rank's tokens' normed rows pushed (tagged) to every rank's XG, where
            # K2b's xq and router poll them (no XRDY, no gather stage)
            cb["tok_par"] = True
            ca["xg"] = ca["peer"]["xg"]  # stage_norm pushes its rows there
            cb["x_xg"] = preg(cb["sym"], cb["peer"]["xg"], "xg")  # xq / the router poll XG
            after_norm = lambda t, part=0: publish_x_tok_par(ca, cb, t, part, k2a.NORM_PARTS)  # noqa: E731
        else:
            after_norm = lambda t: publish(cb["put"], cb["xrdy"], t, 1, ca["tid"] == 0)  # noqa: E731
        k2a.run_attn_post(ca, key_a, ca["bid"], after_norm, tl=tlf, gates=False)
        lds_barrier()  # the LDS union changes hands
        # the FFN gates on the shared expert's CTAs (they take no ug share, ``moe.run_moe``):
        # on their schedule's CTAs 160.. (an ug range each) they held those CTAs' ug back
        # (its critical end, 2026-10-31)
        gate_base = rs.shared0
        # the FFN gates (the next layer's post / comb / pre: nothing in K2 waits on
        # them) past K2b's shared stage, on K2b's RED (free there): off the path
        k2b.run_moe(
            cb, key_b, cb["bid"], tl=tlf, after_shared=lambda: k2a.run_gates(ca, key_a, ca["bid"], red=cb["red"], base=gate_base)
        )
        if const_expr(key.pf_next > 0):
            if nxt != 0:
                prefetch_next(ca, cb, key, ca["bid"], nxt)

    tag = cache_tag(name, key)

    @flyc.jit
    def launch(
        q: Int64, pool: Int64, kmeta: Int64, ring_off: Int32, ring_slots: Int32, sink: Int64,
        qk_scale: Int32, pos: Int64, cos: Int64, sin: Int64, woa: Int64, woa_s: Int64,
        wob: Int64, wob_s: Int64, res_in: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64,
        hc_fn: Int64, hc_scale: Int64, hc_base: Int64, ffn_w: Int64, res_out: Int64,
        post_out: Int64, comb_out: Int64, pre_out: Int64, normed: Int64,
        gate_w: Int64, bias: Int64, w13: Int64, w13_s: Int64, w2: Int64, w2_s: Int64, sgu: Int64,
        sgu_s: Int64, sw2: Int64, sw2_s: Int64, out: Int64,
        scratch: Int64, sym: Int64, peers: Int64, rank: Int32, layer: Int32, tl: Int64, iargs: Int64,
        iact: Int32, sel: Int64, sbound: Int64, nxt: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = tag  # the build in the disk cache's key (``sources.cache_tag``)
        layer_post(
            q, pool, kmeta, ring_off, ring_slots, sink, qk_scale, pos, cos, sin, woa, woa_s,
            wob, wob_s, res_in, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, ffn_w,
            res_out, post_out, comb_out, pre_out, normed,
            gate_w, bias, w13, w13_s, w2, w2_s, sgu, sgu_s, sw2, sw2_s, out,
            scratch, sym, peers, rank, layer, tl, iargs, iact, sel, sbound, nxt,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
