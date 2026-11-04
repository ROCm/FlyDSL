# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""K2a ``attn_post`` on gfx1250: one V4.1 layer from the sparse decode attention
to the MoE input, ported stage for stage from the MI350 mono kernel.

    score  (S x tiles x ``per``): the attention's scores and softmax statistics
           (``attention``) -> AM, AL, AP
    irq    (S x tiles x 16 / groups): PV and combine (``attention``), then the
           inverse RoPE of each head's tail and group-32 FP8 of the fp32 result
           (aiter ``inverse_rope_group_quant``: amax floored at 1e-8, the ceil
           code) -> XO / XOS
    wo_a   (o_rows / 32, two 16-row tiles a task, K split over 8 waves each): the
           grouped FP8 GEMV -> bf16 -> ``quantize_fp8`` of the 32 (floor 1e-4) ->
           X8B / X8BS
    wo_b   (320 x 16 rows): FP8 GEMV -> the rank's bf16 partial -> every rank's
           ATTN peer region (system scope)
    slice  (160): the TP partials of its 32 columns summed in rank order (fp32,
           bf16) -- the attention output -- then the FFN seam as in K1
    norm, gate: ffn_norm -> ``normed`` (the MoE input), then the FFN gates

A selecting layer's indexer (``index_score``: ``index_bound_max`` > 0, the
layer's ``iact`` set) runs first; the attention then waits on each token's
selection (IRDY) and reads its pool rows. Not yet ported: the indexer's FP4
plane and token-parallel build; the standalone K2a build takes no indexer and
no timeline (``layer_post``'s K2 does).
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32, Int64

from kernels.monokernel.dsv41 import attn_pre as k1
from kernels.monokernel.dsv41.attention import (
    COLS,
    GROUPS,
    lds_words,
    only_live,
    pv_split,
    pv_run,
    score_tasks,
    stage_attn_score,
    tile_head,
    tile_task,
)
from kernels.monokernel.dsv41.attention import scratch_pairs as attention_pairs
from kernels.monokernel.dsv41.attn_pre import (
    TILE,
    ld_bf,
    plus,
    poll_copy,
    slice_prefetch,
    stage_gate,
    stage_norm,
    stage_slice,
    tile_rows,
    tile_tasks,
    token_tiles,
)
from kernels.monokernel.dsv41.dims import HEAD_DIM, HEAD_TILE, HIDDEN, ROWS, WOA_ROWS, Dims
from kernels.monokernel.dsv41 import l2, tdm
from kernels.monokernel.dsv41 import index_score as ix
from kernels.monokernel.dsv41.gemv import (
    gemv_fp8_lds_ops, gemv_fp8_loads, gemv_fp8_wmmas, gemv_scale_dwords, group_bytes, lds_row,
)  # fmt: skip
from kernels.monokernel.dsv41.layout import POST_TAG0, mailbox, pair_layout, peer_offsets
from kernels.monokernel.dsv41.mx import FP8_MAX, clamp_fp8, code_ceil, pow2
from kernels.monokernel.dsv41.sources import cache_tag, kernel_name
from kernels.monokernel.dsv41.token_par import K2A_OWN, own_count, own_view, tile_view
from kernels.monokernel.execution_gfx1250 import BLOCKS, THREADS, WAVE, WAVES, first_task
from kernels.monokernel.ops_gfx1250 import (
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    fp8_pack4,
    fresh,
    lane_gather,
    ld_i32,
    load_ptr64,
    uniform,
    xshfl,
)
from kernels.common import buffer_ops as bo
from kernels.monokernel.sync_gfx1250 import DIAG_WORDS, SCOPE_SYS, TH_RT, lds_barrier, preg, realtime, rsrc, sreg, traced
from kernels.monokernel.wmma_gfx1250 import ACC, row_sum

ROPE = k1.ROPE
WOA_SPLIT_K = WAVES * ROWS // WOA_ROWS  # 8: two 16-row tiles, K split over 8 waves each
WOB_TASKS = HIDDEN // ROWS  # 320
RED_WORDS = WAVES * 2 * WAVE * ACC  # K2a's RED (f32 words)
PREFETCH_MAX = TILE  # tokens up to which the seam prefetches across the TP reduce
NORM_PARTS = k1.NORM_PARTS  # a token-parallel norm's column parts (``run_attn_post``)
IRQ_FLOOR = 1e-8
QUANT_FLOOR = 1e-4
SCORE0 = 0
IRQ0 = 64
WOA0 = 160
NORM0 = k1.NORM0
GATE0 = k1.GATE0
_CURRENT_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class AttnPostBuild:
    tokens: int
    index_bound_max: int = 0
    index_fp4: bool = False
    tp: int = 4
    timeline: bool = False
    diag_off: int = -1
    tok_par: bool = False


ARGS = (
    "q", "pool", "kmeta", "ring_off", "ring_slots", "sink", "qk_scale", "pos", "cos", "sin",
    "woa", "woa_s", "wob", "wob_s", "res_in", "post_in", "comb_in", "pre_in", "hc_fn",
    "hc_scale", "hc_base", "ffn_w", "res_out", "post_out", "comb_out", "pre_out", "normed",
    "scratch", "sym", "peers", "rank", "layer",
)  # fmt: skip


def scratch_layout(key: AttnPostBuild) -> dict:
    s, d = key.tokens, Dims(key.tp)
    pairs = {
        "lin": s * HIDDEN // 2,
        "pmix": k1.SLICES * s * k1.PART,
        "xo": s * d.xo_words,
        "xos": s * d.heads * HEAD_DIM // 32,
        **attention_pairs(s, d),
        "x8b": s * d.yb_words,
        "x8bs": s * d.o_rows // 32,
        **(ix.scratch_pairs(s, key.index_bound_max) if key.index_bound_max else {}),
    }
    return pair_layout(pairs.items())


def scratch_bytes(key: AttnPostBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def attn_region_pair(src, t, s, col):
    """Pair of (source rank, token, column) in the ATTN peer region, laid out by
    the sender: [src][wo_b task][token][16 columns], so a task's push to a rank
    is one contiguous run (S x 64 B) written 16 B a lane."""
    return ((src * WOB_TASKS + col // ROWS) * s + t) * (ROWS // 2) + col % ROWS // 2


def sum_partials(poll, own, pair_of, tp, npairs=1):
    """The ``tp`` ranks' bf16 pair partials (``npairs`` adjacent pairs: 2 n
    columns, one load a rank) at ``pair_of(src)`` of ``own``, summed in fp32 in
    rank order 0 .. tp - 1 (every rank in the same order: they agree)."""
    ws = poll([(own, pair_of(src), npairs) for src in range(tp)])
    accs = []
    for q in range_constexpr(npairs):
        accs += [bf_lo(ws[0][q]), bf_hi(ws[0][q])]
    for src in range_constexpr(1, tp):
        for q in range_constexpr(npairs):
            accs[2 * q] = accs[2 * q] + bf_lo(ws[src][q])
            accs[2 * q + 1] = accs[2 * q + 1] + bf_hi(ws[src][q])
    return accs


# ---------------------------------------------------------------- irq
def irq_rope_loads(c, t, dcs):
    """This thread's (cos, sin) of its dim pair in each column group ``dcs`` of
    token t (1, 0 off the rotated tail): loads only."""
    a = c["args"]
    tt = c["tid"] % (COLS // 2)
    pos = ld_i32(a["pos"], 2 * t)
    out = []
    for dc in dcs:
        d = dc * COLS + tt * 2
        live = d >= HEAD_DIM - ROPE
        jj = fx.max((d - (HEAD_DIM - ROPE)) // 2, 0)
        out.append(
            (
                live.select(ld_bf(a["cos"], pos * k1.HALF + jj), fx.Float32(1.0)),
                live.select(ld_bf(a["sin"], pos * k1.HALF + jj), fx.Float32(0.0)),
            )
        )
    return out


@traced
def irq_group(c, t, tile, dc, out, rope):
    """Token t, columns 32 dc .. of a head tile's heads (PV out [head][column] at
    ``out``) -> inverse RoPE, group-32 FP8 -> XO / XOS. 256 threads, a (head, dim
    pair) each; a group of 32 dims is 16 threads of one DPP row."""
    tid, dims = c["tid"], c["d"]
    mine = tid < HEAD_TILE * COLS // 2
    h = fx.min(tid, HEAD_TILE * COLS // 2 - 1) // (COLS // 2)
    head, live = tile_head(dims, tile, h)
    mine = only_live(mine, live)
    tt = tid % (COLS // 2)
    d = dc * COLS + tt * 2
    out = out + h * COLS
    e = fx.ptr_load(out + tt * 2)
    o = fx.ptr_load(out + (tt * 2 + 1))
    cs, sn = rope
    e1 = e * cs + o * sn
    o1 = o * cs - e * sn
    amax = butterfly(fx.max(abs(e1), abs(o1)), (1, 2, 4, 8), fx.max)
    code = code_ceil(fx.max(amax, fx.Float32(IRQ_FLOOR)) * fx.Float32(1.0 / FP8_MAX))
    inv = pow2(254 - code)
    q0, q1 = clamp_fp8(e1 * inv), clamp_fp8(o1 * inv)
    n0, n1 = xshfl(q0, 1), xshfl(q1, 1)
    w = fp8_pack4(q0, q1, n0, n1)
    if mine & (tt % 2 == 0):
        c["put_words"](c["xo"], t * dims.xo_words + (head * HEAD_DIM + d) // 4, [w])
    if mine & (tt % 16 == 0):
        c["put"](c["xos"], t * (dims.heads * HEAD_DIM // 32) + (head * HEAD_DIM + d) // 32, code)


@traced
def stage_attn_irq(c, item, n, nmax):
    """pv items item .. item + n (one token's column groups, n <= nmax, ``pv_split``)
    -> the attention output of those groups (``pv_run``), then each group's inverse
    RoPE and FP8 (``irq_group``) in a runtime loop: one copy of the code; the
    groups' (cos, sin), loaded ahead, picked by a select."""
    tiles = c["d"].head_tiles.count
    q, dc0 = item // GROUPS, item % GROUPS
    t, tile = q // tiles, q % tiles
    ropes = irq_rope_loads(c, t, [fx.min(dc0 + k, GROUPS - 1) for k in range(nmax)])
    pv_run(c, t, tile, dc0, n, nmax)
    if const_expr(c.get("tl") is not None):
        c["tl"]("irq.pv")
    if const_expr(nmax == 1):
        irq_group(c, t, tile, dc0, c["att"]["out"], ropes[0])
    else:
        for i in range(n):
            rope = ropes[0]
            for k in range_constexpr(1, nmax):
                rope = tuple(fx.Float32((i == k).select(ropes[k][r], rope[r])) for r in range(2))
            irq_group(c, t, tile, dc0 + i, c["att"]["out"] + i * HEAD_TILE * COLS, rope)


# ---------------------------------------------------------------- wo_a / wo_b
@traced
def load_rows(c, words_mb, codes_mb, stride_w, off_w, words, stride_g, off_g, groups, t0, n):
    """Tokens t0 .. t0 + n's slice ``[off, off + words)`` of a mailbox of rows
    ``stride`` words apart (and its codes) into LDS rows at the ``lds_row``
    stride; ``n`` traced: a TILE-row copy of which only n rows are polled. Two pairs a
    poll (16 B; strides, offsets and widths even), ``k1.COPY_BATCH`` a round trip, as
    K1's ``load_x8``: one pair a poll, 12 a batch, wo_a's input copy ran 4 round trips
    (8 us from irq's end to the rows in, 2026-10-31)."""
    rows = n if isinstance(n, int) else TILE
    assert words % 2 == 0 and groups % 2 == 0 and stride_w % 2 == 0 and stride_g % 2 == 0
    # the wait on one two-pair unit a thread, CTAs and threads spread over the rows, then
    # the copy (its every tag still checked): a wait in the copy's batches re-read every
    # row a round on every CTA (wo_a: 192 CTAs x 16 rows of 1024 pairs)
    units = (n * words) // 2
    u = (c["tid"] + c["bid"] * THREADS) % units * 2
    c["poll"]([(words_mb, plus(t0, u // words) * stride_w + off_w + u % words, 2)])
    poll_copy(
        c,
        c["tid"],
        THREADS,
        [
            (words_mb, lambda u: plus(t0, u // words) * stride_w + off_w + u % words, rows * words, c["xl"], words),
            (codes_mb, lambda u: plus(t0, u // groups) * stride_g + off_g + u % groups, rows * groups, c["xsl"], groups),
        ],
        live=None if isinstance(n, int) else [n * words, n * groups],
        vec=2,
        batch=k1.COPY_BATCH,
    )
    lds_barrier()


def woa_loads(c, task):
    a = c["args"]
    return gemv_fp8_loads(
        c["lane"], c["wave"], a["woa"], a["woa_s"], c["d"].group_k, task * (WOA_ROWS // ROWS), WOA_SPLIT_K, tiled=True
    )


@traced
def stage_woa(c, task, ops, t0, n):
    """Rows 32 task .. of wo_a -> bf16 -> ``quantize_fp8`` of the 32 -> X8B / X8BS.
    A token's 32 rows are the 32 lanes of one wave."""
    tid, red, d, lane = c["tid"], c["red"], c["d"], c["lane"]
    split = WOA_SPLIT_K
    gemv_fp8_wmmas(lane, c["wave"], d.group_k, c["xl"], c["xsl"], red, ops, n, split, tiled=True)
    lds_barrier()
    mine = tid < WOA_ROWS * n
    tl = fx.min(tid // WOA_ROWS, n - 1)
    t = plus(t0, tl)
    row = tid % WOA_ROWS
    tile = row // ROWS
    y = bf16_round(row_sum(red, row % ROWS, tl, [tile * split + w for w in range(split)]))
    amax = fx.max(butterfly(abs(y), (1, 2, 4, 8, 16), fx.max), fx.Float32(QUANT_FLOOR))
    code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
    q = clamp_fp8(y / pow2(code))
    w = fp8_pack4(
        q, *[lane_gather(q.bitcast(fx.Int32), fx.min(lane + k, WAVE - 1)).bitcast(fx.Float32) for k in (1, 2, 3)]
    )
    if mine & (row % 4 == 0):
        c["put_words"](c["x8b"], t * d.yb_words + (task * WOA_ROWS + row) // 4, [w])
    if mine & (row == 0):
        c["put"](c["x8bs"], t * (d.o_rows // 32) + task, code)
    lds_barrier()


@traced
def woa_tile(c, task, ops, t0, n):
    """wo_a task ``task`` on tokens t0 .. t0 + n: its group's attention output rows
    into LDS, then ``stage_woa``."""
    d = c["d"]
    g = task // (d.woa_tasks // d.groups)
    load_rows(
        c, c["xo"], c["xos"], d.xo_words, g * d.group_k // 4, d.group_k // 4,
        d.heads * HEAD_DIM // 32, g * d.group_k // 32, d.group_k // 32, t0, n,
    )  # fmt: skip
    if const_expr(c.get("tl") is not None):
        c["tl"]("woa.x")  # a timeline build: the group's attention rows in
    stage_woa(c, task, ops, t0, n)


def wob_loads(c, task):
    a = c["args"]
    return gemv_fp8_loads(c["lane"], c["wave"], a["wob"], a["wob_s"], c["d"].o_rows, task)


# ---------------------------------------------------------------- wo_b TDM preload
WOB_TDM_MAX_BYTES = 128 * 1024  # LDS a CTA may hold its wo_b groups in


def wob_slots(key) -> int:
    """The wo_b tasks a CTA may run: two past BLOCKS at one token tile, else
    ``tile_tasks``' count."""
    nt = len(token_tiles(key.tokens))
    return 2 if nt == 1 else -(-WOB_TASKS // (BLOCKS // nt))


def wob_tdm_bytes(key) -> int:
    return wob_slots(key) * group_bytes(Dims(key.tp).o_rows)


def wob_tdm(key) -> bool:
    """Whether wo_b's weights are TDM-preloaded into LDS at the kernel's start
    (else loaded into registers ahead of x8b's wait, as before)."""
    return wob_tdm_bytes(key) <= WOB_TDM_MAX_BYTES


def wob_tasks(key, bid):
    """This CTA's wo_b tasks (traced; may pass WOB_TASKS: the caller guards)
    and, past one token tile, its tile."""
    if len(token_tiles(key.tokens)) == 1:
        first = first_task(bid, schedule(key)["wob"].base)
        return [first, first + BLOCKS], None
    tile, tasks = tile_tasks(bid, schedule(key)["wob"].base, len(token_tiles(key.tokens)), WOB_TASKS)
    return tasks, tile


@traced
def wob_prefetch(c, key, bid):
    """At the kernel's start: wave i < slots TDMs task i's 16-row group (its
    contiguous ``16 o_rows`` bytes) into LDS slot i -- one tensor op a wave, the
    weights' trip hidden under the attention."""
    k = c["d"].o_rows
    nbytes = group_bytes(k)
    tasks, _ = wob_tasks(key, bid)
    wave = c["wave"]
    for i in range_constexpr(len(tasks)):
        if wave == i:
            task = fx.min(tasks[i], WOB_TASKS - 1)
            tdm.load_block(
                c["wobw"] + i * (nbytes // 4), c["args"]["wob"] + fx.Int64(task) * fx.Int64(nbytes), nbytes, TH_RT
            )


@traced
def l2_prefetch(c, key, bid):
    """K2's start (the attention ahead, HBM idle): the GL2 lines of this CTA's
    wo_a task rows (32 rows of group_k: two contiguous groups) and, unless they
    are TDM-preloaded, its wo_b groups -- read after the attention."""
    d, tid, a = c["d"], c["tid"], c["args"]
    sch = schedule(key)
    nb = WOA_ROWS * d.group_k
    for unit in range(first_task(bid, sch["woa"].base), sch["woa"].n, BLOCKS):
        l2.block(a["woa"] + fx.Int64(unit % d.woa_tasks) * nb, nb, tid)
    if const_expr(not (wob_tdm(key) and "wobw" in c)):
        tasks, _ = wob_tasks(key, bid)
        gb = group_bytes(d.o_rows)
        for i in range_constexpr(len(tasks)):
            if tasks[i] < WOB_TASKS:
                l2.block(a["wob"] + fx.Int64(tasks[i]) * gb, gb, tid)


@traced
def wob_ready(c, slots):
    """The preloaded groups landed: each issuing wave drains its op, then the CTA."""
    if c["wave"] < slots:
        tdm.wait(0)
    lds_barrier()


def wob_lds_ops(c, slot, task, scales=None):
    a = c["args"]
    k = c["d"].o_rows
    return gemv_fp8_lds_ops(
        c["lane"], c["wave"], c["wobw"] + slot * (group_bytes(k) // 4), a["wob_s"], k, task, scales=scales
    )


@traced
def stage_wob(c, task, ops, t0, n):
    """Rows 16 task .. of wo_b -> this rank's bf16 partial, pushed to every rank's
    ATTN region at system scope."""
    s, tid, red, d = c["S"], c["tid"], c["red"], c["d"]
    # token parallel (a compile-time choice, decided here: a Python local first
    # set inside a traced branch would become one of its results)
    tok_par = bool(c.get("tok_par", False))
    own_n = s
    if const_expr(tok_par):
        own_n = own_count(s, d.tp)
    gemv_fp8_wmmas(c["lane"], c["wave"], d.o_rows, c["xl"], c["xsl"], red, ops, n)
    lds_barrier()
    # four columns (two pairs, 16 B) a thread: the task's rows land as one
    # contiguous run of the region (``attn_region_pair``), whole lines a store
    if tid < ROWS * n // 4:
        q = tid % (ROWS // 4)
        tl = tid // (ROWS // 4)
        t = plus(t0, tl)
        col = task * ROWS + 4 * q
        vs = [row_sum(red, 4 * q + j, tl, range(WAVES)) for j in range(4)]
        if const_expr(tok_par):
            # token parallel: to the token's owner alone, at its own index there
            # (a reduce-scatter: the owner sums the ranks' partials of its tokens)
            for p in range_constexpr(d.tp):
                if t // own_n == p:
                    dst = preg(c["peer_addr"](p), c["peer"]["attn"], "attn")
                    c["put_bf"](dst, 2 * attn_region_pair(c["rank"], t - p * own_n, own_n, col), vs)
        else:
            for p in range_constexpr(d.tp):
                dst = preg(c["peer_addr"](p), c["peer"]["attn"], "attn")
                c["put_bf"](dst, 2 * attn_region_pair(c["rank"], t, s, col), vs)
    lds_barrier()


# ---------------------------------------------------------------- reduce
@traced
def reduce_attn_pair(c, task, idx):
    s = c["S"]
    # a token tile (``tile_view``): the ATTN region laid out over the whole step
    tt0, s_all = c.get("tt0", 0), c.get("s_all", s)
    c0 = task * k1.COLS
    if idx < s * k1.COLS // 4:
        # four columns (two pairs, one 16 B load a rank) a thread
        t = idx // (k1.COLS // 4)
        col = c0 + 4 * (idx % (k1.COLS // 4))
        own = preg(c["sym"], c["peer"]["attn"], "attn")
        accs = sum_partials(c["poll"], own, lambda src: attn_region_pair(src, k1.plus(tt0, t), s_all, col), c["d"].tp, npairs=2)
        lds = c["pend_lds"] + (t * k1.COLS + col - c0)
        for j in range_constexpr(4):
            fx.ptr_store(bf16_round(accs[j]), lds + j)


@traced
def stage_reduce(c, task):
    s, tid = c["S"], c["tid"]
    for pas in range_constexpr(0, s * k1.COLS // 4, THREADS):
        reduce_attn_pair(c, task, plus(pas, tid))
    lds_barrier()


# ---------------------------------------------------------------- the kernel
@dataclass(frozen=True)
class Stage:
    """A stage's place in the launch: tasks 0 .. n placed from CTA ``base``.
    ``kind``: "stride" (CTA b runs first_task(b, base) + BLOCKS k), "one" (at most
    one task a CTA, n <= BLOCKS), "two" (the task and the task + BLOCKS) or "tiled"
    (``tile_tasks`` over ``tiles`` token tiles)."""

    name: str
    base: int
    n: int
    kind: str = "stride"
    tiles: int = 1


def schedule(key: AttnPostBuild) -> dict:
    """The launch's stages in order: the one table the kernel and the host
    preflight both read."""
    s, d = key.tokens, Dims(key.tp)
    tiles = d.head_tiles.count
    nt = len(token_tiles(s))
    st = [
        Stage("score", SCORE0, s * tiles * score_tasks(s, tiles)),
        # irq: cpt CTAs a (token, head tile), each a run of its column groups (``pv_split``)
        Stage("irq", IRQ0, s * tiles * pv_split(s, tiles)[0]),
        Stage("woa", WOA0, d.woa_tasks * nt, "one" if nt == 1 else "stride"),
        Stage("wob", WOA0 + d.woa_tasks * nt, WOB_TASKS, "two" if nt == 1 else "tiled", nt),
        Stage("slice", 0, k1.SLICES),
        Stage("norm", NORM0, s // d.tp * NORM_PARTS if key.tok_par else s),
        Stage("gate", GATE0, s),
    ]
    return {x.name: x for x in st}


@traced
def run_attn_post(c, key, bid, after_norm=None, tl=None, gates=True):
    """K2a's stages; ``after_norm(t)``: a merged launch's hand-off of token t's
    ``normed`` row (the MoE's input) once its norm stored it; ``tl(k)``: a
    timeline build's stamp of stage point k (``layer_post.TL_STAGES``); ``gates``
    False: the FFN gates left to the caller (``run_gates``: nothing here waits
    on them, the next layer reads them)."""
    s, d = key.tokens, c["d"]
    sch = schedule(key)
    use_tdm = wob_tdm(key) and ("wobw" in c)
    if const_expr(key.index_bound_max > 0):
        # a selecting layer's indexer first (``c["args"]`` an ``index_score.IndexArgs``);
        # the stage LDS then the attention's
        if const_expr(c.get("itl") is not None):
            # every CTA's indexer sub-points (a running indexer's overwrite them)
            c["itl"]("iscore")
            c["itl"]("isel")
            for p in ("in", "keys", "extra", "passes", "emit", "topk", "pub"):
                c["itl"](f"isel0.{p}")
        if c["args"]["iact"] != 0:
            ix.run_indexer(c, s, bid, key.index_bound_max, blocks=False)
            lds_barrier()
    if const_expr(tl is not None):
        tl("index")
    # wo_b's groups into LDS (TDM) once the indexer's latency-bound loads are out of
    # the way: issued at the start, they shared HBM with them; still under the attention
    if const_expr(use_tdm):
        wob_prefetch(c, key, bid)
    # and their scale dwords, not loaded at the task's start: while a wave's are one a
    # task (o_rows <= 128 WAVES: TP >= 4; TP2's two spill 6 VGPRs at S = 17)
    early_scales = use_tdm and d.o_rows // 128 <= WAVES
    if const_expr(early_scales):
        wsc = [gemv_scale_dwords(c["wave"], c["args"]["wob_s"], d.o_rows, fx.min(t, WOB_TASKS - 1))
               for t in wob_tasks(key, bid)[0]]  # fmt: skip
    tiles = d.head_tiles.count
    per = score_tasks(s, tiles)
    for task in range(first_task(bid, sch["score"].base), sch["score"].n, BLOCKS):
        stage_attn_score(c, task, per)
    lds_barrier()
    if const_expr(tl is not None):
        tl("score")
    if const_expr(tl is not None):
        # every CTA's irq sub-points (an irq CTA's own overwrite them)
        tl("irq.stat")
        tl("irq.pv")
    c["tl"] = tl
    # CTA r: (token, tile) r // cpt, its column groups [j G / cpt, (j + 1) G / cpt), j = r % cpt
    # (``pv_split``: one token a CTA, one ``pv_run``)
    cpt, nmax = pv_split(s, tiles)
    r = first_task(bid, sch["irq"].base)
    if r < sch["irq"].n:
        q, j = r // cpt, r % cpt
        g0 = j * GROUPS // cpt
        if const_expr(GROUPS % cpt == 0):
            stage_attn_irq(c, q * GROUPS + g0, nmax, nmax)
        else:
            stage_attn_irq(c, q * GROUPS + g0, (j + 1) * GROUPS // cpt - g0, nmax)
    lds_barrier()
    if const_expr(tl is not None):
        tl("irq")
    if const_expr(key.index_bound_max > 0):
        # a candidate-producing layer's top blocks: read by the next REINDEX layers, not
        # here -- on CTAs 0 .. S that wait from here to wo_b's input (no wo_a task)
        if c["args"]["iact"] != 0:
            ix.run_index_blocks(c, s, bid, key.index_bound_max)
    if const_expr(tl is not None):
        tl("woa.x")  # every CTA's (a wo_a CTA's own overwrites it)
    assert d.woa_tasks <= BLOCKS
    ttiles = token_tiles(s)
    if const_expr(len(ttiles) == 1):
        woa_task = first_task(bid, sch["woa"].base)
        if woa_task < d.woa_tasks:
            ops = woa_loads(c, woa_task)
            woa_tile(c, woa_task, ops, 0, s)
    else:
        for unit in range(first_task(bid, sch["woa"].base), sch["woa"].n, BLOCKS):
            woa_task = unit % d.woa_tasks
            t0 = unit // d.woa_tasks * TILE
            ops = woa_loads(c, woa_task)
            woa_tile(c, woa_task, ops, t0, fx.min(fx.Int32(TILE), s - t0))
    if const_expr(tl is not None):
        tl("woa")
    assert d.yb_words <= d.group_k // 4
    assert BLOCKS < WOB_TASKS <= 2 * BLOCKS
    if const_expr(use_tdm):
        # the groups preloaded at the start (``wob_prefetch``): x8b's rows, the
        # groups' wait, then each task's GEMV off LDS
        tasks, tile = wob_tasks(key, bid)
        if const_expr(tile is None):
            t0, n = 0, s
        else:
            t0 = tile * TILE
            n = fx.min(fx.Int32(TILE), s - t0)
        load_rows(c, c["x8b"], c["x8bs"], d.yb_words, 0, d.yb_words, d.o_rows // 32, 0, d.o_rows // 32, t0, n)
        if const_expr(tl is not None):
            tl("wob.x8b")
        wob_ready(c, len(tasks))
        for i in range_constexpr(len(tasks)):
            if tasks[i] < WOB_TASKS:
                stage_wob(c, tasks[i], wob_lds_ops(c, i, fx.min(tasks[i], WOB_TASKS - 1), wsc[i] if early_scales else None), t0, n)
            if const_expr(tl is not None and i == 0):
                tl("wob.task1")
    elif const_expr(len(ttiles) == 1):
        wob_task = first_task(bid, sch["wob"].base)
        wob_second = wob_task + BLOCKS
        has_second = wob_second < WOB_TASKS
        ops = wob_loads(c, wob_task)
        ops2 = wob_loads(c, has_second.select(wob_second, wob_task))
        load_rows(c, c["x8b"], c["x8bs"], d.yb_words, 0, d.yb_words, d.o_rows // 32, 0, d.o_rows // 32, 0, s)
        if const_expr(tl is not None):
            tl("wob.x8b")
        stage_wob(c, wob_task, ops, 0, s)
        if const_expr(tl is not None):
            tl("wob.task1")
        if has_second:
            stage_wob(c, wob_second, ops2, 0, s)
    else:
        tile, wtasks = tile_tasks(bid, sch["wob"].base, len(ttiles), WOB_TASKS)
        ops = [wob_loads(c, fx.min(t, WOB_TASKS - 1)) for t in wtasks]
        t0 = tile * TILE
        n = fx.min(fx.Int32(TILE), s - t0)
        load_rows(c, c["x8b"], c["x8bs"], d.yb_words, 0, d.yb_words, d.o_rows // 32, 0, d.o_rows // 32, t0, n)
        if const_expr(tl is not None):
            tl("wob.x8b")
        for k in range_constexpr(len(wtasks)):
            if wtasks[k] < WOB_TASKS:
                stage_wob(c, wtasks[k], ops[k], t0, n)
            if const_expr(tl is not None and k == 0):
                tl("wob.task1")
    if const_expr(tl is not None):
        tl("wob")
    # token parallel: this rank's tokens from here on (``after_norm`` gets the
    # token's index among them)
    co = c
    if const_expr(key.tok_par):  # a statement: an if-expression would be traced as a select
        co = own_view(c, K2A_OWN)
    n_own = co["S"]
    if const_expr(tl is not None):
        tl("seam.red")  # a CTA without a slice task: the stage empty
    # past one token tile (S a multiple of 16), the seam's tasks are (32 columns,
    # 16 tokens): 160 x S / 16, every CTA busy, instead of 160 each running all S
    # tokens (19 us at S = 48); the mailboxes stay laid out over the step
    seam_tiles = 1
    if const_expr(n_own > TILE and n_own % TILE == 0):
        seam_tiles = n_own // TILE
    if const_expr(seam_tiles > 1):
        for unit in range(first_task(bid, sch["slice"].base), k1.SLICES * seam_tiles, BLOCKS):
            task = unit % k1.SLICES
            cs = tile_view(co, K2A_OWN, unit // k1.SLICES * TILE, TILE)
            pre = slice_prefetch(cs, task)
            stage_reduce(cs, task)
            if const_expr(tl is not None):
                tl("seam.red")
            stage_slice(cs, task, pre)
    else:
        for task in range(first_task(bid, sch["slice"].base), sch["slice"].n, BLOCKS):
            # the seam's loads issued before the reduce's polls: held across them, so
            # only while they fit (S = 48, TP8 spills 20 VGPRs with them)
            pre = slice_prefetch(co, task) if n_own <= PREFETCH_MAX else None
            stage_reduce(co, task)
            if const_expr(tl is not None):
                tl("seam.red")  # the TP partials of its columns in: the wait, apart from the seam
            stage_slice(co, task, pre)
    if const_expr(tl is not None):
        tl("reduce+seam")
    if const_expr(key.tok_par and after_norm is not None):
        # (token, column part) units: each a row's whole sum of squares, its part's
        # columns quantized and pushed to every rank's XG (``after_norm(t, part)``) --
        # a token a CTA pushed 4 x 20 KB of tagged pairs alone (S = 48: norm 9 us)
        for unit in range(first_task(bid, sch["norm"].base), n_own * NORM_PARTS, BLOCKS):
            stage_norm(co, unit // NORM_PARTS, unit % NORM_PARTS, NORM_PARTS)
            after_norm(unit // NORM_PARTS, unit % NORM_PARTS)
    else:
        for t in range(first_task(bid, sch["norm"].base), n_own, BLOCKS):
            stage_norm(co, t)
            if const_expr(after_norm is not None):
                after_norm(t)
    if const_expr(tl is not None):
        tl("norm")
    if const_expr(gates):
        run_gates(c, key, bid)
    if const_expr(tl is not None):
        tl("gate")


@traced
def run_gates(c, key, bid, red=None, base=None):
    """The FFN seam's gates (``stage_gate``) -> the next layer's post / comb / pre.
    ``red``: an LDS f32 area of 20 x 25 to use for RED (a merged launch runs them
    in K2b, past the shared stage, where K2a's LDS is K2b's). ``base``: the first gate
    task's CTA, if not the schedule's (a merged launch's: CTAs without an ug share)."""
    sch = schedule(key)
    co = c
    if const_expr(key.tok_par):  # a statement: an if-expression would be traced as a select
        co = own_view(c, K2A_OWN)
    cg = co
    if const_expr(red is not None):
        cg = {**co, "red": red}
    for t in range(first_task(bid, sch["gate"].base if base is None else base), cg["S"], BLOCKS):
        stage_gate(cg, t)


def post_lds(key: AttnPostBuild):
    """K2a's LDS as one flat struct (so a merged launch can union it with K2b's):
    RED, XBUF, then a stage area holding one of the attention's buffers, the
    GEMV's x tile, or the seam's rows at a time. -> (struct, stage offsets
    (words) with each buffer's element type)."""
    s, d = key.tokens, Dims(key.tp)
    rows = tile_rows(s)
    aw = lds_words(s, d)
    off, attn = 0, {}
    for k, n in aw.items():
        attn[k] = (off, fx.Float32 if k != "kv" else fx.Int32)
        off += -(-n // 4) * 4
    xl = rows * lds_row(d.group_k // 4)
    gemv = {"xl": (0, fx.Int32), "xsl": (-(-xl // 4) * 4, fx.Int32)}
    gemv_words = gemv["xsl"][0] + rows * lds_row(d.group_k // 32) + 4
    rl, fl = s * k1.KT, k1.MIX * k1.KT
    seam = {"rl": (0, fx.Float32), "fl": (rl, fx.Float32), "pl": (rl + fl, fx.Float32)}
    seam_words = rl + fl + s * k1.COLS
    # the indexer's selection (``topk.lds_topk``): first in the launch, done before the attention
    sel, sel_words = {}, 0
    if key.index_bound_max:
        for k, n in ix.sel_lds_words().items():
            sel[k] = (sel_words, fx.Int32)
            sel_words += -(-n // 4) * 4
    fields = {
        "red": fx.Array[fx.Float32, RED_WORDS, 16],
        "xbuf": fx.Array[fx.Float32, THREADS, 16],
        "stage": fx.Array[fx.Int32, max(off, gemv_words, seam_words, sel_words), 16],
    }
    if wob_tdm(key):
        # wo_b's preloaded 16-row groups (``wob_prefetch``): apart from every stage
        # buffer, the TDM writes land while the attention runs
        fields["wobw"] = fx.Array[fx.Int32, wob_tdm_bytes(key) // 4, 128]
    lds = fx.struct(type("K2aLds", (), {"__annotations__": fields, "__module__": __name__}))
    return lds, {"attn": attn, "gemv": gemv, "seam": seam, "sel": sel}


def post_lds_bytes(key: AttnPostBuild) -> int:
    """K2a's LDS bytes (``post_lds``'s fields, each 128 B aligned at most)."""
    s, d = key.tokens, Dims(key.tp)
    rows = tile_rows(s)
    attn = sum(-(-n // 4) * 4 for n in lds_words(s, d).values())
    xl = -(-(rows * lds_row(d.group_k // 4)) // 4) * 4
    gemv = xl + rows * lds_row(d.group_k // 32) + 4
    seam = s * k1.KT + k1.MIX * k1.KT + s * k1.COLS
    sel = sum(-(-n // 4) * 4 for n in ix.sel_lds_words().values()) if key.index_bound_max else 0
    words = WAVES * 2 * WAVE * ACC + THREADS + max(attn, gemv, seam, sel)
    return 4 * words + (wob_tdm_bytes(key) + 128 if wob_tdm(key) else 0)


def post_context(key, lds, offsets, mb, bases, layout, scratch, args, rank, sym):
    """K2a's stage context over LDS ``lds`` (a ``post_lds`` struct, peeked)."""
    tid, bid = fx.thread_idx.x, fx.block_idx.x
    s, d = key.tokens, Dims(key.tp)
    base = lds.stage.ptr

    def at(off, ty):
        return fx.recast_iter(ty, base + off)

    att = {k: at(o, ty) for k, (o, ty) in offsets["attn"].items()}
    gm = {k: at(o, ty) for k, (o, ty) in offsets["gemv"].items()}
    sl = {k: at(o, ty) for k, (o, ty) in offsets["seam"].items()}
    c = {
        "S": s, "tid": tid, "bid": bid, "lane": tid % WAVE, "wave": tid // WAVE, "d": d,
        "red": lds.red.ptr, "xbuf": lds.xbuf.ptr, "att": att,
        "xl": gm["xl"], "xsl": gm["xsl"], "rl": sl["rl"], "fl": sl["fl"], "pend_lds": sl["pl"],
        "put": mb.put, "put_bf": mb.put_bf, "put_words": mb.put_words, "poll": mb.poll,
        "peer_addr": lambda p: bases[p], "rank": rank, "sym": sym, "peer": peer_offsets(s, key.tp),
        "fold": True, "index": False, "ffn": True, "args": args, "tok_par": key.tok_par,
    }  # fmt: skip
    if wob_tdm(key):
        c["wobw"] = lds.wobw.ptr
    if offsets["sel"]:
        c["sel_lds"] = {k: at(o, ty) for k, (o, ty) in offsets["sel"].items()}
    for region, (off, _) in layout.items():
        c[region] = sreg(scratch, off, region)
    return c


def post_args(q, pool, kmeta, ring_off, ring_slots, sink, qk_scale, pos, cos, sin, woa, woa_s, wob, wob_s, res_in,
              post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, ffn_w, res_out, post_out, comb_out, pre_out,
              normed):  # fmt: skip
    return {
        "q": q, "pool": pool, "kmeta": kmeta, "ring_off": ring_off, "ring_slots": ring_slots,
        "sink": sink, "qk_scale": qk_scale, "pos": pos, "cos": cos, "sin": sin,
        "woa": woa, "woa_s": woa_s, "wob": wob, "wob_s": wob_s, "res_in": res_in,
        "post_in": post_in, "comb_in": comb_in, "pre_in": pre_in, "hc_fn": hc_fn,
        "hc_scale": hc_scale, "hc_base": hc_base, "attn_w": ffn_w, "res_out": res_out,
        "post_out": post_out, "comb_out": comb_out, "pre_out": pre_out, "normed": normed,
    }  # fmt: skip


@traced
def stamp_start(key, scratch, bid, tid):
    """Debug builds: the CTA's start on the 100 MHz clock, after the wait records
    (a CTA that starts late was not resident: a co-residency stall)."""
    if const_expr(key.diag_off >= 0):
        if tid == 0:
            t_start = fx.Int32(realtime() & 0x7FFFFFFF)
            starts = scratch + fx.Int64(key.diag_off + BLOCKS * DIAG_WORDS * 4)
            bo.buffer_store(t_start, rsrc(starts), bid, cache_modifier=SCOPE_SYS)


def build_attn_post(key: AttnPostBuild):
    s = key.tokens
    assert 1 <= s <= k1.MAX_ROWS
    if key.index_bound_max or key.tok_par or key.timeline:
        raise NotImplementedError("gfx1250 K2a: indexer, token-parallel and timeline builds are not ported yet")
    layout = scratch_layout(key)
    Lds, offsets = post_lds(key)
    name = kernel_name("dsv41_attn_post", s=s, tp=key.tp)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def attn_post(
        q: Int64, pool: Int64, kmeta: Int64, ring_off: Int32, ring_slots: Int32, sink: Int64,
        qk_scale: Int32, pos: Int64, cos: Int64, sin: Int64, woa: Int64, woa_s: Int64,
        wob: Int64, wob_s: Int64, res_in: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64,
        hc_fn: Int64, hc_scale: Int64, hc_base: Int64, ffn_w: Int64, res_out: Int64,
        post_out: Int64, comb_out: Int64, pre_out: Int64, normed: Int64, scratch: Int64,
        sym: Int64, peers: Int64, rank: Int32, layer: Int32,
    ):  # fmt: skip
        lds = fx.SharedAllocator().allocate(Lds).peek()
        mb = mailbox(layer, scratch, key.diag_off, layout, tag0=POST_TAG0)
        bases = [load_ptr64(peers, p) for p in range(key.tp)]
        args = post_args(
            q, pool, kmeta, ring_off, ring_slots, sink, qk_scale, pos, cos, sin, woa, woa_s, wob, wob_s,
            res_in, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, ffn_w, res_out, post_out,
            comb_out, pre_out, normed,
        )  # fmt: skip
        c = post_context(key, lds, offsets, mb, bases, layout, scratch, args, rank, sym)
        stamp_start(key, scratch, c["bid"], c["tid"])
        run_attn_post(c, key, c["bid"])

    tag = cache_tag(name, key)

    @flyc.jit
    def launch(
        q: Int64, pool: Int64, kmeta: Int64, ring_off: Int32, ring_slots: Int32, sink: Int64,
        qk_scale: Int32, pos: Int64, cos: Int64, sin: Int64, woa: Int64, woa_s: Int64,
        wob: Int64, wob_s: Int64, res_in: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64,
        hc_fn: Int64, hc_scale: Int64, hc_base: Int64, ffn_w: Int64, res_out: Int64,
        post_out: Int64, comb_out: Int64, pre_out: Int64, normed: Int64, scratch: Int64,
        sym: Int64, peers: Int64, rank: Int32, layer: Int32, stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = tag  # the build in the disk cache's key (``sources.cache_tag``)
        attn_post(
            q, pool, kmeta, ring_off, ring_slots, sink, qk_scale, pos, cos, sin, woa, woa_s,
            wob, wob_s, res_in, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, ffn_w,
            res_out, post_out, comb_out, pre_out, normed, scratch, sym, peers, rank, layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
