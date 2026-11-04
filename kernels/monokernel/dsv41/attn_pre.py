# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""K1 ``attn_pre`` on gfx1250: one V4.1 layer from the attention seam to the
rotated query, ported stage for stage from the MI350 mono kernel.

One launch a layer and rank, ``BLOCKS`` CTAs x ``THREADS`` threads (16 wave32
waves), S = ``tokens`` rows. Stages, each a task table placed round the CTAs
and handed off through the tagged mailbox at device scope:

    slice   (160, 32 hidden columns x 4 streams): the owed post folded into the
            residual (R_new, bf16 out), the collapsed layer input (LIN) and the
            partial mix projection / sum of squares (PMIX)
    norm    (S): attn_norm of LIN -> bf16 -> quantize_fp8 (X8)
    wqkv_a  (112 x 16 rows): FP8 GEMV -> the q latent | kv row (QKV)
    gate    (S): the 160 partials -> post / comb (Sinkhorn) and the next pre
    qkv     (S): q_norm -> MXFP8 (QX8); kv_norm -> RoPE -> FP8 QAT -> ring row
    wq_b    (heads x 512 / 16): FP8 GEMV -> bf16 -> RoPE of a head's tail -> q

What changed from MI350, and why it keeps the numbers:

- Lanes and waves are wave32. A reduction the MI350 kernel runs over 64 lanes
  with a fixed offset order (the q / kv norms, the seam's RMSNorm: the
  reference's own order) runs as ``ops_gfx1250.butterfly64``, the same
  association bit for bit.
- The FP8 projections are scaled WMMAs (``gemv``) and the mix projection an
  f32 WMMA: a matrix instruction's internal K order is the hardware's, so these
  agree with MI350 to rounding, as MI350 agreed with the original kernels.
- Every elementwise step (fold, collapse, quantizers, RoPE, QAT, Sinkhorn) is
  the MI350 kernel's expression for expression.

Not yet ported: the token-parallel and aux builds; a build asking for them is
refused. ``timeline``: each CTA's thread 0 stamps the end of every stage (``TL_STAGES``,
MI355's names).
"""

import math
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from kernels.common import buffer_ops as bo
from kernels.monokernel.dsv41.dims import HC, HEAD_DIM, HIDDEN, KV_DIM, Q_RANK, ROWS, TILE, Dims
from kernels.monokernel.dsv41.gemv import gemv_fp8_loads, gemv_fp8_wmmas, lds_row
from kernels.monokernel.dsv41.token_par import own_view, tile_view
from kernels.monokernel.dsv41.index_query import DIM as IDX_DIM
from kernels.monokernel.dsv41.index_query import HEADS as IDX_HEADS
from kernels.monokernel.dsv41.index_query import (
    IDX_ROPE,
    IDX_ROWS,
    IW_TASKS,
    VWAVES,
    stage_iquant,
    stage_iquant_fp4,
    iw_tasks,
    stage_iw,
)
from kernels.monokernel.dsv41.index_query import scratch_pairs as indexer_pairs
from kernels.monokernel.dsv41.layout import mailbox, pair_layout, peer_offsets
from kernels.monokernel.dsv41.sources import cache_tag, kernel_name
from kernels.monokernel.dsv41.mx import FP8_MAX, clamp_fp8, code_ceil, code_round, fp8_value, pow2
from kernels.monokernel.execution_gfx1250 import BLOCKS, THREADS, WAVE, WAVES, first_task
from kernels.monokernel.ops_gfx1250 import (
    batched_rounds,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    butterfly64,
    div_rn,
    first_rounds,
    fp8_pack4,
    fresh,
    hw_exp2,
    hw_rcp,
    hw_rsq,
    ld_i32,
    load_ptr64,
    xshfl,
)
from kernels.monokernel.sync_gfx1250 import (
    POLL_MAX,
    lds_barrier,
    preg,
    realtime,
    rsrc,
    sreg,
    to_bf16,
    traced,
)
from kernels.monokernel.wmma_gfx1250 import ACC, red_index, row_sum, wmma_f32

MAX_ROWS = 48
K_MIX = HC * HIDDEN
MIX = HC * (HC + 2)  # 24 projections: pre 4 | post 4 | comb 16
QKV_ROWS = Q_RANK + KV_DIM  # 1792
ROPE = 64
HALF = ROPE // 2
EPS = 1e-20
HC_EPS = 1e-6
SINKHORN = 20
POST_MULT = 2.0
LOG2E = 1.0 / math.log(2.0)

COLS = 32  # hidden columns a slice task
SLICES = HIDDEN // COLS  # 160
K1_SEAM = ("res_in", "pend", "post_in", "comb_in", "pre_in", "res_out")  # the seam's per-token arguments
# a token-parallel launch's per-token arguments of slice .. qkv (``own_view``)
K1_OWN = (
    "res_in", "pend", "post_in", "comb_in", "pre_in", "res_out", "post_out", "comb_out", "pre_out", "pos",
    "ring_rows", "normed", "qr_out", "qrs_out",
)  # fmt: skip
KT = HC * COLS  # projection columns a slice task
assert MIX * KT % THREADS == 0
PART = MIX + 1  # a slice's partials a token: the mixes, then the sum of squares
WQKV_TASKS = QKV_ROWS // ROWS  # 112
X8_WORDS = HIDDEN // 4
QX8_WORDS = Q_RANK // 4

GATE0 = SLICES
NORM0 = GATE0 + 8
WQKV0 = NORM0 + 8
QKV0 = (WQKV0 + WQKV_TASKS) % BLOCKS
WQB0 = QKV0 + 8
IW0 = 0
IQ_TASKS = IDX_ROWS // ROWS  # 256
IQ0 = IW_TASKS
IQUANT0 = 0

# a slice's mix projection: f32 16 x 16 x 4 WMMAs, the KT columns split over the
# waves, two 16-row tiles of the 24 mixes
# load_x8's two-pair polls a round trip (every span in one batch: K1 +0.8 us, 2026-10-30 -- the
# copies wait on the data, not on round trips, and a retry of more polls costs more)
COPY_BATCH = 18
NORM_PARTS = 16  # a token-parallel norm's column parts (K1 and K2a: ``stage_norm``'s part)
RED_TILE = WAVES * WAVE * ACC  # a GEMV's partials in RED (``wmma_gfx1250.red_index``)
MIX_K = 4
KW = KT // WAVES  # columns a wave
assert KW % MIX_K == 0

_CURRENT_STREAM = fx.Stream(None)
# a timeline build's stamps: point 0 the CTA's start, then the end of each stage
# (its task loop) on that CTA, the REALTIME counter's low 32 bits (100 MHz)
TL_STAGES = (
    "start", "slice", "norm", "x8.w", "x8.c", "wqkv.x", "wqkv_a", "gate", "qkv", "iw", "q.w", "q.c",
    "wqb.x", "wq_b", "index",
    "ring",
)
TL = {name: k for k, name in enumerate(TL_STAGES)}
TL_POINTS = len(TL_STAGES)


@traced
def tl_stamp(tl, k):
    """Thread 0 of the CTA: the counter at point k -> tl[bid][k]."""
    if fx.thread_idx.x == 0:
        bo.buffer_store(fx.Int32(realtime() & 0xFFFFFFFF), rsrc(tl), fx.block_idx.x * TL_POINTS + k)


@dataclass(frozen=True)
class AttnPreBuild:
    tokens: int
    fold: bool = True
    feeds_index: bool = False
    index_fp4: bool = False
    aux: bool = False
    tp: int = 4
    timeline: bool = False
    diag_off: int = -1
    tok_par: bool = False
    # a layer that feeds no indexer: X8 straight from the slice (``_slice_x8``), no norm
    # stage. Its rstd is a token's scale that q_norm / kv_norm after wqkv_a cancel; only
    # its roundings move (bf16 of x w, the MX scale of an unscaled group). S = 48 TP4: K1
    # 25.2 -> 23.3, K2 104.9 -> 103.7 us, but the ring rows' rel L2 5.2e-2 against the
    # golden (2e-2 budget: the FP8 QAT flips with the rounding), 2026-11-04: off
    slice_x8: bool = False
    # token parallel, a layer feeding the indexer: iw (the index weights' projection) at the
    # norm's end on the CTAs wqkv_a and qkv leave idle, not after qkv (it needs only the
    # normed rows; on the critical path there ~7.6 us). The 40-layer mean unmoved (8 of
    # them select), 2026-11-04: off
    iw_early: bool = False


ARGS = (
    "res_in", "pend", "post_in", "comb_in", "pre_in",
    "hc_fn", "hc_scale", "hc_base", "attn_w", "wqkv", "wqkv_s", "qn_w", "kvn_w",
    "wqb", "wqb_s", "cos", "sin", "pos", "ring", "ring_rows", "ring_off",
    "res_out", "post_out", "comb_out", "pre_out", "q_out", "normed", "qr_out",
    "qrs_out", "aux", "iq_w", "iq_ws", "iw_w", "iq_out", "iqs_out", "iw_out",
    "scratch", "rank", "layer", "tl", "sym", "peers",
)  # fmt: skip


def scratch_layout(tokens: int) -> dict:
    """Mailbox region -> (byte offset, bytes) in K1's scratch; 8 B a pair."""
    pairs = {
        "lin": tokens * HIDDEN // 2,
        "pmix": SLICES * tokens * PART,
        "x8": tokens * X8_WORDS,
        "x8s": tokens * HIDDEN // 32,
        "qkv": tokens * QKV_ROWS,
        "qx8": tokens * QX8_WORDS,
        "qx8s": tokens * Q_RANK // 32,
        **indexer_pairs(tokens, HIDDEN),
    }
    return pair_layout(pairs.items())


def scratch_bytes(tokens: int) -> int:
    return sum(n for _, n in scratch_layout(tokens).values())


def fma(a, b, c):
    return fx.Float32(fmath.fma(a, b, c))


def sigmoid(z):
    """Triton's ``tl.sigmoid``: 1 / (1 + exp2(-z log2(e)))."""
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


def ld_bf(ptr, i):
    return fx.Float32(fx.BFloat16(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16)))


def nth_task(first, i):
    return first if i == 0 else first + BLOCKS * i


def token_tiles(s):
    return [(t0, min(TILE, s - t0)) for t0 in range(0, s, TILE)]


def tile_rows(s):
    return min(s, TILE)


def lds_at(u, row):
    """Element u of a span of rows ``row`` wide, at the ``lds_row`` stride."""
    return u + u // row * 4


def plus(t0, v):
    return v if isinstance(t0, int) and t0 == 0 else v + t0


def tile_tasks(bid, base, ntiles, ntasks):
    b = first_task(bid, base)
    tile = b % ntiles
    per_tile = (BLOCKS - tile + ntiles - 1) // ntiles
    most = -(-ntasks // (BLOCKS // ntiles))
    return tile, [b // ntiles + k * per_tile for k in range(most)]


# ---------------------------------------------------------------- slice
@traced
def slice_input(c, task, idx):
    """Thread ``idx``'s pair of the layer input (token idx / 16, columns
    32 task + 2 (idx % 16) ..)."""
    s, rl = c["S"], c["rl"]
    tt0 = c.get("tt0", 0)  # a token tile's first token of the step (``tile_view``)
    sx = c.get("slice_x8", False)  # a plain flag, read before the traced ``if``
    a = c["args"]
    c0 = task * COLS
    if idx < s * COLS // 2:
        t = idx // (COLS // 2)
        cc = 2 * (idx % (COLS // 2))
        outs = []
        for d in range_constexpr(2):
            acc = fx.Float32(0.0)
            for h in range_constexpr(HC):
                rb = fx.ptr_load(rl + (t * KT + h * COLS + cc + d))
                acc = acc + ld_f32(a["pre_in"], t * HC + h) * rb
            outs.append(acc)
        c["put_bf"](c["lin"], plus(tt0, t) * HIDDEN + c0 + cc, outs)
        if const_expr(sx):
            _slice_x8(c, plus(tt0, t), c0, cc, outs)


@traced
def _slice_x8(c, t, c0, cc, xs):
    """``AttnPreBuild.slice_x8``: token t's columns c0 + cc, + 1 (``xs``, f32) -> the
    group's X8 word and code, as ``stage_norm`` quantizes but of bf16(bf16(x) w) (no
    rstd: q_norm / kv_norm cancel it). A token's 32 columns are 16 consecutive lanes
    (the slice's pairs): the group's amax over xor 1..8, a word from a lane and its
    xor-1 partner (columns cc .. cc + 3, cc % 4 = 0) and the group's code (X8S)."""
    a = c["args"]
    ys = [bf16_round(bf16_round(xs[d]) * ld_bf(a["attn_w"], c0 + cc + d)) for d in range(2)]
    amax = fx.max(abs(ys[0]), abs(ys[1]))
    amax = fx.max(butterfly(amax, (1, 2, 4, 8), fx.max), fx.Float32(1e-4))
    code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
    step = pow2(code)
    q = [clamp_fp8(y / step) for y in ys]
    p = [xshfl(v, 1) for v in q]  # the partner's columns cc ^ 2, + 1
    if cc % 4 == 0:
        c["put_words"](c["x8"], t * X8_WORDS + (c0 + cc) // 4, [fp8_pack4(q[0], q[1], p[0], p[1])])
        if cc == 0:
            c["put"](c["x8s"], t * (HIDDEN // 32) + c0 // 32, code)


def pend_value(c, t, col, cc):
    """The owed post's sublayer output at (token t, column col): from ``pend``, or
    where a caller staged it (``pend_lds`` [token][32] of this task's columns: K2a's
    TP reduction)."""
    if const_expr(c.get("pend_lds") is None):
        return ld_bf(c["args"]["pend"], t * HIDDEN + col)
    return fx.ptr_load(c["pend_lds"] + (t * COLS + cc))


def slice_at(c0, e):
    return e // KT, (e % KT) // COLS, c0 + e % COLS


def slice_loads(c, c0, e):
    a = c["args"]
    t, o, col = slice_at(c0, e)
    if const_expr(not c["fold"]):
        return ld_bf(a["res_in"], (t * HC + o) * HIDDEN + col)
    r = [ld_bf(a["res_in"], (t * HC + h) * HIDDEN + col) for h in range(HC)]
    cm = [ld_f32(a["comb_in"], (t * HC + h) * HC + o) for h in range(HC)]
    return r, cm, ld_f32(a["post_in"], t * HC + o)


def slice_row(c, c0, e, got):
    a = c["args"]
    t, o, col = slice_at(c0, e)
    if const_expr(c["fold"]):
        r, cm, post = got
        # aiter's mhc_fused_post_pre_delayed_rmsnorm: ((post x + c0 r0) + c1 r1)
        # + c2 r2, each product rounded, then fma(c3, r3, ..)
        v = post * pend_value(c, t, col, e % COLS) + (r[0] * cm[0])
        for h in range_constexpr(1, HC - 1):
            v = v + cm[h] * r[h]
        v = bf16_round(fma(cm[HC - 1], r[HC - 1], v))
        bo.buffer_store(v.to(fx.BFloat16), rsrc(a["res_out"]), (t * HC + o) * HIDDEN + col)
    else:
        v = got
    fx.ptr_store(v, c["rl"] + (t * KT + o * COLS + e % COLS))


def slice_fn_loads(c, task):
    a = c["args"]
    c0 = task * COLS
    fns = []
    for i in range_constexpr(MIX * KT // THREADS):
        e = c["tid"] + THREADS * i
        j = e // KT
        o = (e % KT) // COLS
        fns.append(ld_f32(a["hc_fn"], j * K_MIX + o * HIDDEN + c0 + e % COLS))
    return fns


SLICE_FN_EARLY_MAX = 24  # rows up to which a prefetch also holds the mix weights


def slice_prefetch(c, task):
    """``stage_slice``'s first residual / gate loads (and, up to SLICE_FN_EARLY_MAX
    rows, its weights), issued before a wait they do not depend on (K2a's)."""
    c0 = task * COLS
    rows = first_rounds(c["tid"], c["S"] * KT, lambda e: slice_loads(c, c0, e))
    fns = slice_fn_loads(c, task) if c["S"] <= SLICE_FN_EARLY_MAX else None
    return fns, rows


@traced
def stage_slice(c, task, pre=None):
    """R_new, the layer input and the partial projections of hidden columns
    32 task .. +32, all four streams, every token. ``pre``: ``slice_prefetch``'s."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rl, fl = c["rl"], c["fl"]
    # a token tile (``tile_view``): PMIX laid out over the whole step
    tt0, s_all = c.get("tt0", 0), c.get("s_all", s)
    c0 = task * COLS
    if const_expr(pre is None):
        fns = slice_fn_loads(c, task)
        rows = first_rounds(tid, s * KT, lambda e: slice_loads(c, c0, e))
    else:
        fns, rows = pre
        if const_expr(fns is None):
            fns = slice_fn_loads(c, task)
    batched_rounds(tid, s * KT, lambda e: slice_loads(c, c0, e), lambda e, got: slice_row(c, c0, e, got), first=rows)
    for i in range_constexpr(MIX * KT // THREADS):
        # the original's bf16 MFMAs take fn as hi = bf16(fn), lo = bf16(fn - hi)
        hi = bf16_round(fns[i])
        fx.ptr_store(hi + bf16_round(fns[i] - hi), fl + (tid + THREADS * i))
    lds_barrier()
    for pas in range_constexpr(0, s * COLS // 2, THREADS):
        slice_input(c, task, plus(pas, tid))
    # sum of squares: token t on wave t % WAVES, four columns a lane
    for t in range_constexpr(s):
        if wave == t % WAVES:
            x = [fx.ptr_load(rl + (t * KT + 4 * lane + d)) for d in range(4)]
            sq = butterfly((x[0] * x[0] + x[1] * x[1]) + (x[2] * x[2] + x[3] * x[3]), (16, 8, 4, 2, 1))
            if lane == 0:
                c["put"](c["pmix"], (task * s_all + plus(tt0, t)) * PART + MIX, sq)
    for t0, n in token_tiles(s):
        slice_mix(c, task, t0, n)


@traced
def slice_mix(c, task, t0, n):
    """Mix projection partials of tokens t0 .. t0 + n: f32 WMMA 16 x 16 x 4, A = fn
    rows (two 16-row tiles), B = R_new^T. Operand lane l: row / token l % 16, K
    columns 2 (l // 16) and + 1 of the 4."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    rl, fl, red = c["rl"], c["fl"], c["red"]
    tt0, s_all = c.get("tt0", 0), c.get("s_all", c["S"])  # a token tile: PMIX over the step
    tok = plus(t0, fx.min(lane % 16, n - 1))
    cs = [fx.Vector.filled(ACC, 0.0, fx.Float32) for _ in range(2)]
    for kk in range_constexpr(KW // MIX_K):
        col0 = wave * KW + MIX_K * kk + 2 * (lane // 16)
        b = fx.Vector.from_elements([fx.ptr_load(rl + (tok * KT + col0 + d)) for d in range(2)], fx.Float32)
        for h in range_constexpr(2):
            row = h * 16 + lane % 16
            av = [fx.ptr_load(fl + (fx.min(row, MIX - 1) * KT + col0 + d)) for d in range(2)]
            av = fx.Vector.from_elements([(row < MIX).select(v, fx.Float32(0.0)) for v in av], fx.Float32)
            cs[h] = wmma_f32(av, b, cs[h])
    for h in range_constexpr(2):
        fx.ptr_store(cs[h], red + red_index(wave * 2 + h, lane, 0))
    lds_barrier()
    if tid < MIX * n:
        tl = tid // MIX
        t = plus(t0, tl)
        j = tid % MIX
        h = j // 16
        m = j % 16
        ln = 16 * (m // 8) + tl
        tot = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            tot = tot + fx.ptr_load(red + red_index(w * 2 + h, ln, m % 8))
        c["put"](c["pmix"], (task * s_all + plus(tt0, t)) * PART + j, tot)
    lds_barrier()


# ---------------------------------------------------------------- gate
GATE_BLOCK = 8
COMB0 = 16  # comb[r][c] at lane COMB0 + 4 r + c: rows lanes xor 1, 2; columns xor 4, 8


def _row_sum(v):
    return butterfly(v, (1, 2))


def _col_sum(v):
    return butterfly(v, (4, 8))


@traced
def stage_gate(c, t):
    """Token t's 160 slices, in slice order, -> post / comb / next pre."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    red = c["red"]
    a = c["args"]
    groups = SLICES // GATE_BLOCK  # 20
    if tid < groups * PART:
        g = tid // PART
        j = tid % PART
        vals = c["poll"]([(c["pmix"], ((g * GATE_BLOCK + k) * s + t) * PART + j, 1) for k in range(GATE_BLOCK)])
        acc = vals[0][0].bitcast(fx.Float32)
        for k in range_constexpr(1, GATE_BLOCK):
            acc = acc + vals[k][0].bitcast(fx.Float32)
        fx.ptr_store(acc, red + (g * PART + j))
    lds_barrier()
    if wave == 0:
        j = (lane < COMB0).select(lane % (2 * HC), fx.min(lane - COMB0 + 2 * HC, MIX - 1))
        mix = fx.ptr_load(red + j)
        sq = fx.ptr_load(red + MIX)
        for g in range_constexpr(1, groups):
            mix = mix + fx.ptr_load(red + (g * PART + j))
            sq = sq + fx.ptr_load(red + (g * PART + MIX))
        sc = [ld_f32(a["hc_scale"], i) for i in range(3)]
        base = ld_f32(a["hc_base"], j)
        v = mix * hw_rsq(fma(sq, fx.Float32(1.0 / K_MIX), fx.Float32(EPS)))
        if lane < HC:
            pre = sigmoid(fma(v, sc[0], base))
            bo.buffer_store(pre + fx.Float32(HC_EPS), rsrc(a["pre_out"]), t * HC + lane)
        if (lane >= HC) & (lane < 2 * HC):
            post = sigmoid(fma(v, sc[1], base))
            bo.buffer_store(post * fx.Float32(POST_MULT), rsrc(a["post_out"]), t * HC + lane - HC)
        cv = fma(v, sc[2], base)
        m = butterfly(cv, (1, 2), fx.max)
        cv = hw_exp2((cv - m) * fx.Float32(LOG2E))
        cv = fma(cv, hw_rcp(_row_sum(cv)), fx.Float32(HC_EPS))
        cv = cv * hw_rcp(_col_sum(cv) + fx.Float32(HC_EPS))
        for _ in range_constexpr(SINKHORN - 1):
            cv = cv * hw_rcp(_row_sum(cv) + fx.Float32(HC_EPS))
            cv = cv * hw_rcp(_col_sum(cv) + fx.Float32(HC_EPS))
        if (lane >= COMB0) & (lane < COMB0 + HC * HC):
            bo.buffer_store(cv, rsrc(a["comb_out"]), t * HC * HC + lane - COMB0)
    lds_barrier()


# ---------------------------------------------------------------- norm
XG_LDS0 = 64  # ``stage_norm``'s staged row in RED (words): past the sum of squares' (4)
XG_LDS_WORDS = 256 * 3 * 4  # a thread's 3 chunks of 8 bf16


@traced
def stage_norm(c, t, part=0, parts=1):
    """The seam's RMSNorm (256 threads, 3 chunks of 8 at 8 t + 2048 c), then
    quantize_fp8 of the bf16 row -> X8 words and codes. The sum of squares is the
    MI350 kernel's 64-lane tree (``butterfly64``) then its 4-way sum. Threads
    256.. mirror thread 255 and store nothing. ``parts`` > 1: the row's whole sum of
    squares, then only columns [part, part + 1) HIDDEN / parts quantized, stored and
    staged (a token-parallel K2's XG push spread over CTAs, ``publish_x_tok_par``)"""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    # plain Python flags, read before any traced ``if`` (whose state capture would
    # otherwise see the context dict)
    index, ffn, ncm = c["index"], c.get("ffn", False), c.get("normed_cm", 0)
    # token parallel (K2): the packed row also staged in LDS (RED past the sum of
    # squares' words) for the push to the other ranks' XG after the XRDY flag
    # (``layer_post.publish_x_tok_par``): pushed from here, the remote stores in
    # flight held the flag's drain (21 us at S = 48), read back from global it
    # cost a load round trip
    xg = c.get("xg")
    nb_dsts = c.get("nb_dsts")  # token parallel, an indexer layer: every other rank's NBG
    if const_expr(xg is not None):
        xg_lds = fx.recast_iter(fx.Int32, red) + XG_LDS0
    tt = fx.min(tid, 255)
    mine = tid < 256
    cols = [tt * 8 + 2048 * ch for ch in range(3)]
    wts = [[ld_bf(a["attn_w"], fx.min(cols[ch], HIDDEN - 8) + j) for j in range(8)] for ch in range(3)]
    bases = [t * HIDDEN + fx.min(cols[ch], HIDDEN - 8) for ch in range(3)]
    words = c["poll"]([(c["lin"], bases[ch] // 2 + k, 1) for ch in range(3) for k in range(4)])
    xs = []
    for ch in range_constexpr(3):
        v = []
        for k in range_constexpr(4):
            v += [bf_lo(words[4 * ch + k][0]), bf_hi(words[4 * ch + k][0])]
        xs.append([(cols[ch] < HIDDEN).select(x, fx.Float32(0.0)) for x in v])
    acc = fx.Float32(0.0)
    for ch in range_constexpr(3):
        for x in xs[ch]:
            acc = acc + x * x
    acc = butterfly64(acc, (1, 2, 4, 8, 16, 32), lane, wave, c["xbuf"])
    if (lane == 0) & (wave % 2 == 0) & mine:
        fx.ptr_store(acc, red + wave // 2)
    lds_barrier()
    tot = butterfly(fx.ptr_load(red + lane % 4), (1, 2))
    r = hw_rsq(fma(tot, fx.Float32(1.0 / HIDDEN), fx.Float32(EPS)))
    for ch in range_constexpr(3):
        col = fx.min(cols[ch], HIDDEN - 8)
        ys = [bf16_round((xs[ch][j] * r) * wts[ch][j]) for j in range(8)]
        amax = fx.Float32(0.0)
        for y in ys:
            amax = fx.max(amax, abs(y))
        amax = fx.max(butterfly(amax, (1, 2), fx.max), fx.Float32(1e-4))
        code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
        step = pow2(code)
        live = mine & (cols[ch] < HIDDEN)
        if const_expr(parts > 1):
            live = live & (cols[ch] >= part * (HIDDEN // parts)) & (cols[ch] < (part + 1) * (HIDDEN // parts))
        if live:
            if const_expr(index or ffn):
                # the compressor (an indexer layer) or the MoE (the FFN seam) reads
                # the bf16 row; device scope when this launch also reads it (the 8
                # values one 16-byte store)
                packed = fx.Vector.from_elements(ys, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
                bo.buffer_store(packed, rsrc(a["normed"]), (t * HIDDEN + col) // 2, cache_modifier=ncm)
                if const_expr(xg is not None):
                    fx.ptr_store(packed, xg_lds + (tt * 3 + ch) * 4)
            if const_expr(index):
                c["put_bf"](c["nb"], t * HIDDEN + col, ys[0:4])
                c["put_bf"](c["nb"], t * HIDDEN + col + 4, ys[4:8])
                if const_expr(nb_dsts is not None):
                    # token parallel: the row to every rank's NBG, this one's too (the compressor
                    # and every rank's iw read every token's from there)
                    tg = c["t0"] + t
                    for p in range_constexpr(len(nb_dsts)):
                        c["put_bf"](nb_dsts[p], tg * HIDDEN + col, ys[0:4])
                        c["put_bf"](nb_dsts[p], tg * HIDDEN + col + 4, ys[4:8])
            if const_expr(not ffn):
                w0 = fp8_pack4(*[clamp_fp8(ys[i] / step) for i in range(4)])
                w1 = fp8_pack4(*[clamp_fp8(ys[4 + i] / step) for i in range(4)])
                # the group's code in the tag words (``load_x8`` reads it there); X8S
                # still written for the reference checks
                c["put_words"](c["x8"], t * X8_WORDS + col // 4, [w0, w1])
                if tt % 4 == 0:
                    c["put"](c["x8s"], t * (HIDDEN // 32) + col // 32, code)
    lds_barrier()


# ---------------------------------------------------------------- mailbox -> LDS
@traced
def poll_copy(c, idx, width, spans, live=None, vec=1, batch=POLL_MAX):
    """Mailbox spans [(region, pair_of, n, dst, row)] into LDS: element u (pair
    ``pair_of(u)``) to dst + u, or ``lds_at(u, row)`` for rows ``row`` words
    wide; ``batch`` polls a round trip, each batch stored before the next is
    polled. ``live``: a span's elements the reader needs (traced, <= n), a span
    each; a batch past them is not polled, an element past them not written.
    ``vec`` 2: two adjacent pairs a poll (one 16-byte load; ``pair_of`` contiguous,
    n and ``row`` even): half the loads and the batches' round trips."""
    idx = fresh(idx)
    if const_expr(vec == 2):
        for k, (region, pair_of, n, dst, row) in enumerate(spans):
            assert n % 2 == 0 and (row is None or row % 2 == 0), (n, row)
            need = n if live is None else live[k]
            units = n // 2
            chunks = [(region, pair_of, need, (dst, row), idx + width * i) for i in range((units + width - 1) // width)]
            for b0 in range_constexpr(0, len(chunks), batch):
                if 2 * chunks[b0][4] < need:
                    _poll_store_pairs(c, chunks[b0 : b0 + batch])
            if const_expr(c.get("xtl") is not None):
                c["xtl"](k)  # a timeline build: span k in
        return
    if const_expr(live is None):
        _poll_store_batches(
            c,
            [
                (region, pair_of, n, (dst, row), idx + width * i)
                for region, pair_of, n, dst, row in spans
                for i in range((n + width - 1) // width)
            ],
            batch,
        )
        return
    for (region, pair_of, n, dst, row), need in zip(spans, live):
        chunks = [(region, pair_of, need, (dst, row), idx + width * i) for i in range((n + width - 1) // width)]
        for b0 in range_constexpr(0, len(chunks), batch):
            if chunks[b0][4] < need:
                _poll_store_batches(c, chunks[b0 : b0 + batch], batch)


@traced
def _poll_store_batches(c, chunks, batch=POLL_MAX):
    for b0 in range_constexpr(0, len(chunks), batch):
        part = chunks[b0 : b0 + batch]
        got = c["poll"]([(r, p(fx.min(u, n - 1)), 1) for r, p, n, _, u in part], batch)
        for (_, _, n, (dst, row), u), v in zip(part, got):
            at = u if row is None else lds_at(u, row)
            if u < n:
                fx.ptr_store(v[0], dst + at)


@traced
def _poll_store_pairs(c, chunks):
    """``poll_copy``'s two-pair polls: unit u is elements 2 u, 2 u + 1 (n the
    elements needed, even within a row)."""
    got = c["poll"]([(r, p(fx.min(2 * u, n - 2)), 2) for r, p, n, _, u in chunks], len(chunks))
    for (_, _, n, (dst, row), u), v in zip(chunks, got):
        for k in range_constexpr(2):
            e = 2 * u + k
            at = e if row is None else lds_at(e, row)
            if e < n:
                fx.ptr_store(v[k], dst + at)


@traced
def load_x8(c, words_mb, codes_mb, words, groups, xl, xsl, t0, n, tl=None):
    """Tokens t0 .. t0 + n's MXFP8 rows from the mailbox into LDS rows 0 .. (codes
    an i32 a group). ``n`` traced (a tile a CTA picks at run time): a TILE-row
    copy of which only n rows are polled. (A spin on one word a thread first -- the
    token's last scale code -- then the copy: K1 +2 us, 2026-10-30: the round trip it
    adds once the rows are in costs ~1 us, the batches' polls little while waiting.
    The codes in the words' tags, one span: K1 +3 us, 2026-11-04.)"""
    rows = n if isinstance(n, int) else TILE
    # (a wait on one two-pair poll a thread first, CTAs and threads spread over the
    # rows -- 1/15 of the spin's GL2 reads, no hot line -- then the copy: neutral,
    # 2026-10-30; the rows are seen ~1.5 us after the writers' stores drain either way)
    poll_copy(
        c if tl is None else {**c, "xtl": tl},
        c["tid"],
        THREADS,
        [
            (words_mb, lambda u: plus(t0 * words, u), rows * words, xl, words),
            (codes_mb, lambda u: plus(t0 * groups, u), rows * groups, xsl, groups),
        ],
        live=None if isinstance(n, int) else [n * words, n * groups],
        vec=2,
        batch=COPY_BATCH,
    )
    lds_barrier()


@traced
def stage_wqkv(c, task, ops, t0, n):
    """Rows 16 task .. of wqkv_a against tokens t0 .. t0 + n (x in LDS)."""
    tid, red = c["tid"], c["red"]
    gemv_fp8_wmmas(c["lane"], c["wave"], HIDDEN, c["xl"], c["xsl"], red, ops, n)
    lds_barrier()
    if tid < ROWS * n:
        r = tid % ROWS
        t = plus(t0, tid // ROWS)
        v = row_sum(red, r, tid // ROWS, range(WAVES))
        c["put"](c["qkv"], t * QKV_ROWS + task * ROWS + r, bf16_round(v))
    lds_barrier()


# ---------------------------------------------------------------- q / kv norms
@traced
def stage_qkv(c, t):
    """Token t: q_norm -> MXFP8 (the Triton dual norm's order: 64-lane tree
    (8, 4, 2, 1, 16, 32), then the 4 MI350 waves) and kv_norm -> GPT-J RoPE of
    the tail -> bf16 -> group-32 FP8 QAT -> the window ring row."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    # token parallel (a compile-time choice: a Python local first set in a traced
    # branch would become one of its results)
    tok_par = bool(c.get("tok_par", False))
    tt = fx.min(tid, 255)
    mine = tid < 256
    qc = [fx.min(tt * 8 + j, Q_RANK - 1) for j in range(8)]
    qnw = [ld_bf(a["qn_w"], qc[j]) for j in range(8)]
    kvnw = [ld_bf(a["kvn_w"], tt * 2 + j) for j in range(2)]
    pos = ld_i32(a["pos"], 2 * t)
    jr = tt - (KV_DIM - ROPE) // 2
    live = jr >= 0
    jj = fx.max(jr, 0)
    cs = live.select(ld_bf(a["cos"], pos * HALF + jj), fx.Float32(1.0))
    sn = live.select(ld_bf(a["sin"], pos * HALF + jj), fx.Float32(0.0))
    rel = ld_i32(a["ring_rows"], t)
    got = c["poll"](
        [(c["qkv"], t * QKV_ROWS + qc[j], 1) for j in range(8)]
        + [(c["qkv"], t * QKV_ROWS + Q_RANK + tt * 2 + j, 1) for j in range(2)]
    )
    qv, kvv = got[:8], got[8:]
    xq = [((tt * 8 + j) < Q_RANK).select(qv[j][0].bitcast(fx.Float32), fx.Float32(0.0)) for j in range(8)]
    xk = [kvv[j][0].bitcast(fx.Float32) for j in range(2)]
    ssq = xq[0] * xq[0]
    for j in range_constexpr(1, 8):
        ssq = ssq + xq[j] * xq[j]
    ssq = butterfly64(ssq, (8, 4, 2, 1, 16, 32), lane, wave, c["xbuf"])
    ssk = butterfly64(xk[0] * xk[0] + xk[1] * xk[1], (8, 4, 2, 1, 16, 32), lane, wave, c["xbuf"])
    if (lane == WAVE - 1) & (wave % 2 == 1) & mine:
        fx.ptr_store(ssq, red + wave // 2)
        fx.ptr_store(ssk, red + 4 + wave // 2)
    lds_barrier()
    tq = butterfly(fx.ptr_load(red + lane % 4), (2, 1))
    tk = butterfly(fx.ptr_load(red + 4 + lane % 4), (2, 1))
    nq = hw_rsq(div_rn(tq, fx.Float32(float(Q_RANK)), fx.Float32(1.0 / Q_RANK)) + fx.Float32(EPS))
    nk = hw_rsq(div_rn(tk, fx.Float32(float(KV_DIM)), fx.Float32(1.0 / KV_DIM)) + fx.Float32(EPS))
    ys = [(xq[j] * nq) * qnw[j] for j in range(8)]
    amax = fx.Float32(0.0)
    for y in ys:
        amax = fx.max(amax, abs(y))
    code = code_round(butterfly(amax, (1, 2), fx.max))
    mul = pow2(254 - code)
    if mine & (tt * 8 < Q_RANK):
        w0 = fp8_pack4(*[ys[i] * mul for i in range(4)])
        w1 = fp8_pack4(*[ys[4 + i] * mul for i in range(4)])
        if const_expr(tok_par):
            # token parallel: to every rank's QG / QGS, at the token's row of the step
            # (wq_b reads every token's from there; the raw q latent pushed from wqkv_a
            # instead, normed and quantized on every CTA: the copy 0.7 us sooner, the
            # 48 tokens' quantization a CTA +6.5 us -- K1 +6.6 us, 2026-10-30)
            tg = c["t0"] + t
            for p in range_constexpr(c["d"].tp):
                qw, qs = c["q_dsts"][p]
                c["put_words"](qw, tg * QX8_WORDS + 2 * tt, [w0, w1])
                if tt % 4 == 0:
                    c["put"](qs, tg * (Q_RANK // 32) + tt // 4, code)
        else:
            c["put_words"](c["qx8"], t * QX8_WORDS + 2 * tt, [w0, w1])
            if tt % 4 == 0:
                c["put"](c["qx8s"], t * (Q_RANK // 32) + tt // 4, code)
        if const_expr(c["index"]):
            # the (qr, qr_scale) pair the original indexer reads
            bo.buffer_store(fx.Vector.from_elements([w0, w1], fx.Int32), rsrc(a["qr_out"]), t * QX8_WORDS + 2 * tt)
            if tt % 4 == 0:
                bo.buffer_store(fx.Int8(code), rsrc(a["qrs_out"]), t * (Q_RANK // 32) + tt // 4)
    kn = [bf16_round((xk[j] * nk) * kvnw[j]) for j in range(2)]
    e0 = bf16_round(kn[0] * cs - kn[1] * sn)
    e1 = bf16_round(kn[0] * sn + kn[1] * cs)
    amax = butterfly(fx.max(abs(e0), abs(e1)), (1, 2, 4, 8), fx.max)
    step = pow2(code_ceil(fx.max(amax, fx.Float32(1e-4)) * fx.Float32(1.0 / FP8_MAX)))
    qat = [fp8_value(clamp_fp8(v / step)) * step for v in (e0, e1)]
    row = rel + a["ring_off"]
    if mine & (rel >= 0):
        dst = rsrc(a["ring"] + fx.Int64(row) * (KV_DIM * 2))
        for j in range_constexpr(2):
            bo.buffer_store(qat[j].to(fx.BFloat16), dst, tt * 2 + j)
    if const_expr(tok_par):  # noqa: SIM102
        # token parallel: the row to every other rank's KG (``ring_row_from`` writes it)
        if mine:
            tg = c["t0"] + t
            for p in range_constexpr(c["d"].tp):
                if c["rank"] != p:
                    c["put_bf"](c["kv_dsts"][p], tg * KV_DIM + 2 * tt, qat)
    lds_barrier()


@traced
def normed_row_from(c, nbg, t, n):
    """Token t, when another rank owns it (``n`` tokens a rank): its normed row from
    NBG -> ``normed`` (``stage_norm``'s store), two bf16 pairs a thread a round."""
    tid = c["tid"]
    a = c["args"]
    if t // n != c["rank"]:
        for k in range_constexpr(-(-(HIDDEN // 4) // THREADS)):
            q = tid + THREADS * k
            if q < HIDDEN // 4:
                (w,) = c["poll"]([(nbg, t * (HIDDEN // 2) + 2 * q, 2)])
                bo.buffer_store(fx.Vector.from_elements([w[0], w[1]], fx.Int32), rsrc(a["normed"]), t * (HIDDEN // 2) + 2 * q)


@traced
def ring_row_from(c, kg, t, n):
    """Token t, when another rank owns it (``n`` tokens a rank): its window row from
    KG -> the ring row (``stage_qkv``'s store), a bf16 pair a thread."""
    tid = c["tid"]
    a = c["args"]
    if t // n != c["rank"]:
        rel = ld_i32(a["ring_rows"], t)
        if tid < KV_DIM // 2:
            (w,) = c["poll"]([(kg, t * (KV_DIM // 2) + tid, 1)])
            if rel >= 0:
                dst = rsrc(a["ring"] + fx.Int64(rel + a["ring_off"]) * (KV_DIM * 2))
                bo.buffer_store(w[0], dst, tid)


# ---------------------------------------------------------------- wq_b
@traced
def rope_pair(c, t0, n, dim0):
    a = c["args"]
    tl = c["tid"] // (ROWS // 2)
    pos = ld_i32(a["pos"], 2 * plus(t0, fx.min(tl, n - 1)))
    jj = fx.max(dim0, 0) // 2
    return ld_bf(a["cos"], pos * HALF + jj), ld_bf(a["sin"], pos * HALF + jj)


@traced
def wqb_rope(c, task, t0, n):
    d = (task * ROWS + 2 * (c["tid"] % (ROWS // 2))) % HEAD_DIM
    return rope_pair(c, t0, n, d - (HEAD_DIM - ROPE))


@traced
def stage_wqb(c, task, ops, t0, n, rope):
    """Rows 16 task .. +16 of the rank's heads x 512: FP8 GEMV -> bf16 -> GPT-J
    RoPE of a head's last 64 dims -> q out."""
    tid, red = c["tid"], c["red"]
    a = c["args"]
    gemv_fp8_wmmas(c["lane"], c["wave"], Q_RANK, c["qxl"], c["qxsl"], red, ops, n)
    lds_barrier()
    if tid < ROWS * n // 2:
        rp = tid % (ROWS // 2)
        tl = tid // (ROWS // 2)
        t = plus(t0, tl)
        r0 = task * ROWS + 2 * rp
        v0 = bf16_round(row_sum(red, 2 * rp, tl, range(WAVES)))
        v1 = bf16_round(row_sum(red, 2 * rp + 1, tl, range(WAVES)))
        d = r0 % HEAD_DIM
        live = d >= HEAD_DIM - ROPE
        cs, sn = rope
        e0 = fx.Float32(live.select(v0 * cs - v1 * sn, v0))
        e1 = fx.Float32(live.select(v0 * sn + v1 * cs, v1))
        out = t * c["d"].heads * HEAD_DIM + r0
        bo.buffer_store(to_bf16(e0), rsrc(a["q_out"]), out)
        bo.buffer_store(to_bf16(e1), rsrc(a["q_out"]), out + 1)
    lds_barrier()


@traced
def stage_wqb_all(c, task, ops, rope, tiles):
    """``stage_wqb`` over every token tile at once: the tiles' WMMAs back to back
    (one weight operand, the tiles' rows in LDS from ``qxl``), their partials in
    RED a tile apart, one barrier, then every token's rows summed, RoPE'd and
    stored. A tile at a time, each tile's GEMV paid its own two barriers and
    reduction (S = 48: 6 GEMVs a CTA, wq_b ~3 us of compute). ``rope``:
    ``wqb_rope`` over the whole step (t0 0, n S)."""
    tid, red = c["tid"], c["red"]
    a = c["args"]
    s = c["S"]
    for i, (t0, n) in enumerate(tiles):
        gemv_fp8_wmmas(
            c["lane"], c["wave"], Q_RANK, c["qxl"] + t0 * lds_row(QX8_WORDS), c["qxsl"] + t0 * lds_row(Q_RANK // 32),
            red + i * RED_TILE, ops, n,
        )  # fmt: skip
    lds_barrier()
    if tid < ROWS * s // 2:
        rp = tid % (ROWS // 2)
        t = tid // (ROWS // 2)
        tile, tl = t // TILE, t % TILE
        r0 = task * ROWS + 2 * rp
        rt = red + tile * RED_TILE
        v0 = bf16_round(row_sum(rt, 2 * rp, tl, range(WAVES)))
        v1 = bf16_round(row_sum(rt, 2 * rp + 1, tl, range(WAVES)))
        d = r0 % HEAD_DIM
        live = d >= HEAD_DIM - ROPE
        cs, sn = rope
        e0 = fx.Float32(live.select(v0 * cs - v1 * sn, v0))
        e1 = fx.Float32(live.select(v0 * sn + v1 * cs, v1))
        out = t * c["d"].heads * HEAD_DIM + r0
        bo.buffer_store(to_bf16(e0), rsrc(a["q_out"]), out)
        bo.buffer_store(to_bf16(e1), rsrc(a["q_out"]), out + 1)
    lds_barrier()


@traced
def stage_iq_all(c, task, ops, rope, tiles):
    """``stage_iq`` over every token tile at once (``stage_wqb_all``'s pass: the
    tiles' WMMAs back to back, one barrier, every token's rows summed). ``rope``:
    ``iq_rope`` over the whole step (t0 0, n S)."""
    tid, red = c["tid"], c["red"]
    s = c["S"]
    for i, (t0, n) in enumerate(tiles):
        gemv_fp8_wmmas(
            c["lane"], c["wave"], Q_RANK, c["qxl"] + t0 * lds_row(QX8_WORDS), c["qxsl"] + t0 * lds_row(Q_RANK // 32),
            red + i * RED_TILE, ops, n,
        )  # fmt: skip
    lds_barrier()
    if tid < ROWS * s // 2:
        rp = tid % (ROWS // 2)
        t = tid // (ROWS // 2)
        tile, tl = t // TILE, t % TILE
        r0 = task * ROWS + 2 * rp
        rt = red + tile * RED_TILE
        v0 = bf16_round(row_sum(rt, 2 * rp, tl, range(WAVES)))
        v1 = bf16_round(row_sum(rt, 2 * rp + 1, tl, range(WAVES)))
        d = r0 % IDX_DIM
        live = d >= IDX_DIM - IDX_ROPE
        cs, sn = rope
        e0 = fx.Float32(live.select(v0 * cs - v1 * sn, v0))
        e1 = fx.Float32(live.select(v0 * sn + v1 * cs, v1))
        c["put_bf"](c["iq"], t * IDX_ROWS + r0, [e0, e1])
    lds_barrier()


@traced
def iq_rope(c, task, t0, n):
    d = (task * ROWS + 2 * (c["tid"] % (ROWS // 2))) % IDX_DIM
    return rope_pair(c, t0, n, d - (IDX_DIM - IDX_ROPE))


@traced
def stage_iq(c, task, ops, t0, n, rope):
    """Rows 16 task .. of the indexer's wq_b [4096, 1280]: FP8 GEMV of the q latent
    -> bf16 -> GPT-J RoPE of a head's dims 64..127 -> IQ."""
    tid, red = c["tid"], c["red"]
    gemv_fp8_wmmas(c["lane"], c["wave"], Q_RANK, c["qxl"], c["qxsl"], red, ops, n)
    lds_barrier()
    if tid < ROWS * n // 2:
        rp = tid % (ROWS // 2)
        tl = tid // (ROWS // 2)
        t = plus(t0, tl)
        r0 = task * ROWS + 2 * rp
        v0 = bf16_round(row_sum(red, 2 * rp, tl, range(WAVES)))
        v1 = bf16_round(row_sum(red, 2 * rp + 1, tl, range(WAVES)))
        d = r0 % IDX_DIM
        live = d >= IDX_DIM - IDX_ROPE
        cs, sn = rope
        e0 = fx.Float32(live.select(v0 * cs - v1 * sn, v0))
        e1 = fx.Float32(live.select(v0 * sn + v1 * cs, v1))
        c["put_bf"](c["iq"], t * IDX_ROWS + r0, [e0, e1])
    lds_barrier()


# ---------------------------------------------------------------- the kernel
def build_attn_pre(key: AttnPreBuild):
    s = key.tokens
    assert 1 <= s <= MAX_ROWS
    if key.aux:
        raise NotImplementedError("gfx1250 K1: the aux build is not ported yet")
    if key.tok_par:
        assert s % key.tp == 0, f"token parallel: {s} rows over TP {key.tp}"
    own_n = s // key.tp if key.tok_par else s
    own_tiles = token_tiles(own_n)
    layout = scratch_layout(s)
    tiles, rows = token_tiles(s), tile_rows(s)
    dims = Dims(key.tp)
    assert dims.wqb_tasks % BLOCKS == 0
    wqb_per_cta = dims.wqb_tasks // BLOCKS

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, max(2, len(tiles)) * RED_TILE, 16]  # wq_b: a tile's partials apart
        xbuf: fx.Array[fx.Float32, THREADS, 16]
        qxl: fx.Array[fx.Int32, s * lds_row(QX8_WORDS), 16]
        qxsl: fx.Array[fx.Int32, s * lds_row(Q_RANK // 32) + 4, 16]

    @fx.struct
    class SliceLds:
        rl: fx.Array[fx.Float32, s * KT, 16]
        fl: fx.Array[fx.Float32, MIX * KT, 16]

    @fx.struct
    class WqkvLds:
        xl: fx.Array[fx.Int32, rows * lds_row(X8_WORDS), 16]
        xsl: fx.Array[fx.Int32, rows * lds_row(HIDDEN // 32) + 4, 16]

    @fx.struct
    class IwLds:
        # a group of VWAVES tokens' normed rows (``stage_iw``)
        nbl: fx.Array[fx.Int32, min(s, VWAVES) * HIDDEN // 2 if key.feeds_index else 1, 16]

    @fx.union
    class StageLds:
        slice: SliceLds
        wqkv: WqkvLds
        iw: IwLds

    # X8 from the slice: a layer whose normed row nothing else reads (no indexer)
    slice_x8 = key.slice_x8 and not key.feeds_index
    # iw at the norm's end on the CTAs between qkv's and wqkv_a's (token parallel, one
    # token tile: wqkv_a WQKV0 .. +WQKV_TASKS wrapping past BLOCKS, then qkv's own_n)
    iw_early = key.iw_early and key.feeds_index and key.tok_par
    iw_lo, iw_hi = QKV0 + own_n, WQKV0
    if iw_early:
        assert len(own_tiles) == 1 and WQKV0 + WQKV_TASKS > BLOCKS and QKV0 + own_n <= WQKV0, (own_n,)
    name = kernel_name("dsv41_attn_pre", s=s, f=key.fold, ix=key.feeds_index, i4=key.index_fp4, tp=key.tp,
                       tl=int(key.timeline), tpar=int(key.tok_par), sx=int(slice_x8), ie=int(iw_early))

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def attn_pre(
        res_in: Int64, pend: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64,
        hc_fn: Int64, hc_scale: Int64, hc_base: Int64, attn_w: Int64, wqkv: Int64,
        wqkv_s: Int64, qn_w: Int64, kvn_w: Int64, wqb: Int64, wqb_s: Int64,
        cos: Int64, sin: Int64, pos: Int64, ring: Int64, ring_rows: Int64, ring_off: Int32,
        res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64,
        q_out: Int64, normed: Int64, qr_out: Int64, qrs_out: Int64, aux: Int64,
        iq_w: Int64, iq_ws: Int64, iw_w: Int64, iq_out: Int64, iqs_out: Int64, iw_out: Int64,
        scratch: Int64, rank: Int32, layer: Int32, tl: Int64, sym: Int64, peers: Int64,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        stage_lds = alloc.allocate(StageLds)
        sl, wl, il = stage_lds.slice.peek(), stage_lds.wqkv.peek(), stage_lds.iw.peek()
        mb = mailbox(layer, scratch, key.diag_off, layout)
        c = {
            "S": s, "tid": tid, "bid": bid, "lane": tid % WAVE, "wave": tid // WAVE,
            "rl": sl.rl.ptr, "fl": sl.fl.ptr, "red": lds.red.ptr, "xbuf": lds.xbuf.ptr,
            "xl": wl.xl.ptr, "xsl": wl.xsl.ptr, "qxl": lds.qxl.ptr, "qxsl": lds.qxsl.ptr,
            "put": mb.put, "put_bf": mb.put_bf, "put_words": mb.put_words, "poll": mb.poll,
            "args": {
                "res_in": res_in, "pend": pend, "post_in": post_in, "comb_in": comb_in,
                "pre_in": pre_in, "hc_fn": hc_fn, "hc_scale": hc_scale, "hc_base": hc_base,
                "attn_w": attn_w, "qn_w": qn_w, "kvn_w": kvn_w, "cos": cos, "sin": sin,
                "pos": pos, "ring": ring, "ring_rows": ring_rows, "ring_off": ring_off,
                "res_out": res_out, "post_out": post_out, "comb_out": comb_out,
                "pre_out": pre_out, "q_out": q_out, "normed": normed, "qr_out": qr_out,
                "qrs_out": qrs_out, "iw_w": iw_w, "iq_out": iq_out, "iqs_out": iqs_out,
                "iw_out": iw_out,
            },
            "fold": key.fold, "index": key.feeds_index, "d": dims, "rank": rank,
            "slice_x8": slice_x8,
        }  # fmt: skip
        for region, (off, _) in layout.items():
            c[region] = sreg(scratch, off, region)
        stamp = (lambda name: tl_stamp(tl, TL[name])) if key.timeline else (lambda name: None)
        x8_tl = lambda k: stamp(("x8.w", "x8.c")[k])  # noqa: E731  (the copies' spans in: a timeline build)
        stamp("start")
        # (a GL2 prefetch here of the CTA's wqkv / wq_b / iq weight groups: no gain,
        # 2026-10-30 -- the copies did not wait on the weights' loads)
        # token parallel (``tok_par``): slice .. qkv on this rank's tokens (``own_view``);
        # qkv pushes each token's q latent to every rank's QG / QGS (wq_b reads every
        # token's from there) and its window row to every other rank's KG, which the
        # launch's last stage writes into the ring
        co = c
        if const_expr(key.tok_par):  # a statement: an if-expression would be traced as a select
            bases = [load_ptr64(peers, p) for p in range(key.tp)]
            pl = peer_offsets(s, key.tp)
            co = own_view(c, K1_OWN)
            co["tok_par"] = True
            co["q_dsts"] = [(preg(bases[p], pl["qg"], "qg"), preg(bases[p], pl["qgs"], "qgs")) for p in range(key.tp)]
            co["kv_dsts"] = [preg(bases[p], pl["kg"], "kg") for p in range(key.tp)]
            if const_expr(key.feeds_index):
                # norm pushes its normed rows to every rank's NBG (every rank's compressor and
                # iw read every token's from there)
                co["nb_dsts"] = [preg(bases[p], pl["nbg"], "nbg") for p in range(key.tp)]
        # past one token tile (S a multiple of 16): (32 columns, 16 tokens) tasks,
        # 160 x S / 16, as K2a's seam (``attn_post.run_attn_post``); the mailboxes
        # stay laid out over the step
        seam_tiles = 1
        if const_expr(own_n > TILE and own_n % TILE == 0):
            seam_tiles = own_n // TILE
        if const_expr(seam_tiles > 1):
            for unit in range(first_task(bid, 0), SLICES * seam_tiles, BLOCKS):
                stage_slice(tile_view(co, K1_SEAM, unit // SLICES * TILE, TILE), unit % SLICES)
        else:
            for task in range(first_task(bid, 0), SLICES, BLOCKS):
                stage_slice(co, task)
        stamp("slice")
        # (``slice_x8``: X8 written by the slice, ``_slice_x8`` -- no norm stage)
        if const_expr(key.tok_par and key.feeds_index):
            # an indexer layer's token-parallel norm in (token, column part) units: a row's
            # whole sum of squares each, its part's columns quantized, stored and pushed to every
            # rank's NBG (a token a CTA pushed 4 x 20 KB of tagged pairs alone: norm +2 us; a
            # layer pushing nothing gains nothing from the parts: +1.1 us, 2026-10-30)
            for unit in range(first_task(bid, NORM0), own_n * NORM_PARTS, BLOCKS):
                stage_norm(co, unit // NORM_PARTS, unit % NORM_PARTS, NORM_PARTS)
        elif const_expr(not slice_x8):
            for t in range(first_task(bid, NORM0), own_n, BLOCKS):
                stage_norm(co, t)
        stamp("norm")

        def iw_context():
            """iw's context (``nb``: NBG under token parallel) and its group copy."""
            c_iw = c
            if const_expr(key.tok_par):
                c_iw = {**c, "nb": preg(sym, pl["nbg"], "nbg")}

            def load_group(g, n):
                # a group's VWAVES rows, of which the n < VWAVES of a last group polled
                span = (c_iw["nb"], lambda u: plus(g * HIDDEN // 2, u), VWAVES * HIDDEN // 2, il.nbl.ptr, None)
                poll_copy(c_iw, tid, THREADS, [span], live=[n * HIDDEN // 2], vec=2, batch=COPY_BATCH)
                lds_barrier()

            return c_iw, load_group

        if const_expr(iw_early):
            # iw at the norm's end on the CTAs wqkv_a and qkv leave idle (``iw_early``)
            c_iw, load_group = iw_context()
            if (bid >= iw_lo) & (bid < iw_hi):
                for task in range(bid - iw_lo, iw_tasks(s), iw_hi - iw_lo):
                    stage_iw(c_iw, task, HIDDEN, il.nbl.ptr, load_group)
        stamp("wqkv.x")
        if const_expr(len(own_tiles) == 1):
            wqkv_task = first_task(bid, WQKV0)
            if wqkv_task < WQKV_TASKS:
                ops = gemv_fp8_loads(co["lane"], co["wave"], wqkv, wqkv_s, HIDDEN, wqkv_task)
                load_x8(co, co["x8"], co["x8s"], X8_WORDS, HIDDEN // 32, co["xl"], co["xsl"], 0, own_n, x8_tl)
                stamp("wqkv.x")  # the rows in (a CTA without a task: at norm's end)
                stage_wqkv(co, wqkv_task, ops, 0, own_n)
        else:
            tile, wqkv_tasks = tile_tasks(bid, WQKV0, len(own_tiles), WQKV_TASKS)
            ops = [
                gemv_fp8_loads(co["lane"], co["wave"], wqkv, wqkv_s, HIDDEN, fx.min(t, WQKV_TASKS - 1))
                for t in wqkv_tasks
            ]
            t0 = tile * TILE
            n = fx.min(fx.Int32(TILE), own_n - t0)
            load_x8(co, co["x8"], co["x8s"], X8_WORDS, HIDDEN // 32, co["xl"], co["xsl"], t0, n, x8_tl)
            stamp("wqkv.x")  # the rows in (a CTA without a task: at norm's end)
            for k in range_constexpr(len(wqkv_tasks)):
                if wqkv_tasks[k] < WQKV_TASKS:
                    stage_wqkv(co, wqkv_tasks[k], ops[k], t0, n)
        stamp("wqkv_a")
        for t in range(first_task(bid, GATE0), own_n, BLOCKS):
            stage_gate(co, t)
        stamp("gate")
        for t in range(first_task(bid, QKV0), own_n, BLOCKS):
            stage_qkv(co, t)
        stamp("qkv")
        if const_expr(key.tok_par):
            # wq_b reads every token's q latent where qkv pushed it (a gather of QG into
            # the QX8 scratch, a token a CTA, so wq_b reads through the L2: K1 +0.6 us, the
            # extra hop costing more than the 35 MB of system-scope reads it saved; 2026-10-30)
            c["qx8"] = preg(sym, pl["qg"], "qg")
            c["qx8s"] = preg(sym, pl["qgs"], "qgs")
        wqb_task = first_task(bid, WQB0)
        wqb_ops = [
            gemv_fp8_loads(c["lane"], c["wave"], wqb, wqb_s, Q_RANK, nth_task(wqb_task, i))
            for i in range(wqb_per_cta)
        ]
        assert IQ_TASKS == BLOCKS
        iq_task = first_task(bid, IQ0)
        if const_expr(key.feeds_index):
            iq_ops = gemv_fp8_loads(c["lane"], c["wave"], iq_w, iq_ws, Q_RANK, iq_task)
        if const_expr(key.feeds_index and not iw_early):
            # iw on every token (each rank its own copy: no IWG wait before the quantization),
            # its normed rows from NBG under token parallel (pushed at the norm). Here, before
            # the q latent copy: it needs only the normed rows (in since the norm), and fills the
            # CTAs' wait for the other ranks' q (after the wq_b / iq tiles it ran 11.7 us of
            # the index layers' K1 critical path, 2026-10-30); the weights' loads in flight
            c_iw, load_group = iw_context()
            for task in range(first_task(bid, IW0), iw_tasks(s), BLOCKS):
                stage_iw(c_iw, task, HIDDEN, il.nbl.ptr, load_group)
        stamp("iw")
        # the RoPE (cos, sin) loads, then every token's q latent into LDS at once (one
        # copy, not a tile at a time between the tiles' GEMVs), then the GEMVs, every tile in
        # one pass a task (``stage_wqb_all`` / ``stage_iq_all``; an indexer layer's tile at a
        # time, 9 GEMV passes of their two barriers and reduction a CTA, ran wq_b 6.5 us)
        # (a source rank's block at a time, this rank's first, its GEMVs while the others'
        # q were in flight: K1 +0.8 us -- a barrier and reduction a block, 2026-11-04)
        ropes_w = [wqb_rope(c, nth_task(wqb_task, b), 0, s) for b in range(wqb_per_cta)]
        if const_expr(key.feeds_index):
            iq_rope_w = iq_rope(c, iq_task, 0, s)
        q_tl = lambda k: stamp(("q.w", "q.c")[k])  # noqa: E731
        load_x8(c, c["qx8"], c["qx8s"], QX8_WORDS, Q_RANK // 32, c["qxl"], c["qxsl"], 0, s, q_tl)
        stamp("wqb.x")
        for b in range_constexpr(wqb_per_cta):
            stage_wqb_all(c, nth_task(wqb_task, b), wqb_ops[b], ropes_w[b], tiles)
        if const_expr(key.feeds_index):
            stage_iq_all(c, iq_task, iq_ops, iq_rope_w, tiles)
        stamp("wq_b")
        if const_expr(key.feeds_index):
            # a CTA's quant tasks side by side, a wave each (they poll IW: one round trip,
            # not one a task in turn)
            per_cta = -(-(s * IDX_HEADS) // BLOCKS)
            assert per_cta <= WAVES
            for k in range_constexpr(per_cta):
                task = first_task(bid, IQUANT0) + k * BLOCKS
                if task < s * IDX_HEADS:
                    if const_expr(key.index_fp4):
                        stage_iquant_fp4(c, task, k)
                    else:
                        stage_iquant(c, task, k)

        stamp("index")
        if const_expr(key.tok_par):
            # the other ranks' tokens' window rows into this rank's ring
            kg = preg(sym, pl["kg"], "kg")
            for t in range(first_task(bid, QKV0), s, BLOCKS):
                ring_row_from(c, kg, t, own_n)
            if const_expr(key.feeds_index):
                # the compressor reads every token's normed row: the others' from NBG
                nbg = preg(sym, pl["nbg"], "nbg")
                for t in range(first_task(bid, NORM0), s, BLOCKS):
                    normed_row_from(c, nbg, t, own_n)
        stamp("ring")

    tag = cache_tag(name, key)

    @flyc.jit
    def launch(
        res_in: Int64, pend: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64,
        hc_fn: Int64, hc_scale: Int64, hc_base: Int64, attn_w: Int64, wqkv: Int64,
        wqkv_s: Int64, qn_w: Int64, kvn_w: Int64, wqb: Int64, wqb_s: Int64,
        cos: Int64, sin: Int64, pos: Int64, ring: Int64, ring_rows: Int64, ring_off: Int32,
        res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64,
        q_out: Int64, normed: Int64, qr_out: Int64, qrs_out: Int64, aux: Int64,
        iq_w: Int64, iq_ws: Int64, iw_w: Int64, iq_out: Int64, iqs_out: Int64, iw_out: Int64,
        scratch: Int64, rank: Int32, layer: Int32, tl: Int64, sym: Int64, peers: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = tag  # the build in the disk cache's key (``sources.cache_tag``)
        attn_pre(
            res_in, pend, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, attn_w,
            wqkv, wqkv_s, qn_w, kvn_w, wqb, wqb_s, cos, sin, pos, ring, ring_rows, ring_off,
            res_out, post_out, comb_out, pre_out, q_out, normed, qr_out, qrs_out, aux,
            iq_w, iq_ws, iw_w, iq_out, iqs_out, iw_out, scratch, rank, layer, tl, sym, peers,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
