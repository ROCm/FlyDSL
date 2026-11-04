# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""K2a's indexer on gfx1250 (a layer that selects: FULL, and REINDEX within
layer 20's candidates), ported from ATOM ``mono/kernels/index_score.py``
(MI355): the scores of every visible index row, the top-512 of them and, on the
candidate-producing layer, the top-2048 blocks -- the answer of the original
``score_topk_paged`` (logits: ``deepgemm_fp8_paged_mqa_logits``; selection:
``top_k_per_row_decode(stable=True)``; blocks: ``pick_candidate_blocks``). Which
cells each stage writes and polls is decided per step only in ``index_plan``.

    iscore  (a task: a run of a request's tokens x up to TASK_CHUNKS chunks of
            128 columns, ``index_plan.score_tasks``): every key load first, then
            each token's scores -> global ``ilog``; a candidate producer's 8-row
            block maxima (NaN ignored, capped at the largest finite, the newest
            block +inf) -> global ``ibmax``; then, once a task, the flags (ISF)
            of its (token, chunk)s.
    ilist   (a level's lists, per token): ``topk.lds_topk`` of a segment's values
            (level 0, after its chunks' flags) or of the lists it merges (level
            l > 0, from ISEL<l-1> / IBLK<l-1>), in index order. A list that is its
            token's last -> global (``isel``, lifted to real rows on a REINDEX
            layer; ``irow`` the pool rows the attention reads; ``icout``) and, for
            the selection, the ready flag (IRDY) the attention waits on; else ->
            ISEL<l> / IBLK<l>.

The score on wave32 WMMA: a chunk is 16 tiles of 8 rows, a wave a tile pair (16
keys) in every other chunk of the task (waves 0-7 the even, 8-15 the odd). One
scaled 16 x 16 x 128 FP8 WMMA (unit scales) a 16-head half: A = the token's 16
query heads, B = the pair's 16 keys, so a lane holds one key's 8 heads of each
half (``wmma_gfx1250``'s accumulator) -- relu(dot x k-scale) x weight summed in
the lane, then one swap across the halves: no 16-lane head tree. The plane is
MI355's (8-row tiles, 16 B chunk-major: byte ((c 8 + r) 16 + b) of dim 16 c + b,
then 8 fp32 scales); a lane's K bytes [32 j + 16 h, +16) are chunk 2 j + h, so
q and keys pair dims in natural order. WMMA K order and the head sum are not
MI355's: logits agree to rounding, not bit for bit.

Data too large for tagged pairs (the logits, up to max_model_len a token) goes
through global memory at device scope: stored, the stores drained, then the
flag (``publish``): a task's drain and barrier, hence several chunks a task.

Not ported: the FP4 plane (``index_fp4``), token parallel.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, T

from kernels.common import buffer_ops as bo
from kernels.monokernel.dsv41 import index_plan as ip
from kernels.monokernel.dsv41.attention import KMETA
from kernels.monokernel.dsv41.topk import key_bits, lds_topk
from kernels.monokernel.dsv41.topk import lds_words as topk_words
from kernels.monokernel.execution_gfx1250 import BLOCKS, THREADS, WAVE, WAVES, first_task
from kernels.monokernel.ops_gfx1250 import (
    ballot, butterfly, global_load, lane_gather, ld_dev, ld_i32, popcount, readlane, s_load_i32, s_load_ptr64,
    wave_incl_scan, xred,
)  # fmt: skip
from kernels.monokernel.sync_gfx1250 import SCOPE_DEV, lds_barrier, publish, rsrc, traced
from kernels.monokernel.wmma_gfx1250 import ACC, FP8, wmma_scaled

TILE = ip.BLOCK_ROWS
# a score task's chunks at most: 1 << this; a wave holds its tile pair of half
TASK_CHUNKS_LOG2 = 3
TASK_CHUNKS = 1 << TASK_CHUNKS_LOG2
PAIRS = ip.CHUNK // (2 * TILE)  # tile pairs a chunk: 8
PARITIES = WAVES // PAIRS  # waves a tile pair: 2, each every other chunk
SLOTS = TASK_CHUNKS // PARITIES  # chunks a wave scores a task: 4
TILE_BYTES = TILE * ip.DIM + TILE * 4  # 8 rows chunk-major, then 8 fp32 scales
PER_THREAD = ip.SEG // THREADS  # a select task's values a thread: 16
SMALL_PER = 4  # a short list's values a thread (``stage_ilist``): n <= 2048
SMALL_MAX = THREADS * SMALL_PER
NEG = float("-inf")
FLT_MAX = 3.4028234663852886e38
UNIT_SCALES = 0x7F7F7F7F  # four E8M0 1.0
assert PAIRS * PARITIES == WAVES and TASK_CHUNKS % PARITIES == 0
assert ip.HEADS == 32 and ip.DIM == 128

# the indexer's arguments, MI355's names (``attn_post.ABI``) in its order, as a
# table of i64 words at the kernel's ``iargs`` (pointers, or an int in the low
# word): one argument whatever the build, loaded where a stage reads them
INDEX_ARGS = (
    "iact", "iq", "iw", "plane", "iqs", "pscale", "iplan", "itab", "itab_stride",
    "itab_len", "ibatch", "ishift", "ibound", "ilog", "lstride", "isel", "icand",
    "ilift", "iprod", "ibmax", "bstride", "icout", "irow", "sbound",
    "sel", "table", "rows_per_page", "page_rows", "main_off",
)  # fmt: skip
INDEX_INTS = ("iact", "itab_stride", "itab_len", "lstride", "ilift", "iprod", "bstride", "rows_per_page",
              "page_rows", "main_off", "ishift")  # fmt: skip


def index_arg(table, name):
    """Argument ``name`` from the ``iargs`` table: scalar loads (SGPRs, no wait on
    the vector loads in flight, e.g. the attention's KV)."""
    i = INDEX_ARGS.index(name)
    if name in INDEX_INTS:
        return s_load_i32(table, 2 * i)
    return s_load_ptr64(table, i)


class IndexArgs(dict):
    """Kernel arguments (``c["args"]``) and the ``iargs`` table's, each of the
    latter loaded where it is read (a scalar load, a K$ hit): loaded once up
    front, the table's ~50 SGPRs stayed live through the indexer and K2 spilled
    SGPRs to scratch (S = 48, 2026-10-30)."""

    def __init__(self, base, table):
        super().__init__(base)
        self.table = table

    def __getitem__(self, name):
        if name in INDEX_ARGS and not dict.__contains__(self, name):
            return index_arg(self.table, name)
        return dict.__getitem__(self, name)

    def get(self, name, default=None):
        return self[name] if name in self else default

    def __contains__(self, name):
        return name in INDEX_ARGS or dict.__contains__(self, name)


def scratch_pairs(tokens: int, bound_max: int) -> dict:
    """The indexer's mailbox regions. The last level's list is always a token's
    final one (global), so only levels below it have a region."""
    out = {"isf": tokens * ip.chunks(bound_max), "irdy": tokens}
    for kind, k in ip.KINDS.items():
        for level in range(ip.top_level(kind, bound_max)):
            slots = ip.list_slots(kind, level, bound_max)
            out[f"i{kind}{level}"] = tokens * slots * k * 2
    return out


def sel_lds_words() -> dict:
    """The selection's LDS (32-bit words) by buffer: ``topk.lds_topk``'s at
    PER_THREAD values a thread."""
    return topk_words(PER_THREAD)


def _tl(c, name):
    """A timeline build's stamp of indexer point ``name`` (``c["itl"]``, set by
    a timeline build)."""
    if const_expr(c.get("itl") is not None):
        c["itl"](name)


def _st_dev(v, ptr, i):
    bo.buffer_store(v, rsrc(ptr), i, cache_modifier=SCOPE_DEV)


def _plane_addr(plane, tile, byte):
    return plane + fx.Int64(tile) * TILE_BYTES + fx.Int64(byte)


def _chunk_keys(a, t, ch0, width, wave, lane):
    """This lane's keys of the task's chunks through token t's tile table: wave w
    holds tile pair w % PAIRS of chunks ch0 + m, m = PARITIES k + w // PAIRS
    (lane: key lane % 16 of the pair, K half lane // 16). Every table load, then
    every key load, none waited on. A slot past ``width`` re-reads the last
    chunk (cache hits); a tile past the table reads its last."""
    pair, parity = wave % PAIRS, wave // PAIRS
    h, r8 = lane // 16, lane % 8
    ms = [PARITIES * k + parity for k in range(SLOTS)]
    blks = [(ch0 + fx.min(width - 1, m)) * (ip.CHUNK // TILE) + 2 * pair + lane % 16 // 8 for m in ms]
    tr = rsrc(a["itab"])
    tiles = [
        fx.Int32(bo.buffer_load(tr, t * a["itab_stride"] + fx.min(blk, a["itab_len"] - 1), vec_width=1, dtype=T.i32))
        for blk in blks
    ]
    keys = []
    for m, blk, tile in zip(ms, blks, tiles):
        ks = fx.Int32(global_load(_plane_addr(a["plane"], tile, TILE * ip.DIM + r8 * 4))).bitcast(fx.Float32)
        kw = [fx.Vector(global_load(_plane_addr(a["plane"], tile, ((2 * j + h) * TILE + r8) * 16), 4)) for j in range(4)]
        kv = fx.Vector.from_elements([kw[j][e] for j in range(4) for e in range(4)], fx.Int32)
        keys.append((m, blk, ks, kv))
    return keys


def _token_query(a, t, lane):
    """Token t's index query as this lane's A operands (head 16 half + lane % 16,
    K half lane // 16), and the weights of the heads its accumulator holds (8
    (lane // 16) + i of each half): K1's FP8 rows (``index_query.stage_iquant``,
    the query scale folded into the weight)."""
    h = lane // 16
    qr, wr = rsrc(a["iq"]), rsrc(a["iw"])
    query = []
    for half in range(2):
        row = t * ip.HEADS + 16 * half + lane % 16
        qw = [
            fx.Vector(bo.buffer_load(qr, (row * ip.DIM + 32 * j + 16 * h) // 4, vec_width=4, dtype=T.i32))
            for j in range(4)
        ]
        qv = fx.Vector.from_elements([qw[j][e] for j in range(4) for e in range(4)], fx.Int32)
        w0 = t * ip.HEADS + 16 * half + 8 * h
        ws = [fx.Vector(bo.buffer_load(wr, w0 + 4 * p, vec_width=4, dtype=T.f32)) for p in range(2)]
        query.append((qv, [ws[i // 4][i % 4] for i in range(ACC)]))
    return query


@traced
def _score_chunk(c, t, bound, blocks, query, blk, ks, kv, out=None):
    """Token t's scores (``bound`` its columns, ``blocks`` its candidate blocks)
    of the tile pair holding ``blk``, and its block's best on a candidate
    producer; see the module. ``out``: the scores also into LDS there (by column:
    ``stage_ifused``)"""
    lane = c["lane"]
    a = c["args"]
    r8 = lane % 8
    col = blk * TILE + r8
    zero = fx.Float32(0.0)
    unit = fx.Int32(UNIT_SCALES)
    halves = []
    for half in range_constexpr(2):
        qv, ws = query[half]
        acc = wmma_scaled(qv, kv, fx.Vector.filled(ACC, 0.0, fx.Float32), unit, unit, FP8, FP8)
        # heads 8 (lane // 16) + i of the half: relu(dot x k-scale) x weight, in order
        v = fx.max(acc[0] * ks, zero) * ws[0]
        for i in range_constexpr(1, ACC):
            v = fx.Float32(fmath.fma(fx.max(acc[i] * ks, zero), ws[i], v))
        halves.append(v)
    # the two halves, then the other 16 heads' lane (xor 16)
    total = xred(halves[0] + halves[1], 16, lambda x, y: x + y)
    live = col < bound
    score = fx.Float32(live.select(total, fx.Float32(NEG)))
    if lane < 16:
        _st_dev(score, a["ilog"], t * a["lstride"] + col)
        if const_expr(out is not None):
            fx.ptr_store(score.bitcast(fx.Int32), out + col)
    if a["iprod"] != 0:
        # a block's best visible score: its 8 rows are 8 lanes
        best = fx.Float32(fx.isnan(score).select(fx.Float32(NEG), score))
        best = fx.Float32(live.select(best, fx.Float32(NEG)))
        best = butterfly(best, (1, 2, 4), fx.max)
        best = fx.min(best, fx.Float32(FLT_MAX))
        newest = blk == blocks - 1
        best = fx.Float32(newest.select(fx.Float32(float("inf")), best))
        if (lane < 16) & (r8 == 0) & (blk < blocks):
            _st_dev(best, a["ibmax"], t * a["bstride"] + blk)


@traced
def stage_iscore(c, lead, span, ch0, width, max_chunks):
    """Chunks [ch0, ch0 + width) of the ``span`` tokens from ``lead``, which
    read lead's tile table (``index_plan.score_tasks``): the keys loaded once,
    every (token, chunk) scored, then one drain and the chunks' flags."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    a = c["args"]
    # a thread a (token, chunk) flag; its bound loaded with the keys
    row = lead + tid // TASK_CHUNKS
    m = tid % TASK_CHUNKS
    row_bound = ld_i32(a["ibound"], fx.min(row, c["S"] - 1))
    keys = _chunk_keys(a, lead, ch0, width, wave, lane)
    for r in range(span):
        tok = lead + r
        tok_bound = ld_i32(a["ibound"], tok)
        query = _token_query(a, tok, lane)
        tok_blocks = ip.blocks(tok_bound)
        for k in range_constexpr(SLOTS):
            km, blk, ks, kv = keys[k]
            if ip.task_writes(tok_bound, ch0, width, km):
                _score_chunk(c, tok, tok_bound, tok_blocks, query, blk, ks, kv)
    who = (tid < span * TASK_CHUNKS) & ip.task_writes(row_bound, ch0, width, m)
    publish(c["put"], c["isf"], row * max_chunks + ch0 + m, 1, who)


@traced
def stage_ilist(c, t, j, raw, kind, level, bound_max, max_chunks):
    """List j of ``level`` for token t (``raw`` its column bound); see the
    module. Only the lists ``index_plan.list_count`` gives a token run."""
    tid = c["tid"]
    a = c["args"]
    k = ip.KINDS[kind]
    slots = ip.list_slots(kind, level, bound_max)
    bound = ip.kind_bound(kind, raw)
    if j < ip.list_count(bound, level, k):
        n = ip.list_inputs(bound, level, j, k)
        final = ip.is_final(bound, level, k)
        # the top level has one list, a token's last: it has no region above it
        to_final = True if level == ip.top_level(kind, bound_max) else final
        got = []
        if const_expr(level == 0):
            if const_expr(kind == "sel"):
                first, count = ip.CHUNKS_PER_SEG * j, ip.seg_chunks(bound, j)
            else:
                first, count = ip.block_chunks(raw, j)
            if tid < count:
                c["poll"]([(c["isf"], t * max_chunks + first + tid, 1)])
            lds_barrier()
        else:
            below = ip.list_slots(kind, level - 1, bound_max)
            base = tid * PER_THREAD
            got = c["poll"](
                [
                    (
                        c[f"i{kind}{level - 1}"],
                        ((t * below + j * ip.fan(k) + fx.min(base + d, n - 1) // k) * k + fx.min(base + d, n - 1) % k)
                        * 2,
                        2,
                    )
                    for d in range(PER_THREAD)
                ]
            )
        src, stride = (a["ilog"], a["lstride"]) if kind == "sel" else (a["ibmax"], a["bstride"])

        def load(q, i):
            if const_expr(level == 0):
                # a quad stays in the row: strides are whole chunks / blocks
                v = fx.Vector(ld_dev(src, t * stride + j * ip.SEG + i, 4))
                return [fx.Int32(v[e]) for e in range(4)]
            return [got[4 * q + e][0] for e in range(4)]

        keys = c["sel_lds"]["keys"]

        def ident_of(d, i):
            return j * ip.SEG + i if level == 0 else got[d][1]

        def run_list(per):
            """The list's top-k at ``per`` values a thread."""
            first_i = tid * per

            # a selection's rows (``_selected_rows``), in flight during the passes
            def selected_rows():
                return _selected_rows(c, t, [ident_of(d, fx.min(first_i + d, n - 1)) for d in range(per)])

            def emit(pos, d, i, rows):
                if const_expr(d is None):
                    ident, bits = fx.Int32(-1), fx.Float32(NEG).bitcast(fx.Int32)
                    picked = (fx.Int32(-1), fx.Int32(0))
                else:
                    ident = ident_of(d, i)
                    bits = key_bits(fx.ptr_load(keys + i))
                    picked = rows[d] if kind == "sel" else None
                if to_final:
                    _final(c, kind, t, pos, ident, picked)
                else:
                    c["put_words"](c[f"i{kind}{level}"], ((t * slots + j) * k + pos) * 2, [bits, ident])

            lds_topk(
                n, k, load, emit, c["sel_lds"], tid, per,
                selected_rows if kind == "sel" else None, lambda p: _tl(c, f"i{kind}{level}.{p}"),
            )  # fmt: skip

        _tl(c, f"i{kind}{level}.in")
        if const_expr(level == 0):
            # a short segment (a step's usual: bounds of a few thousand) at SMALL_PER values a
            # thread: a quarter of the passes' LDS atomics, of the emits and of the row loads
            # (16 a thread left 476 of 512 threads counting only padding at n = 566)
            if n <= SMALL_MAX:
                run_list(SMALL_PER)
            else:
                run_list(PER_THREAD)
        else:
            run_list(PER_THREAD)
        _tl(c, f"i{kind}{level}.topk")
        # a selection's flag for the attention
        if const_expr(kind == "sel"):  # noqa: SIM102
            if final:
                publish(c["put"], c["irdy"], t, 1, tid == 0)
        _tl(c, f"i{kind}{level}.pub")


@traced
def _selected_rows(c, t, idents):
    """Columns ``idents`` of token t as a selection holds them (a REINDEX layer's
    compacted columns lifted back to the rows they stand for) and as the
    attention reads them: their pool rows, through the block table (every layer
    sharing this selection shares its KV owner, so its rows) -> [(row, pool row)].
    Issued from ``after_keys``, before the top-k passes, so the table reads
    overlap them. The candidate loads only on a REINDEX layer -- one branch for
    the layer, none an element (that would serialize the prefetches through the
    in-order load counter) -- and branch-free inside it, each in its token's
    candidate row (padding discards its value); the table offset a scalar load:
    a FULL layer's table reads wait on nothing (S = 48: they waited on the
    candidates' and the offset's round trips, extra + emit 3.6 us, 2026-10-30)."""
    a = c["args"]
    rows = list(idents)
    if a["ilift"] != 0:
        cands = [
            fx.Int32(bo.buffer_load(rsrc(a["icand"]), ip.candidate_offset(t, i), vec_width=1, dtype=T.i32))
            for i in idents
        ]
        rows = [fx.Int32((i >= 0).select(cd * TILE + fx.max(i, 0) % TILE, i)) for cd, i in zip(cands, idents)]
    toff = s_load_i32(a["kmeta"], t * KMETA + 2)
    rpp = a["rows_per_page"]
    out = []
    for row in rows:
        live = fx.max(row, 0)
        page = fx.Int32(bo.buffer_load(rsrc(a["table"]), toff + live // rpp, vec_width=1, dtype=T.i32))
        out.append((row, page * a["page_rows"] + a["main_off"] + live % rpp))
    return out


def _final(c, kind, t, pos, ident, rows=None):
    a = c["args"]
    if const_expr(kind == "sel"):
        row, pool_row = rows
        # the attention reads the pool rows; the selection only check mode compares
        _st_dev(row, a["isel"], t * ip.TOPK + pos)
        _st_dev(pool_row, a["irow"], t * ip.TOPK + pos)
    else:
        _st_dev(Int32(ident), a["icout"], t * ip.TOPK_BLOCKS + pos)


# ---------------------------------------------------------------- the step's plan, lane parallel
# ``index_plan``'s per-step plan (``score_tasks``, ``flat_task``, ``list_count``) over
# the step's S <= 64 tokens, a token a lane (token lane + 32 h in half h): scans,
# ballots and lane reads instead of S-long select chains. Unrolled over 48 tokens
# those were thousands of scalar instructions a stage, run cold, with SGPRs spilled
# to lanes -- ~4 us of a 22 us indexer, three times (S = 48, 2026-10-30). Same
# answers: the device check compares every output.
TOKEN_LANES = 2 * WAVE


def _halves(s):
    return -(-s // WAVE)


def _token(lane, h):
    return lane + WAVE * h


def _lanes_load(s, lane, load):
    """Token lane + 32 h's ``load(token)`` (a token past s reads token s - 1)."""
    return [load(fx.min(_token(lane, h), s - 1)) for h in range(_halves(s))]


def _lanes_valid(s, lane):
    return [_token(lane, h) < s for h in range(_halves(s))]


def _lanes_at(vals, idx):
    """vals of token ``idx`` (each lane its own: a lane gather)."""
    lo = lane_gather(vals[0], fx.max(fx.min(idx, WAVE - 1), 0))
    if const_expr(len(vals) == 1):
        return lo
    hi = lane_gather(vals[1], fx.max(fx.min(idx - WAVE, WAVE - 1), 0))
    return fx.Int32((idx < WAVE).select(lo, hi))


def _lanes_pick(vals, t):
    """vals of token t, t wave-uniform: a scalar."""
    lo = readlane(vals[0], fx.max(fx.min(t, WAVE - 1), 0))
    if const_expr(len(vals) == 1):
        return lo
    hi = readlane(vals[1], fx.max(fx.min(t - WAVE, WAVE - 1), 0))
    return fx.Int32((t < WAVE).select(lo, hi))


def _lanes_scan(vals, lane):
    """(inclusive prefix sums over the tokens, total): ``flat_total`` and the
    running sums ``flat_task`` walks."""
    incl = [wave_incl_scan(vals[0], lane)]
    if const_expr(len(vals) == 2):
        incl.append(wave_incl_scan(vals[1], lane) + readlane(incl[0], WAVE - 1))
    return incl, readlane(incl[-1], WAVE - 1)


def _lanes_task(incl, valid, task):
    """``index_plan.flat_task``: (token, its index) of flat task ``task`` < total --
    the token is the count of tokens whose running sum is <= task."""
    t = Int32(0)
    for h in range_constexpr(len(incl)):
        t = t + popcount(ballot((incl[h] <= task) & valid[h]))
    start = fx.Int32((t == 0).select(Int32(0), _lanes_pick(incl, t - 1)))
    return t, task - start


def _ctz(x):
    """Trailing zeros of a nonzero Int32."""
    return popcount((x & -x) - 1)


def _lanes_bit(masks, idx):
    """Bit ``idx`` (per lane, 0 <= idx < 64) of a 64-bit mask held as 32-bit halves."""
    word = masks[0] if len(masks) == 1 else fx.Int32((idx < WAVE).select(masks[0], masks[1]))
    return ((word >> (idx % WAVE)) & 1) != 0


def _score_plan(s, lane, bounds, owners, ctas, width_log2_max):
    """``index_plan.score_tasks`` (see there), a token a lane -> (counts, spans,
    width_log2): counts / spans per lane, width_log2 uniform."""
    hs = _halves(s)
    valid = _lanes_valid(s, lane)
    own = [fx.Int32(valid[h].select(ip.score_chunks(bounds[h]), Int32(0))) for h in range(hs)]
    # diff[t]: token t + 1 does not share t's owner (or there is none): a run ends at t
    diff = []
    for h in range_constexpr(hs):
        t = _token(lane, h)
        diff.append((owners[h] != _lanes_at(owners, t + 1)) | (t + 1 >= s))
    # a run's end: the first token at or after t whose next does not share its owner;
    # a half's bit of its last lane is set whenever no later token of the half is
    ends = [ballot(diff[h]) for h in range(hs)]
    last, run = [], []
    below = (Int32(1) << lane) - 1
    for h in range_constexpr(hs):
        t = _token(lane, h)
        mine = ends[h] & ~below  # this half's end bits at or after t
        later = WAVE * (h + 1) + _ctz(ends[h + 1]) if h + 1 < hs else Int32(WAVE * (h + 1) - 1)
        e = fx.Int32((mine != 0).select(WAVE * h + _ctz(mine), later))
        last.append(e)
        run.append(e - t + 1)
    # widest[t]: the most own over t .. its run's end (a segmented suffix max, doubling)
    w = list(own)
    off = 1
    while off < s:
        nw = []
        for h in range_constexpr(hs):
            t = _token(lane, h)
            cand = _lanes_at(w, t + off)
            nw.append(fx.Int32((t + off <= last[h]).select(fx.max(w[h], cand), w[h])))
        w = nw
        off *= 2
    # shared: a run's widest at its first token (one whose predecessor's run ended)
    shared = []
    for h in range_constexpr(hs):
        t = _token(lane, h)
        starts = (t == 0) | _lanes_bit(ends, fx.max(t - 1, 0))
        shared.append(fx.Int32(starts.select(w[h], Int32(0))))
    _, tot_shared = _lanes_scan(shared, lane)
    together = tot_shared >= ctas
    need = [fx.Int32(together.select(shared[h], own[h])) for h in range(hs)]
    spans = [fx.Int32(together.select(run[h], Int32(1))) for h in range(hs)]
    _, total = _lanes_scan(need, lane)
    width_log2 = Int32(0)
    for g in range_constexpr(1, width_log2_max + 1):
        width_log2 = fx.Int32(((total >> g) >= ctas).select(Int32(g), width_log2))
    counts = [(need[h] + (Int32(1) << width_log2) - 1) >> width_log2 for h in range(hs)]
    return counts, spans, width_log2


# ---------------------------------------------------------------- a short token's selection, one CTA
# A token of bound <= FUSED_MAX (a step's usual: S = 48 bounds of ~570 / ~1150) has
# its whole selection on one CTA (``stage_ifused``): its tile pairs scored straight into
# LDS, then the top-k off it -- no logits round trip through global memory, no chunk
# flags, no list task waiting on them. The same arithmetic and answer as the score
# stage and the level-0 list (``ilog`` / ``ibmax`` still written: the check reads them).
FUSED_MAX = THREADS * SMALL_PER  # columns: SMALL_PER values a thread in the top-k
FUSED_SLOTS = 4  # tile pairs a wave scores a round (their keys in flight together)
FUSED_ROUNDS = -(-(FUSED_MAX // (2 * TILE)) // (WAVES * FUSED_SLOTS))


@traced
def stage_ifused(c, t, raw):
    """Token t (``raw`` <= FUSED_MAX columns): scores (``_score_chunk``) into LDS
    ``keys`` by column, ``lds_topk`` of them, its rows (``_final``) and IRDY; on a
    candidate producer its blocks, every one of them (``blocks(raw)`` <= TOPK_BLOCKS:
    the block top-k takes them all, in order)."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    a = c["args"]
    keys = c["sel_lds"]["keys"]
    h, r8 = lane // 16, lane % 8
    npairs = ip.chunks(raw) * PAIRS
    blocks = ip.blocks(raw)
    lds_barrier()  # ``keys``: the last token's top-k done reading them
    if ip.scored(raw):
        query = _token_query(a, t, lane)
        tr = rsrc(a["itab"])
        for rnd in range_constexpr(FUSED_ROUNDS):
            ps = [wave + WAVES * (rnd * FUSED_SLOTS + k) for k in range(FUSED_SLOTS)]
            blks = [
                fx.min(p, npairs - 1) // PAIRS * (ip.CHUNK // TILE) + 2 * (fx.min(p, npairs - 1) % PAIRS) + lane % 16 // 8
                for p in ps
            ]
            tiles = [
                fx.Int32(bo.buffer_load(tr, t * a["itab_stride"] + fx.min(blk, a["itab_len"] - 1), vec_width=1, dtype=T.i32))
                for blk in blks
            ]
            kvs = []
            for tile in tiles:
                ks = fx.Int32(global_load(_plane_addr(a["plane"], tile, TILE * ip.DIM + r8 * 4))).bitcast(fx.Float32)
                kw = [fx.Vector(global_load(_plane_addr(a["plane"], tile, ((2 * j + h) * TILE + r8) * 16), 4)) for j in range(4)]
                kvs.append((ks, fx.Vector.from_elements([kw[j][e] for j in range(4) for e in range(4)], fx.Int32)))
            for k in range_constexpr(FUSED_SLOTS):
                if ps[k] < npairs:
                    _score_chunk(c, t, raw, blocks, query, blks[k], kvs[k][0], kvs[k][1], keys)
    lds_barrier()

    def load(q, i):
        v = fx.Vector(fx.ptr_load(keys + i, result_type=fx.Vector.make_type(4, fx.Int32)))
        return [fx.Int32(v[e]) for e in range(4)]

    def selected_rows():
        first_i = tid * SMALL_PER
        return _selected_rows(c, t, [fx.min(first_i + d, raw - 1) for d in range(SMALL_PER)])

    def emit(pos, d, i, rows):
        if const_expr(d is None):
            _final(c, "sel", t, pos, fx.Int32(-1), (fx.Int32(-1), fx.Int32(0)))
        else:
            _final(c, "sel", t, pos, i, rows[d])

    lds_topk(raw, ip.TOPK, load, emit, c["sel_lds"], tid, SMALL_PER, selected_rows)
    publish(c["put"], c["irdy"], t, 1, tid == 0)
    if a["iprod"] != 0:
        # the candidate blocks: every block of the token's, ascending, then -1
        for r in range_constexpr(ip.TOPK_BLOCKS // THREADS):
            pos = tid + THREADS * r
            _st_dev(fx.Int32((pos < blocks).select(pos, fx.Int32(-1))), a["icout"], t * ip.TOPK_BLOCKS + pos)


@traced
def run_fused(c, s, bid):
    """``stage_ifused`` for each token of bound <= FUSED_MAX, a token a CTA."""
    a = c["args"]
    for t in range(first_task(bid, 0), s, BLOCKS):
        raw = fx.max(ld_i32(a["ibound"], t), Int32(0))
        if raw <= FUSED_MAX:
            stage_ifused(c, t, raw)


@traced
def run_indexer(c, s, bid, bound_max, blocks=True):
    """Every indexer stage of a selecting layer, the tasks of each laid out from
    the step's bounds (``index_plan``, a token a lane): only live tasks. The
    bounds were written before this launch: plain loads. ``c["args"]``: an
    ``IndexArgs`` (with ``kmeta``). ``blocks`` False: the candidate blocks left
    to the caller (``run_index_blocks``: nothing in this launch reads them)."""
    assert s <= TOKEN_LANES, s
    a = c["args"]
    max_chunks = ip.chunks(bound_max)
    bounds = _lanes_load(s, c["lane"], lambda t: ld_i32(a["ibound"], t))
    # a short token's whole selection on one CTA (``run_fused``): the general stages
    # see its bound as 0 (no score task) and run no list of it
    fused = [b <= FUSED_MAX for b in bounds]
    general = [fx.Int32(f.select(Int32(0), b)) for f, b in zip(fused, bounds)]
    run_scores(c, s, bid, general, max_chunks)
    run_fused(c, s, bid)
    _tl(c, "iscore")
    _run_tree(c, s, bid, "sel", bounds, bound_max, max_chunks, fused)
    _tl(c, "isel")
    if const_expr(blocks):
        if a["iprod"] != 0:
            _run_tree(c, s, bid, "blk", bounds, bound_max, max_chunks, fused)
        _tl(c, "iblk")


@traced
def run_index_blocks(c, s, bid, bound_max):
    """A candidate-producing layer's top blocks (``run_indexer``'s last tree) on
    their own: the next REINDEX layers read them, this launch does not, so a
    merged launch runs them where their CTAs would wait anyway."""
    a = c["args"]
    if a["iprod"] != 0:
        bounds = _lanes_load(s, c["lane"], lambda t: ld_i32(a["ibound"], t))
        _run_tree(c, s, bid, "blk", bounds, bound_max, ip.chunks(bound_max), [b <= FUSED_MAX for b in bounds])


@traced
def run_scores(c, n, bid, bounds, max_chunks):
    """The score stage's tasks (``index_plan.score_tasks``) over rows 0 .. n. A
    request's tokens share its tile table (key metadata word 2: its block table
    offset); a REINDEX layer's are each token's own candidates."""
    assert n * TASK_CHUNKS <= THREADS, n  # a thread a (token, chunk) flag
    a = c["args"]
    lane = c["lane"]
    reindex = a["ilift"] != 0
    kown = _lanes_load(n, lane, lambda t: ld_i32(a["kmeta"], t * KMETA + 2))
    owners = [fx.Int32(reindex.select(_token(lane, h), kown[h])) for h in range(len(kown))]
    counts, spans, width_log2 = _score_plan(n, lane, bounds, owners, BLOCKS, TASK_CHUNKS_LOG2)
    incl, total = _lanes_scan(counts, lane)
    valid = _lanes_valid(n, lane)
    width = Int32(1) << width_log2
    for task in range(first_task(bid, 0), total, BLOCKS):
        lead, i = _lanes_task(incl, valid, task)
        stage_iscore(c, lead, _lanes_pick(spans, lead), i << width_log2, width, max_chunks)


@traced
def _run_tree(c, n, bid, kind, bounds, bound_max, max_chunks, fused):
    k = ip.KINDS[kind]
    lane = c["lane"]
    valid = _lanes_valid(n, lane)
    for level in range_constexpr(ip.top_level(kind, bound_max) + 1):
        counts = [ip.list_count(ip.kind_bound(kind, bounds[h]), level, k) for h in range(len(bounds))]
        counts = [fx.Int32(fused[h].select(Int32(0), valid[h].select(n_, Int32(0)))) for h, n_ in enumerate(counts)]
        incl, total = _lanes_scan(counts, lane)
        for task in range(first_task(bid, 0), total, BLOCKS):
            t, j = _lanes_task(incl, valid, task)
            stage_ilist(c, t, j, _lanes_pick(bounds, t), kind, level, bound_max, max_chunks)


def check_build(bound_max: int, index_fp4: bool, tokens: int, tok_par: bool) -> None:
    """The builds this port covers (else a clear error at build time). A token-parallel
    K2 runs the indexer on every row on every rank (its inputs -- K1's iq / iw, the
    plane -- are every row's on each, its selection read on that rank alone): no
    exchange, MI355's own-rows indexer with its peer outputs not ported."""
    if index_fp4:
        raise NotImplementedError("gfx1250 indexer: the FP4 plane (index_score_fp4) is not ported yet")
    assert tokens * TASK_CHUNKS <= THREADS, f"indexer: {tokens} rows x {TASK_CHUNKS} chunk flags > {THREADS} threads"
    assert bound_max > ip.TOPK, bound_max
    assert ip.SEG % THREADS == 0 and PER_THREAD % 4 == 0
