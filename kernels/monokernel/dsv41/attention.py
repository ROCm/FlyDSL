# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""K2a's decode sparse attention on gfx1250: the original decode's split +
reduce (64 splits of one 16-key tile, D = 512, bf16 pool, a 16-head tile), its
softmax statistics and combine as the MI350 mono kernel keeps them, the matrix
products on bf16 WMMA 16 x 16 x 32.

score (S x tiles x ``per`` tasks; wave w of task (t, tile, sg): splits
       (sg + per k) WAVES + w): C[key][head] = 16 chained WMMAs over D (A = the
       split's 16 KV rows, B = the tile's 16 query heads), x qk_scale, NEG past
       kv_len. A lane holds one head's 8 keys (``wmma_gfx1250``'s accumulator
       layout), so m (max of 8, then xor 16), p = exp2(s - m) and l are in-lane
       plus one swap -> AM / AL; the lane's 8 bf16 p are exactly its slice of the
       PV WMMA's B operand -> AP, read back by the same lane.
pv    (S x tiles x GROUPS / groups tasks, 32 columns a group): wave w holds the
       splits w + WAVES r; per split a WMMA with A = V^T (16 columns x the
       split's 16 keys, the other K half zero) and B = its p; per head m_max,
       alpha_s = exp2(m_s - m_max), l = sum l_s alpha_s; acc = sum acc_s alpha_s
       (in-wave, then the waves in order); the sink fold; bf16 -> LDS ``out``
       [group][head][32].

The WMMA's K order is the hardware's: scores and partial sums agree with the
MI350 kernel (and the original Triton split) to rounding, not bit for bit.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from kernels.common import buffer_ops as bo
from kernels.monokernel.dsv41.dims import HEAD_DIM, HEAD_TILE
from kernels.monokernel.execution_gfx1250 import BLOCKS, WAVE, WAVES
from kernels.monokernel.ops_gfx1250 import bf16_round, fresh, global_load, hw_exp2, ld_dev, ld_i32, uniform, xshfl
from kernels.monokernel.sync_gfx1250 import f32_pinned, lds_barrier, rsrc, traced, wave_lds_fence
from kernels.monokernel.wmma_gfx1250 import ACC, wmma_bf16

SPLITS = 64  # a (token, head tile)'s splits: the original decode's
BK = 16  # keys a split
KMETA = 4  # a token's window key metadata words: first, count, table offset, slot base
COLS = 32  # the pv task's columns: one FP8 quant group of the output
GROUPS = HEAD_DIM // COLS
NEG = -3.4028234663852886e38
LOG2E = 1.4426950408889634
ROW_BYTES = HEAD_DIM * 2
D_STEPS = HEAD_DIM // 32  # WMMAs over D
TOPK = 512  # a selecting layer's top rows


def score_tasks(s, tiles) -> int:
    """Score tasks a (token, head tile): the most, up to a split a wave, that keep
    a step's tasks to one round of the CTAs."""
    per = SPLITS // WAVES
    while s * tiles * per > BLOCKS and per > 1:
        per //= 2
    return per


def pv_groups(s, tiles) -> int:
    groups = 1
    while s * tiles * GROUPS // groups > BLOCKS and groups < GROUPS:
        groups *= 2
    return groups


def pv_per(s, tiles) -> int:
    """(token, head tile, column group) pv items a CTA, contiguous, the items spread
    over every CTA (``attn_post``'s irq): S = 48's 768 are 3 a CTA on 256, where a
    power-of-two ``pv_groups`` ran 4 a CTA on 192."""
    per = -(-(s * tiles * GROUPS) // BLOCKS)
    assert per <= GROUPS
    return per


def pv_split(s, tiles):
    """(CTAs a (token, head tile), column groups a CTA at most): a (token, tile)'s
    GROUPS groups split over ``cpt`` CTAs, CTA j taking [j GROUPS / cpt, (j + 1)
    GROUPS / cpt) -- no CTA's items cross a token. S = 48: 5 a token (3, 3, 3, 3, 4)
    on 240 CTAs; ``pv_per``'s 3 contiguous items a CTA put 96 CTAs across two tokens,
    two ``pv_run``s back to back (their split statistics polled, combined and their
    V loaded twice: irq 16.5 us on those, 5.6 on the rest)."""
    cpt = max(1, min(GROUPS, BLOCKS // (s * tiles)))
    return cpt, -(-GROUPS // cpt)


def scratch_pairs(tokens: int, d) -> dict:
    return {
        "am": tokens * SPLITS * d.heads,
        "al": tokens * SPLITS * d.heads,
        "ap": tokens * d.head_tiles.count * SPLITS * WAVE * 4,
    }


def imin(a, b):
    return fx.min(a, b) if not (isinstance(a, int) and isinstance(b, int)) else min(a, b)


def written_splits(n):
    """Splits of ``n`` keys the score stage writes (split 0 for a key-less pad row)."""
    return fx.max((n + BK - 1) // BK, 1)


def tile_task(task, per, tiles):
    if tiles.count == 1:
        return task // per, 0, task % per
    return task // (per * tiles.count), task // per % tiles.count, task % per


def tile_head(d, tile, h):
    """Head h of ``tile`` -> (the rank's head it reads, live)."""
    tiles = d.head_tiles
    if tiles.count == 1 and not tiles.ragged:
        return h, True
    head = tile * HEAD_TILE + h
    if not tiles.ragged:
        return head, True
    return fx.min(head, d.heads - 1), head < d.heads


def only_live(cond, live):
    return cond if live is True else cond & live


def _keys(a, t):
    """Token t's key count and ``_key_row``'s inputs: its selected count (the
    selection's valid ids, at most TOPK; 0 without an indexer), first window
    position, slot row base and key count."""
    first, count, base = [ld_i32(a["kmeta"], t * KMETA + i) for i in (0, 1, 3)]
    if const_expr(a.get("sbound") is None):
        n_sel = fx.Int32(0)
    else:
        bound = fx.Int32(ld_dev(a["sbound"], t))
        n_sel = fx.max(fx.min(bound, TOPK), 0)
    return n_sel + count, (n_sel, first, base, n_sel + count)


def _key_row(a, t, meta, key):
    """Pool row of token t's key ``key`` (0 <= key < SPLITS BK): the selected rows
    (``sel``), then the window rows oldest first. A key past the token's count
    reads a valid row (zeroed at use). The selection's word is read at the key
    itself, not at the key clamped to the count, so it need not wait on the count
    (``sbound``) -- the compiler still issues it after the count returns. Neither
    loading it ahead of the count in the program (S = 48 K2 +3 us with the indexer)
    nor branching around it for a split with no selected key (+2.4 us) helped
    (2026-10-30)."""
    n_sel, first, base, kv_len = meta
    win = fx.max(fx.min(key, kv_len - 1), n_sel)  # a window key, or the first one
    window = base + a["ring_off"] + (first + win - n_sel) % a["ring_slots"]
    if const_expr(a.get("sel") is None):
        return window
    picked = key < n_sel
    selected = fx.Int32(picked.select(ld_dev(a["sel"], t * TOPK + fx.min(key, TOPK - 1)), 0))
    return fx.Int32(picked.select(selected, window))


def _row_addr(pool, row, byte):
    """Pool row ``row`` at ``byte`` (64-bit: the pool passes 4 GiB)."""
    return pool + fx.Int64(row) * ROW_BYTES + fx.Int64(byte)


def _ms(d, t, s, h):
    return (t * SPLITS + s) * d.heads + h


def _ap(d, t, tile, s, lane):
    """Split s of token t's head tile in AP: a lane's 4 words (its 8 bf16 p)."""
    return (((t * d.head_tiles.count + tile) * SPLITS + s) * WAVE + lane) * 4


@traced
def await_selection(c, t):
    """A layer that selects: its token t's selection is written this launch
    (``index_score``). A build without the indexer has none to wait for, nor
    has a layer that reads an earlier layer's (``iact`` 0)."""
    if const_expr("irdy" in c):  # noqa: SIM102
        if c["args"]["iact"] != 0:
            c["poll"]([(c["irdy"], t, 1)])


def _operand_bf16(addr_of, ch, h):
    """A bf16 WMMA operand row's step ``ch`` (K = 32): elements 32 ch + 8 h .. +8
    and 32 ch + 16 + 8 h .. +8, two 16-byte loads from ``addr_of(byte)``."""
    lo = addr_of((32 * ch + 8 * h) * 2)
    hi = addr_of((32 * ch + 16 + 8 * h) * 2)
    return lo, hi


def _vec8(lo, hi):
    lo, hi = fx.Vector(lo), fx.Vector(hi)
    return fx.Vector.from_elements([lo[i] for i in range(4)] + [hi[i] for i in range(4)], fx.Int32)


QROW = HEAD_DIM // 2 + 4  # a staged q row's words: 512 bf16 + 16 B (a b128 read of 16 heads hits 16 bank groups)
Q_WORDS = HEAD_TILE * QROW
assert Q_WORDS <= WAVES * WAVE * 2 * ACC  # inside ``tree`` (``lds_words``)


def _q_fetch(c, t, tile):
    """Thread tid's 32 B of the tile's q rows (head tid / 32, bytes 32 (tid % 32) ..):
    the CTA's 512 threads hold all 16 heads' 1 KB rows once (every wave's B)."""
    tid, d, a = c["tid"], c["d"], c["args"]
    qh = tile_head(d, tile, tid // 32)[0]
    at = ((t * d.heads + qh) * HEAD_DIM * 2 + tid % 32 * 32) // 4
    r = rsrc(a["q"])
    return [fx.Vector(bo.buffer_load(r, at + 4 * j, vec_width=4, dtype=T.i32)) for j in range(2)]


def _q_store(c, ql, words):
    tid = c["tid"]
    for j in range_constexpr(2):
        fx.ptr_store(words[j], ql + (tid // 32 * QROW + tid % 32 * 8 + 4 * j))


def _q_operand(ql, ch, lane):
    """Step ch's B operand (``_operand_bf16``'s elements) of head lane % 16 from LDS."""
    base = lane % 16 * QROW + 16 * ch + 4 * (lane // 16)
    v4 = fx.Vector.make_type(4, fx.Int32)
    return _vec8(fx.ptr_load(ql + base, result_type=v4), fx.ptr_load(ql + (base + 8), result_type=v4))


def _kv_split(a, t, meta, kv_len, s, lane):
    """Split s's KV operand of this lane (key 16 s + lane % 16), every D step's
    loads in flight at once (one HBM trip a split, not one a step); a key past
    kv_len reads a valid row (``_key_row``) and is zeroed at use."""
    row = _key_row(a, t, meta, s * BK + lane % 16)
    return [
        _vec8(*_operand_bf16(lambda b: global_load(_row_addr(a["pool"], row, b), 4), ch, lane // 16))
        for ch in range(D_STEPS)
    ]


@traced
def stage_attn_score(c, task, per):
    """Score task ``task`` (``per`` a (token, head tile)): this wave's splits.
    The CTA stages the tile's q in LDS once (the attention's ``tree``: pv's, written
    only past its barriers) while each wave's first split of KV is in flight."""
    lane, wave, d = c["lane"], c["wave"], c["d"]
    a = c["args"]
    head = lane % 16
    t, tile, sg = tile_task(task, per, d.head_tiles)
    qh, live = tile_head(d, tile, head)
    qw = _q_fetch(c, t, tile)  # independent of the selection: in flight through its wait
    await_selection(c, t)
    kv_len, meta = _keys(a, t)
    nsplit = SPLITS // WAVES // per
    ql = fx.recast_iter(fx.Int32, c["att"]["tree"])
    first = _kv_split(a, t, meta, kv_len, sg * WAVES + wave, lane)
    lds_barrier()  # a previous task's waves done reading ql (the KV loads stay in flight)
    _q_store(c, ql, qw)
    lds_barrier()
    for k in range_constexpr(nsplit):
        s = (sg + per * k) * WAVES + wave
        ok = s * BK + head < kv_len
        kvs = first if k == 0 else _kv_split(a, t, meta, kv_len, s, lane)
        sc = fx.Vector.filled(ACC, 0.0, fx.Float32)
        for ch in range_constexpr(D_STEPS):
            kv = fx.Vector.from_elements([ok.select(kvs[ch][e], fx.Int32(0)) for e in range(8)], fx.Int32)
            sc = wmma_bf16(kv.bitcast(fx.BFloat16), _q_operand(ql, ch, lane).bitcast(fx.BFloat16), sc)
        if s < written_splits(kv_len):
            _score_out(c, t, tile, s, qh, live, kv_len, sc)


@traced
def _score_out(c, t, tile, s, qh, live, kv_len, sc):
    """A split's scores (lane: head lane % 16, keys 8 (lane // 16) + i) -> m, l, p."""
    lane, d = c["lane"], c["d"]
    a = c["args"]
    h = lane // 16
    scale = fx.Int32(a["qk_scale"]).bitcast(fx.Float32)
    sv = [fx.Float32((s * BK + 8 * h + i < kv_len).select(sc[i] * scale, fx.Float32(NEG))) for i in range(ACC)]
    m = sv[0]
    for i in range_constexpr(1, ACC):
        m = fx.max(m, sv[i])
    m = fx.max(m, xshfl(m, 16, lane))
    p = [hw_exp2(sv[i] - m) for i in range(ACC)]
    ssum = ((p[0] + p[4]) + (p[2] + p[6])) + ((p[1] + p[5]) + (p[3] + p[7]))
    ssum = ssum + xshfl(ssum, 16, lane)
    if only_live(h == 0, live):
        c["put"](c["am"], _ms(d, t, s, qh), m)
        c["put"](c["al"], _ms(d, t, s, qh), ssum)
    pw = fx.Vector.from_elements([f32_pinned(v) for v in p], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
    base = _ap(d, t, tile, s, lane)
    c["put_words"](c["ap"], base, [pw[0], pw[1]])
    c["put_words"](c["ap"], base + 2, [pw[2], pw[3]])


@traced
def stage_attn_pv(c, task, groups):
    """Token t, ``groups`` column groups of 32 of every head of a tile (pv task
    ``task``) -> ``att["out"]`` [group][head][column] (``pv_run``)."""
    t, tile, g0 = tile_task(task, GROUPS // groups, c["d"].head_tiles)
    pv_run(c, t, tile, g0 * groups, groups, groups)


@traced
def pv_run(c, t, tile, dc0, n, nmax):
    """Token t, column groups dc0 .. dc0 + n of every head of a tile (n <= nmax, n
    traced or nmax itself) -> ``att["out"]`` [group][head][column] (f32 of bf16),
    barriers included: the split statistics polled once, then the groups."""
    lane, wave, tid, d = c["lane"], c["wave"], c["tid"], c["d"]
    att = c["att"]
    a = c["args"]
    h = lane // 16
    head = lane % 16
    await_selection(c, t)
    kv_len, meta = _keys(a, t)
    act = (kv_len + BK - 1) // BK
    last = written_splits(kv_len) - 1
    splits = [wave + WAVES * r for r in range(SPLITS // WAVES)]
    # the V rows need no score: the first group's in flight through the polls below
    rows = [_key_row(a, t, meta, sp * BK + lane % 16) for sp in splits]
    pre = _pv_loads(a, rows, dc0, lane)
    # per split: m, l of this lane's head (lane = head), and its p words
    specs = []
    for sp in splits:
        src = fx.min(sp, last)
        qh = tile_head(d, tile, head)[0]
        specs += [(c["am"], _ms(d, t, src, qh), 1), (c["al"], _ms(d, t, src, qh), 1)]
        base = _ap(d, t, tile, src, lane)
        specs += [(c["ap"], base, 2), (c["ap"], base + 2, 2)]
    got = c["poll"](specs)
    ms, ls, pws = [], [], []
    for r, sp in enumerate(splits):
        alive = sp < act
        ms.append(fx.Float32(alive.select(got[4 * r][0].bitcast(fx.Float32), fx.Float32(NEG))))
        ls.append(fx.Float32(alive.select(got[4 * r + 1][0].bitcast(fx.Float32), fx.Float32(0.0))))
        pws.append(got[4 * r + 2] + got[4 * r + 3])
    # m_max over every split of this head: in-wave, then across the waves (LDS)
    mloc = ms[0]
    for r in range_constexpr(1, len(splits)):
        mloc = fx.max(mloc, ms[r])
    if h == 0:
        fx.ptr_store(mloc, att["stat"] + (wave * HEAD_TILE + head))
    lds_barrier()
    m_max = fx.ptr_load(att["stat"] + head)
    for w in range_constexpr(1, WAVES):
        m_max = fx.max(m_max, fx.ptr_load(att["stat"] + (w * HEAD_TILE + head)))
    alphas = [hw_exp2(ms[r] - m_max) for r in range(len(splits))]
    lw = fx.Float32(fmath.fma(ls[0], alphas[0], fx.Float32(0.0)))
    for r in range_constexpr(1, len(splits)):
        lw = fx.Float32(fmath.fma(ls[r], alphas[r], lw))
    lds_barrier()
    if h == 0:
        fx.ptr_store(lw, att["stat"] + (wave * HEAD_TILE + head))
    lds_barrier()
    l_comb = fx.ptr_load(att["stat"] + head)
    for w in range_constexpr(1, WAVES):
        l_comb = l_comb + fx.ptr_load(att["stat"] + (w * HEAD_TILE + head))
    sink = fx.Float32(bo.buffer_load(rsrc(a["sink"]), tile_head(d, tile, head)[0], vec_width=1, dtype=T.f32))
    if (wave == 0) & (h == 0):
        fx.ptr_store(m_max, att["mmax"] + head)
        fx.ptr_store(l_comb, att["lcomb"] + head)
        fx.ptr_store(sink, att["sink"] + head)
    lds_barrier()
    if const_expr(c.get("tl") is not None):
        c["tl"]("irq.stat")  # the split statistics polled and combined
    # group dc0 off V words issued before the polls; the rest a runtime loop (an
    # opaque trip count, not unrolled): a persistent kernel runs this code once a
    # layer, cold, and S = 48's four unrolled groups were ~2.8k instructions.
    # (V sets ping-ponged a group ahead: irq.pv 5.8 -> 7.0 us a CTA, K2 +2 us,
    # 243 VGPRs -- the groups are bound by the LDS transpose, not the loads; 2026-10-30)
    _pv_group(c, kv_len, splits, rows, pws, alphas, dc0, 0, pre)
    if const_expr(nmax > 1):
        for i in range(1, n):
            _pv_group(c, kv_len, splits, rows, pws, alphas, dc0 + i, i, _pv_loads(a, rows, dc0 + i, lane))


def _pv_loads(a, rows, dc, lane):
    """Group dc's V words of this lane in every split (key lane % 16's 32 columns:
    lanes 0-15 the first 16, 16-31 the rest), all in flight at once."""
    h = lane // 16
    return [
        [fx.Vector(global_load(_row_addr(a["pool"], row, (dc * COLS + 16 * h + 8 * j) * 2), 4)) for j in range(2)]
        for row in rows
    ]


def _pv_split(lane, kt, ok, vw_r, pw_r):
    """One split of a group: its V words (``ok``: this lane's key live, else zero)
    through the wave's LDS tile ``kt`` transposed into A = V^T, against its p (B)
    -> the two 16-column WMMA results (unscaled)."""
    h = lane // 16
    w, w2 = vw_r
    vals = [ok.select(w[e], fx.Int32(0)) for e in range(4)] + [ok.select(w2[e], fx.Int32(0)) for e in range(4)]
    fx.ptr_store(fx.Vector.from_elements(vals, fx.Int32), kt + ((lane % 16) * (COLS // 2) + 8 * h))
    # kt is this wave's own: a wave fence, no CTA barrier (one a split lined
    # the waves up behind the slowest V load)
    wave_lds_fence()
    pb = fx.Vector.from_elements([pw_r[0], pw_r[1], pw_r[2], pw_r[3]] + [fx.Int32(0)] * 4, fx.Int32).bitcast(
        fx.BFloat16
    )
    out = []
    for tt in range_constexpr(2):
        col = tt * 16 + lane % 16
        # A = V^T: row = column ``col``, K = keys 8 h .. 8 h + 7 (the other half zero)
        halves = []
        for j in range_constexpr(8):
            word = fx.Int32(fx.ptr_load(kt + ((8 * h + j) * (COLS // 2) + col // 2)))
            halves.append((word >> (col % 2 * 16)) & 0xFFFF)
        av = fx.Vector.from_elements(
            [halves[2 * q] | (halves[2 * q + 1] << 16) for q in range(4)] + [fx.Int32(0)] * 4, fx.Int32
        ).bitcast(fx.BFloat16)
        out.append(wmma_bf16(av, pb, fx.Vector.filled(ACC, 0.0, fx.Float32)))
    wave_lds_fence()
    return out[0], out[1]


@traced
def _pv_group(c, kv_len, splits, rows, pws, alphas, dc, i, vw):
    """Group i (columns 32 dc ..; ``vw`` its V words, ``_pv_loads``): per split,
    two WMMAs (16 columns each) of V^T against its p, scaled by alpha and summed
    in this wave; the waves' sums in order; the sink fold -> ``att["out"]``."""
    lane, wave, tid = c["lane"], c["wave"], c["tid"]
    att = c["att"]
    kt = att["kv"] + wave * (BK * COLS // 2)  # this wave's [key][32 columns] bf16 tile
    act = (kv_len + BK - 1) // BK  # splits holding a key: the rest add exactly nothing
    acc0 = fx.Vector.filled(ACC, 0.0, fx.Float32)
    acc1 = fx.Vector.filled(ACC, 0.0, fx.Float32)
    for r in range_constexpr(len(splits)):
        sp = splits[r]
        # a split past the keys (alpha = exp2(NEG - m_max) = 0) skipped: its V store,
        # the transposes and the WMMAs -- what bounds a group (the LDS transpose), so a
        # dead split cost a live one's. Wave-uniform (sp: the wave's, act: the token's);
        # the sum is the same bits (acc x 0 + accs = accs). S = 48, 1k keys: a wave's
        # live splits 3 of 4 on a selecting layer's 640 keys, 1 of 4 on a window layer
        if sp < act:
            pv0, pv1 = _pv_split(lane, kt, sp * BK + lane % 16 < kv_len, vw[r], pws[r])
            acc0 = fx.Vector.from_elements(
                [fx.Float32(fmath.fma(pv0[e], alphas[r], acc0[e])) for e in range(ACC)], fx.Float32
            )
            acc1 = fx.Vector.from_elements(
                [fx.Float32(fmath.fma(pv1[e], alphas[r], acc1[e])) for e in range(ACC)], fx.Float32
            )
    # acc<tt>[e]: column tt 16 + 8 h + e, head lane % 16 -> the waves' sums in order
    for tt, acc_tt in ((0, acc0), (1, acc1)):
        fx.ptr_store(acc_tt, att["tree"] + ((wave * WAVE + lane) * 2 + tt) * ACC)
    lds_barrier()
    hh = tid // COLS
    col = tid % COLS
    if tid < HEAD_TILE * COLS:
        tt, rem = col // 16, col % 16
        owner = 16 * (rem // 8) + hh
        tot = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            tot = tot + fx.ptr_load(att["tree"] + ((w * WAVE + owner) * 2 + tt) * ACC + rem % 8)
        mm = fx.ptr_load(att["mmax"] + hh)
        lc = fx.ptr_load(att["lcomb"] + hh)
        sk = fx.ptr_load(att["sink"] + hh)
        log2e = fx.Float32(LOG2E)
        m_final = fx.max(mm, sk * log2e)
        alpha_kv = hw_exp2(mm - m_final)
        alpha_sink = hw_exp2(fx.Float32(fmath.fma(sk, log2e, -m_final)))
        l_final = fx.Float32(fmath.fma(lc, alpha_kv, alpha_sink))
        o = (tot * alpha_kv) / fx.max(l_final, fx.Float32(1e-30))
        o = fx.Float32((l_final > fx.Float32(0.0)).select(o, fx.Float32(0.0)))
        fx.ptr_store(bf16_round(o), att["out"] + ((i * HEAD_TILE + hh) * COLS + col))
    lds_barrier()


def lds_words(s, d) -> dict:
    """The attention's LDS (f32 / i32 words) by buffer."""
    return {
        "kv": WAVES * BK * COLS // 2,
        "tree": WAVES * WAVE * 2 * ACC,
        "stat": WAVES * HEAD_TILE,
        "mmax": HEAD_TILE,
        "lcomb": HEAD_TILE,
        "sink": HEAD_TILE,
        "out": max(pv_groups(s, d.head_tiles.count), pv_split(s, d.head_tiles.count)[1]) * HEAD_TILE * COLS,
    }
