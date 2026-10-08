# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device helpers the resident MonoKernels (GLM, Kimi-K3 MLA, DeepSeek-V4) share.

Every helper is a plain function whose first argument is the kernel's ``LaunchState``: the values of
one launch (``tag``, ``lane`` / ``wave`` / ``tid``, scratch layout, LDS pointers, task bases, ...).
``LaunchState`` also exposes each helper bound to it (``st.put``, ``st.poll``, ...), and helpers call
one another through it, so a kernel that installs its own variant (``st.poll = partial(poll, st,
retry_spin_pause=True)``, or a function of its own) changes it for every helper. Variants are
keyword arguments (``poll``, ``stamp``, ``peer_reduce``) or separate functions (``unit_mxfp4_atom``,
``unit_mxfp4_rows``), each a build-time choice.
"""

import hashlib
from functools import partial
from pathlib import Path

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel import ops
from kernels.monokernel.config import EPS, FP8_MAX, SCALE_BM
from kernels.monokernel.layout import CM_DEV, CM_SYS, POLL_MAX, ROW_TILE, THREADS, TL_COLS, WAVES
from kernels.monokernel.ops import (
    bf2_f32,
    bf16_pair,
    f8_word,
    fp8_to_bf16x8,
    ld_f32,
    lds_ld,
    lds_st,
    mem_realtime,
    mxfp4_to_bf16x8,
    rcp,
    rsq,
    rsrc,
    spin_pause,
    uniform,
    wave_max,
    wave_sum,
    xshfl,
)

# A kernel's JIT disk-cache key covers only the sources in its own directory. A kernel that binds these
# helpers captures this digest of the shared device sources, so editing this file or ops.py recompiles it.
SHARED_SOURCE_KEY = hashlib.sha256(Path(__file__).read_bytes() + Path(ops.__file__).read_bytes()).hexdigest()[:16]

# Helpers with dynamic control flow need flydsl's AST rewriting (as a @flyc.kernel body gets), so that
# their ``if`` / ``while`` / ``for range`` over traced values lower to scf ops. ASTRewriter is not part of
# flydsl's stable API; this is the one place that depends on it.
traced = ASTRewriter.transform


# --------------------------------------------------------------------------- tagged-pair mailboxes


def mb(st, name):
    SC = st.SC
    scratch = st.scratch
    return scratch + fx.Int64(SC[name])


def put(st, base_addr, i, v, cm=CM_DEV):
    """Pair i := (v, tag); ``v`` f32 (or int32 bits)."""
    tag = st.tag
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    bo.buffer_store(fx.Vector.from_elements([bits, tag], fx.Int32), rsrc(base_addr), i * 2, cache_modifier=cm)


def put2(st, base_addr, i, v0, v1, cm=CM_DEV):
    """Pairs i, i+1 (i even) in one 16-byte store."""
    tag = st.tag
    vec = fx.Vector.from_elements(
        [fx.Float32(v0).bitcast(fx.Int32), tag, fx.Float32(v1).bitcast(fx.Int32), tag], fx.Int32
    )
    bo.buffer_store(vec, rsrc(base_addr), i * 2, cache_modifier=cm)


def put_bf(st, base_addr, i, vs, cm=CM_DEV, store_tag=None):
    """Elements i .. i + len(vs) (2 or 4, i aligned) as packed bf16 pairs: pair
    i / 2 + j := (bf16(vs[2j]) | bf16(vs[2j + 1]) << 16, tag), one 8 / 16-byte store."""
    tag = st.tag
    write_tag = tag if store_tag is None else store_tag
    words = []
    for j in range_constexpr(len(vs) // 2):
        words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), write_tag]
    bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), rsrc(base_addr), i, cache_modifier=cm)


@traced
def poll(
    st,
    specs,
    scope="agent",
    batch=POLL_MAX,
    expected_tag=None,
    *,
    retry_spin_pause=False,
    hang=None,
    poll_timeout_ticks=None,
):
    """Batched poll of mailbox pairs ``specs`` = [(base_addr, pair index, npairs in {1, 2})].

    All pairs are loaded together with plain 8 / 16-byte coherent buffer loads (sc1 locally,
    sc0 sc1 for peer memory); while any tag is not ``expected_tag`` (default: this launch's)
    the whole batch is re-loaded, so a batch costs one round trip after its last producer
    lands. A side-effecting op in the retry loop keeps the loads from being hoisted. With
    ``poll_timeout_ticks`` a timed-out poll stores the tag into ``hang``, and later polls
    seeing it give up. Returns one list of Int32 value bits per spec."""
    tag = st.tag
    poll = st.poll
    if const_expr(len(specs) == 0):
        return []
    if const_expr(len(specs) > batch):  # bound live registers
        return poll(specs[:batch], scope, batch, expected_tag) + poll(specs[batch:], scope, batch, expected_tag)
    cm = CM_DEV if const_expr(scope == "agent") else CM_SYS
    wanted_tag = tag if expected_tag is None else expected_tag

    def load_all():
        words = []
        for b, i, n in specs:
            w = fx.Vector(bo.buffer_load(rsrc(b), fx.Int32(i) * 2, vec_width=2 * n, dtype=T.i32, cache_modifier=cm))
            words += [w[e] for e in range(2 * n)]
        return fx.Vector.from_elements(words, fx.Int32)

    nw = sum(2 * n for _, _, n in specs)

    def unpack(v):
        outs_, e = [], 0
        for _, _, n in specs:
            outs_.append([v[e + 2 * q] for q in range(n)])
            e += 2 * n
        return outs_

    def pending(v):
        bad = v[1] != wanted_tag
        for e in range_constexpr(3, nw, 2):
            bad = bad | (v[e] != wanted_tag)
        return bad

    v = load_all()
    if const_expr(poll_timeout_ticks is not None):
        t0 = mem_realtime()
        stop = fx.Int32(0)
        while pending(v) & (stop == 0):
            rocdl.s_nop(0)
            v = load_all()
            flagged = uniform(bo.buffer_load(rsrc(hang), 0, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV)) == tag
            late = (mem_realtime() - t0) > fx.Int64(poll_timeout_ticks)
            stop = late.select(fx.Int32(2), flagged.select(fx.Int32(1), fx.Int32(0)))
        if stop == 2:
            bo.buffer_store(tag, rsrc(hang), 0, cache_modifier=CM_DEV)
    else:
        while pending(v):
            if const_expr(retry_spin_pause):
                spin_pause()
            else:
                rocdl.s_nop(0)
            v = load_all()
    return unpack(v)


def get(st, base_addr, i):
    poll = st.poll
    return poll([(base_addr, i, 1)])[0][0]


def getf(st, base_addr, i):
    get = st.get
    return get(base_addr, i).bitcast(fx.Float32)


def getf_many(st, specs):
    """[(base, i)] single pairs -> list of f32."""
    poll = st.poll
    return [v[0].bitcast(fx.Float32) for v in poll([(b, i, 1) for b, i in specs])]


def get2(st, base_addr, i):
    get2_many = st.get2_many
    return get2_many([(base_addr, i)])[0]


def get2_many(st, specs):
    """[(base, i)] double pairs (i even) -> list of (f32, f32)."""
    poll = st.poll
    return [(v[0].bitcast(fx.Float32), v[1].bitcast(fx.Float32)) for v in poll([(b, i, 2) for b, i in specs])]


def get_bf2_many(st, specs):
    """[(base, i)] packed bf16 elements i, i + 1 (i even) -> list of (f32, f32)."""
    poll = st.poll
    return [bf2_f32(v[0]) for v in poll([(b, i // 2, 1) for b, i in specs])]


@traced
def pre_poll(st, n, addr_of):
    """Wave 0 spins on one small pair per producer (lane j -> producer j < n <= 64)
    before a large payload poll, so waiting CTAs do not flood memory."""
    lane = st.lane
    wave = st.wave
    poll = st.poll
    if wave == 0:
        b, i = addr_of(fx.min(lane, n - 1))
        poll([(b, i, 1)])
    gpu.barrier()


# --------------------------------------------------------------------- timeline and task placement


@traced
def stamp(st, name, t, which, lead=0, pred=None, *, stamp_fence=False, timeline_addr=None):
    """Timeline column ``which`` of stage ``name``'s task ``t`` := s_memrealtime, written by
    thread ``lead`` (and only where ``pred``); timeline builds only."""
    first = st.first
    tid = st.tid
    timeline = st.timeline
    timeline_buf = st.timeline_buf
    tl_cols = st.tl_cols
    if const_expr(timeline):
        if const_expr(stamp_fence):
            # nothing may cross the clock read (constrains timeline builds only)
            rocdl.sched_barrier(0)
        # a masked-off rep must not overwrite the live task's row
        ok = (tid == lead) if const_expr(pred is None) else ((tid == lead) & pred)
        if ok:
            now = mem_realtime()
            tl_addr = timeline_buf if const_expr(timeline_addr is None) else timeline_addr()
            fx.generic_store(
                fx.inttoptr(
                    fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8),
                    tl_addr + fx.Int64((first[name] + t) * tl_cols + which) * 8,
                ),
                now,
            )


def hint_wait(st, n, addr_of, mark=None):
    """Block barrier before a stage polls its inputs; consumers poll their payload
    directly (tight per-wave spins). ``mark`` stamps timeline column 1."""
    stamp = st.stamp
    if const_expr(mark is not None):
        stamp(mark[0], mark[1], 1)
    gpu.barrier()


def start(st, name):
    G = st.G
    base = st.base
    bid = st.bid
    return (bid + (G - base[name])) & (G - 1)


# -------------------------------------------------------------------------------- block reductions


@traced
def block_sums(st, vs):
    """Block-wide sums of several per-thread values with one LDS exchange."""
    lane = st.lane
    red = st.red
    wave = st.wave
    ws = [wave_sum(v) for v in vs]
    if lane == 0:
        for i in range_constexpr(len(vs)):
            lds_st(red, i * WAVES + wave, ws[i])
    gpu.barrier()
    tots = []
    for i in range_constexpr(len(vs)):
        t = lds_ld(red, i * WAVES)
        for w in range_constexpr(1, WAVES):
            t = t + lds_ld(red, i * WAVES + w)
        tots.append(t)
    gpu.barrier()
    return tots


@traced
def block_sum(st, v):
    lane = st.lane
    red = st.red
    wave = st.wave
    w = wave_sum(v)
    if lane == 0:
        lds_st(red, wave, w)
    gpu.barrier()
    t = lds_ld(red, 0)
    for i in range_constexpr(1, WAVES):
        t = t + lds_ld(red, i)
    gpu.barrier()
    return t


# --------------------------------------------------------------------------------- MFMA GEMV units


def unit_fp8(st, w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word, coef=None, ln=None):
    """Issue one 64-k chunk of row group ``rg`` of a packed FP8 matrix; the
    bf16 activation chunk starts at LDS word ``b_word``."""
    lane = st.lane
    ln = lane if ln is None else ln
    wv = fx.Vector(bo.buffer_load(w_rsrc, ((rg * NKC + kc) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
    s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // BK) + kc * 64 // BK)
    if const_expr(callable(coef)):  # factor known only after a later wait
        return ("fp8", [wv], lambda: s * coef(), b_word + (lane // 16) * 4)
    if const_expr(coef is not None):
        s = s * coef
    return ("fp8", [wv], s, b_word + (lane // 16) * 4)


def unit_fp8x2(st, w_rsrc, s_rsrc, rg, kc, NKC, K, b_word, coef=None, ln=None, scale_rows=SCALE_BM):
    """Issue both 64-k halves of one 128-k FP8 weight-scale block."""
    lane = st.lane
    ln = lane if ln is None else ln
    wv = [
        fx.Vector(bo.buffer_load(w_rsrc, ((rg * NKC + kc + h) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
        for h in range(2)
    ]
    s = ld_f32(s_rsrc, (rg * 16 // scale_rows) * (K // 128) + kc // 2)
    if const_expr(callable(coef)):
        return ("fp8x2", wv, lambda: s * coef(), b_word + (lane // 16) * 4)
    if const_expr(coef is not None):
        s = s * coef
    return ("fp8x2", wv, s, b_word + (lane // 16) * 4)


def unit_f8f8(st, w_rsrc, s_rsrc, rg, kc, NKC, K, b_word, coef, ln=None):
    """Issue one 128-k chunk (64-k chunks kc, kc + 1) of row group ``rg`` against the
    FP8 activation at LDS word ``b_word``; ``coef()`` = activation scale (x route weight)."""
    lane = st.lane
    ln = lane if ln is None else ln
    wv = [
        fx.Vector(bo.buffer_load(w_rsrc, ((rg * NKC + kc + h) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
        for h in range(2)
    ]
    s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // 128) + kc // 2)
    return ("f8f8", wv, lambda: s * coef(), b_word + (lane // 16) * 4)


def unit_fp8mx(st, w_rsrc, s_rsrc, rg, s_rg, kc, K, b_word, coef=None, ln=None):
    """One 128-K chunk of row group ``rg`` of an FP8 128x128-scaled matrix (``s_rg`` =
    the row group of this lane's output rows) against bf16 at LDS word ``b_word``."""
    lane = st.lane
    ln = lane if ln is None else ln
    wv = [
        fx.Vector(bo.buffer_load(w_rsrc, ((rg * (K // 64) + kc * 2 + h) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
        for h in range(2)
    ]
    s = ld_f32(s_rsrc, (s_rg * 16 // SCALE_BM) * (K // 128) + kc)
    return ("fp8mx", wv, (s, coef), b_word + (lane // 16) * 4)


def unit_bf16(st, w_rsrc, rg, kc, NKC, b_word, ln=None):
    lane = st.lane
    ln = lane if ln is None else ln
    wv = [
        fx.Vector(bo.buffer_load(w_rsrc, (((rg * NKC + kc) * 2 + sp) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
        for sp in range(2)
    ]
    return ("bf16", wv, None, b_word + (lane // 16) * 4)


def unit_mxfp4_atom(st, w_rsrc, s_rsrc, rg, kc, K, b_word, coef=None, ln=None):
    """Issue one 128-K tile of an MXFP4 bank in ATOM's gfx950 layout and its E8M0 scale (lane
    ``16 * kl + r`` holds one 32-K scale block of row ``r``; the scale dword's four bytes cover
    tiles kc (even, odd) x row groups (even, odd))."""
    lane = st.lane
    ln = lane if ln is None else ln
    k1n = -(-(K // 32) // 8)  # scale K blocks, padded to 8, in groups of 8
    raw = fx.Vector(bo.buffer_load(w_rsrc, ((rg * (K // 128) + kc) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
    word = fx.Int32(bo.buffer_load(s_rsrc, ((rg // 2) * k1n + kc // 2) * 64 + ln, vec_width=1, dtype=T.i32))
    sc_byte = word.shrui(((kc % 2) * 2 + rg % 2) * 8) & fx.Int32(0xFF)
    return ("mxfp4_atom", (raw, sc_byte), coef, b_word + (lane // 16) * 16)


def unit_mxfp4_rows(st, w_rsrc, s_rsrc, rg, kc, K, b_word, coef=None, ln=None, *, act):
    """Issue one packed 128-K MXFP4 tile and its four per-row E8M0 scales, against ``act``
    activations: "fp8", "bf16", or "split" (bf16, one MFMA chain and factor per 32-K part)."""
    lane = st.lane
    ln = lane if ln is None else ln
    raw = fx.Vector(bo.buffer_load(w_rsrc, ((rg * (K // 128) + kc) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
    row = rg * 16 + ln % 16
    packed_scale = fx.Int32(bo.buffer_load(s_rsrc, row * (K // 128) + kc, vec_width=1, dtype=T.i32))
    scales = [
        ((packed_scale.shrui(fx.Int32(sp * 8)) & fx.Int32(0xFF)) << fx.Int32(23)).bitcast(fx.Float32)
        for sp in range_constexpr(4)
    ]
    return ("mxfp4_rows_" + act, (raw, scales), coef, b_word + (lane // 16) * 4)


def mma_units(st, acc, units):
    """acc[4] += coef * (W_chunk @ X_chunk) for every issued unit."""
    xs = st.xs
    v4f = fx.Vector.make_type(4, fx.Float32)
    for unit_format, wv, coef, bw in units:
        if const_expr(unit_format == "fp8mx"):
            # one factor per unit, so the four K32 MFMAs chain into one partial
            ws, f = coef
            c = fx.Vector.filled(4, 0.0, fx.Float32)
            for sp in range_constexpr(4):
                a = fp8_to_bf16x8(wv[sp // 2][(sp % 2) * 2], wv[sp // 2][(sp % 2) * 2 + 1])
                b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(fx.BFloat16)
                c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
            f = ws if const_expr(f is None) else ws * (f() if const_expr(callable(f)) else f)
            acc = [acc[e] + c[e] * f for e in range(4)]
            continue
        if const_expr(callable(coef) and unit_format not in ("mxfp4_atom", "mxfp4_rows_split")):
            coef = coef()
        if const_expr(unit_format == "mxfp4_atom"):
            # step sp takes K 32 * kl + 8 * sp .. of the lane's block (B words 16 * kl + 4 * sp)
            raw, sc_byte = wv
            assert not isinstance(coef, list), "per-K32 factors are folded into the operands"
            c = fx.Vector.from_elements(acc, fx.Float32) if coef is None else fx.Vector.filled(4, 0.0, fx.Float32)
            sc = (sc_byte << fx.Int32(23)).bitcast(fx.Float32)
            for sp in range_constexpr(4):
                a = mxfp4_to_bf16x8(raw[sp], sc)
                b = fx.ptr_load(xs + (bw + sp * 4), result_type=v4f).bitcast(fx.BFloat16)
                c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
            if const_expr(coef is None):
                acc = [c[e] for e in range(4)]
            else:
                f = coef() if const_expr(callable(coef)) else coef
                acc = [acc[e] + c[e] * f for e in range(4)]
            continue
        if const_expr(unit_format == "mxfp4_rows_split"):
            # one MFMA per 32-K part, each with its own factor (``coef`` may be a list)
            raw, scales = wv
            for sp in range_constexpr(4):
                a = mxfp4_to_bf16x8(raw[sp], scales[sp])
                b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(fx.BFloat16)
                c = fx.Vector.filled(4, 0.0, fx.Float32)
                c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                part_coef = coef[sp] if const_expr(isinstance(coef, list)) else coef
                if const_expr(callable(part_coef)):
                    part_coef = part_coef()
                if const_expr(part_coef is None):
                    acc = [acc[e] + c[e] for e in range(4)]
                else:
                    acc = [acc[e] + c[e] * part_coef for e in range(4)]
            continue
        c = fx.Vector.filled(4, 0.0, fx.Float32)
        if const_expr(unit_format in ("mxfp4_rows_fp8", "mxfp4_rows_bf16")):
            raw, scales = wv
            for sp in range_constexpr(4):
                a = mxfp4_to_bf16x8(raw[sp], scales[sp])
                if const_expr(unit_format == "mxfp4_rows_fp8"):
                    wh, ws = sp // 2, sp % 2
                    bv = fx.Vector(fx.ptr_load(xs + (bw + wh * 16), result_type=v4f)).bitcast(fx.Int32)
                    b = fp8_to_bf16x8(bv[ws * 2], bv[ws * 2 + 1])
                else:
                    b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(fx.BFloat16)
                c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
        if const_expr(unit_format == "f8f8"):  # one FP8 x FP8 MFMA (E8M0 scales = 1)
            a = fx.Vector.from_elements([wv[h][e] for h in range(2) for e in range(4)], fx.Int32)
            bv = [fx.Vector(fx.ptr_load(xs + (bw + h * 16), result_type=v4f)).bitcast(fx.Int32) for h in range(2)]
            b = fx.Vector.from_elements([bv[h][e] for h in range(2) for e in range(4)], fx.Int32)
            one = fx.Int32(127)
            c = fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [a, b, c, 0, 0, 0, one, 0, one]))
        nsp = (
            4
            if unit_format == "fp8x2"
            else 2 if unit_format not in ("f8f8", "mxfp4_rows_fp8", "mxfp4_rows_bf16") else 0
        )
        for sp in range_constexpr(nsp):
            if const_expr(unit_format in ("fp8", "fp8x2")):
                wh = sp // 2 if unit_format == "fp8x2" else 0
                ws = sp % 2 if unit_format == "fp8x2" else sp
                a = fp8_to_bf16x8(wv[wh][ws * 2], wv[wh][ws * 2 + 1])
            else:
                a = wv[sp].bitcast(fx.BFloat16)
            b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(fx.BFloat16)
            c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
        if const_expr(coef is None):
            acc = [acc[e] + c[e] for e in range(4)]
        else:
            acc = [acc[e] + c[e] * coef for e in range(4)]
    return acc


def run_units(st, make_unit, cpw, batch, pre=None):
    """Software pipelined: issue batch b+1's loads before computing batch b.
    ``pre`` = the already-issued first batch (prefetched before a wait)."""
    mma_units = st.mma_units
    acc = [fx.Float32(0.0) for _ in range(4)]
    starts = list(range(0, cpw, batch))
    cur = pre if pre is not None else [make_unit(c) for c in range(0, min(batch, cpw))]
    for bi in range_constexpr(len(starts)):
        nxt = None
        if const_expr(bi + 1 < len(starts)):
            n0 = starts[bi + 1]
            nxt = [make_unit(c) for c in range(n0, min(n0 + batch, cpw))]
        acc = mma_units(acc, cur)
        cur = nxt
    return acc


@traced
def reduce_rows(st, R, acc, emit, count=None):
    """Sum per-wave MFMA tiles; emit(row_local, local sample column, value)."""
    S = st.S
    lane = st.lane
    red = st.red
    tid = st.tid
    wave = st.wave
    count = S if count is None else count
    wpr = WAVES // R
    fx.ptr_store(fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4)
    gpu.barrier()
    n_out = R * 16 * count
    for i in range_constexpr((n_out + THREADS - 1) // THREADS):
        t = tid + i * THREADS
        if t < n_out:
            rl = t % (R * 16)
            n = t // (R * 16)
            r = rl % 16
            tot = fx.Float32(0.0)
            for j in range_constexpr(wpr):
                ww = (rl // 16) * wpr + j
                tot = tot + lds_ld(red, (ww * 64 + n + 16 * (r // 4)) * 4 + r % 4)
            emit(rl, n, tot)


def emit_out(st, stride):
    outs = st.outs

    def f(rl, n, v):
        lds_st(outs, n * stride + rl, v)

    return f


def n_sel(st, count=None):
    """This lane's local MFMA B column; inactive columns duplicate the last one."""
    S = st.S
    lane = st.lane
    count = S if count is None else count
    return fx.min(lane % 16, count - 1)


# ------------------------------------------------------------------------------ activation staging


def rmsnorm_tail_ks(st, n):
    """This thread's group-of-4 starting indices of n elements. A ragged tail is
    clamped to the last group (idempotent rewrites); ``active`` masks its
    contribution to sums (None when there is no tail)."""
    tid = st.tid
    nq = n // 4
    full = nq // THREADS
    ks = [(tid + i * THREADS) * 4 for i in range(full)]
    active = None
    if const_expr(nq % THREADS):
        w = tid + full * THREADS
        active = w < nq
        ks.append(fx.min(w, nq - 1) * 4)
    return ks, active


def load_x_rmsnorm(st, ld4s, n, gamma, count=None):
    """The gamma loads (issued ahead of the wait), then ld4s -> (gammas, x values)."""
    S = st.S
    rmsnorm_tail_ks = st.rmsnorm_tail_ks
    count = S if count is None else count
    rg_ = rsrc(gamma)
    ks, _ = rmsnorm_tail_ks(n)
    gs = []
    for k in ks:
        g = fx.Vector(bo.buffer_load(rg_, k // 2, vec_width=2, dtype=T.i32)).bitcast(fx.BFloat16).to(fx.Float32)
        gs.append([g[j] for j in range(4)])
    return gs, ld4s([(s, k) for s in range(count) for k in ks])


def stage_x_rmsnorm(st, ld4s, n, gamma, mark=None, loaded=None, count=None):
    """LDS bf16 X[s][0:n] = bf16(rmsnorm(x_s) * gamma) for every sample s, where
    ld4s([(s, k)]) -> [(x_s[k], .., x_s[k+3])] (one batched load); returns the rstds.
    ``loaded``: the (gamma, x) loads already issued by load_x_rmsnorm; ``mark``: the
    (stage, task) whose timeline columns 6 / 7 bracket the block reduction."""
    S = st.S
    eps = st.eps
    xs = st.xs
    block_sums = st.block_sums
    load_x_rmsnorm = st.load_x_rmsnorm
    rmsnorm_tail_ks = st.rmsnorm_tail_ks
    stamp = st.stamp
    count = S if count is None else count
    ks, active = rmsnorm_tail_ks(n)
    per = len(ks)
    gs, vals = loaded if loaded is not None else load_x_rmsnorm(ld4s, n, gamma, count)
    sss = []
    for s in range_constexpr(count):
        ss = fx.Float32(0.0)
        for i in range_constexpr(per):
            for a in vals[s * per + i]:
                term = a * a
                if const_expr(active is not None and i == per - 1):
                    term = active.select(term, fx.Float32(0.0))
                ss = ss + term
        sss.append(ss)
    if const_expr(mark is not None):
        stamp(mark[0], mark[1], 6)
    rstds = [rsq(tot * (1.0 / n) + eps) for tot in block_sums(sss)]
    if const_expr(mark is not None):
        stamp(mark[0], mark[1], 7)
    for s in range_constexpr(count):
        for i in range_constexpr(per):
            a = vals[s * per + i]
            for j in range_constexpr(2):
                lds_st(
                    xs,
                    (s * n + ks[i]) // 2 + j,
                    bf16_pair(a[2 * j] * rstds[s] * gs[i][2 * j], a[2 * j + 1] * rstds[s] * gs[i][2 * j + 1]),
                )
    return rstds


@traced
def stage_x_pairs(st, name, n_total, src_of):
    """LDS bf16 X[k] = packed bf16 mailbox ``name`` element src_of(k) for k < n_total
    (src_of contiguous over aligned groups of 4): one 16-byte poll per 4 elements."""
    tid = st.tid
    xs = st.xs
    mb = st.mb
    poll = st.poll
    nq = n_total // 4
    full = nq // THREADS
    vals = poll([(mb(name), src_of((tid + i * THREADS) * 4) // 2, 2) for i in range(full)])
    for i in range_constexpr(full):
        for j in range_constexpr(2):
            lds_st(xs, (tid + i * THREADS) * 2 + j, vals[i][j].bitcast(fx.Float32))
    if const_expr(nq % THREADS):
        w = tid + full * THREADS
        if w < nq:
            v = poll([(mb(name), src_of(w * 4) // 2, 2)])[0]
            for j in range_constexpr(2):
                lds_st(xs, w * 2 + j, v[j].bitcast(fx.Float32))


def quant_scaled(st, a0, a1):
    """Per-wave FP8 quant of a 128-block held as 2 f32 per lane -> (scaled q0, q1, scale)."""
    amax = wave_max(fx.max(fmath.absf(a0), fmath.absf(a1)))
    nz = amax > 0.0
    qs = nz.select(amax * (1.0 / FP8_MAX), fx.Float32(1.0))
    inv = nz.select(rcp(amax) * FP8_MAX, fx.Float32(1.0))
    q0 = fx.min(fx.max(a0 * inv, -FP8_MAX), FP8_MAX)
    q1 = fx.min(fx.max(a1 * inv, -FP8_MAX), FP8_MAX)
    return q0, q1, qs


@traced
def st_f8(st, k, q0, q1):
    """LDS FP8 activation bytes k, k + 1 (k even, held by this lane; lane ^ 1 holds
    k ^ 2) in ``f8_word`` order.  Call from the whole wave."""
    lane = st.lane
    xs = st.xs
    w = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
    nb = xshfl(w, 1)
    if lane % 2 == 0:
        lds_st(xs, f8_word(k), (w | (nb << 16)).bitcast(fx.Float32))


def load_bias(st):
    """This lane's 4 expert biases (issue before the scores wait)."""
    N_EXPERTS = st.N_EXPERTS
    bias = st.bias
    lane = st.lane
    return [ld_f32(rsrc(bias), lane + i * 64) for i in range(N_EXPERTS // 64)]


# ---------------------------------------------------------------------------------- TP peer reduce


@traced
def owner_peer_reduce(st, region_base, t, residual, out_fn, tile, *, r_peers=None):
    """``owner_reduce``: the rank owning a tile's rows sums it once and sends the result back."""
    HIDDEN = st.HIDDEN
    S = st.S
    W = st.W
    lane = st.lane
    outs = st.outs
    peer_dst = st.peer_dst
    rank = st.rank
    sym = st.sym
    tag = st.tag
    tid = st.tid
    wave = st.wave
    poll = st.poll
    put_bf = st.put_bf
    pair_count = S * tile // 2
    owner_rank = (t * tile) // (HIDDEN // W)
    owner_words = fx.Vector(bo.buffer_load(r_peers, owner_rank * 2, vec_width=2, dtype=T.i32))
    owner_dst = (fx.Int64(uniform(owner_words[1])) << 32) | fx.Int64(fx.Uint32(uniform(owner_words[0])))
    if wave == 0:
        for batch in range_constexpr((pair_count + 63) // 64):
            pair = lane + batch * 64
            if pair < pair_count:
                si = pair // (tile // 2)
                ri = (pair % (tile // 2)) * 2
                put_bf(
                    owner_dst + region_base,
                    (rank * S + si) * HIDDEN + t * tile + ri,
                    [lds_ld(outs, si * tile + ri), lds_ld(outs, si * tile + ri + 1)],
                    CM_SYS,
                )
    gpu.barrier()

    if (rank == owner_rank) & (tid < pair_count):
        s = tid // (tile // 2)
        r = (tid % (tile // 2)) * 2
        row = t * tile + r
        r0, r1 = residual(s, row)
        own = sym + region_base
        got = poll(
            [(own, ((src * S + s) * HIDDEN + row) // 2, 1) for src in range(W)],
            "one-as",
        )
        t0 = fx.Float32(0.0)
        t1 = fx.Float32(0.0)
        for src in range_constexpr(W):
            p0, p1 = bf2_f32(got[src][0])
            t0 = t0 + p0
            t1 = t1 + p1
        lds_st(outs, tid, bf16_pair(r0 + t0, r1 + t1))
    gpu.barrier()

    result_tag = tag + fx.Int32(1 << 30)
    if (rank == owner_rank) & (wave < W):
        for batch in range_constexpr((pair_count + 63) // 64):
            pair = lane + batch * 64
            if pair < pair_count:
                s = pair // (tile // 2)
                r = (pair % (tile // 2)) * 2
                packed = lds_ld(outs, pair).bitcast(fx.Int32)
                mailbox = ((owner_rank * S + s) * HIDDEN + t * tile + r) // 2
                bo.buffer_store(
                    fx.Vector.from_elements([packed, result_tag], fx.Int32),
                    rsrc(peer_dst + region_base),
                    mailbox * 2,
                    cache_modifier=CM_SYS,
                )
    gpu.barrier()

    if tid < pair_count:
        s = tid // (tile // 2)
        r = (tid % (tile // 2)) * 2
        row = t * tile + r
        own = sym + region_base
        got = poll(
            [(own, ((owner_rank * S + s) * HIDDEN + row) // 2, 1)],
            "one-as",
            expected_tag=result_tag,
        )
        v0, v1 = bf2_f32(got[0][0])
        out_fn(s, row, v0, v1)


@traced
def peer_reduce(st, region, t, residual, out_fn, tile=ROW_TILE, *, owner_reduce=False, r_peers=None):
    """Push BF16 partials in tagged pairs to every peer, then sum all ranks' pairs from the own
    symmetric buffer in rank order (W = 1: no exchange). ``residual`` is fn(s, row) -> (r0, r1)
    (plain loads, issued first), a mailbox base (pairs s * HIDDEN + row, polled with the peers),
    or None when the caller adds its own. With ``peer_slot`` consecutive epochs use alternating
    ``_part_stride`` slots: a rank cannot finish epoch k + 1 before every peer enters it, so no
    rank reaches k + 2 soon enough to overwrite epoch k while it is still being consumed. With
    ``owner_reduce`` each tile is summed once by the rank that owns its rows and sent back."""
    HIDDEN = st.HIDDEN
    S = st.S
    SY = st.SY
    W = st.W
    lane = st.lane
    outs = st.outs
    peer_dst = st.peer_dst
    peer_slot = st.peer_slot
    rank = st.rank
    sym = st.sym
    tid = st.tid
    wave = st.wave
    owner_peer_reduce = st.owner_peer_reduce
    poll = st.poll
    put_bf = st.put_bf
    region_base = fx.Int64(SY[region])
    if const_expr(peer_slot is not None):
        region_base = region_base + fx.Int64(peer_slot) * fx.Int64(SY["_part_stride"])
    if const_expr(owner_reduce and W > 1):
        owner_peer_reduce(region_base, t, residual, out_fn, tile, r_peers=r_peers)
        return
    if const_expr(W > 1):
        # one wave per destination peer
        if wave < W:
            pair_count = S * tile // 2
            for batch in range_constexpr((pair_count + 63) // 64):
                pair = lane + batch * 64
                if pair < pair_count:
                    si = pair // (tile // 2)
                    ri = (pair % (tile // 2)) * 2
                    put_bf(
                        peer_dst + region_base,
                        (rank * S + si) * HIDDEN + t * tile + ri,
                        [lds_ld(outs, si * tile + ri), lds_ld(outs, si * tile + ri + 1)],
                        CM_SYS,
                    )
        gpu.barrier()
    if tid < S * tile // 2:
        s = tid // (tile // 2)
        r = (tid % (tile // 2)) * 2
        row = t * tile + r
        r0 = fx.Float32(0.0)
        r1 = fx.Float32(0.0)
        if const_expr(callable(residual)):
            r0, r1 = residual(s, row)
        v0 = lds_ld(outs, s * tile + r)
        v1 = lds_ld(outs, s * tile + r + 1)
        if const_expr(W == 1):
            parts = [(v0, v1)]
            got = []
            if const_expr(residual is not None and not callable(residual)):
                got = poll([(residual, (s * HIDDEN + row) // 2, 1)])
        else:
            own = sym + region_base
            specs = [(own, ((src * S + s) * HIDDEN + row) // 2, 1) for src in range(W)]
            if const_expr(residual is not None and not callable(residual)):
                specs.append((residual, (s * HIDDEN + row) // 2, 1))
            got = poll(specs, "one-as")
            parts = [bf2_f32(v[0]) for v in got[:W]]
            got = got[W:]
        if const_expr(residual is not None and not callable(residual)):
            r0, r1 = bf2_f32(got[0][0])
        t0 = fx.Float32(0.0)
        t1 = fx.Float32(0.0)
        for src in range_constexpr(W):
            t0 = t0 + parts[src][0]
            t1 = t1 + parts[src][1]
        out_fn(s, row, r0 + t0, r1 + t1)


HELPERS = (
    mb,
    put,
    put2,
    put_bf,
    poll,
    get,
    getf,
    getf_many,
    get2,
    get2_many,
    get_bf2_many,
    pre_poll,
    stamp,
    hint_wait,
    start,
    block_sums,
    block_sum,
    unit_fp8,
    unit_fp8x2,
    unit_f8f8,
    unit_fp8mx,
    unit_bf16,
    unit_mxfp4_atom,
    unit_mxfp4_rows,
    mma_units,
    run_units,
    reduce_rows,
    emit_out,
    n_sel,
    rmsnorm_tail_ks,
    load_x_rmsnorm,
    stage_x_rmsnorm,
    stage_x_pairs,
    quant_scaled,
    st_f8,
    load_bias,
    owner_peer_reduce,
    peer_reduce,
)


class LaunchState:
    """One launch's values for the shared helpers, and the helpers bound to it (``st.put`` ...).

    ``source_key`` is the kernel's captured ``SHARED_SOURCE_KEY`` (see there)."""

    def __init__(
        self,
        *,
        source_key,
        S,
        tid,
        lane,
        wave,
        bid,
        G,
        base,
        scratch,
        SC,
        tag,
        red,
        outs,
        xs,
        bias=None,
        N_EXPERTS=None,
        timeline=False,
        timeline_buf=None,
        first=None,
        tl_cols=TL_COLS,
        rank=None,
        W=1,
        HIDDEN=None,
        peer_dst=None,
        sym=None,
        SY=None,
        peer_slot=None,
        eps=EPS,
    ):
        assert source_key == SHARED_SOURCE_KEY, "the kernel was built against older shared helper sources"
        self.S = S
        self.tid = tid
        self.lane = lane
        self.wave = wave
        self.bid = bid
        self.G = G
        self.base = base
        self.scratch = scratch
        self.SC = SC
        self.tag = tag
        self.red = red
        self.outs = outs
        self.xs = xs
        self.bias = bias
        self.N_EXPERTS = N_EXPERTS
        self.timeline = timeline
        self.timeline_buf = timeline_buf
        self.first = first
        self.tl_cols = tl_cols
        self.rank = rank
        self.W = W
        self.HIDDEN = HIDDEN
        self.peer_dst = peer_dst
        self.sym = sym
        self.SY = SY
        self.peer_slot = peer_slot
        self.eps = eps
        for fn in HELPERS:
            setattr(self, fn.__name__, partial(fn, self))
