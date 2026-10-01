# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""GLM-5 indexed decode MonoKernel for gfx1250: one persistent launch per TP rank.

This is the wave32 WMMA counterpart of ``kernel.py``.  It keeps that kernel's
stage list, CTA placement, tagged-pair mailbox protocol, scratch and symmetric
layouts (except the compact indices below), and numerics, so the host wrapper,
golden and tests are shared.  One
launch of ``grid = 256 CTAs x 512 threads`` (one CTA per gfx1250 WGP) runs the
whole decoder-layer body for this rank's TP shard.

Differences from the gfx950 kernel:

* A CTA has 16 wave32 waves instead of 8 wave64 waves.  Stages that assign one
  wave per head, row group or key group keep eight working waves; the other
  eight repeat that work and their results are never read.
* GEMVs run on ``v_wmma_f32_16x16x32_bf16`` with weights in the gfx1250 tile
  order (``pack_*_gfx1250``): a lane owns 32 bytes of every 1 KB 16-row tile.  FP8
  and MXFP4 weights are widened exactly to BF16 with ``v_cvt_scale_pk8``, and FP8
  activations are staged in LDS as their exact BF16 values, so every product is
  exact and accumulated in FP32.  The native FP8 WMMA truncates its internal sum
  to about 2^-15 of the absolute product sum.
* One 16x16 FP32 accumulator tile is eight values per lane: lane ``l`` holds
  column ``l % 16`` and rows ``(l // 16) * 8 + e``.
* A 128-element FP8 activation block is quantized by one wave with four values
  per lane, and top-8 routing sorts eight expert candidates per lane.
* Mailboxes use the gfx12 ``scope`` field: ``SCOPE_DEV`` within the GPU and
  ``SCOPE_SYS`` for peer buffers.
* The index-select CTA writes each compact top-k index as a tagged pair, which
  the attention CTAs poll directly, instead of plain words published by a
  release fence and one ``indices_ready`` tag.
* ``timeline=True`` reads the 100 MHz steady counter.
* ``poll_limit`` bounds the re-polls of every mailbox wait.  An expired wait
  sets its stage's ``poll_err`` scratch word and continues, so a protocol defect
  finishes the launch with wrong values instead of spinning indefinitely.  The
  bound counts iterations: reading the steady counter in thousands of spinning
  waves stalled the other waves of the grid for as long as they spun.

TileRT shared/reuse MonoKernel reference (fusion-boundary comparison):
https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.config import (
    EPS,
    FP8_MAX,
    HIDDEN,
    INTER,
    KV_LORA,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    ROUTE_SCALE,
    SCALE_BM,
    SHARED_EXPERT,
    SOFTMAX_SCALE,
    TOP_K,
    V_DIM,
)
from kernels.monokernel.glm.layout import (
    BLOCKS,
    INDEX_DIM,
    INDEX_HEADS,
    INDEX_KEYS_PER_TASK,
    INDEX_Q_ROWS,
    INDEX_TILE,
    N_QKV_A,
    N_ROUTER,
    N_ROW_TILES,
    N_UG_PER_SLOT,
    POLL_STAGES,
    Q_B_TILE,
    QKV_A_TILE,
    ROUTER_TILE,
    ROW_TILE,
    UG_TILE,
    UK_TILE,
    UV_TILE,
    XQ_BLOCKS,
    XQ_WAVES,
    dn_tile,
    layout,
    sparse_keys_per_task,
    stage_tasks,
)
from kernels.monokernel.layout import LAYER_SLOTS, NEG, POLL_MAX, THREADS, TL_COLS
from kernels.monokernel.ops import (
    bpermute_i32,
    fp4x8_to_bf16_gfx1250,
    fp8x8_to_bf16_gfx1250,
    spin_pause,
    steady_counter,
    wave32_umax,
    wmma_bf16_gfx1250,
    write_lane_i32,
)
from kernels.monokernel.ops import (
    exp as _exp,
)
from kernels.monokernel.ops import (
    rcp as _rcp,
)
from kernels.monokernel.ops import (
    rsq as _rsq,
)
from kernels.monokernel.ops import (
    rsrc as _rsrc,
)
from kernels.monokernel.ops import (
    uniform as _uniform,
)
from kernels.monokernel.ops import (
    uniform_f32 as _uniform_f32,
)
from kernels.monokernel.ops import (
    xred as _xred,
)
from kernels.monokernel.ops import (
    xshfl as _xshfl,
)

WAVE = 32
WAVES = THREADS // WAVE
ACC = 8  # FP32 accumulator values per lane of one 16x16 WMMA tile
CM_DEV = 16  # gfx12 scope:SCOPE_DEV
CM_SYS = 24  # gfx12 scope:SCOPE_SYS
LDS_BYTES = 327680
# With lanes 0-15 holding the E8M0 bytes (s0, s0, s1, s1) of their row and lanes
# 16-31 holding (s2, s2, s3, s3), these v_cvt_scale_pk8 selectors give every lane
# the scale of 32-K step 0, 1, 2 and 3 of a 128-K MXFP4 tile.
MXFP4_SCALE_SEL = (0, 2, 1, 3)
# Expert candidates per lane in top-8 routing, and a sorting network for them.
ROUTE_CANDIDATES = N_EXPERTS // WAVE
SORT8 = (
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (2, 4),
    (3, 5),
    (1, 4),
    (3, 6),
    (1, 2),
    (3, 4),
    (5, 6),
)
# Stages whose natural decomposition has eight parts keep eight working waves.
WORK_WAVES = 8


def build_glm5_monokernel_gfx1250(
    S: int = 1,
    heads: int = 8,
    npes: int = 8,
    topk: int = 2048,
    launches_per_step: int = 1,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    expert_mxfp4: bool = False,
    scale: float = SOFTMAX_SCALE,
    uv_scale_rows: int = 128,
    timeline: bool = False,
    poll_limit: int | None = None,
):
    """Return the ``@flyc.jit`` launcher for one rank's whole layer on gfx1250.

    The launcher ABI, scratch layout and ``timeline`` buffer match
    :func:`kernels.monokernel.glm.kernel.build_glm5_monokernel`.
    """
    assert uv_scale_rows in (64, 128)
    assert heads == 8, "the split-attention mapping uses one wave per local head"
    assert ROUTE_CANDIDATES == len({i for pair in SORT8 for i in pair})
    SPLIT_KEYS = sparse_keys_per_task(S)
    assert topk % SPLIT_KEYS == 0 and 1 <= S <= 8
    assert 1 <= launches_per_step <= LAYER_SLOTS
    assert not with_indexer or (topk == 2048 and index_max_seq % INDEX_KEYS_PER_TASK == 0)
    assert index_max_seq < 1 << 16, "the index-select compaction scans two key counts packed in 16-bit halves"
    assert poll_limit is None or poll_limit > 0
    H = heads
    W = npes
    G = BLOCKS
    SC, SY = layout(S, H, W, topk, with_indexer, index_max_seq, tagged_indices=True)
    N_SPLIT = topk // SPLIT_KEYS
    QB_ROWS = H * (NOPE_DIM + PE_DIM)
    N_QB = QB_ROWS // Q_B_TILE
    assert not with_indexer or (INDEX_Q_ROWS // INDEX_TILE == G and N_QB * 2 == G)
    QB_PER_HEAD = (NOPE_DIM + PE_DIM) // Q_B_TILE
    N_UK = H * KV_LORA // UK_TILE
    UK_PER_HEAD = KV_LORA // UK_TILE
    assert UK_TILE // 16 == WORK_WAVES
    N_UV = H * V_DIM // UV_TILE
    O_K = H * V_DIM
    N_UG = S * MOE_SLOTS * N_UG_PER_SLOT
    QK_DIM = KV_LORA + PE_DIM
    # split LDS: bf16 q of all heads, then the KV latent / k_pe tiles (bf16 pairs);
    # row strides are padded by 4 words so the WMMA operand rows spread over the banks
    QS = QK_DIM // 2 + 4
    KS = KV_LORA // 2 + 4
    PS = PE_DIM // 2 + 4
    KT_OFF = H * QS
    PT_OFF = KT_OFF + SPLIT_KEYS * KS
    KEY_REPS = SPLIT_KEYS // WAVE  # keys per lane in the split-local softmax
    # The 320 KB LDS holds every sample's normalized input (S * XW words, already
    # reserved for the MoE activation), so the input projection reads its
    # weights once instead of once per four samples.
    SAMPLE_TILE = S
    DN_TILE = dn_tile(S, expert_mxfp4)
    N_DN_TILES = HIDDEN // DN_TILE
    RED_WORDS = WAVES * WAVE * ACC
    # index select: keys[0:256] hold the radix histogram, keys[256:256 + WAVES] the
    # compaction's wave totals and keys[SEL_OUT:SEL_OUT + 2] each pass's digit
    SEL_OUT = 256 + WAVES
    LDS_KEYS = max(SEL_OUT + 2 if with_indexer else 0, SPLIT_KEYS, S * MOE_SLOTS)
    XW = HIDDEN // 2  # LDS words of one sample's BF16 activation

    # TileRT lineage: one phase-overlaid arena.  FP8 activations are staged as
    # BF16, so the all-sample MoE activation is twice the gfx950 FP8 tile; the
    # 320 KB WGP LDS holds it with room to spare.
    SPLIT_X_WORDS = PT_OFF + SPLIT_KEYS * PS
    X_WORDS = max(S * XW, SPLIT_X_WORDS, index_max_seq)
    MISC_OFF = X_WORDS
    MISC_WORDS = max(8 + S * XQ_BLOCKS, S * MOE_SLOTS * (INTER // 128), N_SPLIT)
    KEYS_OFF = MISC_OFF + MISC_WORDS
    DNW_OFF = KEYS_OFF + LDS_KEYS
    RED_OFF = (DNW_OFF + S * MOE_SLOTS + 7) // 8 * 8  # 32-byte aligned accumulator tiles
    OUT_OFF = RED_OFF + RED_WORDS
    # UK writes its 128-row result straight from the reduction tile to q_lat;
    # all remaining stages need at most these compact output tiles.
    OUT_WORDS = max(S * ROW_TILE, S * 2 * UG_TILE)
    WORK_WORDS = OUT_OFF + OUT_WORDS
    assert WORK_WORDS * 4 <= LDS_BYTES, "keep static LDS within the gfx1250 WGP budget"

    base, first, acc = {}, {}, 0
    for name, n in stage_tasks(S, H, topk, with_indexer, index_max_seq, expert_mxfp4):
        first[name] = acc
        acc += n
    # CTA placement: split before uk, so every split tile lands on a CTA freed by
    # qkv_a (uk shares the q_b CTAs it waits on anyway)
    tasks = dict(stage_tasks(S, H, topk, with_indexer, index_max_seq, expert_mxfp4))
    acc = 0
    for name in ("qkv_a", "q_norm", "cache", "q_b", "split", "uk", "uv", "o", "router", "ug", "down"):
        base[name] = acc % G
        acc += tasks[name]
    if with_indexer:
        # q_b occupies exactly half the grid.  Put the second half of index-Q on
        # the complementary CTAs while q_b CTAs reuse their normalized q_lora
        # tile for the first half, instead of serializing two index-Q tiles on
        # every q_b CTA.
        base["index_q"] = (base["q_b"] + N_QB) % G
        base["index_score"] = 101
        base["index_select"] = 100

    @fx.struct
    class Smem:
        work: fx.Array[fx.Float32, WORK_WORDS, 16]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def glm5_monokernel_gfx1250(
        h_in: fx.Int64,
        x_out: fx.Int64,
        cur_pos: fx.Int64,
        kv_cache: fx.Int64,
        pe_cache: fx.Int64,
        indices: fx.Int64,
        rope_cos: fx.Int64,
        rope_sin: fx.Int64,
        g_in: fx.Int64,
        g_q: fx.Int64,
        g_kv: fx.Int64,
        g_post: fx.Int64,
        w_qkv_a: fx.Int64,
        s_qkv_a: fx.Int64,
        w_q_b: fx.Int64,
        s_q_b: fx.Int64,
        w_uk: fx.Int64,
        s_uk: fx.Int64,
        w_uv: fx.Int64,
        s_uv: fx.Int64,
        w_o: fx.Int64,
        s_o: fx.Int64,
        w_r: fx.Int64,
        bias: fx.Int64,
        w_ug: fx.Int64,
        s_ug: fx.Int64,
        w_dn: fx.Int64,
        s_dn: fx.Int64,
        scratch: fx.Int64,
        sym: fx.Int64,
        peers: fx.Int64,
        timeline_buf: fx.Int64,
        step: fx.Int64,
        rank: fx.Int32,
        layer: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % WAVE
        wave = tid // WAVE
        allocator = fx.SharedAllocator()
        lds = allocator.allocate(Smem).peek()
        xs = lds.work.ptr
        misc = xs + MISC_OFF
        keys = fx.recast_iter(fx.Int32, xs + KEYS_OFF)
        dnw = xs + DNW_OFF
        red = xs + RED_OFF
        outs = xs + OUT_OFF
        pl = xs  # score Q is dead before split probabilities are written
        attn_keys = keys
        ktile = xs + KT_OFF  # f32-typed views holding raw bf16 pairs
        petile = xs + PT_OFF
        v4f = fx.Vector.make_type(4, fx.Float32)

        r_h = _rsrc(h_in)
        # this launch's epoch: every mailbox tag must equal it.  ``step`` is a
        # device counter bumped once per decode step (graph friendly); ``layer``
        # makes it unique per layer within the step.
        step_value = _uniform(bo.buffer_load(_rsrc(step), 0, vec_width=1, dtype=T.i32))
        tag = step_value * LAYER_SLOTS + layer + 1
        peer_slot = (step_value * launches_per_step + layer) & 1
        pos0 = _uniform(bo.buffer_load(_rsrc(cur_pos), 0, vec_width=1, dtype=T.i32))
        r_peers = _rsrc(peers)
        # One wave sends to one peer, so retain only that wave's destination.
        pv = fx.Vector(bo.buffer_load(r_peers, fx.min(wave, W - 1) * 2, vec_width=2, dtype=T.i32))
        peer_dst = (fx.Int64(_uniform(pv[1])) << 32) | fx.Int64(fx.Uint32(_uniform(pv[0])))

        # ------------------------------------------------------------ helpers
        def ld_f32(r, i):
            return fx.Float32(bo.buffer_load(r, i, vec_width=1, dtype=T.f32))

        def ld_bf16(r, i):
            return fx.Float32(fx.BFloat16(bo.buffer_load(r, i, vec_width=1, dtype=T.bf16)))

        def lds_ld(ptr, i):
            return fx.ptr_load(ptr + i)

        def lds_st(ptr, i, v):
            fx.ptr_store(v, ptr + i)

        def bf16_pair(a, b):
            """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
            return fx.Vector.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Float32)[0]

        def bf16_round(a):
            return fx.Float32(fx.Float32(a).to(fx.BFloat16))

        def index_arg(i):
            """Load one uniform pointer from the compact indexer parameter table.

            Fused-indexer launches carry this table in the otherwise independent
            timeline argument, keeping the no-indexer kernel ABI identical to the
            original layer.  The indices argument similarly carries index_cache.
            """
            pv = fx.Vector(bo.buffer_load(_rsrc(timeline_buf), i * 2, vec_width=2, dtype=T.i32))
            return (fx.Int64(_uniform(pv[1])) << 32) | fx.Int64(fx.Uint32(_uniform(pv[0])))

        # ---- tagged-pair mailboxes
        # TileRT lineage: payload + launch epoch is the progress protocol for
        # resident CTAs; the helpers below are the FlyDSL/ROCm adaptation.
        stage_now = ["qkv_a"]  # the stage being traced: names a timed-out poll

        def enter(name):
            stage_now[0] = name

        def mb(name):
            return scratch + fx.Int64(SC[name])

        def put(base_addr, i, v, cm=CM_DEV):
            """Pair i := (v, tag); ``v`` f32 (or int32 bits)."""
            bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
            bo.buffer_store(fx.Vector.from_elements([bits, tag], fx.Int32), _rsrc(base_addr), i * 2, cache_modifier=cm)

        def put2(base_addr, i, v0, v1, cm=CM_DEV):
            """Pairs i, i+1 (i even) in one 16-byte store."""
            vec = fx.Vector.from_elements(
                [fx.Float32(v0).bitcast(fx.Int32), tag, fx.Float32(v1).bitcast(fx.Int32), tag], fx.Int32
            )
            bo.buffer_store(vec, _rsrc(base_addr), i * 2, cache_modifier=cm)

        def put_bf(base_addr, i, vs, cm=CM_DEV):
            """Elements i .. i + len(vs) (2 or 4, i aligned) as packed bf16 pairs: pair
            i / 2 + j := (bf16(vs[2j]) | bf16(vs[2j + 1]) << 16, tag), one 8 / 16-byte store."""
            words = []
            for j in range_constexpr(len(vs) // 2):
                words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), tag]
            bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), _rsrc(base_addr), i, cache_modifier=cm)

        def bf2_f32(w):
            """Packed bf16 pair word -> (f32 low, f32 high)."""
            return (w << 16).bitcast(fx.Float32), (w & fx.Int32(-65536)).bitcast(fx.Float32)

        def poll(specs, scope="agent", batch=POLL_MAX):
            """Batched poll of mailbox pairs: ``specs`` = [(base_addr, pair index, npairs in {1, 2})].

            All pairs are loaded together with plain 8 / 16-byte scoped buffer loads
            (SCOPE_DEV locally, SCOPE_SYS for peer memory); while any tag is not this
            launch's the whole batch is re-loaded.  A side-effecting asm statement in
            the retry loop keeps the loads from being hoisted.  Returns one list of
            Int32 value bits per spec."""
            if const_expr(len(specs) == 0):
                return []
            if const_expr(len(specs) > batch):  # bound live registers
                return poll(specs[:batch], scope, batch) + poll(specs[batch:], scope, batch)
            cm = CM_DEV if const_expr(scope == "agent") else CM_SYS

            def load_all():
                words = []
                for b, i, n in specs:
                    w = fx.Vector(
                        bo.buffer_load(_rsrc(b), fx.Int32(i) * 2, vec_width=2 * n, dtype=T.i32, cache_modifier=cm)
                    )
                    words += [w[e] for e in range(2 * n)]
                return fx.Vector.from_elements(words, fx.Int32)

            nw = sum(2 * n for _, _, n in specs)

            def pending(v):
                bad = v[1] != tag
                for e in range_constexpr(3, nw, 2):
                    bad = bad | (v[e] != tag)
                return bad

            v = load_all()
            if const_expr(poll_limit is None):
                while pending(v):
                    spin_pause()
                    v = load_all()
            else:
                retries = fx.Int32(0)
                while pending(v) & (retries < poll_limit):
                    spin_pause()
                    v = load_all()
                    retries = retries + 1
                if pending(v):
                    bo.buffer_store(fx.Int32(1), _rsrc(mb("poll_err")), POLL_STAGES.index(stage_now[0]))
            outs_, e = [], 0
            for _, _, n in specs:
                outs_.append([v[e + 2 * q] for q in range(n)])
                e += 2 * n
            return outs_

        def hint_wait(n, addr_of, mark=None):
            """Consumers poll their payload directly (tight per-wave spins); a wave-0
            pre-poll of each producer's last pair only added a hop of latency."""
            if const_expr(mark is not None):
                stamp(mark[0], mark[1], 1)
            gpu.barrier()

        def pre_poll(n, addr_of):
            """Wave 0 spins on one small pair per producer (lane j -> producer j) before
            a large payload poll, so waiting CTAs do not flood memory."""
            if wave == 0:
                b, i = addr_of(fx.min(lane, n - 1))
                poll([(b, i, 1)])
            gpu.barrier()

        def get(base_addr, i):
            return poll([(base_addr, i, 1)])[0][0]

        def getf(base_addr, i):
            return get(base_addr, i).bitcast(fx.Float32)

        def getf_many(specs):
            """[(base, i)] single pairs -> list of f32."""
            return [v[0].bitcast(fx.Float32) for v in poll([(b, i, 1) for b, i in specs])]

        def get2_many(specs):
            """[(base, i)] double pairs (i even) -> list of (f32, f32)."""
            return [(v[0].bitcast(fx.Float32), v[1].bitcast(fx.Float32)) for v in poll([(b, i, 2) for b, i in specs])]

        def get2(base_addr, i):
            return get2_many([(base_addr, i)])[0]

        def get_bf2_many(specs):
            """[(base, i)] packed bf16 elements i, i + 1 (i even) -> list of (f32, f32)."""
            return [bf2_f32(v[0]) for v in poll([(b, i // 2, 1) for b, i in specs])]

        # ---- wave reductions
        def wave_sum(v):
            for sh in range_constexpr(5):
                v = _xred(v, 16 >> sh, lambda a, b: a + b)
            return v

        def wave_max(v):
            for sh in range_constexpr(5):
                v = _xred(v, 16 >> sh, fx.max)
            return v

        def block_sums(vs):
            """Block-wide sums of several per-thread values with one LDS exchange."""
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

        def block_sum(v):
            w = wave_sum(v)
            if lane == 0:
                lds_st(red, wave, w)
            gpu.barrier()
            t = lds_ld(red, 0)
            for i in range_constexpr(1, WAVES):
                t = t + lds_ld(red, i)
            gpu.barrier()
            return t

        # ------------------------------------------------ WMMA GEMV machinery
        def b16_lds(word):
            """The 16 BF16 of one K=32 WMMA operand from LDS: words [word, word + 4)
            and [word + 8, word + 12) (the lane group offset is part of ``word``)."""
            lo = fx.Vector(fx.ptr_load(xs + word, result_type=v4f))
            hi = fx.Vector(fx.ptr_load(xs + (word + 8), result_type=v4f))
            return fx.Vector.from_elements([lo[i] for i in range(4)] + [hi[i] for i in range(4)], fx.Float32).bitcast(
                fx.BFloat16
            )

        def cat8(lo, hi):
            """Two 8-value BF16 vectors -> one 16-value WMMA operand."""
            lo, hi = lo.bitcast(fx.Int32), hi.bitcast(fx.Int32)
            return fx.Vector.from_elements([lo[i] for i in range(4)] + [hi[i] for i in range(4)], fx.Int32).bitcast(
                fx.BFloat16
            )

        def a16_fp8(w4):
            """16 BF16 WMMA A values from one 16-byte FP8 load (two 8-value blocks)."""
            return cat8(fp8x8_to_bf16_gfx1250(w4[0], w4[1]), fp8x8_to_bf16_gfx1250(w4[2], w4[3]))

        def a16_bf16(w4a, w4b):
            words = [w4a[i] for i in range(4)] + [w4b[i] for i in range(4)]
            return fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16)

        def w_tile(w_rsrc, tile, ln, h):
            """Half ``h`` (16 bytes) of lane ``ln``'s 32-byte slice of 1 KB tile ``tile``."""
            return fx.Vector(bo.buffer_load(w_rsrc, (tile * WAVE + ln) * 8 + h * 4, vec_width=4, dtype=T.i32))

        def unit_fp8(w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word, coef=None):
            """Issue one 64-k chunk (two WMMAs) of row group ``rg`` of a packed FP8 matrix;
            the bf16 activation chunk starts at LDS word ``b_word``."""
            wv = [w_tile(w_rsrc, rg * NKC + kc, lane, h) for h in range(2)]
            s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // BK) + kc * 64 // BK)
            if const_expr(callable(coef)):  # factor known only after a later wait
                return ("fp8", wv, lambda: s * coef(), b_word + (lane // 16) * 4)
            if const_expr(coef is not None):
                s = s * coef
            return ("fp8", wv, s, b_word + (lane // 16) * 4)

        def unit_fp8x2(w_rsrc, s_rsrc, rg, kc, NKC, K, b_word, coef=None, scale_rows=SCALE_BM, ln=None, s_rg=None):
            """Issue both 64-k halves of one 128-k FP8 weight-scale block (four WMMAs).
            ``ln`` = the lane whose weights are loaded; ``s_rg`` = the row group of
            this lane's accumulator rows when it differs from ``rg``."""
            ln = lane if ln is None else ln
            wv = [w_tile(w_rsrc, rg * NKC + kc + c, ln, h) for c in range(2) for h in range(2)]
            srow = rg if s_rg is None else s_rg
            s = ld_f32(s_rsrc, (srow * 16 // scale_rows) * (K // 128) + kc // 2)
            if const_expr(callable(coef)):
                return ("fp8x2", wv, lambda: s * coef(), b_word + (lane // 16) * 4)
            if const_expr(coef is not None):
                s = s * coef
            return ("fp8x2", wv, s, b_word + (lane // 16) * 4)

        def unit_bf16(w_rsrc, rg, kc, NKC, b_word, ln=None):
            ln = lane if ln is None else ln
            wv = [w_tile(w_rsrc, (rg * NKC + kc) * 2 + sp, ln, h) for sp in range(2) for h in range(2)]
            return ("bf16", wv, None, b_word + (lane // 16) * 4)

        def unit_mxfp4(w_rsrc, s_rsrc, rg, kt, K, b_word, coef, ln=None):
            """Issue one native packed 128-K MXFP4 tile and this row's four E8M0 scales."""
            ln = lane if ln is None else ln
            raw = [w_tile(w_rsrc, rg * (K // 128) + kt, ln, h) for h in range(2)]
            row = rg * 16 + ln % 16
            packed = fx.Int32(bo.buffer_load(s_rsrc, row * (K // 128) + kt, vec_width=1, dtype=T.i32))
            shift = (lane // 16) * 16
            lo = packed.shrui(fx.Int32(shift)) & fx.Int32(0xFF)
            hi = packed.shrui(fx.Int32(shift + 8)) & fx.Int32(0xFF)
            return ("mxfp4", (raw, (lo | (hi << 16)) * fx.Int32(0x101)), coef, b_word + (lane // 16) * 4)

        def mma_units(acc, units):
            """acc[8] += coef * (W_chunk @ X_chunk) for every issued unit."""
            for fmt, wv, coef, bw in units:
                if const_expr(callable(coef)):
                    coef = coef()
                c = fx.Vector.filled(ACC, 0.0, fx.Float32)
                if const_expr(fmt == "mxfp4"):
                    raw, sc = wv
                    for sp in range_constexpr(4):
                        w = raw[sp // 2]
                        q = (sp % 2) * 2
                        a = cat8(
                            fp4x8_to_bf16_gfx1250(w[q], sc, MXFP4_SCALE_SEL[sp]),
                            fp4x8_to_bf16_gfx1250(w[q + 1], sc, MXFP4_SCALE_SEL[sp]),
                        )
                        c = wmma_bf16_gfx1250(a, b16_lds(bw + sp * 16), c)
                else:
                    nsp = 4 if const_expr(fmt == "fp8x2") else 2
                    for sp in range_constexpr(nsp):
                        if const_expr(fmt == "bf16"):
                            a = a16_bf16(wv[2 * sp], wv[2 * sp + 1])
                        else:
                            a = a16_fp8(wv[sp])
                        c = wmma_bf16_gfx1250(a, b16_lds(bw + sp * 16), c)
                if const_expr(coef is None):
                    acc = [acc[e] + c[e] for e in range(ACC)]
                else:
                    acc = [acc[e] + c[e] * coef for e in range(ACC)]
            return acc

        def zero_acc():
            return [fx.Float32(0.0) for _ in range(ACC)]

        def run_units(make_unit, cpw, batch, pre=None):
            """Software pipelined: issue batch b+1's loads before computing batch b.
            ``pre`` = the already-issued first batch (prefetched before a wait)."""
            acc = zero_acc()
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

        def c_word(ww, r, n):
            """LDS word of accumulator element (row r, column n) of wave ww's tile in ``red``."""
            return (ww * WAVE + n + 16 * (r // 8)) * ACC + r % 8

        def store_acc(acc):
            fx.ptr_store(fx.Vector.from_elements(acc, fx.Float32), red + (wave * WAVE + lane) * ACC)

        def reduce_rows(R, acc, emit, count=S):
            """Sum per-wave WMMA tiles; emit(row_local, local sample column, value)."""
            wpr = WAVES // R
            store_acc(acc)
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
                        tot = tot + lds_ld(red, c_word((rl // 16) * wpr + j, r, n))
                    emit(rl, n, tot)

        def emit_out(stride):
            def f(rl, n, v):
                lds_st(outs, n * stride + rl, v)

            return f

        def stage_x_rmsnorm(ld4s, n, gamma, mark=None, loaded=None, count=S):
            """LDS bf16 X[s][0:n] = bf16(rmsnorm(x_s) * gamma) for every sample s, where
            ld4s([(s, k)]) -> [(x_s[k], .., x_s[k+3])] (one batched load); returns the rstds.
            ``loaded``: the (gamma, x) loads already issued by load_x_rmsnorm."""
            per = n // (4 * THREADS)
            ks = [(tid + i * THREADS) * 4 for i in range(per)]
            gs, vals = loaded if loaded is not None else load_x_rmsnorm(ld4s, n, gamma, count)
            sss = []
            for s in range_constexpr(count):
                ss = fx.Float32(0.0)
                for i in range_constexpr(per):
                    for a in vals[s * per + i]:
                        ss = ss + a * a
                sss.append(ss)
            if const_expr(mark is not None):
                stamp(mark[0], mark[1], 6)
            rstds = [_rsq(tot * (1.0 / n) + EPS) for tot in block_sums(sss)]
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

        def load_x_rmsnorm(ld4s, n, gamma, count=S):
            """The gamma loads (issued ahead of the wait), then ld4s -> (gammas, x values)."""
            rg_ = _rsrc(gamma)
            ks = [(tid + i * THREADS) * 4 for i in range(n // (4 * THREADS))]
            gs = []
            for k in ks:
                g = fx.Vector(bo.buffer_load(rg_, k // 2, vec_width=2, dtype=T.i32)).bitcast(fx.BFloat16).to(fx.Float32)
                gs.append([g[j] for j in range(4)])
            return gs, ld4s([(s, k) for s in range(count) for k in ks])

        def stage_x_pairs(name, n_total, src_of):
            """LDS bf16 X[k] = packed bf16 mailbox ``name`` element src_of(k) for k < n_total
            (src_of contiguous over aligned groups of 4): one 16-byte poll per 4 elements."""
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

        # ---- FP8 activations: one wave per 128-block, four values per lane
        def quant4(vals):
            """Per-wave FP8 quant of a 128-block -> (clamped scaled values, block scale)."""
            local = fx.max(
                fx.max(fmath.absf(vals[0]), fmath.absf(vals[1])), fx.max(fmath.absf(vals[2]), fmath.absf(vals[3]))
            )
            amax = wave_max(local)
            nz = amax > 0.0
            qs = nz.select(amax * (1.0 / FP8_MAX), fx.Float32(1.0))
            inv = nz.select(_rcp(amax) * FP8_MAX, fx.Float32(1.0))  # hardware rcp, no IEEE divide
            return [fx.min(fx.max(v * inv, -FP8_MAX), FP8_MAX) for v in vals], qs

        def fp8_word4(q):
            """Four clamped f32 -> one dword of E4M3 bytes in element order."""
            lo = rocdl.cvt_pk_fp8_f32(T.i32, q[0], q[1], fx.Int32(0), False)
            return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q[2], q[3], lo, True))

        def fp8_word_bf16(w):
            """One dword of E4M3 bytes -> two exact packed-BF16 words.

            Decoded with v_cvt_scale_pk8 rather than v_cvt_pk_f32_fp8, whose gfx1250
            VOP1 form LLVM prints in a way its own assembler rejects."""
            v = fp8x8_to_bf16_gfx1250(w, fx.Int32(0)).bitcast(fx.Int32)
            return v[0], v[1]

        def st_x4(k, vals):
            """LDS BF16 X[k .. k + 4) := vals (k a multiple of 4)."""
            lds_st(xs, k // 2, bf16_pair(vals[0], vals[1]))
            lds_st(xs, k // 2 + 1, bf16_pair(vals[2], vals[3]))

        def st_fp8_words(k, w):
            """LDS X[k .. k + 4) := the four E4M3 values of ``w`` as exact BF16."""
            lo, hi = fp8_word_bf16(w)
            lds_st(xs, k // 2, lo.bitcast(fx.Float32))
            lds_st(xs, k // 2 + 1, hi.bitcast(fx.Float32))

        def st_fp8x4(k, q):
            """LDS X[k .. k + 4) := the E4M3 roundings of q, as exact BF16 values."""
            st_fp8_words(k, fp8_word4(q))

        def stage_xq(samples):
            """Poll the router's packed FP8 activation + block scales of ``samples`` into
            LDS slot j = X[j * HIDDEN:] (exact BF16 values) and misc[8 + j * XQ_BLOCKS:]."""
            nxw = HIDDEN // 4 // THREADS
            got = poll(
                [(mb("xq"), sx * (HIDDEN // 4) + tid + i * THREADS, 1) for sx in samples for i in range(nxw)]
                + [(mb("xqs"), sx * XQ_BLOCKS + fx.min(tid, XQ_BLOCKS - 1), 1) for sx in samples]
            )
            for j in range_constexpr(len(samples)):
                for i in range_constexpr(nxw):
                    st_fp8_words(j * HIDDEN + (tid + i * THREADS) * 4, got[j * nxw + i][0])
                if tid < XQ_BLOCKS:
                    lds_st(misc, 8 + j * XQ_BLOCKS + tid, got[len(samples) * nxw + j][0].bitcast(fx.Float32))

        def load_bias():
            """This lane's 8 expert biases (issue before the scores wait)."""
            return [ld_f32(_rsrc(bias), lane + i * WAVE) for i in range(ROUTE_CANDIDATES)]

        def route_top8(s, raws=None, bs=None):
            """Top-8 of sample s (call from one whole wave, after the router scores landed).

            Preserve all FP32 bits of (sigmoid + bias). Each round reduces the score
            first, then the expert ID only among exactly equal winners.  Candidate i of
            this lane is expert lane + 32 i.  Returns (expert id, route weight = raw
            score / sum of the 8 raw scores * ROUTE_SCALE) of pick ``lane`` in score
            order, valid in lanes < TOP_K."""
            if const_expr(bs is None):
                bs = load_bias()
            if const_expr(raws is None):
                raws = getf_many([(mb("scores"), s * N_EXPERTS + lane + i * WAVE) for i in range(ROUTE_CANDIDATES)])
                stamp("ug", bid, 7)
            ks = []
            for i in range_constexpr(ROUTE_CANDIDATES):
                kb = (raws[i] + bs[i]).bitcast(fx.Int32)
                ok = (kb >= 0).select(kb ^ fx.Int32(-(2**31)), ~kb)
                ks.append(fx.Uint32(ok))
            ids = [lane + i * WAVE for i in range(ROUTE_CANDIDATES)]
            # sort this lane's keys descending; each round then takes the wave max of
            # the lane heads and shifts the winning lane's list (0 is below every key)
            for a, b in SORT8:
                first = (ks[a] > ks[b]) | ((ks[a] == ks[b]) & (ids[a] < ids[b]))
                ka, kb, ia, ib = ks[a], ks[b], ids[a], ids[b]
                ks[a], ks[b] = first.select(ka, kb), first.select(kb, ka)
                ids[a], ids[b] = first.select(ia, ib), first.select(ib, ia)
            ks = [fx.Int32(k) for k in ks] + [fx.Int32(0)]
            ids = ids + [fx.Int32(N_EXPERTS)]
            mv = fx.Int32(0)  # lane k: the expert ID of pick k
            for k in range_constexpr(TOP_K):
                m = wave32_umax(ks[0])
                winner = wave32_umax((ks[0] == m).select(255 - ids[0], fx.Int32(0)))
                expert = 255 - winner
                hit = (ks[0] == m) & (ids[0] == expert)
                ks = [hit.select(ks[i + 1], ks[i]) for i in range(ROUTE_CANDIDATES)] + [ks[ROUTE_CANDIDATES]]
                ids = [hit.select(ids[i + 1], ids[i]) for i in range(ROUTE_CANDIDATES)] + [ids[ROUTE_CANDIDATES]]
                mv = write_lane_i32(expert, k, mv)
            e = mv
            src = (e % WAVE) * 4
            got = [bpermute_i32(src, r.bitcast(fx.Int32)) for r in raws]
            raw = got[0]
            for i in range_constexpr(1, ROUTE_CANDIDATES):
                raw = (e // WAVE == i).select(got[i], raw)
            raw = (lane < TOP_K).select(raw.bitcast(fx.Float32), fx.Float32(0.0))
            tot = raw
            for off in (1, 2, 4):
                tot = _xred(tot, off, lambda a, b: a + b)
            return e, raw * (_rcp(tot) * ROUTE_SCALE)

        def peer_reduce(region, t, residual, out_fn, tile=ROW_TILE):
            """Push BF16 partials to every peer, then sum them in rank order.

            One wave owns each destination, allowing the peer stores to progress
            concurrently without retaining all peer pointers in every wave."""
            region_base = fx.Int64(SY[region]) + fx.Int64(peer_slot) * fx.Int64(SY["_part_stride"])
            if const_expr(W > 1):
                if wave < W:
                    pair_count = S * tile // 2
                    for batch in range_constexpr((pair_count + WAVE - 1) // WAVE):
                        pair = lane + batch * WAVE
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
                if const_expr(callable(residual)):
                    r0, r1 = residual(s, row)
                v0 = lds_ld(outs, s * tile + r)
                v1 = lds_ld(outs, s * tile + r + 1)
                if const_expr(W == 1):  # no TP peers: the sum is the local value
                    parts = [(v0, v1)]
                    got = []
                    if const_expr(not callable(residual)):
                        got = poll([(residual, (s * HIDDEN + row) // 2, 1)])
                else:
                    own = sym + region_base
                    specs = [(own, ((src * S + s) * HIDDEN + row) // 2, 1) for src in range(W)]
                    if const_expr(not callable(residual)):  # packed bf16 pair
                        specs.append((residual, (s * HIDDEN + row) // 2, 1))
                    got = poll(specs, "one-as")
                    parts = [bf2_f32(v[0]) for v in got[:W]]
                    got = got[W:]
                if const_expr(not callable(residual)):
                    r0, r1 = bf2_f32(got[0][0])
                t0 = fx.Float32(0.0)
                t1 = fx.Float32(0.0)
                for src in range_constexpr(W):
                    t0 = t0 + parts[src][0]
                    t1 = t1 + parts[src][1]
                out_fn(s, row, r0 + t0, r1 + t1)

        def start(name):
            return (bid + (G - base[name])) & (G - 1)

        def stamp(name, t, which, lead=0):
            if const_expr(timeline):
                if tid == lead:
                    now = steady_counter()
                    tl_addr = index_arg(7) if const_expr(with_indexer) else timeline_buf
                    fx.generic_store(
                        fx.inttoptr(
                            fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8),
                            tl_addr + fx.Int64((first[name] + t) * TL_COLS + which) * 8,
                        ),
                        now,
                    )

        def n_sel(count=S):
            """This lane's local WMMA B column; inactive columns duplicate the last one."""
            return fx.min(lane % 16, count - 1)

        # ================================================= 1. q_a / kv_a GEMV
        # 1 row group x 96 chunks: 16 waves split K, 6 chunks each (all prefetched)
        r_wqa, r_sqa = _rsrc(w_qkv_a), _rsrc(s_qkv_a)
        QA_NKC = HIDDEN // 64
        QA_UNITS = QA_NKC // (2 * WAVES)
        if const_expr(with_indexer):
            r_wik, r_sik = _rsrc(index_arg(0)), _rsrc(index_arg(1))
            r_wiw = _rsrc(index_arg(2))
        for t in range(start("qkv_a"), N_QKV_A, G):
            t = fx.Int32(t)
            stamp("qkv_a", t, 0)
            for sample_base in range_constexpr(0, S, SAMPLE_TILE):
                group_count = min(SAMPLE_TILE, S - sample_base)

                def u_qa(c):
                    kc = (wave * QA_UNITS + c) * 2
                    return unit_fp8x2(
                        r_wqa,
                        r_sqa,
                        t,
                        kc,
                        QA_NKC,
                        HIDDEN,
                        (n_sel(group_count) * HIDDEN + kc * 64) // 2,
                    )

                def ld_h(sks):
                    res = []
                    for s, k in sks:
                        w = fx.Vector(
                            bo.buffer_load(
                                r_h,
                                ((sample_base + s) * HIDDEN + k) // 2,
                                vec_width=2,
                                dtype=T.i32,
                            )
                        )
                        v = w.bitcast(fx.BFloat16).to(fx.Float32)
                        res.append([v[j] for j in range(4)])
                    return res

                # Up to four samples, the first weight units are issued ahead of the
                # input wait; with eight samples' inputs live they spill.
                if const_expr(group_count <= 4):
                    h_ld = load_x_rmsnorm(ld_h, HIDDEN, g_in, group_count)
                    pre = [u_qa(c) for c in range(QA_UNITS)]
                    stage_x_rmsnorm(ld_h, HIDDEN, g_in, loaded=h_ld, count=group_count)
                else:
                    stage_x_rmsnorm(ld_h, HIDDEN, g_in, count=group_count)
                    pre = [u_qa(c) for c in range(QA_UNITS)]
                gpu.barrier()
                stamp("qkv_a", t, 2)
                acc = run_units(u_qa, QA_UNITS, QA_UNITS, pre)
                reduce_rows(1, acc, emit_out(QKV_A_TILE), group_count)
                stamp("qkv_a", t, 3)
                gpu.barrier()
                if tid < group_count * QKV_A_TILE:
                    s = sample_base + tid // QKV_A_TILE
                    row = t * QKV_A_TILE + tid % QKV_A_TILE
                    v = lds_ld(outs, tid)
                    if row < Q_LORA:
                        put(mb("q_a"), s * Q_LORA + row, v)
                    else:
                        put(mb("kv_a"), s * (KV_LORA + PE_DIM) + row - Q_LORA, v)

                if const_expr(with_indexer):
                    IW_CPW = QA_NKC // WAVES

                    def u_index_w(c):
                        kc = wave * IW_CPW + c
                        return unit_bf16(
                            r_wiw,
                            t,
                            kc,
                            QA_NKC,
                            (n_sel(group_count) * HIDDEN + kc * 64) // 2,
                        )

                    if t < INDEX_DIM // QKV_A_TILE:

                        def u_index_k(c):
                            kc = (wave * QA_UNITS + c) * 2
                            return unit_fp8x2(
                                r_wik,
                                r_sik,
                                t,
                                kc,
                                QA_NKC,
                                HIDDEN,
                                (n_sel(group_count) * HIDDEN + kc * 64) // 2,
                            )

                        ik_acc = run_units(u_index_k, QA_UNITS, QA_UNITS)
                        iw_acc = zero_acc()
                        if t < INDEX_HEADS // QKV_A_TILE:
                            iw_acc = run_units(u_index_w, IW_CPW, IW_CPW)
                        reduce_rows(1, ik_acc, emit_out(INDEX_TILE), group_count)
                        gpu.barrier()
                        if tid < group_count * INDEX_TILE:
                            s = sample_base + tid // INDEX_TILE
                            row = t * INDEX_TILE + tid % INDEX_TILE
                            put(mb("index_k"), s * INDEX_DIM + row, lds_ld(outs, tid))
                        if t < INDEX_HEADS // QKV_A_TILE:
                            reduce_rows(1, iw_acc, emit_out(INDEX_TILE), group_count)
                            gpu.barrier()
                            if tid < group_count * INDEX_TILE:
                                s = sample_base + tid // INDEX_TILE
                                row = t * INDEX_TILE + tid % INDEX_TILE
                                put(mb("index_w"), s * INDEX_HEADS + row, lds_ld(outs, tid))
            stamp("qkv_a", t, 4)

        def ld_qa(sks):
            v = get2_many([(mb("q_a"), s * Q_LORA + k + j) for s, k in sks for j in (0, 2)])
            return [list(v[2 * i]) + list(v[2 * i + 1]) for i in range(len(sks))]

        # ===================== 2. one q_a RMSNorm CTA per sample, shared downstream
        enter("q_norm")
        for s_norm in range(start("q_norm"), S, G):
            s_norm = fx.Int32(s_norm)
            stamp("q_norm", s_norm, 0)
            hint_wait(
                Q_LORA // QKV_A_TILE,
                lambda k: (mb("q_a"), s_norm * Q_LORA + k * QKV_A_TILE + QKV_A_TILE - 1),
                mark=("q_norm", s_norm),
            )

            def ld_qa_one(sks):
                return ld_qa([(s_norm, k) for _, k in sks])

            stage_x_rmsnorm(ld_qa_one, Q_LORA, g_q, count=1)
            stamp("q_norm", s_norm, 2)
            gpu.barrier()
            k = tid * 4
            w0 = lds_ld(xs, k // 2)
            w1 = lds_ld(xs, k // 2 + 1)
            a0, a1 = bf2_f32(w0.bitcast(fx.Int32))
            a2, a3 = bf2_f32(w1.bitcast(fx.Int32))
            put_bf(mb("q_an"), s_norm * Q_LORA + k, [a0, a1, a2, a3])
            stamp("q_norm", s_norm, 4)

        # ================ 3. KV RMSNorm + k_pe RoPE -> cache (+ this launch's rows)
        enter("cache")
        for t in range(start("cache"), 1, G):
            stamp("cache", t, 0)
            r_kv = _rsrc(kv_cache)
            r_pe = _rsrc(pe_cache)
            # gamma and the RoPE factors are issued ahead of the wait
            g = ld_bf16(_rsrc(g_kv), tid)
            tpe = tid % (PE_DIM // 2)
            cs = [ld_f32(_rsrc(rope_cos), (pos0 + s) * (PE_DIM // 2) + tpe) for s in range(S)]
            sns = [ld_f32(_rsrc(rope_sin), (pos0 + s) * (PE_DIM // 2) + tpe) for s in range(S)]
            hint_wait(
                (KV_LORA + PE_DIM) // QKV_A_TILE,
                lambda k: (mb("kv_a"), (S - 1) * (KV_LORA + PE_DIM) + k * QKV_A_TILE + QKV_A_TILE - 1),
                mark=("cache", t),
            )
            # every sample's kv latent and k_pe pair in one poll, one block reduction
            vs = getf_many([(mb("kv_a"), s * (KV_LORA + PE_DIM) + tid) for s in range(S)])
            pes = get2_many(
                [(mb("kv_a"), s * (KV_LORA + PE_DIM) + KV_LORA + (tid % (PE_DIM // 2)) * 2) for s in range(S)]
            )
            stamp("cache", t, 2)
            ssq = block_sums([v * v for v in vs])
            for s in range_constexpr(S):
                pos = pos0 + s
                kvn = bf16_round(vs[s] * _rsq(ssq[s] * (1.0 / KV_LORA) + EPS) * g)
                bo.buffer_store(kvn.to(fx.BFloat16), r_kv, pos * KV_LORA + tid)
                put(mb("kvnew"), s * KV_LORA + tid, kvn)
                if tid < PE_DIM // 2:
                    x0, x1 = pes[s]
                    c, sn = cs[s], sns[s]
                    p0 = bf16_round(x0 * c - x1 * sn)
                    p1 = bf16_round(x0 * sn + x1 * c)
                    bo.buffer_store(p0.to(fx.BFloat16), r_pe, pos * PE_DIM + tid * 2)
                    bo.buffer_store(p1.to(fx.BFloat16), r_pe, pos * PE_DIM + tid * 2 + 1)
                    put2(mb("penew"), s * PE_DIM + tid * 2, p0, p1)

            if const_expr(with_indexer):
                # Index keys use LayerNorm (not RMSNorm), interleaved RoPE on
                # the first 64 dimensions, and BF16 cache storage. Hadamard is
                # omitted because applying the same orthogonal transform to Q
                # and K leaves their dot products unchanged.
                r_gik, r_bik = _rsrc(index_arg(5)), _rsrc(index_arg(6))
                r_index_cache = _rsrc(indices)
                # every sample's index key in one poll and two block reductions
                live = tid < INDEX_DIM
                iks = getf_many([(mb("index_k"), s * INDEX_DIM + fx.min(tid, INDEX_DIM - 1)) for s in range(S)])
                means = [m * (1.0 / INDEX_DIM) for m in block_sums([live.select(ik, fx.Float32(0.0)) for ik in iks])]
                centered = [live.select(iks[s] - means[s], fx.Float32(0.0)) for s in range(S)]
                rstds = [_rsq(v * (1.0 / INDEX_DIM) + 1.0e-6) for v in block_sums([c * c for c in centered])]
                if tid < INDEX_DIM // 2:
                    i0 = tid * 2
                    kks = get2_many([(mb("index_k"), s * INDEX_DIM + i0) for s in range(S)])
                    g0, g1 = ld_f32(r_gik, i0), ld_f32(r_gik, i0 + 1)
                    b0, b1 = ld_f32(r_bik, i0), ld_f32(r_bik, i0 + 1)
                    for s in range_constexpr(S):
                        k0, k1 = kks[s]
                        v0 = (k0 - means[s]) * rstds[s] * g0 + b0
                        v1 = (k1 - means[s]) * rstds[s] * g1 + b1
                        if tid < PE_DIM // 2:
                            c, sn = cs[s], sns[s]
                            v0, v1 = v0 * c - v1 * sn, v0 * sn + v1 * c
                        bo.buffer_store(
                            fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16),
                            r_index_cache,
                            (pos0 + s) * INDEX_DIM + i0,
                        )
                        put_bf(mb("index_k_new"), s * INDEX_DIM + i0, [v0, v1])
                gpu.barrier()  # the next stage's ``red`` writes wait for these block reductions
            stamp("cache", t, 4)

        # =============================================== 4. normalized q_a -> q_b (+RoPE)
        r_wqb, r_sqb = _rsrc(w_q_b), _rsrc(s_q_b)
        QB_NKC = Q_LORA // 64
        QB_UNITS = QB_NKC // (2 * WAVES)
        if const_expr(with_indexer):
            r_wiq, r_siq = _rsrc(index_arg(3)), _rsrc(index_arg(4))

        enter("q_b")
        for t in range(start("q_b"), N_QB, G):
            t = fx.Int32(t)
            stamp("q_b", t, 0)

            def u_qb(c):
                kc = (wave * QB_UNITS + c) * 2
                return unit_fp8x2(r_wqb, r_sqb, t, kc, QB_NKC, Q_LORA, (n_sel() * Q_LORA + kc * 64) // 2)

            pre = [u_qb(c) for c in range(QB_UNITS)]
            hint_wait(S, lambda s: (mb("q_an"), (s * Q_LORA + Q_LORA - 2) // 2), mark=("q_b", t))
            stage_x_pairs("q_an", S * Q_LORA, lambda k: k)
            stamp("q_b", t, 2)
            gpu.barrier()
            acc = run_units(u_qb, QB_UNITS, QB_UNITS, pre)
            reduce_rows(1, acc, emit_out(Q_B_TILE))
            stamp("q_b", t, 3)
            gpu.barrier()
            head = t // QB_PER_HEAD
            hoff = (t % QB_PER_HEAD) * Q_B_TILE
            if hoff < NOPE_DIM:
                if tid < S * Q_B_TILE // 4:
                    s = tid // (Q_B_TILE // 4)
                    r = (tid % (Q_B_TILE // 4)) * 4
                    put_bf(
                        mb("q_nope"),
                        (s * H + head) * NOPE_DIM + hoff + r,
                        [lds_ld(outs, s * Q_B_TILE + r + j) for j in range(4)],
                    )
            else:
                if tid < S * Q_B_TILE // 2:
                    s = tid // (Q_B_TILE // 2)
                    pr = tid % (Q_B_TILE // 2)
                    i = hoff - NOPE_DIM + pr * 2
                    x0 = lds_ld(outs, s * Q_B_TILE + pr * 2)
                    x1 = lds_ld(outs, s * Q_B_TILE + pr * 2 + 1)
                    c = ld_f32(_rsrc(rope_cos), (pos0 + s) * (PE_DIM // 2) + i // 2)
                    sn = ld_f32(_rsrc(rope_sin), (pos0 + s) * (PE_DIM // 2) + i // 2)
                    put_bf(mb("q_pe"), (s * H + head) * PE_DIM + i, [x0 * c - x1 * sn, x0 * sn + x1 * c])
            stamp("q_b", t, 4)

            if const_expr(with_indexer):
                # Reuse this CTA's normalized q_lora tile for one index-query
                # row tile.  Complementary CTAs compute the remaining half below.
                iq_t = t
                stamp("index_q", iq_t, 0)

                def u_index_q(c):
                    kc = (wave * QB_UNITS + c) * 2
                    return unit_fp8x2(
                        r_wiq,
                        r_siq,
                        iq_t,
                        kc,
                        QB_NKC,
                        Q_LORA,
                        (n_sel() * Q_LORA + kc * 64) // 2,
                    )

                iq_acc = run_units(u_index_q, QB_UNITS, QB_UNITS)
                reduce_rows(1, iq_acc, emit_out(INDEX_TILE))
                stamp("index_q", iq_t, 3)
                gpu.barrier()
                if tid < S * INDEX_TILE // 4:
                    s = tid // (INDEX_TILE // 4)
                    r = (tid % (INDEX_TILE // 4)) * 4
                    put_bf(
                        mb("index_q"),
                        s * INDEX_Q_ROWS + iq_t * INDEX_TILE + r,
                        [lds_ld(outs, s * INDEX_TILE + r + j) for j in range(4)],
                    )
                stamp("index_q", iq_t, 4)

        if const_expr(with_indexer):
            # The 128 CTAs without q_b work produce the other 128 index-query
            # tiles concurrently.  They reload q_a, but remove one full GEMV
            # from the q_b CTAs' serialized critical path.
            N_INDEX_Q_EXTRA = INDEX_Q_ROWS // INDEX_TILE - N_QB
            enter("index_q")
            for tt in range(start("index_q"), N_INDEX_Q_EXTRA, G):
                tt = fx.Int32(tt)
                iq_t = N_QB + tt
                stamp("index_q", iq_t, 0)

                def u_index_q_extra(c):
                    kc = (wave * QB_UNITS + c) * 2
                    return unit_fp8x2(
                        r_wiq,
                        r_siq,
                        iq_t,
                        kc,
                        QB_NKC,
                        Q_LORA,
                        (n_sel() * Q_LORA + kc * 64) // 2,
                    )

                pre = [u_index_q_extra(c) for c in range(QB_UNITS)]
                hint_wait(S, lambda s: (mb("q_an"), (s * Q_LORA + Q_LORA - 2) // 2), mark=("index_q", iq_t))
                stage_x_pairs("q_an", S * Q_LORA, lambda k: k)
                stamp("index_q", iq_t, 2)
                gpu.barrier()
                iq_acc = run_units(u_index_q_extra, QB_UNITS, QB_UNITS, pre)
                reduce_rows(1, iq_acc, emit_out(INDEX_TILE))
                stamp("index_q", iq_t, 3)
                gpu.barrier()
                if tid < S * INDEX_TILE // 4:
                    s = tid // (INDEX_TILE // 4)
                    r = (tid % (INDEX_TILE // 4)) * 4
                    put_bf(
                        mb("index_q"),
                        s * INDEX_Q_ROWS + iq_t * INDEX_TILE + r,
                        [lds_ld(outs, s * INDEX_TILE + r + j) for j in range(4)],
                    )
                stamp("index_q", iq_t, 4)

        # ==================================== 4. absorbed query: q_lat = W_UK^T q_nope
        # 8 row groups (128 latent rows of one head) x 3 chunks: one row group per
        # working wave; waves 8-15 repeat waves 0-7
        r_wuk, r_suk = _rsrc(w_uk), _rsrc(s_uk)
        UK_NKC = NOPE_DIM // 64
        enter("uk")
        for t in range(start("uk"), N_UK, G):
            t = fx.Int32(t)
            stamp("uk", t, 0)
            head = t // UK_PER_HEAD

            def u_uk(c):
                return unit_fp8(
                    r_wuk,
                    r_suk,
                    t * WORK_WAVES + wave % WORK_WAVES,
                    c,
                    UK_NKC,
                    NOPE_DIM,
                    64,
                    (n_sel() * NOPE_DIM + c * 64) // 2,
                )

            pre = [u_uk(c) for c in range(UK_NKC)]
            hint_wait(
                NOPE_DIM // Q_B_TILE,
                lambda k: (mb("q_nope"), ((S - 1) * H + head) * NOPE_DIM + k * Q_B_TILE + Q_B_TILE - 1),
                mark=("uk", t),
            )
            stage_x_pairs("q_nope", S * NOPE_DIM, lambda k: ((k // NOPE_DIM) * H + head) * NOPE_DIM + k % NOPE_DIM)
            stamp("uk", t, 2)
            gpu.barrier()
            acc = run_units(u_uk, UK_NKC, UK_NKC, pre)
            store_acc(acc)
            stamp("uk", t, 3)
            gpu.barrier()
            if tid < S * UK_TILE // 4:
                k = tid * 4
                s = k // UK_TILE
                r0 = k % UK_TILE
                vals = [lds_ld(red, c_word((r0 + j) // 16, (r0 + j) % 16, s)) for j in range(4)]
                put_bf(
                    mb("q_lat"),
                    (s * H + head) * KV_LORA + (t % UK_PER_HEAD) * UK_TILE + r0,
                    vals,
                )
            stamp("uk", t, 4)

        # ====================== 4b. fused sparse index score + exact top-2048
        if const_expr(with_indexer):
            r_index_cache = _rsrc(indices)
            N_INDEX_SPLIT = index_max_seq // INDEX_KEYS_PER_TASK

            def load_index_q16(head, k):
                """Index-query WMMA A operand: elements k .. k + 8 and k + 16 .. k + 24."""
                return b16_lds((head * INDEX_DIM + k) // 2)

            enter("index_score")
            for tt in range(start("index_score"), S * N_INDEX_SPLIT, G):
                tt = fx.Int32(tt)
                s = tt // N_INDEX_SPLIT
                tile = tt % N_INDEX_SPLIT
                stamp("index_score", tt, 0)
                bound = pos0 + s + 1
                if tile * INDEX_KEYS_PER_TASK < bound:
                    # 4 key groups x 2 head groups of 16 on the working waves
                    key_group = (wave % WORK_WAVES) // 2
                    head_group = wave % 2
                    key_pos = tile * INDEX_KEYS_PER_TASK + key_group * 16 + lane % 16
                    safe_key = fx.min(key_pos, bound - 1)
                    head = head_group * 16 + lane % 16
                    # Cache rows before pos0 were written by earlier launches, so their
                    # loads are issued ahead of this launch's waits; the rows of this
                    # launch come from the tagged index_k_new pairs below.
                    cached_keys = []
                    for k32 in range_constexpr(INDEX_DIM // 32):
                        k = k32 * 32 + (lane // 16) * 8
                        halves = [
                            fx.Vector(
                                bo.buffer_load(
                                    r_index_cache, (safe_key * INDEX_DIM + k + hk) // 2, vec_width=4, dtype=T.i32
                                )
                            ).bitcast(fx.BFloat16)
                            for hk in (0, 16)
                        ]
                        cached_keys.append(cat8(halves[0], halves[1]))
                    if tid < INDEX_HEADS:
                        lds_st(keys, tid, get(mb("index_w"), s * INDEX_HEADS + tid))
                    # Every key group reuses the same 32x128 query.  Stage and RoPE
                    # it once per scoring CTA instead of polling and rotating it in
                    # each of the four key-group wave pairs.
                    Q_REPS = (INDEX_Q_ROWS // 2) // THREADS
                    q_all = get_bf2_many(
                        [(mb("index_q"), s * INDEX_Q_ROWS + (tid + b * THREADS) * 2) for b in range(Q_REPS)]
                    )
                    for b in range_constexpr(Q_REPS):
                        q_pair = tid + b * THREADS
                        q_elem = q_pair * 2
                        kq = q_elem % INDEX_DIM
                        q0, q1 = q_all[b]
                        if kq < PE_DIM:
                            c = ld_f32(_rsrc(rope_cos), (pos0 + s) * (PE_DIM // 2) + kq // 2)
                            sn = ld_f32(_rsrc(rope_sin), (pos0 + s) * (PE_DIM // 2) + kq // 2)
                            q0, q1 = q0 * c - q1 * sn, q0 * sn + q1 * c
                        lds_st(xs, q_pair, bf16_pair(q0, q1))
                    gpu.barrier()
                    score_frag = fx.Vector.filled(ACC, 0.0, fx.Float32)
                    for k32 in range_constexpr(INDEX_DIM // 32):
                        k = k32 * 32 + (lane // 16) * 8
                        qv = load_index_q16(head, k)
                        kv = cached_keys[k32]
                        if safe_key >= pos0:
                            sn = safe_key - pos0
                            kv_pairs = get_bf2_many(
                                [
                                    (mb("index_k_new"), sn * INDEX_DIM + k + blk * 16 + j * 2)
                                    for blk in range(2)
                                    for j in range(4)
                                ]
                            )
                            kv_values = []
                            for pair in kv_pairs:
                                kv_values += list(pair)
                            kv = fx.Vector.from_elements(kv_values, fx.Float32).to(fx.BFloat16)
                        score_frag = wmma_bf16_gfx1250(qv, kv, score_frag)
                    partial = fx.Float32(0.0)
                    for e in range_constexpr(ACC):
                        index_head = head_group * 16 + (lane // 16) * 8 + e
                        weight = lds_ld(keys, index_head).bitcast(fx.Float32)
                        partial = partial + fx.max(score_frag[e], fx.Float32(0.0)) * weight
                    partial = _xred(partial, 16, lambda a, b: a + b)
                    if lane < 16:
                        lds_st(red, wave * 16 + lane, partial)
                    gpu.barrier()
                    if (wave < WORK_WAVES) & (wave % 2 == 0) & (lane < 16) & (key_pos < bound):
                        score = lds_ld(red, wave * 16 + lane) + lds_ld(red, (wave + 1) * 16 + lane)
                        put(mb("index_scores"), s * index_max_seq + key_pos, score)
                stamp("index_score", tt, 4)

            enter("index_select")
            for s in range(start("index_select"), S, G):
                s = fx.Int32(s)
                stamp("index_select", s, 0)
                bound = pos0 + s + 1

                def select_digit(shift, prefix, remain):
                    """Wave 0 finds the digit holding the remain-th largest key, publishes
                    it to keys[SEL_OUT:SEL_OUT + 2] and clears the histogram for the next
                    pass: one wave scan and one barrier, no cross-wave totals."""
                    if wave == 0:
                        top = 255 - lane * 8  # this lane's eight digits, in descending order
                        counts = [lds_ld(keys, top - j) for j in range(8)]
                        total = counts[0]
                        for j in range_constexpr(1, 8):
                            total = total + counts[j]
                        above = fx.coop.warp_inclusive_scan(total, fx.ReductionOp.ADD, width=WAVE) - total
                        for j in range_constexpr(8):
                            if (above < remain) & (above + counts[j] >= remain):
                                lds_st(keys, SEL_OUT, fx.Int32(prefix | (fx.Uint32(top - j) << shift)))
                                lds_st(keys, SEL_OUT + 1, remain - above)
                            above = above + counts[j]
                            lds_st(keys, top - j, fx.Int32(0))
                    gpu.barrier()
                    return fx.Uint32(lds_ld(keys, SEL_OUT)), lds_ld(keys, SEL_OUT + 1)

                # Clearing the histogram before the wait hides its barrier behind it.
                if tid < 256:
                    lds_st(keys, tid, fx.Int32(0))
                gpu.barrier()
                SEL_REPS = (index_max_seq + THREADS - 1) // THREADS
                # One batched poll for this thread's scores; indices past the bound
                # re-read the last valid score and are masked below.
                scores_all = getf_many(
                    [
                        (mb("index_scores"), s * index_max_seq + fx.min(tid + b * THREADS, bound - 1))
                        for b in range(SEL_REPS)
                    ]
                )
                # Monotonic integer keys stay in registers for every radix pass.
                sel_keys, sel_live = [], []
                for b in range_constexpr(SEL_REPS):
                    i = tid + b * THREADS
                    bits = scores_all[b].bitcast(fx.Int32)
                    key = (bits >= 0).select(bits ^ fx.Int32(-(2**31)), ~bits)
                    sel_keys.append(fx.Uint32(key))
                    sel_live.append(i < bound)
                    if i < bound:
                        # The selector CTA no longer needs the large GEMV staging
                        # region, so reuse it for the 4-K radix keys instead of
                        # increasing the monokernel's LDS allocation.
                        lds_st(xs, i, key.bitcast(fx.Float32))
                need = fx.min(fx.Int32(topk), bound)
                prefix, remain = fx.Uint32(0), need
                prefix_mask = 0
                for shift in (24, 16, 8, 0):
                    if const_expr(shift == 24):
                        # Most scores share their high byte (sign and exponent), and
                        # same-address LDS atomics serialize per lane: each thread adds
                        # every distinct high byte of its keys once, with its count.
                        digits = [
                            sel_live[b].select(fx.Int32(sel_keys[b] >> 24), fx.Int32(256)) for b in range(SEL_REPS)
                        ]
                        for j in range_constexpr(SEL_REPS):
                            new_digit = digits[j] < 256
                            for i in range_constexpr(j):
                                new_digit = new_digit & (digits[i] != digits[j])
                            count = fx.Int32(1)
                            for i in range_constexpr(j + 1, SEL_REPS):
                                count = count + (digits[i] == digits[j]).select(fx.Int32(1), fx.Int32(0))
                            if new_digit:
                                fx.atomic_add(keys + digits[j], count, syncscope="workgroup")
                    else:
                        for b in range_constexpr(SEL_REPS):
                            key = sel_keys[b]
                            if sel_live[b] & ((key & fx.Uint32(prefix_mask)) == prefix):
                                digit = fx.Int32((key >> shift) & fx.Uint32(255))
                                fx.atomic_add(keys + digit, fx.Int32(1), syncscope="workgroup")
                    gpu.barrier()
                    prefix, remain = select_digit(shift, prefix, remain)
                    prefix_mask |= 255 << shift

                # ``prefix`` is now the key of the need-th largest score and ``remain``
                # the number of keys equal to it that are selected.
                threshold = prefix
                out_gt = need - remain
                items = index_max_seq // THREADS
                item_indices = [tid * items + j for j in range_constexpr(items)]
                item_keys = []
                for h in range_constexpr(items // 4):
                    quad = fx.Vector(fx.ptr_load(xs + (tid * items + h * 4), result_type=v4f))
                    item_keys += [fx.Uint32(quad[q].bitcast(fx.Int32)) for q in range(4)]
                gt = [(i < bound) & (key > threshold) for i, key in zip(item_indices, item_keys)]
                eq = [(i < bound) & (key == threshold) for i, key in zip(item_indices, item_keys)]

                # Thread-major exclusive offsets of both flag sets in one scan:
                # greater-than counts in the low 16 bits, equal counts in the high
                # 16 bits.  The wave totals use keys[256:256 + WAVES].
                local = fx.Int32(0)
                local_offsets = []
                for g, e in zip(gt, eq):
                    local_offsets.append(local)
                    local = local + g.select(fx.Int32(1), fx.Int32(0)) + e.select(fx.Int32(1 << 16), fx.Int32(0))
                inclusive = fx.coop.warp_inclusive_scan(local, fx.ReductionOp.ADD, width=WAVE)
                if lane == WAVE - 1:
                    lds_st(keys, 256 + wave, inclusive)
                gpu.barrier()
                before_wave = fx.Int32(0)
                for w in range_constexpr(WAVES):
                    before_wave = before_wave + (wave > w).select(lds_ld(keys, 256 + w), fx.Int32(0))
                thread_base = before_wave + inclusive - local
                # Every one of the topk tagged slots is written each launch, so the
                # attention CTAs poll their indices directly: no release fence and
                # no separate readiness tag on the critical path.
                for j in range_constexpr(items):
                    offset = thread_base + local_offsets[j]
                    if gt[j]:
                        put(mb("indices"), s * topk + (offset & fx.Int32(0xFFFF)), fx.Int32(item_indices[j]))
                    eq_offset = offset.shrui(fx.Int32(16))
                    if eq[j] & (eq_offset < remain):
                        put(mb("indices"), s * topk + out_gt + eq_offset, fx.Int32(item_indices[j]))
                for batch in range_constexpr((topk + THREADS - 1) // THREADS):
                    j = tid + batch * THREADS
                    if j >= bound:
                        put(mb("indices"), s * topk + j, fx.Int32(0))
                stamp("index_select", s, 4)

        # ================================== 5. sparse MLA split: 64 (S>4: 32) keys x 8 heads
        r_kv = _rsrc(kv_cache)
        r_pe = _rsrc(pe_cache)
        r_idx = _rsrc(indices)
        KPW = SPLIT_KEYS // WAVES

        def split_keys(t, s):
            """(nkeys, sparse) of sample s; wave 0 writes this split's cache rows to LDS."""
            kv_len = pos0 + s + 1
            sparse = kv_len > topk
            nkeys = sparse.select(fx.Int32(topk), kv_len)
            if wave == 0:
                slots = [lane + rep * WAVE for rep in range(KEY_REPS)]
                k_cls = []
                for j in slots:
                    k_pos = t * SPLIT_KEYS + j
                    k_cls.append((k_pos < nkeys).select(k_pos, 0))
                if const_expr(with_indexer):
                    if sparse:
                        tagged = poll([(mb("indices"), s * topk + k_cl, 1) for k_cl in k_cls])
                        for j, idx in zip(slots, tagged):
                            lds_st(attn_keys, j, idx[0])
                    else:
                        for j, k_cl in zip(slots, k_cls):
                            lds_st(attn_keys, j, k_cl)
                else:
                    for j, k_cl in zip(slots, k_cls):
                        idx = fx.Int32(bo.buffer_load(r_idx, s * topk + k_cl, vec_width=1, dtype=T.i32))
                        lds_st(attn_keys, j, sparse.select(idx, k_cl))
            return nkeys, sparse

        def gather_old_kv():
            """Each wave copies its keys' KV latent (1 KB) + k_pe (128 B) cache rows into
            the LDS tiles (rows of this launch are patched in by patch_new_kv)."""
            krows = [lds_ld(attn_keys, wave * KPW + jj) for jj in range(KPW)]
            for jj in range_constexpr(KPW):
                j = wave * KPW + jj
                for hh in range_constexpr(2):
                    kv8 = fx.Vector(
                        bo.buffer_load(r_kv, krows[jj] * (KV_LORA // 2) + lane * 8 + hh * 4, vec_width=4, dtype=T.i32)
                    )
                    fx.ptr_store(kv8.bitcast(fx.Float32), ktile + (j * KS + lane * 8 + hh * 4))
                lds_st(petile, j * PS + lane, ld_f32(r_pe, krows[jj] * (PE_DIM // 2) + lane))

        def patch_new_kv():
            """Rows appended by this launch come from the cache task's kvnew / penew pairs."""
            for jj in range_constexpr(KPW):
                j = wave * KPW + jj
                kr = lds_ld(attn_keys, j)
                if kr >= pos0:
                    sn = kr - pos0
                    # the row's latent and k_pe pairs in one poll
                    kvp = get2_many(
                        [(mb("kvnew"), sn * KV_LORA + lane * 16 + m * 2) for m in range(8)]
                        + [(mb("penew"), sn * PE_DIM + lane * 2)]
                    )
                    w = [bf16_pair(a0, a1) for a0, a1 in kvp[:8]]
                    for hh in range_constexpr(2):
                        fx.ptr_store(
                            fx.Vector.from_elements(w[hh * 4 : hh * 4 + 4], fx.Float32),
                            ktile + (j * KS + lane * 8 + hh * 4),
                        )
                    lds_st(petile, j * PS + lane, bf16_pair(*kvp[8]))

        enter("split")
        for tt in range(start("split"), S * N_SPLIT, G):
            tt = fx.Int32(tt)
            stamp("split", tt, 0)
            s = tt // N_SPLIT  # sample
            t = tt % N_SPLIT  # sparse-key chunk
            h = wave % H  # waves 8-15 repeat the softmax of heads 0-7
            nkeys, sparse = split_keys(t, s)
            gpu.barrier()
            gather_old_kv()  # before waiting for q: these rows are from earlier launches
            N_PE_T = PE_DIM // Q_B_TILE
            hint_wait(
                N_UK + H * N_PE_T + 1,
                lambda k: (
                    (k < N_UK).select(
                        fx.Int64(SC["q_lat"]),
                        (k < N_UK + H * N_PE_T).select(fx.Int64(SC["q_pe"]), fx.Int64(SC["penew"])),
                    )
                    + scratch,
                    (k < N_UK).select(
                        (s * H + k // UK_PER_HEAD) * KV_LORA + (k % UK_PER_HEAD) * UK_TILE + UK_TILE - 1,
                        (k < N_UK + H * N_PE_T).select(
                            (s * H + (k - N_UK) // N_PE_T) * PE_DIM + ((k - N_UK) % N_PE_T) * Q_B_TILE + Q_B_TILE - 1,
                            s * PE_DIM + PE_DIM - 1,
                        ),
                    ),
                ),
                mark=("split", tt),
            )
            # q of all heads -> bf16 Q[h][576] (words h * QS + d / 2): latent 512 then pe 64
            NQ = H * KV_LORA // 4 // THREADS
            tpe = fx.min(tid, H * PE_DIM // 4 - 1)
            qv = poll(
                [(mb("q_lat"), (s * H * KV_LORA + (tid + i * THREADS) * 4) // 2, 2) for i in range(NQ)]
                + [(mb("q_pe"), (s * H * PE_DIM + tpe * 4) // 2, 2)]
            )
            for i in range_constexpr(NQ):
                w4 = tid + i * THREADS
                qw = (w4 // (KV_LORA // 4)) * QS + (w4 % (KV_LORA // 4)) * 2
                lds_st(xs, qw, qv[i][0].bitcast(fx.Float32))
                lds_st(xs, qw + 1, qv[i][1].bitcast(fx.Float32))
            if tid < H * PE_DIM // 4:
                hh = tid // (PE_DIM // 4)
                qw = hh * QS + KV_LORA // 2 + (tid % (PE_DIM // 4)) * 2
                lds_st(xs, qw, qv[NQ][0].bitcast(fx.Float32))
                lds_st(xs, qw + 1, qv[NQ][1].bitcast(fx.Float32))
            patch_new_kv()
            stamp("split", tt, 2)
            gpu.barrier()
            stamp("split", tt, 5)
            # scores = K Q^T on WMMA.  The 64-key tile maps the working waves to (four
            # key row groups, two K halves); the batch-8 32-key tile uses two row
            # groups and computes every tile twice.  Only the first copy is stored.
            sw = wave % WORK_WAVES
            hn = fx.min(lane % 16, H - 1)
            rgk = sw % (SPLIT_KEYS // 16)
            c = fx.Vector.filled(ACC, 0.0, fx.Float32)
            for st in range_constexpr(QK_DIM // 32 // 2):
                kst = ((sw // (SPLIT_KEYS // 16)) % 2) * (QK_DIM // 32 // 2) + st
                key = rgk * 16 + lane % 16
                kw = (kst < KV_LORA // 32).select(
                    KT_OFF + key * KS + kst * 16,
                    PT_OFF + key * PS + (kst - KV_LORA // 32) * 16,
                )
                a = b16_lds(kw + (lane // 16) * 4)
                b = b16_lds(hn * QS + kst * 16 + (lane // 16) * 4)
                c = wmma_bf16_gfx1250(a, b, c)
            if wave < 2 * (SPLIT_KEYS // 16):
                fx.ptr_store(c, red + (wave * WAVE + lane) * ACC)
            gpu.barrier()
            stamp("split", tt, 6)
            # split-local softmax: wave h, lane + 32 rep = key j (score = sum of the two K halves)
            half_stride = SPLIT_KEYS // 16
            r16 = lane % 16
            cw = (h + 16 * (r16 // 8)) * ACC + r16 % 8
            valids, scores_ = [], []
            for rep in range_constexpr(KEY_REPS):
                key_rg = lane // 16 + rep * (WAVE // 16)
                valid = t * SPLIT_KEYS + lane + rep * WAVE < nkeys
                raw = lds_ld(red, key_rg * WAVE * ACC + cw) + lds_ld(red, (key_rg + half_stride) * WAVE * ACC + cw)
                valids.append(valid)
                scores_.append(valid.select(raw * scale, fx.Float32(NEG)))
            m_loc = scores_[0]
            for rep in range_constexpr(1, KEY_REPS):
                m_loc = fx.max(m_loc, scores_[rep])
            m = wave_max(m_loc)
            ps = [valids[rep].select(_exp(scores_[rep] - m), fx.Float32(0.0)) for rep in range(KEY_REPS)]
            p_loc = ps[0]
            for rep in range_constexpr(1, KEY_REPS):
                p_loc = p_loc + ps[rep]
            lsum = wave_sum(p_loc)
            for rep in range_constexpr(KEY_REPS):
                p_n = _xshfl(ps[rep], 1)
                if (lane % 2 == 0) & (wave < H):
                    lds_st(pl, h * (SPLIT_KEYS // 2) + (lane + rep * WAVE) // 2, bf16_pair(ps[rep], p_n))
            gpu.barrier()
            stamp("split", tt, 3)
            # O = P V on WMMA: heads M, keys K, latent dims N.  Each V word holds a dim
            # pair (even dim low), so one read feeds two WMMAs (even / odd dims): each
            # wave owns one group of 32 dims.  V is read key-strided from the tile.
            for g in range_constexpr(KV_LORA // 32 // WAVES):
                dw = (wave * (KV_LORA // 32 // WAVES) + g) * 16 + lane % 16  # dim pair word
                c0 = fx.Vector.filled(ACC, 0.0, fx.Float32)
                c1 = fx.Vector.filled(ACC, 0.0, fx.Float32)
                for js in range_constexpr(SPLIT_KEYS // 32):
                    a = b16_lds(hn * (SPLIT_KEYS // 2) + js * 16 + (lane // 16) * 4)
                    ws = [
                        fx.ptr_load(ktile + ((js * 32 + blk * 16 + (lane // 16) * 8 + i) * KS + dw)).bitcast(fx.Int32)
                        for blk in range(2)
                        for i in range(8)
                    ]
                    w_lo = [(ws[2 * i] & 0xFFFF) | (ws[2 * i + 1] << 16) for i in range(8)]
                    w_hi = [fx.Int32(fx.Uint32(ws[2 * i]) >> 16) | (ws[2 * i + 1] & -65536) for i in range(8)]
                    b0 = fx.Vector.from_elements(w_lo, fx.Int32).bitcast(fx.BFloat16)
                    b1 = fx.Vector.from_elements(w_hi, fx.Int32).bitcast(fx.BFloat16)
                    c0 = wmma_bf16_gfx1250(a, b0, c0)
                    c1 = wmma_bf16_gfx1250(a, b1, c1)
                if lane < 16:  # rows (heads) e < 8 live in lanes 0-15
                    for e in range_constexpr(ACC):
                        put_bf(mb("sp_acc"), ((s * N_SPLIT + t) * H + e) * KV_LORA + dw * 2, [c0[e], c1[e]])
            if (lane == 0) & (wave < H):  # written last: the merge's readiness hint
                put(mb("sp_m"), (s * N_SPLIT + t) * H + h, m)
                put(mb("sp_l"), (s * N_SPLIT + t) * H + h, lsum)
            stamp("split", tt, 4)

        # ========================== 6. split merge + W_UV: o = W_UV (softmax . KV)
        # 4 row groups x 8 chunks: 4 waves per row group, 2 chunks each
        r_wuv, r_suv = _rsrc(w_uv), _rsrc(s_uv)
        UV_NKC = KV_LORA // 64
        UV_R = UV_TILE // 16
        UV_WPR = WAVES // UV_R
        UV_UNITS = UV_NKC // (2 * UV_WPR)
        SPLIT_REPS = (N_SPLIT + WAVE - 1) // WAVE
        enter("uv")
        for tt in range(start("uv"), S * N_UV, G):
            tt = fx.Int32(tt)
            stamp("uv", tt, 0)
            s = tt // N_UV  # sample
            t = tt % N_UV  # 64-row tile
            head = t // (V_DIM // UV_TILE)

            def u_uv(c):
                kc = ((wave % UV_WPR) * UV_UNITS + c) * 2
                return unit_fp8x2(
                    r_wuv,
                    r_suv,
                    t * UV_R + wave // UV_WPR,
                    kc,
                    UV_NKC,
                    KV_LORA,
                    (kc * 64) // 2,
                    scale_rows=uv_scale_rows,
                )

            pre = [u_uv(c) for c in range(UV_UNITS)]
            hint_wait(N_SPLIT, lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head), mark=("uv", tt))
            pre_poll(N_SPLIT, lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head))
            stamp("uv", tt, 5)
            # Four quarters of the splits keep each poll and its live payload bounded.
            # Every thread owns one (quarter, half-of-latent) pair and processes
            # its two latent-pair positions sequentially; red holds quarter sums.
            SPH = N_SPLIT // 4
            dp_lo = tid % (KV_LORA // 4)
            hf = tid // (KV_LORA // 4)
            spis = [fx.min(lane + rep * WAVE, N_SPLIT - 1) for rep in range(SPLIT_REPS)]
            ml_specs = [(mb("sp_m"), (s * N_SPLIT + sp) * H + head, 1) for sp in spis] + [
                (mb("sp_l"), (s * N_SPLIT + sp) * H + head, 1) for sp in spis
            ]

            def acc_specs(dh):
                dp = dp_lo + dh * (KV_LORA // 4)
                return [
                    (mb("sp_acc"), ((s * N_SPLIT + hf * SPH + j) * H + head) * (KV_LORA // 2) + dp, 1)
                    for j in range(SPH)
                ]

            # pre_poll has seen every split's sp_l, so the weights and both payload
            # halves are polled in one batch while it stays small (18 pairs, S <= 4).
            UV_ONE_POLL = len(ml_specs) + 2 * SPH <= 18
            if const_expr(UV_ONE_POLL):
                polled = poll(ml_specs + acc_specs(0) + acc_specs(1), batch=len(ml_specs) + 2 * SPH)
                ml_got = polled[: len(ml_specs)]
            else:
                ml_got = poll(ml_specs)
            if wave == 0:  # per-split weights exp(m - M) / L for this head -> misc[sp]
                oks = [lane + rep * WAVE < N_SPLIT for rep in range(SPLIT_REPS)]
                m_sp = [oks[r].select(ml_got[r][0].bitcast(fx.Float32), fx.Float32(NEG)) for r in range(SPLIT_REPS)]
                l_sp = [
                    oks[r].select(ml_got[SPLIT_REPS + r][0].bitcast(fx.Float32), fx.Float32(0.0))
                    for r in range(SPLIT_REPS)
                ]
                m_loc = m_sp[0]
                for r in range_constexpr(1, SPLIT_REPS):
                    m_loc = fx.max(m_loc, m_sp[r])
                m_all = wave_max(m_loc)
                w_sp = [_exp(m_sp[r] - m_all) for r in range(SPLIT_REPS)]
                d_loc = l_sp[0] * w_sp[0]
                for r in range_constexpr(1, SPLIT_REPS):
                    d_loc = d_loc + l_sp[r] * w_sp[r]
                inv_den = _rcp(wave_sum(d_loc))
                for r in range_constexpr(SPLIT_REPS):
                    if oks[r]:
                        lds_st(misc, lane + r * WAVE, w_sp[r] * inv_den)
            stamp("uv", tt, 2)
            gpu.barrier()
            for dh in range_constexpr(2):
                dp = dp_lo + dh * (KV_LORA // 4)
                if const_expr(UV_ONE_POLL):
                    got = polled[len(ml_specs) + dh * SPH : len(ml_specs) + (dh + 1) * SPH]
                else:
                    got = poll(acc_specs(dh))
                o0 = fx.Float32(0.0)
                o1 = fx.Float32(0.0)
                for j in range_constexpr(SPH):
                    wj = lds_ld(misc, hf * SPH + j)
                    a0, a1 = bf2_f32(got[j][0])
                    o0 = o0 + a0 * wj
                    o1 = o1 + a1 * wj
                lds_st(red, (hf * (KV_LORA // 2) + dp) * 2, o0)
                lds_st(red, (hf * (KV_LORA // 2) + dp) * 2 + 1, o1)
            gpu.barrier()
            if tid < KV_LORA // 2:
                o0 = fx.Float32(0.0)
                o1 = fx.Float32(0.0)
                for q in range_constexpr(4):
                    o0 = o0 + lds_ld(red, (q * (KV_LORA // 2) + tid) * 2)
                    o1 = o1 + lds_ld(red, (q * (KV_LORA // 2) + tid) * 2 + 1)
                lds_st(xs, tid, bf16_pair(o0, o1))
            gpu.barrier()
            acc = run_units(u_uv, UV_UNITS, UV_UNITS, pre)
            reduce_rows(UV_R, acc, emit_out(UV_TILE))
            stamp("uv", tt, 3)
            gpu.barrier()
            if tid < UV_TILE // 4:
                r = tid * 4
                put_bf(mb("o"), s * O_K + t * UV_TILE + r, [lds_ld(outs, r + j) for j in range(4)])
            stamp("uv", tt, 4)

        # ====================== 7. W_o + attention TP peer reduce + residual -> a
        # 2 row groups x 32 chunks: 8 waves per row group, 4 chunks each
        r_wo, r_so = _rsrc(w_o), _rsrc(s_o)
        O_NKC = O_K // 64
        O_R = ROW_TILE // 16
        O_WPR = WAVES // O_R
        O_UNITS = O_NKC // (2 * O_WPR)
        enter("o")
        for t in range(start("o"), N_ROW_TILES, G):
            t = fx.Int32(t)
            stamp("o", t, 0)

            def u_o(c):
                kc = ((wave % O_WPR) * O_UNITS + c) * 2
                return unit_fp8x2(r_wo, r_so, t * O_R + wave // O_WPR, kc, O_NKC, O_K, (n_sel() * O_K + kc * 64) // 2)

            pre = [u_o(c) for c in range(O_UNITS)]
            hint_wait(
                S * N_UV, lambda k: (mb("o"), (k // N_UV) * O_K + (k % N_UV) * UV_TILE + UV_TILE - 1), mark=("o", t)
            )
            stage_x_pairs("o", S * O_K, lambda k: k)
            stamp("o", t, 2)
            gpu.barrier()
            acc = run_units(u_o, O_UNITS, O_UNITS, pre)
            reduce_rows(O_R, acc, emit_out(ROW_TILE))
            stamp("o", t, 3)
            gpu.barrier()

            def resid_h(s, row):
                w = fx.Vector.from_elements(
                    [fx.Int32(bo.buffer_load(r_h, (s * HIDDEN + row) // 2, vec_width=1, dtype=T.i32))], fx.Int32
                )
                v = w.bitcast(fx.BFloat16).to(fx.Float32)
                return v[0], v[1]

            peer_reduce(
                "attn",
                t,
                resid_h,
                lambda s, row, v0, v1: put_bf(mb("a"), s * HIDDEN + row, [v0, v1]),
            )
            stamp("o", t, 4)

        # ====== 8. post-attn RMSNorm -> router scores + this task's FP8 activation blocks
        # One sample per CTA: 1 row group x 96 chunks (bf16), 16 waves split K.
        # S > 1 gets S times as many independent router CTAs instead of serializing
        # every sample's normalization and output columns inside one CTA.
        r_wr = _rsrc(w_r)
        R_NKC = HIDDEN // 64
        enter("router")
        for tt in range(start("router"), S * N_ROUTER, G):
            tt = fx.Int32(tt)
            t = tt if const_expr(S == 1) else tt % N_ROUTER
            router_sample = fx.Int32(0) if const_expr(S == 1) else tt // N_ROUTER
            stamp("router", tt, 0)

            # K-fold: WMMA rows / B columns 0..7 take this wave's first K half, rows /
            # columns 8..15 the second, so every loaded weight row is distinct and the
            # whole K slice is prefetched; logit = C[r][0] + C[8 + r][8]
            r_sub = t * ROUTER_TILE % 16  # this task's rows of the 16-row group
            r_ln = (lane & -16) | (r_sub + lane % ROUTER_TILE)
            R_CPW = R_NKC // WAVES // 2
            r_fold = (lane % 16) // ROUTER_TILE
            r_ns = fx.Int32(0)

            def u_r(c):
                kc = wave * (R_NKC // WAVES) + r_fold * R_CPW + c
                return unit_bf16(r_wr, t * ROUTER_TILE // 16, kc, R_NKC, (r_ns * HIDDEN + kc * 64) // 2, r_ln)

            pre = [u_r(c) for c in range(R_CPW)]
            hint_wait(
                N_ROW_TILES,
                lambda k: (mb("a"), router_sample * HIDDEN + k * ROW_TILE + ROW_TILE - 1),
                mark=("router", tt),
            )
            # this task's FP8 activation block inputs ride along with the staging loads:
            # wave w quantizes block w * N_ROUTER + t of this CTA's sample.
            r_gp = _rsrc(g_post)
            x_blk = wave * N_ROUTER + t
            x_s = router_sample
            x_ok = (wave < XQ_WAVES) & (x_blk < XQ_BLOCKS)
            xk = fx.min(x_blk, XQ_BLOCKS - 1) * 128 + lane * 4
            xg = [ld_bf16(r_gp, xk + j) for j in range(4)]
            xa = []

            def ld_a(sks):
                specs = [(mb("a"), (router_sample * HIDDEN + k) // 2, 2) for s, k in sks]
                specs.append((mb("a"), (x_s * HIDDEN + xk) // 2, 2))
                v = poll(specs, batch=len(specs))
                stamp("router", tt, 5, lead=THREADS - WAVE)
                xa.append(list(bf2_f32(v[-1][0])) + list(bf2_f32(v[-1][1])))
                return [list(bf2_f32(w[0])) + list(bf2_f32(w[1])) for w in v[:-1]]

            rstds = stage_x_rmsnorm(ld_a, HIDDEN, g_post, mark=("router", tt), count=1)
            stamp("router", tt, 2)
            # this task's FP8 activation blocks go out ahead of the gate GEMV
            if x_ok:
                x_rstd = rstds[0]
                q, qs = quant4([xa[0][j] * x_rstd * xg[j] for j in range(4)])
                w8 = fp8_word4(q)
                put(mb("xq"), (x_s * HIDDEN + xk) // 4, w8)
                lo, hi = fp8_word_bf16(w8)
                d = list(bf2_f32(lo)) + list(bf2_f32(hi))
                bo.buffer_store(
                    fx.Vector.from_elements([d[j] * qs for j in range(4)], fx.Float32),
                    _rsrc(mb("xqd")),
                    x_s * HIDDEN + xk,
                )
                if lane == 0:
                    put(mb("xqs"), x_s * XQ_BLOCKS + x_blk, qs)
            gpu.barrier()
            acc = run_units(u_r, R_CPW, R_CPW, pre)
            store_acc(acc)
            gpu.barrier()
            stamp("router", tt, 3)
            if tid < ROUTER_TILE:
                r = tid % ROUTER_TILE
                logit = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    for f in range_constexpr(2):
                        logit = logit + lds_ld(red, c_word(w, f * ROUTER_TILE + r, f * ROUTER_TILE))
                put(mb("scores"), router_sample * N_EXPERTS + t * ROUTER_TILE + r, _rcp(1.0 + _exp(-logit)))
            stamp("router", tt, 4)

        def dn_route(bs):
            """Expert-down routing (wave s -> sample s): expert ids -> keys[s * 9 + slot],
            route weights -> dnw[]; the scores must have landed."""
            if wave < S:
                e, w = route_top8(wave, bs=bs)
                if lane < MOE_SLOTS:  # slot 0: the shared expert, then pick lane (slot lane + 1)
                    q = wave * MOE_SLOTS + (lane + 1) % MOE_SLOTS
                    lds_st(keys, q, (lane == TOP_K).select(fx.Int32(SHARED_EXPERT), e))
                    lds_st(dnw, q, (lane == TOP_K).select(fx.Float32(1.0), w))

        # ================================ 9. expert up/gate + SiLU
        UG_NKC = HIDDEN // 64
        UG_W_BYTES = 2 * INTER * HIDDEN // (2 if expert_mxfp4 else 1)
        UG_S_BYTES = 2 * INTER * (HIDDEN // 32) if expert_mxfp4 else 2 * INTER // SCALE_BM * (HIDDEN // 128) * 4
        UG8 = 8
        # WMMA rows 0-7: gate rows c * 8 .., rows 8-15: the matching up rows.  Lane ``w_ln``
        # holds the packed row this lane's A operand needs; accumulator rows of lanes 0-15
        # are gate rows and of lanes 16-31 up rows, so the weight-scale row group follows.
        UG_ROW_GROUPS = INTER // 16

        def ug_lanes(c):
            w_rg = ((lane % 16) // 8) * UG_ROW_GROUPS + c // 2
            w_ln = (lane & -16) | ((c % 2) * 8 + lane % 8)
            s_rg = (lane // 16) * UG_ROW_GROUPS + c // 2
            return w_rg, w_ln, s_rg

        def expert_rsrc(base_addr, e, nbytes, live=None):
            num = None if live is None else live.select(fx.Int32(nbytes), fx.Int32(0))
            return bo.create_buffer_resource_from_addr(
                base_addr + fx.Int64(e) * fx.Int64(nbytes), num_records_bytes=num
            )

        enter("ug")
        if const_expr(S == 1):
            # one task per CTA: task u takes intermediates (u % 32) * 8 of routed slot
            # u // 32 (slot 8 for u < 32, which also take the shared expert's); the 8 gate
            # + 8 up rows are one WMMA row group and all waves split K
            UG8_CPW = UG_NKC // WAVES
            for u in range(start("ug"), G, G):
                u = fx.Int32(u)
                stamp("ug", u, 0)
                s_u, c = fx.Int32(0), u % (INTER // UG8)
                has_sh = u < INTER // UG8
                slot = has_sh.select(fx.Int32(MOE_SLOTS - 1), u // (INTER // UG8))
                # the FP8 activation is computed here from the post-attention state (in
                # parallel with the router): RMSNorm, then per-128 quant with one wave per
                # block -> X[0] (FP8 values in bf16), block scales -> misc[8:]
                NB = XQ_BLOCKS // WAVES
                ks_ = [(wave + j * WAVES) * 128 + lane * 4 for j in range(NB)]
                r_gp = _rsrc(g_post)
                gps = [[ld_bf16(r_gp, k + i) for i in range(4)] for k in ks_]  # issued ahead of the wait
                bs = load_bias()
                w_rg, w_ln, s_rg = ug_lanes(c)

                def u_ug8(cc, e, live=None):  # expert e's weights (loads return 0 unless live)
                    r_wug = expert_rsrc(w_ug, e, UG_W_BYTES, live)
                    r_sug = expert_rsrc(s_ug, e, UG_S_BYTES, live)
                    unit = wave * (UG8_CPW // 2) + cc
                    kc = unit * 2
                    if const_expr(expert_mxfp4):
                        return unit_mxfp4(
                            r_wug,
                            r_sug,
                            w_rg,
                            unit,
                            HIDDEN,
                            unit * 64,
                            lambda: _uniform_f32(lds_ld(misc, 8 + unit)),
                            w_ln,
                        )
                    return unit_fp8x2(
                        r_wug,
                        r_sug,
                        w_rg,
                        kc,
                        UG_NKC,
                        HIDDEN,
                        kc * 32,
                        lambda: _uniform_f32(lds_ld(misc, 8 + kc // 2)),
                        ln=w_ln,
                        s_rg=s_rg,
                    )

                # the shared expert's weights do not depend on routing: prefetch them (the
                # later zero-weight MMAs of the other tasks are cheaper than a branch)
                pre = [u_ug8(cc, fx.Int32(SHARED_EXPERT), has_sh) for cc in range(UG8_CPW // 2)]
                hint_wait(N_ROW_TILES, lambda k: (mb("a"), s_u * HIDDEN + k * ROW_TILE + ROW_TILE - 1), mark=("ug", u))
                # the sum of squares takes the router's element partition and order
                # (stage_x_rmsnorm), so rstd -- and every FP8 rounding -- is bit-identical
                NQ4 = HIDDEN // (4 * THREADS)
                got = poll(
                    [(mb("a"), (s_u * HIDDEN + (tid + i * THREADS) * 4) // 2, 2) for i in range(NQ4)]
                    + [(mb("a"), (s_u * HIDDEN + k) // 2, 2) for k in ks_]
                )
                av = [list(bf2_f32(w[0])) + list(bf2_f32(w[1])) for w in got[NQ4:]]
                ss = fx.Float32(0.0)
                for w in got[:NQ4]:
                    for a in list(bf2_f32(w[0])) + list(bf2_f32(w[1])):
                        ss = ss + a * a
                rstd = _rsq(block_sum(ss) * (1.0 / HIDDEN) + EPS)
                for j in range_constexpr(NB):
                    q, qs = quant4([av[j][i] * rstd * gps[j][i] for i in range(4)])
                    st_fp8x4(ks_[j], q)
                    if lane == 0:
                        lds_st(misc, 8 + wave + j * WAVES, qs)
                if wave == 0:
                    e, w = route_top8(s_u, bs=bs)
                    if lane == slot - 1:
                        lds_st(keys, 0, e)
                        lds_st(misc, 0, w)
                stamp("ug", u, 2)
                gpu.barrier()
                e_sel = _uniform(lds_ld(keys, 0))
                post = [u_ug8(cc, e_sel) for cc in range(UG8_CPW // 2)]
                reduce_rows(1, mma_units(zero_acc(), pre), emit_out(16))
                gpu.barrier()
                reduce_rows(1, mma_units(zero_acc(), post), lambda rl, n, v: lds_st(outs, 16 + rl, v))
                stamp("ug", u, 3)
                gpu.barrier()
                if tid < UG8:  # threads 0-3: the shared expert's rows, 4-7: the routed slot's
                    r = (tid % (UG8 // 2)) * 2
                    o = (tid // (UG8 // 2)) * 16
                    g0, g1 = lds_ld(outs, o + r), lds_ld(outs, o + r + 1)
                    u0, u1 = lds_ld(outs, o + UG8 + r), lds_ld(outs, o + UG8 + r + 1)
                    if has_sh | (tid >= UG8 // 2):
                        put2(
                            mb("mid"),
                            (tid < UG8 // 2).select(fx.Int32(0), slot) * INTER + c * UG8 + r,
                            g0 * _rcp(1.0 + _exp(-g0)) * u0,
                            g1 * _rcp(1.0 + _exp(-g1)) * u1,
                        )
                if (c == 0) & (tid == 0):  # routing record (debug / tests)
                    put(mb("sel"), slot, e_sel)
                    put(mb("prob"), slot, lds_ld(misc, 0))
                    if has_sh:
                        put(mb("sel"), 0, fx.Int32(SHARED_EXPERT))
                        put(mb("prob"), 0, fx.Float32(1.0))
                stamp("ug", u, 4)
        else:
            # One eight-intermediate tile per CTA.  Shared-expert weights feed
            # all sample columns of one WMMA, while routed-expert weights are
            # prefetched one sample ahead.
            UG8_UNITS = (HIDDEN // 128) // WAVES
            u = fx.Int32(start("ug"))
            c = u % (INTER // UG8)
            has_sh = u < INTER // UG8
            slot = has_sh.select(fx.Int32(MOE_SLOTS - 1), u // (INTER // UG8))
            w_rg, w_ln, s_rg = ug_lanes(c)

            def ug8_units(e, sample, live=None):
                rw = expert_rsrc(w_ug, e, UG_W_BYTES, live)
                rs = expert_rsrc(s_ug, e, UG_S_BYTES, live)
                sn = n_sel() if sample is None else fx.Int32(sample)
                units = []
                for cc in range_constexpr(UG8_UNITS):
                    unit = wave * UG8_UNITS + cc
                    kc = unit * 2
                    if const_expr(expert_mxfp4):
                        units.append(
                            unit_mxfp4(
                                rw,
                                rs,
                                w_rg,
                                unit,
                                HIDDEN,
                                sn * XW + unit * 64,
                                lambda unit=unit, sn=sn: lds_ld(misc, 8 + sn * XQ_BLOCKS + unit),
                                w_ln,
                            )
                        )
                        continue
                    units.append(
                        unit_fp8x2(
                            rw,
                            rs,
                            w_rg,
                            kc,
                            UG_NKC,
                            HIDDEN,
                            sn * XW + kc * 32,
                            lambda kc=kc, sn=sn: lds_ld(misc, 8 + sn * XQ_BLOCKS + kc // 2),
                            ln=w_ln,
                            s_rg=s_rg,
                        )
                    )
                return units

            def ug8_emit(sample, shared):
                if tid < (S if shared else 1) * UG8 // 2:
                    n = tid // (UG8 // 2)
                    r = (tid % (UG8 // 2)) * 2
                    g0, g1 = lds_ld(outs, n * 16 + r), lds_ld(outs, n * 16 + r + 1)
                    v0 = lds_ld(outs, n * 16 + UG8 + r)
                    v1 = lds_ld(outs, n * 16 + UG8 + r + 1)
                    sn = n if shared else fx.Int32(sample)
                    sl = fx.Int32(0) if shared else slot
                    put2(
                        mb("mid"),
                        (sn * MOE_SLOTS + sl) * INTER + c * UG8 + r,
                        g0 * _rcp(1.0 + _exp(-g0)) * v0,
                        g1 * _rcp(1.0 + _exp(-g1)) * v1,
                    )
                if (c == 0) & (tid < S if shared else tid == 0):
                    sn = tid if shared else fx.Int32(sample)
                    sl = fx.Int32(0) if shared else slot
                    put(mb("sel"), sn * MOE_SLOTS + sl, lds_ld(keys, sn * MOE_SLOTS + sl))
                    put(mb("prob"), sn * MOE_SLOTS + sl, lds_ld(dnw, sn * MOE_SLOTS + sl))

            dn_route(load_bias())
            gpu.barrier()
            shared_pre = ug8_units(fx.Int32(SHARED_EXPERT), None, has_sh)
            cur = ug8_units(_uniform(lds_ld(keys, slot)), 0)
            stage_xq(list(range(S)))
            gpu.barrier()
            if has_sh:
                reduce_rows(1, mma_units(zero_acc(), shared_pre), emit_out(16))
                gpu.barrier()
                ug8_emit(0, True)
            for sample in range_constexpr(S):
                stamp("ug", sample * G + u, 0)
                pre = cur
                if const_expr(sample + 1 < S):
                    cur = ug8_units(_uniform(lds_ld(keys, (sample + 1) * MOE_SLOTS + slot)), sample + 1)
                reduce_rows(1, mma_units(zero_acc(), pre), emit_out(16))
                gpu.barrier()
                ug8_emit(sample, False)
                stamp("ug", sample * G + u, 4)

        # ======== 10. mid FP8 quant + expert down + route weighting + MoE TP reduce
        # 2 row groups x (sample tile * 9 slots * 4) chunks: 8 waves per group.
        DN_NKC = INTER // 64
        DN_R = (DN_TILE + 15) // 16  # 16-row groups touched by a tile (24-row tiles start at row 0 or 8 of one)
        DN_WPR = WAVES // DN_R
        # A 128-K unit is 16 VGPRs per wave32 lane, twice gfx950's; nine in flight
        # spilled at S=4 under the 256-VGPR budget of a 16-wave CTA.
        DN_BATCH = 4 if S > 4 else 5
        DN_W_BYTES = HIDDEN * INTER // (2 if expert_mxfp4 else 1)
        DN_S_BYTES = HIDDEN * (INTER // 32) if expert_mxfp4 else HIDDEN // SCALE_BM * (INTER // 128) * 4
        enter("down")
        for t in range(start("down"), N_DN_TILES, G):
            t = fx.Int32(t)
            stamp("down", t, 0)
            dn_route(load_bias())
            gpu.barrier()
            gu = wave // DN_WPR
            dn_rg = t * DN_TILE // 16
            dn_off = t * DN_TILE % 16
            # this lane's row, as a tile row; rows outside the tile load their lane ^ 8 twin
            # (same cache lines) and are dropped in the output
            dn_lr = gu * 16 + lane % 16 - dn_off
            dn_ln = ((dn_lr >= 0) & (dn_lr < DN_TILE)).select(lane, lane ^ 8)

            DN_NU = S * MOE_SLOTS * DN_NKC // 2
            DN_UPW = (DN_NU + DN_WPR - 1) // DN_WPR
            DN_BLK = S * MOE_SLOTS * INTER // 128

            def u_dn(cc):  # cc: 128-k chunk of this wave
                qu = (wave % DN_WPR) * DN_UPW + cc
                live = qu < DN_NU
                unit = fx.min(qu, DN_NU - 1)
                q = unit * 2  # 64-k chunk index over (s, slot, kc)
                s_q = q // (MOE_SLOTS * DN_NKC)
                slot_q = (q // DN_NKC) % MOE_SLOTS
                kc = q % DN_NKC
                e = _uniform(lds_ld(keys, s_q * MOE_SLOTS + slot_q))
                wb = expert_rsrc(w_dn, e, DN_W_BYTES, None if DN_NU % DN_WPR == 0 else live)
                sb = expert_rsrc(s_dn, e, DN_S_BYTES, None if DN_NU % DN_WPR == 0 else live)

                def coef():  # mid block scale * route weight, only in this sample's column
                    return (lane % 16 == s_q).select(_uniform_f32(lds_ld(misc, q // 2)), fx.Float32(0.0))

                if const_expr(expert_mxfp4):

                    def bf16_coef():
                        return (lane % 16 == s_q).select(
                            _uniform_f32(lds_ld(dnw, s_q * MOE_SLOTS + slot_q)), fx.Float32(0.0)
                        )

                    return unit_mxfp4(
                        wb,
                        sb,
                        dn_rg + gu,
                        unit % (INTER // 128),
                        INTER,
                        unit * 64,
                        bf16_coef,
                        dn_ln,
                    )
                return unit_fp8x2(wb, sb, dn_rg + gu, kc, DN_NKC, INTER, q * 32, coef, ln=dn_ln)

            if const_expr(S <= 4):
                pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
            hint_wait(
                N_UG,
                lambda k: (
                    mb("mid"),
                    (k // (MOE_SLOTS * N_UG_PER_SLOT) * MOE_SLOTS + (k // N_UG_PER_SLOT) % MOE_SLOTS) * INTER
                    + (k % N_UG_PER_SLOT) * UG_TILE
                    + UG_TILE
                    - 1,
                ),
                mark=("down", t),
            )
            MID_REPS = (DN_BLK + WAVES - 1) // WAVES
            mids = get2_many(
                [
                    (mb("mid"), fx.min(wave + b * WAVES, DN_BLK - 1) * 128 + lane * 4 + 2 * hh)
                    for b in range(MID_REPS)
                    for hh in range(2)
                ]
            )
            stamp("down", t, 2)
            for b in range_constexpr(MID_REPS):
                blk = wave + b * WAVES
                if blk < DN_BLK:
                    vals = list(mids[2 * b]) + list(mids[2 * b + 1])
                    if const_expr(expert_mxfp4):
                        st_x4(blk * 128 + lane * 4, vals)
                    else:
                        q, qs = quant4(vals)
                        st_fp8x4(blk * 128 + lane * 4, q)
                        if lane == 0:
                            lds_st(misc, blk, qs * lds_ld(dnw, blk // (INTER // 128)))
            gpu.barrier()
            if const_expr(S > 4):
                pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
            acc = run_units(u_dn, DN_UPW, DN_BATCH, pre)

            def emit_dn(rl, n, v):
                if (rl >= dn_off) & (rl < dn_off + DN_TILE):
                    lds_st(outs, n * DN_TILE + rl - dn_off, v)

            reduce_rows(DN_R, acc, emit_dn)
            stamp("down", t, 3)
            gpu.barrier()

            def store_x(s, row, v0, v1):
                bo.buffer_store(
                    fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16), _rsrc(x_out), s * HIDDEN + row
                )

            peer_reduce("ffn", t, mb("a"), store_x, tile=DN_TILE)
            gpu.barrier()
            stamp("down", t, 4)

    @flyc.jit
    def launch(
        h_in: fx.Int64,
        x_out: fx.Int64,
        cur_pos: fx.Int64,
        kv_cache: fx.Int64,
        pe_cache: fx.Int64,
        indices: fx.Int64,
        rope_cos: fx.Int64,
        rope_sin: fx.Int64,
        g_in: fx.Int64,
        g_q: fx.Int64,
        g_kv: fx.Int64,
        g_post: fx.Int64,
        w_qkv_a: fx.Int64,
        s_qkv_a: fx.Int64,
        w_q_b: fx.Int64,
        s_q_b: fx.Int64,
        w_uk: fx.Int64,
        s_uk: fx.Int64,
        w_uv: fx.Int64,
        s_uv: fx.Int64,
        w_o: fx.Int64,
        s_o: fx.Int64,
        w_r: fx.Int64,
        bias: fx.Int64,
        w_ug: fx.Int64,
        s_ug: fx.Int64,
        w_dn: fx.Int64,
        s_dn: fx.Int64,
        scratch: fx.Int64,
        sym: fx.Int64,
        peers: fx.Int64,
        timeline_buf: fx.Int64,
        step: fx.Int64,
        rank: fx.Int32,
        layer: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        glm5_monokernel_gfx1250(
            h_in,
            x_out,
            cur_pos,
            kv_cache,
            pe_cache,
            indices,
            rope_cos,
            rope_sin,
            g_in,
            g_q,
            g_kv,
            g_post,
            w_qkv_a,
            s_qkv_a,
            w_q_b,
            s_q_b,
            w_uk,
            s_uk,
            w_uv,
            s_uv,
            w_o,
            s_o,
            w_r,
            bias,
            w_ug,
            s_ug,
            w_dn,
            s_dn,
            scratch,
            sym,
            peers,
            timeline_buf,
            step,
            rank,
            layer,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)

    return launch
