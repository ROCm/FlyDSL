# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Sparse MLA, split merge, output projection and TP residual."""

from typing import Callable, NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.config import FP8_MAX, HIDDEN, KV_LORA, PE_DIM, Q_LORA, V_DIM
from kernels.monokernel.glm.layout import (
    N_QKV_A,
    N_ROW_TILES,
    Q_B_TILE,
    QKV_A_TILE,
    ROW_TILE,
    UK_TILE,
    UV_TILE,
    WAVES,
    dcp_local_uv_tile,
    split_acc_head,
    split_score_column,
)
from kernels.monokernel.layout import CM_DEV, NEG, THREADS
from kernels.monokernel.ops import div_rn, spin_pause
from kernels.monokernel.ops import exp as _exp
from kernels.monokernel.ops import fp8_to_bf16x8 as _fp8_to_bf16x8
from kernels.monokernel.ops import rcp as _rcp
from kernels.monokernel.ops import rsrc as _rsrc
from kernels.monokernel.ops import uniform as _uniform
from kernels.monokernel.ops import xshfl as _xshfl


class AttentionStages(NamedTuple):
    sparse: Callable
    merge: Callable
    output: Callable


@flyc.jit
def build_attention(plan, state, tools):
    COHORT_ROW_TILES = plan.COHORT_ROW_TILES
    DEFER_OFF_CACHE = plan.DEFER_OFF_CACHE
    DN_UNION_CANDIDATE = plan.DN_UNION_CANDIDATE
    FRONT_PAYLOAD_RETIRE = plan.FRONT_PAYLOAD_RETIRE
    G = plan.G
    GLOBAL_UV = plan.GLOBAL_UV
    H = plan.H
    KS = plan.KS
    KT_OFF = plan.KT_OFF
    L = plan.L
    N_SPLIT = plan.N_SPLIT
    N_SPLIT_TASKS = plan.N_SPLIT_TASKS
    N_UK = plan.N_UK
    N_UV = plan.N_UV
    O_K = plan.O_K
    O_PAD4 = plan.O_PAD4
    PS = plan.PS
    PT_OFF = plan.PT_OFF
    QK_DIM = plan.QK_DIM
    QS = plan.QS
    S = plan.S
    SC = plan.SC
    SPLIT_ALL_HEADS = plan.SPLIT_ALL_HEADS
    SPLIT_GATHER_KEYS = plan.SPLIT_GATHER_KEYS
    SPLIT_KEYS = plan.SPLIT_KEYS
    SPLIT_TASK_PAIR = plan.SPLIT_TASK_PAIR
    TILE_SCRATCH_BYTES = plan.TILE_SCRATCH_BYTES
    UK_PER_HEAD = plan.UK_PER_HEAD
    UV_MERGE_OFF = plan.UV_MERGE_OFF
    UV_REUSE = plan.UV_REUSE
    attention_k_chunks_per_unit = plan.attention_k_chunks_per_unit
    attention_ptpc = plan.attention_ptpc
    cache_fp8 = plan.cache_fp8
    index_request_width = plan.index_request_width
    indexer_cp = plan.indexer_cp
    native_fp4_mfma = plan.native_fp4_mfma
    npes = plan.npes
    scale = plan.scale
    topk = plan.topk
    total_samples = plan.total_samples
    use_atom_kv_cache = plan.use_atom_kv_cache
    uv_scale_rows = plan.uv_scale_rows
    with_indexer = plan.with_indexer
    attn_keys = state.attn_keys
    full_bid = state.full_bid
    full_scratch = state.full_scratch
    full_slots = state.full_slots
    indices = state.indices
    ktile = state.ktile
    kv_cache = state.kv_cache
    lane = state.lane
    misc = state.misc
    outs = state.outs
    pe_cache = state.pe_cache
    petile = state.petile
    pl = state.pl
    pos0 = state.pos0
    r_h = state.r_h
    red = state.red
    row_offset = state.row_offset
    s_o = state.s_o
    s_uv = state.s_uv
    scratch = state.scratch
    tag = state.tag
    tid = state.tid
    v4f = state.v4f
    w_o = state.w_o
    w_uv = state.w_uv
    wave = state.wave
    xs = state.xs
    ld_f32 = tools.ld_f32
    row_index_bounds = tools.row_index_bounds
    row_slot = tools.row_slot
    row_writes_cache = tools.row_writes_cache
    index_page_slot = tools.index_page_slot
    lds_ld = tools.lds_ld
    lds_st = tools.lds_st
    bf16_pair = tools.bf16_pair
    bf16_round = tools.bf16_round
    mb = tools.mb
    put = tools.put
    put_bf = tools.put_bf
    bf2_f32 = tools.bf2_f32
    poll = tools.poll
    hint_wait = tools.hint_wait
    pre_poll = tools.pre_poll
    get = tools.get
    get2_many = tools.get2_many
    get2 = tools.get2
    wave_sum = tools.wave_sum
    wave_max = tools.wave_max
    unit_attention = tools.unit_attention
    run_units = tools.run_units
    reduce_rows = tools.reduce_rows
    emit_out = tools.emit_out
    attention_b_word = tools.attention_b_word
    emit_ptpc = tools.emit_ptpc
    emit_ptpc_scalar = tools.emit_ptpc_scalar
    stage_x_pairs_ptpc = tools.stage_x_pairs_ptpc
    stage_x_pairs = tools.stage_x_pairs
    st_ptpc = tools.st_ptpc
    o_owner_reduce = tools.o_owner_reduce
    peer_reduce = tools.peer_reduce
    dcp_merge_latent = tools.dcp_merge_latent
    start = tools.start
    n_sel = tools.n_sel

    def sparse():
        # ================================== 5. sparse MLA split: 32 keys x 8 heads
        if const_expr(SPLIT_ALL_HEADS):
            r_kv = _rsrc(kv_cache)
            r_pe = _rsrc(pe_cache)
            r_idx = _rsrc(indices)
            KPW = SPLIT_KEYS // WAVES

            def split_keys(t, s):
                """(nkeys, sparse) of sample s; wave 0 writes this split's cache rows to LDS."""
                if const_expr(use_atom_kv_cache):
                    index_base, index_end = row_index_bounds(s)
                    nkeys = index_end - index_base
                    sparse = index_end > index_base
                else:
                    index_base = s * topk
                    kv_len = pos0 + s + 1
                    sparse = kv_len > topk
                    nkeys = sparse.select(fx.Int32(topk), kv_len)
                if const_expr(with_indexer):
                    if sparse:
                        if wave == 0:
                            if lane == 0:
                                get(mb("indices_ready"), s)
                            # Only wave 0 consumes the compact index payload; its
                            # divergent ready poll reconverges before this acquire.
                            fx.memory_fence(
                                ordering=fx.AtomicOrdering.Acquire,
                                syncscope="agent",
                            )
                if wave == 0:
                    if lane < SPLIT_KEYS:
                        k_pos = t * SPLIT_KEYS + lane
                        k_cl = (k_pos < nkeys).select(k_pos, 0)
                        if const_expr(with_indexer):
                            idx = k_cl
                            if sparse:
                                idx = fx.Int32(
                                    bo.buffer_load(
                                        _rsrc(mb("indices")),
                                        s * topk + k_cl,
                                        vec_width=1,
                                        dtype=T.i32,
                                        cache_modifier=CM_DEV,
                                    )
                                )
                            lds_st(attn_keys, lane, index_page_slot(s, idx))
                        elif const_expr(use_atom_kv_cache):
                            idx = fx.Int32(0)
                            if sparse:
                                idx = fx.Int32(
                                    bo.buffer_load(
                                        r_idx,
                                        index_base + k_cl,
                                        vec_width=1,
                                        dtype=T.i32,
                                    )
                                )
                            lds_st(attn_keys, lane, idx)
                        else:
                            idx = fx.Int32(
                                bo.buffer_load(
                                    r_idx,
                                    index_base + k_cl,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            )
                            lds_st(attn_keys, lane, sparse.select(idx, k_cl))
                return nkeys, sparse

            def gather_old_kv():
                """Copy persistent KV/PE rows to LDS before patching incoming cache versions."""
                krows = [lds_ld(attn_keys, wave * KPW + jj) for jj in range(KPW)]
                if const_expr(use_atom_kv_cache):
                    # Each gathered row is wave-uniform; retain two independent
                    # row loads while putting its full byte address in the base.
                    cache_row_bytes = QK_DIM * (1 if cache_fp8 else 2)
                    for gather_base in range_constexpr(0, KPW, 2):
                        descriptors = [
                            _rsrc(kv_cache + fx.Int64(_uniform(krows[gather_base + part])) * cache_row_bytes)
                            for part in range(min(2, KPW - gather_base))
                        ]
                        raw_values = [
                            fx.Vector(
                                bo.buffer_load(
                                    descriptors[part],
                                    lane * (2 if cache_fp8 else 4),
                                    vec_width=2 if cache_fp8 else 4,
                                    dtype=T.i32,
                                )
                            )
                            for part in range(min(2, KPW - gather_base))
                        ]
                        for part in range_constexpr(min(2, KPW - gather_base)):
                            raw = raw_values[part]
                            value = _fp8_to_bf16x8(raw[0], raw[1]) if const_expr(cache_fp8) else raw
                            fx.ptr_store(
                                value.bitcast(fx.Float32),
                                ktile + ((wave * KPW + gather_base + part) * KS + lane * 4),
                            )
                        if lane < (PE_DIM // 8 if const_expr(cache_fp8) else PE_DIM // 2):
                            pe_values = [
                                bo.buffer_load(
                                    _rsrc(pe_cache + fx.Int64(_uniform(krows[gather_base + part])) * cache_row_bytes),
                                    KV_LORA // (4 if cache_fp8 else 2) + lane * (2 if cache_fp8 else 1),
                                    vec_width=2 if cache_fp8 else 1,
                                    dtype=T.i32,
                                )
                                for part in range(min(2, KPW - gather_base))
                            ]
                            for part in range_constexpr(min(2, KPW - gather_base)):
                                if const_expr(cache_fp8):
                                    raw = fx.Vector(pe_values[part])
                                    fx.ptr_store(
                                        _fp8_to_bf16x8(raw[0], raw[1]).bitcast(fx.Float32),
                                        petile + ((wave * KPW + gather_base + part) * PS + lane * 4),
                                    )
                                else:
                                    lds_st(
                                        petile,
                                        (wave * KPW + gather_base + part) * PS + lane,
                                        fx.Int32(pe_values[part]).bitcast(fx.Float32),
                                    )
                elif const_expr(cache_fp8 and KPW % 2 == 0):
                    # Keep two independent rows outstanding before conversion.
                    # The complete gather still precedes q/patch dependencies.
                    for gather_base in range_constexpr(0, KPW, 2):
                        gather_kv = [
                            fx.Vector(
                                bo.buffer_load(
                                    r_kv,
                                    krows[gather_base + part] * (QK_DIM // 4) + lane * 2,
                                    vec_width=2,
                                    dtype=T.i32,
                                )
                            )
                            for part in range(2)
                        ]
                        for part in range_constexpr(2):
                            raw = gather_kv[part]
                            fx.ptr_store(
                                _fp8_to_bf16x8(raw[0], raw[1]).bitcast(fx.Float32),
                                ktile + ((wave * KPW + gather_base + part) * KS + lane * 4),
                            )
                        if lane < PE_DIM // 8:
                            gather_pe = [
                                fx.Vector(
                                    bo.buffer_load(
                                        r_pe,
                                        krows[gather_base + part] * (QK_DIM // 4) + KV_LORA // 4 + lane * 2,
                                        vec_width=2,
                                        dtype=T.i32,
                                    )
                                )
                                for part in range(2)
                            ]
                            for part in range_constexpr(2):
                                raw = gather_pe[part]
                                fx.ptr_store(
                                    _fp8_to_bf16x8(raw[0], raw[1]).bitcast(fx.Float32),
                                    petile + ((wave * KPW + gather_base + part) * PS + lane * 4),
                                )
                elif const_expr(
                    COHORT_ROW_TILES and not with_indexer and native_fp4_mfma and npes == 4 and not cache_fp8
                ):
                    # Two independent persistent-cache rows outstanding per wave.
                    # Complete the original gather before any q/patch dependency.
                    for gather_base in range_constexpr(0, KPW, 2):
                        gather_kv = [
                            fx.Vector(
                                bo.buffer_load(
                                    r_kv,
                                    krows[gather_base + part] * (QK_DIM // 2) + lane * 4,
                                    vec_width=4,
                                    dtype=T.i32,
                                )
                            )
                            for part in range(2)
                        ]
                        for gather_part in range_constexpr(2):
                            fx.ptr_store(
                                gather_kv[gather_part].bitcast(fx.Float32),
                                ktile + ((wave * KPW + gather_base + gather_part) * KS + lane * 4),
                            )
                        if lane < PE_DIM // 2:
                            gather_pe = [
                                ld_f32(r_pe, krows[gather_base + part] * (QK_DIM // 2) + KV_LORA // 2 + lane)
                                for part in range(2)
                            ]
                            for gather_part in range_constexpr(2):
                                lds_st(
                                    petile,
                                    (wave * KPW + gather_base + gather_part) * PS + lane,
                                    gather_pe[gather_part],
                                )
                else:
                    for jj in range_constexpr(KPW):
                        j = wave * KPW + jj
                        if const_expr(cache_fp8):
                            row = krows[jj] * (QK_DIM // 4)
                            raw = fx.Vector(bo.buffer_load(r_kv, row + lane * 2, vec_width=2, dtype=T.i32))
                            fx.ptr_store(
                                _fp8_to_bf16x8(raw[0], raw[1]).bitcast(fx.Float32),
                                ktile + (j * KS + lane * 4),
                            )
                            if lane < PE_DIM // 8:
                                pe_raw = fx.Vector(
                                    bo.buffer_load(
                                        r_pe,
                                        row + KV_LORA // 4 + lane * 2,
                                        vec_width=2,
                                        dtype=T.i32,
                                    )
                                )
                                fx.ptr_store(
                                    _fp8_to_bf16x8(pe_raw[0], pe_raw[1]).bitcast(fx.Float32),
                                    petile + (j * PS + lane * 4),
                                )
                        else:
                            kv_row_words = QK_DIM // 2 if const_expr(use_atom_kv_cache) else KV_LORA // 2
                            kv8 = fx.Vector(
                                bo.buffer_load(
                                    r_kv,
                                    krows[jj] * kv_row_words + lane * 4,
                                    vec_width=4,
                                    dtype=T.i32,
                                )
                            )
                            fx.ptr_store(kv8.bitcast(fx.Float32), ktile + (j * KS + lane * 4))
                            if lane < PE_DIM // 2:
                                pe_row = (
                                    krows[jj] * (QK_DIM // 2) + KV_LORA // 2 + lane
                                    if const_expr(use_atom_kv_cache)
                                    else krows[jj] * (PE_DIM // 2) + lane
                                )
                                lds_st(petile, j * PS + lane, ld_f32(r_pe, pe_row))

            def patch_new_kv():
                """Patch the cache version visible at this tile's serial boundary."""
                if const_expr(DEFER_OFF_CACHE):
                    # One coalesced incoming-slot load per lane, then one ballot
                    # per key replaces a serial row-slot search in wave zero.
                    incoming_slot = fx.Int32(
                        bo.buffer_load(_rsrc(full_slots), fx.min(lane, total_samples - 1) * 2, vec_width=1, dtype=T.i32)
                    )
                    visible_row = (lane < row_offset + S) & (lane < total_samples) & (incoming_slot >= 0)
                    for jj in range_constexpr(KPW):
                        j = wave * KPW + jj
                        physical_key = _uniform(lds_ld(attn_keys, j))
                        version_mask = fx.Int64(rocdl.ballot(T.i64, visible_row & (incoming_slot == physical_key)))
                        if version_mask != 0:
                            incoming_row = fx.Int32(63) - fx.Int32(fmath.ctlz(version_mask))
                            source_tile = incoming_row // S
                            source_row = incoming_row % S
                            source_scratch = full_scratch + fx.Int64(source_tile) * TILE_SCRATCH_BYTES
                            kvp = get2_many(
                                [
                                    (source_scratch + SC["kvnew"], source_row * KV_LORA + lane * 8 + m * 2)
                                    for m in range(4)
                                ]
                            )
                            words = [bf16_pair(a0, a1) for a0, a1 in kvp]
                            fx.ptr_store(
                                fx.Vector.from_elements(words, fx.Float32),
                                ktile + (j * KS + lane * 4),
                            )
                            if lane < PE_DIM // 2:
                                a0, a1 = get2(source_scratch + SC["penew"], source_row * PE_DIM + lane * 2)
                                lds_st(petile, j * PS + lane, bf16_pair(a0, a1))
                else:
                    if const_expr(use_atom_kv_cache):
                        new_active = [row_writes_cache(new_s) for new_s in range(S)]
                        new_slots = [row_slot(new_s) for new_s in range(S)]
                    for jj in range_constexpr(KPW):
                        j = wave * KPW + jj
                        kr = lds_ld(attn_keys, j)
                        if const_expr(use_atom_kv_cache):
                            sn = fx.Int32(-1)
                            for new_s in range_constexpr(S):
                                match = new_active[new_s] & (kr == new_slots[new_s])
                                sn = match.select(fx.Int32(new_s), sn)
                            is_new = sn >= 0
                        else:
                            is_new = kr >= pos0
                            sn = kr - pos0
                        if is_new:
                            kvp = get2_many([(mb("kvnew"), sn * KV_LORA + lane * 8 + m * 2) for m in range(4)])
                            w = [bf16_pair(a0, a1) for a0, a1 in kvp]
                            fx.ptr_store(
                                fx.Vector.from_elements(w, fx.Float32),
                                ktile + (j * KS + lane * 4),
                            )
                            if lane < PE_DIM // 2:
                                a0, a1 = get2(mb("penew"), sn * PE_DIM + lane * 2)
                                lds_st(petile, j * PS + lane, bf16_pair(a0, a1))

            N_HEAD_GROUPS = 1
            for tt in range(start("split"), S * N_HEAD_GROUPS * N_SPLIT, G):
                tt = fx.Int32(tt)
                s = tt // (N_HEAD_GROUPS * N_SPLIT)
                head_group = (tt // N_SPLIT) % N_HEAD_GROUPS
                t = tt % N_SPLIT
                h = wave
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
                                (s * H + (k - N_UK) // N_PE_T) * PE_DIM
                                + ((k - N_UK) % N_PE_T) * Q_B_TILE
                                + Q_B_TILE
                                - 1,
                                s * PE_DIM + PE_DIM - 1,
                            ),
                        ),
                    ),
                )
                # q of all heads -> bf16 Q[h][576] (words h * 288 + d / 2): latent 512 then pe 64
                NQ = H * KV_LORA // 4 // THREADS
                tpe = fx.min(tid, H * PE_DIM // 4 - 1)
                qv = poll(
                    [
                        (
                            mb("q_lat"),
                            (s * H * KV_LORA + (tid + i * THREADS) * 4) // 2,
                            2,
                        )
                        for i in range(NQ)
                    ]
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
                gpu.barrier()
                # scores = K Q^T on MFMA.  The 64-key tile maps the eight waves to
                # (four row groups, two K halves); the batch-8 32-key tile uses two
                # row groups and four waves per K half.  All waves subsequently own
                # one attention head for softmax and P@V.
                hn = lane % 16
                rgk = wave % (SPLIT_KEYS // 16)
                c = fx.full(4, 0.0, fx.Float32)
                for st in range_constexpr(QK_DIM // 32 // 2):
                    kst = ((wave // (SPLIT_KEYS // 16)) % 2) * (QK_DIM // 32 // 2) + st
                    key = rgk * 16 + lane % 16
                    kw = (kst < KV_LORA // 32).select(
                        KT_OFF + key * KS + kst * 16,
                        PT_OFF + key * PS + (kst - KV_LORA // 32) * 16,
                    )
                    a = fx.ptr_load(xs + (kw + (lane // 16) * 4), result_type=v4f).bitcast(fx.BFloat16)
                    b = fx.ptr_load(
                        xs + (hn * QS + kst * 16 + (lane // 16) * 4),
                        result_type=v4f,
                    ).bitcast(fx.BFloat16)
                    c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                if const_expr(SPLIT_KEYS == 64) or wave < 4:
                    fx.ptr_store(c, red + (wave * 64 + lane) * 4)
                gpu.barrier()
                # split-local softmax: wave h, lane = key j (score = sum of the two K halves)
                kidx = t * SPLIT_KEYS + lane
                valid = (lane < SPLIT_KEYS) & (kidx < nkeys)
                r16 = lane % 16

                def softmax_one(softmax_head):
                    cl = split_score_column(softmax_head, lane)
                    key_rg = fx.min(lane // 16, SPLIT_KEYS // 16 - 1)
                    half_stride = SPLIT_KEYS // 16
                    raw = lds_ld(red, (key_rg * 64 + cl) * 4 + r16 % 4) + lds_ld(
                        red, ((key_rg + half_stride) * 64 + cl) * 4 + r16 % 4
                    )
                    sc_v = valid.select(raw * scale, fx.Float32(NEG))
                    m = wave_max(sc_v)
                    p = valid.select(_exp(sc_v - m), fx.Float32(0.0))
                    lsum = wave_sum(p)
                    p_n = _xshfl(p, 1)
                    if (lane < SPLIT_KEYS) & (lane % 2 == 0):
                        lds_st(
                            pl,
                            softmax_head * (SPLIT_KEYS // 2) + lane // 2,
                            bf16_pair(p, p_n),
                        )
                    return m, lsum

                m0, l0 = softmax_one(wave)
                m1, l1 = softmax_one(wave + WAVES)
                gpu.barrier()
                # O = P V on MFMA: heads M, keys K, latent dims N.  Each V word holds
                # a dim pair (even dim low), so one read feeds two MFMAs (even / odd dims):
                # each wave owns 2 groups of 32 dims.  V is read key-strided from the tile.
                for g in range_constexpr(KV_LORA // 32 // WAVES):
                    dw = (wave * (KV_LORA // 32 // WAVES) + g) * 16 + lane % 16  # dim pair word
                    c0 = fx.full(4, 0.0, fx.Float32)
                    c1 = fx.full(4, 0.0, fx.Float32)
                    for js in range_constexpr(SPLIT_KEYS // 32):
                        a = fx.ptr_load(
                            pl + (hn * (SPLIT_KEYS // 2) + js * 16 + (lane // 16) * 4),
                            result_type=v4f,
                        ).bitcast(fx.BFloat16)
                        ws = [
                            fx.ptr_load(ktile + ((js * 32 + (lane // 16) * 8 + i) * KS + dw)).bitcast(fx.Int32)
                            for i in range(8)
                        ]
                        w_lo = [(ws[2 * i] & 0xFFFF) | (ws[2 * i + 1] << 16) for i in range(4)]
                        w_hi = [fx.Int32(fx.Uint32(ws[2 * i]) >> 16) | (ws[2 * i + 1] & -65536) for i in range(4)]
                        b0 = fx.Vector.from_elements(w_lo, fx.Int32).bitcast(fx.BFloat16)
                        b1 = fx.Vector.from_elements(w_hi, fx.Int32).bitcast(fx.BFloat16)
                        c0 = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b0, c0]))
                        c1 = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b1, c1]))
                    if lane < 64:
                        for e in range_constexpr(4):
                            hh = (lane // 16) * 4 + e
                            put_bf(
                                mb("sp_acc"),
                                ((s * N_SPLIT + t) * H + hh) * KV_LORA + dw * 2,
                                [c0[e], c1[e]],
                            )
                if lane == 0:  # written last: the merge's readiness hint
                    put(mb("sp_m"), (s * N_SPLIT + t) * H + h, m0)
                    put(mb("sp_m"), (s * N_SPLIT + t) * H + h + WAVES, m1)
                    put(mb("sp_l"), (s * N_SPLIT + t) * H + h, l0)
                    put(mb("sp_l"), (s * N_SPLIT + t) * H + h + WAVES, l1)
        else:
            r_kv = _rsrc(kv_cache)
            r_pe = _rsrc(pe_cache)
            r_idx = _rsrc(indices)
            KPW = SPLIT_GATHER_KEYS // WAVES

            def split_keys(t, s):
                """(nkeys, sparse) of sample s; wave 0 writes this split's cache rows to LDS."""
                if const_expr(use_atom_kv_cache):
                    index_base, index_end = row_index_bounds(s)
                    nkeys = index_end - index_base
                    sparse = index_end > index_base
                else:
                    index_base = s * topk
                    kv_len = pos0 + s + 1
                    sparse = kv_len > topk
                    nkeys = sparse.select(fx.Int32(topk), kv_len)
                if const_expr(with_indexer):
                    if sparse:
                        if wave == 0:
                            if lane == 0:
                                get(mb("indices_ready"), s)
                            # Only wave 0 consumes the compact index payload; its
                            # divergent ready poll reconverges before this acquire.
                            fx.memory_fence(
                                ordering=fx.AtomicOrdering.Acquire,
                                syncscope="agent",
                            )
                if wave == 0:
                    if lane < SPLIT_GATHER_KEYS:
                        k_pos = t * SPLIT_GATHER_KEYS + lane
                        k_cl = (k_pos < nkeys).select(k_pos, 0)
                        if const_expr(with_indexer):
                            idx = k_cl
                            if sparse:
                                idx = fx.Int32(
                                    bo.buffer_load(
                                        _rsrc(mb("indices")),
                                        s * topk + k_cl,
                                        vec_width=1,
                                        dtype=T.i32,
                                        cache_modifier=CM_DEV,
                                    )
                                )
                            lds_st(attn_keys, lane, index_page_slot(s, idx))
                        elif const_expr(use_atom_kv_cache):
                            idx = fx.Int32(0)
                            if sparse:
                                idx = fx.Int32(
                                    bo.buffer_load(
                                        r_idx,
                                        index_base + k_cl,
                                        vec_width=1,
                                        dtype=T.i32,
                                    )
                                )
                            lds_st(attn_keys, lane, idx)
                        else:
                            idx = fx.Int32(
                                bo.buffer_load(
                                    r_idx,
                                    index_base + k_cl,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            )
                            lds_st(attn_keys, lane, sparse.select(idx, k_cl))
                return nkeys, sparse

            def gather_old_kv():
                """Copy persistent KV/PE rows to LDS before patching incoming cache versions."""
                krows = [lds_ld(attn_keys, wave * KPW + jj) for jj in range(KPW)]
                if const_expr(use_atom_kv_cache):
                    # Each gathered row is wave-uniform; retain two independent
                    # row loads while putting its full byte address in the base.
                    cache_row_bytes = QK_DIM * (1 if cache_fp8 else 2)
                    for gather_base in range_constexpr(0, KPW, 2):
                        descriptors = [
                            _rsrc(kv_cache + fx.Int64(_uniform(krows[gather_base + part])) * cache_row_bytes)
                            for part in range(min(2, KPW - gather_base))
                        ]
                        raw_values = [
                            fx.Vector(
                                bo.buffer_load(
                                    descriptors[part],
                                    lane * (2 if cache_fp8 else 4),
                                    vec_width=2 if cache_fp8 else 4,
                                    dtype=T.i32,
                                )
                            )
                            for part in range(min(2, KPW - gather_base))
                        ]
                        for part in range_constexpr(min(2, KPW - gather_base)):
                            raw = raw_values[part]
                            value = _fp8_to_bf16x8(raw[0], raw[1]) if const_expr(cache_fp8) else raw
                            fx.ptr_store(
                                value.bitcast(fx.Float32),
                                ktile + ((wave * KPW + gather_base + part) * KS + lane * 4),
                            )
                        if lane < (PE_DIM // 8 if const_expr(cache_fp8) else PE_DIM // 2):
                            pe_values = [
                                bo.buffer_load(
                                    _rsrc(pe_cache + fx.Int64(_uniform(krows[gather_base + part])) * cache_row_bytes),
                                    KV_LORA // (4 if cache_fp8 else 2) + lane * (2 if cache_fp8 else 1),
                                    vec_width=2 if cache_fp8 else 1,
                                    dtype=T.i32,
                                )
                                for part in range(min(2, KPW - gather_base))
                            ]
                            for part in range_constexpr(min(2, KPW - gather_base)):
                                if const_expr(cache_fp8):
                                    raw = fx.Vector(pe_values[part])
                                    fx.ptr_store(
                                        _fp8_to_bf16x8(raw[0], raw[1]).bitcast(fx.Float32),
                                        petile + ((wave * KPW + gather_base + part) * PS + lane * 4),
                                    )
                                else:
                                    lds_st(
                                        petile,
                                        (wave * KPW + gather_base + part) * PS + lane,
                                        fx.Int32(pe_values[part]).bitcast(fx.Float32),
                                    )
                elif const_expr(cache_fp8 and KPW % 2 == 0):
                    # Keep two independent rows outstanding before conversion.
                    # The complete gather still precedes q/patch dependencies.
                    for gather_base in range_constexpr(0, KPW, 2):
                        gather_kv = [
                            fx.Vector(
                                bo.buffer_load(
                                    r_kv,
                                    krows[gather_base + part] * (QK_DIM // 4) + lane * 2,
                                    vec_width=2,
                                    dtype=T.i32,
                                )
                            )
                            for part in range(2)
                        ]
                        for part in range_constexpr(2):
                            raw = gather_kv[part]
                            fx.ptr_store(
                                _fp8_to_bf16x8(raw[0], raw[1]).bitcast(fx.Float32),
                                ktile + ((wave * KPW + gather_base + part) * KS + lane * 4),
                            )
                        if lane < PE_DIM // 8:
                            gather_pe = [
                                fx.Vector(
                                    bo.buffer_load(
                                        r_pe,
                                        krows[gather_base + part] * (QK_DIM // 4) + KV_LORA // 4 + lane * 2,
                                        vec_width=2,
                                        dtype=T.i32,
                                    )
                                )
                                for part in range(2)
                            ]
                            for part in range_constexpr(2):
                                raw = gather_pe[part]
                                fx.ptr_store(
                                    _fp8_to_bf16x8(raw[0], raw[1]).bitcast(fx.Float32),
                                    petile + ((wave * KPW + gather_base + part) * PS + lane * 4),
                                )
                else:
                    for jj in range_constexpr(KPW):
                        j = wave * KPW + jj
                        if const_expr(cache_fp8):
                            row = krows[jj] * (QK_DIM // 4)
                            raw = fx.Vector(bo.buffer_load(r_kv, row + lane * 2, vec_width=2, dtype=T.i32))
                            fx.ptr_store(
                                _fp8_to_bf16x8(raw[0], raw[1]).bitcast(fx.Float32),
                                ktile + (j * KS + lane * 4),
                            )
                            if lane < PE_DIM // 8:
                                pe_raw = fx.Vector(
                                    bo.buffer_load(
                                        r_pe,
                                        row + KV_LORA // 4 + lane * 2,
                                        vec_width=2,
                                        dtype=T.i32,
                                    )
                                )
                                fx.ptr_store(
                                    _fp8_to_bf16x8(pe_raw[0], pe_raw[1]).bitcast(fx.Float32),
                                    petile + (j * PS + lane * 4),
                                )
                        else:
                            kv_row_words = QK_DIM // 2 if const_expr(use_atom_kv_cache) else KV_LORA // 2
                            kv8 = fx.Vector(
                                bo.buffer_load(
                                    r_kv,
                                    krows[jj] * kv_row_words + lane * 4,
                                    vec_width=4,
                                    dtype=T.i32,
                                )
                            )
                            fx.ptr_store(kv8.bitcast(fx.Float32), ktile + (j * KS + lane * 4))
                            if lane < PE_DIM // 2:
                                pe_row = (
                                    krows[jj] * (QK_DIM // 2) + KV_LORA // 2 + lane
                                    if const_expr(use_atom_kv_cache)
                                    else krows[jj] * (PE_DIM // 2) + lane
                                )
                                lds_st(petile, j * PS + lane, ld_f32(r_pe, pe_row))

            def patch_new_kv():
                """Patch the cache version visible at this tile's serial boundary."""
                if const_expr(DEFER_OFF_CACHE):
                    # One coalesced incoming-slot load per lane, then one ballot
                    # per key replaces a serial row-slot search in wave zero.
                    incoming_slot = fx.Int32(
                        bo.buffer_load(_rsrc(full_slots), fx.min(lane, total_samples - 1) * 2, vec_width=1, dtype=T.i32)
                    )
                    visible_row = (lane < row_offset + S) & (lane < total_samples) & (incoming_slot >= 0)
                    for jj in range_constexpr(KPW):
                        j = wave * KPW + jj
                        physical_key = _uniform(lds_ld(attn_keys, j))
                        version_mask = fx.Int64(rocdl.ballot(T.i64, visible_row & (incoming_slot == physical_key)))
                        if version_mask != 0:
                            incoming_row = fx.Int32(63) - fx.Int32(fmath.ctlz(version_mask))
                            source_tile = incoming_row // S
                            source_row = incoming_row % S
                            source_scratch = full_scratch + fx.Int64(source_tile) * TILE_SCRATCH_BYTES
                            kvp = get2_many(
                                [
                                    (source_scratch + SC["kvnew"], source_row * KV_LORA + lane * 8 + m * 2)
                                    for m in range(4)
                                ]
                            )
                            words = [bf16_pair(a0, a1) for a0, a1 in kvp]
                            fx.ptr_store(
                                fx.Vector.from_elements(words, fx.Float32),
                                ktile + (j * KS + lane * 4),
                            )
                            if lane < PE_DIM // 2:
                                a0, a1 = get2(source_scratch + SC["penew"], source_row * PE_DIM + lane * 2)
                                lds_st(petile, j * PS + lane, bf16_pair(a0, a1))
                else:
                    if const_expr(use_atom_kv_cache):
                        new_active = [row_writes_cache(new_s) for new_s in range(S)]
                        new_slots = [row_slot(new_s) for new_s in range(S)]
                    for jj in range_constexpr(KPW):
                        j = wave * KPW + jj
                        kr = lds_ld(attn_keys, j)
                        if const_expr(use_atom_kv_cache):
                            sn = fx.Int32(-1)
                            for new_s in range_constexpr(S):
                                match = new_active[new_s] & (kr == new_slots[new_s])
                                sn = match.select(fx.Int32(new_s), sn)
                            is_new = sn >= 0
                        else:
                            is_new = kr >= pos0
                            sn = kr - pos0
                        if is_new:
                            kvp = get2_many([(mb("kvnew"), sn * KV_LORA + lane * 8 + m * 2) for m in range(4)])
                            w = [bf16_pair(a0, a1) for a0, a1 in kvp]
                            fx.ptr_store(
                                fx.Vector.from_elements(w, fx.Float32),
                                ktile + (j * KS + lane * 4),
                            )
                            if lane < PE_DIM // 2:
                                a0, a1 = get2(mb("penew"), sn * PE_DIM + lane * 2)
                                lds_st(petile, j * PS + lane, bf16_pair(a0, a1))

            N_HEAD_GROUPS = H // WAVES
            for tt in range(start("split"), S * N_HEAD_GROUPS * N_SPLIT_TASKS, G):
                tt = fx.Int32(tt)
                s = tt // (N_HEAD_GROUPS * N_SPLIT_TASKS)
                head_group = (tt // N_SPLIT_TASKS) % N_HEAD_GROUPS
                t = tt % N_SPLIT_TASKS
                h = head_group * WAVES + wave
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
                                (s * H + (k - N_UK) // N_PE_T) * PE_DIM
                                + ((k - N_UK) % N_PE_T) * Q_B_TILE
                                + Q_B_TILE
                                - 1,
                                s * PE_DIM + PE_DIM - 1,
                            ),
                        ),
                    ),
                )
                # q of all heads -> bf16 Q[h][576] (words h * 288 + d / 2): latent 512 then pe 64
                NQ = H * KV_LORA // 4 // THREADS
                tpe = fx.min(tid, H * PE_DIM // 4 - 1)
                qv = poll(
                    [
                        (
                            mb("q_lat"),
                            (s * H * KV_LORA + (tid + i * THREADS) * 4) // 2,
                            2,
                        )
                        for i in range(NQ)
                    ]
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
                gpu.barrier()
                # scores = K Q^T on MFMA.  The 64-key tile maps the eight waves to
                # (four row groups, two K halves); the batch-8 32-key tile uses two
                # row groups and four waves per K half.  All waves subsequently own
                # one attention head for softmax and P@V.
                hn = head_group * WAVES + fx.min(lane % 16, WAVES - 1)
                rgk = wave % (SPLIT_KEYS // 16)
                c = fx.full(4, 0.0, fx.Float32)
                for st in range_constexpr(QK_DIM // 32 // 2):
                    kst = ((wave // (SPLIT_KEYS // 16)) % 2) * (QK_DIM // 32 // 2) + st
                    key = rgk * 16 + lane % 16 + (wave // 4) * SPLIT_KEYS * (SPLIT_TASK_PAIR - 1)
                    kw = (kst < KV_LORA // 32).select(
                        KT_OFF + key * KS + kst * 16,
                        PT_OFF + key * PS + (kst - KV_LORA // 32) * 16,
                    )
                    a = fx.ptr_load(xs + (kw + (lane // 16) * 4), result_type=v4f).bitcast(fx.BFloat16)
                    b = fx.ptr_load(
                        xs + (hn * QS + kst * 16 + (lane // 16) * 4),
                        result_type=v4f,
                    ).bitcast(fx.BFloat16)
                    c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                if const_expr(SPLIT_KEYS == 64 or SPLIT_TASK_PAIR == 2) or wave < 4:
                    fx.ptr_store(c, red + (wave * 64 + lane) * 4)
                gpu.barrier()
                for split_part in range_constexpr(SPLIT_TASK_PAIR):
                    # split-local softmax: wave h, lane = key j (score = sum of the two K halves)
                    logical_split = t * SPLIT_TASK_PAIR + split_part
                    kidx = logical_split * SPLIT_KEYS + lane
                    valid = (lane < SPLIT_KEYS) & (kidx < nkeys)
                    r16 = lane % 16
                    cl = split_score_column(wave, lane)
                    key_rg = fx.min(lane // 16, SPLIT_KEYS // 16 - 1)
                    half_stride = SPLIT_KEYS // 16
                    raw = lds_ld(red, ((key_rg + split_part * 4) * 64 + cl) * 4 + r16 % 4) + lds_ld(
                        red, ((key_rg + half_stride + split_part * 4) * 64 + cl) * 4 + r16 % 4
                    )
                    sc_v = valid.select(raw * scale, fx.Float32(NEG))
                    m = wave_max(sc_v)
                    p = valid.select(_exp(sc_v - m), fx.Float32(0.0))
                    lsum = wave_sum(p)
                    p_n = _xshfl(p, 1)
                    if (lane < SPLIT_KEYS) & (lane % 2 == 0):
                        lds_st(pl, h * (SPLIT_KEYS // 2) + lane // 2, bf16_pair(p, p_n))
                    gpu.barrier()
                    # O = P V on MFMA: heads M, keys K, latent dims N.  Each V word holds
                    # a dim pair (even dim low), so one read feeds two MFMAs (even / odd dims):
                    # each wave owns 2 groups of 32 dims.  V is read key-strided from the tile.
                    for g in range_constexpr(KV_LORA // 32 // WAVES):
                        dw = (wave * (KV_LORA // 32 // WAVES) + g) * 16 + lane % 16  # dim pair word
                        c0 = fx.full(4, 0.0, fx.Float32)
                        c1 = fx.full(4, 0.0, fx.Float32)
                        for js in range_constexpr(SPLIT_KEYS // 32):
                            a = fx.ptr_load(
                                pl + (hn * (SPLIT_KEYS // 2) + js * 16 + (lane // 16) * 4),
                                result_type=v4f,
                            ).bitcast(fx.BFloat16)
                            ws = [
                                fx.ptr_load(
                                    ktile + ((split_part * SPLIT_KEYS + js * 32 + (lane // 16) * 8 + i) * KS + dw)
                                ).bitcast(fx.Int32)
                                for i in range(8)
                            ]
                            w_lo = [(ws[2 * i] & 0xFFFF) | (ws[2 * i + 1] << 16) for i in range(4)]
                            w_hi = [fx.Int32(fx.Uint32(ws[2 * i]) >> 16) | (ws[2 * i + 1] & -65536) for i in range(4)]
                            b0 = fx.Vector.from_elements(w_lo, fx.Int32).bitcast(fx.BFloat16)
                            b1 = fx.Vector.from_elements(w_hi, fx.Int32).bitcast(fx.BFloat16)
                            c0 = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b0, c0]))
                            c1 = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b1, c1]))
                        if lane < 32:
                            for e in range_constexpr(4):
                                hh = split_acc_head(head_group, lane // 16, e)
                                put_bf(
                                    mb("sp_acc"),
                                    ((s * N_SPLIT + logical_split) * H + hh) * KV_LORA + dw * 2,
                                    [c0[e], c1[e]],
                                )
                    if lane == 0:  # written last: the merge's readiness hint
                        put(mb("sp_m"), (s * N_SPLIT + logical_split) * H + h, m)
                        put(mb("sp_l"), (s * N_SPLIT + logical_split) * H + h, lsum)
                    if const_expr(split_part + 1 < SPLIT_TASK_PAIR):
                        gpu.barrier()

    def merge():
        if const_expr(FRONT_PAYLOAD_RETIRE):
            # Each global QKV task reads all rows of one 16-row group.
            # Its q_a/kv_a stores follow the last operand read, MFMA and
            # a CTA barrier. One current-epoch pair per task therefore
            # certifies that task has retired all input-norm readers.
            # A cohort can intersect two groups; cover both explicitly.
            def retire_input_group(representative):
                if tid < N_QKV_A:
                    retire_scratch = full_scratch + fx.Int64(representative // S) * TILE_SCRATCH_BYTES
                    retire_element = tid * QKV_A_TILE
                    if retire_element < Q_LORA:
                        get(retire_scratch + SC["q_a"], (representative % S) * Q_LORA + retire_element)
                    else:
                        get(
                            retire_scratch + SC["kv_a"],
                            (representative % S) * (KV_LORA + PE_DIM) + retire_element - Q_LORA,
                        )
                # All retirement-poll waves finish before any UV writer.
                gpu.barrier()

            retire_first = (row_offset // 16) * 16
            retire_last = ((row_offset + S - 1) // 16) * 16
            retire_input_group(retire_first)
            if retire_last != retire_first:
                retire_input_group(retire_last)

        # ========================== 6. split merge + W_UV: o = W_UV (softmax . KV)
        # 4 row groups x 8 chunks: 2 waves per row group, 4 chunks each
        r_wuv, r_suv = _rsrc(w_uv), _rsrc(s_uv)
        UV_NKC = KV_LORA // 64
        UV_R = UV_TILE // 16
        UV_WPR = WAVES // UV_R
        UV_UNITS = UV_NKC // (attention_k_chunks_per_unit * UV_WPR)
        if const_expr(GLOBAL_UV):
            # Retired input-norm payloads already have the ordinary epoch.
            # A separate phase bit prevents UV from accepting them as ready.
            uv_tag = tag ^ fx.Int32(-2147483648)

        def uv_merged_pair(base_addr, i):
            def load_pair():
                return fx.Vector(
                    bo.buffer_load(_rsrc(base_addr), i * 2, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV)
                )

            words = load_pair()
            while (words[1] != uv_tag) | (words[3] != uv_tag):
                spin_pause()
                words = load_pair()
            return words[0], words[2]

        for tt in range(start("uv"), S * N_UV // UV_REUSE, G):
            tt = fx.Int32(tt)
            s = tt // (N_UV // UV_REUSE)
            t = (tt % (N_UV // UV_REUSE)) * UV_REUSE
            local_t = dcp_local_uv_tile(t, L)
            head = t // (V_DIM // UV_TILE)

            def u_uv_at(c, uv_local_t):
                kc = ((wave % UV_WPR) * UV_UNITS + c) * attention_k_chunks_per_unit
                return unit_attention(
                    r_wuv,
                    r_suv,
                    uv_local_t * UV_R + wave // UV_WPR,
                    kc,
                    UV_NKC,
                    KV_LORA,
                    128,
                    attention_b_word(kc * 64),
                    scale_rows=uv_scale_rows,
                    coef=((lambda: lds_ld(misc, N_SPLIT + 2 + kc // 2)) if const_expr(attention_ptpc) else None),
                )

            if const_expr(not GLOBAL_UV):
                pre = [u_uv_at(c, local_t) for c in range(UV_UNITS)]
            hint_wait(
                N_SPLIT,
                lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head),
            )
            pre_poll(N_SPLIT, lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head))
            # Four 16-split quarters keep each poll and its live payload bounded.
            # Every thread owns one (quarter, half-of-latent) pair and processes
            # its two latent-pair positions sequentially; red holds quarter sums.
            SPH = N_SPLIT // 4
            dp_lo = tid % (KV_LORA // 4)
            hf = tid // (KV_LORA // 4)
            spi = fx.min(lane, N_SPLIT - 1)
            ml = (s * N_SPLIT + spi) * H + head
            ml_got = poll([(mb("sp_m"), ml, 1), (mb("sp_l"), ml, 1)])
            if wave == 0:  # per-split weights exp(m - M) / L for this head -> misc[sp]
                ok_sp = lane < N_SPLIT
                m_sp = ok_sp.select(ml_got[0][0].bitcast(fx.Float32), fx.Float32(NEG))
                l_sp = ok_sp.select(ml_got[1][0].bitcast(fx.Float32), fx.Float32(0.0))
                m_all = wave_max(m_sp)
                w_sp = _exp(m_sp - m_all)
                den = wave_sum(l_sp * w_sp)
                if ok_sp:
                    lds_st(
                        misc,
                        lane,
                        w_sp * (den > 0.0).select(_rcp(den), fx.Float32(0.0)),
                    )
                if lane == 0:
                    lds_st(misc, N_SPLIT, m_all)
                    lds_st(misc, N_SPLIT + 1, den)
            gpu.barrier()
            for dh in range_constexpr(2):
                dp = dp_lo + dh * (KV_LORA // 4)
                got = poll(
                    [
                        (
                            mb("sp_acc"),
                            ((s * N_SPLIT + hf * SPH + j) * H + head) * (KV_LORA // 2) + dp,
                            1,
                        )
                        for j in range(SPH)
                    ]
                )
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
            uv_valid = tid < KV_LORA // 2
            uv0, uv1 = fx.Float32(0.0), fx.Float32(0.0)
            if uv_valid:
                o0 = fx.Float32(0.0)
                o1 = fx.Float32(0.0)
                for q in range_constexpr(4):
                    o0 = o0 + lds_ld(red, (q * (KV_LORA // 2) + tid) * 2)
                    o1 = o1 + lds_ld(red, (q * (KV_LORA // 2) + tid) * 2 + 1)
                uv0, uv1 = bf16_round(o0), bf16_round(o1)
                if const_expr(not attention_ptpc):
                    lds_st(xs, tid, bf16_pair(uv0, uv1))
            if const_expr(attention_ptpc):
                uv_max = wave_max(uv_valid.select(fx.max(fmath.absf(uv0), fmath.absf(uv1)), fx.Float32(0.0)))
                if (lane == 0) & (wave < KV_LORA // 128):
                    lds_st(
                        misc,
                        N_SPLIT + 2 + wave,
                        (uv_max == 0.0).select(fx.Float32(1.0), uv_max * (1.0 / FP8_MAX)),
                    )
            gpu.barrier()
            if const_expr(attention_ptpc):
                if uv_valid:
                    uv_scale = lds_ld(misc, N_SPLIT + 2 + tid // 64)
                    uv_rcp = 1.0 / uv_scale
                    st_ptpc(
                        tid * 2,
                        div_rn(uv0, uv_scale, uv_rcp),
                        div_rn(uv1, uv_scale, uv_rcp),
                    )
                gpu.barrier()
            if const_expr(GLOBAL_UV):
                # This range is disjoint from packed mid values/scales.
                # Input-norm readers retire before merge through the grid join
                # or QKV completion polls. The distinct UV phase tag prevents consumers from
                # accepting those old payloads before these stores complete.
                if uv_valid:
                    uv_packed = bf16_pair(uv0, uv1).bitcast(fx.Int32)
                    bo.buffer_store(
                        fx.Vector.from_elements([uv_packed, uv_tag], fx.Int32),
                        _rsrc(mb("ugp")),
                        (UV_MERGE_OFF + (s * H + head) * (KV_LORA // 2) + tid) * 2,
                        cache_modifier=CM_DEV,
                    )
                gpu.barrier()
            else:
                owner = dcp_merge_latent(s, t)
                if owner:
                    for uv_pass in range_constexpr(UV_REUSE):
                        local_t = dcp_local_uv_tile(t + uv_pass, L)
                        if const_expr(uv_pass > 0):
                            pre = [u_uv_at(c, local_t) for c in range(UV_UNITS)]
                        acc = run_units(lambda c: u_uv_at(c, local_t), UV_UNITS, UV_UNITS, pre)
                        if const_expr(attention_ptpc):
                            reduce_rows(UV_R, acc, emit_ptpc_scalar(UV_TILE, ld_f32(r_suv, 0)), count=1)
                        else:
                            reduce_rows(UV_R, acc, emit_out(UV_TILE), count=1)
                        gpu.barrier()
                        if tid < UV_TILE // 4:
                            r = tid * 4
                            put_bf(
                                mb("o"),
                                s * O_K + local_t * UV_TILE + r,
                                [lds_ld(outs, r + j) for j in range(4)],
                            )
                        # All output readers finish before the next pass reuses outs.
                        if const_expr(UV_REUSE > 1):
                            gpu.barrier()

        if const_expr(GLOBAL_UV):
            # Every CTA has finished its merge producers before waiting here.
            # Timeline columns 6/7 use the first 64 slots/cohort for 192 global
            # projector tasks; columns0..5 remain the original merge tasks.
            if full_bid < ((total_samples + 15) // 16) * N_UV:
                uv_task = full_bid
                uv_first = uv_task // N_UV * 16
                uv_count = fx.min(fx.Int32(16), total_samples - uv_first)
                uv_t = uv_task % N_UV
                uv_head = uv_t // (V_DIM // UV_TILE)

                def global_uv_unit(c):
                    kc = ((wave % UV_WPR) * UV_UNITS + c) * attention_k_chunks_per_unit
                    return unit_attention(
                        r_wuv,
                        r_suv,
                        uv_t * UV_R + wave // UV_WPR,
                        kc,
                        UV_NKC,
                        KV_LORA,
                        128,
                        fx.min(lane % 16, uv_count - 1) * 260 + kc * 32,
                        scale_rows=uv_scale_rows,
                    )

                uv_pre = [global_uv_unit(c) for c in range(UV_UNITS)]
                for uv_batch in range_constexpr(0, 16, 4):
                    if uv_batch < uv_count:
                        uv_local = uv_batch + tid // 128
                        uv_row = uv_first + uv_local
                        uv_src = full_scratch + fx.Int64(uv_row // S) * TILE_SCRATCH_BYTES
                        uv_words = uv_merged_pair(
                            uv_src + SC["ugp"],
                            UV_MERGE_OFF + ((uv_row % S) * H + uv_head) * (KV_LORA // 2) + (tid % 128) * 2,
                        )
                        lds_st(xs, uv_local * 260 + (tid % 128) * 2, uv_words[0].bitcast(fx.Float32))
                        lds_st(xs, uv_local * 260 + (tid % 128) * 2 + 1, uv_words[1].bitcast(fx.Float32))
                gpu.barrier()
                uv_acc = run_units(global_uv_unit, UV_UNITS, UV_UNITS, uv_pre)
                fx.ptr_store(fx.Vector.from_elements(uv_acc, fx.Float32), red + (wave * 64 + lane) * 4)
                gpu.barrier()
                if tid < uv_count * UV_TILE // 2:
                    uv_local = tid // (UV_TILE // 2)
                    uv_r = (tid % (UV_TILE // 2)) * 2
                    uv_v0, uv_v1 = fx.Float32(0.0), fx.Float32(0.0)
                    for uv_w in range_constexpr(UV_WPR):
                        uv_ww = (uv_r // 16) * UV_WPR + uv_w
                        uv_address = (uv_ww * 64 + uv_local + 16 * ((uv_r % 16) // 4)) * 4 + uv_r % 4
                        uv_v0 = uv_v0 + lds_ld(red, uv_address)
                        uv_v1 = uv_v1 + lds_ld(red, uv_address + 1)
                    uv_row = uv_first + uv_local
                    uv_dst = full_scratch + fx.Int64(uv_row // S) * TILE_SCRATCH_BYTES
                    put_bf(uv_dst + SC["o"], (uv_row % S) * O_K + uv_t * UV_TILE + uv_r, [uv_v0, uv_v1])
                gpu.barrier()

    def output():
        # ====================== 7. W_o + attention TP peer reduce + residual -> a
        # 2 row groups x 32 chunks: 4 waves per row group, 8 chunks each
        r_wo, r_so = _rsrc(w_o), _rsrc(s_o)
        O_NKC = O_K // 64
        O_R = ROW_TILE // 16
        O_WPR = WAVES // O_R
        O_UNITS = O_NKC // (attention_k_chunks_per_unit * O_WPR)
        for t in range(start("o"), N_ROW_TILES, G):
            t = fx.Int32(t)

            def u_o(c):
                kc = ((wave % O_WPR) * O_UNITS + c) * attention_k_chunks_per_unit
                return unit_attention(
                    r_wo,
                    r_so,
                    t * O_R + wave // O_WPR,
                    kc,
                    O_NKC,
                    O_K,
                    128,
                    (
                        (n_sel() * (O_K // 2 + 4) + kc * 32)
                        if const_expr(O_PAD4)
                        else attention_b_word(n_sel() * O_K + kc * 64)
                    ),
                )

            pre = [u_o(c) for c in range(O_UNITS)]
            hint_wait(
                S * N_UV,
                lambda k: (
                    mb("o"),
                    (k // N_UV) * O_K + (k % N_UV) * UV_TILE + UV_TILE - 1,
                ),
            )
            if const_expr(attention_ptpc):
                o_scales = stage_x_pairs_ptpc("o", S, O_K, lambda k: k)
            else:
                stage_x_pairs("o", S * O_K, lambda k: k, padded_row=(O_K if O_PAD4 else 0))
            gpu.barrier()
            acc = run_units(u_o, O_UNITS, O_UNITS, pre)
            if const_expr(attention_ptpc):
                reduce_rows(O_R, acc, emit_ptpc(ROW_TILE, t * ROW_TILE, r_so, o_scales))
            else:
                reduce_rows(O_R, acc, emit_out(ROW_TILE))
            gpu.barrier()

            def resid_h(s, row):
                w = fx.Vector.from_elements(
                    [
                        bo.buffer_load(
                            r_h,
                            (s * HIDDEN + row) // 2,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    ],
                    fx.Int32,
                )
                v = w.bitcast(fx.BFloat16).to(fx.Float32)
                return v[0], v[1]

            if const_expr(DN_UNION_CANDIDATE and index_request_width == 6 and not indexer_cp):
                o_owner_reduce(
                    t,
                    resid_h,
                    lambda s, row, v0, v1: put_bf(mb("a"), s * HIDDEN + row, [v0, v1]),
                )
            else:
                peer_reduce(
                    "attn",
                    t,
                    resid_h,
                    lambda s, row, v0, v1: put_bf(mb("a"), s * HIDDEN + row, [v0, v1]),
                )

    return AttentionStages(sparse=sparse, merge=merge, output=output)
