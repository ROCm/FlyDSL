# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Input/QKV normalization, projections, RoPE and cache publication."""

from typing import Callable, NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.config import EPS, HIDDEN, KV_LORA, NOPE_DIM, PE_DIM, Q_LORA
from kernels.monokernel.glm.layout import (
    BLOCKS,
    INDEX_DIM,
    INDEX_HEADS,
    INDEX_Q_ROWS,
    INDEX_TILE,
    N_QKV_A,
    Q_B_TILE,
    QKV_A_TILE,
    UK_TILE,
    WAVES,
    fp8_kv_upper_pair_lane,
    fp8_pe_upper_pair_lane,
)
from kernels.monokernel.layout import CM_SYS, THREADS
from kernels.monokernel.ops import bpermute_i32
from kernels.monokernel.ops import fp8_roundtrip as _fp8_roundtrip
from kernels.monokernel.ops import rsq as _rsq
from kernels.monokernel.ops import rsrc as _rsrc
from kernels.monokernel.ops import uniform as _uniform
from kernels.monokernel.ops import xshfl as _xshfl


class PreattnStages(NamedTuple):
    input_qkv: Callable
    query_norm: Callable
    cache: Callable
    query_projection: Callable
    index_quant: Callable
    absorbed_query: Callable


@flyc.jit
def build_preattn(plan, state, tools):
    CACHE_TILE = plan.CACHE_TILE
    COHORT_ROW_TILES = plan.COHORT_ROW_TILES
    CP_RANKS = plan.CP_RANKS
    DEFER_OFF_CACHE = plan.DEFER_OFF_CACHE
    G = plan.G
    GLOBAL_QKV_A = plan.GLOBAL_QKV_A
    GLOBAL_Q_B = plan.GLOBAL_Q_B
    H = plan.H
    INDEX_BF16_OFF = plan.INDEX_BF16_OFF
    INDEX_FP4_BYTES = plan.INDEX_FP4_BYTES
    INDEX_PRE_ROPE = plan.INDEX_PRE_ROPE
    INDEX_Q_BF16_OFF = plan.INDEX_Q_BF16_OFF
    INDEX_ROT_OFF = plan.INDEX_ROT_OFF
    INPUT_NORM_OFF = plan.INPUT_NORM_OFF
    INPUT_NORM_ONCE = plan.INPUT_NORM_ONCE
    INPUT_QA_BATCH = plan.INPUT_QA_BATCH
    N_QB = plan.N_QB
    N_UK = plan.N_UK
    QB_PER_HEAD = plan.QB_PER_HEAD
    QK_DIM = plan.QK_DIM
    S = plan.S
    SAMPLE_TILE = plan.SAMPLE_TILE
    SC = plan.SC
    SHARD_INDEX_Q = plan.SHARD_INDEX_Q
    TILED_INDEXER = plan.TILED_INDEXER
    TILE_SCRATCH_BYTES = plan.TILE_SCRATCH_BYTES
    UK_PER_HEAD = plan.UK_PER_HEAD
    W = plan.W
    _partition = plan.partition
    attention_k_chunks_per_unit = plan.attention_k_chunks_per_unit
    attention_ptpc = plan.attention_ptpc
    cache_fp8 = plan.cache_fp8
    indexer_quant = plan.indexer_quant
    total_samples = plan.total_samples
    use_atom_kv_cache = plan.use_atom_kv_cache
    with_indexer = plan.with_indexer
    bid = state.bid
    full_bid = state.full_bid
    full_indptr = state.full_indptr
    full_positions = state.full_positions
    full_scratch = state.full_scratch
    full_slots = state.full_slots
    g_in = state.g_in
    g_kv = state.g_kv
    g_q = state.g_q
    indices = state.indices
    kv_cache = state.kv_cache
    lane = state.lane
    outs = state.outs
    pe_cache = state.pe_cache
    peer_dst = state.peer_dst
    r_h = state.r_h
    red = state.red
    rope_cos = state.rope_cos
    rope_sin = state.rope_sin
    row_offset = state.row_offset
    s_q_b = state.s_q_b
    s_qkv_a = state.s_qkv_a
    s_uk = state.s_uk
    sym = state.sym
    tid = state.tid
    w_q_b = state.w_q_b
    w_qkv_a = state.w_qkv_a
    w_uk = state.w_uk
    wave = state.wave
    xs = state.xs
    ld_f32 = tools.ld_f32
    ld_bf16 = tools.ld_bf16
    ld_rope = tools.ld_rope
    quant_index_pair = tools.quant_index_pair
    row_position = tools.row_position
    row_slot = tools.row_slot
    row_writes_cache = tools.row_writes_cache
    index_page_slot = tools.index_page_slot
    lds_ld = tools.lds_ld
    lds_st = tools.lds_st
    bf16_round = tools.bf16_round
    index_arg = tools.index_arg
    mb = tools.mb
    cp_local_rank = tools.cp_local_rank
    cp_peer = tools.cp_peer
    cp_buffer = tools.cp_buffer
    put = tools.put
    put2 = tools.put2
    put_bf = tools.put_bf
    bf2_f32 = tools.bf2_f32
    poll = tools.poll
    hint_wait = tools.hint_wait
    getf = tools.getf
    getf_many = tools.getf_many
    get2_many = tools.get2_many
    get2 = tools.get2
    block_sums = tools.block_sums
    block_sum = tools.block_sum
    unit_fp8x2 = tools.unit_fp8x2
    unit_bf16 = tools.unit_bf16
    unit_attention = tools.unit_attention
    mma_units = tools.mma_units
    run_units = tools.run_units
    reduce_rows = tools.reduce_rows
    emit_out = tools.emit_out
    pick = tools.pick
    attention_b_word = tools.attention_b_word
    emit_ptpc = tools.emit_ptpc
    stage_x_rmsnorm_ptpc = tools.stage_x_rmsnorm_ptpc
    stage_x_pairs_ptpc = tools.stage_x_pairs_ptpc
    stage_x_rmsnorm = tools.stage_x_rmsnorm
    load_x_rmsnorm = tools.load_x_rmsnorm
    stage_x_pairs = tools.stage_x_pairs
    start = tools.start
    n_sel = tools.n_sel

    def ld_qa(sks):
        v = get2_many([(mb("q_a"), s * Q_LORA + k + j) for s, k in sks for j in (0, 2)])
        return [list(v[2 * i]) + list(v[2 * i + 1]) for i in range(len(sks))]

    def input_qkv():
        if const_expr(INPUT_NORM_ONCE):
            # One resident CTA per token publishes the exact normalized BF16 input.
            # Reader retirement precedes UV reuse through the grid join or QKV completion polls.
            if bid < S:

                def input_norm_load(sks):
                    result = []
                    for sn, k in sks:
                        packed = fx.Vector(
                            bo.buffer_load(
                                r_h,
                                (bid * HIDDEN + k) // 2,
                                vec_width=2,
                                dtype=T.i32,
                            )
                        )
                        values = packed.bitcast(fx.BFloat16).to(fx.Float32)
                        result.append([values[j] for j in range(4)])
                    return result

                stage_x_rmsnorm(input_norm_load, HIDDEN, g_in, count=1)
                gpu.barrier()
                for input_norm_round in range_constexpr(HIDDEN // (2 * THREADS)):
                    input_norm_word = tid + input_norm_round * THREADS
                    input_norm_value = lds_ld(xs, input_norm_word).bitcast(fx.Int32)
                    put(
                        mb("ugp"),
                        INPUT_NORM_OFF + bid * (HIDDEN // 2) + input_norm_word,
                        input_norm_value,
                    )
                gpu.barrier()

        # ================================================= 1. q_a / kv_a GEMV
        # 1 row group x 96 chunks: 8 waves split K, 12 chunks each (all prefetched)
        r_wqa, r_sqa = _rsrc(w_qkv_a), _rsrc(s_qkv_a)
        QA_NKC = HIDDEN // 64
        QA_UNITS = QA_NKC // (attention_k_chunks_per_unit * WAVES)
        if const_expr(with_indexer):
            r_wik, r_sik = _rsrc(index_arg(0)), _rsrc(index_arg(1))
            r_wiw = _rsrc(index_arg(2))
        if const_expr(GLOBAL_QKV_A):
            # Assign one projection to each global task. Index K/weight
            # tasks preserve the same K partition and reduction order,
            # while avoiding three simultaneously live accumulators.
            GLOBAL_QA_TASKS = N_QKV_A + ((INDEX_DIM + INDEX_HEADS) // QKV_A_TILE if with_indexer else 0)
            if const_expr(with_indexer):
                assert INDEX_TILE == QKV_A_TILE
            for qa_task in range(full_bid, ((total_samples + 15) // 16) * GLOBAL_QA_TASKS, BLOCKS):
                qa_task = fx.Int32(qa_task)
                qa_first = qa_task // GLOBAL_QA_TASKS * 16
                qa_count = fx.min(fx.Int32(16), total_samples - qa_first)
                qa_projection = qa_task % GLOBAL_QA_TASKS

                def global_input_projection(kind, projection_tile, rw, rs):
                    qa_acc = fx.full(4, 0.0, fx.Float32)
                    for qa_base in range_constexpr(0, QA_UNITS, 2):
                        if const_expr(kind == "index_w"):
                            qa_pre = [
                                unit_bf16(
                                    rw,
                                    projection_tile,
                                    (wave * QA_UNITS + qa_base) * 2 + c,
                                    QA_NKC,
                                    fx.min(lane % 16, qa_count - 1) * 1028 + wave * 128 + c * 32,
                                )
                                for c in range(4)
                            ]
                        else:
                            qa_pre = [
                                unit_fp8x2(
                                    rw,
                                    rs,
                                    projection_tile,
                                    (wave * QA_UNITS + qa_base + c) * 2,
                                    QA_NKC,
                                    HIDDEN,
                                    fx.min(lane % 16, qa_count - 1) * 1028 + (wave * 2 + c) * 64,
                                )
                                for c in range(2)
                            ]
                        for qa_batch in range_constexpr(0, 16, 4):
                            if qa_batch < qa_count:
                                qa_specs = []
                                for qa_part in range_constexpr(4):
                                    qa_row = qa_first + qa_batch + qa_part
                                    qa_src = full_scratch + fx.Int64(qa_row // S) * TILE_SCRATCH_BYTES
                                    qa_k = (tid // 64) * QA_UNITS * 128 + qa_base * 128 + (tid % 64) * 4
                                    qa_specs.append(
                                        (
                                            qa_src + SC["ugp"],
                                            INPUT_NORM_OFF + ((qa_row % S) * HIDDEN + qa_k) // 2,
                                            2,
                                        )
                                    )
                                qa_values = poll(qa_specs)
                                for qa_part in range_constexpr(4):
                                    for qa_pair in range_constexpr(2):
                                        lds_st(
                                            xs,
                                            (qa_batch + qa_part) * 1028 + tid * 2 + qa_pair,
                                            qa_values[qa_part][qa_pair].bitcast(fx.Float32),
                                        )
                        gpu.barrier()
                        qa_acc = fx.Vector.from_elements(mma_units([qa_acc[e] for e in range(4)], qa_pre), fx.Float32)
                        gpu.barrier()
                    reduce_rows(1, [qa_acc[e] for e in range(4)], emit_out(QKV_A_TILE), count=16)
                    gpu.barrier()
                    if tid < qa_count * QKV_A_TILE:
                        qa_local = tid // QKV_A_TILE
                        qa_row = qa_first + qa_local
                        qa_dst = full_scratch + fx.Int64(qa_row // S) * TILE_SCRATCH_BYTES
                        qa_output_row = projection_tile * QKV_A_TILE + tid % QKV_A_TILE
                        qa_value = lds_ld(outs, tid)
                        if const_expr(kind == "qkv"):
                            if qa_output_row < Q_LORA:
                                put(qa_dst + SC["q_a"], (qa_row % S) * Q_LORA + qa_output_row, qa_value)
                            else:
                                put(
                                    qa_dst + SC["kv_a"],
                                    (qa_row % S) * (KV_LORA + PE_DIM) + qa_output_row - Q_LORA,
                                    qa_value,
                                )
                        else:
                            channels = INDEX_DIM if kind == "index_k" else INDEX_HEADS
                            put(qa_dst + SC[kind], (qa_row % S) * channels + qa_output_row, qa_value)

                if qa_projection < N_QKV_A:
                    global_input_projection("qkv", qa_projection, r_wqa, r_sqa)
                elif const_expr(with_indexer):
                    if qa_projection < N_QKV_A + INDEX_DIM // QKV_A_TILE:
                        global_input_projection("index_k", qa_projection - N_QKV_A, r_wik, r_sik)
                    else:
                        global_input_projection(
                            "index_w", qa_projection - N_QKV_A - INDEX_DIM // QKV_A_TILE, r_wiw, r_sqa
                        )
        elif const_expr(INPUT_NORM_ONCE):
            for t in range(start("qkv_a"), N_QKV_A, G):
                t = fx.Int32(t)
                qin_acc = fx.full(4, 0.0, fx.Float32)
                qin_k_acc = fx.full(4, 0.0, fx.Float32)
                qin_w_acc = fx.full(4, 0.0, fx.Float32)
                for qin_base in range_constexpr(0, QA_UNITS, INPUT_QA_BATCH):
                    QIN_PART = min(INPUT_QA_BATCH, QA_UNITS - qin_base)
                    QIN_K = WAVES * QIN_PART * 128

                    def qin_source(element):
                        qin_row = element // QIN_K
                        qin_wave = (element % QIN_K) // (QIN_PART * 128)
                        qin_local = element % (QIN_PART * 128)
                        return (
                            INPUT_NORM_OFF * 2
                            + qin_row * HIDDEN
                            + qin_wave * QA_UNITS * 128
                            + qin_base * 128
                            + qin_local
                        )

                    def qin_unit(c):
                        qin_kc = (wave * QA_UNITS + qin_base + c) * 2
                        return unit_fp8x2(
                            r_wqa,
                            r_sqa,
                            t,
                            qin_kc,
                            QA_NKC,
                            HIDDEN,
                            n_sel() * (QIN_K // 2 + 4) + (wave * QIN_PART + c) * 64,
                        )

                    qin_pre = [qin_unit(c) for c in range(QIN_PART)]
                    stage_x_pairs("ugp", S * QIN_K, qin_source, padded_row=QIN_K)
                    gpu.barrier()
                    qin_acc = fx.Vector.from_elements(
                        mma_units([qin_acc[e] for e in range(4)], qin_pre),
                        fx.Float32,
                    )
                    if const_expr(with_indexer):
                        if t < INDEX_DIM // QKV_A_TILE:
                            qin_k_units = [
                                unit_fp8x2(
                                    r_wik,
                                    r_sik,
                                    t,
                                    (wave * QA_UNITS + qin_base + c) * 2,
                                    QA_NKC,
                                    HIDDEN,
                                    n_sel() * (QIN_K // 2 + 4) + (wave * QIN_PART + c) * 64,
                                )
                                for c in range(QIN_PART)
                            ]
                            qin_k_acc = fx.Vector.from_elements(
                                mma_units([qin_k_acc[e] for e in range(4)], qin_k_units),
                                fx.Float32,
                            )
                            if t < INDEX_HEADS // QKV_A_TILE:
                                qin_w_units = [
                                    unit_bf16(
                                        r_wiw,
                                        t,
                                        (wave * QA_UNITS + qin_base) * 2 + c,
                                        QA_NKC,
                                        n_sel() * (QIN_K // 2 + 4) + wave * QIN_PART * 64 + c * 32,
                                    )
                                    for c in range(QIN_PART * 2)
                                ]
                                qin_w_acc = fx.Vector.from_elements(
                                    mma_units(
                                        [qin_w_acc[e] for e in range(4)],
                                        qin_w_units,
                                    ),
                                    fx.Float32,
                                )
                    # All MFMA consumers finish before the next K slice reuses xs.
                    gpu.barrier()
                reduce_rows(1, [qin_acc[e] for e in range(4)], emit_out(QKV_A_TILE))
                gpu.barrier()
                if tid < S * QKV_A_TILE:
                    qin_sample = tid // QKV_A_TILE
                    qin_row = t * QKV_A_TILE + tid % QKV_A_TILE
                    qin_value = lds_ld(outs, tid)
                    if qin_row < Q_LORA:
                        put(mb("q_a"), qin_sample * Q_LORA + qin_row, qin_value)
                    else:
                        put(
                            mb("kv_a"),
                            qin_sample * (KV_LORA + PE_DIM) + qin_row - Q_LORA,
                            qin_value,
                        )
                if const_expr(with_indexer):
                    if t < INDEX_DIM // QKV_A_TILE:
                        gpu.barrier()
                        reduce_rows(
                            1,
                            [qin_k_acc[e] for e in range(4)],
                            emit_out(INDEX_TILE),
                        )
                        gpu.barrier()
                        if tid < S * INDEX_TILE:
                            qin_sample = tid // INDEX_TILE
                            qin_row = t * INDEX_TILE + tid % INDEX_TILE
                            put(
                                mb("index_k"),
                                qin_sample * INDEX_DIM + qin_row,
                                lds_ld(outs, tid),
                            )
                        if t < INDEX_HEADS // QKV_A_TILE:
                            gpu.barrier()
                            reduce_rows(
                                1,
                                [qin_w_acc[e] for e in range(4)],
                                emit_out(INDEX_TILE),
                            )
                            gpu.barrier()
                            if tid < S * INDEX_TILE:
                                qin_sample = tid // INDEX_TILE
                                qin_row = t * INDEX_TILE + tid % INDEX_TILE
                                put(
                                    mb("index_w"),
                                    qin_sample * INDEX_HEADS + qin_row,
                                    lds_ld(outs, tid),
                                )
        else:
            for t in range(start("qkv_a"), N_QKV_A, G):
                t = fx.Int32(t)
                for sample_base in range_constexpr(0, S, SAMPLE_TILE):
                    group_count = min(SAMPLE_TILE, S - sample_base)

                    def u_qa(c):
                        kc = (wave * QA_UNITS + c) * attention_k_chunks_per_unit
                        return unit_attention(
                            r_wqa,
                            r_sqa,
                            t,
                            kc,
                            QA_NKC,
                            HIDDEN,
                            128,
                            attention_b_word(n_sel(group_count) * HIDDEN + kc * 64),
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

                    if const_expr(S <= SAMPLE_TILE):
                        h_ld = load_x_rmsnorm(ld_h, HIDDEN, g_in, group_count)
                        pre = [u_qa(c) for c in range(QA_UNITS)]
                        if const_expr(attention_ptpc):
                            qa_scales = stage_x_rmsnorm_ptpc(ld_h, HIDDEN, g_in, loaded=h_ld, count=group_count)
                        else:
                            stage_x_rmsnorm(ld_h, HIDDEN, g_in, loaded=h_ld, count=group_count)
                    else:
                        if const_expr(attention_ptpc):
                            qa_scales = stage_x_rmsnorm_ptpc(ld_h, HIDDEN, g_in, count=group_count)
                        else:
                            stage_x_rmsnorm(ld_h, HIDDEN, g_in, count=group_count)
                        pre = [u_qa(c) for c in range(QA_UNITS)]
                    gpu.barrier()
                    acc = run_units(u_qa, QA_UNITS, QA_UNITS, pre)
                    if const_expr(attention_ptpc):
                        reduce_rows(
                            1,
                            acc,
                            emit_ptpc(QKV_A_TILE, t * QKV_A_TILE, r_sqa, qa_scales),
                            group_count,
                        )
                    else:
                        reduce_rows(1, acc, emit_out(QKV_A_TILE), group_count)
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
                                (INDEX_BF16_OFF if const_expr(attention_ptpc) else 0)
                                + (n_sel(group_count) * HIDDEN + kc * 64) // 2,
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
                                    (INDEX_BF16_OFF if const_expr(attention_ptpc) else 0)
                                    + (n_sel(group_count) * HIDDEN + kc * 64) // 2,
                                )

                            ik_acc = run_units(u_index_k, QA_UNITS, QA_UNITS)
                            iw_acc = [fx.Float32(0.0) for _ in range(4)]
                            if t < INDEX_HEADS // QKV_A_TILE:
                                iw_acc = run_units(u_index_w, IW_CPW, IW_CPW)
                            reduce_rows(1, ik_acc, emit_out(INDEX_TILE), group_count)
                            gpu.barrier()
                            if tid < group_count * INDEX_TILE:
                                s = sample_base + tid // INDEX_TILE
                                row = t * INDEX_TILE + tid % INDEX_TILE
                                put(
                                    mb("index_k"),
                                    s * INDEX_DIM + row,
                                    lds_ld(outs, tid),
                                )
                            if t < INDEX_HEADS // QKV_A_TILE:
                                reduce_rows(1, iw_acc, emit_out(INDEX_TILE), group_count)
                                gpu.barrier()
                                if tid < group_count * INDEX_TILE:
                                    s = sample_base + tid // INDEX_TILE
                                    row = t * INDEX_TILE + tid % INDEX_TILE
                                    put(
                                        mb("index_w"),
                                        s * INDEX_HEADS + row,
                                        lds_ld(outs, tid),
                                    )

    def query_norm():
        # ===================== 2. one q_a RMSNorm CTA per sample, shared downstream
        for s_norm in range(start("q_norm"), S, G):
            s_norm = fx.Int32(s_norm)
            hint_wait(
                Q_LORA // QKV_A_TILE,
                lambda k: (
                    mb("q_a"),
                    s_norm * Q_LORA + k * QKV_A_TILE + QKV_A_TILE - 1,
                ),
            )

            def ld_qa_one(sks):
                return ld_qa([(s_norm, k) for _, k in sks])

            # FP4 amplifies a one-ULP normalized query difference into a
            # different quantized value. Keep this small reduction precise.
            stage_x_rmsnorm(ld_qa_one, Q_LORA, g_q, count=1, precise=indexer_quant)
            gpu.barrier()
            k = tid * 4
            w0 = lds_ld(xs, k // 2)
            w1 = lds_ld(xs, k // 2 + 1)
            a0, a1 = bf2_f32(w0.bitcast(fx.Int32))
            a2, a3 = bf2_f32(w1.bitcast(fx.Int32))
            put_bf(mb("q_an"), s_norm * Q_LORA + k, [a0, a1, a2, a3])

    def cache():
        # ================ 3. KV RMSNorm + k_pe RoPE -> cache (+ this launch's rows)
        for t in range(start("cache"), S // CACHE_TILE, G):
            sample_base = fx.Int32(t) * CACHE_TILE if const_expr(CACHE_TILE < S) else fx.Int32(0)
            r_kv = _rsrc(kv_cache)
            r_pe = _rsrc(pe_cache)
            # gamma and the RoPE factors are issued ahead of the wait
            g = ld_bf16(_rsrc(g_kv), tid)
            tpe = tid % (PE_DIM // 2)
            cs = [
                ld_rope(
                    _rsrc(rope_cos),
                    row_position(sample_base + s) * (PE_DIM // 2) + tpe,
                )
                for s in range(CACHE_TILE)
            ]
            sns = [
                ld_rope(
                    _rsrc(rope_sin),
                    row_position(sample_base + s) * (PE_DIM // 2) + tpe,
                )
                for s in range(CACHE_TILE)
            ]
            hint_wait(
                (KV_LORA + PE_DIM) // QKV_A_TILE,
                lambda k: (
                    mb("kv_a"),
                    (sample_base + CACHE_TILE - 1) * (KV_LORA + PE_DIM) + k * QKV_A_TILE + QKV_A_TILE - 1,
                ),
            )
            # This task's query tile shares one poll and one block reduction.
            vs = getf_many([(mb("kv_a"), (sample_base + s) * (KV_LORA + PE_DIM) + tid) for s in range(CACHE_TILE)])
            pes = get2_many(
                [
                    (
                        mb("kv_a"),
                        (sample_base + s) * (KV_LORA + PE_DIM) + KV_LORA + (tid % (PE_DIM // 2)) * 2,
                    )
                    for s in range(CACHE_TILE)
                ]
            )
            ssq = block_sums([v * v for v in vs])
            for local_s in range_constexpr(CACHE_TILE):
                s = sample_base + local_s
                pos = row_position(s)
                slot = row_slot(s)
                # Buffer offsets are 32-bit bytes. Long paged caches can
                # exceed 4 GiB; rebase the descriptor at this uniform row.
                store_slot = slot
                if const_expr(use_atom_kv_cache):
                    cache_row_bytes = QK_DIM * (1 if cache_fp8 else 2)
                    r_kv = _rsrc(kv_cache + fx.Int64(_uniform(slot)) * cache_row_bytes)
                    r_pe = _rsrc(pe_cache + fx.Int64(_uniform(slot)) * cache_row_bytes)
                    store_slot = fx.Int32(0)
                active = row_writes_cache(s) & (not DEFER_OFF_CACHE)
                if const_expr(use_atom_kv_cache and not DEFER_OFF_CACHE and CACHE_TILE < S):
                    # Indexed cache CTAs can also collide on an output slot.
                    # Concurrent cohorts elect the last writer over all rows;
                    # serial tiles retain their original per-tile commit order.
                    # Every row still publishes kvnew/penew for its consumers.
                    if const_expr(COHORT_ROW_TILES):
                        later_slot = fx.Int32(
                            bo.buffer_load(
                                _rsrc(full_slots), fx.min(lane, total_samples - 1) * 2, vec_width=1, dtype=T.i32
                            )
                        )
                        overwritten = (lane > row_offset + s) & (lane < total_samples) & (later_slot == slot)
                    else:
                        later_slot = row_slot(fx.min(lane, S - 1))
                        overwritten = (lane > s) & (lane < S) & (later_slot == slot)
                    overwritten_mask = fx.Int64(rocdl.ballot(T.i64, overwritten))
                    active = active & (overwritten_mask == 0)
                value = vs[local_s] * _rsq(ssq[local_s] * (1.0 / KV_LORA) + EPS) * g
                if const_expr(cache_fp8):
                    value = bf16_round(value)
                    kvn, _ = _fp8_roundtrip(value, value)
                else:
                    kvn = bf16_round(value)
                if const_expr(use_atom_kv_cache):
                    if active:
                        if const_expr(cache_fp8):
                            pair = fx.Int32(
                                rocdl.cvt_pk_fp8_f32(T.i32, kvn, _xshfl(kvn, 1), fx.Int32(0), False)
                            ) & fx.Int32(0xFFFF)
                            upper_pair = bpermute_i32(fp8_kv_upper_pair_lane(lane) * 4, pair)
                            if lane % 4 == 0:
                                bo.buffer_store(
                                    pair | (upper_pair << 16),
                                    r_kv,
                                    store_slot * (QK_DIM // 4) + tid // 4,
                                )
                        else:
                            bo.buffer_store(kvn.to(fx.BFloat16), r_kv, store_slot * QK_DIM + tid)
                else:
                    bo.buffer_store(kvn.to(fx.BFloat16), r_kv, pos * KV_LORA + tid)
                put(mb("kvnew"), s * KV_LORA + tid, kvn)
                if tid < PE_DIM // 2:
                    x0, x1 = pes[local_s]
                    c, sn = cs[local_s], sns[local_s]
                    p0, p1 = x0 * c - x1 * sn, x0 * sn + x1 * c
                    if const_expr(cache_fp8):
                        p0, p1 = bf16_round(p0), bf16_round(p1)
                        p0, p1 = _fp8_roundtrip(p0, p1)
                    else:
                        p0, p1 = bf16_round(p0), bf16_round(p1)
                    if const_expr(use_atom_kv_cache):
                        if active:
                            if const_expr(cache_fp8):
                                pair = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, p0, p1, fx.Int32(0), False)) & fx.Int32(
                                    0xFFFF
                                )
                                upper_pair = bpermute_i32(fp8_pe_upper_pair_lane(lane) * 4, pair)
                                if lane % 2 == 0:
                                    bo.buffer_store(
                                        pair | (upper_pair << 16),
                                        r_pe,
                                        store_slot * (QK_DIM // 4) + KV_LORA // 4 + tid // 2,
                                    )
                            else:
                                pe_offset = store_slot * QK_DIM + KV_LORA + tid * 2
                                bo.buffer_store(p0.to(fx.BFloat16), r_pe, pe_offset)
                                bo.buffer_store(p1.to(fx.BFloat16), r_pe, pe_offset + 1)
                    else:
                        pe_offset = pos * PE_DIM + tid * 2
                        bo.buffer_store(p0.to(fx.BFloat16), r_pe, pe_offset)
                        bo.buffer_store(p1.to(fx.BFloat16), r_pe, pe_offset + 1)
                    put2(mb("penew"), s * PE_DIM + tid * 2, p0, p1)

            if const_expr(with_indexer):
                # Index keys use LayerNorm (not RMSNorm), interleaved RoPE on
                # the first 64 dimensions, and a BF16 rounding boundary.
                # Optional MXFP4 quantizes in this same coordinate system;
                # it does not add a Hadamard transform to Q or K.
                r_gik, r_bik = _rsrc(index_arg(5)), _rsrc(index_arg(6))
                r_index_cache = _rsrc(indices)
                for local_s in range_constexpr(CACHE_TILE):
                    s = sample_base + local_s
                    ik = getf(mb("index_k"), s * INDEX_DIM + fx.min(tid, INDEX_DIM - 1))
                    live = tid < INDEX_DIM
                    iv = live.select(ik, fx.Float32(0.0))
                    mean = block_sum(iv) * (1.0 / INDEX_DIM)
                    centered = live.select(ik - mean, fx.Float32(0.0))
                    rstd = _rsq(block_sum(centered * centered) * (1.0 / INDEX_DIM) + 1.0e-6)
                    if tid < INDEX_DIM // 2:
                        i0 = tid * 2
                        k0, k1 = get2(mb("index_k"), s * INDEX_DIM + i0)
                        v0 = (k0 - mean) * rstd * ld_f32(r_gik, i0) + ld_f32(r_bik, i0)
                        v1 = (k1 - mean) * rstd * ld_f32(r_gik, i0 + 1) + ld_f32(r_bik, i0 + 1)
                        if tid < PE_DIM // 2:
                            c, sn = cs[local_s], sns[local_s]
                            v0, v1 = v0 * c - v1 * sn, v0 * sn + v1 * c
                        if const_expr(indexer_quant):
                            pair4, word4, exponent = quant_index_pair(v0, v1)
                            scale4 = (exponent << 23).bitcast(fx.Float32)
                            key_base = indices + fx.Int64(index_page_slot(s, row_position(s))) * INDEX_FP4_BYTES
                            if lane % 4 == 0:
                                bo.buffer_store(word4, _rsrc(key_base), i0 // 8)
                                put(mb("index_k_quant"), s * 20 + i0 // 8, word4)
                            if lane % 16 == 0:
                                bo.buffer_store(exponent.to(fx.Int8), _rsrc(key_base), INDEX_DIM // 2 + i0 // 32)
                                put(mb("index_k_quant"), s * 20 + 16 + i0 // 32, exponent)
                            # Consumers of the current invocation must see the cache's
                            # quantized values, including when a later row reuses a slot.
                            restored = fx.Vector(rocdl.cvt_scalef32_pk_bf16_fp4(T.vec(2, T.bf16), pair4, scale4, 0)).to(
                                fx.Float32
                            )
                            v0, v1 = restored[0], restored[1]
                        elif const_expr(TILED_INDEXER):
                            # Uniform current-key row base, full 64-bit physical address.
                            key_base = indices + fx.Int64(index_page_slot(s, row_position(s))) * (INDEX_DIM * 2)
                            bo.buffer_store(
                                fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16),
                                _rsrc(key_base),
                                i0,
                            )
                        else:
                            bo.buffer_store(
                                fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16),
                                r_index_cache,
                                index_page_slot(s, row_position(s)) * INDEX_DIM + i0,
                            )
                        put_bf(mb("index_k_new"), s * INDEX_DIM + i0, [v0, v1])
                    gpu.barrier()
                    if tid == 0:
                        put(mb("index_ready"), s, fx.Int32(1))

    def query_projection():
        # =============================================== 4. normalized q_a -> q_b (+RoPE)
        def emit_index_query(qs, qt, qr):
            values = [lds_ld(outs, qs * INDEX_TILE + qr + j) for j in range(4)]
            offset = qs * INDEX_Q_ROWS + qt * INDEX_TILE + qr
            put_bf(mb("index_q"), offset, values)
            if const_expr(INDEX_PRE_ROPE and not indexer_quant):
                # Keep the first BF16 rounding used by the raw query mailbox.
                rounded = [bf16_round(v) for v in values]
                rotated = []
                for pair in range_constexpr(2):
                    q0, q1 = rounded[pair * 2], rounded[pair * 2 + 1]
                    qk = (qt * INDEX_TILE + qr + pair * 2) % INDEX_DIM
                    if qk < PE_DIM:
                        qc = ld_rope(_rsrc(rope_cos), row_position(qs) * (PE_DIM // 2) + qk // 2)
                        qsn = ld_rope(_rsrc(rope_sin), row_position(qs) * (PE_DIM // 2) + qk // 2)
                        q0, q1 = q0 * qc - q1 * qsn, q0 * qsn + q1 * qc
                    rotated += [q0, q1]
                # Keep rotated queries above the later mid quantization region.
                # Its readiness tags must never alias within this epoch.
                put_bf(mb("ugp"), 2 * INDEX_ROT_OFF + offset, rotated)

        r_wqb, r_sqb = _rsrc(w_q_b), _rsrc(s_q_b)
        QB_NKC = Q_LORA // 64
        QB_UNITS = QB_NKC // (attention_k_chunks_per_unit * WAVES)
        if const_expr(with_indexer):
            r_wiq, r_siq = _rsrc(index_arg(3)), _rsrc(index_arg(4))

        if const_expr(GLOBAL_Q_B):
            # All output tiles cover the same global 16-row chunk. Each
            # output element retains its original eight-wave K reduction.
            for qb_chunk in range((total_samples + 15) // 16):
                qb_chunk = fx.Int32(qb_chunk)
                qb_first = qb_chunk * 16
                qb_count = fx.min(fx.Int32(16), total_samples - qb_first)
                t = full_bid
                # Three
                # task rounds occupy its first192 Q_B slots;64 stay unused.

                def u_global_qb(c):
                    kc = (wave * QB_UNITS + c) * 2
                    return unit_fp8x2(
                        r_wqb,
                        r_sqb,
                        t,
                        kc,
                        QB_NKC,
                        Q_LORA,
                        fx.min(lane % 16, qb_count - 1) * (Q_LORA // 2 + 4) + kc * 32,
                    )

                pre = [u_global_qb(c) for c in range(QB_UNITS)]
                gpu.barrier()
                # Four whole rows per batch bounds live mailbox values.
                # The last chunk stages eight rows; MFMA padding duplicates
                # row seven and publishes no padded output.
                for qb_batch in range_constexpr(0, 16, 4):
                    if qb_batch < qb_count:
                        qb_specs = []
                        for qb_part in range_constexpr(4):
                            qb_row = qb_first + qb_batch + qb_part
                            qb_src = full_scratch + fx.Int64(qb_row // S) * TILE_SCRATCH_BYTES
                            qb_specs.append((qb_src + SC["q_an"], ((qb_row % S) * Q_LORA + tid * 4) // 2, 2))
                        qb_values = poll(qb_specs)
                        for qb_part in range_constexpr(4):
                            for qb_pair in range_constexpr(2):
                                lds_st(
                                    xs,
                                    (qb_batch + qb_part) * (Q_LORA // 2 + 4) + tid * 2 + qb_pair,
                                    qb_values[qb_part][qb_pair].bitcast(fx.Float32),
                                )
                gpu.barrier()
                acc = run_units(u_global_qb, QB_UNITS, QB_UNITS, pre)
                reduce_rows(1, acc, emit_out(Q_B_TILE), count=16)
                gpu.barrier()
                head = t // QB_PER_HEAD
                hoff = (t % QB_PER_HEAD) * Q_B_TILE
                if hoff < NOPE_DIM:
                    if tid < qb_count * Q_B_TILE // 4:
                        qb_local = tid // (Q_B_TILE // 4)
                        qb_row = qb_first + qb_local
                        qr = tid % (Q_B_TILE // 4) * 4
                        qb_dst = full_scratch + fx.Int64(qb_row // S) * TILE_SCRATCH_BYTES
                        put_bf(
                            qb_dst + SC["q_nope"],
                            ((qb_row % S) * H + head) * NOPE_DIM + hoff + qr,
                            [lds_ld(outs, qb_local * Q_B_TILE + qr + j) for j in range(4)],
                        )
                else:
                    if tid < qb_count * Q_B_TILE // 2:
                        qb_local = tid // (Q_B_TILE // 2)
                        qb_row = qb_first + qb_local
                        qr = tid % (Q_B_TILE // 2) * 2
                        qk = hoff - NOPE_DIM + qr
                        qb_dst = full_scratch + fx.Int64(qb_row // S) * TILE_SCRATCH_BYTES
                        qpos = fx.Int32(bo.buffer_load(_rsrc(full_positions), qb_row * 2, vec_width=1, dtype=T.i32))
                        qb_begin = fx.Int32(bo.buffer_load(_rsrc(full_indptr), qb_row, vec_width=1, dtype=T.i32))
                        qb_end = fx.Int32(bo.buffer_load(_rsrc(full_indptr), qb_row + 1, vec_width=1, dtype=T.i32))
                        qb_slot = fx.Int32(bo.buffer_load(_rsrc(full_slots), qb_row * 2, vec_width=1, dtype=T.i32))
                        qpos = ((qb_end > qb_begin) | (qb_slot >= 0)).select(qpos, fx.Int32(0))
                        qcos = ld_rope(_rsrc(rope_cos), qpos * (PE_DIM // 2) + qk // 2)
                        qsin = ld_rope(_rsrc(rope_sin), qpos * (PE_DIM // 2) + qk // 2)
                        q0 = lds_ld(outs, qb_local * Q_B_TILE + qr)
                        q1 = lds_ld(outs, qb_local * Q_B_TILE + qr + 1)
                        put_bf(
                            qb_dst + SC["q_pe"],
                            ((qb_row % S) * H + head) * PE_DIM + qk,
                            [q0 * qcos - q1 * qsin, q0 * qsin + q1 * qcos],
                        )

                if const_expr(with_indexer and not SHARD_INDEX_Q):
                    # Raw and precomputed rotated index queries retain their
                    # BF16 rounding and ten-row cohort mailbox addresses.
                    gpu.barrier()

                    def u_global_index_q(c):
                        kc = (wave * QB_UNITS + c) * 2
                        return unit_fp8x2(
                            r_wiq,
                            r_siq,
                            t,
                            kc,
                            QB_NKC,
                            Q_LORA,
                            fx.min(lane % 16, qb_count - 1) * (Q_LORA // 2 + 4) + kc * 32,
                        )

                    index_acc = run_units(u_global_index_q, QB_UNITS, QB_UNITS)
                    reduce_rows(1, index_acc, emit_out(INDEX_TILE), count=16)
                    gpu.barrier()
                    if tid < qb_count * INDEX_TILE // 4:
                        index_local = tid // (INDEX_TILE // 4)
                        index_row = qb_first + index_local
                        index_r = tid % (INDEX_TILE // 4) * 4
                        index_dst = full_scratch + fx.Int64(index_row // S) * TILE_SCRATCH_BYTES
                        index_offset = (index_row % S) * INDEX_Q_ROWS + t * INDEX_TILE + index_r
                        index_values = [lds_ld(outs, index_local * INDEX_TILE + index_r + j) for j in range(4)]
                        put_bf(index_dst + SC["index_q"], index_offset, index_values)
                        if const_expr(not indexer_quant):
                            rounded = [bf16_round(v) for v in index_values]
                            rotated = []
                            for pair in range_constexpr(2):
                                q0, q1 = rounded[pair * 2], rounded[pair * 2 + 1]
                                qk = (t * INDEX_TILE + index_r + pair * 2) % INDEX_DIM
                                if qk < PE_DIM:
                                    qp = fx.Int32(
                                        bo.buffer_load(_rsrc(full_positions), index_row * 2, vec_width=1, dtype=T.i32)
                                    )
                                    qb = fx.Int32(
                                        bo.buffer_load(_rsrc(full_indptr), index_row, vec_width=1, dtype=T.i32)
                                    )
                                    qe = fx.Int32(
                                        bo.buffer_load(_rsrc(full_indptr), index_row + 1, vec_width=1, dtype=T.i32)
                                    )
                                    qs = fx.Int32(
                                        bo.buffer_load(_rsrc(full_slots), index_row * 2, vec_width=1, dtype=T.i32)
                                    )
                                    qp = ((qe > qb) | (qs >= 0)).select(qp, fx.Int32(0))
                                    qc = ld_rope(_rsrc(rope_cos), qp * (PE_DIM // 2) + qk // 2)
                                    qsn = ld_rope(_rsrc(rope_sin), qp * (PE_DIM // 2) + qk // 2)
                                    q0, q1 = q0 * qc - q1 * qsn, q0 * qsn + q1 * qc
                                rotated += [q0, q1]
                            put_bf(index_dst + SC["ugp"], 2 * INDEX_ROT_OFF + index_offset, rotated)
        else:
            for t in range(start("q_b"), N_QB, G):
                t = fx.Int32(t)

                def u_qb(c):
                    kc = (wave * QB_UNITS + c) * attention_k_chunks_per_unit
                    return unit_attention(
                        r_wqb,
                        r_sqb,
                        t,
                        kc,
                        QB_NKC,
                        Q_LORA,
                        128,
                        attention_b_word(n_sel() * Q_LORA + kc * 64),
                    )

                pre = [u_qb(c) for c in range(QB_UNITS)]
                hint_wait(
                    S,
                    lambda s: (mb("q_an"), (s * Q_LORA + Q_LORA - 2) // 2),
                )
                if const_expr(attention_ptpc):
                    qb_scales = stage_x_pairs_ptpc("q_an", S, Q_LORA, lambda k: k, retain_bf16=with_indexer)
                else:
                    stage_x_pairs("q_an", S * Q_LORA, lambda k: k)
                gpu.barrier()
                acc = run_units(u_qb, QB_UNITS, QB_UNITS, pre)
                if const_expr(attention_ptpc):
                    reduce_rows(1, acc, emit_ptpc(Q_B_TILE, t * Q_B_TILE, r_sqb, qb_scales))
                else:
                    reduce_rows(1, acc, emit_out(Q_B_TILE))
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
                        c = ld_rope(_rsrc(rope_cos), row_position(s) * (PE_DIM // 2) + i // 2)
                        sn = ld_rope(_rsrc(rope_sin), row_position(s) * (PE_DIM // 2) + i // 2)
                        put_bf(
                            mb("q_pe"),
                            (s * H + head) * PE_DIM + i,
                            [x0 * c - x1 * sn, x0 * sn + x1 * c],
                        )

                if const_expr(with_indexer and not SHARD_INDEX_Q):
                    # Reuse this CTA's normalized q_lora tile for one index-query
                    # row tile.  Complementary CTAs compute the remaining half below.
                    gpu.barrier()
                    iq_t = t

                    def u_index_q(c):
                        kc = (wave * QB_UNITS + c) * 2
                        return unit_fp8x2(
                            r_wiq,
                            r_siq,
                            iq_t,
                            kc,
                            QB_NKC,
                            Q_LORA,
                            (INDEX_Q_BF16_OFF if const_expr(attention_ptpc) else 0) + (n_sel() * Q_LORA + kc * 64) // 2,
                        )

                    iq_acc = run_units(u_index_q, QB_UNITS, QB_UNITS)
                    reduce_rows(1, iq_acc, emit_out(INDEX_TILE))
                    gpu.barrier()
                    if tid < S * INDEX_TILE // 4:
                        s = tid // (INDEX_TILE // 4)
                        r = (tid % (INDEX_TILE // 4)) * 4
                        emit_index_query(s, iq_t, r)

        if const_expr(with_indexer):
            # The 128 CTAs without q_b work produce the other 128 index-query
            # tiles concurrently.  They reload q_a, but remove one full GEMV
            # from the q_b CTAs' serialized critical path.
            N_INDEX_Q_EXTRA = (
                INDEX_Q_ROWS // INDEX_TILE // CP_RANKS if SHARD_INDEX_Q else INDEX_Q_ROWS // INDEX_TILE - N_QB
            )
            for tt in range(start("index_q"), N_INDEX_Q_EXTRA, G):
                tt = fx.Int32(tt)
                iq_t = cp_local_rank() * N_INDEX_Q_EXTRA + tt if const_expr(SHARD_INDEX_Q) else N_QB + tt

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
                hint_wait(
                    S,
                    lambda s: (mb("q_an"), (s * Q_LORA + Q_LORA - 2) // 2),
                )
                stage_x_pairs("q_an", S * Q_LORA, lambda k: k)
                gpu.barrier()
                iq_acc = run_units(u_index_q_extra, QB_UNITS, QB_UNITS, pre)
                reduce_rows(1, iq_acc, emit_out(INDEX_TILE))
                gpu.barrier()
                if const_expr(SHARD_INDEX_Q):
                    if (wave < CP_RANKS) & (lane < S * INDEX_TILE // 4):
                        s = lane // (INDEX_TILE // 4)
                        r = (lane % (INDEX_TILE // 4)) * 4
                        put_bf(
                            cp_buffer(
                                "index_cp_q",
                                (cp_peer(wave) if const_expr(CP_RANKS < W) else peer_dst),
                            ),
                            s * INDEX_Q_ROWS + iq_t * INDEX_TILE + r,
                            [lds_ld(outs, s * INDEX_TILE + r + j) for j in range(4)],
                            CM_SYS,
                        )
                else:
                    if tid < S * INDEX_TILE // 4:
                        s = tid // (INDEX_TILE // 4)
                        r = (tid % (INDEX_TILE // 4)) * 4
                        emit_index_query(s, iq_t, r)

    def index_quant():
        if const_expr(with_indexer and indexer_quant):
            # Each head is quantized once after all of its projection tiles.
            # Producers precede this stage on every worker, so waiting here
            # never depends on a task beyond this stage.
            for qtask in range(bid, S * INDEX_HEADS, G):
                qtask = fx.Int32(qtask)
                qs, qhead = qtask // INDEX_HEADS, qtask % INDEX_HEADS
                if tid < INDEX_DIM // 2:
                    qk = tid * 2
                    q_source = cp_buffer("index_cp_q", sym) if const_expr(SHARD_INDEX_Q) else mb("index_q")
                    raw = poll(
                        [(q_source, (qs * INDEX_Q_ROWS + qhead * INDEX_DIM + qk) // 2, 1)],
                        scope="system" if SHARD_INDEX_Q else "agent",
                    )[0][0]
                    q0, q1 = bf2_f32(raw)
                    if qk < PE_DIM:
                        c = ld_rope(_rsrc(rope_cos), row_position(qs) * (PE_DIM // 2) + qk // 2)
                        sn = ld_rope(_rsrc(rope_sin), row_position(qs) * (PE_DIM // 2) + qk // 2)
                        q0, q1 = q0 * c - q1 * sn, q0 * sn + q1 * c
                    _, word4, exponent = quant_index_pair(q0, q1)
                    if lane % 4 == 0:
                        put(mb("index_q_quant"), qtask * 20 + qk // 8, word4)
                    if lane % 16 == 0:
                        put(mb("index_q_quant"), qtask * 20 + 16 + qk // 32, exponent)
                gpu.barrier()

    def absorbed_query():
        # ==================================== 4. absorbed query: q_lat = W_UK^T q_nope
        # 8 row groups (128 latent rows of one head) x 3 chunks: one row group per wave
        r_wuk, r_suk = _rsrc(w_uk), _rsrc(s_uk)
        UK_NKC = NOPE_DIM // 64
        for t in range(start("uk"), N_UK, G):
            t = fx.Int32(t)
            head = t // UK_PER_HEAD

            def u_uk(c):
                return unit_attention(
                    r_wuk,
                    r_suk,
                    t * WAVES + wave,
                    c,
                    UK_NKC,
                    NOPE_DIM,
                    64,
                    attention_b_word(n_sel() * NOPE_DIM + c * 64),
                    coef=(
                        (
                            lambda: pick(
                                [uk_scales[sample * ((NOPE_DIM + 127) // 128) + c // 2] for sample in range(S)],
                                n_sel(),
                            )
                        )
                        if const_expr(attention_ptpc)
                        else None
                    ),
                )

            pre = [u_uk(c) for c in range(UK_NKC)]
            hint_wait(
                NOPE_DIM // Q_B_TILE,
                lambda k: (
                    mb("q_nope"),
                    ((S - 1) * H + head) * NOPE_DIM + k * Q_B_TILE + Q_B_TILE - 1,
                ),
            )
            if const_expr(attention_ptpc):
                uk_scales = stage_x_pairs_ptpc(
                    "q_nope",
                    S,
                    NOPE_DIM,
                    lambda k: ((k // NOPE_DIM) * H + head) * NOPE_DIM + k % NOPE_DIM,
                    group=128,
                )
            else:
                stage_x_pairs(
                    "q_nope",
                    S * NOPE_DIM,
                    lambda k: ((k // NOPE_DIM) * H + head) * NOPE_DIM + k % NOPE_DIM,
                )
            gpu.barrier()
            acc = run_units(u_uk, UK_NKC, UK_NKC, pre)
            fx.ptr_store(
                fx.Vector.from_elements(acc, fx.Float32),
                red + (wave * 64 + lane) * 4,
            )
            gpu.barrier()
            if tid < S * UK_TILE // 4:
                k = tid * 4
                s = k // UK_TILE
                r0 = k % UK_TILE
                vals = []
                if const_expr(attention_ptpc):
                    ptpc_scale = ld_f32(r_suk, 0)
                for j in range_constexpr(4):
                    r = r0 + j
                    ww = r // 16
                    value = lds_ld(red, (ww * 64 + s + 16 * ((r % 16) // 4)) * 4 + r % 4)
                    if const_expr(attention_ptpc):
                        value = bf16_round(value * ptpc_scale)
                    vals.append(value)
                put_bf(
                    mb("q_lat"),
                    (s * H + head) * KV_LORA + (t % UK_PER_HEAD) * UK_TILE + r0,
                    vals,
                )

    return PreattnStages(
        input_qkv=input_qkv,
        query_norm=query_norm,
        cache=cache,
        query_projection=query_projection,
        index_quant=index_quant,
        absorbed_query=absorbed_query,
    )
