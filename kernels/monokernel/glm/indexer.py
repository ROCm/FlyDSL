# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Index scores and exact sparse selection, including supported CP selectors."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.config import PE_DIM
from kernels.monokernel.glm.layout import INDEX_DIM, INDEX_HEADS, INDEX_KEYS_PER_TASK, INDEX_Q_ROWS, WAVES
from kernels.monokernel.layout import CM_DEV, CM_SYS, THREADS
from kernels.monokernel.ops import read_lane_i32
from kernels.monokernel.ops import rsrc as _rsrc
from kernels.monokernel.ops import uniform as _uniform
from kernels.monokernel.ops import xred as _xred


@flyc.jit
def emit_indexer(plan, state, tools):
    CP_PARTS = plan.CP_PARTS
    CP_RANKS = plan.CP_RANKS
    G = plan.G
    INDEX_CAPACITY = plan.INDEX_CAPACITY
    INDEX_CP_TILE = plan.INDEX_CP_TILE
    INDEX_FP4_BYTES = plan.INDEX_FP4_BYTES
    INDEX_PRE_ROPE = plan.INDEX_PRE_ROPE
    INDEX_QUERY_REUSE = plan.INDEX_QUERY_REUSE
    INDEX_RADIX_BINS = plan.INDEX_RADIX_BINS
    INDEX_RADIX_DIGITS = plan.INDEX_RADIX_DIGITS
    INDEX_ROT_OFF = plan.INDEX_ROT_OFF
    INDEX_SCORE_REUSE = plan.INDEX_SCORE_REUSE
    LOCAL_INDEX_CAPACITY = plan.LOCAL_INDEX_CAPACITY
    RESIDENT_INDEX_SELECT = plan.RESIDENT_INDEX_SELECT
    S = plan.S
    SHARD_INDEX_Q = plan.SHARD_INDEX_Q
    TILED_INDEXER = plan.TILED_INDEXER
    W = plan.W
    _partition = plan.partition
    indexer_cp = plan.indexer_cp
    indexer_quant = plan.indexer_quant
    topk = plan.topk
    with_indexer = plan.with_indexer
    bid = state.bid
    indices = state.indices
    keys = state.keys
    lane = state.lane
    peer_dst = state.peer_dst
    red = state.red
    rope_cos = state.rope_cos
    rope_sin = state.rope_sin
    sym = state.sym
    tag = state.tag
    tid = state.tid
    wave = state.wave
    xs = state.xs
    ld_rope = tools.ld_rope
    row_position = tools.row_position
    index_request_start = tools.index_request_start
    index_page_slot = tools.index_page_slot
    lds_ld = tools.lds_ld
    lds_st = tools.lds_st
    bf16_pair = tools.bf16_pair
    mb = tools.mb
    cp_local_rank = tools.cp_local_rank
    cp_peer = tools.cp_peer
    cp_buffer = tools.cp_buffer
    put = tools.put
    bf2_f32 = tools.bf2_f32
    poll = tools.poll
    get = tools.get
    getf = tools.getf
    get_bf2_many = tools.get_bf2_many
    start = tools.start

    # ====================== 4b. fused sparse index score + exact top-2048
    if const_expr(with_indexer):
        r_index_cache = _rsrc(indices)
        N_INDEX_SPLIT = INDEX_CAPACITY // INDEX_KEYS_PER_TASK

        def use_long_index(s):
            if const_expr(indexer_quant and indexer_cp):
                return fx.Boolean(TILED_INDEXER)
            if const_expr(indexer_cp):
                return row_position(s) + 1 > LOCAL_INDEX_CAPACITY
            return fx.Boolean(True)

        def use_short_index(s):
            if const_expr(indexer_quant and indexer_cp):
                return fx.Boolean(not TILED_INDEXER)
            # A short compiled capacity cannot cross the selector boundary.
            # Preserve the original branch-free short path with CP off.
            if const_expr(not TILED_INDEXER):
                return fx.Boolean(True)
            return row_position(s) + 1 <= LOCAL_INDEX_CAPACITY

        def index_shard(s):
            bound = row_position(s) + 1
            if const_expr(indexer_cp):
                stride = (
                    (bound + CP_RANKS * INDEX_KEYS_PER_TASK - 1) // (CP_RANKS * INDEX_KEYS_PER_TASK)
                ) * INDEX_KEYS_PER_TASK
                begin = cp_local_rank() * stride
                return begin, fx.max(fx.Int32(0), fx.min(stride, bound - begin))
            return fx.Int32(0), bound

        def load_index_q8(head, k):
            words = fx.Vector.from_elements(
                [lds_ld(xs, (head * INDEX_DIM + k) // 2 + j) for j in range(4)],
                fx.Float32,
            )
            return words.bitcast(fx.BFloat16)

        if const_expr(RESIDENT_INDEX_SELECT):
            EARLY_HIST_ACCUM_BYTES = INDEX_RADIX_DIGITS * S * INDEX_RADIX_BINS * 4
            early_hist_done = mb("index_cp_hist") + fx.Int64(EARLY_HIST_ACCUM_BYTES)
            early_hist_init = early_hist_done + fx.Int64(INDEX_RADIX_DIGITS * S * CP_PARTS * 8)

            def early_hist_pointer(digit, sample, bucket):
                return fx.inttoptr(
                    fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4),
                    mb("index_cp_hist") + fx.Int64(((digit * S + sample) * INDEX_RADIX_BINS + bucket) * 4),
                )

            if const_expr(RESIDENT_INDEX_SELECT):
                for sample in range(bid, S, G):
                    sample = fx.Int32(sample)
                    if use_long_index(sample):
                        for bin_chunk in range_constexpr(max(1, INDEX_RADIX_BINS // THREADS)):
                            bucket_lane = bin_chunk * THREADS + tid
                            if bucket_lane < INDEX_RADIX_BINS:
                                for digit in range_constexpr(INDEX_RADIX_DIGITS):
                                    fx.generic_store(early_hist_pointer(digit, sample, bucket_lane), fx.Int32(0))
                        fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope="agent")
                        gpu.barrier()
                        if tid == 0:
                            put(early_hist_init, sample, fx.Int32(1))

        # The aligned CP capacity can contain an odd count of key tiles.
        # Keep the final partial reuse group; its padded part has no
        # payload slot. Live-shard bounds guard the math.
        N_INDEX_GROUPS = (N_INDEX_SPLIT + INDEX_SCORE_REUSE - 1) // INDEX_SCORE_REUSE
        for index_group in range(start("index_score"), (S // INDEX_QUERY_REUSE) * N_INDEX_GROUPS, G):
            index_group = fx.Int32(index_group)
            s = (index_group // N_INDEX_GROUPS) * INDEX_QUERY_REUSE
            index_tile_base = (index_group % N_INDEX_GROUPS) * INDEX_SCORE_REUSE
            tile = index_tile_base
            tt = s * N_INDEX_SPLIT + tile
            bound = row_position(s) + 1
            request_start = index_request_start(s)
            first_position = row_position(request_start)
            shard_begin, shard_size = index_shard(s)
            group_size = shard_size
            if const_expr(INDEX_QUERY_REUSE > 1):
                _, last_size = index_shard(s + INDEX_QUERY_REUSE - 1)
                group_size = fx.max(group_size, last_size)
            if tile * INDEX_KEYS_PER_TASK < group_size:
                if const_expr(_partition == "index"):
                    if tid < INDEX_HEADS * INDEX_QUERY_REUSE:
                        word = fx.Int32(
                            bo.buffer_load(
                                _rsrc(mb("index_w")),
                                (s * INDEX_HEADS + tid) * 2,
                                vec_width=1,
                                dtype=T.i32,
                                cache_modifier=CM_DEV,
                            )
                        )
                        lds_st(keys, tid, word)
                else:
                    if const_expr(indexer_quant):
                        poll(
                            [(mb("index_ready"), s + query, 1) for query in range(INDEX_QUERY_REUSE)],
                            batch=INDEX_QUERY_REUSE,
                        )
                    else:
                        for query in range_constexpr(INDEX_QUERY_REUSE):
                            get(mb("index_ready"), s + query)
                    if tid < INDEX_HEADS * INDEX_QUERY_REUSE:
                        lds_st(keys, tid, get(mb("index_w"), s * INDEX_HEADS + tid))
                # Every key group reuses the same 32x128 query.  Stage and RoPE
                # it once per scoring CTA instead of polling and rotating it in
                # each of the four key-group wave pairs.
                # Peer reads benefit from independent outstanding polls.
                # Keep the original one-pair staging order when CP is off.
                if const_expr(indexer_quant):
                    # Each wave owns 32 keys and all 32 heads. Two scaled
                    # 32x32x64 MFMAs accumulate the complete 128-wide dot.
                    # Keep one query's operands live while sharing K across the group.
                    # Stage the shared query once per CTA, instead of
                    # polling the same fragments independently in every wave.
                    Q_WORDS = INDEX_HEADS * 20 * INDEX_QUERY_REUSE
                    Q_BATCHES = (Q_WORDS + 2 * THREADS - 1) // (2 * THREADS)
                    if const_expr(_partition == "index"):
                        q_pairs = []
                        for qbatch in range_constexpr(Q_BATCHES):
                            qword = qbatch * 2 * THREADS + tid * 2
                            qword = fx.min(fx.Int32(Q_WORDS - 2), qword)
                            raw = fx.Vector(
                                bo.buffer_load(
                                    _rsrc(mb("index_q_quant")),
                                    (s * INDEX_HEADS * 20 + qword) * 2,
                                    vec_width=4,
                                    dtype=T.i32,
                                    cache_modifier=CM_DEV,
                                )
                            )
                            q_pairs.append(fx.Vector.from_elements([raw[0], raw[2]], fx.Int32))
                    else:
                        q_pairs = poll(
                            [
                                (
                                    mb("index_q_quant"),
                                    s * INDEX_HEADS * 20
                                    + (
                                        qbatch * 2 * THREADS + tid * 2
                                        if (qbatch + 1) * 2 * THREADS <= Q_WORDS
                                        else fx.min(fx.Int32(Q_WORDS - 2), qbatch * 2 * THREADS + tid * 2)
                                    ),
                                    2,
                                )
                                for qbatch in range(Q_BATCHES)
                            ],
                            batch=Q_BATCHES,
                        )
                    for q_batch4 in range_constexpr(Q_BATCHES):
                        qword = q_batch4 * 2 * THREADS + tid * 2
                        if qword < Q_WORDS:
                            lds_st(xs, qword, q_pairs[q_batch4][0].bitcast(fx.Float32))
                            lds_st(xs, qword + 1, q_pairs[q_batch4][1].bitcast(fx.Float32))
                else:
                    q_batch = (INDEX_Q_ROWS // 2) // THREADS if indexer_cp else 1
                    for batch in range_constexpr(0, (INDEX_Q_ROWS // 2) // THREADS, q_batch):
                        q_pairs = [tid + (batch + b) * THREADS for b in range(q_batch)]
                        q_source = (
                            cp_buffer("index_cp_q", sym)
                            if const_expr(SHARD_INDEX_Q)
                            else (mb("ugp") + fx.Int64(INDEX_ROT_OFF * 8) if INDEX_PRE_ROPE else mb("index_q"))
                        )
                        q_words = poll(
                            [(q_source, s * INDEX_Q_ROWS // 2 + pair, 1) for pair in q_pairs],
                            scope="system" if SHARD_INDEX_Q else "agent",
                        )
                        for q_pair, q_word in zip(q_pairs, q_words):
                            if const_expr(INDEX_PRE_ROPE):
                                lds_st(xs, q_pair, q_word[0].bitcast(fx.Float32))
                            else:
                                q_elem = q_pair * 2
                                kq = q_elem % INDEX_DIM
                                q0, q1 = bf2_f32(q_word[0])
                                if kq < PE_DIM:
                                    c = ld_rope(
                                        _rsrc(rope_cos),
                                        row_position(s) * (PE_DIM // 2) + kq // 2,
                                    )
                                    sn = ld_rope(
                                        _rsrc(rope_sin),
                                        row_position(s) * (PE_DIM // 2) + kq // 2,
                                    )
                                    q0, q1 = q0 * c - q1 * sn, q0 * sn + q1 * c
                                lds_st(xs, q_pair, bf16_pair(q0, q1))
                gpu.barrier()
                if const_expr(indexer_quant):
                    # Adjacent queries can straddle one CP stride boundary.
                    # Preload the middle query's shard so the majority keeps
                    # sharing K; each minority query retains its exact reload.
                    reuse_shard_begin, _ = index_shard(s + INDEX_QUERY_REUSE // 2)
                    # Issue the four independent key payload/scale loads
                    # before consuming their MFMA operands.
                    key_tiles4 = []
                    for preload_part in range_constexpr(0, INDEX_SCORE_REUSE, 4):
                        preload_tile = index_tile_base + preload_part + wave // 2
                        preload_key = preload_tile * INDEX_KEYS_PER_TASK + (wave % 2) * 32 + lane % 32
                        safe_key = fx.min(reuse_shard_begin + preload_key, row_position(s + INDEX_QUERY_REUSE - 1))
                        key_base4 = indices + fx.Int64(index_page_slot(s, safe_key)) * INDEX_FP4_BYTES
                        operands4 = []
                        for kstep in range_constexpr(2):
                            key_ptr4 = fx.inttoptr(
                                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4),
                                key_base4 + fx.Int64(kstep * 32 + (lane // 32) * 16),
                            )
                            kv4 = fx.generic_load(key_ptr4, count=4)
                            scale_ptr4 = fx.inttoptr(
                                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4),
                                key_base4 + fx.Int64(INDEX_DIM // 2),
                            )
                            raw_scale4 = fx.Int32(fx.generic_load(scale_ptr4))
                            operands4.append((kv4, raw_scale4))
                        key_tiles4.append(operands4)
                    for query in range_constexpr(INDEX_QUERY_REUSE):
                        queries4 = []
                        scales4 = []
                        for half in range_constexpr(2):
                            head4 = lane % 32
                            qbase4 = (query * INDEX_HEADS + head4) * 20
                            queries4.append(
                                fx.Vector.from_elements(
                                    [
                                        lds_ld(xs, qbase4 + (half % 2) * 8 + (lane // 32) * 4 + j).bitcast(fx.Int32)
                                        for j in range(4)
                                    ],
                                    fx.Int32,
                                )
                            )
                            scales4.append(lds_ld(xs, qbase4 + 16 + (half % 2) * 2 + lane // 32).bitcast(fx.Int32))
                        weights4 = [
                            lds_ld(keys, query * INDEX_HEADS + (lane // 32) * 4 + (e // 4) * 8 + e % 4).bitcast(
                                fx.Float32
                            )
                            for e in range(16)
                        ]
                        query_bound = row_position(s + query) + 1
                        query_begin, query_size = index_shard(s + query)
                        for index_part in range_constexpr(0, INDEX_SCORE_REUSE, 4):
                            tile = index_tile_base + index_part + wave // 2
                            tt = (s + query) * N_INDEX_SPLIT + tile
                            if tile * INDEX_KEYS_PER_TASK < query_size:
                                local_key = tile * INDEX_KEYS_PER_TASK + (wave % 2) * 32 + lane % 32
                                key_pos = query_begin + local_key
                                safe_key = fx.min(key_pos, query_bound - 1)
                                frag4 = fx.full(16, 0.0, fx.Float32)
                                # Prefetch only this key batch's two MFMA operands.
                                # Bound live alternate operands to preserve two-CTA residency.
                                selected_tiles4 = []
                                for load_step in range_constexpr(2):
                                    selected_kv, selected_scale = key_tiles4[index_part // 4][load_step]
                                    if const_expr(INDEX_QUERY_REUSE > 1):
                                        if query_begin != reuse_shard_begin:
                                            other_base4 = (
                                                indices
                                                + fx.Int64(index_page_slot(s + query, safe_key)) * INDEX_FP4_BYTES
                                            )
                                            other_ptr4 = fx.inttoptr(
                                                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4),
                                                other_base4 + fx.Int64(load_step * 32 + (lane // 32) * 16),
                                            )
                                            selected_kv = fx.generic_load(other_ptr4, count=4)
                                            other_scale4 = fx.inttoptr(
                                                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4),
                                                other_base4 + fx.Int64(INDEX_DIM // 2),
                                            )
                                            selected_scale = fx.Int32(fx.generic_load(other_scale4))
                                    selected_tiles4.append((selected_kv, selected_scale))
                                for kstep in range_constexpr(2):
                                    kv4, raw_scale4 = selected_tiles4[kstep]
                                    ks4 = (raw_scale4 >> ((kstep * 2 + lane // 32) * 8)) & 255
                                    if safe_key >= first_position:
                                        sn = request_start + safe_key - first_position
                                        new4 = poll(
                                            [
                                                (
                                                    mb("index_k_quant"),
                                                    sn * 20 + kstep * 8 + (lane // 32) * 4,
                                                    2,
                                                ),
                                                (
                                                    mb("index_k_quant"),
                                                    sn * 20 + kstep * 8 + (lane // 32) * 4 + 2,
                                                    2,
                                                ),
                                                (
                                                    mb("index_k_quant"),
                                                    sn * 20 + 16 + kstep * 2 + lane // 32,
                                                    1,
                                                ),
                                            ]
                                        )
                                        kv4 = fx.Vector.from_elements([new4[j // 2][j % 2] for j in range(4)], fx.Int32)
                                        ks4 = new4[2][0]
                                    frag4 = fx.Vector(
                                        rocdl.mfma_scale_f32_32x32x64_f8f6f4(
                                            T.vec(16, T.f32),
                                            [
                                                queries4[kstep],
                                                kv4,
                                                frag4,
                                                4,
                                                4,
                                                0,
                                                scales4[kstep],
                                                0,
                                                ks4,
                                            ],
                                        )
                                    )
                                partial4 = fx.Float32(0.0)
                                for e in range_constexpr(16):
                                    partial4 = partial4 + fx.max(frag4[e], fx.Float32(0.0)) * weights4[e]
                                score4 = _xred(partial4, 32, lambda a, b: a + b)
                                if (lane < 32) & (key_pos < query_bound):
                                    if const_expr(indexer_cp):
                                        if query_bound <= LOCAL_INDEX_CAPACITY:
                                            for dst in range_constexpr(CP_RANKS):
                                                put(
                                                    cp_buffer("index_cp_scores", cp_peer(fx.Int32(dst))),
                                                    (s + query) * LOCAL_INDEX_CAPACITY + key_pos,
                                                    score4,
                                                    CM_SYS,
                                                )
                                        else:
                                            put(
                                                mb("index_scores"),
                                                (s + query) * INDEX_CAPACITY + local_key,
                                                score4,
                                            )
                                    else:
                                        put(
                                            mb("index_scores"),
                                            (s + query) * INDEX_CAPACITY + local_key,
                                            score4,
                                        )
                    # No wave may overwrite the next group's weights before
                    # every wave has consumed this group's LDS values.
                    gpu.barrier()
                else:
                    # Keep the query tile live while scoring adjacent key tiles.
                    for index_part in range_constexpr(INDEX_SCORE_REUSE):
                        tile = index_tile_base + index_part
                        tt = s * N_INDEX_SPLIT + tile
                        if tile * INDEX_KEYS_PER_TASK < shard_size:
                            key_group = wave // 2
                            head_group = wave % 2
                            local_key = tile * INDEX_KEYS_PER_TASK + key_group * 16 + lane % 16
                            key_pos = shard_begin + local_key
                            safe_key = fx.min(key_pos, bound - 1)
                            head = head_group * 16 + lane % 16
                            score_frag = fx.full(4, 0.0, fx.Float32)
                            for k32 in range_constexpr(INDEX_DIM // 32):
                                k = k32 * 32 + (lane // 16) * 8
                                qv = load_index_q8(head, k)
                                if const_expr(TILED_INDEXER):
                                    # Lanes may refer to different physical pages. A global
                                    # pointer preserves the full address without assuming
                                    # contiguous pages or narrowing to a buffer byte offset.
                                    key_addr = (
                                        indices
                                        + fx.Int64(index_page_slot(s, safe_key)) * (INDEX_DIM * 2)
                                        + fx.Int64(k * 2)
                                    )
                                    key_ptr = fx.inttoptr(
                                        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 16),
                                        key_addr,
                                    )
                                    kv = fx.generic_load(key_ptr, count=4).bitcast(fx.BFloat16)
                                else:
                                    kv = fx.Vector(
                                        bo.buffer_load(
                                            r_index_cache,
                                            (index_page_slot(s, safe_key) * INDEX_DIM + k) // 2,
                                            vec_width=4,
                                            dtype=T.i32,
                                        )
                                    ).bitcast(fx.BFloat16)
                                if safe_key >= first_position:
                                    sn = request_start + safe_key - first_position
                                    kv_pairs = get_bf2_many(
                                        [(mb("index_k_new"), sn * INDEX_DIM + k + j * 2) for j in range(4)]
                                    )
                                    kv_values = []
                                    for pair in kv_pairs:
                                        kv_values += list(pair)
                                    kv = fx.Vector.from_elements(kv_values, fx.Float32).to(fx.BFloat16)
                                score_frag = fx.Vector(
                                    rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [qv, kv, score_frag])
                                )
                            partial = fx.Float32(0.0)
                            for e in range_constexpr(4):
                                index_head = head_group * 16 + (lane // 16) * 4 + e
                                weight = lds_ld(keys, index_head).bitcast(fx.Float32)
                                partial = partial + fx.max(score_frag[e], fx.Float32(0.0)) * weight
                            partial = _xred(partial, 16, lambda a, b: a + b)
                            partial = _xred(partial, 32, lambda a, b: a + b)
                            if lane < 16:
                                lds_st(red, wave * 16 + lane, partial)
                            gpu.barrier()
                            if (wave % 2 == 0) & (lane < 16) & (key_pos < bound):
                                score = lds_ld(red, wave * 16 + lane) + lds_ld(red, (wave + 1) * 16 + lane)
                                if const_expr(indexer_cp):
                                    if bound <= LOCAL_INDEX_CAPACITY:
                                        lds_st(red, 128 + key_group * 16 + lane, score)
                                    else:
                                        put(
                                            mb("index_scores"),
                                            s * INDEX_CAPACITY + local_key,
                                            score,
                                        )
                                else:
                                    put(
                                        mb("index_scores"),
                                        s * INDEX_CAPACITY + local_key,
                                        score,
                                    )
                            if const_expr(indexer_cp):
                                if bound <= LOCAL_INDEX_CAPACITY:
                                    # One wave per peer, one complete score exchange.
                                    gpu.barrier()
                                    out_key = shard_begin + tile * INDEX_KEYS_PER_TASK + lane
                                    if (wave < CP_RANKS) & (out_key < bound):
                                        put(
                                            cp_buffer(
                                                "index_cp_scores",
                                                (cp_peer(wave) if const_expr(CP_RANKS < W) else peer_dst),
                                            ),
                                            s * LOCAL_INDEX_CAPACITY + out_key,
                                            lds_ld(red, 128 + lane),
                                            CM_SYS,
                                        )
                        if const_expr(index_part + 1 < INDEX_SCORE_REUSE):
                            # All readers finish before another key tile reuses red.
                            gpu.barrier()
            else:
                for query in range_constexpr(INDEX_QUERY_REUSE):
                    for index_part in range_constexpr(INDEX_SCORE_REUSE):
                        tt = (s + query) * N_INDEX_SPLIT + index_tile_base + index_part

        if const_expr(TILED_INDEXER):
            # Context-parallel exact radix selection. Scoring owns a
            # contiguous, 64-key-aligned shard of the *live* context on each
            # rank. Histograms and compaction use bounded key-tile tasks;
            # neither LDS nor the compiled loop body scales with context.
            # Each rank completes its local stage before entering the next
            # one. All producers are resident, including rank-0 coordinators.
            def cp_get(name, i):
                return poll([(cp_buffer(name, sym), i, 1)], scope="system")[0][0]

            def cp_threshold(digit, s):
                values = poll(
                    [
                        (
                            cp_buffer("index_cp_threshold", sym),
                            (digit * S + s) * 2,
                            2,
                        )
                    ],
                    scope="system",
                )[0]
                return fx.Uint32(_uniform(values[0])), _uniform(values[1])

            def score_tile(s, part, size):
                """Read independent score mailboxes in one bounded batch."""
                count = INDEX_CP_TILE // THREADS
                bits = fx.full(count, 0, fx.Int32)
                if (part + 1) * INDEX_CP_TILE <= size:
                    words = poll(
                        [
                            (
                                mb("index_scores"),
                                s * INDEX_CAPACITY + part * INDEX_CP_TILE + b * THREADS + tid,
                                1,
                            )
                            for b in range(count)
                        ],
                        batch=count,
                    )
                    bits = fx.Vector.from_elements([word[0] for word in words], fx.Int32)
                else:
                    tail = []
                    for b in range_constexpr(count):
                        i = part * INDEX_CP_TILE + b * THREADS + tid
                        value = fx.Int32(0)
                        if i < size:
                            value = get(mb("index_scores"), s * INDEX_CAPACITY + i)
                        tail.append(value)
                    bits = fx.Vector.from_elements(tail, fx.Int32)
                return fx.Vector.from_elements(
                    [fx.Uint32((bits[b] >= 0).select(bits[b] ^ fx.Int32(-(2**31)), ~bits[b])) for b in range(count)],
                    fx.Uint32,
                )

            # Scoring and radix tasks have disjoint LDS lifetimes.
            # Keep the scan scratch in keys and reuse Q staging for FP4 bins.
            cp_bins = fx.recast_iter(fx.Int32, xs) if const_expr(indexer_quant) else keys

            def cp_scan(value):
                """Exclusive scan in ascending thread order; all threads participate."""
                inclusive = fx.coop.warp_inclusive_scan(value, fx.ReductionOp.ADD, width=64)
                if lane == 63:
                    lds_st(keys, 256 + wave, inclusive)
                gpu.barrier()
                before = fx.Int32(0)
                total = fx.Int32(0)
                for w in range_constexpr(WAVES):
                    count = lds_ld(keys, 256 + w)
                    before = before + (wave > w).select(count, fx.Int32(0))
                    total = total + count
                offset = before + inclusive - value
                gpu.barrier()
                return offset, total

            def cp_select_digit(shift, prefix, remain, bins=INDEX_RADIX_BINS):
                if const_expr(indexer_quant):
                    # Consecutive descending buckets per thread preserve global order.
                    items = bins // THREADS
                    counts = [lds_ld(cp_bins, bins - 1 - tid * items - j) for j in range(items)]
                    subtotal = fx.Int32(0)
                    for count in counts:
                        subtotal = subtotal + count
                    above, _ = cp_scan(subtotal)
                    for j in range_constexpr(items):
                        count = counts[j]
                        hit = (above < remain) & (above + count >= remain)
                        if hit:
                            bucket = bins - 1 - tid * items - j
                            lds_st(keys, 256, fx.Int32(prefix | (fx.Uint32(bucket) << shift)))
                            lds_st(keys, 257, remain - above)
                        above = above + count
                    gpu.barrier()
                    result = fx.Uint32(lds_ld(keys, 256)), lds_ld(keys, 257)
                    gpu.barrier()
                    return result

                else:
                    digit = 255 - fx.min(tid, 255)
                    count = (tid < 256).select(lds_ld(keys, digit), fx.Int32(0))
                    above, _ = cp_scan(count)
                    hit = (tid < 256) & (above < remain) & (above + count >= remain)
                    if hit:
                        lds_st(
                            keys,
                            256,
                            fx.Int32(prefix | (fx.Uint32(digit) << shift)),
                        )
                        lds_st(keys, 257, remain - above)
                    gpu.barrier()
                    result = fx.Uint32(lds_ld(keys, 256)), lds_ld(keys, 257)
                    gpu.barrier()
                    return result

            # The FP4 selector reuses the existing histogram allocation:
            # raw accumulated bins, tagged task completions, then init tags.
            # Even CP_PARTS=1 leaves enough space in the original paired bins.
            HIST_ACCUM_BYTES = INDEX_RADIX_DIGITS * S * INDEX_RADIX_BINS * 4
            hist_done = mb("index_cp_hist") + fx.Int64(HIST_ACCUM_BYTES)
            hist_init = hist_done + fx.Int64(INDEX_RADIX_DIGITS * S * CP_PARTS * 8)

            def hist_pointer(digit, sample, bucket):
                return fx.inttoptr(
                    fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4),
                    mb("index_cp_hist") + fx.Int64(((digit * S + sample) * INDEX_RADIX_BINS + bucket) * 4),
                )

            if const_expr(indexer_quant and not RESIDENT_INDEX_SELECT):
                for sample in range(bid, S, G):
                    sample = fx.Int32(sample)
                    if use_long_index(sample):
                        for bin_chunk in range_constexpr(max(1, INDEX_RADIX_BINS // THREADS)):
                            bucket_lane = bin_chunk * THREADS + tid
                            if bucket_lane < INDEX_RADIX_BINS:
                                for digit in range_constexpr(INDEX_RADIX_DIGITS):
                                    fx.generic_store(hist_pointer(digit, sample, bucket_lane), fx.Int32(0))
                        fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope="agent")
                        gpu.barrier()
                        if tid == 0:
                            put(hist_init, sample, fx.Int32(1))

            cp_rank = cp_local_rank() if const_expr(indexer_cp) else fx.Int32(0)
            owner = cp_peer(fx.Int32(0))

            def resident_hist(digit, s, part, tt, cached=False):
                ACTIVE_BINS = 1024 if digit == 0 else INDEX_RADIX_BINS
                shift = (22 - digit * 11) if indexer_quant else (24 - digit * 8)
                mask = (0xFFFFFFFF << (shift + (11 if indexer_quant else 8))) & 0xFFFFFFFF
                if use_long_index(s):
                    # Cached radix digits execute consecutively in one CTA.
                    # Digit zero already acquired this invocation's init tag;
                    # raw histogram storage is not reset again within the call.
                    if const_expr(indexer_quant and (not cached or digit == 0)):
                        get(hist_init, s)
                        # The clear completed before its coherent tag.
                        # Atomics access L2 directly; only compiler
                        # ordering is needed after the tagged wait.
                        llvm.InlineAsmOp(None, [], "s_nop 0", "~{memory}", has_side_effects=True)
                    _, size = index_shard(s)
                    prefix = fx.Uint32(0)
                    if const_expr(digit > 0):
                        prefix, _ = cp_threshold(digit - 1, s)
                    for bin_chunk in range_constexpr(max(1, ACTIVE_BINS // THREADS)):
                        bucket_lane = bin_chunk * THREADS + tid
                        if bucket_lane < ACTIVE_BINS:
                            lds_st(cp_bins, bucket_lane, fx.Int32(0))
                    gpu.barrier()
                    if const_expr(cached):
                        tile_keys = fx.Vector.from_elements(
                            [
                                fx.Uint32(lds_ld(xs, INDEX_RADIX_BINS + batch * THREADS + tid).bitcast(fx.Int32))
                                for batch in range(INDEX_CP_TILE // THREADS)
                            ],
                            fx.Uint32,
                        )
                    else:
                        tile_keys = score_tile(s, part, size)
                    for batch in range_constexpr(INDEX_CP_TILE // THREADS):
                        i = part * INDEX_CP_TILE + batch * THREADS + tid
                        if i < size:
                            key = tile_keys[batch]
                            if (key & fx.Uint32(mask)) == prefix:
                                bucket = fx.Int32((key >> shift) & fx.Uint32(INDEX_RADIX_BINS - 1))
                                fx.atomic_add(
                                    cp_bins + bucket,
                                    fx.Int32(1),
                                    syncscope="workgroup",
                                )
                    gpu.barrier()
                    if const_expr(indexer_quant):
                        for bin_chunk in range_constexpr(max(1, ACTIVE_BINS // THREADS)):
                            bucket_lane = bin_chunk * THREADS + tid
                            if bucket_lane < ACTIVE_BINS:
                                count = lds_ld(cp_bins, bucket_lane)
                                if count != 0:
                                    fx.atomic_add(hist_pointer(digit, s, bucket_lane), count, syncscope="agent")
                        # Match gfx950 native top-k: atomics already
                        # commit at L2. Drain them before publishing the
                        # tag, without writing back the entire L2 cache.
                        llvm.InlineAsmOp(None, [], "s_waitcnt vmcnt(0)", "~{memory}", has_side_effects=True)
                        gpu.barrier()
                        if tid == 0:
                            put(hist_done, digit * S * CP_PARTS + tt, fx.Int32(1))
                    else:
                        for bin_chunk in range_constexpr(max(1, ACTIVE_BINS // THREADS)):
                            bucket_lane = bin_chunk * THREADS + tid
                            if bucket_lane < ACTIVE_BINS:
                                put(
                                    mb("index_cp_hist"),
                                    (digit * S * CP_PARTS + tt) * INDEX_RADIX_BINS + bucket_lane,
                                    lds_ld(cp_bins, bucket_lane),
                                )
                        gpu.barrier()

            def resident_prefix(digit, s):
                ACTIVE_BINS = 1024 if digit == 0 else INDEX_RADIX_BINS
                shift = (22 - digit * 11) if indexer_quant else (24 - digit * 8)
                if use_long_index(s):
                    if const_expr(indexer_quant):
                        for batch in range_constexpr((CP_PARTS + THREADS - 1) // THREADS):
                            part = batch * THREADS + tid
                            if part < CP_PARTS:
                                get(hist_done, (digit * S + s) * CP_PARTS + part)
                        gpu.barrier()
                        # Completion tags follow drained L2 atomics.
                        # Read counters coherently, so no cached stale
                        # counter can survive from an earlier launch.
                        llvm.InlineAsmOp(None, [], "s_nop 0", "~{memory}", has_side_effects=True)
                    if const_expr(indexer_quant):
                        # Two adjacent integer bins per thread; tagged layout unchanged.
                        for bin_chunk in range_constexpr(ACTIVE_BINS // (2 * THREADS)):
                            bucket_lane = bin_chunk * (2 * THREADS) + 2 * tid
                            count_pair = fx.Vector(
                                bo.buffer_load(
                                    _rsrc(mb("index_cp_hist")),
                                    (digit * S + s) * INDEX_RADIX_BINS + bucket_lane,
                                    vec_width=2,
                                    dtype=T.i32,
                                    cache_modifier=CM_DEV,
                                )
                            )
                            peer_pair = (
                                (cp_rank * INDEX_RADIX_DIGITS + digit) * S + s
                            ) * INDEX_RADIX_BINS + bucket_lane
                            bo.buffer_store(
                                fx.Vector.from_elements([count_pair[0], tag, count_pair[1], tag], fx.Int32),
                                _rsrc(cp_buffer("index_cp_hist", cp_peer(s % CP_RANKS))),
                                peer_pair * 2,
                                cache_modifier=CM_SYS,
                            )
                    else:
                        for bin_chunk in range_constexpr(max(1, ACTIVE_BINS // THREADS)):
                            bucket_lane = bin_chunk * THREADS + tid
                            if bucket_lane < ACTIVE_BINS:
                                if const_expr(indexer_quant):
                                    count = fx.Int32(
                                        bo.buffer_load(
                                            _rsrc(mb("index_cp_hist")),
                                            (digit * S + s) * INDEX_RADIX_BINS + bucket_lane,
                                            vec_width=1,
                                            dtype=T.i32,
                                            cache_modifier=CM_DEV,
                                        )
                                    )
                                else:
                                    count = fx.Int32(0)
                                    # Independent partition counters can be polled together;
                                    # each batch waits only for its last producer.
                                    for part_base in range(0, CP_PARTS - CP_PARTS % 8, 8):
                                        counts = poll(
                                            [
                                                (
                                                    mb("index_cp_hist"),
                                                    ((digit * S + s) * CP_PARTS + part_base + p) * INDEX_RADIX_BINS
                                                    + bucket_lane,
                                                    1,
                                                )
                                                for p in range(8)
                                            ],
                                            batch=8,
                                        )
                                        for values in counts:
                                            count = count + values[0]
                                    for part in range_constexpr(CP_PARTS - CP_PARTS % 8, CP_PARTS):
                                        count = count + get(
                                            mb("index_cp_hist"),
                                            ((digit * S + s) * CP_PARTS + part) * INDEX_RADIX_BINS + bucket_lane,
                                        )
                                put(
                                    cp_buffer(
                                        "index_cp_hist",
                                        cp_peer(s % CP_RANKS) if const_expr(indexer_quant) else owner,
                                    ),
                                    ((cp_rank * INDEX_RADIX_DIGITS + digit) * S + s) * INDEX_RADIX_BINS + bucket_lane,
                                    count,
                                    CM_SYS,
                                )
                    gpu.barrier()
                    if cp_rank == (s % CP_RANKS if const_expr(indexer_quant) else fx.Int32(0)):
                        if const_expr(indexer_quant):
                            for bin_chunk in range_constexpr(ACTIVE_BINS // (2 * THREADS)):
                                bucket_lane = bin_chunk * (2 * THREADS) + 2 * tid
                                count0, count1 = fx.Int32(0), fx.Int32(0)
                                counts = poll(
                                    [
                                        (
                                            cp_buffer("index_cp_hist", sym),
                                            ((src * INDEX_RADIX_DIGITS + digit) * S + s) * INDEX_RADIX_BINS
                                            + bucket_lane,
                                            2,
                                        )
                                        for src in range(CP_RANKS)
                                    ],
                                    batch=CP_RANKS,
                                    scope="system",
                                )
                                for values in counts:
                                    count0 = count0 + values[0]
                                    count1 = count1 + values[1]
                                lds_st(cp_bins, bucket_lane, count0)
                                lds_st(cp_bins, bucket_lane + 1, count1)
                        else:
                            for bin_chunk in range_constexpr(max(1, ACTIVE_BINS // THREADS)):
                                bucket_lane = bin_chunk * THREADS + tid
                                if bucket_lane < ACTIVE_BINS:
                                    count = fx.Int32(0)
                                    if const_expr(indexer_quant):
                                        counts = poll(
                                            [
                                                (
                                                    cp_buffer("index_cp_hist", sym),
                                                    ((src * INDEX_RADIX_DIGITS + digit) * S + s) * INDEX_RADIX_BINS
                                                    + bucket_lane,
                                                    1,
                                                )
                                                for src in range(CP_RANKS)
                                            ],
                                            batch=CP_RANKS,
                                            scope="system",
                                        )
                                        for values in counts:
                                            count = count + values[0]
                                    else:
                                        for src in range_constexpr(CP_RANKS):
                                            count = count + cp_get(
                                                "index_cp_hist",
                                                ((src * INDEX_RADIX_DIGITS + digit) * S + s) * INDEX_RADIX_BINS
                                                + bucket_lane,
                                            )
                                    lds_st(cp_bins, bucket_lane, count)
                        gpu.barrier()
                        prefix = fx.Uint32(0)
                        remain = fx.min(fx.Int32(topk), row_position(s) + 1)
                        if const_expr(digit > 0):
                            prefix, remain = cp_threshold(digit - 1, s)
                        prefix, remain = cp_select_digit(shift, prefix, remain, bins=ACTIVE_BINS)
                        # One wave-uniform destination per publishing wave.
                        # Keep prefix/remain as integer bits and publish both
                        # tagged pairs with the existing coherent store format.
                        if (lane == 0) & (wave < CP_RANKS):
                            address = cp_buffer("index_cp_threshold", cp_peer(fx.Int32(wave)))
                            bo.buffer_store(
                                fx.Vector.from_elements([fx.Int32(prefix), tag, remain, tag], fx.Int32),
                                _rsrc(address),
                                (digit * S + s) * 4,
                                cache_modifier=CM_SYS,
                            )
                    gpu.barrier()

            def resident_count(s, part, tt, cached=False):
                if use_long_index(s):
                    _, size = index_shard(s)
                    threshold, _ = cp_threshold(INDEX_RADIX_DIGITS - 1, s)
                    n_gt, n_eq = fx.Int32(0), fx.Int32(0)
                    if const_expr(cached):
                        tile_keys = fx.Vector.from_elements(
                            [
                                fx.Uint32(lds_ld(xs, INDEX_RADIX_BINS + batch * THREADS + tid).bitcast(fx.Int32))
                                for batch in range(INDEX_CP_TILE // THREADS)
                            ],
                            fx.Uint32,
                        )
                    else:
                        tile_keys = score_tile(s, part, size)
                    for batch in range_constexpr(INDEX_CP_TILE // THREADS):
                        i = part * INDEX_CP_TILE + batch * THREADS + tid
                        if i < size:
                            key = tile_keys[batch]
                            n_gt = n_gt + (key > threshold).select(fx.Int32(1), fx.Int32(0))
                            n_eq = n_eq + (key == threshold).select(fx.Int32(1), fx.Int32(0))
                    # Each partition has at most 9216 keys. Pack two
                    # independent 16-bit counters into one scan without
                    # carry from the strict-winner count into the tie count.
                    _, packed_counts = cp_scan(n_gt | (n_eq << 16))
                    total_gt = packed_counts & 65535
                    total_eq = fx.Int32(fx.Uint32(packed_counts) >> 16)
                    if tid == 0:
                        put(mb("index_cp_counts"), tt * 2, total_gt)
                        put(mb("index_cp_counts"), tt * 2 + 1, total_eq)
                    gpu.barrier()

            def resident_offsets(s):
                if use_long_index(s):
                    total_gt, total_eq = fx.Int32(0), fx.Int32(0)
                    for batch in range_constexpr((CP_PARTS + THREADS - 1) // THREADS):
                        part = batch * THREADS + tid
                        n_gt, n_eq = fx.Int32(0), fx.Int32(0)
                        if part < CP_PARTS:
                            if const_expr(indexer_quant):
                                n_gt, n_eq = poll([(mb("index_cp_counts"), (s * CP_PARTS + part) * 2, 2)])[0]
                            else:
                                n_gt = get(mb("index_cp_counts"), (s * CP_PARTS + part) * 2)
                                n_eq = get(
                                    mb("index_cp_counts"),
                                    (s * CP_PARTS + part) * 2 + 1,
                                )
                        off_gt, batch_gt = cp_scan(n_gt)
                        off_eq, batch_eq = cp_scan(n_eq)
                        if part < CP_PARTS:
                            put(
                                mb("index_cp_offsets"),
                                (s * CP_PARTS + part) * 2,
                                total_gt + off_gt,
                            )
                            put(
                                mb("index_cp_offsets"),
                                (s * CP_PARTS + part) * 2 + 1,
                                total_eq + off_eq,
                            )
                        total_gt = total_gt + batch_gt
                        total_eq = total_eq + batch_eq
                    if const_expr(indexer_quant):
                        # Rank r contributes only to prefixes of higher ranks.
                        # One wave-uniform destination per sender wave.
                        if (lane == 0) & (wave < CP_RANKS) & (wave > cp_rank):
                            bo.buffer_store(
                                fx.Vector.from_elements([total_gt, tag, total_eq, tag], fx.Int32),
                                _rsrc(cp_buffer("index_cp_counts", cp_peer(fx.Int32(wave)))),
                                (cp_rank * S + s) * 4,
                                cache_modifier=CM_SYS,
                            )
                    else:
                        if tid == 0:
                            put(
                                cp_buffer(
                                    "index_cp_counts",
                                    cp_peer(s % CP_RANKS) if const_expr(indexer_quant) else owner,
                                ),
                                (cp_rank * S + s) * 2,
                                total_gt,
                                CM_SYS,
                            )
                            put(
                                cp_buffer(
                                    "index_cp_counts",
                                    cp_peer(s % CP_RANKS) if const_expr(indexer_quant) else owner,
                                ),
                                (cp_rank * S + s) * 2 + 1,
                                total_eq,
                                CM_SYS,
                            )
                    gpu.barrier()
                    if const_expr(indexer_quant):
                        # Consume direct count pushes from lower ranks in local
                        # symmetric storage, then publish this rank's prefix.
                        # No centralized count owner or return hop is involved.
                        for destination in range_constexpr(CP_RANKS):
                            if cp_rank == destination:
                                if tid == 0:
                                    counts = poll(
                                        [
                                            (cp_buffer("index_cp_counts", sym), (src * S + s) * 2, 2)
                                            for src in range(destination)
                                        ],
                                        batch=CP_RANKS,
                                        scope="system",
                                    )
                                    before_gt, before_eq = fx.Int32(0), fx.Int32(0)
                                    for values in counts:
                                        before_gt = before_gt + values[0]
                                        before_eq = before_eq + values[1]
                                    bo.buffer_store(
                                        fx.Vector.from_elements([before_gt, tag, before_eq, tag], fx.Int32),
                                        _rsrc(cp_buffer("index_cp_offsets", sym)),
                                        s * 4,
                                        cache_modifier=CM_SYS,
                                    )
                    else:
                        if cp_rank == 0:
                            before_gt, before_eq = fx.Int32(0), fx.Int32(0)
                            for src in range_constexpr(CP_RANKS):
                                n_gt = cp_get("index_cp_counts", (src * S + s) * 2)
                                n_eq = cp_get("index_cp_counts", (src * S + s) * 2 + 1)
                                if tid == 0:
                                    address = cp_buffer("index_cp_offsets", cp_peer(fx.Int32(src)))
                                    put(address, s * 2, before_gt, CM_SYS)
                                    put(address, s * 2 + 1, before_eq, CM_SYS)
                                before_gt = before_gt + n_gt
                                before_eq = before_eq + n_eq
                    gpu.barrier()

            def resident_emit(s, part, tt, cached=False):
                if use_long_index(s):
                    begin, size = index_shard(s)
                    threshold, need_eq = cp_threshold(INDEX_RADIX_DIGITS - 1, s)
                    if const_expr(indexer_quant):
                        peer_offsets = poll([(cp_buffer("index_cp_offsets", sym), s * 2, 2)], scope="system")[0]
                        local_offsets = poll([(mb("index_cp_offsets"), tt * 2, 2)])[0]
                        before_gt = peer_offsets[0] + local_offsets[0]
                        before_eq = peer_offsets[1] + local_offsets[1]
                    else:
                        before_gt = cp_get("index_cp_offsets", s * 2) + get(mb("index_cp_offsets"), tt * 2)
                        before_eq = cp_get("index_cp_offsets", s * 2 + 1) + get(mb("index_cp_offsets"), tt * 2 + 1)
                    out_gt = fx.min(fx.Int32(topk), row_position(s) + 1) - need_eq
                    # Each lane owns a fixed contiguous key group. Scan its
                    # total counts once, then emit in lane/key order.
                    # This preserves ascending IDs with one CTA scan per
                    # partition instead of one scan per 512-key batch.
                    EMIT_KEYS = INDEX_CP_TILE // THREADS
                    first_key = part * INDEX_CP_TILE + tid * EMIT_KEYS
                    if const_expr(cached):
                        # LDS stores monotonic integer keys, transposed here into lane/key order.
                        bits8 = fx.Vector.from_elements(
                            [
                                lds_ld(xs, INDEX_RADIX_BINS + tid * EMIT_KEYS + e).bitcast(fx.Int32)
                                for e in range(EMIT_KEYS)
                            ],
                            fx.Int32,
                        )
                    else:
                        bits8 = fx.full(EMIT_KEYS, 0, fx.Int32)
                        if first_key + EMIT_KEYS <= size:
                            words8 = poll(
                                [
                                    (mb("index_scores"), s * INDEX_CAPACITY + first_key + e, 2)
                                    for e in range(0, EMIT_KEYS, 2)
                                ]
                            )
                            bits8 = fx.Vector.from_elements([words8[e // 2][e % 2] for e in range(EMIT_KEYS)], fx.Int32)
                        else:
                            tail8 = []
                            for e in range_constexpr(EMIT_KEYS):
                                bits = fx.Int32(0)
                                if first_key + e < size:
                                    bits = get(mb("index_scores"), s * INDEX_CAPACITY + first_key + e)
                                tail8.append(bits)
                            bits8 = fx.Vector.from_elements(tail8, fx.Int32)
                    gt8, eq8 = [], []
                    n_gt, n_eq = fx.Int32(0), fx.Int32(0)
                    for e in range_constexpr(EMIT_KEYS):
                        bits = bits8[e]
                        if const_expr(cached):
                            key = fx.Uint32(bits)
                        else:
                            key = fx.Uint32((bits >= 0).select(bits ^ fx.Int32(-(2**31)), ~bits))
                        valid = first_key + e < size
                        is_gt = (valid & (key > threshold)).select(fx.Int32(1), fx.Int32(0))
                        is_eq = (valid & (key == threshold)).select(fx.Int32(1), fx.Int32(0))
                        gt8.append(is_gt)
                        eq8.append(is_eq)
                        n_gt, n_eq = n_gt + is_gt, n_eq + is_eq
                    if const_expr(indexer_quant and cached):
                        # All cached score reads finish before cp_scan's barriers.
                        # Reuse that retired LDS range for a dense selected-ID list.
                        packed_off, packed_total = cp_scan(n_gt | (n_eq << 16))
                        part_gt = packed_total & 65535
                        part_eq = fx.Int32(fx.Uint32(packed_total) >> 16)
                        take_eq = fx.min(part_eq, fx.max(fx.Int32(0), need_eq - before_eq))
                        local_gt = packed_off & 65535
                        local_eq = fx.Int32(fx.Uint32(packed_off) >> 16)
                        for e in range_constexpr(EMIT_KEYS):
                            is_gt, is_eq = gt8[e], eq8[e]
                            selected = (is_gt != 0) | ((is_eq != 0) & (local_eq < take_eq))
                            compact_slot = (is_gt != 0).select(local_gt, part_gt + local_eq)
                            if selected:
                                lds_st(
                                    xs,
                                    INDEX_RADIX_BINS + compact_slot,
                                    (begin + first_key + e).bitcast(fx.Float32),
                                )
                            local_gt = local_gt + is_gt
                            local_eq = local_eq + is_eq
                        gpu.barrier()
                        for batch in range_constexpr((topk + THREADS - 1) // THREADS):
                            compact_slot = batch * THREADS + tid
                            if compact_slot < part_gt + take_eq:
                                slot = (compact_slot < part_gt).select(
                                    before_gt + compact_slot, out_gt + before_eq + compact_slot - part_gt
                                )
                                put(
                                    cp_buffer("index_cp_indices", cp_peer(s % CP_RANKS)),
                                    s * topk + slot,
                                    lds_ld(xs, INDEX_RADIX_BINS + compact_slot).bitcast(fx.Int32),
                                    CM_SYS,
                                )
                    else:
                        packed_off, _ = cp_scan(n_gt | (n_eq << 16))
                        before_gt = before_gt + (packed_off & 65535)
                        before_eq = before_eq + fx.Int32(fx.Uint32(packed_off) >> 16)
                        for e in range_constexpr(EMIT_KEYS):
                            is_gt, is_eq = gt8[e], eq8[e]
                            selected = (is_gt != 0) | ((is_eq != 0) & (before_eq < need_eq))
                            slot = (is_gt != 0).select(before_gt, out_gt + before_eq)
                            if selected:
                                # Each query has one result owner; peers gather its
                                # contiguous list instead of receiving scattered writes.
                                put(
                                    cp_buffer("index_cp_indices", cp_peer(s % CP_RANKS)),
                                    s * topk + slot,
                                    begin + first_key + e,
                                    CM_SYS,
                                )
                            before_gt = before_gt + is_gt
                            before_eq = before_eq + is_eq
                    gpu.barrier()

            if const_expr(RESIDENT_INDEX_SELECT):
                # All CTAs finish their assigned score production before taking these roles.
                # Dedicated row coordinators never retain a histogram producer's work.
                if bid < S:
                    s = fx.Int32(bid)
                    for digit in range_constexpr(INDEX_RADIX_DIGITS):
                        resident_prefix(digit, s)
                    resident_offsets(s)
                elif bid < S + S * CP_PARTS:
                    tt = fx.Int32(bid - S)
                    s, part = tt // CP_PARTS, tt % CP_PARTS
                    if use_long_index(s):
                        _, size = index_shard(s)
                        cached_keys = score_tile(s, part, size)
                        for batch in range_constexpr(INDEX_CP_TILE // THREADS):
                            lds_st(
                                xs,
                                INDEX_RADIX_BINS + batch * THREADS + tid,
                                cached_keys[batch].bitcast(fx.Float32),
                            )
                        # Emit later reads keys produced by other lanes in this CTA.
                        gpu.barrier()
                    for digit in range_constexpr(INDEX_RADIX_DIGITS):
                        resident_hist(digit, s, part, tt, cached=True)
                    resident_count(s, part, tt, cached=True)
                    resident_emit(s, part, tt, cached=True)
            else:
                for digit in range_constexpr(INDEX_RADIX_DIGITS):
                    shift = (22 - digit * 11) if indexer_quant else (24 - digit * 8)
                    mask = (0xFFFFFFFF << (shift + (11 if indexer_quant else 8))) & 0xFFFFFFFF
                    for tt in range(start(f"index_hist_{digit}"), S * CP_PARTS, G):
                        tt = fx.Int32(tt)
                        s, part = tt // CP_PARTS, tt % CP_PARTS
                        if use_long_index(s):
                            if const_expr(indexer_quant):
                                get(hist_init, s)
                                # The clear completed before its coherent tag.
                                # Atomics access L2 directly; only compiler
                                # ordering is needed after the tagged wait.
                                llvm.InlineAsmOp(None, [], "s_nop 0", "~{memory}", has_side_effects=True)
                            _, size = index_shard(s)
                            prefix = fx.Uint32(0)
                            if const_expr(digit > 0):
                                prefix, _ = cp_threshold(digit - 1, s)
                            for bin_chunk in range_constexpr(max(1, INDEX_RADIX_BINS // THREADS)):
                                bucket_lane = bin_chunk * THREADS + tid
                                if bucket_lane < INDEX_RADIX_BINS:
                                    lds_st(cp_bins, bucket_lane, fx.Int32(0))
                            gpu.barrier()
                            tile_keys = score_tile(s, part, size)
                            for batch in range_constexpr(INDEX_CP_TILE // THREADS):
                                i = part * INDEX_CP_TILE + batch * THREADS + tid
                                if i < size:
                                    key = tile_keys[batch]
                                    if (key & fx.Uint32(mask)) == prefix:
                                        bucket = fx.Int32((key >> shift) & fx.Uint32(INDEX_RADIX_BINS - 1))
                                        fx.atomic_add(
                                            cp_bins + bucket,
                                            fx.Int32(1),
                                            syncscope="workgroup",
                                        )
                            gpu.barrier()
                            if const_expr(indexer_quant):
                                for bin_chunk in range_constexpr(max(1, INDEX_RADIX_BINS // THREADS)):
                                    bucket_lane = bin_chunk * THREADS + tid
                                    if bucket_lane < INDEX_RADIX_BINS:
                                        count = lds_ld(cp_bins, bucket_lane)
                                        if count != 0:
                                            fx.atomic_add(
                                                hist_pointer(digit, s, bucket_lane),
                                                count,
                                                syncscope="agent",
                                            )
                                # Match gfx950 native top-k: atomics already
                                # commit at L2. Drain them before publishing the
                                # tag, without writing back the entire L2 cache.
                                llvm.InlineAsmOp(None, [], "s_waitcnt vmcnt(0)", "~{memory}", has_side_effects=True)
                                gpu.barrier()
                                if tid == 0:
                                    put(hist_done, digit * S * CP_PARTS + tt, fx.Int32(1))
                            else:
                                for bin_chunk in range_constexpr(max(1, INDEX_RADIX_BINS // THREADS)):
                                    bucket_lane = bin_chunk * THREADS + tid
                                    if bucket_lane < INDEX_RADIX_BINS:
                                        put(
                                            mb("index_cp_hist"),
                                            (digit * S * CP_PARTS + tt) * INDEX_RADIX_BINS + bucket_lane,
                                            lds_ld(cp_bins, bucket_lane),
                                        )
                                gpu.barrier()

                    for s in range(start(f"index_prefix_{digit}"), S, G):
                        s = fx.Int32(s)
                        if use_long_index(s):
                            if const_expr(indexer_quant):
                                for batch in range_constexpr((CP_PARTS + THREADS - 1) // THREADS):
                                    part = batch * THREADS + tid
                                    if part < CP_PARTS:
                                        get(hist_done, (digit * S + s) * CP_PARTS + part)
                                gpu.barrier()
                                # Completion tags follow drained L2 atomics.
                                # Read counters coherently, so no cached stale
                                # counter can survive from an earlier launch.
                                llvm.InlineAsmOp(None, [], "s_nop 0", "~{memory}", has_side_effects=True)
                            for bin_chunk in range_constexpr(max(1, INDEX_RADIX_BINS // THREADS)):
                                bucket_lane = bin_chunk * THREADS + tid
                                if bucket_lane < INDEX_RADIX_BINS:
                                    if const_expr(indexer_quant):
                                        count = fx.Int32(
                                            bo.buffer_load(
                                                _rsrc(mb("index_cp_hist")),
                                                (digit * S + s) * INDEX_RADIX_BINS + bucket_lane,
                                                vec_width=1,
                                                dtype=T.i32,
                                                cache_modifier=CM_DEV,
                                            )
                                        )
                                    else:
                                        count = fx.Int32(0)
                                        # Independent partition counters can be polled together;
                                        # each batch waits only for its last producer.
                                        for part_base in range(0, CP_PARTS - CP_PARTS % 8, 8):
                                            counts = poll(
                                                [
                                                    (
                                                        mb("index_cp_hist"),
                                                        ((digit * S + s) * CP_PARTS + part_base + p) * INDEX_RADIX_BINS
                                                        + bucket_lane,
                                                        1,
                                                    )
                                                    for p in range(8)
                                                ],
                                                batch=8,
                                            )
                                            for values in counts:
                                                count = count + values[0]
                                        for part in range_constexpr(CP_PARTS - CP_PARTS % 8, CP_PARTS):
                                            count = count + get(
                                                mb("index_cp_hist"),
                                                ((digit * S + s) * CP_PARTS + part) * INDEX_RADIX_BINS + bucket_lane,
                                            )
                                    put(
                                        cp_buffer(
                                            "index_cp_hist",
                                            cp_peer(s % CP_RANKS) if const_expr(indexer_quant) else owner,
                                        ),
                                        ((cp_rank * INDEX_RADIX_DIGITS + digit) * S + s) * INDEX_RADIX_BINS
                                        + bucket_lane,
                                        count,
                                        CM_SYS,
                                    )
                            gpu.barrier()
                            if cp_rank == (s % CP_RANKS if const_expr(indexer_quant) else fx.Int32(0)):
                                for bin_chunk in range_constexpr(max(1, INDEX_RADIX_BINS // THREADS)):
                                    bucket_lane = bin_chunk * THREADS + tid
                                    if bucket_lane < INDEX_RADIX_BINS:
                                        count = fx.Int32(0)
                                        if const_expr(indexer_quant):
                                            counts = poll(
                                                [
                                                    (
                                                        cp_buffer("index_cp_hist", sym),
                                                        ((src * INDEX_RADIX_DIGITS + digit) * S + s) * INDEX_RADIX_BINS
                                                        + bucket_lane,
                                                        1,
                                                    )
                                                    for src in range(CP_RANKS)
                                                ],
                                                batch=CP_RANKS,
                                                scope="system",
                                            )
                                            for values in counts:
                                                count = count + values[0]
                                        else:
                                            for src in range_constexpr(CP_RANKS):
                                                count = count + cp_get(
                                                    "index_cp_hist",
                                                    ((src * INDEX_RADIX_DIGITS + digit) * S + s) * INDEX_RADIX_BINS
                                                    + bucket_lane,
                                                )
                                        lds_st(cp_bins, bucket_lane, count)
                                gpu.barrier()
                                prefix = fx.Uint32(0)
                                remain = fx.min(fx.Int32(topk), row_position(s) + 1)
                                if const_expr(digit > 0):
                                    prefix, remain = cp_threshold(digit - 1, s)
                                prefix, remain = cp_select_digit(shift, prefix, remain)
                                if tid == 0:
                                    for dst in range_constexpr(CP_RANKS):
                                        address = cp_buffer(
                                            "index_cp_threshold",
                                            cp_peer(fx.Int32(dst)),
                                        )
                                        put(
                                            address,
                                            (digit * S + s) * 2,
                                            fx.Int32(prefix),
                                            CM_SYS,
                                        )
                                        put(
                                            address,
                                            (digit * S + s) * 2 + 1,
                                            remain,
                                            CM_SYS,
                                        )
                            gpu.barrier()

                # Count strict winners and threshold ties separately. Prefixes
                # in rank/partition/token order preserve the original selector's
                # output: ascending token IDs above threshold, then earliest ties.
                for tt in range(start("index_count"), S * CP_PARTS, G):
                    tt = fx.Int32(tt)
                    s, part = tt // CP_PARTS, tt % CP_PARTS
                    if use_long_index(s):
                        _, size = index_shard(s)
                        threshold, _ = cp_threshold(INDEX_RADIX_DIGITS - 1, s)
                        n_gt, n_eq = fx.Int32(0), fx.Int32(0)
                        tile_keys = score_tile(s, part, size)
                        for batch in range_constexpr(INDEX_CP_TILE // THREADS):
                            i = part * INDEX_CP_TILE + batch * THREADS + tid
                            if i < size:
                                key = tile_keys[batch]
                                n_gt = n_gt + (key > threshold).select(fx.Int32(1), fx.Int32(0))
                                n_eq = n_eq + (key == threshold).select(fx.Int32(1), fx.Int32(0))
                        # Each partition has at most 9216 keys. Pack two
                        # independent 16-bit counters into one scan without
                        # carry from the strict-winner count into the tie count.
                        _, packed_counts = cp_scan(n_gt | (n_eq << 16))
                        total_gt = packed_counts & 65535
                        total_eq = fx.Int32(fx.Uint32(packed_counts) >> 16)
                        if tid == 0:
                            put(mb("index_cp_counts"), tt * 2, total_gt)
                            put(mb("index_cp_counts"), tt * 2 + 1, total_eq)
                        gpu.barrier()

                for s in range(start("index_offsets"), S, G):
                    s = fx.Int32(s)
                    if use_long_index(s):
                        total_gt, total_eq = fx.Int32(0), fx.Int32(0)
                        for batch in range_constexpr((CP_PARTS + THREADS - 1) // THREADS):
                            part = batch * THREADS + tid
                            n_gt, n_eq = fx.Int32(0), fx.Int32(0)
                            if part < CP_PARTS:
                                if const_expr(indexer_quant):
                                    n_gt, n_eq = poll([(mb("index_cp_counts"), (s * CP_PARTS + part) * 2, 2)])[0]
                                else:
                                    n_gt = get(mb("index_cp_counts"), (s * CP_PARTS + part) * 2)
                                    n_eq = get(
                                        mb("index_cp_counts"),
                                        (s * CP_PARTS + part) * 2 + 1,
                                    )
                            off_gt, batch_gt = cp_scan(n_gt)
                            off_eq, batch_eq = cp_scan(n_eq)
                            if part < CP_PARTS:
                                put(
                                    mb("index_cp_offsets"),
                                    (s * CP_PARTS + part) * 2,
                                    total_gt + off_gt,
                                )
                                put(
                                    mb("index_cp_offsets"),
                                    (s * CP_PARTS + part) * 2 + 1,
                                    total_eq + off_eq,
                                )
                            total_gt = total_gt + batch_gt
                            total_eq = total_eq + batch_eq
                        if tid == 0:
                            put(
                                cp_buffer(
                                    "index_cp_counts",
                                    cp_peer(s % CP_RANKS) if const_expr(indexer_quant) else owner,
                                ),
                                (cp_rank * S + s) * 2,
                                total_gt,
                                CM_SYS,
                            )
                            put(
                                cp_buffer(
                                    "index_cp_counts",
                                    cp_peer(s % CP_RANKS) if const_expr(indexer_quant) else owner,
                                ),
                                (cp_rank * S + s) * 2 + 1,
                                total_eq,
                                CM_SYS,
                            )
                        gpu.barrier()
                        if cp_rank == (s % CP_RANKS if const_expr(indexer_quant) else fx.Int32(0)):
                            if const_expr(indexer_quant):
                                if tid == 0:
                                    counts = poll(
                                        [
                                            (cp_buffer("index_cp_counts", sym), (src * S + s) * 2, 2)
                                            for src in range(CP_RANKS)
                                        ],
                                        batch=CP_RANKS,
                                        scope="system",
                                    )
                                    before_gt, before_eq = fx.Int32(0), fx.Int32(0)
                                    for src in range_constexpr(CP_RANKS):
                                        address = cp_buffer("index_cp_offsets", cp_peer(fx.Int32(src)))
                                        put(address, s * 2, before_gt, CM_SYS)
                                        put(address, s * 2 + 1, before_eq, CM_SYS)
                                        before_gt = before_gt + counts[src][0]
                                        before_eq = before_eq + counts[src][1]
                            else:
                                before_gt, before_eq = fx.Int32(0), fx.Int32(0)
                                for src in range_constexpr(CP_RANKS):
                                    n_gt = cp_get("index_cp_counts", (src * S + s) * 2)
                                    n_eq = cp_get("index_cp_counts", (src * S + s) * 2 + 1)
                                    if tid == 0:
                                        address = cp_buffer("index_cp_offsets", cp_peer(fx.Int32(src)))
                                        put(address, s * 2, before_gt, CM_SYS)
                                        put(address, s * 2 + 1, before_eq, CM_SYS)
                                    before_gt = before_gt + n_gt
                                    before_eq = before_eq + n_eq
                        gpu.barrier()

                for tt in range(start("index_emit"), S * CP_PARTS, G):
                    tt = fx.Int32(tt)
                    s, part = tt // CP_PARTS, tt % CP_PARTS
                    if use_long_index(s):
                        begin, size = index_shard(s)
                        threshold, need_eq = cp_threshold(INDEX_RADIX_DIGITS - 1, s)
                        if const_expr(indexer_quant):
                            peer_offsets = poll([(cp_buffer("index_cp_offsets", sym), s * 2, 2)], scope="system")[0]
                            local_offsets = poll([(mb("index_cp_offsets"), tt * 2, 2)])[0]
                            before_gt = peer_offsets[0] + local_offsets[0]
                            before_eq = peer_offsets[1] + local_offsets[1]
                        else:
                            before_gt = cp_get("index_cp_offsets", s * 2) + get(mb("index_cp_offsets"), tt * 2)
                            before_eq = cp_get("index_cp_offsets", s * 2 + 1) + get(mb("index_cp_offsets"), tt * 2 + 1)
                        out_gt = fx.min(fx.Int32(topk), row_position(s) + 1) - need_eq
                        # Each lane owns a fixed contiguous key group. Scan its
                        # total counts once, then emit in lane/key order.
                        # This preserves ascending IDs with one CTA scan per
                        # partition instead of one scan per 512-key batch.
                        EMIT_KEYS = INDEX_CP_TILE // THREADS
                        first_key = part * INDEX_CP_TILE + tid * EMIT_KEYS
                        bits8 = fx.full(EMIT_KEYS, 0, fx.Int32)
                        if first_key + EMIT_KEYS <= size:
                            words8 = poll(
                                [
                                    (mb("index_scores"), s * INDEX_CAPACITY + first_key + e, 2)
                                    for e in range(0, EMIT_KEYS, 2)
                                ]
                            )
                            bits8 = fx.Vector.from_elements([words8[e // 2][e % 2] for e in range(EMIT_KEYS)], fx.Int32)
                        else:
                            tail8 = []
                            for e in range_constexpr(EMIT_KEYS):
                                bits = fx.Int32(0)
                                if first_key + e < size:
                                    bits = get(mb("index_scores"), s * INDEX_CAPACITY + first_key + e)
                                tail8.append(bits)
                            bits8 = fx.Vector.from_elements(tail8, fx.Int32)
                        gt8, eq8 = [], []
                        n_gt, n_eq = fx.Int32(0), fx.Int32(0)
                        for e in range_constexpr(EMIT_KEYS):
                            bits = bits8[e]
                            key = fx.Uint32((bits >= 0).select(bits ^ fx.Int32(-(2**31)), ~bits))
                            valid = first_key + e < size
                            is_gt = (valid & (key > threshold)).select(fx.Int32(1), fx.Int32(0))
                            is_eq = (valid & (key == threshold)).select(fx.Int32(1), fx.Int32(0))
                            gt8.append(is_gt)
                            eq8.append(is_eq)
                            n_gt, n_eq = n_gt + is_gt, n_eq + is_eq
                        packed_off, _ = cp_scan(n_gt | (n_eq << 16))
                        before_gt = before_gt + (packed_off & 65535)
                        before_eq = before_eq + fx.Int32(fx.Uint32(packed_off) >> 16)
                        for e in range_constexpr(EMIT_KEYS):
                            is_gt, is_eq = gt8[e], eq8[e]
                            selected = (is_gt != 0) | ((is_eq != 0) & (before_eq < need_eq))
                            slot = (is_gt != 0).select(before_gt, out_gt + before_eq)
                            if selected:
                                # Each query has one result owner; peers gather its
                                # contiguous list instead of receiving scattered writes.
                                put(
                                    cp_buffer("index_cp_indices", cp_peer(s % CP_RANKS)),
                                    s * topk + slot,
                                    begin + first_key + e,
                                    CM_SYS,
                                )
                            before_gt = before_gt + is_gt
                            before_eq = before_eq + is_eq
                        gpu.barrier()

            for s in range(start("index_select"), S, G):
                s = fx.Int32(s)
                if use_long_index(s):
                    bound = row_position(s) + 1
                    if const_expr(indexer_quant and topk % (2 * THREADS) == 0):
                        if bound >= topk:
                            # Two adjacent tagged indices fit one coherent 16-byte load.
                            # Issue independent pairs together and keep both epoch checks.
                            index_pairs = poll(
                                [
                                    (
                                        cp_buffer("index_cp_indices", cp_peer(s % CP_RANKS)),
                                        s * topk + pair * (2 * THREADS) + 2 * tid,
                                        2,
                                    )
                                    for pair in range(topk // (2 * THREADS))
                                ],
                                batch=topk // (2 * THREADS),
                                scope="system",
                            )
                            for pair in range_constexpr(topk // (2 * THREADS)):
                                bo.buffer_store(
                                    fx.Vector.from_elements(index_pairs[pair], fx.Int32),
                                    _rsrc(mb("indices")),
                                    s * topk + pair * (2 * THREADS) + 2 * tid,
                                    cache_modifier=CM_DEV,
                                )
                        else:
                            for batch in range_constexpr((topk + THREADS - 1) // THREADS):
                                j = batch * THREADS + tid
                                value = fx.Int32(0)
                                if j < fx.min(fx.Int32(topk), bound):
                                    value = poll(
                                        [
                                            (
                                                cp_buffer("index_cp_indices", cp_peer(s % CP_RANKS)),
                                                s * topk + j,
                                                1,
                                            )
                                        ],
                                        scope="system",
                                    )[0][0]
                                bo.buffer_store(
                                    value,
                                    _rsrc(mb("indices")),
                                    s * topk + j,
                                    cache_modifier=CM_DEV,
                                )
                    else:
                        for batch in range_constexpr((topk + THREADS - 1) // THREADS):
                            j = batch * THREADS + tid
                            value = fx.Int32(0)
                            if j < fx.min(fx.Int32(topk), bound):
                                value = poll(
                                    [
                                        (
                                            cp_buffer("index_cp_indices", cp_peer(s % CP_RANKS)),
                                            s * topk + j,
                                            1,
                                        )
                                    ],
                                    scope="system",
                                )[0][0]
                            bo.buffer_store(
                                value,
                                _rsrc(mb("indices")),
                                s * topk + j,
                                cache_modifier=CM_DEV,
                            )
                    fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope="agent")
                    gpu.barrier()
                    if tid == 0:
                        put(mb("indices_ready"), s, fx.Int32(1))

        if const_expr(not TILED_INDEXER or indexer_cp):
            for s in range(start("index_select"), S, G):
                s = fx.Int32(s)
                if use_short_index(s):
                    bound = row_position(s) + 1

                    def select_digit(shift, prefix, remain):
                        digit = 255 - fx.min(tid, 255)
                        count = (tid < 256).select(lds_ld(keys, digit), fx.Int32(0))
                        inclusive = fx.coop.warp_inclusive_scan(count, fx.ReductionOp.ADD, width=64)
                        wave_total = read_lane_i32(inclusive, 63)
                        if (wave < 4) & (lane == 63):
                            lds_st(keys, 256 + wave, wave_total)
                        gpu.barrier()
                        before_wave = fx.Int32(0)
                        for w in range_constexpr(4):
                            before_wave = before_wave + (wave > w).select(lds_ld(keys, 256 + w), fx.Int32(0))
                        above = before_wave + inclusive - count
                        hit = (tid < 256) & (above < remain) & (above + count >= remain)
                        gpu.barrier()
                        if hit:
                            lds_st(
                                keys,
                                256,
                                fx.Int32(prefix | (fx.Uint32(digit) << shift)),
                            )
                            lds_st(keys, 257, remain - above)
                        gpu.barrier()
                        result = fx.Uint32(lds_ld(keys, 256)), lds_ld(keys, 257)
                        # The following scan reuses these words for wave totals.
                        # All waves must read the final threshold before any
                        # wave overwrites it, or compaction can exceed topk.
                        gpu.barrier()
                        return result

                    # Transform scores to monotonic integer keys and build the high
                    # byte histogram in the same pass, avoiding one extra 4-K LDS
                    # scan before the remaining radix digits.
                    if tid < 256:
                        lds_st(keys, tid, fx.Int32(0))
                    gpu.barrier()

                    def stage_index_key(i, bits):
                        if i < bound:
                            key = (bits >= 0).select(bits ^ fx.Int32(-(2**31)), ~bits)
                            # The selector CTA no longer needs the large GEMV staging
                            # region, so reuse it for the 4-K radix keys instead of
                            # increasing the monokernel's LDS allocation.
                            lds_st(xs, i, key.bitcast(fx.Float32))
                            digit = fx.Int32((fx.Uint32(key) >> 24) & 255)
                            fx.atomic_add(keys + digit, fx.Int32(1), syncscope="workgroup")

                    if const_expr(indexer_cp):
                        # Issue several independent peer-mailbox reads together.
                        # Tail lanes poll a valid key but never enter the histogram.
                        score_batches = (LOCAL_INDEX_CAPACITY + THREADS - 1) // THREADS
                        for batch in range_constexpr(0, score_batches, 4):
                            ids = [tid + b * THREADS for b in range(batch, min(batch + 4, score_batches))]
                            values = poll(
                                [
                                    (
                                        cp_buffer("index_cp_scores", sym),
                                        s * LOCAL_INDEX_CAPACITY + fx.min(i, bound - 1),
                                        1,
                                    )
                                    for i in ids
                                ],
                                scope="system",
                            )
                            for i, value in zip(ids, values):
                                stage_index_key(i, value[0])
                    else:
                        for batch in range_constexpr((LOCAL_INDEX_CAPACITY + THREADS - 1) // THREADS):
                            i = tid + batch * THREADS
                            if i < bound:
                                stage_index_key(
                                    i,
                                    getf(
                                        mb("index_scores"),
                                        s * LOCAL_INDEX_CAPACITY + i,
                                    ).bitcast(fx.Int32),
                                )
                    gpu.barrier()
                    prefix, remain = select_digit(24, fx.Uint32(0), fx.min(fx.Int32(topk), bound))
                    prefix_mask = 255 << 24
                    for shift in (16, 8, 0):
                        if tid < 256:
                            lds_st(keys, tid, fx.Int32(0))
                        gpu.barrier()
                        for batch in range_constexpr((LOCAL_INDEX_CAPACITY + THREADS - 1) // THREADS):
                            i = tid + batch * THREADS
                            if i < bound:
                                key = fx.Uint32(lds_ld(xs, i).bitcast(fx.Int32))
                                if (key & fx.Uint32(prefix_mask)) == prefix:
                                    digit = fx.Int32((key >> shift) & fx.Uint32(255))
                                    fx.atomic_add(
                                        keys + digit,
                                        fx.Int32(1),
                                        syncscope="workgroup",
                                    )
                        gpu.barrier()
                        prefix, remain = select_digit(shift, prefix, remain)
                        prefix_mask |= 255 << shift

                    threshold = prefix
                    r_index_out = _rsrc(mb("indices"))
                    items = (LOCAL_INDEX_CAPACITY + THREADS - 1) // THREADS
                    item_indices = [tid * items + j for j in range_constexpr(items)]
                    item_keys = [fx.Uint32(lds_ld(xs, fx.min(i, bound - 1)).bitcast(fx.Int32)) for i in item_indices]
                    gt = [(i < bound) & (key > threshold) for i, key in zip(item_indices, item_keys)]
                    eq = [(i < bound) & (key == threshold) for i, key in zip(item_indices, item_keys)]

                    def scan_flags(flags):
                        """Thread-major exclusive offsets for eight flags without extra LDS.

                        The radix histogram already reserves keys[256:264] for wave
                        totals, so the scan can reuse those words and keep S=8 at the
                        96-KiB static-LDS boundary.
                        """
                        local = fx.Int32(0)
                        local_offsets = []
                        for flag in flags:
                            local_offsets.append(local)
                            local = local + flag.select(fx.Int32(1), fx.Int32(0))
                        inclusive = fx.coop.warp_inclusive_scan(local, fx.ReductionOp.ADD, width=64)
                        wave_total = read_lane_i32(inclusive, 63)
                        if lane == 63:
                            lds_st(keys, 256 + wave, wave_total)
                        gpu.barrier()
                        before_wave = fx.Int32(0)
                        total = fx.Int32(0)
                        for w in range_constexpr(WAVES):
                            wave_count = lds_ld(keys, 256 + w)
                            before_wave = before_wave + (wave > w).select(wave_count, fx.Int32(0))
                            total = total + wave_count
                        thread_base = before_wave + inclusive - local
                        return [thread_base + off for off in local_offsets], total

                    gt_offsets, out_gt = scan_flags(gt)
                    gpu.barrier()
                    eq_offsets, _ = scan_flags(eq)
                    gpu.barrier()
                    for j in range_constexpr(items):
                        if gt[j]:
                            bo.buffer_store(
                                fx.Int32(item_indices[j]),
                                r_index_out,
                                s * topk + gt_offsets[j],
                                cache_modifier=CM_DEV,
                            )
                        if eq[j] & (eq_offsets[j] < topk):
                            lds_st(
                                red,
                                eq_offsets[j],
                                item_indices[j].bitcast(fx.Float32),
                            )
                    gpu.barrier()
                    need_eq = fx.min(fx.Int32(topk), bound) - out_gt
                    for batch in range_constexpr((topk + THREADS - 1) // THREADS):
                        j = tid + batch * THREADS
                        if j >= bound:
                            bo.buffer_store(
                                fx.Int32(0),
                                r_index_out,
                                s * topk + j,
                                cache_modifier=CM_DEV,
                            )
                        if j < need_eq:
                            i = lds_ld(red, j).bitcast(fx.Int32)
                            bo.buffer_store(
                                i,
                                r_index_out,
                                s * topk + out_gt + j,
                                cache_modifier=CM_DEV,
                            )
                    # Every lane contributed compact indices.  Make all of those
                    # device-memory stores visible before lane 0 publishes the one
                    # readiness tag consumed by the attention CTAs.
                    fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope="agent")
                    gpu.barrier()
                    if tid == 0:
                        put(mb("indices_ready"), s, fx.Int32(1))
