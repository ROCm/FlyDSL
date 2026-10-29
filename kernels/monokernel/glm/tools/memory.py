# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared memory primitives, inlined during kernel IR construction."""

from typing import Callable, NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.layout import CM_DEV, CM_SYS, POLL_MAX
from kernels.monokernel.ops import rsrc as _rsrc
from kernels.monokernel.ops import spin_pause
from kernels.monokernel.ops import uniform as _uniform


class MemoryTools(NamedTuple):
    ld_f32: Callable
    ld_bf16: Callable
    ld_rope: Callable
    row_index_bounds: Callable
    row_active: Callable
    row_position: Callable
    row_slot: Callable
    row_writes_cache: Callable
    index_request_start: Callable
    index_page_slot: Callable
    lds_ld: Callable
    lds_st: Callable
    bf16_pair: Callable
    bf16_round: Callable
    index_arg: Callable
    mb: Callable
    cp_local_rank: Callable
    cp_peer: Callable
    cp_buffer: Callable
    put: Callable
    put2: Callable
    put_bf: Callable
    bf2_f32: Callable
    poll: Callable
    hint_wait: Callable
    pre_poll: Callable
    get: Callable
    getf: Callable
    getf_many: Callable
    get2_many: Callable
    get2: Callable
    get_bf2_many: Callable
    pick: Callable
    start: Callable
    n_sel: Callable


@flyc.jit
def build_memory(plan, state):
    CP_RANKS = plan.CP_RANKS
    G = plan.G
    S = plan.S
    SC = plan.SC
    SY = plan.SY
    W = plan.W
    base = plan.base
    index_block_table_stride = plan.index_block_table_stride
    index_page_size = plan.index_page_size
    index_request_width = plan.index_request_width
    indexer_cp = plan.indexer_cp
    rope_dtype = plan.rope_dtype
    use_atom_kv_cache = plan.use_atom_kv_cache
    bid = state.bid
    cur_pos = state.cur_pos
    lane = state.lane
    peer_offset = state.peer_offset
    peer_slot = state.peer_slot
    pos0 = state.pos0
    positions = state.positions
    r_peers = state.r_peers
    rank = state.rank
    row_offset = state.row_offset
    scratch = state.scratch
    slot_mapping = state.slot_mapping
    sparse_kv_indptr = state.sparse_kv_indptr
    sym = state.sym
    tag = state.tag
    index_params = state.index_params
    wave = state.wave

    def ld_f32(r, i):
        return fx.Float32(bo.buffer_load(r, i, vec_width=1, dtype=T.f32))

    def ld_bf16(r, i):
        return fx.Float32(fx.BFloat16(bo.buffer_load(r, i, vec_width=1, dtype=T.bf16)))

    def ld_rope(r, i):
        if const_expr(rope_dtype == "bf16"):
            return ld_bf16(r, i)
        return ld_f32(r, i)

    def row_index_bounds(s):
        begin = fx.Int32(bo.buffer_load(_rsrc(sparse_kv_indptr), s, vec_width=1, dtype=T.i32))
        end = fx.Int32(bo.buffer_load(_rsrc(sparse_kv_indptr), s + 1, vec_width=1, dtype=T.i32))
        return begin, end

    def row_active(s):
        if const_expr(use_atom_kv_cache):
            begin, end = row_index_bounds(s)
            return end > begin
        return True

    def row_position(s):
        if const_expr(use_atom_kv_cache):
            position = fx.Int32(bo.buffer_load(_rsrc(positions), s * 2, vec_width=1, dtype=T.i32))
            present = row_active(s) | (row_slot(s) >= 0)
            return present.select(position, fx.Int32(0))
        return pos0 + s

    def row_slot(s):
        if const_expr(use_atom_kv_cache):
            return fx.Int32(bo.buffer_load(_rsrc(slot_mapping), s * 2, vec_width=1, dtype=T.i32))
        return pos0 + s

    def row_writes_cache(s):
        return row_slot(s) >= 0

    def index_request_start(s):
        return fx.max(
            fx.Int32(0),
            ((row_offset + s) // index_request_width) * index_request_width - row_offset,
        )

    def index_page_slot(s, logical_pos):
        if const_expr(not use_atom_kv_cache):
            return logical_pos
        request = (row_offset + s) // index_request_width
        page = fx.Int32(
            bo.buffer_load(
                _rsrc(cur_pos),
                request * index_block_table_stride + logical_pos // index_page_size,
                vec_width=1,
                dtype=T.i32,
            )
        )
        return page * index_page_size + logical_pos % index_page_size

    def lds_ld(ptr, i):
        return fx.ptr_load(ptr + i)

    def lds_st(ptr, i, v):
        fx.ptr_store(v, ptr + i)

    def bf16_pair(a, b):
        """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
        return fx.Vector.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Float32)[0]

    def bf16_round(a):
        return fx.Float32(a.to(fx.BFloat16))

    def index_arg(i):
        """Load one uniform pointer from the compact indexer parameter table.

        The indices argument separately carries the index cache.
        """
        pv = fx.Vector(bo.buffer_load(_rsrc(index_params), i * 2, vec_width=2, dtype=T.i32))
        return (fx.Int64(_uniform(pv[1])) << 32) | fx.Int64(fx.Uint32(_uniform(pv[0])))

    def mb(name):
        return scratch + fx.Int64(SC[name])

    def cp_local_rank():
        if const_expr(indexer_cp and CP_RANKS < W):
            return rank % CP_RANKS
        return rank

    def cp_peer(peer):
        if const_expr(not indexer_cp):
            return sym
        if const_expr(CP_RANKS < W):
            peer = (rank // CP_RANKS) * CP_RANKS + peer
        words = fx.Vector(bo.buffer_load(r_peers, peer * 2, vec_width=2, dtype=T.i32))
        return ((fx.Int64(_uniform(words[1])) << 32) | fx.Int64(fx.Uint32(_uniform(words[0])))) + peer_offset

    def cp_buffer(name, address):
        return address + fx.Int64(SY[name]) + fx.Int64(peer_slot) * SY[name + "_stride"]

    def put(base_addr, i, v, cm=CM_DEV):
        """Pair i := (v, tag); ``v`` f32 (or int32 bits)."""
        bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
        bo.buffer_store(
            fx.Vector.from_elements([bits, tag], fx.Int32),
            _rsrc(base_addr),
            i * 2,
            cache_modifier=cm,
        )

    def put2(base_addr, i, v0, v1, cm=CM_DEV):
        """Pairs i, i+1 (i even) in one 16-byte store."""
        vec = fx.Vector.from_elements(
            [
                fx.Float32(v0).bitcast(fx.Int32),
                tag,
                fx.Float32(v1).bitcast(fx.Int32),
                tag,
            ],
            fx.Int32,
        )
        bo.buffer_store(vec, _rsrc(base_addr), i * 2, cache_modifier=cm)

    def put_bf(base_addr, i, vs, cm=CM_DEV):
        """Elements i .. i + len(vs) (2 or 4, i aligned) as packed bf16 pairs: pair
        i / 2 + j := (bf16(vs[2j]) | bf16(vs[2j + 1]) << 16, tag), one 8 / 16-byte store.
        """
        words = []
        for j in range_constexpr(len(vs) // 2):
            words += [
                bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32),
                tag,
            ]
        bo.buffer_store(
            fx.Vector.from_elements(words, fx.Int32),
            _rsrc(base_addr),
            i,
            cache_modifier=cm,
        )

    def bf2_f32(w):
        """Packed bf16 pair word -> (f32 low, f32 high)."""
        return (w << 16).bitcast(fx.Float32), (w & fx.Int32(-65536)).bitcast(fx.Float32)

    def poll(specs, scope="agent", batch=POLL_MAX):
        """Batched poll of mailbox pairs: ``specs`` = [(base_addr, pair index, npairs in {1, 2})].

        All pairs are loaded together with plain 8 / 16-byte coherent buffer loads
        (sc1 locally, sc0 sc1 for peer memory); while any tag is not this launch's
        the whole batch is re-loaded, so a batch costs one round trip after its
        last producer lands.  A side-effecting (compiler-opaque) asm statement in
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
                    bo.buffer_load(
                        _rsrc(b),
                        fx.Int32(i) * 2,
                        vec_width=2 * n,
                        dtype=T.i32,
                        cache_modifier=cm,
                    )
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
        while pending(v):
            spin_pause()
            v = load_all()
        outs_, e = [], 0
        for _, _, n in specs:
            outs_.append([v[e + 2 * q] for q in range(n)])
            e += 2 * n
        return outs_

    def hint_wait(n, addr_of):
        """Consumers poll their payload directly (tight per-wave spins); a wave-0
        pre-poll of each producer's last pair only added a hop of latency."""
        gpu.barrier()

    def pre_poll(n, addr_of):
        """Wave 0 spins on one small pair per producer (lane j -> producer j < n <= 64)
        before a large payload poll, so waiting CTAs do not flood memory."""
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

    def pick(values, index):
        value = values[0]
        for i in range_constexpr(1, len(values)):
            value = (index == i).select(values[i], value)
        return value

    def start(name):
        return (bid + (G - base[name])) & (G - 1)

    def n_sel(count=S):
        """This lane's local MFMA B column; inactive columns duplicate the last one."""
        return fx.min(lane % 16, count - 1)

    return MemoryTools(
        ld_f32=ld_f32,
        ld_bf16=ld_bf16,
        ld_rope=ld_rope,
        row_index_bounds=row_index_bounds,
        row_active=row_active,
        row_position=row_position,
        row_slot=row_slot,
        row_writes_cache=row_writes_cache,
        index_request_start=index_request_start,
        index_page_slot=index_page_slot,
        lds_ld=lds_ld,
        lds_st=lds_st,
        bf16_pair=bf16_pair,
        bf16_round=bf16_round,
        index_arg=index_arg,
        mb=mb,
        cp_local_rank=cp_local_rank,
        cp_peer=cp_peer,
        cp_buffer=cp_buffer,
        put=put,
        put2=put2,
        put_bf=put_bf,
        bf2_f32=bf2_f32,
        poll=poll,
        hint_wait=hint_wait,
        pre_poll=pre_poll,
        get=get,
        getf=getf,
        getf_many=getf_many,
        get2_many=get2_many,
        get2=get2,
        get_bf2_many=get_bf2_many,
        pick=pick,
        start=start,
        n_sel=n_sel,
    )
