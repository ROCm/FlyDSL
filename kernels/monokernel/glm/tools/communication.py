# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared communication primitives, inlined during kernel IR construction."""

from typing import Callable, NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.config import HIDDEN, KV_LORA
from kernels.monokernel.glm.layout import DCP_SUMMARY_PAIRS, ROW_TILE, dcp_summary_index, dcp_uv_owner
from kernels.monokernel.layout import CM_SYS, NEG
from kernels.monokernel.ops import exp as _exp
from kernels.monokernel.ops import rcp as _rcp
from kernels.monokernel.ops import uniform as _uniform


class CommunicationTools(NamedTuple):
    ffn_owner_reduce: Callable
    o_owner_reduce: Callable
    peer_reduce: Callable
    dcp_merge_latent: Callable


@flyc.jit
def build_communication(plan, state, memory, reductions):
    D = plan.D
    DN_TILE = plan.DN_TILE
    L = plan.L
    N_SPLIT = plan.N_SPLIT
    N_UV = plan.N_UV
    S = plan.S
    SY = plan.SY
    W = plan.W
    lane = state.lane
    misc = state.misc
    outs = state.outs
    peer_dst = state.peer_dst
    peer_offset = state.peer_offset
    peer_slot = state.peer_slot
    r_peers = state.r_peers
    rank = state.rank
    sym = state.sym
    tid = state.tid
    wave = state.wave
    xs = state.xs
    lds_ld = memory.lds_ld
    lds_st = memory.lds_st
    bf16_pair = memory.bf16_pair
    put = memory.put
    put2 = memory.put2
    put_bf = memory.put_bf
    bf2_f32 = memory.bf2_f32
    poll = memory.poll
    wave_sum = reductions.wave_sum
    wave_max = reductions.wave_max

    def owner_reduce(region, t, residual, out_fn, tile):
        """Gather rounded partials at a tile owner, then broadcast the raw FP32 sum."""
        region_base = fx.Int64(SY[region]) + fx.Int64(peer_slot) * fx.Int64(SY["_part_stride"])
        owner = t % W
        owner_words = fx.Vector(bo.buffer_load(r_peers, owner * 2, vec_width=2, dtype=T.i32))
        owner_addr = (
            (fx.Int64(_uniform(owner_words[1])) << 32) | fx.Int64(fx.Uint32(_uniform(owner_words[0])))
        ) + peer_offset
        if tid < S * tile // 2:
            s = tid // (tile // 2)
            r = (tid % (tile // 2)) * 2
            put_bf(
                owner_addr + region_base,
                (rank * S + s) * HIDDEN + t * tile + r,
                [lds_ld(outs, s * tile + r), lds_ld(outs, s * tile + r + 1)],
                CM_SYS,
            )
        gpu.barrier()
        if rank == owner:
            if tid < S * tile // 2:
                s = tid // (tile // 2)
                r = (tid % (tile // 2)) * 2
                row = t * tile + r
                specs = [(sym + region_base, ((src * S + s) * HIDDEN + row) // 2, 1) for src in range(W)]
                got = poll(specs, "one-as")
                parts = [bf2_f32(v[0]) for v in got]
                t0, t1 = fx.Float32(0.0), fx.Float32(0.0)
                for src in range_constexpr(W):
                    t0 = t0 + parts[src][0]
                    t1 = t1 + parts[src][1]
                lds_st(outs, s * tile + r, t0)
                lds_st(outs, s * tile + r + 1, t1)
            gpu.barrier()
            if wave < W:
                pair_count = S * tile // 2
                for batch in range_constexpr((pair_count + 63) // 64):
                    pair = lane + batch * 64
                    if pair < pair_count:
                        s = pair // (tile // 2)
                        r = (pair % (tile // 2)) * 2
                        result_base = peer_dst + region_base + fx.Int64(W * S * HIDDEN * 4)
                        put2(
                            result_base,
                            s * HIDDEN + t * tile + r,
                            lds_ld(outs, s * tile + r),
                            lds_ld(outs, s * tile + r + 1),
                            CM_SYS,
                        )
        gpu.barrier()
        if tid < S * tile // 2:
            s = tid // (tile // 2)
            r = (tid % (tile // 2)) * 2
            row = t * tile + r
            result_base = sym + region_base + fx.Int64(W * S * HIDDEN * 4)
            got = poll([(result_base, s * HIDDEN + row, 2)], "one-as")
            r0, r1 = residual(s, row)
            out_fn(
                s,
                row,
                r0 + got[0][0].bitcast(fx.Float32),
                r1 + got[0][1].bitcast(fx.Float32),
            )

    def ffn_owner_reduce(t, residual, out_fn, tile=DN_TILE):
        owner_reduce("ffn", t, residual, out_fn, tile)

    def o_owner_reduce(t, residual, out_fn, tile=ROW_TILE):
        owner_reduce("attn", t, residual, out_fn, tile)

    def peer_reduce(region, t, residual, out_fn, tile=ROW_TILE):
        """Push BF16 partials to every peer, then sum them in rank order.

        One wave owns each destination, allowing the peer stores to progress
        concurrently without retaining all peer pointers in every wave."""
        region_base = fx.Int64(SY[region]) + fx.Int64(peer_slot) * fx.Int64(SY["_part_stride"])
        if const_expr(W > 1):
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
                            [
                                lds_ld(outs, si * tile + ri),
                                lds_ld(outs, si * tile + ri + 1),
                            ],
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

    def dcp_merge_latent(s, t):
        if const_expr(D == 1):
            return True
        region_base = fx.Int64(SY["dcp"]) + fx.Int64(peer_slot) * fx.Int64(SY["_dcp_part_stride"])
        summary_base = dcp_summary_index(rank, s, t, 0, S, N_UV)
        if wave < D:
            for batch in range_constexpr((DCP_SUMMARY_PAIRS + 63) // 64):
                item = lane + batch * 64
                if item < DCP_SUMMARY_PAIRS:
                    value = fx.Float32(0.0)
                    if item < KV_LORA // 2:
                        value = lds_ld(xs, item)
                    elif item == KV_LORA // 2:
                        value = lds_ld(misc, N_SPLIT)
                    else:
                        value = lds_ld(misc, N_SPLIT + 1)
                    put(
                        peer_dst + region_base,
                        summary_base + item,
                        value,
                        CM_SYS,
                    )
        gpu.barrier()
        owner = dcp_uv_owner(t, L, rank)
        if owner:
            own = sym + region_base
            if wave == 0:
                src = fx.min(lane, D - 1)
                src_base = dcp_summary_index(src, s, t, 0, S, N_UV)
                values = poll(
                    [
                        (own, src_base + KV_LORA // 2, 1),
                        (own, src_base + KV_LORA // 2 + 1, 1),
                    ],
                    "one-as",
                )
                valid = lane < D
                m = valid.select(values[0][0].bitcast(fx.Float32), fx.Float32(NEG))
                local_sum = valid.select(values[1][0].bitcast(fx.Float32), fx.Float32(0.0))
                m_all = wave_max(m)
                z = local_sum * _exp(m - m_all)
                den = wave_sum(z)
                if valid:
                    lds_st(
                        misc,
                        lane,
                        z * (den > 0.0).select(_rcp(den), fx.Float32(0.0)),
                    )
            gpu.barrier()
            if tid < KV_LORA // 2:
                parts = poll(
                    [(own, dcp_summary_index(src, s, t, tid, S, N_UV), 1) for src in range(D)],
                    "one-as",
                )
                o0, o1 = fx.Float32(0.0), fx.Float32(0.0)
                for src in range_constexpr(D):
                    a0, a1 = bf2_f32(parts[src][0])
                    weight = lds_ld(misc, src)
                    o0, o1 = o0 + a0 * weight, o1 + a1 * weight
                lds_st(xs, tid, bf16_pair(o0, o1))
            gpu.barrier()
        return owner

    return CommunicationTools(
        ffn_owner_reduce=ffn_owner_reduce,
        o_owner_reduce=o_owner_reduce,
        peer_reduce=peer_reduce,
        dcp_merge_latent=dcp_merge_latent,
    )
