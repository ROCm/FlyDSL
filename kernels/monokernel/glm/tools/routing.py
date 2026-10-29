# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared routing primitives, inlined during kernel IR construction."""

from typing import Callable, NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from kernels.monokernel.config import MOE_SLOTS, N_EXPERTS, ROUTE_SCALE, SHARED_EXPERT, TOP_K
from kernels.monokernel.glm.layout import WAVES, sample_wave_batches
from kernels.monokernel.ops import bpermute_i32, wave_umax_dpp, write_lane_i32
from kernels.monokernel.ops import rcp as _rcp
from kernels.monokernel.ops import rsrc as _rsrc


class RoutingTools(NamedTuple):
    load_bias: Callable
    route_top8: Callable
    dn_route: Callable


@flyc.jit
def build_routing(plan, state, memory):
    S = plan.S
    bias = state.bias
    dnw = state.dnw
    keys = state.keys
    lane = state.lane
    wave = state.wave
    ld_f32 = memory.ld_f32
    lds_st = memory.lds_st
    mb = memory.mb
    getf_many = memory.getf_many

    def load_bias():
        """This lane's 4 expert biases (issue before the scores wait)."""
        return [ld_f32(_rsrc(bias), lane + i * 64) for i in range(N_EXPERTS // 64)]

    def route_top8(s, raws=None, bs=None):
        """Top-8 of sample s (call from one whole wave, after the router scores landed).

        Preserve all FP32 bits of (sigmoid + bias). Each round reduces the score
        first, then the expert ID only among exactly equal winners. Packing the
        ID into the score's low byte can change GLM's top-k on real checkpoints.
        Candidate i of this lane is expert lane + 64 i.
        Returns (expert id, route weight = raw score / sum of
        the 8 raw scores * ROUTE_SCALE) of pick ``lane`` in score order, valid in
        lanes < TOP_K."""
        if const_expr(bs is None):
            bs = load_bias()
        if const_expr(raws is None):
            raws = getf_many([(mb("scores"), s * N_EXPERTS + lane + i * 64) for i in range(N_EXPERTS // 64)])
        ks = []
        for i in range_constexpr(N_EXPERTS // 64):
            kb = (raws[i] + bs[i]).bitcast(fx.Int32)
            ok = (kb >= 0).select(kb ^ fx.Int32(-(2**31)), ~kb)
            ks.append(fx.Uint32(ok))
        ids = [lane + i * 64 for i in range(N_EXPERTS // 64)]
        # sort this lane's 4 keys descending; each round then takes the wave max of
        # the lane heads and shifts the winning lane's list (0 is below every key)
        for a, b in ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2)):
            first = (ks[a] > ks[b]) | ((ks[a] == ks[b]) & (ids[a] < ids[b]))
            ka, kb, ia, ib = ks[a], ks[b], ids[a], ids[b]
            ks[a], ks[b] = first.select(ka, kb), first.select(kb, ka)
            ids[a], ids[b] = first.select(ia, ib), first.select(ib, ia)
        ks = [fx.Int32(k) for k in ks] + [fx.Int32(0)]
        ids = ids + [fx.Int32(N_EXPERTS)]
        mv = fx.Int32(0)  # lane k: the expert ID of pick k
        for k in range_constexpr(TOP_K):
            m = wave_umax_dpp(ks[0])
            winner = wave_umax_dpp((ks[0] == m).select(255 - ids[0], fx.Int32(0)))
            expert = 255 - winner
            hit = (ks[0] == m) & (ids[0] == expert)
            ks = [hit.select(ks[i + 1], ks[i]) for i in range(4)] + [ks[4]]
            ids = [hit.select(ids[i + 1], ids[i]) for i in range(4)] + [ids[4]]
            mv = write_lane_i32(expert, k, mv)
        e = mv
        src = (e % 64) * 4
        got = [bpermute_i32(src, r.bitcast(fx.Int32)) for r in raws]
        raw = got[0]
        for i in range_constexpr(1, N_EXPERTS // 64):
            raw = (e // 64 == i).select(got[i], raw)
        raw = (lane < TOP_K).select(raw.bitcast(fx.Float32), fx.Float32(0.0))
        tot = fx.coop.warp_reduce(raw, fx.ReductionOp.ADD, width=8)
        return e, raw * (_rcp(tot) * ROUTE_SCALE)

    def dn_route(bs):
        """Expert-down routing: one whole wave per sample, in wave-sized batches."""
        for sample_batch in range_constexpr(sample_wave_batches(S)):
            route_sample = wave + sample_batch * WAVES
            if route_sample < S:
                e, w = route_top8(route_sample, bs=bs)
                if lane < MOE_SLOTS:  # slot 0: the shared expert, then pick lane (slot lane + 1)
                    q = route_sample * MOE_SLOTS + (lane + 1) % MOE_SLOTS
                    lds_st(
                        keys,
                        q,
                        (lane == TOP_K).select(fx.Int32(SHARED_EXPERT), e),
                    )
                    lds_st(dnw, q, (lane == TOP_K).select(fx.Float32(1.0), w))

    return RoutingTools(
        load_bias=load_bias,
        route_top8=route_top8,
        dn_route=dn_route,
    )
