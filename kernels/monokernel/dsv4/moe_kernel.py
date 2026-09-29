# SPDX-License-Identifier: Apache-2.0
"""DSV4 router + A8W4 experts + TP reduction in one resident GPU launch.

The GLM/K3 resident-grid and tagged-payload protocol is retained. Each CTA
advances a bounded, independent token through the whole MoE before visiting
the next token. Service CTAs are offset from the router producers. No global
barrier, sort kernel, quantization launch or host-side epoch update is used.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import Int32, Int64, Stream, T
from kernels.common import buffer_ops as bo
from kernels.monokernel.layout import CM_DEV, CM_SYS
from kernels.monokernel.ops import exp, rcp, rsrc, spin_pause, uniform, xshfl

BLOCKS = 256
THREADS = 512


def scratch_layout(config, samples, tp, shared_fp8=False):
    h, inter, slots = config.hidden, config.intermediate // tp, config.top_k + 1
    sizes = {
        "logits": samples * config.experts,
        "ids": samples * slots,
        "probs": samples * slots,
        "xq": samples * h // 4,
        "xs": samples * h // 32,
        "mid": samples * slots * inter // 4,
        "ms": samples * slots * inter // 32,
    }
    if shared_fp8:
        sizes.update(shared_xq=samples * h // 4, shared_xs=samples * h // 32, shared_mid=samples * inter)
    offsets, offset = {}, 0
    for name, size in sizes.items():
        offsets[name] = offset
        offset += (size * 8 + 255) // 256 * 256
    offsets["_bytes"] = offset
    return offsets


@functools.cache
def _build_serial_dsv4_moe(
    config, samples, tp=1, hash_routing=False, stage=None, shared_fp8=False, hash_vocab=0, aiter_experts=False
):
    config.validate_moe(samples, tp)
    H, I, E, K = config.hidden, config.intermediate // tp, config.experts, config.top_k  # noqa: E741
    SLOTS = K + 1
    SC = scratch_layout(config, samples, tp, shared_fp8)
    # One token is staged at a time; all eight batch buckets use this LDS size.
    XWORDS = max(H // 2, (H // 4 + H // 32), SLOTS * (I // 4 + I // 32))
    REDWORDS = 8 * 64 * 4
    ROUTER_TASKS = E // 16
    QUANT_BASE = (ROUTER_TASKS + 1) % BLOCKS
    UP_BASE = (QUANT_BASE + (H + 1023) // 1024) % BLOCKS
    DOWN_BASE = (UP_BASE + SLOTS * (I // 32)) % BLOCKS
    MAX_PAIRS = samples * H // 2
    SLOT_BYTES = tp * MAX_PAIRS * 8

    @fx.struct
    class Smem:
        x: fx.Array[fx.Float32, XWORDS, 16]
        reduction: fx.Array[fx.Float32, REDWORDS, 16]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def dsv4_moe(
        hidden: Int64,
        router: Int64,
        bias: Int64,
        hash_ids: Int64,
        hash_table: Int64,
        w_up: Int64,
        s_up: Int64,
        w_down: Int64,
        s_down: Int64,
        shared_up: Int64,
        shared_us: Int64,
        shared_down: Int64,
        shared_ds: Int64,
        output: Int64,
        scratch: Int64,
        epochs: Int64,
        symmetric: Int64,
        peers: Int64,
        rank: Int32,
    ):
        bid, tid = gpu.block_idx.x, gpu.thread_idx.x
        lane, wave = tid % 64, tid // 64
        storage = fx.SharedAllocator().allocate(Smem).peek()
        x, red = storage.x.ptr, storage.reduction.ptr
        tag = uniform(bo.buffer_load(rsrc(epochs), bid, vec_width=1, dtype=T.i32)) + 1
        slot_base = fx.Int64(tag & 1) * fx.Int64(SLOT_BYTES)

        def mb(name):
            return scratch + fx.Int64(SC[name])

        def put(base, index, value, cm=CM_DEV):
            bits = value.bitcast(fx.Int32) if isinstance(value, fx.Float32) else fx.Int32(value)
            bo.buffer_store(fx.Vector.from_elements([bits, tag], fx.Int32), rsrc(base), index * 2, cache_modifier=cm)

        def get(base, index, cm=CM_DEV):
            def read():
                return fx.Vector(bo.buffer_load(rsrc(base), index * 2, vec_width=2, dtype=T.i32, cache_modifier=cm))

            pair = read()
            while pair[1] != tag:
                spin_pause()
                pair = read()
            return pair[0]

        def ld(ptr, index):
            return fx.ptr_load(ptr + index)

        def st(ptr, index, value):
            fx.ptr_store(fx.Float32(value), ptr + index)

        def bf(value):
            return fx.Float32(fx.BFloat16(value))

        def bf_pair(lo, hi):
            return fx.Vector.from_elements([fx.BFloat16(lo), fx.BFloat16(hi)], fx.BFloat16).bitcast(fx.Int32)[0]

        def load_bf(base, index):
            return fx.Float32(fx.BFloat16(bo.buffer_load(rsrc(base), index, vec_width=1, dtype=T.bf16)))

        def reduce_wave(value, offsets, maximum=False):
            for delta in offsets:
                other = xshfl(value, delta)
                value = fx.max(value, other) if const_expr(maximum) else value + other
            return value

        def quant_pair(lo, hi, block=32, fused_epilogue=False):
            amax = fx.max(fmath.absf(lo), fmath.absf(hi))
            amax = reduce_wave(amax, (32, 16, 8, 4, 2, 1) if block == 128 else (8, 4, 2, 1), True)
            if const_expr(fused_epilogue):
                # Native AITER *_fp8 GEMM1 keeps the activation in FP32 and
                # rounds the amax exponent before reserving eight FP8 bits.
                bits = amax.bitcast(fx.Int32)
                exponent = fx.max(((bits + 0x400000) & -0x800000).shrui(fx.Int32(23)) - 8, 0)
            else:
                bits = (amax * (1.0 / 448.0)).bitcast(fx.Int32)
                exponent = fx.min(fx.max((bits + 0x7FFFFF).shrui(fx.Int32(23)), 1), 254)
            # Exact reciprocal power of two, safe for all-zero blocks.
            inv = ((254 - exponent) << 23).bitcast(fx.Float32)
            raw = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, lo * inv, hi * inv, fx.Int32(0), False))
            adjacent = xshfl(raw, 1)
            return (raw & 0xFFFF) | (adjacent << 16), exponent

        atom = fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                16,
                16,
                128,
                fx.Float8E4M3FN,
                fx.Float4E2M1FN,
                opsel_a=0,
                opsel_b=0,
            )
        )

        def weight_fragment(w, scales, row_group, chunk, k_size, up=False):
            weight_group = row_group
            if const_expr(aiter_experts and up):
                # ATOM's gfx950 GU-interleaved layout alternates whole
                # 16-row gate/up groups; MFMA fragments within a group match.
                weight_group = (row_group % (I // 16)) * 2 + row_group // (I // 16)
            raw = fx.Vector(
                bo.buffer_load(
                    rsrc(w),
                    ((weight_group * (k_size // 128) + chunk) * 64 + lane) * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            scale_word = fx.Int32(
                bo.buffer_load(
                    rsrc(scales),
                    (row_group * 16 + lane % 16) * (k_size // 128) + chunk,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            scale = scale_word.shrui((lane // 16) * 8) & 255
            return raw, scale

        def mma(acc, fragment, chunk, k_size, input_base, scale_base):
            # E4M3 operand ABI: two 16-byte halves separated by 64 elements.
            lo = fx.ptr_load(
                x + input_base + chunk * 32 + (lane // 16) * 4, result_type=fx.Vector.make_type(4, fx.Float32)
            ).bitcast(fx.Int32)
            hi = fx.ptr_load(
                x + input_base + chunk * 32 + (lane // 16) * 4 + 16, result_type=fx.Vector.make_type(4, fx.Float32)
            ).bitcast(fx.Int32)
            activation = fx.make_rmem_tensor(8, fx.Int32)
            activation.store(lo.shuffle(hi, list(range(8))))
            weight = fx.make_rmem_tensor(4, fx.Int32)
            weight.store(fragment[0])
            act_scale = ld(x, scale_base + chunk * 4 + lane // 16).bitcast(fx.Int32)
            fx.gemm(atom, acc, activation, weight, acc, scale_a=act_scale, scale_b=fragment[1])

        if const_expr(shared_fp8):
            shared_atom = fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(
                    16,
                    16,
                    128,
                    fx.Float8E4M3FN,
                    fx.Float8E4M3FN,
                    opsel_a=0,
                    opsel_b=0,
                )
            )

        def shared_fragment(w, scales, row_group, chunk, k_size):
            index = ((row_group * (k_size // 128) + chunk) * 64 + lane) * 8
            low = fx.Vector(bo.buffer_load(rsrc(w), index, vec_width=4, dtype=T.i32))
            high = fx.Vector(bo.buffer_load(rsrc(w), index + 4, vec_width=4, dtype=T.i32))
            raw = low.shuffle(high, list(range(8)))
            scale = (
                fx.Int32(
                    bo.buffer_load(
                        rsrc(scales),
                        (row_group // 8) * (k_size // 128) + chunk,
                        vec_width=1,
                        dtype=T.i8,
                    )
                )
                & 255
            )
            return raw, scale

        def shared_mma(acc, fragment, chunk, input_base, scale_base):
            lo = fx.ptr_load(
                x + input_base + chunk * 32 + (lane // 16) * 4, result_type=fx.Vector.make_type(4, fx.Float32)
            ).bitcast(fx.Int32)
            hi = fx.ptr_load(
                x + input_base + chunk * 32 + (lane // 16) * 4 + 16, result_type=fx.Vector.make_type(4, fx.Float32)
            ).bitcast(fx.Int32)
            activation = fx.make_rmem_tensor(8, fx.Int32)
            activation.store(lo.shuffle(hi, list(range(8))))
            weight = fx.make_rmem_tensor(8, fx.Int32)
            weight.store(fragment[0])
            act_scale = ld(x, scale_base + chunk * 4 + lane // 16).bitcast(fx.Int32)
            fx.gemm(shared_atom, acc, activation, weight, acc, scale_a=act_scale, scale_b=fragment[1])

        def stage_quantized(value_name, scale_name, value_base, scale_base, length):
            for r in range_constexpr((length // 4 + THREADS - 1) // THREADS):
                i = tid + r * THREADS
                if i < length // 4:
                    st(x, i, get(mb(value_name), value_base + i).bitcast(fx.Float32))
            for r in range_constexpr((length // 32 + THREADS - 1) // THREADS):
                i = tid + r * THREADS
                if i < length // 32:
                    st(x, length // 4 + i, get(mb(scale_name), scale_base + i).bitcast(fx.Float32))
            gpu.barrier()

        sample = fx.Int32(0)
        while sample < samples:
            if const_expr(stage is None or stage == "router"):
                # Router tasks own 16 logits and split K across eight waves.
                task = bid
                while task < ROUTER_TASKS:
                    for r in range_constexpr((H // 2 + THREADS - 1) // THREADS):
                        p = tid + r * THREADS
                        if p < H // 2:
                            raw = fx.Int32(bo.buffer_load(rsrc(hidden), sample * H // 2 + p, vec_width=1, dtype=T.i32))
                            st(x, p, raw.bitcast(fx.Float32))
                    gpu.barrier()
                    acc = fx.Vector.filled(4, 0.0, fx.Float32)
                    for c in range_constexpr((H // 64 + 7) // 8):
                        chunk = wave + c * 8
                        if chunk < H // 64:
                            for half in range_constexpr(2):
                                a = fx.Vector(
                                    bo.buffer_load(
                                        rsrc(router),
                                        (((task * (H // 64) + chunk) * 2 + half) * 64 + lane) * 4,
                                        vec_width=4,
                                        dtype=T.i32,
                                    )
                                ).bitcast(fx.BFloat16)
                                b = fx.ptr_load(
                                    x + chunk * 32 + half * 16 + (lane // 16) * 4,
                                    result_type=fx.Vector.make_type(4, fx.Float32),
                                ).bitcast(fx.BFloat16)
                                acc = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, acc]))
                    fx.ptr_store(acc, red + tid * 4)
                    gpu.barrier()
                    if tid < 16:
                        v = fx.Float32(0.0)
                        for w in range_constexpr(8):
                            v = v + ld(red, (w * 64 + (tid // 4) * 16) * 4 + tid % 4)
                        put(mb("logits"), sample * E + task * 16 + tid, bf(v))
                    gpu.barrier()
                    task = task + BLOCKS

            if const_expr(stage is None or stage == "route"):
                # One independent selector CTA/token. Bias affects ids, never weights.
                if bid == ROUTER_TASKS % BLOCKS:
                    if wave == 0:
                        scores, corrected = [], []
                        for i in range_constexpr((E + 63) // 64):
                            eid = lane + i * 64
                            logit = get(mb("logits"), sample * E + fx.min(eid, E - 1)).bitcast(fx.Float32)
                            # Hash routing can select very negative logits. log(1+exp)
                            # loses these probabilities when the addition rounds to 1.
                            z = exp(-fmath.absf(logit))
                            # This degree-6 log1p series avoids a device-library
                            # call and has relative truncation error < 5.5e-7 on
                            # [0,1/8]; above 1/8, log(1+z) is well conditioned.
                            small = z * (1.0 + z * (-0.5 + z * (1.0 / 3.0 + z * (-0.25 + z * (0.2 - z / 6.0)))))
                            tail = (z < 0.125).select(small, fmath.log(1.0 + z))
                            score = fmath.sqrt(fx.max(logit, 0.0) + tail)
                            scores.append(score)
                            correction = fx.Float32(
                                bo.buffer_load(rsrc(bias), fx.min(eid, E - 1), vec_width=1, dtype=T.f32)
                            )
                            corrected.append((eid < E).select(score + correction, fx.Float32(float("-inf"))))
                        total = fx.Float32(0.0)
                        selected = []
                        for k in range_constexpr(K):
                            if const_expr(hash_routing):
                                if const_expr(hash_vocab > 0):
                                    token = fx.Int64(bo.buffer_load(rsrc(hash_ids), sample, vec_width=1, dtype=T.i64))
                                    valid_token = (token >= 0) & (token < hash_vocab)
                                    safe_token = valid_token.select(fx.Int32(token), fx.Int32(0))
                                    loaded_id = fx.Int32(
                                        bo.buffer_load(rsrc(hash_table), safe_token * K + k, vec_width=1, dtype=T.i32)
                                    )
                                    valid_id = valid_token & (loaded_id >= 0) & (loaded_id < E)
                                else:
                                    loaded_id = fx.Int32(
                                        bo.buffer_load(rsrc(hash_ids), sample * K + k, vec_width=1, dtype=T.i32)
                                    )
                                    valid_id = (loaded_id >= 0) & (loaded_id < E)
                                # Invalid dynamic indices produce a nonfinite result,
                                # never an out-of-bounds weight/table access.
                                best_id = valid_id.select(loaded_id, fx.Int32(0))
                            else:
                                best, best_id = corrected[0], fx.Int32(lane)
                                for i in range_constexpr(1, (E + 63) // 64):
                                    candidate = fx.Int32(lane + i * 64)
                                    take = (corrected[i] > best) | (
                                        (ArithValue(corrected[i]) == ArithValue(best)) & (candidate < best_id)
                                    )
                                    best, best_id = take.select(corrected[i], best), take.select(candidate, best_id)
                                for delta in (32, 16, 8, 4, 2, 1):
                                    other, other_id = xshfl(best, delta), xshfl(best_id, delta)
                                    take = (other > best) | (
                                        (ArithValue(other) == ArithValue(best)) & (other_id < best_id)
                                    )
                                    best, best_id = take.select(other, best), take.select(other_id, best_id)
                            raw = fx.Float32(0.0)
                            for i in range_constexpr((E + 63) // 64):
                                matched = lane + i * 64 == best_id
                                raw = raw + matched.select(scores[i], fx.Float32(0.0))
                                corrected[i] = matched.select(fx.Float32(float("-inf")), corrected[i])
                            raw = reduce_wave(raw, (32, 16, 8, 4, 2, 1))
                            if const_expr(hash_routing):
                                raw = valid_id.select(raw, fx.Float32(float("nan")))
                            selected.append(raw)
                            total = total + raw
                            if lane == 0:
                                put(mb("ids"), sample * SLOTS + k, best_id)
                        if lane == 0:
                            for k in range_constexpr(K):
                                put(mb("probs"), sample * SLOTS + k, selected[k] * rcp(total) * config.route_scale)
                            put(mb("ids"), sample * SLOTS + K, fx.Int32(E))
                            put(mb("probs"), sample * SLOTS + K, fx.Float32(1.0))

            if const_expr(stage is None or stage == "quant"):
                # Quantization is disjoint from router/selector CTA ownership.
                task = (bid - QUANT_BASE + BLOCKS) % BLOCKS
                while task < (H + 1023) // 1024:
                    pair = task * 512 + tid
                    if pair < H // 2:
                        lo = load_bf(hidden, sample * H + pair * 2)
                        hi = load_bf(hidden, sample * H + pair * 2 + 1)
                        packed, exponent = quant_pair(lo, hi)
                        if tid % 2 == 0:
                            put(mb("xq"), sample * H // 4 + pair // 2, packed)
                        if tid % 16 == 0:
                            put(mb("xs"), sample * H // 32 + pair // 16, exponent)
                        if const_expr(shared_fp8):
                            packed128, exponent128 = quant_pair(lo, hi, 128)
                            if tid % 2 == 0:
                                put(mb("shared_xq"), sample * H // 4 + pair // 2, packed128)
                            if tid % 16 == 0:
                                put(mb("shared_xs"), sample * H // 32 + pair // 16, exponent128)
                    task = task + BLOCKS

            if const_expr(stage is None or stage == "up"):
                # Each producer computes a complete 32-value quantization group.
                task = (bid - UP_BASE + BLOCKS) % BLOCKS
                while task < (K if shared_fp8 else SLOTS) * (I // 32):
                    route_slot, tile = task // (I // 32), task % (I // 32)
                    expert = uniform(get(mb("ids"), sample * SLOTS + route_slot))
                    wu = w_up + fx.Int64(expert) * fx.Int64(I * H)
                    su = s_up + fx.Int64(expert) * fx.Int64(2 * I * (H // 32))
                    row_group = tile * 2 + (wave // 2) % 2 + (wave // 4) * (I // 16)
                    split = wave % 2
                    current = weight_fragment(wu, su, row_group, split, H, up=True)
                    stage_quantized("xq", "xs", sample * H // 4, sample * H // 32, H)
                    acc = fx.make_rmem_tensor(4, fx.Float32)
                    acc.store(fx.Vector.filled(4, 0.0, fx.Float32))
                    for c in range_constexpr(H // 256):
                        chunk = c * 2 + split
                        following = None
                        if const_expr(c + 1 < H // 256):
                            following = weight_fragment(wu, su, row_group, chunk + 2, H, up=True)
                        mma(acc, current, chunk, H, fx.Int32(0), fx.Int32(H // 4))
                        current = following
                    fx.ptr_store(acc.load(), red + tid * 4)
                    gpu.barrier()
                    if tid < 16:
                        vals = []
                        for item in range_constexpr(2):
                            row = tid * 2 + item
                            g, u = fx.Float32(0.0), fx.Float32(0.0)
                            for split_i in range_constexpr(2):
                                w = (row // 16) * 2 + split_i
                                g = g + ld(red, (w * 64 + row % 16) * 4)
                                u = u + ld(red, ((w + 4) * 64 + row % 16) * 4)
                            if const_expr(config.swiglu_limit > 0):
                                g = fx.min(g, config.swiglu_limit)
                                u = fx.min(fx.max(u, -config.swiglu_limit), config.swiglu_limit)
                            vals.append(g * rcp(1.0 + exp(-g)) * u)
                        packed, exponent = quant_pair(vals[0], vals[1], fused_epilogue=True)
                        route = sample * SLOTS + route_slot
                        if tid % 2 == 0:
                            put(mb("mid"), route * I // 4 + tile * 8 + tid // 2, packed)
                        if tid == 0:
                            put(mb("ms"), route * I // 32 + tile, exponent)
                    gpu.barrier()
                    task = task + BLOCKS

                if const_expr(shared_fp8):
                    # Keep native checkpoint FP8 shared weights and block128 activations.
                    task = (bid - UP_BASE - K * (I // 32) + 4 * BLOCKS) % BLOCKS
                    while task < I // 32:
                        row_group = task * 2 + (wave // 2) % 2 + (wave // 4) * (I // 16)
                        split = wave % 2
                        current = shared_fragment(shared_up, shared_us, row_group, split, H)
                        stage_quantized("shared_xq", "shared_xs", sample * H // 4, sample * H // 32, H)
                        acc = fx.make_rmem_tensor(4, fx.Float32)
                        acc.store(fx.Vector.filled(4, 0.0, fx.Float32))
                        for c in range_constexpr(H // 256):
                            chunk = c * 2 + split
                            following = None
                            if const_expr(c + 1 < H // 256):
                                following = shared_fragment(shared_up, shared_us, row_group, chunk + 2, H)
                            shared_mma(acc, current, chunk, fx.Int32(0), fx.Int32(H // 4))
                            current = following
                        fx.ptr_store(acc.load(), red + tid * 4)
                        gpu.barrier()
                        if tid < 32:
                            g, u = fx.Float32(0.0), fx.Float32(0.0)
                            for split_i in range_constexpr(2):
                                w = (tid // 16) * 2 + split_i
                                g = g + ld(red, (w * 64 + tid % 16) * 4)
                                u = u + ld(red, ((w + 4) * 64 + tid % 16) * 4)
                            # ATOM materializes shared gate/up BF16 before activation.
                            g, u = bf(g), bf(u)
                            if const_expr(config.swiglu_limit > 0):
                                g = fx.min(g, config.swiglu_limit)
                                u = fx.min(fx.max(u, -config.swiglu_limit), config.swiglu_limit)
                            value = bf(g * rcp(1.0 + exp(-g)) * u)
                            put(mb("shared_mid"), sample * I + task * 32 + tid, value)
                        gpu.barrier()
                        task = task + BLOCKS
                    task = (bid - UP_BASE - K * (I // 32) - I // 32 + 4 * BLOCKS) % BLOCKS
                    while task < I // 128:
                        if wave == 0:
                            base = sample * I + task * 128 + lane * 2
                            lo = get(mb("shared_mid"), base).bitcast(fx.Float32)
                            hi = get(mb("shared_mid"), base + 1).bitcast(fx.Float32)
                            packed, exponent = quant_pair(lo, hi, 128)
                            route = sample * SLOTS + K
                            if lane % 2 == 0:
                                put(mb("mid"), route * I // 4 + task * 32 + lane // 2, packed)
                            if lane % 16 == 0:
                                put(mb("ms"), route * I // 32 + task * 4 + lane // 16, exponent)
                        task = task + BLOCKS

            if const_expr(stage is None or stage == "down"):
                task = (bid - DOWN_BASE + BLOCKS) % BLOCKS
                while task < H // 16:
                    route_slot = fx.min(wave, SLOTS - 1)
                    expert = uniform(get(mb("ids"), sample * SLOTS + route_slot))
                    probability = get(mb("probs"), sample * SLOTS + route_slot).bitcast(fx.Float32)
                    wd = w_down + fx.Int64(expert) * fx.Int64(H * I // 2)
                    sd = s_down + fx.Int64(expert) * fx.Int64(H * I // 32)
                    acc = fx.make_rmem_tensor(4, fx.Float32)
                    acc.store(fx.Vector.filled(4, 0.0, fx.Float32))
                    # All waves must cross the same LDS barrier, before the
                    # routed/shared instruction branches diverge.
                    stage_quantized("mid", "ms", sample * SLOTS * I // 4, sample * SLOTS * I // 32, SLOTS * I)
                    if const_expr(shared_fp8):
                        if wave >= K:
                            current8 = shared_fragment(shared_down, shared_ds, task, fx.Int32(0), I)
                            for chunk in range_constexpr(I // 128):
                                following8 = None
                                if const_expr(chunk + 1 < I // 128):
                                    following8 = shared_fragment(shared_down, shared_ds, task, fx.Int32(chunk + 1), I)
                                shared_mma(
                                    acc,
                                    current8,
                                    fx.Int32(chunk),
                                    route_slot * (I // 4),
                                    SLOTS * I // 4 + route_slot * (I // 32),
                                )
                                current8 = following8
                        else:
                            current4 = weight_fragment(wd, sd, task, fx.Int32(0), I)
                            for chunk in range_constexpr(I // 128):
                                following4 = None
                                if const_expr(chunk + 1 < I // 128):
                                    following4 = weight_fragment(wd, sd, task, fx.Int32(chunk + 1), I)
                                mma(
                                    acc,
                                    current4,
                                    fx.Int32(chunk),
                                    I,
                                    route_slot * (I // 4),
                                    SLOTS * I // 4 + route_slot * (I // 32),
                                )
                                current4 = following4
                    else:
                        current = weight_fragment(wd, sd, task, fx.Int32(0), I)
                        for chunk in range_constexpr(I // 128):
                            following = None
                            if const_expr(chunk + 1 < I // 128):
                                following = weight_fragment(wd, sd, task, fx.Int32(chunk + 1), I)
                            mma(
                                acc,
                                current,
                                fx.Int32(chunk),
                                I,
                                route_slot * (I // 4),
                                SLOTS * I // 4 + route_slot * (I // 32),
                            )
                            current = following
                    vals = acc.load()
                    weighted = fx.Vector.from_elements(
                        [(wave < SLOTS).select(vals[j] * probability, fx.Float32(0.0)) for j in range(4)], fx.Float32
                    )
                    fx.ptr_store(weighted, red + tid * 4)
                    gpu.barrier()
                    if tid < 8:
                        local = []
                        for item in range_constexpr(2):
                            v = fx.Float32(0.0)
                            for w in range_constexpr(K if shared_fp8 else SLOTS):
                                v = v + ld(red, (w * 64 + tid * 2 + item) * 4)
                            if const_expr(shared_fp8):
                                shared_value = ld(red, (K * 64 + tid * 2 + item) * 4)
                                v = bf(v) + bf(shared_value)
                            local.append(v)
                        st(x, tid, bf_pair(local[0], local[1]).bitcast(fx.Float32))
                    gpu.barrier()
                    if const_expr(tp > 1):
                        if wave < tp:
                            pw = fx.Vector(bo.buffer_load(rsrc(peers), wave * 2, vec_width=2, dtype=T.i32))
                            addr = (fx.Int64(uniform(pw[1])) << 32) | fx.Int64(fx.Uint32(uniform(pw[0])))
                            if lane < 8:
                                pair = sample * H // 2 + task * 8 + lane
                                put(addr + slot_base, rank * MAX_PAIRS + pair, ld(x, lane).bitcast(fx.Int32), CM_SYS)
                        gpu.barrier()
                    if tid < 8:
                        pair = sample * H // 2 + task * 8 + tid
                        if const_expr(tp == 1):
                            packed = ld(x, tid).bitcast(fx.Int32)
                        else:
                            lo, hi = fx.Float32(0.0), fx.Float32(0.0)
                            for source in range_constexpr(tp):
                                packed_peer = get(symmetric + slot_base, source * MAX_PAIRS + pair, CM_SYS)
                                lo = lo + (packed_peer << 16).bitcast(fx.Float32)
                                hi = hi + (packed_peer & fx.Int32(-65536)).bitcast(fx.Float32)
                            packed = bf_pair(lo, hi)
                        bo.buffer_store(packed, rsrc(output), pair)
                    gpu.barrier()
                    task = task + BLOCKS
            sample = sample + 1
        # CTA-private epoch counters avoid a grid race and a separate launch.
        if const_expr(stage is None or stage == "down"):
            if tid == 0:
                bo.buffer_store(tag, rsrc(epochs), bid)

    @flyc.jit
    def launch(
        hidden: Int64,
        router: Int64,
        bias: Int64,
        hash_ids: Int64,
        hash_table: Int64,
        w_up: Int64,
        s_up: Int64,
        w_down: Int64,
        s_down: Int64,
        shared_up: Int64,
        shared_us: Int64,
        shared_down: Int64,
        shared_ds: Int64,
        output: Int64,
        scratch: Int64,
        epochs: Int64,
        symmetric: Int64,
        peers: Int64,
        rank: Int32,
        stream: Stream = Stream(None),
    ):
        dsv4_moe(
            hidden,
            router,
            bias,
            hash_ids,
            hash_table,
            w_up,
            s_up,
            w_down,
            s_down,
            shared_up,
            shared_us,
            shared_down,
            shared_ds,
            output,
            scratch,
            epochs,
            symmetric,
            peers,
            rank,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch


@functools.cache
def _build_stage_batch_dsv4_moe(
    config, samples, tp=1, hash_routing=False, stage=None, shared_fp8=False, hash_vocab=0, aiter_experts=False
):
    config.validate_moe(samples, tp)
    H, I, E, K = config.hidden, config.intermediate // tp, config.experts, config.top_k  # noqa: E741
    SLOTS = K + 1
    SC = scratch_layout(config, samples, tp, shared_fp8)
    # One token is staged at a time; all eight batch buckets use this LDS size.
    XWORDS = max(H // 2, (H // 4 + H // 32), SLOTS * (I // 4 + I // 32))
    REDWORDS = 8 * 64 * 4
    ROUTER_TASKS = E // 16
    # Task IDs cover every token within a phase. All scratch still uses
    # the original per-token tagged payloads and one-token LDS staging.
    QUANT_TASKS = (H + 1023) // 1024
    ROUTED_UP_TASKS = (K if shared_fp8 else SLOTS) * (I // 32)
    ROUTE_BASE = (samples * ROUTER_TASKS) % BLOCKS
    QUANT_BASE = (ROUTE_BASE + samples) % BLOCKS
    UP_BASE = (QUANT_BASE + samples * QUANT_TASKS) % BLOCKS
    SHARED_UP_BASE = (UP_BASE + samples * ROUTED_UP_TASKS) % BLOCKS
    SHARED_QUANT_BASE = (SHARED_UP_BASE + samples * (I // 32)) % BLOCKS
    DOWN_BASE = (UP_BASE + samples * SLOTS * (I // 32)) % BLOCKS
    MAX_PAIRS = samples * H // 2
    SLOT_BYTES = tp * MAX_PAIRS * 8

    @fx.struct
    class Smem:
        x: fx.Array[fx.Float32, XWORDS, 16]
        reduction: fx.Array[fx.Float32, REDWORDS, 16]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def dsv4_moe_stage_batch_v1(
        hidden: Int64,
        router: Int64,
        bias: Int64,
        hash_ids: Int64,
        hash_table: Int64,
        w_up: Int64,
        s_up: Int64,
        w_down: Int64,
        s_down: Int64,
        shared_up: Int64,
        shared_us: Int64,
        shared_down: Int64,
        shared_ds: Int64,
        output: Int64,
        scratch: Int64,
        epochs: Int64,
        symmetric: Int64,
        peers: Int64,
        rank: Int32,
    ):
        bid, tid = gpu.block_idx.x, gpu.thread_idx.x
        lane, wave = tid % 64, tid // 64
        storage = fx.SharedAllocator().allocate(Smem).peek()
        x, red = storage.x.ptr, storage.reduction.ptr
        tag = uniform(bo.buffer_load(rsrc(epochs), bid, vec_width=1, dtype=T.i32)) + 1
        slot_base = fx.Int64(tag & 1) * fx.Int64(SLOT_BYTES)

        def mb(name):
            return scratch + fx.Int64(SC[name])

        def put(base, index, value, cm=CM_DEV):
            bits = value.bitcast(fx.Int32) if isinstance(value, fx.Float32) else fx.Int32(value)
            bo.buffer_store(fx.Vector.from_elements([bits, tag], fx.Int32), rsrc(base), index * 2, cache_modifier=cm)

        def get(base, index, cm=CM_DEV):
            def read():
                return fx.Vector(bo.buffer_load(rsrc(base), index * 2, vec_width=2, dtype=T.i32, cache_modifier=cm))

            pair = read()
            while pair[1] != tag:
                spin_pause()
                pair = read()
            return pair[0]

        def ld(ptr, index):
            return fx.ptr_load(ptr + index)

        def st(ptr, index, value):
            fx.ptr_store(fx.Float32(value), ptr + index)

        def bf(value):
            return fx.Float32(fx.BFloat16(value))

        def bf_pair(lo, hi):
            return fx.Vector.from_elements([fx.BFloat16(lo), fx.BFloat16(hi)], fx.BFloat16).bitcast(fx.Int32)[0]

        def load_bf(base, index):
            return fx.Float32(fx.BFloat16(bo.buffer_load(rsrc(base), index, vec_width=1, dtype=T.bf16)))

        def reduce_wave(value, offsets, maximum=False):
            for delta in offsets:
                other = xshfl(value, delta)
                value = fx.max(value, other) if const_expr(maximum) else value + other
            return value

        def quant_pair(lo, hi, block=32, fused_epilogue=False):
            amax = fx.max(fmath.absf(lo), fmath.absf(hi))
            amax = reduce_wave(amax, (32, 16, 8, 4, 2, 1) if block == 128 else (8, 4, 2, 1), True)
            if const_expr(fused_epilogue):
                # Native AITER *_fp8 GEMM1 keeps the activation in FP32 and
                # rounds the amax exponent before reserving eight FP8 bits.
                bits = amax.bitcast(fx.Int32)
                exponent = fx.max(((bits + 0x400000) & -0x800000).shrui(fx.Int32(23)) - 8, 0)
            else:
                bits = (amax * (1.0 / 448.0)).bitcast(fx.Int32)
                exponent = fx.min(fx.max((bits + 0x7FFFFF).shrui(fx.Int32(23)), 1), 254)
            # Exact reciprocal power of two, safe for all-zero blocks.
            inv = ((254 - exponent) << 23).bitcast(fx.Float32)
            raw = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, lo * inv, hi * inv, fx.Int32(0), False))
            adjacent = xshfl(raw, 1)
            return (raw & 0xFFFF) | (adjacent << 16), exponent

        atom = fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                16,
                16,
                128,
                fx.Float8E4M3FN,
                fx.Float4E2M1FN,
                opsel_a=0,
                opsel_b=0,
            )
        )

        def weight_fragment(w, scales, row_group, chunk, k_size, up=False):
            weight_group = row_group
            if const_expr(aiter_experts and up):
                # ATOM's gfx950 GU-interleaved layout alternates whole
                # 16-row gate/up groups; MFMA fragments within a group match.
                weight_group = (row_group % (I // 16)) * 2 + row_group // (I // 16)
            raw = fx.Vector(
                bo.buffer_load(
                    rsrc(w),
                    ((weight_group * (k_size // 128) + chunk) * 64 + lane) * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            scale_word = fx.Int32(
                bo.buffer_load(
                    rsrc(scales),
                    (row_group * 16 + lane % 16) * (k_size // 128) + chunk,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            scale = scale_word.shrui((lane // 16) * 8) & 255
            return raw, scale

        def mma(acc, fragment, chunk, k_size, input_base, scale_base):
            # E4M3 operand ABI: two 16-byte halves separated by 64 elements.
            lo = fx.ptr_load(
                x + input_base + chunk * 32 + (lane // 16) * 4, result_type=fx.Vector.make_type(4, fx.Float32)
            ).bitcast(fx.Int32)
            hi = fx.ptr_load(
                x + input_base + chunk * 32 + (lane // 16) * 4 + 16, result_type=fx.Vector.make_type(4, fx.Float32)
            ).bitcast(fx.Int32)
            activation = fx.make_rmem_tensor(8, fx.Int32)
            activation.store(lo.shuffle(hi, list(range(8))))
            weight = fx.make_rmem_tensor(4, fx.Int32)
            weight.store(fragment[0])
            act_scale = ld(x, scale_base + chunk * 4 + lane // 16).bitcast(fx.Int32)
            fx.gemm(atom, acc, activation, weight, acc, scale_a=act_scale, scale_b=fragment[1])

        if const_expr(shared_fp8):
            shared_atom = fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(
                    16,
                    16,
                    128,
                    fx.Float8E4M3FN,
                    fx.Float8E4M3FN,
                    opsel_a=0,
                    opsel_b=0,
                )
            )

        def shared_fragment(w, scales, row_group, chunk, k_size):
            index = ((row_group * (k_size // 128) + chunk) * 64 + lane) * 8
            low = fx.Vector(bo.buffer_load(rsrc(w), index, vec_width=4, dtype=T.i32))
            high = fx.Vector(bo.buffer_load(rsrc(w), index + 4, vec_width=4, dtype=T.i32))
            raw = low.shuffle(high, list(range(8)))
            scale = (
                fx.Int32(
                    bo.buffer_load(
                        rsrc(scales),
                        (row_group // 8) * (k_size // 128) + chunk,
                        vec_width=1,
                        dtype=T.i8,
                    )
                )
                & 255
            )
            return raw, scale

        def shared_mma(acc, fragment, chunk, input_base, scale_base):
            lo = fx.ptr_load(
                x + input_base + chunk * 32 + (lane // 16) * 4, result_type=fx.Vector.make_type(4, fx.Float32)
            ).bitcast(fx.Int32)
            hi = fx.ptr_load(
                x + input_base + chunk * 32 + (lane // 16) * 4 + 16, result_type=fx.Vector.make_type(4, fx.Float32)
            ).bitcast(fx.Int32)
            activation = fx.make_rmem_tensor(8, fx.Int32)
            activation.store(lo.shuffle(hi, list(range(8))))
            weight = fx.make_rmem_tensor(8, fx.Int32)
            weight.store(fragment[0])
            act_scale = ld(x, scale_base + chunk * 4 + lane // 16).bitcast(fx.Int32)
            fx.gemm(shared_atom, acc, activation, weight, acc, scale_a=act_scale, scale_b=fragment[1])

        def stage_quantized(value_name, scale_name, value_base, scale_base, length):
            for r in range_constexpr((length // 4 + THREADS - 1) // THREADS):
                i = tid + r * THREADS
                if i < length // 4:
                    st(x, i, get(mb(value_name), value_base + i).bitcast(fx.Float32))
            for r in range_constexpr((length // 32 + THREADS - 1) // THREADS):
                i = tid + r * THREADS
                if i < length // 32:
                    st(x, length // 4 + i, get(mb(scale_name), scale_base + i).bitcast(fx.Float32))
            gpu.barrier()

        if const_expr(stage is None or stage == "router"):
            # Router tasks own 16 logits and split K across eight waves.
            task = bid
            while task < samples * ROUTER_TASKS:
                sample = task // ROUTER_TASKS
                router_task = task % ROUTER_TASKS
                for r in range_constexpr((H // 2 + THREADS - 1) // THREADS):
                    p = tid + r * THREADS
                    if p < H // 2:
                        raw = fx.Int32(bo.buffer_load(rsrc(hidden), sample * H // 2 + p, vec_width=1, dtype=T.i32))
                        st(x, p, raw.bitcast(fx.Float32))
                gpu.barrier()
                acc = fx.Vector.filled(4, 0.0, fx.Float32)
                for c in range_constexpr((H // 64 + 7) // 8):
                    chunk = wave + c * 8
                    if chunk < H // 64:
                        for half in range_constexpr(2):
                            a = fx.Vector(
                                bo.buffer_load(
                                    rsrc(router),
                                    (((router_task * (H // 64) + chunk) * 2 + half) * 64 + lane) * 4,
                                    vec_width=4,
                                    dtype=T.i32,
                                )
                            ).bitcast(fx.BFloat16)
                            b = fx.ptr_load(
                                x + chunk * 32 + half * 16 + (lane // 16) * 4,
                                result_type=fx.Vector.make_type(4, fx.Float32),
                            ).bitcast(fx.BFloat16)
                            acc = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, acc]))
                fx.ptr_store(acc, red + tid * 4)
                gpu.barrier()
                if tid < 16:
                    v = fx.Float32(0.0)
                    for w in range_constexpr(8):
                        v = v + ld(red, (w * 64 + (tid // 4) * 16) * 4 + tid % 4)
                    put(mb("logits"), sample * E + router_task * 16 + tid, bf(v))
                gpu.barrier()
                task = task + BLOCKS

        if const_expr(stage is None or stage == "route"):
            # One independent selector CTA/token. Bias affects ids, never weights.
            task = (bid - ROUTE_BASE + BLOCKS) % BLOCKS
            while task < samples:
                sample = task
                if wave == 0:
                    scores, corrected = [], []
                    for i in range_constexpr((E + 63) // 64):
                        eid = lane + i * 64
                        logit = get(mb("logits"), sample * E + fx.min(eid, E - 1)).bitcast(fx.Float32)
                        # Hash routing can select very negative logits. log(1+exp)
                        # loses these probabilities when the addition rounds to 1.
                        z = exp(-fmath.absf(logit))
                        # This degree-6 log1p series avoids a device-library
                        # call and has relative truncation error < 5.5e-7 on
                        # [0,1/8]; above 1/8, log(1+z) is well conditioned.
                        small = z * (1.0 + z * (-0.5 + z * (1.0 / 3.0 + z * (-0.25 + z * (0.2 - z / 6.0)))))
                        tail = (z < 0.125).select(small, fmath.log(1.0 + z))
                        score = fmath.sqrt(fx.max(logit, 0.0) + tail)
                        scores.append(score)
                        correction = fx.Float32(
                            bo.buffer_load(rsrc(bias), fx.min(eid, E - 1), vec_width=1, dtype=T.f32)
                        )
                        corrected.append((eid < E).select(score + correction, fx.Float32(float("-inf"))))
                    total = fx.Float32(0.0)
                    selected = []
                    for k in range_constexpr(K):
                        if const_expr(hash_routing):
                            if const_expr(hash_vocab > 0):
                                token = fx.Int64(bo.buffer_load(rsrc(hash_ids), sample, vec_width=1, dtype=T.i64))
                                valid_token = (token >= 0) & (token < hash_vocab)
                                safe_token = valid_token.select(fx.Int32(token), fx.Int32(0))
                                loaded_id = fx.Int32(
                                    bo.buffer_load(rsrc(hash_table), safe_token * K + k, vec_width=1, dtype=T.i32)
                                )
                                valid_id = valid_token & (loaded_id >= 0) & (loaded_id < E)
                            else:
                                loaded_id = fx.Int32(
                                    bo.buffer_load(rsrc(hash_ids), sample * K + k, vec_width=1, dtype=T.i32)
                                )
                                valid_id = (loaded_id >= 0) & (loaded_id < E)
                            # Invalid dynamic indices produce a nonfinite result,
                            # never an out-of-bounds weight/table access.
                            best_id = valid_id.select(loaded_id, fx.Int32(0))
                        else:
                            best, best_id = corrected[0], fx.Int32(lane)
                            for i in range_constexpr(1, (E + 63) // 64):
                                candidate = fx.Int32(lane + i * 64)
                                take = (corrected[i] > best) | (
                                    (ArithValue(corrected[i]) == ArithValue(best)) & (candidate < best_id)
                                )
                                best, best_id = take.select(corrected[i], best), take.select(candidate, best_id)
                            for delta in (32, 16, 8, 4, 2, 1):
                                other, other_id = xshfl(best, delta), xshfl(best_id, delta)
                                take = (other > best) | ((ArithValue(other) == ArithValue(best)) & (other_id < best_id))
                                best, best_id = take.select(other, best), take.select(other_id, best_id)
                        raw = fx.Float32(0.0)
                        for i in range_constexpr((E + 63) // 64):
                            matched = lane + i * 64 == best_id
                            raw = raw + matched.select(scores[i], fx.Float32(0.0))
                            corrected[i] = matched.select(fx.Float32(float("-inf")), corrected[i])
                        raw = reduce_wave(raw, (32, 16, 8, 4, 2, 1))
                        if const_expr(hash_routing):
                            raw = valid_id.select(raw, fx.Float32(float("nan")))
                        selected.append(raw)
                        total = total + raw
                        if lane == 0:
                            put(mb("ids"), sample * SLOTS + k, best_id)
                    if lane == 0:
                        for k in range_constexpr(K):
                            put(mb("probs"), sample * SLOTS + k, selected[k] * rcp(total) * config.route_scale)
                        put(mb("ids"), sample * SLOTS + K, fx.Int32(E))
                        put(mb("probs"), sample * SLOTS + K, fx.Float32(1.0))
                task = task + BLOCKS

        if const_expr(stage is None or stage == "quant"):
            # Quantization is disjoint from router/selector CTA ownership.
            task = (bid - QUANT_BASE + BLOCKS) % BLOCKS
            while task < samples * QUANT_TASKS:
                sample = task // QUANT_TASKS
                pair = (task % QUANT_TASKS) * 512 + tid
                if pair < H // 2:
                    lo = load_bf(hidden, sample * H + pair * 2)
                    hi = load_bf(hidden, sample * H + pair * 2 + 1)
                    packed, exponent = quant_pair(lo, hi)
                    if tid % 2 == 0:
                        put(mb("xq"), sample * H // 4 + pair // 2, packed)
                    if tid % 16 == 0:
                        put(mb("xs"), sample * H // 32 + pair // 16, exponent)
                    if const_expr(shared_fp8):
                        packed128, exponent128 = quant_pair(lo, hi, 128)
                        if tid % 2 == 0:
                            put(mb("shared_xq"), sample * H // 4 + pair // 2, packed128)
                        if tid % 16 == 0:
                            put(mb("shared_xs"), sample * H // 32 + pair // 16, exponent128)
                task = task + BLOCKS

        if const_expr(stage is None or stage == "up"):
            # Each producer computes a complete 32-value quantization group.
            task = (bid - UP_BASE + BLOCKS) % BLOCKS
            while task < samples * ROUTED_UP_TASKS:
                sample = task // ROUTED_UP_TASKS
                local_task = task % ROUTED_UP_TASKS
                route_slot, tile = local_task // (I // 32), local_task % (I // 32)
                expert = uniform(get(mb("ids"), sample * SLOTS + route_slot))
                wu = w_up + fx.Int64(expert) * fx.Int64(I * H)
                su = s_up + fx.Int64(expert) * fx.Int64(2 * I * (H // 32))
                row_group = tile * 2 + (wave // 2) % 2 + (wave // 4) * (I // 16)
                split = wave % 2
                current = weight_fragment(wu, su, row_group, split, H, up=True)
                stage_quantized("xq", "xs", sample * H // 4, sample * H // 32, H)
                acc = fx.make_rmem_tensor(4, fx.Float32)
                acc.store(fx.Vector.filled(4, 0.0, fx.Float32))
                for c in range_constexpr(H // 256):
                    chunk = c * 2 + split
                    following = None
                    if const_expr(c + 1 < H // 256):
                        following = weight_fragment(wu, su, row_group, chunk + 2, H, up=True)
                    mma(acc, current, chunk, H, fx.Int32(0), fx.Int32(H // 4))
                    current = following
                fx.ptr_store(acc.load(), red + tid * 4)
                gpu.barrier()
                if tid < 16:
                    vals = []
                    for item in range_constexpr(2):
                        row = tid * 2 + item
                        g, u = fx.Float32(0.0), fx.Float32(0.0)
                        for split_i in range_constexpr(2):
                            w = (row // 16) * 2 + split_i
                            g = g + ld(red, (w * 64 + row % 16) * 4)
                            u = u + ld(red, ((w + 4) * 64 + row % 16) * 4)
                        if const_expr(config.swiglu_limit > 0):
                            g = fx.min(g, config.swiglu_limit)
                            u = fx.min(fx.max(u, -config.swiglu_limit), config.swiglu_limit)
                        vals.append(g * rcp(1.0 + exp(-g)) * u)
                    packed, exponent = quant_pair(vals[0], vals[1], fused_epilogue=True)
                    route = sample * SLOTS + route_slot
                    if tid % 2 == 0:
                        put(mb("mid"), route * I // 4 + tile * 8 + tid // 2, packed)
                    if tid == 0:
                        put(mb("ms"), route * I // 32 + tile, exponent)
                gpu.barrier()
                task = task + BLOCKS

            if const_expr(shared_fp8):
                # Keep native checkpoint FP8 shared weights and block128 activations.
                task = (bid - SHARED_UP_BASE + BLOCKS) % BLOCKS
                while task < samples * (I // 32):
                    sample = task // (I // 32)
                    tile = task % (I // 32)
                    row_group = tile * 2 + (wave // 2) % 2 + (wave // 4) * (I // 16)
                    split = wave % 2
                    current = shared_fragment(shared_up, shared_us, row_group, split, H)
                    stage_quantized("shared_xq", "shared_xs", sample * H // 4, sample * H // 32, H)
                    acc = fx.make_rmem_tensor(4, fx.Float32)
                    acc.store(fx.Vector.filled(4, 0.0, fx.Float32))
                    for c in range_constexpr(H // 256):
                        chunk = c * 2 + split
                        following = None
                        if const_expr(c + 1 < H // 256):
                            following = shared_fragment(shared_up, shared_us, row_group, chunk + 2, H)
                        shared_mma(acc, current, chunk, fx.Int32(0), fx.Int32(H // 4))
                        current = following
                    fx.ptr_store(acc.load(), red + tid * 4)
                    gpu.barrier()
                    if tid < 32:
                        g, u = fx.Float32(0.0), fx.Float32(0.0)
                        for split_i in range_constexpr(2):
                            w = (tid // 16) * 2 + split_i
                            g = g + ld(red, (w * 64 + tid % 16) * 4)
                            u = u + ld(red, ((w + 4) * 64 + tid % 16) * 4)
                        # ATOM materializes shared gate/up BF16 before activation.
                        g, u = bf(g), bf(u)
                        if const_expr(config.swiglu_limit > 0):
                            g = fx.min(g, config.swiglu_limit)
                            u = fx.min(fx.max(u, -config.swiglu_limit), config.swiglu_limit)
                        value = bf(g * rcp(1.0 + exp(-g)) * u)
                        put(mb("shared_mid"), sample * I + tile * 32 + tid, value)
                    gpu.barrier()
                    task = task + BLOCKS
                task = (bid - SHARED_QUANT_BASE + BLOCKS) % BLOCKS
                while task < samples * (I // 128):
                    sample = task // (I // 128)
                    tile = task % (I // 128)
                    if wave == 0:
                        base = sample * I + tile * 128 + lane * 2
                        lo = get(mb("shared_mid"), base).bitcast(fx.Float32)
                        hi = get(mb("shared_mid"), base + 1).bitcast(fx.Float32)
                        packed, exponent = quant_pair(lo, hi, 128)
                        route = sample * SLOTS + K
                        if lane % 2 == 0:
                            put(mb("mid"), route * I // 4 + tile * 32 + lane // 2, packed)
                        if lane % 16 == 0:
                            put(mb("ms"), route * I // 32 + tile * 4 + lane // 16, exponent)
                    task = task + BLOCKS

        if const_expr(stage is None or stage == "down"):
            task = (bid - DOWN_BASE + BLOCKS) % BLOCKS
            while task < samples * (H // 16):
                sample = task // (H // 16)
                tile = task % (H // 16)
                route_slot = fx.min(wave, SLOTS - 1)
                expert = uniform(get(mb("ids"), sample * SLOTS + route_slot))
                probability = get(mb("probs"), sample * SLOTS + route_slot).bitcast(fx.Float32)
                wd = w_down + fx.Int64(expert) * fx.Int64(H * I // 2)
                sd = s_down + fx.Int64(expert) * fx.Int64(H * I // 32)
                acc = fx.make_rmem_tensor(4, fx.Float32)
                acc.store(fx.Vector.filled(4, 0.0, fx.Float32))
                # All waves must cross the same LDS barrier, before the
                # routed/shared instruction branches diverge.
                stage_quantized("mid", "ms", sample * SLOTS * I // 4, sample * SLOTS * I // 32, SLOTS * I)
                if const_expr(shared_fp8):
                    if wave >= K:
                        current8 = shared_fragment(shared_down, shared_ds, tile, fx.Int32(0), I)
                        for chunk in range_constexpr(I // 128):
                            following8 = None
                            if const_expr(chunk + 1 < I // 128):
                                following8 = shared_fragment(shared_down, shared_ds, tile, fx.Int32(chunk + 1), I)
                            shared_mma(
                                acc,
                                current8,
                                fx.Int32(chunk),
                                route_slot * (I // 4),
                                SLOTS * I // 4 + route_slot * (I // 32),
                            )
                            current8 = following8
                    else:
                        current4 = weight_fragment(wd, sd, tile, fx.Int32(0), I)
                        for chunk in range_constexpr(I // 128):
                            following4 = None
                            if const_expr(chunk + 1 < I // 128):
                                following4 = weight_fragment(wd, sd, tile, fx.Int32(chunk + 1), I)
                            mma(
                                acc,
                                current4,
                                fx.Int32(chunk),
                                I,
                                route_slot * (I // 4),
                                SLOTS * I // 4 + route_slot * (I // 32),
                            )
                            current4 = following4
                else:
                    current = weight_fragment(wd, sd, tile, fx.Int32(0), I)
                    for chunk in range_constexpr(I // 128):
                        following = None
                        if const_expr(chunk + 1 < I // 128):
                            following = weight_fragment(wd, sd, tile, fx.Int32(chunk + 1), I)
                        mma(
                            acc,
                            current,
                            fx.Int32(chunk),
                            I,
                            route_slot * (I // 4),
                            SLOTS * I // 4 + route_slot * (I // 32),
                        )
                        current = following
                vals = acc.load()
                weighted = fx.Vector.from_elements(
                    [(wave < SLOTS).select(vals[j] * probability, fx.Float32(0.0)) for j in range(4)], fx.Float32
                )
                fx.ptr_store(weighted, red + tid * 4)
                gpu.barrier()
                if tid < 8:
                    local = []
                    for item in range_constexpr(2):
                        v = fx.Float32(0.0)
                        for w in range_constexpr(K if shared_fp8 else SLOTS):
                            v = v + ld(red, (w * 64 + tid * 2 + item) * 4)
                        if const_expr(shared_fp8):
                            shared_value = ld(red, (K * 64 + tid * 2 + item) * 4)
                            v = bf(v) + bf(shared_value)
                        local.append(v)
                    st(x, tid, bf_pair(local[0], local[1]).bitcast(fx.Float32))
                gpu.barrier()
                if const_expr(tp > 1):
                    if wave < tp:
                        pw = fx.Vector(bo.buffer_load(rsrc(peers), wave * 2, vec_width=2, dtype=T.i32))
                        addr = (fx.Int64(uniform(pw[1])) << 32) | fx.Int64(fx.Uint32(uniform(pw[0])))
                        if lane < 8:
                            pair = sample * H // 2 + tile * 8 + lane
                            put(addr + slot_base, rank * MAX_PAIRS + pair, ld(x, lane).bitcast(fx.Int32), CM_SYS)
                    gpu.barrier()
                if tid < 8:
                    pair = sample * H // 2 + tile * 8 + tid
                    if const_expr(tp == 1):
                        packed = ld(x, tid).bitcast(fx.Int32)
                    else:
                        lo, hi = fx.Float32(0.0), fx.Float32(0.0)
                        for source in range_constexpr(tp):
                            packed_peer = get(symmetric + slot_base, source * MAX_PAIRS + pair, CM_SYS)
                            lo = lo + (packed_peer << 16).bitcast(fx.Float32)
                            hi = hi + (packed_peer & fx.Int32(-65536)).bitcast(fx.Float32)
                        packed = bf_pair(lo, hi)
                    bo.buffer_store(packed, rsrc(output), pair)
                gpu.barrier()
                task = task + BLOCKS
        # CTA-private epoch counters avoid a grid race and a separate launch.
        if const_expr(stage is None or stage == "down"):
            if tid == 0:
                bo.buffer_store(tag, rsrc(epochs), bid)

    @flyc.jit
    def launch_stage_batch_v1(
        hidden: Int64,
        router: Int64,
        bias: Int64,
        hash_ids: Int64,
        hash_table: Int64,
        w_up: Int64,
        s_up: Int64,
        w_down: Int64,
        s_down: Int64,
        shared_up: Int64,
        shared_us: Int64,
        shared_down: Int64,
        shared_ds: Int64,
        output: Int64,
        scratch: Int64,
        epochs: Int64,
        symmetric: Int64,
        peers: Int64,
        rank: Int32,
        stream: Stream = Stream(None),
    ):
        dsv4_moe_stage_batch_v1(
            hidden,
            router,
            bias,
            hash_ids,
            hash_table,
            w_up,
            s_up,
            w_down,
            s_down,
            shared_up,
            shared_us,
            shared_down,
            shared_ds,
            output,
            scratch,
            epochs,
            symmetric,
            peers,
            rank,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch_stage_batch_v1


@functools.cache
def build_dsv4_moe(
    config, samples, tp=1, hash_routing=False, stage=None, shared_fp8=False, hash_vocab=0, aiter_experts=False
):
    # Keep seq1 and the five-stage reference on the exact verified arithmetic
    # and original scheduling. Only fused multi-token execution is changed.
    builder = _build_serial_dsv4_moe if samples == 1 or stage is not None else _build_stage_batch_dsv4_moe
    return builder(config, samples, tp, hash_routing, stage, shared_fp8, hash_vocab, aiter_experts)
