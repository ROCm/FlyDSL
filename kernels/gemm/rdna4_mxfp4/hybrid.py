# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Asymmetric, register-pipelined GEMM for large M on gfx1201.

Native MMA copies fetch A directly into register fragments ahead of use.
Workgroup copies fetch compressed B, decode it once into LDS, and share it
across the M waves. Only B occupies the ping-pong LDS arena. All thread and
fragment ownership comes from FlyDSL copy/MMA layouts.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Vector as Vec

from .common import (
    expand_nibbles,
    global_tensor,
    identity_tensor,
    magnitude_words,
    swizzled_layout,
    tensor_load,
    tiled_wmma,
)


@lru_cache(maxsize=256)
def build_hybrid(
    m,
    n,
    k,
    *,
    tile_m=256,
    tile_n=128,
    tile_k=64,
    waves_m=4,
    waves_n=2,
    stages=2,
    operand_prefetch=2,
    cache_group_m=4,
    fragment_lds=True,
    pad=8,
    b_swizzle=None,
    barrier_mode="lds",
    schedule="barrier",
    operand_schedule="none",
    epilogue_prefetch=False,
    fold_mode="e4m3",
    table_mode="alu",
    normal_only=False,
    separated_nibbles=False,
    output_dtype="bf16",
    weight_bits=4,
    fast_epilogue=True,
    device_id=0,
    aligned_m=False,
):
    assert fold_mode in ("e4m3", "e5m2") and weight_bits in (4, 8)
    assert tile_m % (waves_m * 16) == 0 and tile_n % (waves_n * 16) == 0
    assert tile_k % 32 == 0 and k % tile_k == 0, "hybrid requires complete K tiles"
    assert stages in (1, 2) and 0 <= operand_prefetch < tile_k // 16
    assert schedule in ("none", "barrier", "groups")
    assert barrier_mode in ("gpu", "lds") and operand_schedule in ("none", "barrier")
    assert isinstance(cache_group_m, int) and cache_group_m > 0
    out_type = fx.BFloat16 if output_dtype == "bf16" else fx.Float16
    b_dtype = fx.Float8E5M2 if fold_mode == "e5m2" else fx.Float8E4M3FN
    rm, rn = tile_m // (waves_m * 16), tile_n // (waves_n * 16)
    threads = waves_m * waves_n * 32
    npad, groups = (n + 15) // 16 * 16, k // 32
    nt, nsteps, nblocks = npad // 16, tile_k // 16, k // tile_k
    dynamic_m = m == 0
    static_m = m
    grid_m, grid_n = (m + tile_m - 1) // tile_m, (n + tile_n - 1) // tile_n
    static_grid_m = grid_m
    if not dynamic_m:
        cache_group_m = min(cache_group_m, grid_m)
    slots = tile_n // 16 * nsteps * 32
    wslots = max(s for s in (1, 2, 4) if slots % (threads * s) == 0)
    if weight_bits == 8:
        wslots = min(wslots, 2)
    word_width = weight_bits // 4
    w_thrs_k = min(threads, nsteps * 32 // wslots)
    w_thrs_n = threads // w_thrs_k
    wcount = slots // (threads * wslots)
    table_bias = 15 if fold_mode == "e5m2" else 8
    if b_swizzle is not None:
        assert len(b_swizzle) == 3 and b_swizzle[1] >= 3
        assert tile_n * tile_k % (1 << (b_swizzle[0] + b_swizzle[1])) == 0
    tile_bytes = (
        tile_n * tile_k
        if b_swizzle is not None
        else (tile_n // 16) * nsteps * 34 * 8 if fragment_lds else tile_n * (tile_k + pad)
    )
    arena = stages * tile_bytes
    assert arena <= 65536

    @fx.struct
    class Shared:
        data: fx.Array[fx.Uint8, arena, 4]

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def mxfp4_hybrid(
        A: fx.Tensor,
        W: fx.Tensor,
        S: fx.Tensor,
        XScale: fx.Tensor,
        Y: fx.Tensor,
        Ref: fx.Tensor,
        LUT: fx.Tensor,
    ):
        if const_expr(dynamic_m):
            m = fx.Int32(fx.get_scalar(Y.shape[0]))
            grid_m = (m + tile_m - 1) // tile_m
        else:
            m = static_m
            grid_m = static_grid_m
        tid = fx.Int32(fx.thread_idx.x)
        pid = fx.Int32(fx.block_idx.x)
        if const_expr(cache_group_m == 1):
            pm, pn = pid // grid_n, pid % grid_n
        else:
            group_size = cache_group_m * grid_n
            first_m = pid // group_size * cache_group_m
            group_m = fx.Int32(cache_group_m)
            if const_expr(dynamic_m or static_grid_m % cache_group_m):
                remaining_m = fx.Int32(grid_m) - first_m
                group_m = (remaining_m < group_m).select(remaining_m, group_m)
            pm = first_m + pid % group_size % group_m
            pn = pid % group_size // group_m
        mma = tiled_wmma(waves_m, waves_n, rm, rn, b_dtype)
        thr_mma = mma.thr_slice(tid)
        gA = global_tensor(A, fx.Float8E4M3FN, fx.make_layout((m, k), (k, 1)), m * k)
        gC = global_tensor(Y, out_type, fx.make_layout((m, n), (n, 1)), m * n * 2)
        gX = global_tensor(XScale, fx.Float32, fx.make_layout((m, n), (1, 0)), m * 4)
        gW = global_tensor(
            W,
            fx.Int32,
            fx.make_layout((nt, k // 16 * 32 * word_width), (k // 16 * 32 * word_width, 1)),
            npad * k * weight_bits // 8,
        )
        gS = global_tensor(
            S,
            fx.Uint8,
            fx.make_layout((nt, (16, 2, 2, groups)), (groups * 16, (1, 0, 0, 16))),
            npad * groups,
        )
        gRef = global_tensor(Ref, fx.Int32, fx.make_layout((nt, (16, k // 8)), (16, (1, 0))), npad * 4)
        gRefFlat = global_tensor(Ref, fx.Int32, fx.make_layout(npad, 1), npad * 4)
        gLUT = global_tensor(
            LUT,
            fx.Uint32,
            fx.make_layout((29 if fold_mode == "e5m2" else 15) * 2, 1),
            (29 if fold_mode == "e5m2" else 15) * 8,
        )
        tA = fx.flat_divide(gA, fx.make_tile(tile_m, tile_k))[None, None, pm, None]
        tC = fx.flat_divide(gC, fx.make_tile(tile_m, tile_n))[None, None, pm, pn]
        tX = fx.flat_divide(gX, fx.make_tile(tile_m, tile_n))[None, None, pm, pn]
        tW = fx.flat_divide(gW, fx.make_tile(tile_n // 16, nsteps * 32 * word_width))[None, None, pn, None]
        tS = fx.flat_divide(gS, fx.make_tile(tile_n // 16, nsteps * 32))[None, None, pn, None]
        tRef = fx.flat_divide(gRef, fx.make_tile(tile_n // 16, nsteps * 32))[None, None, pn, 0]
        copy_w = fx.make_copy_atom(fx.rocdl.BufferCopy(32 * wslots * word_width), fx.Int32)
        copy_s = fx.make_copy_atom(fx.rocdl.BufferCopy(8 * wslots), fx.Uint8)
        copy_ref = fx.make_copy_atom(fx.rocdl.BufferCopy(32 * wslots), fx.Int32)
        thread_layout = fx.make_layout((w_thrs_n, w_thrs_k), (w_thrs_k, 1))
        word_plan = fx.make_tiled_copy_tv(
            copy_w,
            thread_layout,
            fx.make_layout((1, wslots * word_width), (wslots * word_width, 1)),
        ).get_slice(tid)
        scale_plan = fx.make_tiled_copy_tv(copy_s, thread_layout, fx.make_layout((1, wslots), (wslots, 1))).get_slice(
            tid
        )
        ref_plan = fx.make_tiled_copy_tv(copy_ref, thread_layout, fx.make_layout((1, wslots), (wslots, 1))).get_slice(
            tid
        )
        pW, pS = word_plan.partition_S(tW), scale_plan.partition_S(tS)
        pRef = ref_plan.partition_S(tRef)
        cW = scale_plan.partition_S(identity_tensor((tile_n // 16, nsteps * 32)))
        load_W = fx.make_fragment_like(pW[None, None, None, 0])
        load_S = fx.make_fragment_like(pS[None, None, None, 0])
        load_Ref = fx.make_fragment_like(pRef)
        if const_expr(weight_bits == 4):
            fx.copy(copy_ref, pRef, load_Ref)
        shared = fx.SharedAllocator().allocate(Shared).peek()
        bp = fx.recast_iter(b_dtype, shared.data.ptr)
        if const_expr(b_swizzle is not None):
            b_layout = swizzled_layout(fx.make_layout((tile_n, tile_k), (tile_k, 1)), b_swizzle)
        elif const_expr(fragment_lds):
            b_layout = fx.make_layout(
                ((16, tile_n // 16), (8, 2, nsteps)),
                ((8, 34 * nsteps * 8), (1, 17 * 8, 34 * 8)),
            )
        else:
            b_layout = fx.make_layout((tile_n, tile_k), (tile_k + pad, 1))
        sB = [fx.make_view(bp + stage * tile_bytes, b_layout) for stage in range_constexpr(stages)]
        a_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Float8E4M3FN)
        b_atom = fx.make_copy_atom(fx.UniversalCopy64b(), b_dtype)
        a_plan = fx.make_tiled_copy_A(a_atom, mma).get_slice(tid)
        b_plan = fx.make_tiled_copy_B(b_atom, mma).get_slice(tid)
        pA = a_plan.partition_S(tA)
        pB = [b_plan.partition_S(stage) for stage in sB]
        frag_A = thr_mma.make_fragment_A(tA[None, None, 0])
        frag_B = thr_mma.make_fragment_B(sB[0])
        frag_C = thr_mma.make_fragment_C(tC)
        copy_A, copy_B = a_plan.retile(frag_A), b_plan.retile(frag_B)
        bcoords = thr_mma.partition_B(identity_tensor((tile_n, 16)))
        refs = []
        for j in range_constexpr(rn):
            nc, _ = bcoords[0, j, 0].unpack()
            refs.append(tensor_load(gRefFlat, (pn * tile_n + fx.Int32(nc),), fx.Int32))
        x_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        x_plan = fx.make_tiled_copy_C(x_atom, mma).get_slice(tid)
        frag_X = fx.make_fragment_like(frag_C)
        if const_expr(epilogue_prefetch):
            fx.copy(x_atom, x_plan.partition_S(tX), x_plan.retile(frag_X))

        def load_tile(block):
            block = fx.Int32(block)
            fx.copy(copy_w, pW[None, None, None, block], load_W)
            if const_expr(weight_bits == 4):
                fx.copy(copy_s, pS[None, None, None, block], load_S)
            return [load_W.load(), load_S.load()]

        def write_tile(data, stage):
            load_W.store(data[0])
            if const_expr(weight_bits == 4):
                load_S.store(data[1])
            for q in range_constexpr(fx.size(cW.shape).unpack()):
                nf, kw = cW[q].unpack()
                nf, kw = fx.Int32(nf), fx.Int32(kw)
                nc = nf * 16 + kw % 16
                kc = kw // 32 * 16 + kw % 32 // 16 * 8
                if const_expr(weight_bits == 8):
                    converted = Vec.from_elements([load_W[q * 2], load_W[q * 2 + 1]], fx.Int32)
                else:
                    exponent = fx.Int32(fx.Uint32(fx.Uint8(load_S[q])))
                    shift = exponent - fx.Int32(load_Ref[q])
                    if const_expr(table_mode == "alu"):
                        t0, t1 = magnitude_words(shift, fold_mode, normal_only)
                    else:
                        table = tensor_load(gLUT, ((shift + table_bias) * 2,), fx.Uint32, 2)
                        t0, t1 = fx.Uint32(table[0]), fx.Uint32(table[1])
                    converted = expand_nibbles(load_W[q], t0, t1, separated_nibbles)
                offset = fx.Int32(fx.crd2idx((nc, kc), b_layout).unpack())
                target = fx.make_view(fx.get_iter(sB[stage]) + offset, fx.make_layout(8, 1))
                decoded = fx.make_rmem_tensor(8, b_dtype)
                decoded.store(converted.bitcast(b_dtype))
                fx.copy(b_atom, decoded, target)

        def load_operands(stage, block, step):
            fx.copy(a_atom, pA[None, None, step, block], copy_A[None, None, step])
            fx.copy(b_atom, pB[stage][None, None, step], copy_B[None, None, step])

        def compute_stage(stage, block):
            block = fx.Int32(block)
            for early in range_constexpr(operand_prefetch):
                load_operands(stage, block, early)
            for step in range_constexpr(nsteps):
                if const_expr(step + operand_prefetch < nsteps):
                    load_operands(stage, block, step + operand_prefetch)
                if const_expr(operand_schedule == "barrier"):
                    fx.rocdl.sched_barrier(0)
                fx.gemm(
                    mma,
                    frag_C,
                    frag_A[None, None, step],
                    frag_B[None, None, step],
                    frag_C,
                    traversal_order=fx.GemmTraversalOrder.MNK,
                )
                if const_expr(operand_schedule == "barrier"):
                    fx.rocdl.sched_barrier(0)

        def synchronize():
            if const_expr(barrier_mode == "gpu"):
                gpu.barrier()
            else:
                fx.rocdl.sched_barrier(0)
                fx.rocdl.s_wait_dscnt(0)
                fx.rocdl.s_barrier_signal(-1)
                fx.rocdl.s_barrier_wait(-1)
                fx.rocdl.sched_barrier(0)

        def after_prefetch():
            if const_expr(schedule == "barrier"):
                fx.rocdl.sched_barrier(0)

        def after_compute():
            if const_expr(schedule == "barrier"):
                fx.rocdl.sched_barrier(0)
            elif const_expr(schedule == "groups"):
                fx.rocdl.sched_vmem((2 if weight_bits == 4 else 1) * wcount)
                fx.rocdl.sched_mfma(nsteps * rm * rn)
                fx.rocdl.sched_barrier(0)

        frag_C.fill(0)
        if const_expr(stages == 1):
            for block, carried in range(0, nblocks, 1, init=[frag_C.load()]):
                frag_C.store(carried[0])
                write_tile(load_tile(block), 0)
                synchronize()
                compute_stage(0, block)
                synchronize()
                result = yield [frag_C.load()]
            frag_C.store(result)
        else:
            write_tile(load_tile(fx.Int32(0)), 0)
            synchronize()
            if const_expr((nblocks - 1) // 2 > 0):
                for pair, carried in range(0, (nblocks - 1) // 2, 1, init=[frag_C.load()]):
                    frag_C.store(carried[0])
                    block = fx.Int32(pair) * 2 + 1
                    data1 = load_tile(block)
                    after_prefetch()
                    compute_stage(0, block - 1)
                    after_compute()
                    write_tile(data1, 1)
                    synchronize()
                    data2 = load_tile(block + 1)
                    after_prefetch()
                    compute_stage(1, block)
                    after_compute()
                    write_tile(data2, 0)
                    synchronize()
                    result = yield [frag_C.load()]
                frag_C.store(result)
            if const_expr((nblocks - 1) % 2 == 1):
                data = load_tile(fx.Int32(nblocks - 1))
                after_prefetch()
                compute_stage(0, nblocks - 2)
                after_compute()
                write_tile(data, 1)
                synchronize()
            compute_stage((nblocks - 1) % 2, nblocks - 1)
        scales = fx.make_fragment_like(frag_C)
        for j in range_constexpr(rn):
            for i in range_constexpr(rm):
                scales[None, i, j].fill((refs[j] << 23).bitcast(fx.Float32))
        frag_C.store(Vec(frag_C.load()) * Vec(scales.load()))
        if const_expr(not epilogue_prefetch):
            fx.copy(x_atom, x_plan.partition_S(tX), x_plan.retile(frag_X))
        frag_out = fx.make_fragment_like(frag_C, out_type)
        frag_out.store((Vec(frag_C.load()) * Vec(frag_X.load())).to(out_type))
        out_atom = fx.make_copy_atom(fx.rocdl.BufferCopy(out_type.width), out_type)
        out_plan = fx.make_tiled_copy_C(out_atom, mma).get_slice(tid)
        pC = out_plan.partition_S(tC)
        if const_expr(fast_epilogue and (aligned_m or not dynamic_m and static_m % tile_m == 0) and n % tile_n == 0):
            fx.copy(out_atom, out_plan.retile(frag_out), pC)
        else:
            coordC = fx.flat_divide(identity_tensor((m, n)), fx.make_tile(tile_m, tile_n))[None, None, pm, pn]
            cC = out_plan.partition_S(coordC)
            pred = fx.make_fragment_like(pC, fx.Boolean)
            for q in range_constexpr(fx.size(pred.shape).unpack()):
                pred[q] = fx.elem_less(cC[q], (m, n))
            fx.copy(out_atom, out_plan.retile(frag_out), pC, pred=pred)

    @flyc.jit
    def launch(
        A: fx.Tensor,
        W: fx.Tensor,
        S: fx.Tensor,
        XScale: fx.Tensor,
        Y: fx.Tensor,
        Ref: fx.Tensor,
        LUT: fx.Tensor,
        stream: fx.Stream,
    ):
        if const_expr(dynamic_m):
            rows = fx.Int32(fx.get_scalar(Y.shape[0]))
            launch_grid_m = (rows + tile_m - 1) // tile_m
        else:
            launch_grid_m = static_grid_m
        mxfp4_hybrid(A, W, S, XScale, Y, Ref, LUT).launch(
            grid=(launch_grid_m * grid_n + 0 * device_id, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    return launch
