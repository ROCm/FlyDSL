# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""A4W4 (MXFP4 E2M1 activation x MXFP4 E2M1 weight) 32x4-preshuffled MXscale GEMM for gfx1250."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster, tdm_ops
from flydsl.expr.typing import Constexpr, T, as_ir_value
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import check_smem_capacity
from kernels.common.gfx1250_cluster import compute_mcast_masks

from .gemm_common_gfx1250 import (
    make_lds_copy_ops,
    pipeline_fence,
    pipeline_fence_signal,
    pipeline_fence_wait,
    workgroup_barrier,
)


@flyc.jit
def launch_gemm_a4w4_256x256(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Pointer,
    arg_scale_b: fx.Pointer,
    i32_m: fx.Int32,
    stream: fx.Stream,
    N: fx.Int32,
    K: fx.Int32,
    i32_lda: fx.Int32,
    i32_ldc: fx.Int32,
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    out_is_f16: Constexpr[int],
    num_buffers: Constexpr[int],
    cluster_m: Constexpr[int],
    cluster_n: Constexpr[int],
):
    """Launch the A4W4 kernel.

    N must divide 1024 and ceil(M/256) must be a multiple of 4 (M itself may be ragged);
    K must be divisible by 1024, with K >= 1024.
    """

    assert (tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, cluster_m, cluster_n) == (
        256,
        256,
        256,
        2,
        2,
        4,
        4,
        4,
    ), "only the tuned 256x256x256, 2x2-wave, 4-buffer profile with a 4x4 cluster is supported"
    cluster_sync_revs = 8
    m_run_max, m_run_min = 32, 8
    FENCE_WAIT_POS = 7  # WMMAs of slack between the READY signal and its matching wait
    WMMA_M = 32
    WMMA_N = 16
    WMMA_K = 128
    WAVE = 32
    PACK_TK = tile_k // 2  # A/B row bytes per K-tile (FP4 packed 2/byte)
    K_STEPS = tile_k // WMMA_K
    SC_WORDS = tile_k // 4  # scale i32 words per super-row per K-tile
    SA_SUPERS = tile_m // 32
    SB_SUPERS = tile_n // 32
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    wmma_m_rep = warp_tile_m // WMMA_N  # A fragments (16 rows each)
    wmma_n_rep = warp_tile_n // WMMA_M  # B fragments (32 rows each)
    half_m, half_n = wmma_m_rep // 2, wmma_n_rep // 2
    WMMA_PER_Q = half_m * half_n * K_STEPS  # WMMAs in one 64x64 quadrant
    n_acc = wmma_m_rep * wmma_n_rep
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    LDS_PAD_A = 16
    A_LDS_ROW = PACK_TK + LDS_PAD_A
    B_LDS_ROW = PACK_TK * 16
    STAGE_A = tile_m * A_LDS_ROW
    STAGE_B = (tile_n // 16) * B_LDS_ROW
    STAGE_SA = SA_SUPERS * tile_k
    STAGE_SB = SB_SUPERS * tile_k
    # Operand-major planar LDS: scales and A below, B on the next 64-KiB boundary above them.
    PLANAR_SA_BASE = 0
    PLANAR_A_BASE = PLANAR_SA_BASE + num_buffers * STAGE_SA
    PLANAR_SB_BASE = PLANAR_A_BASE + num_buffers * STAGE_A
    PLANAR_B_BASE = ((PLANAR_SB_BASE + num_buffers * STAGE_SB + 65535) // 65536) * 65536
    PLANAR_END = PLANAR_B_BASE + num_buffers * STAGE_B

    ARENA_B = PLANAR_END
    check_smem_capacity(ARENA_B, str(get_hip_arch()))

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel_gemm_a4w4_256x256(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Pointer,
        arg_scale_b: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldc: fx.Int32,
    ):

        K_TILES = i32_k // tile_k
        k64 = fx.Int64(i32_k)
        i32_lda_b = i32_lda // 2  # A row stride in bytes (FP4 packed)
        lda64 = fx.Int64(i32_lda_b)
        ldc64 = fx.Int64(i32_ldc)
        Kp16 = (k64 // 2) * 16

        tid = fx.Int32(fx.thread_idx.x)
        bid_x, bid_y, bid_z = fx.block_idx
        wave = rocdl.readfirstlane(T.i32, tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        wave_m = wave // n_warp
        wave_n = wave % n_warp
        local_x, local_y = cluster.compute_cluster_position()
        a_mask, b_mask = compute_mcast_masks(local_x, local_y, cluster_m, cluster_n)
        blk_m = (bid_z * fx.Int32(fx.grid_dim.x) + bid_x) * tile_m
        blk_n = bid_y * tile_n
        blk_m64 = fx.Int64(blk_m)
        blk_n64 = fx.Int64(blk_n)
        mn_oob = i32_m - blk_m  # valid M rows (A / C)
        sa_oob = (i32_m + 31) // 32 - blk_m // 32  # valid M-supers (scale-A)

        arena = fx.SharedAllocator(static=False)
        arena.allocate(ARENA_B)
        base_ptr = arena.base_ptr

        def _planar_base(offset, stride, stage):
            ptr = fx.add_offset(base_ptr, offset + stage * stride)
            return fx.index_cast(T.index, fx.ptrtoint(ptr))

        def _view(ptr, shape, stride):
            return fx.Tensor(fx.make_view(ptr, fx.make_layout(shape, stride)))

        def _gv(base, off, shape, stride):
            return _view(fx.add_offset(base, off), shape, stride)

        oc = fx.Float16 if out_is_f16 else fx.BFloat16
        gA_base = fx.recast_iter(fx.Int8, arg_a)
        gB_base = fx.recast_iter(fx.Int8, arg_b)

        a_off0 = blk_m64 * lda64
        b_off0 = (blk_n64 // 16) * Kp16
        sa_off0 = (blk_m64 // 32) * k64
        sb_off0 = (blk_n64 // 32) * k64

        gA = _gv(gA_base, a_off0, (tile_m, PACK_TK), (PACK_TK, 1))
        gB = _gv(gB_base, b_off0, (tile_n // 16, B_LDS_ROW), (B_LDS_ROW, 1))
        gSA = _gv(arg_scale_a, sa_off0, (SA_SUPERS, tile_k), (tile_k, 1))
        gSB = _gv(arg_scale_b, sb_off0, (SB_SUPERS, tile_k), (tile_k, 1))

        def _build_tdm_desc(owner):
            if const_expr(owner == 0):
                tensor, offset, shape, lds_stride = gA, PLANAR_A_BASE, (tile_m, PACK_TK), A_LDS_ROW
                stride, mask, bound, pad, early = i32_lda_b, a_mask, mn_oob, LDS_PAD_A, False
            elif const_expr(owner == 1):
                tensor, offset, shape, lds_stride = gB, PLANAR_B_BASE, (tile_n // 16, B_LDS_ROW), B_LDS_ROW
                stride, mask, bound, pad, early = i32_k * 8, b_mask, None, 0, False
            elif const_expr(owner == 2):
                tensor, offset, shape, lds_stride = gSA, PLANAR_SA_BASE, (SA_SUPERS, tile_k), tile_k
                stride, mask, bound, pad, early = i32_k, a_mask, sa_oob, 0, True
            else:
                tensor, offset, shape, lds_stride = gSB, PLANAR_SB_BASE, (SB_SUPERS, tile_k), tile_k
                stride, mask, bound, pad, early = i32_k, b_mask, None, 0, False
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=tensor,
                lds_memref=_view(fx.add_offset(base_ptr, offset), shape, (lds_stride, 1)),
                global_offset=(0, 0),
                tensor_shape=shape,
                strides=(stride, 1),
                tile_shape=shape,
                elem_bytes=1,
                pad_interval=shape[1] if pad else 0,
                pad_amount=pad,
                num_warps=1,
                workgroup_mask=mask,
                early_timeout=early,
                oob_outer_bound=bound,
            )
            return desc, shape[0] * lds_stride, shape[1]

        def _owned_tdm_desc(owner):
            desc, lds_step, global_step = _build_tdm_desc(owner)
            return Vec(desc.dgroup0), Vec(desc.dgroup1), fx.Int32(lds_step), fx.Int32(global_step)

        dgroup0 = Vec.from_elements([as_ir_value(fx.Int32(0))] * 4, fx.Int32)
        dgroup1 = Vec.from_elements([as_ir_value(fx.Int32(0))] * 8, fx.Int32)
        tdm_lds_step, tdm_global_step = fx.Int32(0), fx.Int32(0)
        if wave == 0:
            dgroup0, dgroup1, tdm_lds_step, tdm_global_step = _owned_tdm_desc(0)
        elif wave == 1:
            dgroup0, dgroup1, tdm_lds_step, tdm_global_step = _owned_tdm_desc(1)
        elif wave == 2:
            dgroup0, dgroup1, tdm_lds_step, tdm_global_step = _owned_tdm_desc(2)
        else:
            dgroup0, dgroup1, tdm_lds_step, tdm_global_step = _owned_tdm_desc(3)

        tdm_desc = tdm_ops.TDMDescriptor2D(as_ir_value(dgroup0), as_ir_value(dgroup1))
        tdm_base_lds, tdm_base_lo, tdm_base_hi = (
            fx.Int32(dgroup0[1]),
            fx.Int32(dgroup0[2]),
            fx.Int32(dgroup0[3]),
        )

        def _prepare_tdm(slot, tile):
            desc = tdm_ops.update_tensor_descriptor_2d_lds_addr(tdm_desc, tdm_base_lds + tdm_lds_step * fx.Int32(slot))
            return tdm_ops.update_tensor_descriptor_2d_addr64(
                desc,
                tdm_base_lo,
                tdm_base_hi,
                fx.Int32(tile) * tdm_global_step,
            )

        wmb = wave_m * warp_tile_m
        wnb = wave_n * warp_tile_n

        # matrix_b_scale selects which half of the shared A-scale word this 16-row block uses.
        wmma_atoms = [
            fx.make_mma_atom(
                fx.rocdl.WMMAScale(
                    WMMA_M,
                    WMMA_N,
                    WMMA_K,
                    fx.Float4E2M1FN,
                    fx.Float4E2M1FN,
                    fx.Float32,
                    opsel_b=sa_sel,
                )
            )
            for sa_sel in range_constexpr(2)
        ]
        c_frags = [fx.make_rmem_tensor(16, fx.Float32) for _ in range_constexpr(n_acc)]
        for cf in c_frags:
            cf.store(fx.constant_vector(0.0, T.vec(16, T.f32)))

        def _rmem(n, v):
            t = fx.make_rmem_tensor(n, fx.Int32)
            t.store(v)
            return t

        def _mma_one(wm, wn, k, act, wt, sa_k, sb_k):
            idx = wm * wmma_n_rep + wn
            fx.gemm(
                wmma_atoms[wm % 2],
                c_frags[idx],
                wt,
                act,
                c_frags[idx],
                scale_a=sb_k[wn * K_STEPS + k],
                scale_b=sa_k[(wm // 2) * K_STEPS + k],
            )

        def _mma_block_range(wm0, wn0, act, wt, sa_k, sb_k, start, count, n_index_fast=False):
            """Issue `count` WMMAs from linear position `start` of the wm x wn x k block."""
            n_m, n_n = len(act), len(wt)
            n_minor = n_n if n_index_fast else n_m
            for linear in range_constexpr(count):
                pos = start + linear
                minor = pos % n_minor
                k = (pos // n_minor) % K_STEPS
                major = pos // (n_minor * K_STEPS)
                i, j = (major, minor) if n_index_fast else (minor, major)
                _mma_one(wm0 + i, wn0 + j, k, act[i][k], wt[j][k], sa_k, sb_k)

        cluster.cluster_barrier()
        # Keep fragment displacements as DS immediates inside the K loop.
        stage_a_addr, stage_b_addr, stage_sa_addr, stage_sb_addr = [], [], [], []
        sa_row, sb_col = wmb + lane, wnb + lane
        a_byte = fx.index_cast(T.index, (wmb + lane16) * A_LDS_ROW + kgrp * 16)
        b_byte = fx.index_cast(T.index, (wnb // 16) * B_LDS_ROW + kgrp * 256 + lane16 * 16)
        sa_byte = fx.index_cast(T.index, ((sa_row // 32) * SC_WORDS + sa_row % 32) * 4)
        sb_byte = fx.index_cast(T.index, ((sb_col // 32) * SC_WORDS + sb_col % 32) * 4)
        for addr_stage in range_constexpr(num_buffers):
            stage_a_addr.append(_planar_base(PLANAR_A_BASE, STAGE_A, addr_stage) + a_byte)
            stage_b_addr.append(_planar_base(PLANAR_B_BASE, STAGE_B, addr_stage) + b_byte)
            stage_sa_addr.append(_planar_base(PLANAR_SA_BASE, STAGE_SA, addr_stage) + sa_byte)
            stage_sb_addr.append(_planar_base(PLANAR_SB_BASE, STAGE_SB, addr_stage) + sb_byte)

        lds_load_b32, _ = make_lds_copy_ops(32)
        lds_load_b128, _ = make_lds_copy_ops(128)

        class _DsOrder:

            LIMIT = 63

            def __init__(self):
                self.issued = 0

            def mark(self, ops):
                self.issued += ops
                return self.issued

            def wait(self, mark):
                rocdl.s_wait_dscnt(min(self.issued - mark, self.LIMIT))

        ds = _DsOrder()

        DS_A_OPS, DS_B_OPS = 2, 4  # ds_load_b128 per A / B fragment

        def _stage_load_a(stage, wm, k):
            """16 rows x 128 FP4: lane holds row wm*16+lane16, K-bytes kgrp*16 + 32*j."""
            row_off = wm * 16 * A_LDS_ROW + k * (WMMA_K // 2)
            v0 = Vec(lds_load_b128(stage_a_addr[stage], row_off))
            v1 = Vec(lds_load_b128(stage_a_addr[stage], row_off + 32))
            return v0.shuffle(v1, list(range(8))), ds.mark(DS_A_OPS)

        def _stage_load_b(stage, wn, k):
            """32 rows x 128 FP4: two stacked 16-row blocks, low 8 words then high 8."""
            col_off = wn * 2 * B_LDS_ROW + k * (WMMA_K // 2) * 16
            v = [
                Vec(lds_load_b128(stage_b_addr[stage], col_off + blk * B_LDS_ROW + half))
                for blk in range_constexpr(2)
                for half in (0, 512)
            ]
            lo = v[0].shuffle(v[1], list(range(8)))
            hi = v[2].shuffle(v[3], list(range(8)))
            return lo.shuffle(hi, list(range(16))), ds.mark(DS_B_OPS)

        def _stage_load_sa(stage, sm, k):
            return lds_load_b32(stage_sa_addr[stage], (sm * SC_WORDS + k * 32) * 4)[0], ds.mark(1)

        def _stage_load_sb(stage, sn, k):
            return lds_load_b32(stage_sb_addr[stage], (sn * SC_WORDS + k * 32) * 4)[0], ds.mark(1)

        # Separate seed banks prevent WMMA source/address register coalescing.
        def _pipe(count, width):
            return [
                [
                    [fx.make_rmem_tensor(width, fx.Int32) for _ in range_constexpr(K_STEPS)]
                    for _ in range_constexpr(count)
                ]
                for _ in range_constexpr(2)
            ]

        pipe_a = _pipe(half_m, 8)
        pipe_b = _pipe(half_n, 16)
        pipe_sa = _pipe(half_m // 2, 1)
        pipe_sb = _pipe(half_n, 1)
        seed_mark = [0, 0]  # DS mark after which each bank's seeds are complete

        def _scales_a(stage, sm0):
            return [
                _stage_load_sa(stage, sm0 + sm, k)[0]
                for sm in range_constexpr(half_m // 2)
                for k in range_constexpr(K_STEPS)
            ]

        def _scales_b(stage, sn0):
            return [
                _stage_load_sb(stage, sn0 + sn, k)[0]
                for sn in range_constexpr(half_n)
                for k in range_constexpr(K_STEPS)
            ]

        def _seed_a_frag(stage, bank, wm):
            for k in range_constexpr(K_STEPS):
                pipe_a[bank][wm][k].store(_stage_load_a(stage, wm, k)[0])

        def _seed_b_frag(stage, bank, wn):
            for k in range_constexpr(K_STEPS):
                pipe_b[bank][wn][k].store(_stage_load_b(stage, wn, k)[0])

        def _load_seed_a(stage, bank):
            for sm in range_constexpr(half_m // 2):
                for k in range_constexpr(K_STEPS):
                    val, _ = _stage_load_sa(stage, sm, k)
                    pipe_sa[bank][sm][k].store(Vec.from_elements([val], fx.Int32))
            for wm in range_constexpr(half_m):
                _seed_a_frag(stage, bank, wm)

        def _load_seed_b(stage, bank):
            for sn in range_constexpr(half_n):
                for k in range_constexpr(K_STEPS):
                    val, _ = _stage_load_sb(stage, sn, k)
                    pipe_sb[bank][sn][k].store(Vec.from_elements([val], fx.Int32))
            for wn in range_constexpr(half_n):
                _seed_b_frag(stage, bank, wn)

        def _pipe_scale(regs):
            return [reg.load()[0] for row in regs for reg in row]

        def _noop():
            pass

        def _sched_hint(groups):
            """Ask the machine scheduler for one WMMA per DS burst across the produced group."""
            rocdl.sched_mfma(1)
            rocdl.sched_dsrd(2)
            for _ in range_constexpr(groups):
                rocdl.sched_mfma(1)
                rocdl.sched_dsrd(4)
            rocdl.sched_barrier(0)

        def _quadrant(wm0, wn0, act, wt, sa_k, sb_k, n_fast, need, produce, mid=None, mid_pos=0):
            rocdl.sched_barrier(0)
            ds.wait(need)
            rocdl.sched_barrier(0)
            n_lead = len(produce)
            assert n_lead <= WMMA_PER_Q, f"{n_lead} producers do not fit {WMMA_PER_Q} WMMA slots"
            for pos in range_constexpr(WMMA_PER_Q):
                _mma_block_range(wm0, wn0, act, wt, sa_k, sb_k, pos, 1, n_fast)
                if const_expr(mid is not None and pos == mid_pos):
                    mid()
                if const_expr(pos < n_lead):
                    produce[pos]()
                if const_expr(pos == n_lead - 1):
                    _sched_hint(n_lead - 1)
            rocdl.sched_barrier(0)
            return ds.issued

        def _compute_stage(
            stage, next_stage, bank, next_bank, future_slot, future_kt, fence_outstanding, has_next, parity
        ):
            """One scheduled K-tile stage: four quadrants, each hosting one DS producer group."""
            a_top, b_left = pipe_a[bank], pipe_b[bank]
            sa_top, sb_left = _pipe_scale(pipe_sa[bank]), _pipe_scale(pipe_sb[bank])

            a_bottom = [[None] * K_STEPS for _ in range_constexpr(half_m)]
            b_right = [[None] * K_STEPS for _ in range_constexpr(half_n)]
            sa_bottom, sb_right = [], []

            def _produce_b_right():
                out = [lambda: sb_right.extend(_scales_b(stage, half_n))]
                for wn in range_constexpr(half_n):
                    for k in range_constexpr(K_STEPS):

                        def _go(wn=wn, k=k):
                            b_right[wn][k] = _rmem(16, _stage_load_b(stage, half_n + wn, k)[0])

                        out.append(_go)
                return out

            def _produce_a_bottom():
                out = [lambda: sa_bottom.extend(_scales_a(stage, half_m // 2))]
                for wm in range_constexpr(half_m):
                    for k in range_constexpr(K_STEPS):

                        def _go(wm=wm, k=k):
                            a_bottom[wm][k] = _rmem(8, _stage_load_a(stage, half_m + wm, k)[0])

                        out.append(_go)
                return out

            def _produce_seed_a():
                def _scales():
                    if const_expr(has_next):
                        for sm in range_constexpr(half_m // 2):
                            for k in range_constexpr(K_STEPS):
                                val, _ = _stage_load_sa(next_stage, sm, k)
                                pipe_sa[next_bank][sm][k].store(Vec.from_elements([val], fx.Int32))

                out = [_scales]
                for wm in range_constexpr(half_m):

                    def _go(wm=wm):
                        if const_expr(has_next):
                            _seed_a_frag(next_stage, next_bank, wm)

                    out.append(_go)
                return out

            def _produce_seed_b():
                def _scales():
                    if const_expr(has_next):
                        for sn in range_constexpr(half_n):
                            for k in range_constexpr(K_STEPS):
                                val, _ = _stage_load_sb(next_stage, sn, k)
                                pipe_sb[next_bank][sn][k].store(Vec.from_elements([val], fx.Int32))

                out = [_scales]
                for wn in range_constexpr(half_n):

                    def _go(wn=wn):
                        if const_expr(has_next):
                            _seed_b_frag(next_stage, next_bank, wn)

                    out.append(_go)
                return out

            TL, TR = (0, 0), (0, half_n)
            BL, BR = (half_m, 0), (half_m, half_n)
            if const_expr(parity == 0):
                plan = [
                    (TL, a_top, b_left, _produce_b_right),
                    (TR, a_top, b_right, _produce_a_bottom),
                    (BL, a_bottom, b_left, _produce_seed_a),
                    (BR, a_bottom, b_right, _produce_seed_b),
                ]
            else:
                plan = [
                    (TL, a_top, b_left, _produce_a_bottom),
                    (BL, a_bottom, b_left, _produce_b_right),
                    (TR, a_top, b_right, _produce_seed_b),
                    (BR, a_bottom, b_right, _produce_seed_a),
                ]

            need, mid = seed_mark[bank], None
            for idx in range_constexpr(4):
                (wm0, wn0), act, wt, produce = plan[idx]
                if const_expr(idx == 2):
                    need = ds.issued
                    refill = const_expr(future_kt is not None)
                    prepared = _prepare_tdm(future_slot, future_kt) if const_expr(refill) else None
                    rocdl.sched_barrier(0)
                    ds.wait(need)
                    rocdl.sched_barrier(0)
                    if const_expr(has_next):
                        pipeline_fence_signal(outstanding=fence_outstanding, use_cluster=False)

                        def mid():
                            rocdl.sched_barrier(0)
                            pipeline_fence_wait(use_cluster=False)
                            if const_expr(refill):
                                tdm_ops.tensor_load_2d(prepared)
                            rocdl.sched_barrier(0)

                _quadrant(
                    wm0,
                    wn0,
                    act,
                    wt,
                    sa_top + sa_bottom,
                    sb_left + sb_right,
                    idx > 0,
                    need,
                    produce() if const_expr(idx != 2) else [_noop] * (FENCE_WAIT_POS + 1) + produce(),
                    mid,
                    FENCE_WAIT_POS,
                )
                if const_expr(idx == 2):
                    mid = None
                need = ds.issued if const_expr(idx < 2) else need
            seed_mark[next_bank] = ds.issued

        # Keep all four slots in flight and statically expand one revolution.
        for i in range_constexpr(num_buffers):
            tdm_ops.tensor_load_2d(_prepare_tdm(i, i))
        pipeline_fence(outstanding=num_buffers - 1, use_cluster=False)
        _load_seed_a(0, 0)
        _load_seed_b(0, 0)
        seed_mark[0] = ds.issued

        n_steady = K_TILES - num_buffers

        def _run_all(parity):
            for rev in range(n_steady // num_buffers):
                for s in range_constexpr(num_buffers):
                    kt = rev * num_buffers + s
                    _compute_stage(
                        s, (s + 1) % num_buffers, s % 2, (s + 1) % 2, s, kt + num_buffers, num_buffers - 2, True, parity
                    )
                if rev % cluster_sync_revs == cluster_sync_revs - 1:
                    cluster.cluster_barrier()
            for j in range_constexpr(num_buffers):
                # No refills left, so the drain must ratchet the TDM allowance down: stage j
                # seeds slot j+1, whose prologue load must have landed by then.
                _compute_stage(
                    j,
                    (j + 1) % num_buffers,
                    j % 2,
                    (j + 1) % 2,
                    j,
                    None,
                    num_buffers - 2 - j,
                    j < num_buffers - 1,
                    parity,
                )

        wave_parity = rocdl.readfirstlane(T.i32, wave % 2)
        if wave_parity == 0:
            _run_all(0)
        else:
            _run_all(1)

        rocdl.s_wait_dscnt(0)
        accs = [c_frags[idx].load() for idx in range_constexpr(n_acc)]

        pipeline_fence(outstanding=0, use_cluster=True)
        # acc (wm, wn) is 32 N-cols x 16 M-rows: lane holds M = wm*16+lane16 and, per 8-wide
        # half, N = wn*32 + half*16 + kgrp*8 + v.
        for wm in range_constexpr(wmma_m_rep):
            row_rel = wmb + wm * 16 + lane16
            for wn in range_constexpr(wmma_n_rep):
                acc = accs[wm * wmma_n_rep + wn]
                for half in range_constexpr(2):
                    col_rel = wnb + wn * 32 + half * 16 + kgrp * 8
                    h = acc.shuffle(acc, list(range(half * 8, half * 8 + 8))).to(oc)
                    fx.ptr_store(h.bitcast(fx.Int8), base_ptr + (row_rel * tile_n + col_rel) * 2)
        workgroup_barrier(use_cluster=False)
        c_off_rt = blk_m64 * ldc64 + blk_n64
        gC_base = fx.recast_iter(fx.PointerType.get(oc.ir_type, arg_c.address_space), arg_c)
        gtC = _gv(gC_base, c_off_rt, (tile_m, tile_n), (tile_n, 1))
        atomC = fx.rocdl.make_tdm_atom(
            gtC,
            [mn_oob, None],
            strides=[ldc64, None],
            num_warps=num_waves,
        )
        fx.copy(atomC, _view(fx.recast_iter(oc, base_ptr), (tile_m, tile_n), (tile_n, 1)), gtC)
        tdm_ops.tensor_wait(0)

    gx = (i32_m + (tile_m - 1)) // tile_m
    gy = (N + (tile_n - 1)) // tile_n
    # Split gx exactly, so no workgroup is left over to recompute a duplicate tile.
    pow2 = gx & -gx
    capped = (pow2 < m_run_max).select(pow2, fx.Int32(m_run_max))
    m_run = ((gx > m_run_max) & (pow2 >= m_run_min)).select(capped, gx)
    grid_arg = (m_run, gy, gx // m_run)
    # Runtime N/K shape checks belong to the caller.
    kernel_gemm_a4w4_256x256(
        arg_c,
        arg_a,
        arg_b,
        arg_scale_a,
        arg_scale_b,
        i32_m,
        N,
        K,
        i32_lda,
        i32_ldc,
        value_attrs={"rocdl.cluster_dims": f"{cluster_m},{cluster_n},1"},
    ).launch(grid=grid_arg, block=(block, 1, 1), stream=stream, cluster=(cluster_m, cluster_n, 1))


launch_gemm_a4w4_256x256.compile_hints["llvm_options"] = {
    "amdgpu-expert-scheduling-mode": True,
}
