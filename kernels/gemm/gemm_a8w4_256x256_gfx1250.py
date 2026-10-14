# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""A8W4 (FP8 activation x MXFP4 E2M1 weight) 32x4-preshuffled MXscale GEMM for gfx1250."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster, tdm_ops
from flydsl.expr.typing import Constexpr, T, as_ir_value
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import check_smem_capacity
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from kernels.common.gfx1250_cluster import compute_mcast_masks
from kernels.common.kernels_common import format_kernel_name

from .gemm_common_gfx1250 import (
    make_gemm_grid,
    make_lds_copy_ops,
    pipeline_fence,
    pipeline_fence_signal,
    pipeline_fence_wait,
    workgroup_barrier,
)

KPAIR_BY_BUFFERS = {3: 2, 4: 1}


@flyc.jit
def launch_gemm_a8w4_256x256(
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
    split_k: Constexpr[int] = 1,
    a_preshuffle: Constexpr[bool] = False,
):
    """N must cover whole ``tile_n * cluster_n`` groups; M is unrestricted.

    A preshuffled A must pad an odd M to a whole row pair. K per split must be a
    multiple of ``tile_k * KPAIR``, and at least 512 for the 256x256 schedule.
    """

    assert (
        tile_m in (128, 256) and tile_n in (128, 256) and tile_k == 128 and (m_warp, n_warp) == (2, 2)
    ), "supported tile profiles are 128/256 x 128/256 x 128 with 2x2 waves"
    assert num_buffers in KPAIR_BY_BUFFERS, f"num_buffers must be one of {sorted(KPAIR_BY_BUFFERS)}, got {num_buffers}"
    assert (
        1 <= cluster_m < 16 and 1 <= cluster_n < 16 and 1 < cluster_m * cluster_n <= 16
    ), f"cluster_m*cluster_n must be 2..16, got {cluster_m}x{cluster_n}"
    assert split_k >= 1, "split_k must be positive"
    cluster_sync_revs = 8
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    WAVE = fx.num_warp_threads()
    PACK_TK = tile_k // 2  # B row bytes per K-tile (FP4 packed 2/byte)
    SA_SUPERS = tile_m // 32
    SB_SUPERS = tile_n // 32
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    half_m, half_n = wmma_m_rep // 2, wmma_n_rep // 2
    n_slots = half_m * half_n  # WMMAs per quadrant
    n_acc = wmma_m_rep * wmma_n_rep
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    # Each of the num_buffers slots holds KPAIR K-tiles side by side in every LDS row
    KPAIR = KPAIR_BY_BUFFERS[num_buffers]
    A_SEED_ORDER = (0, 2, 1, 3)
    UNROLL = KPAIR * num_buffers
    SUPER_K = tile_k * KPAIR
    LDS_PAD_A = 16
    A_PAIR = 2 if a_preshuffle else 1
    assert not a_preshuffle or tile_m % 2 == 0, "a_preshuffle needs an even tile_m"
    A_LDS_ROWS = tile_m // A_PAIR
    A_TDM_ROW = A_PAIR * SUPER_K
    A_LDS_ROW = A_TDM_ROW + LDS_PAD_A
    C_LDS_ROW = tile_n + 8
    B_LDS_ROW = PACK_TK * 16 * KPAIR
    STAGE_A = A_LDS_ROWS * A_LDS_ROW
    STAGE_B = (tile_n // 16) * B_LDS_ROW
    STAGE_SA = SA_SUPERS * SUPER_K
    STAGE_SB = SB_SUPERS * SUPER_K
    # B first so it keeps a 64-KiB-aligned base; the rest packs behind it.
    PLANAR_B_BASE = 0
    PLANAR_A_BASE = PLANAR_B_BASE + num_buffers * STAGE_B
    PLANAR_SA_BASE = PLANAR_A_BASE + num_buffers * STAGE_A
    PLANAR_SB_BASE = PLANAR_SA_BASE + num_buffers * STAGE_SA
    PLANAR_END = PLANAR_SB_BASE + num_buffers * STAGE_SB

    partial_bytes = 4 if split_k > 1 else 2
    ARENA_B = max(PLANAR_END, tile_m * C_LDS_ROW * partial_bytes)
    check_smem_capacity(ARENA_B, str(get_hip_arch()))
    kernel_name = format_kernel_name(
        f"gemm_a8w4_mx32_compute_t{tile_m}x{tile_n}x{tile_k}"
        f"_mw{m_warp}_nw{n_warp}_nb{num_buffers}_sk{split_k}"
        f"_cm{cluster_m}_cn{cluster_n}"
        + ("_apre" if a_preshuffle else "")
    )

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_gemm_a8w4_256x256(
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

        K_TILES = i32_k // (tile_k * split_k)
        k64 = fx.Int64(i32_k)
        lda64 = fx.Int64(i32_lda)
        ldc64 = fx.Int64(i32_ldc)
        Kp16 = (k64 // 2) * 16

        tid = fx.thread_idx.x
        bid_x, bid_y, bid_z = fx.block_idx
        if const_expr(split_k > 1):
            m_chunks = fx.grid_dim.z // split_k
            split_idx = bid_z // m_chunks
            m_chunk = bid_z - split_idx * m_chunks
            kt_base = fx.Int64(split_idx) * fx.Int64(K_TILES)
        else:
            split_idx = fx.Int32(0)
            m_chunk = bid_z
            kt_base = fx.Int64(0)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // WAVE))
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        wave_m = wave // n_warp
        wave_n = wave % n_warp
        local_x, local_y = cluster.compute_cluster_position()
        a_mask, b_mask = compute_mcast_masks(local_x, local_y, cluster_m, cluster_n)
        blk_m = (m_chunk * fx.grid_dim.x + bid_x) * tile_m
        blk_n = bid_y * tile_n
        blk_m64 = fx.Int64(blk_m)
        blk_n64 = fx.Int64(blk_n)
        mn_oob = i32_m - blk_m  # valid M rows (A / C)
        a_oob = (mn_oob + 1) >> 1 if const_expr(a_preshuffle) else mn_oob
        sa_oob = (i32_m + 31) // 32 - blk_m // 32  # valid M-supers (scale-A)

        arena = fx.SharedAllocator(static=False)
        arena.allocate(ARENA_B)
        base_ptr = arena.base_ptr

        def _planar_base(offset, stride, stage):
            ptr = fx.add_offset(base_ptr, offset + stage * stride)
            return fx.Int64(fx.ptrtoint(ptr))

        def _view(ptr, shape, stride):
            return fx.Tensor(fx.make_view(ptr, fx.make_layout(shape, stride)))

        def _gv(base, off, shape, stride):
            return _view(fx.add_offset(base, off), shape, stride)

        oc = fx.Float16 if out_is_f16 else fx.BFloat16
        partial_elem = fx.Float32 if split_k > 1 else oc
        gA_base = fx.recast_iter(fx.Int8, arg_a)
        gB_base = fx.recast_iter(fx.Int8, arg_b)

        k_elem0 = kt_base * tile_k
        a_off0 = blk_m64 * lda64 + k_elem0 * A_PAIR
        b_off0 = (blk_n64 // 16) * Kp16 + k_elem0 * 8
        sa_off0 = (blk_m64 // 32) * k64 + k_elem0
        sb_off0 = (blk_n64 // 32) * k64 + k_elem0

        gA = _gv(gA_base, a_off0, (A_LDS_ROWS, A_PAIR * tile_k), (A_PAIR * tile_k, 1))
        gB = _gv(gB_base, b_off0, (tile_n // 16, PACK_TK * 16), (PACK_TK * 16, 1))
        gSA = _gv(arg_scale_a, sa_off0, (SA_SUPERS, tile_k), (tile_k, 1))
        gSB = _gv(arg_scale_b, sb_off0, (SB_SUPERS, tile_k), (tile_k, 1))

        def _build_tdm_desc(owner):
            if const_expr(owner == 0):
                tensor, offset, shape, lds_stride = gA, PLANAR_A_BASE, (A_LDS_ROWS, A_TDM_ROW), A_LDS_ROW
                stride, mask, bound, pad, early = i32_lda * A_PAIR, a_mask, a_oob, LDS_PAD_A, False
            elif const_expr(owner == 1):
                tensor, offset, shape, lds_stride = gB, PLANAR_B_BASE, (tile_n // 16, B_LDS_ROW), B_LDS_ROW
                stride, mask, bound, pad, early = i32_k * 8, b_mask, None, 0, False
            elif const_expr(owner == 2):
                tensor, offset, shape, lds_stride = gSA, PLANAR_SA_BASE, (SA_SUPERS, SUPER_K), SUPER_K
                stride, mask, bound, pad, early = i32_k, a_mask, sa_oob, 0, True
            else:
                tensor, offset, shape, lds_stride = gSB, PLANAR_SB_BASE, (SB_SUPERS, SUPER_K), SUPER_K
                stride, mask, bound, pad, early = i32_k, b_mask, None, 0, True
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
            return Vec(desc.dgroup0), Vec(desc.dgroup1), fx.Int32(shape[0] * lds_stride), fx.Int32(shape[1])

        dgroup0 = Vec.filled(4, 0, fx.Int32)
        dgroup1 = Vec.filled(8, 0, fx.Int32)
        tdm_lds_step, tdm_global_step = fx.Int32(0), fx.Int32(0)
        if wave == 0:
            dgroup0, dgroup1, tdm_lds_step, tdm_global_step = _build_tdm_desc(0)
        elif wave == 1:
            dgroup0, dgroup1, tdm_lds_step, tdm_global_step = _build_tdm_desc(1)
        elif wave == 2:
            dgroup0, dgroup1, tdm_lds_step, tdm_global_step = _build_tdm_desc(2)
        else:
            dgroup0, dgroup1, tdm_lds_step, tdm_global_step = _build_tdm_desc(3)

        tdm_desc = tdm_ops.TDMDescriptor2D(as_ir_value(dgroup0), as_ir_value(dgroup1))
        tdm_base_lds, tdm_base_lo, tdm_base_hi = dgroup0[1], dgroup0[2], dgroup0[3]

        def _prepare_tdm(slot, tile_delta):
            desc = tdm_ops.update_tensor_descriptor_2d_lds_addr(tdm_desc, tdm_base_lds + tdm_lds_step * slot)
            return tdm_ops.update_tensor_descriptor_2d_addr64(desc, tdm_base_lo, tdm_base_hi, tile_delta)

        wmb = wave_m * warp_tile_m
        wnb = wave_n * warp_tile_n

        wmma_atoms = [
            [
                fx.make_mma_atom(
                    fx.rocdl.WMMAScale(
                        WMMA_M,
                        WMMA_N,
                        WMMA_K,
                        fx.Float4E2M1FN,
                        fx.Float8E4M3FN,
                        fx.Float32,
                        opsel_a=sb_sel,
                        opsel_b=sa_sel,
                    )
                )
                for sa_sel in range_constexpr(2)
            ]
            for sb_sel in range_constexpr(2)
        ]
        c_frags = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(n_acc)]
        for cf in c_frags:
            cf.store(Vec.filled(8, 0.0, fx.Float32))

        def _rmem(n, v):
            t = fx.make_rmem_tensor(n, fx.Int32)
            t.store(v)
            return t

        def _mma_one(wm, wn, act, wt, sa_k, sb_k):
            idx = wm * wmma_n_rep + wn
            fx.gemm(
                wmma_atoms[wn % 2][wm % 2],
                c_frags[idx],
                wt,
                act,
                c_frags[idx],
                scale_a=sb_k[wn // 2],
                scale_b=sa_k[wm // 2],
            )

        def _mma_range(quad, start, count, n_fast=False):
            """Issue `count` WMMAs from linear position `start` of a (wm0, wn0, act, wt, sa, sb) block.

            The two orders are not interchangeable: they decide which operand a run of
            WMMAs holds constant, which is what the interleaved DS reads are scheduled
            against.  N-fast walks a row of B, M-fast walks a column of A.
            """
            wm0, wn0, act, wt, sa_k, sb_k = quad
            for pos in range_constexpr(start, start + count):
                i, j = (pos // len(wt), pos % len(wt)) if n_fast else (pos % len(act), pos // len(act))
                _mma_one(wm0 + i, wn0 + j, act[i], wt[j], sa_k, sb_k)

        cluster.cluster_barrier()
        # Keep fragment displacements as DS immediates inside the K loop.
        stage_a_addr, stage_b_addr, stage_sa_addr, stage_sb_addr = [], [], [], []
        sa_row, sb_col = wmb + lane, wnb + lane
        if const_expr(a_preshuffle):
            a_pair = wave_m * (warp_tile_m // 2) + (lane16 >> 1)
            a_byte = fx.Int64(a_pair * A_LDS_ROW + (lane16 & 1) * WMMA_K + kgrp * 16)
            a_par_step, a_frag_row_step = A_PAIR * tile_k, (16 // A_PAIR) * A_LDS_ROW
        else:
            a_byte = fx.Int64((wmb + lane16) * A_LDS_ROW + kgrp * 16)
            a_par_step, a_frag_row_step = tile_k, 16 * A_LDS_ROW
        b_byte = fx.Int64((wnb // 16) * B_LDS_ROW + kgrp * 256 + lane16 * 16)
        sa_byte = fx.Int64((sa_row // 32) * SUPER_K + (sa_row % 32) * 4)
        sb_byte = fx.Int64((sb_col // 32) * SUPER_K + (sb_col % 32) * 4)
        for addr_stage in range_constexpr(UNROLL):
            slot, par = addr_stage // KPAIR, addr_stage % KPAIR
            stage_a_addr.append(_planar_base(PLANAR_A_BASE, STAGE_A, slot) + a_byte + par * a_par_step)
            stage_b_addr.append(_planar_base(PLANAR_B_BASE, STAGE_B, slot) + b_byte + par * PACK_TK * 16)
            stage_sa_addr.append(_planar_base(PLANAR_SA_BASE, STAGE_SA, slot) + sa_byte + par * tile_k)
            stage_sb_addr.append(_planar_base(PLANAR_SB_BASE, STAGE_SB, slot) + sb_byte + par * tile_k)

        lds_load_b32, _ = make_lds_copy_ops(32)
        lds_load_b128, _ = make_lds_copy_ops(128)

        def _stage_load_a(stage, wm):
            row_off = wm * a_frag_row_step
            v = [lds_load_b128(stage_a_addr[stage], row_off + 32 * j) for j in range_constexpr(4)]
            return v[0].shuffle(v[1], list(range(8))).shuffle(v[2].shuffle(v[3], list(range(8))), list(range(16)))

        def _stage_load_b(stage, wn):
            col_off = wn * B_LDS_ROW
            v0 = lds_load_b128(stage_b_addr[stage], col_off)
            v1 = lds_load_b128(stage_b_addr[stage], col_off + 512)
            return v0.shuffle(v1, list(range(8)))

        def _stage_load_sa(stage, sm):
            return lds_load_b32(stage_sa_addr[stage], sm * SUPER_K)[0]

        def _stage_load_sb(stage, sn):
            return lds_load_b32(stage_sb_addr[stage], sn * SUPER_K)[0]

        # Separate seed banks prevent WMMA source/address register coalescing.
        pipe_a = [[fx.make_rmem_tensor(16, fx.Int32) for _ in range_constexpr(half_m)] for _ in range_constexpr(2)]
        pipe_b = [[fx.make_rmem_tensor(8, fx.Int32) for _ in range_constexpr(half_n)] for _ in range_constexpr(2)]
        pipe_sa = [[fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(half_m // 2)] for _ in range_constexpr(2)]
        pipe_sb = [[fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(half_n // 2)] for _ in range_constexpr(2)]

        def _load_seed_a_scales(stage):
            for sm in range_constexpr(half_m // 2):
                pipe_sa[stage % 2][sm].store(Vec.from_elements([_stage_load_sa(stage, sm)], fx.Int32))

        def _load_seed_a_fragment(stage, wm):
            pipe_a[stage % 2][wm].store(_stage_load_a(stage, wm))

        def _load_seed_b_scales(stage):
            for sn in range_constexpr(half_n // 2):
                pipe_sb[stage % 2][sn].store(Vec.from_elements([_stage_load_sb(stage, sn)], fx.Int32))

        def _load_seed_b_fragment(stage, wn):
            pipe_b[stage % 2][wn].store(_stage_load_b(stage, wn))

        def _load_seeds(stage):
            _load_seed_a_scales(stage)
            for wm in range_constexpr(half_m):
                _load_seed_a_fragment(stage, wm)
            _load_seed_b_scales(stage)
            for wn in range_constexpr(half_n):
                _load_seed_b_fragment(stage, wn)

        def _seeded_quadrant(stage):
            """The top-left operands and scales the previous stage seeded for `stage`."""
            bank = stage % 2
            return (
                pipe_a[bank],
                pipe_b[bank],
                [r.load()[0] for r in pipe_sa[bank]],
                [r.load()[0] for r in pipe_sb[bank]],
            )

        def _ds_wait(count):
            rocdl.sched_barrier(0)
            rocdl.s_wait_dscnt(count)
            rocdl.sched_barrier(0)

        def _sched_hint(mfma, lead, groups, tail=0):
            """Interleave `mfma` WMMAs with each DS burst of the producers just issued."""
            if const_expr(lead > 0):
                rocdl.sched_mfma(mfma)
                rocdl.sched_dsrd(lead)
            for _ in range_constexpr(groups):
                rocdl.sched_mfma(mfma)
                rocdl.sched_dsrd(4)
            if const_expr(tail > 0):
                rocdl.sched_mfma(tail)
            rocdl.sched_barrier(0)

        # Producers: WMMA p of a quadrant, then the p-th group of DS reads behind it.
        def _load_bottom_a(stage, quad, n_fast=False):
            _mma_range(quad, 0, 1, n_fast)
            sa = [_stage_load_sa(stage, half_m // 2 + sm) for sm in range_constexpr(half_m // 2)]
            frags = []
            for wm in range_constexpr(half_m):
                _mma_range(quad, 1 + wm, 1, n_fast)
                frags.append(_rmem(16, _stage_load_a(stage, half_m + wm)))
            return frags, sa

        def _load_right_b(stage, quad, n_fast=False):
            _mma_range(quad, 0, 1, n_fast)
            sb = [_stage_load_sb(stage, half_n // 2 + sn) for sn in range_constexpr(half_n // 2)]
            frags = []
            for pair in range_constexpr(half_n // 2):
                _mma_range(quad, 1 + pair, 1, n_fast)
                frags.extend(_rmem(8, _stage_load_b(stage, half_n + 2 * pair + k)) for k in range_constexpr(2))
            return frags, sb

        def _seed_next_a(stage, quad, start, stop, n_fast=False):
            for pos in range_constexpr(start, stop):
                _mma_range(quad, pos, 1, n_fast)
                if const_expr(pos == 0):
                    _load_seed_a_scales(stage)
                else:
                    _load_seed_a_fragment(stage, A_SEED_ORDER[pos - 1])

        def _seed_next_b(stage, quad, n_fast=False):
            _mma_range(quad, 0, 1, n_fast)
            _load_seed_b_scales(stage)
            for pair in range_constexpr(half_n // 2):
                _mma_range(quad, 1 + pair, 1, n_fast)
                for k in range_constexpr(2):
                    _load_seed_b_fragment(stage, 2 * pair + k)

        def _compute_generic_stage(stage):
            """Straight scheduled path for smaller M/N tiles."""
            act = [_rmem(16, _stage_load_a(stage, wm)) for wm in range_constexpr(wmma_m_rep)]
            wt = [_rmem(8, _stage_load_b(stage, wn)) for wn in range_constexpr(wmma_n_rep)]
            sa_k = [_stage_load_sa(stage, sm) for sm in range_constexpr(wmma_m_rep // 2)]
            sb_k = [_stage_load_sb(stage, sn) for sn in range_constexpr(wmma_n_rep // 2)]
            rocdl.s_wait_dscnt(0)
            _mma_range((0, 0, act, wt, sa_k, sb_k), 0, n_acc)

        def _compute_even_stage(stage, future_kt, phase):
            """Even-wave stage ordered to align sibling READY arrival."""
            next_stage, boundary, mfma = (stage + 1) % UNROLL, stage % KPAIR == KPAIR - 1, 3 - phase
            a_top, b_left, sa_top, sb_left = _seeded_quadrant(stage)

            # q00: produce current B while admitting four older reads.
            _ds_wait(4)
            q00 = (0, 0, a_top, b_left, sa_top, sb_left)
            b_right, sb_right = _load_right_b(stage, q00)
            sb_k = sb_left + sb_right
            _sched_hint(mfma, 2, 2)
            _mma_range(q00, 3, 5)
            _ds_wait(10)
            _mma_range(q00, 8, 8)

            # q01: produce current A and signal READY.
            _ds_wait(4)
            q01 = (0, half_n, a_top, b_right, sa_top, sb_k)
            a_bottom, sa_bottom = _load_bottom_a(stage, q01)
            sa_k = sa_top + sa_bottom
            _sched_hint(mfma, 2, 4)
            _mma_range(q01, 5, 3)
            _ds_wait(17)
            if const_expr(boundary):
                rocdl.s_wait_dscnt(0)
                pipeline_fence_signal(outstanding=num_buffers - 2, use_cluster=False)
            rocdl.sched_barrier(0)
            _mma_range(q01, 8, 8)
            rocdl.sched_barrier(0)
            if const_expr(boundary):
                pipeline_fence_wait(use_cluster=False)
                rocdl.sched_barrier(0)
            rocdl.sched_barrier(0)

            # q10: seed the next A; phase 0 issues the refill early, phase 1 mid-quadrant.
            q10 = (half_m, 0, a_bottom, b_left, sa_k, sb_left)
            if const_expr(boundary):
                refill = _prepare_tdm(stage // KPAIR, future_kt)
                if const_expr(phase == 0):
                    tdm_ops.tensor_load_2d(refill)
            _seed_next_a(next_stage, q10, 0, 3)
            _sched_hint(mfma, 2, 2)
            if const_expr(boundary and phase != 0):
                tdm_ops.tensor_load_2d(refill)
            rocdl.sched_barrier(0)
            _seed_next_a(next_stage, q10, 3, 5)
            _sched_hint(mfma, 0, 2)
            _mma_range(q10, 5, 11)

            # q11: leave the next B tail in flight.
            rocdl.sched_barrier(0)
            q11 = (half_m, half_n, a_bottom, b_right, sa_k, sb_k)
            _seed_next_b(next_stage, q11)
            _sched_hint(mfma, 2, 2)
            _mma_range(q11, 3, 13)

        def _compute_odd_stage(stage, future_kt):
            """Odd-wave stage: N-fast quadrants with the READY window in q10."""
            next_stage, boundary = (stage + 1) % UNROLL, stage % KPAIR == KPAIR - 1
            a_top, b_left, sa_top, sb_left = _seeded_quadrant(stage)

            # q00: produce current A while retiring the prior A tail.
            _ds_wait(8)
            q00 = (0, 0, a_top, b_left, sa_top, sb_left)
            a_bottom, sa_bottom = _load_bottom_a(stage, q00, True)
            sa_k = sa_top + sa_bottom
            _sched_hint(3, 2, 4)
            _mma_range(q00, 5, 3, True)
            _ds_wait(17)
            _mma_range(q00, 8, 8, True)
            rocdl.sched_barrier(0)

            # q10: produce current B and retire the older A tail.
            rocdl.s_wait_dscnt(8)
            rocdl.sched_barrier(0)
            q10 = (half_m, 0, a_bottom, b_left, sa_k, sb_left)
            b_right, sb_right = _load_right_b(stage, q10, True)
            sb_k = sb_left + sb_right
            _sched_hint(3, 2, 2)
            _mma_range(q10, 3, 5, True)
            _ds_wait(10)
            # Signal READY early: the window between signal and wait is what absorbs a
            # late sibling, and the TDM it waits on has far more slack than the barrier.
            if const_expr(boundary):
                rocdl.s_wait_dscnt(0)
                pipeline_fence_signal(outstanding=num_buffers - 2, use_cluster=False)
            rocdl.sched_barrier(0)
            _mma_range(q10, 8, 8, True)
            rocdl.sched_barrier(0)
            if const_expr(boundary):
                pipeline_fence_wait(use_cluster=False)
            rocdl.sched_barrier(0)

            # q01: produce next B, then let two more useful WMMAs run
            # before wave1/wave3 submit their B/SB refills.
            if const_expr(boundary):
                refill = _prepare_tdm(stage // KPAIR, future_kt)
            q01 = (0, half_n, a_top, b_right, sa_top, sb_k)
            _seed_next_b(next_stage, q01, True)
            _mma_range(q01, 3, 2, True)
            _sched_hint(3, 2, 2, tail=2)
            _mma_range(q01, 5, 2, True)
            if const_expr(boundary):
                rocdl.sched_barrier(0)
                tdm_ops.tensor_load_2d(refill)
            _mma_range(q01, 7, 9, True)

            # q11: leave the next A tail in flight.
            q11 = (half_m, half_n, a_bottom, b_right, sa_k, sb_k)
            _seed_next_a(next_stage, q11, 0, 5, True)
            _sched_hint(3, 1, 4)
            _mma_range(q11, 5, 11, True)

        def _compute_drain_stage(stage, fence_outstanding, has_next):
            """Shared drain stage: no refill; the next seeds load once q10 has started."""
            next_stage, boundary = (stage + 1) % UNROLL, stage % KPAIR == KPAIR - 1
            a_top, b_left, sa_top, sb_left = _seeded_quadrant(stage)

            # K512 reaches this drain without a steady iteration.
            rocdl.s_wait_dscnt(4)
            _mma_range((0, 0, a_top, b_left, sa_top, sb_left), 0, n_slots)
            a_bottom = [_rmem(16, _stage_load_a(stage, half_m + wm)) for wm in range_constexpr(half_m)]
            sa_bottom = [_stage_load_sa(stage, half_m // 2 + sm) for sm in range_constexpr(half_m // 2)]
            b_right = [_rmem(8, _stage_load_b(stage, half_n + wn)) for wn in range_constexpr(half_n)]
            sb_right = [_stage_load_sb(stage, half_n // 2 + sn) for sn in range_constexpr(half_n // 2)]
            sa_k, sb_k = sa_top + sa_bottom, sb_left + sb_right

            q10 = (half_m, 0, a_bottom, b_left, sa_k, sb_k)
            if const_expr(has_next):
                rocdl.s_wait_dscnt(9)
                _mma_range(q10, 0, 13)
                if const_expr(boundary):
                    rocdl.s_wait_dscnt(0)
                    pipeline_fence_signal(outstanding=fence_outstanding, use_cluster=False)
                rocdl.sched_barrier(0)
                _mma_range(q10, 13, 3)
                if const_expr(boundary):
                    pipeline_fence_wait(use_cluster=False)
                else:
                    rocdl.s_wait_dscnt(0)
                rocdl.sched_barrier(0)
                _load_seeds(next_stage)
            else:
                rocdl.s_wait_dscnt(0)
                _mma_range(q10, 0, n_slots)
            _mma_range((0, half_n, a_top, b_right, sa_k, sb_k), 0, n_slots)
            _mma_range((half_m, half_n, a_bottom, b_right, sa_k, sb_k), 0, n_slots)

        # Keep every slot in flight and statically expand one revolution of KPAIR*num_buffers
        # K-tiles.  One TDM covers a whole slot, so it is issued once per KPAIR stages.
        SUPERS = K_TILES // KPAIR
        if const_expr(tile_m == 256 and tile_n == 256):
            last_delta = (SUPERS - 1) * tdm_global_step
            for i in range_constexpr(num_buffers):
                seed_delta = fx.Int32(i) * tdm_global_step
                seed_delta = (seed_delta < last_delta).select(seed_delta, last_delta)
                tdm_ops.tensor_load_2d(_prepare_tdm(i, seed_delta))
            pipeline_fence(outstanding=num_buffers - 1, use_cluster=False)

            n_full = (SUPERS + num_buffers - 1) // num_buffers - 1
            drain_s = SUPERS - n_full * num_buffers  # 1..num_buffers
            slot_delta = [fx.Int32(c + num_buffers) * tdm_global_step for c in range_constexpr(num_buffers)]

            def _run_steady(owner_parity, phase):
                _load_seeds(0)
                rocdl.s_wait_dscnt(0)
                for rev in range(n_full):
                    rev_delta = (rev * num_buffers) * tdm_global_step
                    for g in range_constexpr(UNROLL):
                        delta = rev_delta + slot_delta[g // KPAIR]
                        delta = (delta < last_delta).select(delta, last_delta)
                        if const_expr(owner_parity == 0):
                            _compute_even_stage(g, delta, phase)
                        else:
                            _compute_odd_stage(g, delta)
                    if rev % cluster_sync_revs == cluster_sync_revs - 1:
                        cluster.cluster_barrier()

            wave_parity = fx.Int32(rocdl.readfirstlane(T.i32, wave % 2))
            if wave_parity == 0:
                if wave < 2:
                    _run_steady(0, 0)
                else:
                    _run_steady(0, 1)
            else:
                _run_steady(1, 0)

            # Retire the last steady producer before the shared drain.
            rocdl.s_wait_dscnt(0)
            rocdl.sched_barrier(0)
            for g in range_constexpr(UNROLL):
                if g < drain_s * KPAIR:
                    _compute_drain_stage(g, num_buffers - 2 - g // KPAIR, g < UNROLL - 1)
        else:
            # Each group fills distinct LDS slots. Peers meet only before the
            # next group can reuse those slots for another multicast.
            for group in range((SUPERS + num_buffers - 1) // num_buffers):
                for slot in range_constexpr(num_buffers):
                    super_idx = group * num_buffers + slot
                    if super_idx < SUPERS:
                        tdm_ops.tensor_load_2d(_prepare_tdm(slot, super_idx * tdm_global_step))
                pipeline_fence(outstanding=0, use_cluster=False)
                for slot in range_constexpr(num_buffers):
                    if group * num_buffers + slot < SUPERS:
                        for par in range_constexpr(KPAIR):
                            _compute_generic_stage(slot * KPAIR + par)
                cluster.cluster_barrier()
        rocdl.s_wait_dscnt(0)
        accs = [c_frags[idx].load() for idx in range_constexpr(n_acc)]

        pipeline_fence(outstanding=0, use_cluster=True)
        for wm in range_constexpr(wmma_m_rep):
            row_rel = wmb + wm * 16 + lane16
            for wn in range_constexpr(wmma_n_rep):
                col_rel = wnb + wn * 16 + kgrp * 8
                h = accs[wm * wmma_n_rep + wn].to(partial_elem)
                fx.ptr_store(h.bitcast(fx.Int8), base_ptr + (row_rel * C_LDS_ROW + col_rel) * partial_bytes)
        workgroup_barrier(use_cluster=False)
        c_off_rt = blk_m64 * ldc64 + blk_n64
        if const_expr(split_k > 1):
            c_off_rt += fx.Int64(split_idx) * fx.Int64(i32_m) * ldc64
        gtC = _gv(fx.recast_iter(partial_elem, arg_c), c_off_rt, (tile_m, C_LDS_ROW), (C_LDS_ROW, 1))
        atomC = fx.rocdl.make_tdm_atom(gtC, [mn_oob, tile_n], strides=[ldc64, None], num_warps=num_waves)
        fx.copy(atomC, _view(fx.recast_iter(partial_elem, base_ptr), (tile_m, C_LDS_ROW), (C_LDS_ROW, 1)), gtC)
        tdm_ops.tensor_wait(0)

    grid_arg = make_gemm_grid(i32_m, N, tile_m, tile_n, cluster_m, split_k=split_k)
    # Runtime N/K shape checks belong to the caller.
    kernel_gemm_a8w4_256x256(
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


launch_gemm_a8w4_256x256.compile_hints["llvm_options"] = {
    "amdgpu-expert-scheduling-mode": True,
}
