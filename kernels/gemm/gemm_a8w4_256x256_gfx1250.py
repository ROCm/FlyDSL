# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""A8W4 (FP8 activation x MXFP4 E2M1 weight) 32x4-preshuffled MXscale GEMM for gfx1250."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_dialect
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
):
    """Launch the 1x2-cluster kernel; N must be divisible by 512, K by 128, with K >= 512."""

    assert (tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, cluster_m, cluster_n) == (
        256,
        256,
        128,
        2,
        2,
        4,
        1,
        2,
    ), "only the tuned 256x256x128, 2x2-wave, 4-buffer, 1x2-cluster profile is supported"
    cluster_sync_revs = 8
    m_run_max, m_run_min = 32, 8
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    WAVE = 32
    PACK_TK = tile_k // 2  # B row bytes per K-tile (FP4 packed 2/byte)
    SC_WORDS = tile_k // 4  # scale i32 words per super-row per K-tile
    SA_SUPERS = tile_m // 32
    SB_SUPERS = tile_n // 32
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    half_m, half_n = wmma_m_rep // 2, wmma_n_rep // 2
    n_acc = wmma_m_rep * wmma_n_rep
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    LDS_PAD_A = 16
    A_LDS_ROW = tile_k + LDS_PAD_A
    B_LDS_ROW = PACK_TK * 16
    STAGE_A = tile_m * A_LDS_ROW
    STAGE_B = (tile_n // 16) * B_LDS_ROW
    STAGE_SA = SA_SUPERS * tile_k
    STAGE_SB = SB_SUPERS * tile_k
    # Operand-major planes keep scale storage compact and B 64-KiB aligned.
    PLANAR_SA_BASE = 0
    PLANAR_A_BASE = PLANAR_SA_BASE + num_buffers * STAGE_SA
    PLANAR_SB_BASE = PLANAR_A_BASE + num_buffers * STAGE_A
    PLANAR_B_BASE = 3 * 65536
    PLANAR_END = PLANAR_B_BASE + num_buffers * STAGE_B

    ARENA_B = PLANAR_END
    check_smem_capacity(ARENA_B, str(get_hip_arch()))

    @flyc.kernel(known_block_size=[block, 1, 1])
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

        K_TILES = i32_k // tile_k
        k64 = fx.Int64(i32_k)
        lda64 = fx.Int64(i32_lda)
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

        base_ptr = fx.SharedAllocator(static=False).allocate(ARENA_B)._ptr

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

        gA = _gv(gA_base, a_off0, (tile_m, tile_k), (tile_k, 1))
        gB = _gv(gB_base, b_off0, (tile_n // 16, PACK_TK * 16), (PACK_TK * 16, 1))
        gSA = _gv(arg_scale_a, sa_off0, (SA_SUPERS, tile_k), (tile_k, 1))
        gSB = _gv(arg_scale_b, sb_off0, (SB_SUPERS, tile_k), (tile_k, 1))

        def _build_tdm_desc(owner):
            if const_expr(owner == 0):
                tensor, offset, shape, lds_stride = gA, PLANAR_A_BASE, (tile_m, tile_k), A_LDS_ROW
                stride, mask, bound, pad, early = i32_lda, a_mask, mn_oob, LDS_PAD_A, False
            elif const_expr(owner == 1):
                tensor, offset, shape, lds_stride = gB, PLANAR_B_BASE, (tile_n // 16, PACK_TK * 16), B_LDS_ROW
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

        def _prepare_tdm(slot, tile_delta):
            desc = tdm_ops.update_tensor_descriptor_2d_lds_addr(tdm_desc, tdm_base_lds + tdm_lds_step * fx.Int32(slot))
            return tdm_ops.update_tensor_descriptor_2d_addr64(
                desc,
                tdm_base_lo,
                tdm_base_hi,
                tile_delta,
            )

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
            cf.store(fx.constant_vector(0.0, T.vec(8, T.f32)))

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

        def _mma_block(wm0, wn0, act, wt, sa_k, sb_k):
            _mma_block_range(wm0, wn0, act, wt, sa_k, sb_k, 0, len(act) * len(wt))

        def _mma_block_range(wm0, wn0, act, wt, sa_k, sb_k, start, count, n_index_fast=False):
            """Issue `count` WMMAs from linear position `start` of the wm x wn block.

            The two orders are not interchangeable: they decide which operand a run of
            WMMAs holds constant, which is what the interleaved DS reads are scheduled
            against.  N-fast walks a row of B, M-fast walks a column of A.
            """
            for linear in range_constexpr(count):
                pos = start + linear
                i, j = (pos // len(wt), pos % len(wt)) if n_index_fast else (pos % len(act), pos // len(act))
                _mma_one(wm0 + i, wn0 + j, act[i], wt[j], sa_k, sb_k)

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

        def _stage_load_a(stage, wm):
            row_off = wm * 16 * A_LDS_ROW
            v = [Vec(lds_load_b128(stage_a_addr[stage], row_off + 32 * j)) for j in range_constexpr(4)]
            return v[0].shuffle(v[1], list(range(8))).shuffle(v[2].shuffle(v[3], list(range(8))), list(range(16)))

        def _stage_load_b(stage, wn):
            col_off = wn * B_LDS_ROW
            v0 = Vec(lds_load_b128(stage_b_addr[stage], col_off))
            v1 = Vec(lds_load_b128(stage_b_addr[stage], col_off + 512))
            return v0.shuffle(v1, list(range(8)))

        def _stage_load_sa(stage, sm):
            return lds_load_b32(stage_sa_addr[stage], sm * SC_WORDS * 4)[0]

        def _stage_load_sb(stage, sn):
            return lds_load_b32(stage_sb_addr[stage], sn * SC_WORDS * 4)[0]

        # Separate seed banks prevent WMMA source/address register coalescing.
        pipe_a = [[fx.make_rmem_tensor(16, fx.Int32) for _ in range_constexpr(half_m)] for _ in range_constexpr(2)]
        pipe_b = [[fx.make_rmem_tensor(8, fx.Int32) for _ in range_constexpr(half_n)] for _ in range_constexpr(2)]
        pipe_sa = [[fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(half_m // 2)] for _ in range_constexpr(2)]
        pipe_sb = [[fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(half_n // 2)] for _ in range_constexpr(2)]

        def _load_seed_a_scales(stage, bank):
            for sm in range_constexpr(half_m // 2):
                pipe_sa[bank][sm].store(Vec.from_elements([_stage_load_sa(stage, sm)], fx.Int32))

        def _load_seed_a_fragment(stage, bank, wm):
            pipe_a[bank][wm].store(_stage_load_a(stage, wm))

        def _load_seed_a(stage, bank):
            _load_seed_a_scales(stage, bank)
            for wm in range_constexpr(half_m):
                _load_seed_a_fragment(stage, bank, wm)

        def _load_seed_b_scales(stage, bank):
            for sn in range_constexpr(half_n // 2):
                pipe_sb[bank][sn].store(Vec.from_elements([_stage_load_sb(stage, sn)], fx.Int32))

        def _load_seed_b_fragment(stage, bank, wn):
            pipe_b[bank][wn].store(_stage_load_b(stage, wn))

        def _load_seed_b(stage, bank):
            _load_seed_b_scales(stage, bank)
            for wn in range_constexpr(half_n):
                _load_seed_b_fragment(stage, bank, wn)

        def _pipe_scale(regs):
            return [reg.load()[0] for reg in regs]

        def _load_current_quadrants(stage):
            a_bottom = [_rmem(16, _stage_load_a(stage, half_m + wm)) for wm in range_constexpr(half_m)]
            sa_bottom = [_stage_load_sa(stage, half_m // 2 + sm) for sm in range_constexpr(half_m // 2)]
            b_right = [_rmem(8, _stage_load_b(stage, half_n + wn)) for wn in range_constexpr(half_n)]
            sb_right = [_stage_load_sb(stage, half_n // 2 + sn) for sn in range_constexpr(half_n // 2)]
            return a_bottom, sa_bottom, b_right, sb_right

        def _compute_even_stage(
            stage,
            next_stage,
            bank,
            next_bank,
            future_slot,
            future_kt,
            fence_outstanding,
        ):
            """Even-wave stage ordered to align sibling READY arrival."""
            a_top = pipe_a[bank]
            b_left = pipe_b[bank]
            sa_top = _pipe_scale(pipe_sa[bank])
            sb_left = _pipe_scale(pipe_sb[bank])

            # q00: produce current B while admitting four older reads.
            rocdl.sched_barrier(0)
            rocdl.s_wait_dscnt(4)
            rocdl.sched_barrier(0)
            _mma_block_range(0, 0, a_top, b_left, sa_top, sb_left, 0, 1)
            sb_right = [_stage_load_sb(stage, half_n // 2 + sn) for sn in range_constexpr(half_n // 2)]
            _mma_block_range(0, 0, a_top, b_left, sa_top, sb_left, 1, 1)
            b_right_0 = _rmem(8, _stage_load_b(stage, half_n + 0))
            b_right_1 = _rmem(8, _stage_load_b(stage, half_n + 1))
            _mma_block_range(0, 0, a_top, b_left, sa_top, sb_left, 2, 1)
            b_right_2 = _rmem(8, _stage_load_b(stage, half_n + 2))
            b_right_3 = _rmem(8, _stage_load_b(stage, half_n + 3))
            b_right = [b_right_0, b_right_1, b_right_2, b_right_3]
            sb_k = sb_left + sb_right

            rocdl.sched_mfma(1)
            rocdl.sched_dsrd(2)
            for _ in range_constexpr(2):
                rocdl.sched_mfma(1)
                rocdl.sched_dsrd(4)
            rocdl.sched_barrier(0)
            _mma_block_range(0, 0, a_top, b_left, sa_top, sb_left, 3, 5)
            rocdl.sched_barrier(0)
            rocdl.s_wait_dscnt(10)
            rocdl.sched_barrier(0)
            _mma_block_range(0, 0, a_top, b_left, sa_top, sb_left, 8, 8)

            # q01: produce current A and signal READY.
            rocdl.sched_barrier(0)
            rocdl.s_wait_dscnt(4)
            rocdl.sched_barrier(0)
            _mma_block_range(0, half_n, a_top, b_right, sa_top, sb_k, 0, 1)
            sa_bottom = [_stage_load_sa(stage, half_m // 2 + sm) for sm in range_constexpr(half_m // 2)]
            _mma_block_range(0, half_n, a_top, b_right, sa_top, sb_k, 1, 1)
            a_bottom_0 = _rmem(16, _stage_load_a(stage, half_m + 0))
            _mma_block_range(0, half_n, a_top, b_right, sa_top, sb_k, 2, 1)
            a_bottom_1 = _rmem(16, _stage_load_a(stage, half_m + 1))
            _mma_block_range(0, half_n, a_top, b_right, sa_top, sb_k, 3, 1)
            a_bottom_2 = _rmem(16, _stage_load_a(stage, half_m + 2))
            _mma_block_range(0, half_n, a_top, b_right, sa_top, sb_k, 4, 1)
            a_bottom_3 = _rmem(16, _stage_load_a(stage, half_m + 3))
            a_bottom = [a_bottom_0, a_bottom_1, a_bottom_2, a_bottom_3]
            sa_k = sa_top + sa_bottom

            rocdl.sched_mfma(1)
            rocdl.sched_dsrd(2)
            for _ in range_constexpr(4):
                rocdl.sched_mfma(1)
                rocdl.sched_dsrd(4)
            rocdl.sched_barrier(0)
            _mma_block_range(0, half_n, a_top, b_right, sa_k, sb_k, 5, 3)
            rocdl.sched_barrier(0)
            rocdl.s_wait_dscnt(17)
            rocdl.sched_barrier(0)
            _mma_block_range(0, half_n, a_top, b_right, sa_k, sb_k, 8, 5)
            rocdl.sched_barrier(0)
            pipeline_fence_signal(outstanding=fence_outstanding, use_cluster=False)
            rocdl.sched_barrier(0)
            _mma_block_range(0, half_n, a_top, b_right, sa_k, sb_k, 13, 3)
            rocdl.sched_barrier(0)
            pipeline_fence_wait(use_cluster=False)
            rocdl.sched_barrier(0)
            rocdl.s_wait_dscnt(0)
            rocdl.sched_barrier(0)

            # q10: produce next A and issue TDM at W3.
            prepared_refill_desc = _prepare_tdm(future_slot, future_kt)
            _mma_block_range(half_m, 0, a_bottom, b_left, sa_k, sb_left, 0, 1)
            _load_seed_a_scales(next_stage, next_bank)
            for pos in range_constexpr(1, 3):
                _mma_block_range(half_m, 0, a_bottom, b_left, sa_k, sb_left, pos, 1)
                _load_seed_a_fragment(next_stage, next_bank, pos - 1)
            rocdl.sched_mfma(1)
            rocdl.sched_dsrd(2)
            for _ in range_constexpr(2):
                rocdl.sched_mfma(1)
                rocdl.sched_dsrd(4)
            rocdl.sched_barrier(0)
            tdm_ops.tensor_load_2d(prepared_refill_desc)
            rocdl.sched_barrier(0)
            for pos in range_constexpr(3, 5):
                _mma_block_range(half_m, 0, a_bottom, b_left, sa_k, sb_left, pos, 1)
                _load_seed_a_fragment(next_stage, next_bank, pos - 1)
            for _ in range_constexpr(2):
                rocdl.sched_mfma(1)
                rocdl.sched_dsrd(4)
            rocdl.sched_barrier(0)
            _mma_block_range(half_m, 0, a_bottom, b_left, sa_k, sb_left, 5, 11)

            # q11: leave the next B tail in flight.
            rocdl.sched_barrier(0)
            _mma_block_range(
                half_m,
                half_n,
                a_bottom,
                b_right,
                sa_k,
                sb_k,
                0,
                1,
            )
            _load_seed_b_scales(next_stage, next_bank)
            _mma_block_range(
                half_m,
                half_n,
                a_bottom,
                b_right,
                sa_k,
                sb_k,
                1,
                1,
            )
            _load_seed_b_fragment(next_stage, next_bank, 0)
            _load_seed_b_fragment(next_stage, next_bank, 1)
            _mma_block_range(
                half_m,
                half_n,
                a_bottom,
                b_right,
                sa_k,
                sb_k,
                2,
                1,
            )
            _load_seed_b_fragment(next_stage, next_bank, 2)
            _load_seed_b_fragment(next_stage, next_bank, 3)
            rocdl.sched_mfma(1)
            rocdl.sched_dsrd(2)
            for _ in range_constexpr(2):
                rocdl.sched_mfma(1)
                rocdl.sched_dsrd(4)
            rocdl.sched_barrier(0)
            _mma_block_range(
                half_m,
                half_n,
                a_bottom,
                b_right,
                sa_k,
                sb_k,
                3,
                13,
            )

        def _compute_stage(
            stage,
            next_stage,
            bank,
            next_bank,
            future_slot,
            future_kt,
            fence_outstanding,
            has_next,
            steady=False,
        ):
            """Odd-wave steady stage or shared drain stage."""

            a_top = pipe_a[bank]
            b_left = pipe_b[bank]
            sa_top = _pipe_scale(pipe_sa[bank])
            sb_left = _pipe_scale(pipe_sb[bank])

            if const_expr(steady):
                # q00: produce current A while retiring the prior A tail.
                rocdl.sched_barrier(0)
                rocdl.s_wait_dscnt(8)
                rocdl.sched_barrier(0)
                _mma_block_range(0, 0, a_top, b_left, sa_top, sb_left, 0, 1, True)
                sa_bottom = [_stage_load_sa(stage, half_m // 2 + sm) for sm in range_constexpr(half_m // 2)]
                _mma_block_range(0, 0, a_top, b_left, sa_top, sb_left, 1, 1, True)
                a_bottom_0 = _rmem(16, _stage_load_a(stage, half_m + 0))
                _mma_block_range(0, 0, a_top, b_left, sa_top, sb_left, 2, 1, True)
                a_bottom_1 = _rmem(16, _stage_load_a(stage, half_m + 1))
                _mma_block_range(0, 0, a_top, b_left, sa_top, sb_left, 3, 1, True)
                a_bottom_2 = _rmem(16, _stage_load_a(stage, half_m + 2))
                _mma_block_range(0, 0, a_top, b_left, sa_top, sb_left, 4, 1, True)
                a_bottom_3 = _rmem(16, _stage_load_a(stage, half_m + 3))
                a_bottom = [a_bottom_0, a_bottom_1, a_bottom_2, a_bottom_3]

                rocdl.sched_mfma(1)
                rocdl.sched_dsrd(2)
                for _ in range_constexpr(4):
                    rocdl.sched_mfma(1)
                    rocdl.sched_dsrd(4)
                rocdl.sched_barrier(0)
                _mma_block_range(0, 0, a_top, b_left, sa_top, sb_left, 5, 3, True)
                rocdl.sched_barrier(0)
                rocdl.s_wait_dscnt(17)
                rocdl.sched_barrier(0)
                _mma_block_range(0, 0, a_top, b_left, sa_top, sb_left, 8, 8, True)
                rocdl.sched_barrier(0)
                sb_k = sb_left
            else:
                # K512 reaches this drain without a steady iteration.
                rocdl.s_wait_dscnt(4)
                _mma_block(0, 0, a_top, b_left, sa_top, sb_left)
                a_bottom, sa_bottom, b_right, sb_right = _load_current_quadrants(stage)
                sb_k = sb_left + sb_right

            sa_k = sa_top + sa_bottom

            if const_expr(steady):
                # q10: produce current B and retire the older A tail.
                rocdl.s_wait_dscnt(8)
                rocdl.sched_barrier(0)
                _mma_block_range(half_m, 0, a_bottom, b_left, sa_k, sb_left, 0, 1, True)
                sb_right = [_stage_load_sb(stage, half_n // 2 + sn) for sn in range_constexpr(half_n // 2)]
                _mma_block_range(half_m, 0, a_bottom, b_left, sa_k, sb_left, 1, 1, True)
                b_right_0 = _rmem(8, _stage_load_b(stage, half_n + 0))
                b_right_1 = _rmem(8, _stage_load_b(stage, half_n + 1))
                _mma_block_range(half_m, 0, a_bottom, b_left, sa_k, sb_left, 2, 1, True)
                b_right_2 = _rmem(8, _stage_load_b(stage, half_n + 2))
                b_right_3 = _rmem(8, _stage_load_b(stage, half_n + 3))
                b_right = [b_right_0, b_right_1, b_right_2, b_right_3]
                sb_k = sb_left + sb_right

                rocdl.sched_mfma(1)
                rocdl.sched_dsrd(2)
                for _ in range_constexpr(2):
                    rocdl.sched_mfma(1)
                    rocdl.sched_dsrd(4)
                rocdl.sched_barrier(0)
                _mma_block_range(half_m, 0, a_bottom, b_left, sa_k, sb_k, 3, 5, True)
                rocdl.sched_barrier(0)
                rocdl.s_wait_dscnt(10)
                rocdl.sched_barrier(0)
                _mma_block_range(half_m, 0, a_bottom, b_left, sa_k, sb_k, 8, 5, True)

                # Signal READY before W13 and wait after W15.
                pipeline_fence_signal(outstanding=fence_outstanding, use_cluster=False)
                rocdl.sched_barrier(0)
                _mma_block_range(half_m, 0, a_bottom, b_left, sa_k, sb_k, 13, 3, True)
                rocdl.sched_barrier(0)
                pipeline_fence_wait(use_cluster=False)
                rocdl.s_wait_dscnt(0)
                rocdl.sched_barrier(0)
                rocdl.sched_barrier(0)

                # q01: produce next B and issue TDM after five WMMAs.
                prepared_refill_desc = _prepare_tdm(future_slot, future_kt)
                _mma_block_range(0, half_n, a_top, b_right, sa_k, sb_k, 0, 1, True)
                _load_seed_b_scales(next_stage, next_bank)
                _mma_block_range(0, half_n, a_top, b_right, sa_k, sb_k, 1, 1, True)
                _load_seed_b_fragment(next_stage, next_bank, 0)
                _load_seed_b_fragment(next_stage, next_bank, 1)
                _mma_block_range(0, half_n, a_top, b_right, sa_k, sb_k, 2, 1, True)
                _load_seed_b_fragment(next_stage, next_bank, 2)
                _load_seed_b_fragment(next_stage, next_bank, 3)
                _mma_block_range(0, half_n, a_top, b_right, sa_k, sb_k, 3, 2, True)
                rocdl.sched_mfma(1)
                rocdl.sched_dsrd(2)
                for _ in range_constexpr(2):
                    rocdl.sched_mfma(1)
                    rocdl.sched_dsrd(4)
                rocdl.sched_mfma(2)
                rocdl.sched_barrier(0)
                tdm_ops.tensor_load_2d(prepared_refill_desc)
                _mma_block_range(0, half_n, a_top, b_right, sa_k, sb_k, 5, 2, True)
                _mma_block_range(0, half_n, a_top, b_right, sa_k, sb_k, 7, 9, True)

                # q11: leave the next A tail in flight.
                _mma_block_range(
                    half_m,
                    half_n,
                    a_bottom,
                    b_right,
                    sa_k,
                    sb_k,
                    0,
                    1,
                    True,
                )
                _load_seed_a_scales(next_stage, next_bank)
                for pos in range_constexpr(1, 5):
                    _mma_block_range(
                        half_m,
                        half_n,
                        a_bottom,
                        b_right,
                        sa_k,
                        sb_k,
                        pos,
                        1,
                        True,
                    )
                    _load_seed_a_fragment(next_stage, next_bank, pos - 1)
                rocdl.sched_mfma(1)
                rocdl.sched_dsrd(1)
                for _ in range_constexpr(4):
                    rocdl.sched_mfma(1)
                    rocdl.sched_dsrd(4)
                rocdl.sched_barrier(0)
                _mma_block_range(
                    half_m,
                    half_n,
                    a_bottom,
                    b_right,
                    sa_k,
                    sb_k,
                    5,
                    11,
                    True,
                )
                return

            if const_expr(has_next):
                rocdl.s_wait_dscnt(9)
                _mma_block_range(half_m, 0, a_bottom, b_left, sa_k, sb_k, 0, 13)
                pipeline_fence_signal(outstanding=fence_outstanding, use_cluster=False)
                rocdl.sched_barrier(0)
                _mma_block_range(half_m, 0, a_bottom, b_left, sa_k, sb_k, 13, 3)
                pipeline_fence_wait(use_cluster=False)
                rocdl.s_wait_dscnt(0)
                rocdl.sched_barrier(0)
                _load_seed_a(next_stage, next_bank)
                _load_seed_b(next_stage, next_bank)
            else:
                rocdl.s_wait_dscnt(0)
                _mma_block(half_m, 0, a_bottom, b_left, sa_k, sb_k)

            _mma_block(0, half_n, a_top, b_right, sa_k, sb_k)
            _mma_block(half_m, half_n, a_bottom, b_right, sa_k, sb_k)

        # Keep all four slots in flight and statically expand one revolution.
        for i in range_constexpr(num_buffers):
            tdm_ops.tensor_load_2d(_prepare_tdm(i, fx.Int32(i) * tdm_global_step))
        pipeline_fence(outstanding=num_buffers - 1, use_cluster=False)
        _load_seed_a(0, 0)
        _load_seed_b(0, 0)
        rocdl.s_wait_dscnt(0)

        n_full = (K_TILES + num_buffers - 1) // num_buffers - 1
        drain_n = K_TILES - n_full * num_buffers  # 1..num_buffers
        last_delta = (K_TILES - 1) * tdm_global_step
        stage_delta = [fx.Int32(s + num_buffers) * tdm_global_step for s in range_constexpr(num_buffers)]

        def _run_steady(owner_parity):
            for rev in range(n_full):
                rev_delta = (rev * num_buffers) * tdm_global_step
                for s in range_constexpr(num_buffers):
                    delta = rev_delta + stage_delta[s]
                    delta = (delta < last_delta).select(delta, last_delta)
                    args = (s, (s + 1) % num_buffers, s % 2, (s + 1) % 2, s, delta, num_buffers - 2)
                    if const_expr(owner_parity == 0):
                        _compute_even_stage(*args)
                    else:
                        _compute_stage(*args, True, True)
                if rev % cluster_sync_revs == cluster_sync_revs - 1:
                    cluster.cluster_barrier()

        wave_parity = fx.Int32(
            llvm_dialect.inline_asm(
                T.i32,
                [as_ir_value(rocdl.wave_id())],
                "s_and_b32 $0, $1, 1",
                "=s,s,~{scc}",
                has_side_effects=True,
            )
        )
        if wave_parity == 0:
            _run_steady(0)
        else:
            _run_steady(1)

        # Retire the last steady producer before the shared drain.
        rocdl.s_wait_dscnt(0)
        rocdl.sched_barrier(0)
        for j in range_constexpr(num_buffers):
            if j < drain_n:
                _compute_stage(
                    j, (j + 1) % num_buffers, j % 2, (j + 1) % 2, j, None, num_buffers - 2 - j, j < num_buffers - 1
                )
        accs = [c_frags[idx].load() for idx in range_constexpr(n_acc)]

        pipeline_fence(outstanding=0, use_cluster=True)
        for wm in range_constexpr(wmma_m_rep):
            row_rel = wmb + wm * 16 + lane16
            for wn in range_constexpr(wmma_n_rep):
                col_rel = wnb + wn * 16 + kgrp * 8
                h = accs[wm * wmma_n_rep + wn].to(oc)
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
    cluster_arg = (cluster_m, cluster_n, 1)
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
    ).launch(grid=grid_arg, block=(block, 1, 1), stream=stream, cluster=cluster_arg)


launch_gemm_a8w4_256x256.compile_hints["llvm_options"] = {
    "amdgpu-expert-scheduling-mode": True,
}
