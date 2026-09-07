# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""A8W4 (FP8 activation x MXFP4 E2M1 weight) 32x4-preshuffled MXscale GEMM for gfx1250."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster, tdm_ops
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import check_smem_capacity
from kernels.common.gfx1250_cluster import compute_mcast_masks

from .gemm_common_gfx1250 import (
    make_lds_copy_ops,
    pipeline_fence,
    workgroup_barrier,
)


@flyc.jit
def launch_gemm_a8w4_mxscale(
    arg_c: fx.Tensor,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Tensor,
    arg_scale_b: fx.Tensor,
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
    """A: [M, K] uint8 (FP8 E4M3). B: [N, K//2] uint8, 16x16-preshuffled FP4 (E2M1)."""

    use_cluster = cluster_m > 1 or cluster_n > 1
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    WAVE = 32
    K_WS = tile_k // WMMA_K
    PACK_TK = tile_k // 2  # B row bytes per K-tile (FP4 packed 2/byte)
    SC_WORDS = tile_k // 4  # scale i32 words per super-row per K-tile
    SA_SUPERS = tile_m // 32
    SB_SUPERS = tile_n // 32
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_acc = wmma_m_rep * wmma_n_rep
    num_waves = m_warp * n_warp
    block = num_waves * WAVE

    LDS_PAD_A = 16
    A_LDS_ROW = tile_k + LDS_PAD_A
    B_LDS_ROW = PACK_TK * 16
    STAGE_A = ((tile_m * A_LDS_ROW + 15) // 16) * 16
    STAGE_B = (((tile_n // 16) * B_LDS_ROW + 15) // 16) * 16
    STAGE_SA = ((SA_SUPERS * tile_k + 15) // 16) * 16
    STAGE_SB = ((SB_SUPERS * tile_k + 15) // 16) * 16
    SA_OFF = STAGE_A + STAGE_B
    SB_OFF = STAGE_A + STAGE_B + STAGE_SA
    PITCH = ((STAGE_A + STAGE_B + STAGE_SA + STAGE_SB + 1023) // 1024) * 1024

    C_STORE_B = ((tile_m * tile_n * 2 + 127) // 128) * 128
    ARENA_B = max(num_buffers * PITCH, C_STORE_B)
    check_smem_capacity(ARENA_B, str(get_hip_arch()))

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel_gemm_a8w4_mxscale(
        arg_c: fx.Tensor,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
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
        bid_x, bid_y, _ = fx.block_idx
        wave = rocdl.readfirstlane(T.i32, tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        wave_m = wave // n_warp
        wave_n = wave % n_warp
        if const_expr(use_cluster):
            local_x, local_y = cluster.compute_cluster_position()
            a_mask, b_mask = compute_mcast_masks(local_x, local_y, cluster_m, cluster_n)
        else:
            a_mask, b_mask = 0, 0
        blk_m = bid_x * tile_m
        blk_n = bid_y * tile_n
        blk_m64 = fx.Int64(blk_m)
        blk_n64 = fx.Int64(blk_n)

        arena = fx.SharedAllocator(static=False)
        arena.allocate(ARENA_B)
        base_ptr = arena.base_ptr

        def _bidx(p):
            return fx.Int64(fx.ptrtoint(p))

        def _buf_ptr(s):
            return fx.add_offset(base_ptr, s * PITCH)

        def _gv(base, off, shape, stride):
            return fx.Tensor(fx.make_view(fx.add_offset(base, off), fx.make_layout(shape, stride)))

        def _lv(ptr, shape, stride):
            return fx.Tensor(fx.make_view(ptr, fx.make_layout(shape, stride)))

        gA_base = fx.recast_iter(fx.Int8, arg_a)
        gB_base = fx.recast_iter(fx.Int8, arg_b)
        gSA_base = fx.recast_iter(fx.Int8, fx.get_iter(arg_scale_a))
        gSB_base = fx.recast_iter(fx.Int8, fx.get_iter(arg_scale_b))

        W_A, W_B, W_SA, W_SB = 0, 1, 2 % num_waves, 3 % num_waves
        WARP1 = fx.make_layout(1, 1)  # single-warp TDM partition for the per-wave loads
        B_INNER = (i32_k // 2) * 16  # full packed-K bytes per N-group (== Kp16 as a shape)

        # New tiled-TDM API: build each atom over the FULL global tensor; the coordinate rest carries
        # K-tiles (replaces imm_offset), the LDS view carries an independent PIPE rest (num_buffers).
        # OOB comes from the tensor extent + boundary_check; A's LDS-row slack encodes the pad.
        gA_full = _gv(gA_base, fx.Int64(0), (i32_m, i32_k), (lda64, 1))
        atomA, coordA = fx.rocdl.cdna5.make_tiled_tdm_atom(
            fx.rocdl.TensorLoad(), gA_full, fx.make_layout((tile_m, tile_k), (A_LDS_ROW, 1)), (tile_m, tile_k), num_warps=1
        )
        atomA = fx.atom_set_value(atomA, "workgroup_mask", a_mask)
        blkA = fx.zipped_divide(coordA, (tile_m, tile_k))[None, (bid_x, None)]
        sA_lds = fx.Tensor(
            fx.make_view(
                fx.recast_iter(fx.Int8, base_ptr),
                fx.make_layout(((tile_m, tile_k), num_buffers), ((A_LDS_ROW, 1), PITCH)),
            )
        )
        tAsA, tAgA = fx.rocdl.cdna5.tdm_partition(atomA, 0, WARP1, sA_lds, blkA)

        gB_full = _gv(gB_base, fx.Int64(0), (i32_n // 16, B_INNER), (Kp16, 1))
        atomB, coordB = fx.rocdl.cdna5.make_tiled_tdm_atom(
            fx.rocdl.TensorLoad(),
            gB_full,
            fx.make_layout((tile_n // 16, PACK_TK * 16), (B_LDS_ROW, 1)),
            (tile_n // 16, PACK_TK * 16),
            num_warps=1,
        )
        atomB = fx.atom_set_value(atomB, "workgroup_mask", b_mask)
        blkB = fx.zipped_divide(coordB, (tile_n // 16, PACK_TK * 16))[None, (bid_y, None)]
        sB_lds = fx.Tensor(
            fx.make_view(
                fx.recast_iter(fx.Int8, fx.add_offset(base_ptr, STAGE_A)),
                fx.make_layout(((tile_n // 16, PACK_TK * 16), num_buffers), ((B_LDS_ROW, 1), PITCH)),
            )
        )
        tBsB, tBgB = fx.rocdl.cdna5.tdm_partition(atomB, 0, WARP1, sB_lds, blkB)

        # SA/SB: (super, K) with K inner-contiguous; select the M/N super-tile, K advances via the rest.
        gSA_full = _gv(gSA_base, fx.Int64(0), ((i32_m + 31) // 32, i32_k), (k64, 1))
        atomSA, coordSA = fx.rocdl.cdna5.make_tiled_tdm_atom(
            fx.rocdl.TensorLoad(), gSA_full, fx.make_layout((SA_SUPERS, tile_k), (tile_k, 1)), (SA_SUPERS, tile_k), num_warps=1
        )
        atomSA = fx.atom_set_value(atomSA, "workgroup_mask", a_mask)
        blkSA = fx.zipped_divide(coordSA, (SA_SUPERS, tile_k))[None, (bid_x, None)]
        sSA_lds = fx.Tensor(
            fx.make_view(
                fx.recast_iter(fx.Int8, fx.add_offset(base_ptr, SA_OFF)),
                fx.make_layout(((SA_SUPERS, tile_k), num_buffers), ((tile_k, 1), PITCH)),
            )
        )
        tSAsSA, tSAgSA = fx.rocdl.cdna5.tdm_partition(atomSA, 0, WARP1, sSA_lds, blkSA)

        gSB_full = _gv(gSB_base, fx.Int64(0), ((i32_n + 31) // 32, i32_k), (k64, 1))
        atomSB, coordSB = fx.rocdl.cdna5.make_tiled_tdm_atom(
            fx.rocdl.TensorLoad(), gSB_full, fx.make_layout((SB_SUPERS, tile_k), (tile_k, 1)), (SB_SUPERS, tile_k), num_warps=1
        )
        atomSB = fx.atom_set_value(atomSB, "workgroup_mask", b_mask)
        blkSB = fx.zipped_divide(coordSB, (SB_SUPERS, tile_k))[None, (bid_y, None)]
        sSB_lds = fx.Tensor(
            fx.make_view(
                fx.recast_iter(fx.Int8, fx.add_offset(base_ptr, SB_OFF)),
                fx.make_layout(((SB_SUPERS, tile_k), num_buffers), ((tile_k, 1), PITCH)),
            )
        )
        tSBsSB, tSBgSB = fx.rocdl.cdna5.tdm_partition(atomSB, 0, WARP1, sSB_lds, blkSB)

        def issue(s, kt):
            if wave == W_A:
                fx.copy(atomA, tAgA[None, kt], tAsA[None, s])
            if wave == W_B:
                fx.copy(atomB, tBgB[None, kt], tBsB[None, s])
            if wave == W_SA:
                fx.copy(atomSA, tSAgSA[None, kt], tSAsSA[None, s])
            if wave == W_SB:
                fx.copy(atomSB, tSBgSB[None, kt], tSBsSB[None, s])

        wmb = wave_m * warp_tile_m
        wnb = wave_n * warp_tile_n

        lds_load_b32, _ = make_lds_copy_ops(32)
        lds_load_b128, _ = make_lds_copy_ops(128)

        def load_a(buf, wm, ks):
            row = wmb + wm * 16 + lane16
            b0 = fx.Int64(row * A_LDS_ROW + ks * WMMA_K + kgrp * 16)
            v = [Vec(lds_load_b128(buf, b0 + 32 * j)) for j in range_constexpr(4)]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            return v01.shuffle(v23, list(range(16)))

        def load_b(buf, wn, ks):
            nbl = wnb // 16 + wn
            b0 = fx.Int64(STAGE_A + nbl * B_LDS_ROW + ks * 1024 + kgrp * 256 + lane16 * 16)
            v0 = Vec(lds_load_b128(buf, b0))
            v1 = Vec(lds_load_b128(buf, b0 + 512))
            return v0.shuffle(v1, list(range(8)))

        def load_sa(buf, wm, ks):
            row = wmb + wm * 16 + lane16
            word = (row // 32) * SC_WORDS + ks * 32 + (row % 32)
            return lds_load_b32(buf, fx.Int64(SA_OFF + word * 4))[0]

        def load_sb(buf, wn, ks):
            col = wnb + wn * 16 + lane16
            word = (col // 32) * SC_WORDS + ks * 32 + (col % 32)
            return lds_load_b32(buf, fx.Int64(SB_OFF + word * 4))[0]

        wmma_atom = fx.make_mma_atom(
            fx.rocdl.WMMAScale(WMMA_M, WMMA_N, WMMA_K, fx.Float4E2M1FN, fx.Float8E4M3FN, fx.Float32)
        )
        c_frags = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(n_acc)]
        for cf in c_frags:
            cf.store(Vec.filled(8, 0.0, fx.Float32))

        def _rmem(n, v):
            t = fx.make_rmem_tensor(n, fx.Int32)
            t.store(v)
            return t

        front_wm = (wmma_m_rep + 1) // 2
        _FRONT = list(range(front_wm))
        _BACK = list(range(front_wm, wmma_m_rep))

        def _mma_rows(wm_list, act, wt, sa_k, sb_k):
            for i in range_constexpr(len(wm_list)):
                wm = wm_list[i]
                for wn_raw in range_constexpr(wmma_n_rep):
                    wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                    idx = wm * wmma_n_rep + wn
                    fx.gemm(wmma_atom, c_frags[idx], wt[wn], act[i], c_frags[idx], scale_a=sb_k[wn], scale_b=sa_k[wm])

        DS_A, DS_B = 4, 2
        _BS_DS = wmma_n_rep * DS_B + wmma_n_rep + wmma_m_rep

        def _load_b_scales(buf, ks):
            wt = [_rmem(8, load_b(buf, wn, ks)) for wn in range_constexpr(wmma_n_rep)]
            sb_k = [load_sb(buf, wn, ks) for wn in range_constexpr(wmma_n_rep)]
            sa_k = [load_sa(buf, wm, ks) for wm in range_constexpr(wmma_m_rep)]
            return wt, sb_k, sa_k

        def _kstep(buf, ks, wt, sb_k, sa_k, nxt_ks, prefetch_kt=None):
            act_f = [_rmem(16, load_a(buf, wm, ks)) for wm in _FRONT]
            if const_expr(len(_BACK) > 0):
                act_b = [_rmem(16, load_a(buf, wm, ks)) for wm in _BACK]
                rocdl.s_wait_dscnt(len(_BACK) * DS_A)
            else:
                rocdl.s_wait_dscnt(0)
            _mma_rows(_FRONT, act_f, wt, sa_k, sb_k)
            if const_expr(prefetch_kt is not None):
                rocdl.sched_barrier(0)
                issue(prefetch_kt % num_buffers, prefetch_kt)
                rocdl.sched_barrier(0)
            if const_expr(len(_BACK) > 0):
                rocdl.s_wait_dscnt(0)
                _mma_rows(_BACK, act_b, wt, sa_k, sb_k)
            return _load_b_scales(buf, nxt_ks) if const_expr(nxt_ks is not None) else None

        def compute_ktile(buf, prefetch_kt):
            prev = _load_b_scales(buf, 0)
            for ks in range_constexpr(K_WS):
                nxt_ks = ks + 1 if const_expr(ks + 1 < K_WS) else None
                pk = prefetch_kt if const_expr(ks == 0) else None
                prev = _kstep(buf, ks, prev[0], prev[1], prev[2], nxt_ks, prefetch_kt=pk)
            _fr, _bk = front_wm * wmma_n_rep, len(_BACK) * wmma_n_rep
            for _ks in range_constexpr(K_WS):
                rocdl.sched_dsrd((_BS_DS if _ks == 0 else 0) + front_wm * DS_A)
                rocdl.sched_mfma(_fr)
                rocdl.sched_dsrd(len(_BACK) * DS_A)
                rocdl.sched_mfma(_bk)
                if const_expr(_ks < K_WS - 1):
                    rocdl.sched_dsrd(_BS_DS)
            rocdl.sched_barrier(0)

        if const_expr(use_cluster):
            cluster.cluster_barrier()
        for i in range_constexpr(num_buffers - 1):
            issue(i, i)
        n_steady = K_TILES - (num_buffers - 1)
        for kt in range(n_steady):
            s = kt % num_buffers
            buf = _bidx(_buf_ptr(s))
            pipeline_fence(outstanding=(num_buffers - 2), use_cluster=False)
            compute_ktile(buf, kt + (num_buffers - 1))
            if const_expr(use_cluster) and kt % num_buffers == num_buffers - 1:
                cluster.cluster_barrier()
        for j in range_constexpr(num_buffers - 1):
            kt = n_steady + j
            s = kt % num_buffers
            buf = _bidx(_buf_ptr(s))
            pipeline_fence(outstanding=(num_buffers - 2 - j), use_cluster=False)
            compute_ktile(buf, None)

        accs = [c_frags[idx].load().ir_value() for idx in range_constexpr(n_acc)]

        pipeline_fence(outstanding=0, use_cluster=use_cluster)
        oc = fx.Float16 if out_is_f16 else fx.BFloat16
        for wm in range_constexpr(wmma_m_rep):
            row_rel = wmb + wm * 16 + lane16
            for wn in range_constexpr(wmma_n_rep):
                col_rel = wnb + wn * 16 + kgrp * 8
                h = Vec(accs[wm * wmma_n_rep + wn]).to(oc)
                fx.ptr_store(h.bitcast(fx.Int8), base_ptr + (row_rel * tile_n + col_rel) * 2)
        workgroup_barrier(use_cluster=False)
        # C store: new tiled-TDM store over the FULL (M, N) tensor, split across all waves. Keep the
        # N-tile grid as the coordinate rest (mode-0 stays the nested box); select bid_x/bid_y.
        gC_full = _gv(fx.recast_iter(oc, fx.get_iter(arg_c)), fx.Int64(0), (i32_m, i32_n), (ldc64, 1))
        atomC, coordC = fx.rocdl.cdna5.make_tiled_tdm_atom(
            fx.rocdl.TensorStore(), gC_full, fx.make_layout((tile_m, tile_n), (tile_n, 1)), (tile_m, tile_n), num_warps=num_waves
        )
        blkC = fx.zipped_divide(coordC, (tile_m, tile_n))[None, (bid_x, None)]
        sC_lds = fx.Tensor(
            fx.make_view(
                fx.recast_iter(oc, base_ptr),
                fx.make_layout(((tile_m, tile_n), 1), ((tile_n, 1), tile_m * tile_n)),
            )
        )
        tCsC, tCgC = fx.rocdl.cdna5.tdm_partition(atomC, wave, fx.make_layout(num_waves, 1), sC_lds, blkC)
        fx.copy(atomC, tCsC[None, 0, 0], tCgC[None, 0, bid_y])
        tdm_ops.tensor_wait(0)

    gx = (i32_m + (tile_m - 1)) // tile_m
    gy = (N + (tile_n - 1)) // tile_n
    if use_cluster:
        gx = ((gx + (cluster_m - 1)) // cluster_m) * cluster_m
    cluster_arg = (cluster_m, cluster_n, 1) if use_cluster else None
    kernel_gemm_a8w4_mxscale(
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
        value_attrs={"rocdl.cluster_dims": f"{cluster_m},{cluster_n},1" if use_cluster else None},
    ).launch(grid=(gx, gy, 1), block=(block, 1, 1), stream=stream, cluster=cluster_arg)


launch_gemm_a8w4_mxscale.compile_hints["llvm_options"] = {
    "amdgpu-expert-scheduling-mode": True,
}
