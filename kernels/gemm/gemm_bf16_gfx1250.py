# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Dense BF16/FP16 GEMM (C = A x B^T) for gfx1250: TDM staging + 16x16x32 WMMA."""

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
def launch_gemm_bf16(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    i32_m: fx.Int32,
    stream: fx.Stream,
    N: fx.Int32,
    K: fx.Int32,
    i32_lda: fx.Int32,
    i32_ldb: fx.Int32,
    i32_ldc: fx.Int32,
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    num_buffers: Constexpr[int],
    is_f16: Constexpr[int] = 0,
    cluster_m: Constexpr[int] = 1,
    cluster_n: Constexpr[int] = 1,
):
    """Requires N % tile_n == 0 and K % tile_k == 0; M is clamped per tile."""
    WMMA_M = WMMA_N = 16
    WMMA_K = 32
    WAVE = 32
    EB = 2  # bytes per element
    KB = tile_k * EB  # K-tile row bytes
    KSB = WMMA_K * EB  # bytes consumed by one WMMA K-step
    K_WS = tile_k // WMMA_K
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_acc = wmma_m_rep * wmma_n_rep
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    use_cluster = cluster_m > 1 or cluster_n > 1

    if tile_k % WMMA_K or warp_tile_m % WMMA_M or warp_tile_n % WMMA_N:
        raise ValueError(f"tile ({tile_m},{tile_n},{tile_k}) / warps ({m_warp},{n_warp}) not WMMA-aligned")
    if K_WS < 2:
        raise ValueError(f"tile_k={tile_k} needs at least 2 WMMA K-steps to hide the LDS reads")

    LDS_PAD = 32  # 32-byte pad keeps rows 32-byte aligned for ds_read_b256
    LDS_ROW = KB + LDS_PAD
    STAGE_A = ((tile_m * LDS_ROW + 15) // 16) * 16
    STAGE_B = ((tile_n * LDS_ROW + 15) // 16) * 16
    PITCH = ((STAGE_A + STAGE_B + 1023) // 1024) * 1024
    elem_cls = fx.Float16 if is_f16 else fx.BFloat16
    C_STORE_B = ((tile_m * tile_n * EB + 127) // 128) * 128
    ARENA_B = max(num_buffers * PITCH, C_STORE_B)
    check_smem_capacity(ARENA_B, str(get_hip_arch()))

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel_gemm_bf16(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        i32_m: fx.Int32,
        i32_k: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldb: fx.Int32,
        i32_ldc: fx.Int32,
    ):
        K_TILES = i32_k // tile_k
        lda_b = fx.Int64(i32_lda) * EB  # global row strides in bytes
        ldb_b = fx.Int64(i32_ldb) * EB
        ldc64 = fx.Int64(i32_ldc)

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
        mn_oob = i32_m - blk_m  # valid M rows (A / C)

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

        def _tdm(gt, outer, stride, mask):  # single-warp row-tile atom, dim0 clamped
            atom = fx.rocdl.make_tdm_atom(
                gt,
                [outer, None],
                strides=[stride, None],
                num_warps=1,
                pad_interval=KB,
                pad_amount=LDS_PAD,
                early_timeout=True,
            )
            return fx.atom_set_value(atom, "workgroup_mask", mask)

        gA_base = fx.recast_iter(fx.Int8, arg_a)
        gB_base = fx.recast_iter(fx.Int8, arg_b)
        gC_base = fx.recast_iter(fx.PointerType.get(elem_cls.ir_type, arg_c.address_space), arg_c)

        W_A, W_B = 0, 1 % num_waves
        gA = _gv(gA_base, fx.Int64(blk_m) * lda_b, (tile_m, KB), (KB, 1))
        atomA = _tdm(gA, mn_oob, lda_b, a_mask)
        gB = _gv(gB_base, fx.Int64(blk_n) * ldb_b, (tile_n, KB), (KB, 1))
        atomB = _tdm(gB, None, ldb_b, b_mask)

        def _wcopy(w, atom, gt, lv, imm_offset):
            if wave == w:
                fx.copy(atom, gt, lv, imm_offset=imm_offset)

        def issue(s, kt):
            pa = _buf_ptr(s)
            koff = fx.Int64(kt) * fx.Int64(KB)
            _wcopy(W_A, atomA, gA, _lv(pa, (tile_m, KB), (LDS_ROW, 1)), koff)
            _wcopy(W_B, atomB, gB, _lv(fx.add_offset(pa, STAGE_A), (tile_n, KB), (LDS_ROW, 1)), koff)

        wmb = wave_m * warp_tile_m
        wnb = wave_n * warp_tile_n
        lds_load_b256, _ = make_lds_copy_ops(256)

        def _frag(buf, b0):
            """One 8-int32 fragment via single ds_read_b256."""
            return Vec(lds_load_b256(buf, b0))

        def load_a(buf, wm, ks):
            row = wmb + wm * WMMA_M + lane16
            # kgrp*32: kgrp=0→bytes[0-31], kgrp=1→bytes[32-63] of K-step row
            return _frag(buf, fx.Int64(row * LDS_ROW + ks * KSB + kgrp * 32))

        def load_b(buf, wn, ks):
            col = wnb + wn * WMMA_N + lane16
            return _frag(buf, fx.Int64(STAGE_A + col * LDS_ROW + ks * KSB + kgrp * 32))

        wmma_atom = fx.make_mma_atom(fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, elem_cls, fx.Float32))
        c_frags = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(n_acc)]
        for cf in c_frags:
            cf.store(Vec.filled(8, 0.0, fx.Float32))

        def _rmem(n, v):
            t = fx.make_rmem_tensor(n, fx.Int32)
            t.store(v)
            return t

        DS_A = DS_B = 1  # ds_read_b256 per fragment (halved from 2×b128)
        KS_DS = wmma_m_rep * DS_A + wmma_n_rep * DS_B  # ds_reads for one WMMA K-step

        def _load_ks(buf, ks):
            """All A/B fragments for one WMMA K-step."""
            act = [_rmem(8, load_a(buf, wm, ks)) for wm in range_constexpr(wmma_m_rep)]
            wt = [_rmem(8, load_b(buf, wn, ks)) for wn in range_constexpr(wmma_n_rep)]
            return act, wt

        def _mma_ks(state):
            act, wt = state
            for wm in range_constexpr(wmma_m_rep):
                for wn_raw in range_constexpr(wmma_n_rep):
                    wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                    idx = wm * wmma_n_rep + wn
                    # B is the instruction's A operand: the accumulator's fast dim is N.
                    fx.gemm(wmma_atom, c_frags[idx], wt[wn], act[wm], c_frags[idx])

        def compute_ktile(buf, prefetch_kt):
            cur = _load_ks(buf, 0)
            for ks in range_constexpr(K_WS):
                nxt = _load_ks(buf, ks + 1) if const_expr(ks + 1 < K_WS) else None
                rocdl.s_wait_dscnt(KS_DS if const_expr(nxt is not None) else 0)
                _mma_ks(cur)
                if const_expr(ks == 0 and prefetch_kt is not None):
                    rocdl.sched_barrier(0)
                    issue(prefetch_kt % num_buffers, prefetch_kt)
                    rocdl.sched_barrier(0)
                if const_expr(nxt is not None):
                    cur = nxt
            rocdl.sched_dsrd(KS_DS)  # prologue group
            for _ks in range_constexpr(K_WS):
                if const_expr(_ks < K_WS - 1):
                    rocdl.sched_dsrd(KS_DS)
                rocdl.sched_mfma(n_acc)
            rocdl.sched_barrier(0)

        if const_expr(use_cluster):
            cluster.cluster_barrier()
        for i in range_constexpr(num_buffers - 1):
            issue(i, i)
        n_steady = K_TILES - (num_buffers - 1)
        for kt in range(n_steady):
            buf = _bidx(_buf_ptr(kt % num_buffers))
            pipeline_fence(outstanding=(num_buffers - 2), use_cluster=False)
            compute_ktile(buf, kt + (num_buffers - 1))
            if const_expr(use_cluster) and kt % num_buffers == num_buffers - 1:
                cluster.cluster_barrier()
        for j in range_constexpr(num_buffers - 1):
            kt = n_steady + j
            buf = _bidx(_buf_ptr(kt % num_buffers))
            pipeline_fence(outstanding=(num_buffers - 2 - j), use_cluster=False)
            compute_ktile(buf, None)

        accs = [c_frags[idx].load() for idx in range_constexpr(n_acc)]
        pipeline_fence(outstanding=0, use_cluster=use_cluster)
        for wm in range_constexpr(wmma_m_rep):
            row_rel = wmb + wm * WMMA_M + lane16
            for wn in range_constexpr(wmma_n_rep):
                col_rel = wnb + wn * WMMA_N + kgrp * 8
                h = accs[wm * wmma_n_rep + wn].to(elem_cls)
                fx.ptr_store(h.bitcast(fx.Int8), base_ptr + (row_rel * tile_n + col_rel) * EB)
        workgroup_barrier(use_cluster=False)
        gtC = _gv(gC_base, fx.Int64(blk_m) * ldc64 + fx.Int64(blk_n), (tile_m, tile_n), (tile_n, 1))
        atomC = fx.rocdl.make_tdm_atom(
            gtC,
            [mn_oob, None],
            strides=[ldc64, None],
            num_warps=num_waves,
            early_timeout=False,
        )
        fx.copy(atomC, _lv(fx.recast_iter(elem_cls, base_ptr), (tile_m, tile_n), (tile_n, 1)), gtC)
        tdm_ops.tensor_wait(0)

    gx = (i32_m + (tile_m - 1)) // tile_m
    gy = N // tile_n
    if use_cluster:
        gx = ((gx + (cluster_m - 1)) // cluster_m) * cluster_m
    cluster_arg = (cluster_m, cluster_n, 1) if use_cluster else None
    kernel_gemm_bf16(
        arg_c,
        arg_a,
        arg_b,
        i32_m,
        K,
        i32_lda,
        i32_ldb,
        i32_ldc,
        value_attrs={"rocdl.cluster_dims": f"{cluster_m},{cluster_n},1" if use_cluster else None},
    ).launch(grid=(gx, gy, 1), block=(block, 1, 1), stream=stream, cluster=cluster_arg)


launch_gemm_bf16.compile_hints["llvm_options"] = {
    "amdgpu-expert-scheduling-mode": True,
}
