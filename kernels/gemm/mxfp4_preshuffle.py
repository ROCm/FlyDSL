# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP4/MXFP6/MXFP8 A x MXFP4 B preshuffle GEMM (gfx950): per-32 E8M0 scales folded into
a scaled 16x16x128 fx.gemm; A streams global->LDS via double-buffered async DMA. Layout
matches the host preshuffle (shuffle_weight_w4(.,16) + shuffle_scale_w4)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fx_math
from flydsl.expr.typing import (
    BFloat16,
    Constexpr,
    Float4E2M1FN,
    Float6E2M3FN,
    Float8E4M3FN,
    Float16,
    Float32,
    Int8,
    Int16,
    Int32,
    T,
)
from flydsl.expr.typing import Vector as Vec
from kernels.common import buffer_ops
from kernels.common.mma.mfma_preshuffle_pipeline import xcd_remap_bx_by

_A_ELEM = {"fp4": Float4E2M1FN, "fp6": Float6E2M3FN, "fp8": Float8E4M3FN}


def _scale_mma_atoms(a_dtype, b_dtype):
    """16 (opsel_a, opsel_b) scaled-MFMA atoms; A elem is fp4/fp6/fp8, B is fp4/fp8."""
    elem_a = _A_ELEM[a_dtype]
    elem_b = _A_ELEM[b_dtype]
    return {
        (osa, osb): fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, elem_a, elem_b, opsel_a=osa, opsel_b=osb))
        for osa in range(4)
        for osb in range(4)
    }


def _bq_view(arg_bq_addr, row_elems, KH4, k_tiles, k_halves, pair):
    """Preshuffled B view for one N-row tile; index [l//16, l%16, kt, half, p, None] -> i32[4].

    `pair` = K0 blocks per 128-K MFMA: 1 for fp4 B (one i32[4]), 2 for fp8 B (lo/hi halves
    packed into i32[8] by load_b). K0 blocks (256 i32 each) run contiguously along K as
    ((kt*k_halves + kh)*pair + p); the fp4 case keeps a size-1 `p` dim (byte-identical view)."""
    col_base = rocdl.readfirstlane(T.i32, row_elems * KH4)
    i32_ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=16)
    off_i64 = fx.Int64(col_base)
    base_iter = fx.inttoptr(i32_ptr_ty, arg_bq_addr + off_i64 * fx.Int64(4))
    shape = (4, 16, k_tiles, k_halves, pair, 4)
    strides = (64, 4, k_halves * pair * 256, pair * 256, 256, 1)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, strides)))
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


@flyc.jit
def launch_gemm(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Pointer,
    arg_scale_b: fx.Pointer,
    arg_bias: fx.Pointer,
    arg_d: fx.Pointer,
    arg_l2: fx.Pointer,
    i32_m: fx.Int32,
    i32_n: fx.Int32,
    stream: fx.Stream,
    N: Constexpr[int],
    K: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    a_dtype: Constexpr[str],
    out_dtype: Constexpr[str],
    b_dtype: Constexpr[str],
    batch: Constexpr[int],
    a_row_stride: Constexpr[int],
    a_batch_stride: Constexpr[int],
    sca_row_stride: Constexpr[int],
    sca_batch_stride: Constexpr[int],
    c_row_stride: Constexpr[int],
    c_batch_stride: Constexpr[int],
    waves_per_eu: Constexpr[int],
    xcd_swizzle: Constexpr[int] = 0,
    epilogue: Constexpr[str] = "none",
    rank: Constexpr[int] = 0,
    svd_use_mfma: Constexpr[bool] = True,
):
    """Direct @flyc.jit launcher. Operands are fx.Pointer (pass ptr_arg(t): raw data_ptr, no
    per-launch DLPack). Compile once with flyc.compile, then cf(*runtime). a_dtype fp4/fp6/fp8
    A x preshuffled b_dtype (fp4/fp8) B, e8m0 scales (a8w8 = a_dtype=fp8, b_dtype=fp8).
    batch>1 = strided-batched over grid.z. The a_/sca_/c_ row/batch strides make A/scale_a/C
    addressing caller-controlled; each <0 keeps the contiguous [B,M,*] bmn default, all set =
    the [M,B,*] mbn layout. waves_per_eu<=0 = unset.
    """
    BM, BN, BK = tile_m, tile_n, tile_k
    if const_expr(out_dtype == "bf16"):
        out_elem = BFloat16
    else:
        out_elem = Float16

    # Fused epilogue: per-N bias add (+ optional activation) folded into the C store.
    # "none"/"bias"/"bias_relu"/"bias_silu"/"bias_gelu" (tanh-approx GELU), plus the
    # SVDQuant low-rank up-projection "svd"/"svd_bias" (y += d @ L2^T, see below).
    _has_bias = epilogue in ("bias", "bias_relu", "bias_silu", "bias_gelu", "svd_bias")
    _has_relu = epilogue == "bias_relu"
    _has_silu = epilogue == "bias_silu"
    _has_gelu = epilogue == "bias_gelu"
    _has_svd = epilogue in ("svd", "svd_bias")
    if _has_svd and rank <= 0:
        raise ValueError(f"epilogue={epilogue!r} requires rank > 0, got {rank!r}")
    # MFMA fast path folds d @ L2^T into the f32 accumulators; needs one bf16 16x16x16 MFMA
    # per 16-wide rank block, so rank must be a multiple of 16 and the operands bf16.
    # Everything else falls back to the per-element scalar dot in the store loop.
    _svd_mfma = _has_svd and svd_use_mfma and rank % 16 == 0 and out_dtype == "bf16"
    _svd_scalar = _has_svd and not _svd_mfma

    # Row sizes + read_a fragment layout (i32 units): fp6/fp8 read two b128 halves -> i32[A_NDW], fp4 one -> i32[4].
    if const_expr(a_dtype == "fp4"):  # 2 codes/byte
        a_row_bytes, A_ROW_B = K // 2, BK // 2
        A_GK_I32, A_KH_I32, A_HI_OFF, A_NDW = 4, 16, 0, 4
    else:
        a_row_bytes, A_ROW_B = K, BK
        if const_expr(a_dtype == "fp8"):
            A_GK_I32, A_KH_I32, A_HI_OFF, A_NDW = 4, 32, 16, 8
        else:  # fp6
            A_GK_I32, A_KH_I32, A_HI_OFF, A_NDW = 8, 32, 4, 6

    A_LDS_B = BM * A_ROW_B  # LDS A buffer bytes (row-major [m][col], shared by 4 N-waves)
    A_ROW_I32 = A_ROW_B // 4
    swz_lds = a_dtype in ("fp4", "fp8")
    k_blk16 = A_ROW_B // 16
    # B fragment layout: fp8 B needs i32[8] (two K0 blocks / 128-K MFMA); fp4 B i32[4] (one).
    if const_expr(b_dtype == "fp8"):
        b_row_bytes, B_NDW, B_BLK_PER_MMA = K, 8, 2
    else:  # fp4
        b_row_bytes, B_NDW, B_BLK_PER_MMA = K // 2, 4, 1
    KH4 = b_row_bytes // 4  # i32 per N-row in preshuffled B (== (K//2)//4 for fp4)
    K_TILES = K // BK
    k_halves = BK // 128  # 16x16x128 MFMA k-steps per K-tile
    # e8m0 scales are 256-K granular, B 128-K: tiles_per_chunk K-tiles share a word (hi/lo 16b = 128-K half).
    tiles_per_chunk = 256 // BK  # 1 for tile_k=256, 2 for tile_k=128
    m_chunks = BM // 16
    num_acc_n = (BN // 4) // 16  # 16-col n-subblocks per wave
    _scale_chunk_dw = (K // 32 // 4 // 2) * 64  # e8m0 stride (dwords), per shuffle_scale_w4
    _scale_k0_dw = 64
    n_coop = A_LDS_B // 256 // 16  # 16B cooperative loads per thread
    n_pairs = max(1, num_acc_n // 2)
    m_pairs = max(1, m_chunks // 2)

    # Scheduler counts per loop iter: MFMAs, A LDS reads/thread (fp6/fp8 2 per (mi,kh)), gmem loads.
    sched_mfma_total = k_halves * m_chunks * num_acc_n
    if const_expr(a_dtype == "fp4"):
        a_ds_per = 1
    else:
        a_ds_per = 2
    sched_num_ds_load = m_chunks * k_halves * a_ds_per
    sched_num_gmem = n_coop + num_acc_n * k_halves * B_BLK_PER_MMA + m_pairs + n_pairs

    @fx.struct
    class SharedA:
        a0: fx.Array[Int8, A_LDS_B, 16]
        a1: fx.Array[Int8, A_LDS_B, 16]

    @flyc.kernel
    def kernel_gemm(
        arg_c: fx.Int64,
        arg_a: fx.Int64,
        arg_b: fx.Int64,
        arg_scale_a: fx.Int64,
        arg_scale_b: fx.Int64,
        arg_bias: fx.Int64,
        arg_d: fx.Int64,
        arg_l2: fx.Int64,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        scale_atoms = _scale_mma_atoms(a_dtype, b_dtype)

        tid = fx.Int32(fx.thread_idx.x)
        bid_x, bid_y, bid_z = fx.block_idx
        wave = rocdl.readfirstlane(T.i32, tid // 64)
        lane = tid % 64
        lane_div_16 = lane // 16
        lane_mod_16 = lane % 16

        if const_expr(xcd_swizzle > 0):
            bid_x, bid_y = xcd_remap_bx_by(
                bid_x,
                bid_y,
                i32_m,
                tile_m=BM,
                tile_n=BN,
                N=N,
                xcd_swizzle=xcd_swizzle,
            )
        bx_m = bid_x * BM
        by_n = bid_y * BN

        # Strided-batched: shift each base to batch bid_z (A/scale_a via explicit strides or
        # the contiguous default; B/scale_b stay batch-contiguous). batch==1 emits no batch math.
        if const_expr(batch > 1):
            a_rstride = fx.Int32(a_row_bytes if a_row_stride < 0 else a_row_stride)
            sca_rstride = fx.Int32(_scale_chunk_dw if sca_row_stride < 0 else sca_row_stride)
            bz = fx.Int64(bid_z)
            if const_expr(a_batch_stride < 0):
                arg_a = arg_a + bz * (fx.Int64(i32_m) * fx.Int64(a_row_bytes))
            else:
                arg_a = arg_a + bz * fx.Int64(a_batch_stride)
            arg_b = arg_b + bz * fx.Int64(N * b_row_bytes)
            if const_expr(sca_batch_stride < 0):
                sc_bstride = fx.Int64((i32_m + 31) // 32) * fx.Int64(_scale_chunk_dw) * fx.Int64(4)
                arg_scale_a = arg_scale_a + bz * sc_bstride
            else:
                arg_scale_a = arg_scale_a + bz * fx.Int64(sca_batch_stride)
            arg_scale_b = arg_scale_b + bz * fx.Int64((N // 32) * _scale_chunk_dw * 4)
        else:
            a_rstride = fx.Int32(a_row_bytes)
            sca_rstride = fx.Int32(_scale_chunk_dw)

        # A source view, bound to the last valid M row (ragged M OOB -> 0).
        _i8g = fx.PointerType.get(T.i8, address_space=fx.AddressSpace.Global, alignment=16)
        if const_expr(batch > 1 and a_row_stride >= 0):
            a_nrec = fx.Int64(i32_m - fx.Int32(1)) * fx.Int64(a_rstride) + fx.Int64(a_row_bytes)
        else:
            a_nrec = fx.Int64(i32_m) * fx.Int64(a_row_bytes)
        a_flat = fx.rocdl.make_buffer_tensor(
            fx.Tensor(
                fx.make_view(
                    fx.inttoptr(_i8g, arg_a),
                    fx.make_layout(65536 * a_row_bytes, 1),
                )
            ),
            max_size=False,
            num_records_bytes=a_nrec,
        )
        a_flat_div = fx.logical_divide(a_flat, fx.make_layout(1, 1))
        lds = fx.SharedAllocator().allocate(SharedA).peek()
        # A-LDS modeled as i32 (16B = 4 i32): fx.copy is dtype-agnostic, only the MMA cares.
        sA0_i32 = fx.recast_iter(Int32, lds.a0.ptr)
        lds_db = fx.Int32(fx.ptrtoint(lds.a1.ptr)) - fx.Int32(fx.ptrtoint(lds.a0.ptr))  # ping/pong byte stride
        lds_db_i32 = lds_db // 4
        lds_copy = fx.make_copy_atom(fx.UniversalCopy128b(), Int32)
        dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        _i8s = fx.PointerType.get(Int8.ir_type, fx.AddressSpace.Shared, 512)
        sA0_i8 = fx.recast_iter(_i8s, lds.a0.ptr)

        def _iter_of(parity):  # parity in {0,1} (runtime) -> i32 LDS iterator
            return fx.add_offset(sA0_i32, parity * lds_db_i32)

        def _lds_view(base_iter, off_i32):
            return fx.make_view(fx.add_offset(base_iter, off_i32), fx.make_layout(4, 1))

        # Async A: gmem->LDS DMA (buffer_load_lds); issued after B/scale loads to overlap the MFMAs.
        def dma_a_to_lds(kt, parity):
            base_off = rocdl.readfirstlane(T.i32, parity * lds_db + wave * (64 * 16))
            lds_ptr = fx.add_offset(sA0_i8, base_off)
            base_k_byte = kt * A_ROW_B
            for i in range_constexpr(n_coop):
                if const_expr(i > 0):
                    lds_ptr = fx.add_offset(lds_ptr, fx.Int32(256 * 16))
                lin = (i * 256 + tid) * 16
                row = lin // A_ROW_B
                col = lin % A_ROW_B
                if const_expr(swz_lds):
                    col = col ^ ((row % k_blk16) * 16)
                gmem_byte = (bx_m + row) * a_rstride + base_k_byte + col
                dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                src = fx.slice(a_flat_div, (None, gmem_byte))
                fx.copy(dma_atom, src, dst)

        def _read16(base_iter, off_i32):
            # ds_read_b128 straight into an i32[4] register fragment.
            t = fx.make_rmem_tensor(4, Int32)
            fx.copy(lds_copy, _lds_view(base_iter, off_i32), t)
            return t

        def read_a(parity):
            base_iter = _iter_of(parity)
            av = []
            for mi in range_constexpr(m_chunks):
                for kh in range_constexpr(k_halves):
                    row = mi * 16 + lane_mod_16
                    row_base = row * A_ROW_I32
                    lo_blk = kh * (A_KH_I32 // 4) + lane_div_16 * (A_GK_I32 // 4)
                    if const_expr(swz_lds):
                        off = row_base + (lo_blk ^ (row % k_blk16)) * 4
                    else:
                        off = row_base + kh * A_KH_I32 + lane_div_16 * A_GK_I32
                    if const_expr(a_dtype == "fp4"):
                        av.append(_read16(base_iter, off))
                    else:
                        # fp6/fp8: pack two halves (64 K apart, f8f6f4 ABI) into i32[A_NDW].
                        if const_expr(swz_lds):
                            hi_off = row_base + ((lo_blk + A_HI_OFF // 4) ^ (row % k_blk16)) * 4
                        else:
                            hi_off = off + A_HI_OFF
                        lo = Vec(fx.memref_load_vec(_read16(base_iter, off)))
                        hi = Vec(fx.memref_load_vec(_read16(base_iter, hi_off)))
                        t = fx.make_rmem_tensor(A_NDW, Int32)
                        t.store(lo.shuffle(hi, list(range(A_NDW))))
                        av.append(t)
            return av

        n_col_base = by_n + wave * (BN // 4)
        bq_views = [
            _bq_view(arg_b, n_col_base + ni * 16, KH4, K_TILES, k_halves, B_BLK_PER_MMA)
            for ni in range_constexpr(num_acc_n)
        ]
        b_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
        bs_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        # e8m0 scale buffers bounded to real size (OOB rows read 0); scale_a to the last 32-row chunk.
        _i32g = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
        _sc_layout = fx.make_layout(1 << 28, 1)
        _a_sc_chunks = (i32_m + 31) // 32
        if const_expr(batch > 1 and sca_row_stride >= 0):
            a_sc_nrec = (fx.Int64(_a_sc_chunks - 1) * fx.Int64(sca_rstride) + fx.Int64(_scale_chunk_dw)) * fx.Int64(4)
        else:
            a_sc_nrec = fx.Int64(_a_sc_chunks) * fx.Int64(_scale_chunk_dw) * fx.Int64(4)
        b_sc_nrec = fx.Int64((N // 32) * _scale_chunk_dw * 4)
        sa_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(_i32g, arg_scale_a), _sc_layout)),
                max_size=False,
                num_records_bytes=a_sc_nrec,
            ),
            fx.make_layout(1, 1),
        )
        sb_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(_i32g, arg_scale_b), _sc_layout)),
                max_size=False,
                num_records_bytes=b_sc_nrec,
            ),
            fx.make_layout(1, 1),
        )
        a_sc_base = [(bx_m // 32 + mp) * sca_rstride for mp in range_constexpr(m_pairs)]
        nsb = by_n // 32 + wave * (BN // 128)
        b_sc_base = [(nsb + np) * _scale_chunk_dw for np in range_constexpr(n_pairs)]
        sc_lane = lane_div_16 * 16 + lane_mod_16

        n_acc = m_chunks * num_acc_n

        def load_b(kt):
            # buffer_load_dwordx4 into i32[4] frags; fp8 B packs two K0 blocks (lo/hi, 64 K
            # apart, f8f6f4 ABI) into i32[8] — same shuffle as read_a's fp6/fp8 A path.
            ops = []
            for ni in range_constexpr(num_acc_n):
                for kh in range_constexpr(k_halves):
                    lo = fx.make_rmem_tensor(4, Int32)
                    fx.copy(b_copy, bq_views[ni][lane_div_16, lane_mod_16, kt, kh, 0, None], lo)
                    if const_expr(b_dtype == "fp4"):
                        ops.append(lo)
                    else:  # fp8: lo ++ hi -> i32[B_NDW]
                        hi = fx.make_rmem_tensor(4, Int32)
                        fx.copy(b_copy, bq_views[ni][lane_div_16, lane_mod_16, kt, kh, 1, None], hi)
                        t = fx.make_rmem_tensor(B_NDW, Int32)
                        t.store(Vec(fx.memref_load_vec(lo)).shuffle(Vec(fx.memref_load_vec(hi)), list(range(B_NDW))))
                        ops.append(t)
            return ops

        def load_sc(chunk_kt):
            # (sa, sb) e8m0 words per m-/n-pair for one 256-K chunk (uniform base -> SGPR soffset).
            koff = chunk_kt * _scale_k0_dw
            sa = [
                Vec(
                    fly.copy_atom_call_ssa(
                        [T.vec(1, T.i32)],
                        bs_copy,
                        sa_flat[
                            None,
                            rocdl.readfirstlane(T.i32, a_sc_base[mp] + koff) + sc_lane,
                        ],
                    )
                )[0]
                for mp in range_constexpr(m_pairs)
            ]
            sb = [
                Vec(
                    fly.copy_atom_call_ssa(
                        [T.vec(1, T.i32)],
                        bs_copy,
                        sb_flat[
                            None,
                            rocdl.readfirstlane(T.i32, b_sc_base[np] + koff) + sc_lane,
                        ],
                    )
                )[0]
                for np in range_constexpr(n_pairs)
            ]
            return sa, sb

        def compute(accs, av, bv, sa_v, sb_v, scale_shift=None):
            # tile_k=128: shift the active 128-K half of the shared 256-K word into the opsel's low bytes.
            if const_expr(scale_shift is not None):
                sa_v = [v.shrui(scale_shift) for v in sa_v]
                sb_v = [v.shrui(scale_shift) for v in sb_v]
            # kh OUTERMOST: consecutive MFMAs hit distinct accumulators (dense issue). Each
            # scaled MFMA = fx.gemm over rank-1 i32[4] A/B frags, e8m0 word on scale_a=/scale_b=.
            c_frags = [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(n_acc)]
            for idx in range_constexpr(n_acc):
                c_frags[idx].store(Vec(accs[idx]))
            for kh in range_constexpr(k_halves):
                for ni in range_constexpr(num_acc_n):
                    np_i, in_b = ni // 2, ni % 2
                    for mi in range_constexpr(m_chunks):
                        mp_i, im = mi // 2, mi % 2
                        cf = c_frags[mi * num_acc_n + ni]
                        fx.gemm(
                            scale_atoms[(kh * 2 + im, kh * 2 + in_b)],
                            cf,
                            av[mi * k_halves + kh],
                            bv[ni * k_halves + kh],
                            cf,
                            scale_a=sa_v[mp_i],
                            scale_b=sb_v[np_i],
                        )
            for idx in range_constexpr(n_acc):
                accs[idx] = c_frags[idx].load().ir_value()
            return accs

        def hot_loop_scheduler():
            # Interleave the MFMAs with the tile's vmem + A-LDS loads: preload all hints, then issue MFMAs 1-by-1.
            rocdl.sched_vmem(sched_num_gmem)
            rocdl.sched_dsrd(sched_num_ds_load)
            for _ in range_constexpr(sched_mfma_total):
                rocdl.sched_mfma(1)
            rocdl.sched_barrier(0)

        accs_init = [Vec.filled(4, 0.0, Float32).ir_value() for _ in range_constexpr(n_acc)]

        # Double-buffered LDS-A: prefetch tile iv+1 into the other buffer while MFMAs compute tile iv.
        dma_a_to_lds(fx.Int32(0), fx.Int32(0))
        rocdl.s_waitcnt(0)
        gpu.barrier()
        for iv, state in range(fx.Index(0), fx.Index(K_TILES), fx.Index(1), init=accs_init):
            accs = list(state)
            kt = fx.Int32(iv)
            cur = kt % 2
            nxt = (kt + 1) % 2
            nkt = kt + 1
            pf_kt = nkt - nkt // K_TILES  # clamp last-iter prefetch to K_TILES-1
            chunk_kt = kt if tiles_per_chunk == 1 else kt // tiles_per_chunk
            scale_shift = None if tiles_per_chunk == 1 else (kt % tiles_per_chunk) * 16
            av = read_a(cur)
            bv = load_b(kt)
            sa_v, sb_v = load_sc(chunk_kt)
            dma_a_to_lds(pf_kt, nxt)  # A DMA after B/scale loads -> overlaps the MFMAs
            accs = compute(accs, av, bv, sa_v, sb_v, scale_shift)
            hot_loop_scheduler()
            rocdl.s_waitcnt(0)  # drain the A DMA before the barrier
            gpu.barrier()
            results = yield accs
        accs = results

        # Epilogue via fx.copy: a lane owns 4 rows per (mi,ni) accm (row m*16+(l//16)*4+ii, col
        # base+l%16), c_stride apart; c_flat bounds ragged-M OOB and honors c_row/batch_stride.
        c_stride = N if c_row_stride < 0 else c_row_stride
        if const_expr(c_row_stride < 0):
            c_nrec = fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(2)
        else:
            c_nrec = (fx.Int64(i32_m - fx.Int32(1)) * fx.Int64(c_stride) + fx.Int64(N)) * fx.Int64(2)
        c_addr = arg_c
        if const_expr(batch > 1):
            c_bstride = fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(2) if c_batch_stride < 0 else fx.Int64(c_batch_stride)
            c_addr = c_addr + fx.Int64(bid_z) * c_bstride
        c_ptr_ty = fx.PointerType.get(out_elem.ir_type, address_space=fx.AddressSpace.Global, alignment=2)
        c_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(c_ptr_ty, c_addr), fx.make_layout(1 << 28, 1))),
                max_size=False,
                num_records_bytes=c_nrec,
            ),
            fx.make_layout(1, 1),
        )
        c_copy = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_elem)
        c_rstride = fx.Int32(c_stride)
        col_w = by_n + wave * (BN // 4) + lane_mod_16
        # Per-N bias resource (out_elem dtype, length N) for the fused epilogue.
        bias_rsrc = None
        if const_expr(_has_bias):
            bias_ptr_ty = fx.PointerType.get(out_elem.ir_type, address_space=fx.AddressSpace.Global, alignment=2)
            bias_view = fx.Tensor(fx.make_view(fx.inttoptr(bias_ptr_ty, arg_bias), fx.make_layout(N, 1)))
            bias_rsrc = buffer_ops.create_buffer_resource(bias_view, max_size=True)
        # SVDQuant low-rank operands: d = x @ L1^T (M x rank) and L2 (N x rank), both
        # out_elem row-major. d is bounded to i32_m rows so ragged-M lanes read 0.
        d_rsrc = None
        l2_rsrc = None
        if const_expr(_has_svd):
            svd_ptr_ty = fx.PointerType.get(out_elem.ir_type, address_space=fx.AddressSpace.Global, alignment=2)
            d_view = fx.Tensor(fx.make_view(fx.inttoptr(svd_ptr_ty, arg_d), fx.make_layout(1 << 28, 1)))
            d_rsrc = buffer_ops.create_buffer_resource(d_view, num_records_bytes=fx.Int64(i32_m) * fx.Int64(rank * 2))
            l2_view = fx.Tensor(fx.make_view(fx.inttoptr(svd_ptr_ty, arg_l2), fx.make_layout(N * rank, 1)))
            l2_rsrc = buffer_ops.create_buffer_resource(l2_view, max_size=True)

        # y += d @ L2^T as a small bf16 GEMM contracted over `rank`, accumulated into the
        # SAME f32 accumulators as the main loop: d is the A operand, L2 the B operand, one
        # mfma_f32_16x16x16_bf16 per 16-wide rank block. Fragments follow the main MFMA
        # convention -- lane_mod_16 picks the row (d) / column (L2), lane_div_16*4 the rank
        # offset -- so each lane reads 4 contiguous elements (stride `rank`) as one vec4.
        if const_expr(_svd_mfma):
            svd_acc_ty = Vec.make_type(4, Float32)
            rank_blocks = rank // 16
            r_lane = lane_div_16 * 4
            a_d = [
                [
                    Vec(
                        buffer_ops.buffer_load(
                            d_rsrc,
                            (bx_m + mi * 16 + lane_mod_16) * rank + kb * 16 + r_lane,
                            vec_width=4,
                            dtype=out_elem.ir_type,
                        )
                    ).bitcast(Int16)
                    for kb in range_constexpr(rank_blocks)
                ]
                for mi in range_constexpr(m_chunks)
            ]
            b_l2 = [
                [
                    Vec(
                        buffer_ops.buffer_load(
                            l2_rsrc,
                            (col_w + ni * 16) * rank + kb * 16 + r_lane,
                            vec_width=4,
                            dtype=out_elem.ir_type,
                        )
                    ).bitcast(Int16)
                    for kb in range_constexpr(rank_blocks)
                ]
                for ni in range_constexpr(num_acc_n)
            ]
            for mi in range_constexpr(m_chunks):
                for ni in range_constexpr(num_acc_n):
                    acc_idx = mi * num_acc_n + ni
                    acc_v = accs[acc_idx]
                    for kb in range_constexpr(rank_blocks):
                        acc_v = rocdl.mfma_f32_16x16x16bf16_1k(svd_acc_ty, [a_d[mi][kb], b_l2[ni][kb], acc_v, 0, 0, 0])
                    accs[acc_idx] = acc_v

        # Scalar fallback: L2[col, :] is row-independent, so hoist it out of the store loop.
        l2_pre = None
        if const_expr(_svd_scalar):
            l2_pre = [
                [
                    fx.Float32(
                        buffer_ops.buffer_load(
                            l2_rsrc, (col_w + ni * 16) * rank + r, vec_width=1, dtype=out_elem.ir_type
                        )
                    )
                    for r in range_constexpr(rank)
                ]
                for ni in range_constexpr(num_acc_n)
            ]
        for mi in range_constexpr(m_chunks):
            row_m = bx_m + mi * 16 + lane_div_16 * 4
            for ni in range_constexpr(num_acc_n):
                col = col_w + ni * 16
                if const_expr(_has_bias or _svd_scalar):
                    # bias is per output column (N); apply + activation in f32.
                    acc_f = Vec(accs[mi * num_acc_n + ni])  # f32 accumulator
                    bval = None
                    if const_expr(_has_bias):
                        bval = fx.Float32(buffer_ops.buffer_load(bias_rsrc, col, vec_width=1, dtype=out_elem.ir_type))
                    outs = []
                    for ii in range_constexpr(4):
                        v = fx.Float32(acc_f[ii])
                        if const_expr(_has_bias):
                            v = v + bval
                        if const_expr(_svd_scalar):
                            # y[row, col] += sum_r d[row, r] * L2[col, r], accumulated in f32.
                            d_row = (row_m + ii) * rank
                            svd_dot = fx.Float32(0.0)
                            for r in range_constexpr(rank):
                                dv = fx.Float32(
                                    buffer_ops.buffer_load(d_rsrc, d_row + r, vec_width=1, dtype=out_elem.ir_type)
                                )
                                svd_dot = svd_dot + dv * l2_pre[ni][r]
                            v = v + svd_dot
                        if const_expr(_has_relu):
                            v = fx.Float32(v).maximumf(fx.Float32(0.0))
                        elif const_expr(_has_silu):
                            v = v * (fx.Float32(1.0) / (fx.Float32(1.0) + fx_math.exp(v * fx.Float32(-1.0))))
                        elif const_expr(_has_gelu):
                            # tanh-approx GELU, overflow-safe (non-positive exponent).
                            x3 = v * v * v
                            y = fx.Float32(0.7978845608) * (v + fx.Float32(0.044715) * x3)
                            abs_y = fx.Float32(y).maximumf(fx.Float32(0.0) - y)
                            e = fx_math.exp(fx.Float32(-2.0) * abs_y)
                            denom = fx.Float32(1.0) + e
                            one_plus_tanh = (y >= fx.Float32(0.0)).select(
                                fx.Float32(2.0) / denom, (fx.Float32(2.0) * e) / denom
                            )
                            v = fx.Float32(0.5) * v * one_plus_tanh
                        outs.append(v)
                    acc = Vec.from_elements(outs, fx.Float32).to(out_elem)
                else:
                    acc = Vec(accs[mi * num_acc_n + ni]).to(out_elem)
                for ii in range_constexpr(4):
                    cf = fx.make_rmem_tensor(1, out_elem)
                    cf.store(Vec.from_elements([acc[ii]], out_elem))
                    off = (row_m + ii) * c_rstride + col
                    fx.copy(c_copy, cf, c_flat[None, off])

    c_addr = fx.Int64(fx.ptrtoint(arg_c))
    a_addr = fx.Int64(fx.ptrtoint(arg_a))
    b_addr = fx.Int64(fx.ptrtoint(arg_b))
    sa_addr = fx.Int64(fx.ptrtoint(arg_scale_a))
    sb_addr = fx.Int64(fx.ptrtoint(arg_scale_b))
    bias_addr = fx.Int64(fx.ptrtoint(arg_bias))
    d_addr = fx.Int64(fx.ptrtoint(arg_d))
    l2_addr = fx.Int64(fx.ptrtoint(arg_l2))
    if const_expr(waves_per_eu > 0):
        wpe = waves_per_eu
    else:
        wpe = None
    gx = (i32_m + (BM - 1)) // BM
    gy = i32_n // BN
    kernel_gemm(
        c_addr,
        a_addr,
        b_addr,
        sa_addr,
        sb_addr,
        bias_addr,
        d_addr,
        l2_addr,
        i32_m,
        i32_n,
        value_attrs={"rocdl.waves_per_eu": wpe},
    ).launch(grid=(gx, gy, batch), block=(256, 1, 1), stream=stream)
