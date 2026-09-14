# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP4/MXFP6/MXFP8 A x MXFP4/MXFP8 B preshuffle GEMM (gfx950): per-32 E8M0 scales folded
into a scaled 16x16x128 fx.gemm; A streams global->LDS via double-buffered async DMA. Layout
matches the host preshuffle (shuffle_weight_w4(.,16) + shuffle_scale_w4).

Two scale modes: ``blockscale="ab"`` puts both operands on a coarse E8M0 grid (A per
1x128, B per 128x128); ``"none"`` keeps the per-1x32 MX scales. The separate ``_bs_a`` /
``_bs_b`` flags below exist because the A and B scale readers differ in lane broadcasting,
not to offer mixed modes. ``k_batch`` > 1 enables split-K, reduced by
``launch_splitk_reduce``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import (
    BFloat16,
    Constexpr,
    Float4E2M1FN,
    Float6E2M3FN,
    Float8E4M3FN,
    Float16,
    Float32,
    Int8,
    Int32,
    T,
)
from flydsl.expr.typing import Vector as Vec
from kernels.common.mma.mfma_preshuffle_pipeline import xcd_remap_bx_by
from kernels.gemm.preshuffle_layout import make_preshuffle_dma_layouts, preshuffle_dma_lane_coord

_A_ELEM = {"fp4": Float4E2M1FN, "fp6": Float6E2M3FN, "fp8": Float8E4M3FN}
_B_ELEM = {"fp4": Float4E2M1FN, "fp8": Float8E4M3FN}


def _scale_mma_atoms(a_dtype, b_dtype):
    """16 (opsel_a, opsel_b) scaled-MFMA atoms; A elem is fp4/fp6/fp8, B is fp4/fp8."""
    elem_a = _A_ELEM[a_dtype]
    elem_b = _B_ELEM[b_dtype]
    return {
        (osa, osb): fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, elem_a, elem_b, opsel_a=osa, opsel_b=osb))
        for osa in range(4)
        for osb in range(4)
    }


def _bq_view(arg_bq_addr, row_elems, KH4, k_tiles, k_halves, pair):
    """Preshuffled B view for one N-row tile; index [l//16, l%16, kt, half, p, None] -> i32[4].

    `pair` = K0 blocks per 128-K MFMA: 1 for fp4 B (one i32[4]), 2 for fp8 B (lo/hi halves
    packed into i32[8] by load_b). K0 blocks (256 i32 each) run contiguously along K as
    ((kt*k_halves + kh)*pair + p); the fp4 case keeps a size-1 `p` dim (byte-identical view).
    """
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
    k_batch: Constexpr[int] = 1,
    blockscale: Constexpr[str] = "none",
):
    """Direct @flyc.jit launcher. Operands are fx.Pointer (pass ptr_arg(t): raw data_ptr, no
    per-launch DLPack). Compile once with flyc.compile, then cf(*runtime). a_dtype fp4/fp6/fp8
    A x preshuffled b_dtype (fp4/fp8) B, e8m0 scales (a8w8 = a_dtype=fp8, b_dtype=fp8).
    batch>1 = strided-batched over grid.z. The a_/sca_/c_ row/batch strides make A/scale_a/C
    addressing caller-controlled; each <0 keeps the contiguous [B,M,*] bmn default, all set =
    the [M,B,*] mbn layout. waves_per_eu<=0 = unset.
    """
    BM, BN, BK = tile_m, tile_n, tile_k
    assert blockscale in ("none", "ab"), f"blockscale must be 'none' or 'ab', got {blockscale!r}"
    assert BK in (128, 256), f"tile_k must be 128 or 256 (tiles_per_chunk = 256 // tile_k), got {BK}"
    assert BM % 32 == 0, f"tile_m must be a multiple of 32 (the A e8m0 scale is 32-row granular), got {BM}"
    assert BN % 16 == 0, f"tile_n must be a multiple of 16, got {BN}"
    if const_expr(out_dtype == "bf16"):
        out_elem = BFloat16
    else:
        out_elem = Float16

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

    A_LDS_B = BM * A_ROW_B  # LDS A buffer bytes (row-major [m][col], shared by the N-waves)
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
    assert K_TILES % k_batch == 0, "K_TILES must be divisible by k_batch"
    k_tiles_local = K_TILES // k_batch
    k_halves = BK // 128  # 16x16x128 MFMA k-steps per K-tile
    # e8m0 scales are 256-K granular, B 128-K: tiles_per_chunk K-tiles share a word (hi/lo 16b = 128-K half).
    tiles_per_chunk = 256 // BK  # 1 for tile_k=256, 2 for tile_k=128
    m_chunks = BM // 16
    num_waves = min(4, BN // 16)
    num_threads = num_waves * 64
    num_acc_n = (BN // num_waves) // 16  # 16-col n-subblocks per wave
    _bs_a = blockscale in ("a", "ab")
    _bs_b = blockscale in ("b", "ab")
    _sc_k0_a = 16 if _bs_a else 64  # per-256K-chunk dword stride (A: drop K_Lane=4)
    _sc_k0_b = 1 if _bs_b else 64  # (B: drop K_Lane*N_Lane=64)
    _scale_chunk_dw_a = ((K + 255) // 256) * _sc_k0_a
    _scale_chunk_dw_b = ((K + 255) // 256) * _sc_k0_b
    _b_sc_rows = (N // 128) if _bs_b else (N // 32)  # B scale super-rows
    a_copy_granularity = num_threads * 16  # bytes one cooperative round covers
    assert A_LDS_B % a_copy_granularity == 0, (
        f"A_LDS_B ({A_LDS_B}B) must be divisible by num_threads*16 ({a_copy_granularity}B), else part "
        "of the A tile is never DMA'd; adjust tile_m / tile_k"
    )
    n_coop = A_LDS_B // a_copy_granularity  # 16B cooperative loads per thread
    assert k_batch == 1 or (c_row_stride < 0 and c_batch_stride < 0), (
        "split-K writes tmp[batch*k_batch, M, N] as a contiguous fp32 slab, so k_batch > 1 "
        "requires the default c_row_stride / c_batch_stride (-1)"
    )
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
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        scale_atoms = _scale_mma_atoms(a_dtype, b_dtype)

        tid = fx.Int32(fx.thread_idx.x)
        bid_x, bid_y, bid_z = fx.block_idx

        if const_expr(k_batch > 1):
            bz_batch = bid_z // k_batch
            kt0 = fx.Int32(bid_z % k_batch) * fx.Int32(k_tiles_local)
        else:
            bz_batch = bid_z
            kt0 = fx.Int32(0)
        wave = rocdl.readfirstlane(T.i32, tid // 64)
        lane = tid % 64
        lane_div_16 = lane // 16
        lane_mod_16 = lane % 16
        # XCD swizzle: remap (bid_x, bid_y) for L2-cache reuse (no-op when xcd_swizzle<=0).
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
            sca_rstride = fx.Int32(_scale_chunk_dw_a if sca_row_stride < 0 else sca_row_stride)
            bz = fx.Int64(bz_batch)
            if const_expr(a_batch_stride < 0):
                arg_a = arg_a + bz * (fx.Int64(i32_m) * fx.Int64(a_row_bytes))
            else:
                arg_a = arg_a + bz * fx.Int64(a_batch_stride)
            arg_b = arg_b + bz * fx.Int64(N * b_row_bytes)
            if const_expr(sca_batch_stride < 0):
                sc_bstride = fx.Int64((i32_m + 31) // 32) * fx.Int64(_scale_chunk_dw_a) * fx.Int64(4)
                arg_scale_a = arg_scale_a + bz * sc_bstride
            else:
                arg_scale_a = arg_scale_a + bz * fx.Int64(sca_batch_stride)
            arg_scale_b = arg_scale_b + bz * fx.Int64(_b_sc_rows * _scale_chunk_dw_b * 4)
        else:
            a_rstride = fx.Int32(a_row_bytes)
            sca_rstride = fx.Int32(_scale_chunk_dw_a)

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

        dma_src_layout, dma_dst_layout = make_preshuffle_dma_layouts(
            BM, A_ROW_B, a_rstride, num_threads, swizzle=swz_lds
        )
        dma_lane = preshuffle_dma_lane_coord(lane, A_ROW_B)
        dma_a_base = fx.add_offset(fx.get_iter(a_flat), bx_m * a_rstride)
        # (dword, row lane, K lane, M tile, K half, register half).
        a_read_layout = fx.make_layout(
            (4, 16, 4, m_chunks, k_halves, 1 if a_dtype == "fp4" else 2),
            (1, A_ROW_I32, A_GK_I32, 16 * A_ROW_I32, A_KH_I32, A_HI_OFF),
        )
        if const_expr(swz_lds):
            swz_bits = k_blk16.bit_length() - 1
            a_read_layout = fx.make_composed_layout(fx.static(fx.SwizzleType.get(swz_bits, 2, swz_bits)), a_read_layout)

        # Async A: gmem->LDS DMA (buffer_load_lds); issued after B/scale loads to overlap the MFMAs.
        def dma_a_to_lds(kt, parity):
            src = fx.Tensor(fx.make_view(fx.add_offset(dma_a_base, kt * A_ROW_B), dma_src_layout))
            base_off = rocdl.readfirstlane(T.i32, parity * lds_db)
            dst = fx.Tensor(fx.make_view(fx.add_offset(sA0_i8, base_off), dma_dst_layout))
            for i in range_constexpr(n_coop):
                fx.copy(dma_atom, src[None, dma_lane, wave, i], dst[None, wave, i])

        def _read16(src):
            # ds_read_b128 straight into an i32[4] register fragment.
            t = fx.make_rmem_tensor(4, Int32)
            fx.copy(lds_copy, src, t)
            return t

        def read_a(parity):
            src = fx.Tensor(fx.make_view(_iter_of(parity), a_read_layout))
            av = []
            for mi in range_constexpr(m_chunks):
                for kh in range_constexpr(k_halves):
                    lo = _read16(src[None, lane_mod_16, lane_div_16, mi, kh, 0])
                    if const_expr(a_dtype == "fp4"):
                        av.append(lo)
                    else:
                        # fp6/fp8: pack two halves (64 K apart, f8f6f4 ABI) into i32[A_NDW].
                        hi = _read16(src[None, lane_mod_16, lane_div_16, mi, kh, 1])
                        t = fx.make_rmem_tensor(A_NDW, Int32)
                        t.store(lo.load().shuffle(hi.load(), list(range(A_NDW))))
                        av.append(t)
            return av

        n_col_base = by_n + wave * (BN // num_waves)
        bq_views = [
            _bq_view(arg_b, n_col_base + ni * 16, KH4, K_TILES, k_halves, B_BLK_PER_MMA)
            for ni in range_constexpr(num_acc_n)
        ]
        b_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
        bs_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        # e8m0 scale buffers bounded to real size (OOB rows read 0); scale_a to the last 32-row chunk.
        _i32g = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
        sc_k_chunks = (K + 255) // 256
        _a_sc_chunks = (i32_m + 31) // 32
        if const_expr(batch > 1 and sca_row_stride >= 0):
            a_sc_nrec = (fx.Int64(_a_sc_chunks - 1) * fx.Int64(sca_rstride) + fx.Int64(_scale_chunk_dw_a)) * fx.Int64(4)
        else:
            a_sc_nrec = fx.Int64(_a_sc_chunks) * fx.Int64(_scale_chunk_dw_a) * fx.Int64(4)
        b_sc_nrec = fx.Int64(_b_sc_rows * _scale_chunk_dw_b * 4)
        sa_view = fx.rocdl.make_buffer_tensor(
            fx.Tensor(
                fx.make_view(
                    fx.inttoptr(_i32g, arg_scale_a),
                    fx.make_layout((1, 16, 4, sc_k_chunks, 2048), (0, 1, 0 if _bs_a else 16, _sc_k0_a, sca_rstride)),
                )
            ),
            max_size=False,
            num_records_bytes=a_sc_nrec,
        )
        sb_view = fx.rocdl.make_buffer_tensor(
            fx.Tensor(
                fx.make_view(
                    fx.inttoptr(_i32g, arg_scale_b),
                    fx.make_layout(
                        (1, 16, 4, sc_k_chunks, _b_sc_rows),
                        (0, 0 if _bs_b else 1, 0 if _bs_b else 16, _sc_k0_b, _scale_chunk_dw_b),
                    ),
                )
            ),
            max_size=False,
            num_records_bytes=b_sc_nrec,
        )
        nsb = (by_n + wave * (BN // num_waves)) // 32
        # B blockscale: 4 consecutive 32-N super-rows share one 128-N block scale.
        b_sc_rows = [(nsb + np) // 4 if _bs_b else nsb + np for np in range_constexpr(n_pairs)]

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
                        t.store(lo.load().shuffle(hi.load(), list(range(B_NDW))))
                        ops.append(t)
            return ops

        def load_sc(chunk_kt):
            # Broadcast strides describe both compact block scales and per-32 MX scales.
            def load_word(src):
                frag = fx.make_rmem_tensor(1, Int32)
                fx.copy(bs_copy, src, frag)
                return frag.load()[0]

            sa = [
                load_word(sa_view[None, lane_mod_16, lane_div_16, chunk_kt, bx_m // 32 + mp])
                for mp in range_constexpr(m_pairs)
            ]

            _bs_b_dedup = _bs_b and (BN // num_waves) <= 128 and (128 % (BN // num_waves) == 0)
            if const_expr(_bs_b_dedup):
                _sb0 = load_word(sb_view[None, lane_mod_16, lane_div_16, chunk_kt, b_sc_rows[0]])
                sb = [_sb0 for _ in range_constexpr(n_pairs)]
            else:
                sb = [
                    load_word(sb_view[None, lane_mod_16, lane_div_16, chunk_kt, b_sc_rows[np]])
                    for np in range_constexpr(n_pairs)
                ]
            return sa, sb

        def compute(accs, av, bv, sa_v, sb_v, scale_shift=None):
            # tile_k=128: shift the active 128-K half of the shared 256-K word into the opsel's low bytes.
            if const_expr(scale_shift is not None):
                sa_v = [v.shrui(scale_shift) for v in sa_v]
                sb_v = [v.shrui(scale_shift) for v in sb_v]
            if const_expr(BN < 128):
                _bnsh = ((by_n + wave * (BN // num_waves)) % 32) // 16 * 8
                sb_v = [v.shrui(_bnsh) for v in sb_v]
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

        dma_a_to_lds(kt0, fx.Int32(0))
        rocdl.s_waitcnt(0)
        gpu.barrier()
        for iv, state in range(fx.Index(0), fx.Index(k_tiles_local), fx.Index(1), init=accs_init):
            accs = list(state)
            ivi = fx.Int32(iv)
            cur = ivi % 2
            nxt = (ivi + 1) % 2
            kt = kt0 + ivi  # absolute K-tile for A/B/scale addressing
            nkt = ivi + 1
            # clamp last-iter prefetch to the local last tile, then rebase to absolute
            pf_kt = kt0 + (nkt - nkt // k_tiles_local)
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

        # Encode each lane's accumulator coordinates in a 2-D flat output view.
        c_stride = N if c_row_stride < 0 else c_row_stride
        # split-K writes an fp32 partial slab (no cast); no-split writes bf16/fp16 out.
        if const_expr(k_batch > 1):
            store_elem = Float32
            _ebytes = 4
            # arg_c is tmp[batch*k_batch, M, N] fp32; this WG's slab index == bid_z.
            c_addr = arg_c + fx.Int64(bid_z) * fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(_ebytes)
        else:
            store_elem = out_elem
            _ebytes = 2
            c_addr = arg_c
            if const_expr(batch > 1):
                c_bstride = (
                    fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(2) if c_batch_stride < 0 else fx.Int64(c_batch_stride)
                )
                c_addr = c_addr + fx.Int64(bz_batch) * c_bstride

        c_tile_addr = c_addr + fx.Int64(bx_m) * fx.Int64(c_stride) * fx.Int64(_ebytes)
        _rows_rem = i32_m - fx.Int32(bx_m)
        _rows_wg = (_rows_rem < fx.Int32(BM)).select(_rows_rem, fx.Int32(BM))
        c_nrec = fx.Int64(_rows_wg) * fx.Int64(c_stride) * fx.Int64(_ebytes)
        c_ptr_ty = fx.PointerType.get(store_elem.ir_type, address_space=fx.AddressSpace.Global, alignment=_ebytes)
        c_flat = fx.rocdl.make_buffer_tensor(
            fx.Tensor(
                fx.make_view(
                    fx.inttoptr(c_ptr_ty, c_tile_addr),
                    fx.make_layout((1, 1 << 28), (0, 1)),
                )
            ),
            max_size=False,
            num_records_bytes=c_nrec,
        )
        if const_expr(k_batch > 1):
            c_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), store_elem)
        else:
            c_copy = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), store_elem)
        c_rstride = fx.Int32(c_stride)
        col_w = by_n + wave * (BN // num_waves) + lane_mod_16
        for mi in range_constexpr(m_chunks):
            row_local = mi * 16 + lane_div_16 * 4
            for ni in range_constexpr(num_acc_n):
                col = col_w + ni * 16
                acc = Vec(accs[mi * num_acc_n + ni]).to(store_elem)
                for ii in range_constexpr(4):
                    cf = fx.make_rmem_tensor(1, store_elem)
                    cf.store(Vec.from_elements([acc[ii]], store_elem))
                    off = (row_local + ii) * c_rstride + col
                    fx.copy(c_copy, cf, c_flat[None, off])

    c_addr = fx.Int64(fx.ptrtoint(arg_c))
    a_addr = fx.Int64(fx.ptrtoint(arg_a))
    b_addr = fx.Int64(fx.ptrtoint(arg_b))
    sa_addr = fx.Int64(fx.ptrtoint(arg_scale_a))
    sb_addr = fx.Int64(fx.ptrtoint(arg_scale_b))
    if const_expr(waves_per_eu > 0):
        wpe = waves_per_eu
    else:
        wpe = None
    gx = (i32_m + (BM - 1)) // BM
    gy = i32_n // BN
    gz = batch * k_batch  # split-K: k_batch splits per (real) batch on grid.z
    kernel_gemm(
        c_addr,
        a_addr,
        b_addr,
        sa_addr,
        sb_addr,
        i32_m,
        i32_n,
        value_attrs={"rocdl.waves_per_eu": wpe},
    ).launch(grid=(gx, gy, gz), block=(num_threads, 1, 1), stream=stream)


# ── split-K reduce ────────────────────────────────────────────────────────────

_REDUCE_BLOCK = 256
_REDUCE_VEC = 2  # out elems per thread == one dword (2 fp32 in per slab)


@flyc.jit
def launch_splitk_reduce(
    arg_tmp: fx.Pointer,
    arg_out: fx.Pointer,
    n_out_dw: fx.Int32,  # output dwords = M*N // 2 (2 out elems per dword)
    slab_stride_dw: fx.Int32,  # dwords per split slab = M*N (fp32: 1 dword/elem)
    stream: fx.Stream,
    split_k: Constexpr[int],
    out_dtype: Constexpr[str],
):
    """Sum ``split_k`` fp32 slabs of ``arg_tmp`` -> bf16/fp16 ``arg_out``.

    arg_tmp: (split_k, M, N) fp32 contiguous. arg_out: (M, N) out_dtype. One output
    dword (= 2 out elems = 2 fp32 inputs per slab) per thread; grid.x covers all.
    """
    if const_expr(out_dtype == "bf16"):
        out_elem = BFloat16
    else:
        out_elem = Float16

    @flyc.kernel
    def reduce_kernel(
        tmp: fx.Pointer,
        out: fx.Pointer,
        n_out_dw_i: fx.Int32,
        slab_dw_i: fx.Int32,
    ):
        frag_layout = fx.make_layout(_REDUCE_VEC, 1)
        vt = fx.Int32(fx.block_idx.x) * fx.Int32(_REDUCE_BLOCK) + fx.Int32(fx.thread_idx.x)
        slab_frags = slab_dw_i // fx.Int32(_REDUCE_VEC)

        def _tiled(ptr, elem, num_records_bytes):
            typed = fx.PointerType.get(
                elem.ir_type,
                address_space=fx.AddressSpace.Global,
                alignment=_REDUCE_VEC * elem.width // 8,
            )
            buf = fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(
                        fx.inttoptr(typed, fx.Int64(fx.ptrtoint(ptr))),
                        fx.make_layout((_REDUCE_VEC, 1 << 29), (1, _REDUCE_VEC)),
                    )
                ),
                max_size=False,
                num_records_bytes=num_records_bytes,
            )
            return buf

        in_t = _tiled(tmp, Float32, fx.Int64(slab_dw_i) * fx.Int64(split_k) * fx.Int64(4))
        out_t = _tiled(out, out_elem, fx.Int64(n_out_dw_i) * fx.Int64(4))
        in_copy = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), Float32)  # 2 x f32
        out_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), out_elem)  # 2 x 16b

        if vt < n_out_dw_i:
            acc = Vec.filled(_REDUCE_VEC, 0.0, Float32)
            for sk in range_constexpr(split_k):
                frag = fx.make_rmem_tensor(frag_layout, Float32)
                fx.copy(
                    in_copy,
                    fx.slice(in_t, (None, vt + fx.Int32(sk) * slab_frags)),
                    frag,
                )
                acc = acc + frag.load()
            out_frag = fx.make_rmem_tensor(frag_layout, out_elem)
            out_frag.store(acc.to(out_elem))
            fx.copy(out_copy, out_frag, fx.slice(out_t, (None, vt)))

    gx = (n_out_dw + (_REDUCE_BLOCK - 1)) // _REDUCE_BLOCK
    reduce_kernel(arg_tmp, arg_out, n_out_dw, slab_stride_dw).launch(
        grid=(gx, 1, 1), block=(_REDUCE_BLOCK, 1, 1), stream=stream
    )
