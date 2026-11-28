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
from flydsl._mlir.dialects import fly
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
    base_iter = fx.inttoptr(i32_ptr_ty, arg_bq_addr + off_i64 * 4)
    shape = (4, 16, k_tiles, k_halves, pair, 4)
    strides = (64, 4, k_halves * pair * 256, pair * 256, 256, 1)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, strides)))
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def _emit_gemm(
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
    pair_tail: Constexpr[int] = 0,
    xcd_hw_order: Constexpr[bool] = False,
):
    """Build and launch the GEMM kernel for one tile geometry, inside launch_gemm's trace."""
    BM, BN, BK = tile_m, tile_n, tile_k
    assert blockscale in ("none", "ab"), f"blockscale must be 'none' or 'ab', got {blockscale!r}"
    assert BK in (128, 256), f"tile_k must be 128 or 256 (tiles_per_chunk = 256 // tile_k), got {BK}"
    assert BM % 32 == 0 or (
        BM == 16 and a_dtype == "fp4" and b_dtype == "fp4" and blockscale == "none"
    ), f"tile_m must be a multiple of 32 (or a 16-row fp4 x fp4 tile), got {BM}"
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
    dma_threads = min(num_threads, A_LDS_B // 16)
    a_copy_granularity = dma_threads * 16  # bytes one cooperative round covers
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
    # The paired loop's prefetch list holds, per N fragment, that fragment of the next two
    # tiles; splitting it into two whole tiles is only right with one N fragment per wave.
    pair_k = (
        BM <= 32
        and num_acc_n == 1
        and k_tiles_local >= 2
        and a_dtype == "fp4"
        and b_dtype == "fp4"
        and k_batch == 1
        and blockscale == "none"
    )
    # The register-carried B/scale prefetch and the large-tile tail peel are tuned for fp4 x fp4;
    # the other formats keep the loop that loads B in the iteration that consumes it.
    b_pipe = a_dtype == "fp4" and b_dtype == "fp4" and k_batch == 1 and blockscale == "none"

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

    @fx.struct
    class SharedA4:
        a0: fx.Array[Int8, A_LDS_B * 4, 16]

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
            if const_expr(xcd_swizzle == 1 and b_pipe):
                # WGM=1 makes group_size_m and first_pid_m unnecessary: after the
                # XCD distribution, (m, n) is simply the linearized range quotient.
                n_tiles = N // BN
                num_wgs = n_tiles * ((i32_m + (BM - 1)) // BM)
                linear = bid_x * n_tiles + bid_y
                # These indices are nonnegative: masks/shifts avoid signed-divide rounding.
                xcd = linear & 7
                in_xcd = linear >> 3
                q = num_wgs >> 3
                r = num_wgs & 7
                clip = (xcd < r).select(xcd, r)
                wgid = xcd * q + clip + in_xcd
                if const_expr((n_tiles & (n_tiles - 1)) == 0):
                    n_shift = n_tiles.bit_length() - 1
                    bid_x = wgid >> n_shift
                    bid_y = wgid & (n_tiles - 1)
                else:
                    bid_x = wgid // n_tiles
                    bid_y = wgid % n_tiles
            elif const_expr(xcd_hw_order):
                # The grid is (M tiles, N tiles) and workgroups reach the 8 XCDs round-robin in
                # dispatch order, M fastest. Give each XCD a contiguous range of that order, then
                # walk each range in groups of xcd_swizzle M tiles so that concurrent workgroups
                # share A rows and B columns in L2.
                n_tiles = N // BN
                m_tiles = (i32_m + (BM - 1)) // BM
                num_wgs = n_tiles * m_tiles
                hw_id = fx.Int32(bid_x) + fx.Int32(bid_y) * m_tiles
                xcd = hw_id & 7
                in_xcd = hw_id >> 3
                q = num_wgs >> 3
                r = num_wgs & 7
                clip = (xcd < r).select(xcd, r)
                wgid = xcd * q + clip + in_xcd
                per_group = xcd_swizzle * n_tiles
                first_m = (wgid // per_group) * xcd_swizzle
                rem_m = m_tiles - first_m
                group_m = (rem_m < xcd_swizzle).select(rem_m, fx.Int32(xcd_swizzle))
                in_group = wgid % per_group
                bid_x = first_m + in_group % group_m
                bid_y = in_group // group_m
            else:
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
                sc_bstride = fx.Int64((i32_m + 31) // 32) * fx.Int64(_scale_chunk_dw_a) * 4
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
            a_nrec = fx.Int64(i32_m - 1) * fx.Int64(a_rstride) + fx.Int64(a_row_bytes)
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
        if const_expr(pair_k):
            lds = fx.SharedAllocator().allocate(SharedA4).peek()
            lds_a1_ptr = fx.add_offset(lds.a0.ptr, A_LDS_B)
        else:
            lds = fx.SharedAllocator().allocate(SharedA).peek()
            lds_a1_ptr = lds.a1.ptr
        # A-LDS modeled as i32 (16B = 4 i32): fx.copy is dtype-agnostic, only the MMA cares.
        sA0_i32 = fx.recast_iter(Int32, lds.a0.ptr)
        lds_db = fx.Int32(fx.ptrtoint(lds_a1_ptr)) - fx.Int32(fx.ptrtoint(lds.a0.ptr))  # ping/pong byte stride
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
                    lds_ptr = fx.add_offset(lds_ptr, fx.Int32(dma_threads * 16))
                lin = (i * dma_threads + tid) * 16
                a_row_shift = A_ROW_B.bit_length() - 1
                row = lin >> a_row_shift
                col = lin & (A_ROW_B - 1)
                if const_expr(swz_lds):
                    col = col ^ ((row & (k_blk16 - 1)) * 16)
                gmem_byte = (bx_m + row) * a_rstride + base_k_byte + col
                dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                src = fx.slice(a_flat_div, (None, gmem_byte))
                if const_expr(dma_threads < num_threads):
                    if tid < dma_threads:
                        fx.copy(dma_atom, src, dst)
                else:
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
                        off = row_base + (lo_blk ^ (row & (k_blk16 - 1))) * 4
                    else:
                        off = row_base + kh * A_KH_I32 + lane_div_16 * A_GK_I32
                    if const_expr(a_dtype == "fp4"):
                        av.append(_read16(base_iter, off))
                    else:
                        # fp6/fp8: pack two halves (64 K apart, f8f6f4 ABI) into i32[A_NDW].
                        if const_expr(swz_lds):
                            hi_off = row_base + ((lo_blk + A_HI_OFF // 4) ^ (row & (k_blk16 - 1))) * 4
                        else:
                            hi_off = off + A_HI_OFF
                        lo = Vec(fx.memref_load_vec(_read16(base_iter, off)))
                        hi = Vec(fx.memref_load_vec(_read16(base_iter, hi_off)))
                        t = fx.make_rmem_tensor(A_NDW, Int32)
                        t.store(lo.shuffle(hi, list(range(A_NDW))))
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
        _sc_layout = fx.make_layout(1 << 28, 1)
        _a_sc_chunks = (i32_m + 31) // 32
        if const_expr(batch > 1 and sca_row_stride >= 0):
            a_sc_nrec = (fx.Int64(_a_sc_chunks - 1) * fx.Int64(sca_rstride) + fx.Int64(_scale_chunk_dw_a)) * 4
        else:
            a_sc_nrec = fx.Int64(_a_sc_chunks) * fx.Int64(_scale_chunk_dw_a) * 4
        b_sc_nrec = fx.Int64(_b_sc_rows * _scale_chunk_dw_b * 4)
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
        nsb = (by_n + wave * (BN // num_waves)) // 32
        # B blockscale: 4 consecutive 32-N super-rows share one 128-N block scale.
        b_sc_base = [((nsb + np) // 4 if _bs_b else (nsb + np)) * _scale_chunk_dw_b for np in range_constexpr(n_pairs)]
        sc_lane = lane_div_16 * 16 + lane_mod_16
        # blockscale drops broadcast dims from the per-lane offset: A drops K_Lane
        # (lane_div_16); B drops K_Lane + N_Lane (all lanes read one word).
        sc_lane_a = lane_mod_16 if _bs_a else sc_lane
        sc_lane_b = fx.Int32(0) if _bs_b else sc_lane

        n_acc = m_chunks * num_acc_n

        def load_b(kt, ni_only=None):
            # buffer_load_dwordx4 into i32[4] frags; fp8 B packs two K0 blocks (lo/hi, 64 K
            # apart, f8f6f4 ABI) into i32[8] — same shuffle as read_a's fp6/fp8 A path.
            ops = []
            ni_range = range_constexpr(num_acc_n) if ni_only is None else [ni_only]
            for ni in ni_range:
                for kh in range_constexpr(k_halves):
                    lo = fx.make_rmem_tensor(4, Int32)
                    fx.copy(b_copy, bq_views[ni][lane_div_16, lane_mod_16, kt, kh, 0, None], lo)
                    if const_expr(b_dtype == "fp4"):
                        ops.append(lo)
                    else:  # fp8: lo ++ hi -> i32[B_NDW]
                        hi = fx.make_rmem_tensor(4, Int32)
                        fx.copy(b_copy, bq_views[ni][lane_div_16, lane_mod_16, kt, kh, 1, None], hi)
                        t = fx.make_rmem_tensor(B_NDW, Int32)
                        t.store(fx.memref_load_vec(lo).shuffle(fx.memref_load_vec(hi), list(range(B_NDW))))
                        ops.append(t)
            return ops

        def load_sc(chunk_kt):
            # (sa, sb) e8m0 words per m-/n-pair for one 256-K chunk (uniform base -> SGPR soffset).
            # blockscale uses smaller per-chunk strides (compact buffer) + broadcast lanes.
            koff_a = chunk_kt * _sc_k0_a
            koff_b = chunk_kt * _sc_k0_b
            sa = [
                Vec(
                    fly.copy_atom_call_ssa(
                        [T.vec(1, T.i32)],
                        bs_copy,
                        sa_flat[
                            None,
                            rocdl.readfirstlane(T.i32, a_sc_base[mp] + koff_a) + sc_lane_a,
                        ],
                    )
                )[0]
                for mp in range_constexpr(m_pairs)
            ]

            if const_expr(BM == 16):
                # A 16-row tile shares a 32-row scale word with its sibling tile. Swap
                # adjacent E8M0 bytes in odd tiles so opsel 0/2 selects rows 16..31.
                odd_m16 = (bx_m & 31) >= 16
                sa = [
                    odd_m16.select(
                        ((v >> 8) & 0x00FF00FF) | ((v << 8) & 0xFF00FF00),
                        v,
                    )
                    for v in sa
                ]

            _bs_b_dedup = _bs_b and (BN // num_waves) <= 128 and (128 % (BN // num_waves) == 0)
            if const_expr(_bs_b_dedup):
                _sb0 = Vec(
                    fly.copy_atom_call_ssa(
                        [T.vec(1, T.i32)],
                        bs_copy,
                        sb_flat[
                            None,
                            rocdl.readfirstlane(T.i32, b_sc_base[0] + koff_b) + sc_lane_b,
                        ],
                    )
                )[0]
                sb = [_sb0 for _ in range_constexpr(n_pairs)]
            else:
                sb = [
                    Vec(
                        fly.copy_atom_call_ssa(
                            [T.vec(1, T.i32)],
                            bs_copy,
                            sb_flat[
                                None,
                                rocdl.readfirstlane(T.i32, b_sc_base[np] + koff_b) + sc_lane_b,
                            ],
                        )
                    )[0]
                    for np in range_constexpr(n_pairs)
                ]
            return sa, sb

        def compute(accs, av, bv, sa_v, sb_v, scale_shift=None, prefetch_b=None):
            # tile_k=128: shift the active 128-K half of the shared 256-K word into the opsel's low bytes.
            if const_expr(scale_shift is not None):
                sa_v = [v.shrui(scale_shift) for v in sa_v]
                sb_v = [v.shrui(scale_shift) for v in sb_v]
            if const_expr(BN < 128):
                _bnsh = ((by_n + wave * (BN // num_waves)) % 32) // 16 * 8
                sb_v = [v.shrui(_bnsh) for v in sb_v]
            # Each scaled MFMA = fx.gemm over rank-1 i32[4] A/B frags, e8m0 word on scale_a=/scale_b=.
            c_frags = [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(n_acc)]
            for idx in range_constexpr(n_acc):
                c_frags[idx].store(Vec(accs[idx]))

            def mma(kh, ni):
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

            next_b = []
            if const_expr(prefetch_b is None):
                # kh OUTERMOST: consecutive MFMAs hit distinct accumulators (dense issue).
                for kh in range_constexpr(k_halves):
                    for ni in range_constexpr(num_acc_n):
                        mma(kh, ni)
            else:
                # Each accumulator still visits kh in reduction order; the next B fragment
                # issues before its corresponding MFMAs.
                for ni in range_constexpr(num_acc_n):
                    # Keep only one N-fragment of the next B tile live alongside the current one.
                    next_b.extend(prefetch_b(ni))
                    for kh in range_constexpr(k_halves):
                        mma(kh, ni)
            for idx in range_constexpr(n_acc):
                accs[idx] = c_frags[idx].load().ir_value()
            return accs, next_b

        def hot_loop_scheduler(gmem_scale=1):
            # Interleave the MFMAs with the tile's vmem + A-LDS loads: preload all hints, then issue MFMAs 1-by-1.
            if const_expr(gmem_scale > 0):
                rocdl.sched_vmem(sched_num_gmem * gmem_scale)
            rocdl.sched_dsrd(sched_num_ds_load)
            for _ in range_constexpr(sched_mfma_total):
                rocdl.sched_mfma(1)
            rocdl.sched_barrier(0)

        accs_init = [fx.full(4, 0.0, Float32).ir_value() for _ in range_constexpr(n_acc)]

        if const_expr(not b_pipe):
            dma_a_to_lds(kt0, fx.Int32(0))
            rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            gpu.barrier()
            for iv, state in range(0, k_tiles_local, 1, init=accs_init):
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
                accs, _ = compute(accs, av, bv, sa_v, sb_v, scale_shift)
                hot_loop_scheduler()
                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)  # drain the A DMA before the barrier
                gpu.barrier()
                results = yield accs
            accs = results
        else:
            n_bregs = num_acc_n * k_halves
            chunk_init = kt0 if tiles_per_chunk == 1 else kt0 >> (tiles_per_chunk - 1)
            b_init = load_b(kt0)
            sa_init, sb_init = load_sc(chunk_init)
            init_state = (
                accs_init
                + [b_init[i].load().ir_value() for i in range_constexpr(n_bregs)]
                + [v.ir_value() for v in sa_init]
                + [v.ir_value() for v in sb_init]
            )

            dma_a_to_lds(kt0, fx.Int32(0))
            if const_expr(pair_k):
                kt1 = kt0 + 1
                b_init1 = load_b(kt1)
                chunk1 = kt1 if tiles_per_chunk == 1 else kt1 >> (tiles_per_chunk - 1)
                sa_init1, sb_init1 = load_sc(chunk1)
                pair_init = (
                    init_state
                    + [b_init1[i].load().ir_value() for i in range_constexpr(n_bregs)]
                    + [v.ir_value() for v in sa_init1]
                    + [v.ir_value() for v in sb_init1]
                )
                dma_a_to_lds(kt1, fx.Int32(1))
            rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            gpu.barrier()
            prefetch_tiles = 0 if pair_k else (k_tiles_local - 1 if BM >= 128 else k_tiles_local)
            for iv, state in range(0, prefetch_tiles, 1, init=init_state):
                accs = list(state[:n_acc])
                bv = []
                for i in range_constexpr(n_bregs):
                    frag = fx.make_rmem_tensor(B_NDW, Int32)
                    frag.store(Vec(state[n_acc + i]))
                    bv.append(frag)
                sa_off = n_acc + n_bregs
                sa_v = [fx.Int32(state[sa_off + mp]) for mp in range_constexpr(m_pairs)]
                sb_off = sa_off + m_pairs
                sb_v = [fx.Int32(state[sb_off + np]) for np in range_constexpr(n_pairs)]
                ivi = fx.Int32(iv)
                cur = ivi & 1
                nxt = (ivi + 1) & 1
                kt = kt0 + ivi  # absolute K-tile for A/B/scale addressing
                nkt = ivi + 1
                # Large tiles peel the tail; small tiles keep the scheduled clamped prefetch.
                pf_kt = kt0 + nkt if BM >= 128 else kt0 + (nkt - nkt // k_tiles_local)
                pf_chunk = pf_kt if tiles_per_chunk == 1 else pf_kt >> (tiles_per_chunk - 1)
                chunk_kt = kt if tiles_per_chunk == 1 else kt >> (tiles_per_chunk - 1)
                scale_shift = None if tiles_per_chunk == 1 else (kt & (tiles_per_chunk - 1)) * 16
                av = read_a(cur)
                next_sa, next_sb = load_sc(pf_chunk)
                dma_a_to_lds(pf_kt, nxt)  # A DMA after scale loads -> overlaps the MFMAs
                accs, next_b = compute(accs, av, bv, sa_v, sb_v, scale_shift, lambda ni: load_b(pf_kt, ni))
                hot_loop_scheduler()
                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)  # drain the A DMA before the barrier
                gpu.barrier()
                results = (
                    yield accs
                    + [next_b[i].load().ir_value() for i in range_constexpr(n_bregs)]
                    + [v.ir_value() for v in next_sa]
                    + [v.ir_value() for v in next_sb]
                )
            accs = list(results[:n_acc])
            if const_expr(BM >= 128):
                # The final large tile is already carried in registers/LDS. Compute it without
                # issuing the clamped first-tile prefetch or the pipeline's final barrier.
                bv = []
                for i in range_constexpr(n_bregs):
                    frag = fx.make_rmem_tensor(B_NDW, Int32)
                    frag.store(Vec(results[n_acc + i]))
                    bv.append(frag)
                sa_off = n_acc + n_bregs
                sa_v = [fx.Int32(results[sa_off + mp]) for mp in range_constexpr(m_pairs)]
                sb_off = sa_off + m_pairs
                sb_v = [fx.Int32(results[sb_off + np]) for np in range_constexpr(n_pairs)]
                last_iv = k_tiles_local - 1
                last_kt = kt0 + last_iv
                scale_shift = None if tiles_per_chunk == 1 else (last_kt & (tiles_per_chunk - 1)) * 16
                av = read_a(fx.Int32(last_iv % 2))
                accs, _ = compute(
                    accs,
                    av,
                    bv,
                    sa_v,
                    sb_v,
                    scale_shift,
                    lambda ni: [bv[ni * k_halves + kh] for kh in range_constexpr(k_halves)],
                )

            if const_expr(pair_k):
                # Four A buffers and two B/scale register sets let one barrier cover two tiles.
                pair1_b = n_acc + n_bregs + m_pairs + n_pairs
                pair1_sa = pair1_b + n_bregs
                pair1_sb = pair1_sa + m_pairs
                peel_pair = (
                    (bool(pair_tail) or (BM == 32 and BN == 64)) and k_tiles_local >= 4 and k_tiles_local % 2 == 0
                )
                pair_main_steps = k_tiles_local // 2 - (1 if peel_pair else 0)
                early_hints = BM == 16 or bool(pair_tail)
                for iv, state in range(0, pair_main_steps, 1, init=pair_init, unroll=4 if pair_tail else 15):
                    accs = list(state[:n_acc])
                    bv0 = []
                    bv1 = []
                    for i in range_constexpr(n_bregs):
                        frag0 = fx.make_rmem_tensor(B_NDW, Int32)
                        frag0.store(Vec(state[n_acc + i]))
                        bv0.append(frag0)
                        frag1 = fx.make_rmem_tensor(B_NDW, Int32)
                        frag1.store(Vec(state[pair1_b + i]))
                        bv1.append(frag1)
                    sa_v0 = [fx.Int32(state[sa_off + mp]) for mp in range_constexpr(m_pairs)]
                    sb_v0 = [fx.Int32(state[sb_off + np]) for np in range_constexpr(n_pairs)]
                    sa_v1 = [fx.Int32(state[pair1_sa + mp]) for mp in range_constexpr(m_pairs)]
                    sb_v1 = [fx.Int32(state[pair1_sb + np]) for np in range_constexpr(n_pairs)]
                    ivi = fx.Int32(iv)
                    parity = ivi & 1
                    kt = kt0 + ivi * 2
                    kt1 = kt + 1
                    cur0 = parity * 2
                    cur1 = cur0 + 1
                    nxt0 = (1 - parity) * 2
                    nxt1 = nxt0 + 1
                    pfi0 = ivi * 2 + 2
                    pfi1 = pfi0 + 1
                    last_i = fx.Int32(k_tiles_local - 1)
                    pfi0 = (pfi0 < last_i).select(pfi0, last_i)
                    pfi1 = (pfi1 < last_i).select(pfi1, last_i)
                    pf0 = kt0 + pfi0
                    pf1 = kt0 + pfi1
                    pfc0 = pf0 if tiles_per_chunk == 1 else pf0 >> (tiles_per_chunk - 1)
                    pfc1 = pf1 if tiles_per_chunk == 1 else pf1 >> (tiles_per_chunk - 1)
                    shift0 = None if tiles_per_chunk == 1 else (kt & (tiles_per_chunk - 1)) * 16
                    shift1 = None if tiles_per_chunk == 1 else (kt1 % tiles_per_chunk) * 16

                    av0 = read_a(cur0)
                    next_sa0, next_sb0 = load_sc(pfc0)
                    next_sa1, next_sb1 = load_sc(pfc1)
                    dma_a_to_lds(pf0, nxt0)
                    dma_a_to_lds(pf1, nxt1)
                    accs, next_b01 = compute(
                        accs,
                        av0,
                        bv0,
                        sa_v0,
                        sb_v0,
                        shift0,
                        lambda ni: load_b(pf0, ni) + load_b(pf1, ni),
                    )
                    next_b0 = next_b01[:n_bregs]
                    next_b1_pre = next_b01[n_bregs:]
                    hot_loop_scheduler(2 if early_hints else 1)
                    av1 = read_a(cur1)
                    accs, next_b1 = compute(
                        accs,
                        av1,
                        bv1,
                        sa_v1,
                        sb_v1,
                        shift1,
                        lambda ni: [next_b1_pre[ni * k_halves + kh] for kh in range_constexpr(k_halves)],
                    )
                    hot_loop_scheduler(0 if early_hints else 1)
                    rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    gpu.barrier()
                    results_pair = (
                        yield accs
                        + [next_b0[i].load().ir_value() for i in range_constexpr(n_bregs)]
                        + [v.ir_value() for v in next_sa0]
                        + [v.ir_value() for v in next_sb0]
                        + [next_b1[i].load().ir_value() for i in range_constexpr(n_bregs)]
                        + [v.ir_value() for v in next_sa1]
                        + [v.ir_value() for v in next_sb1]
                    )
                accs = list(results_pair[:n_acc])
                if const_expr(peel_pair):
                    bv0 = []
                    bv1 = []
                    for i in range_constexpr(n_bregs):
                        frag0 = fx.make_rmem_tensor(B_NDW, Int32)
                        frag0.store(Vec(results_pair[n_acc + i]))
                        bv0.append(frag0)
                        frag1 = fx.make_rmem_tensor(B_NDW, Int32)
                        frag1.store(Vec(results_pair[pair1_b + i]))
                        bv1.append(frag1)
                    sa_v0 = [fx.Int32(results_pair[sa_off + mp]) for mp in range_constexpr(m_pairs)]
                    sb_v0 = [fx.Int32(results_pair[sb_off + np]) for np in range_constexpr(n_pairs)]
                    sa_v1 = [fx.Int32(results_pair[pair1_sa + mp]) for mp in range_constexpr(m_pairs)]
                    sb_v1 = [fx.Int32(results_pair[pair1_sb + np]) for np in range_constexpr(n_pairs)]
                    last_pair = k_tiles_local // 2 - 1
                    cur0 = (last_pair % 2) * 2
                    cur1 = cur0 + 1
                    last_kt = kt0 + k_tiles_local - 2
                    last_kt1 = last_kt + 1
                    shift0 = None if tiles_per_chunk == 1 else (last_kt & (tiles_per_chunk - 1)) * 16
                    shift1 = None if tiles_per_chunk == 1 else (last_kt1 & (tiles_per_chunk - 1)) * 16
                    av0 = read_a(fx.Int32(cur0))
                    accs, _ = compute(
                        accs,
                        av0,
                        bv0,
                        sa_v0,
                        sb_v0,
                        shift0,
                        lambda ni: [bv0[ni * k_halves + kh] for kh in range_constexpr(k_halves)],
                    )
                    hot_loop_scheduler()
                    av1 = read_a(fx.Int32(cur1))
                    accs, _ = compute(
                        accs,
                        av1,
                        bv1,
                        sa_v1,
                        sb_v1,
                        shift1,
                        lambda ni: [bv1[ni * k_halves + kh] for kh in range_constexpr(k_halves)],
                    )
                    hot_loop_scheduler()
                if const_expr(k_tiles_local % 2 == 1):
                    bv = []
                    for i in range_constexpr(n_bregs):
                        frag = fx.make_rmem_tensor(B_NDW, Int32)
                        frag.store(Vec(results_pair[n_acc + i]))
                        bv.append(frag)
                    sa_t = [fx.Int32(results_pair[sa_off + mp]) for mp in range_constexpr(m_pairs)]
                    sb_t = [fx.Int32(results_pair[sb_off + np]) for np in range_constexpr(n_pairs)]
                    last_iv = k_tiles_local - 1
                    last_kt = kt0 + last_iv
                    shift = None if tiles_per_chunk == 1 else (last_kt & (tiles_per_chunk - 1)) * 16
                    av = read_a(fx.Int32(last_iv % 4))
                    accs, _ = compute(
                        accs,
                        av,
                        bv,
                        sa_t,
                        sb_t,
                        shift,
                        lambda ni: [bv[ni * k_halves + kh] for kh in range_constexpr(k_halves)],
                    )

        # Epilogue via fx.copy: a lane owns 4 rows per (mi,ni) accm (row m*16+(l//16)*4+ii, col
        # base+l%16), c_stride apart.
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
                c_bstride = fx.Int64(i32_m) * fx.Int64(N) * 2 if c_batch_stride < 0 else fx.Int64(c_batch_stride)
                c_addr = c_addr + fx.Int64(bz_batch) * c_bstride

        c_tile_addr = c_addr + fx.Int64(bx_m) * fx.Int64(c_stride) * fx.Int64(_ebytes)
        _rows_rem = i32_m - fx.Int32(bx_m)
        _rows_wg = (_rows_rem < fx.Int32(BM)).select(_rows_rem, fx.Int32(BM))
        c_nrec = fx.Int64(_rows_wg) * fx.Int64(c_stride) * fx.Int64(_ebytes)
        c_ptr_ty = fx.PointerType.get(store_elem.ir_type, address_space=fx.AddressSpace.Global, alignment=_ebytes)
        c_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(c_ptr_ty, c_tile_addr), fx.make_layout(1 << 28, 1))),
                max_size=False,
                num_records_bytes=c_nrec,
            ),
            fx.make_layout(1, 1),
        )
        if const_expr(k_batch > 1):
            c_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), store_elem)
        else:
            c_copy = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), store_elem)
        c_rstride = fx.Int32(c_stride)
        col_w = by_n + wave * (BN // num_waves) + lane_mod_16
        for mi in range_constexpr(m_chunks):
            row_local = mi * 16 + lane_div_16 * 4  # relative to this WG (base folded into c_tile_addr)
            for ni in range_constexpr(num_acc_n):
                col = col_w + ni * 16
                acc = Vec(accs[mi * num_acc_n + ni]).to(store_elem)
                for ii in range_constexpr(4):
                    cf = fx.make_rmem_tensor(1, store_elem)
                    cf.store(fx.full(1, acc[ii], store_elem))
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

    Plain fp4 x fp4 launches (tile_k=256, no blockscale, split-K, batching or custom strides)
    treat the 32x128, 64x128 and 64x256 tiles as hints and pick the workgroup geometry from M at
    launch time: small grids need more, narrower workgroups to cover the CUs, and once the grid
    covers the machine a wider N tile cuts B traffic. The 128-row geometries pick their own XCD
    order, so xcd_swizzle applies only to the small-M ones. Every other launch uses the given tile.
    """
    contiguous = all(
        stride < 0
        for stride in (a_row_stride, a_batch_stride, sca_row_stride, sca_batch_stride, c_row_stride, c_batch_stride)
    )
    plain_fp4 = (
        a_dtype == b_dtype == "fp4"
        and batch == 1
        and k_batch == 1
        and blockscale == "none"
        and contiguous
        and tile_k == 256
    )

    def emit(bm, bn, xcd, pair_tail=0, xcd_hw_order=False):
        _emit_gemm(
            arg_c,
            arg_a,
            arg_b,
            arg_scale_a,
            arg_scale_b,
            i32_m,
            i32_n,
            stream,
            N,
            K,
            bm,
            bn,
            tile_k,
            a_dtype,
            out_dtype,
            b_dtype,
            batch,
            a_row_stride,
            a_batch_stride,
            sca_row_stride,
            sca_batch_stride,
            c_row_stride,
            c_batch_stride,
            waves_per_eu,
            xcd,
            k_batch,
            blockscale,
            pair_tail,
            xcd_hw_order,
        )

    if const_expr(plain_fp4 and tile_n == 128 and tile_m in (32, 64)):
        xcd = xcd_swizzle or 1

        @flyc.jit
        def dispatch():
            if i32_m <= fx.Int32(32):
                emit(16, 64, xcd, pair_tail=1)
            elif i32_m <= fx.Int32(64):
                emit(32, 32, xcd, pair_tail=1)
            else:
                emit(32, 64, xcd)

        dispatch()
    elif const_expr(plain_fp4 and (tile_m, tile_n) == (64, 256)):
        if const_expr(N < 2048):
            # A 256-wide N tile leaves too few N tiles below N=2048, whatever the M.
            emit(128, 128, 4, xcd_hw_order=True)
        elif const_expr(K <= 1024):

            @flyc.jit
            def dispatch():
                wg128 = (i32_m + fx.Int32(127)) // fx.Int32(128) * fx.Int32(N // 128)
                if wg128 <= fx.Int32(512):
                    emit(128, 128, 4, xcd_hw_order=True)
                elif wg128 <= fx.Int32(1024):
                    emit(128, 256, 8, xcd_hw_order=True)
                else:
                    # Four or fewer K tiles cannot fill the 128-row pipelines on a grid this large.
                    emit(64, 256, xcd_swizzle)

            dispatch()
        else:

            @flyc.jit
            def dispatch():
                # Up to two waves of 128x128 workgroups on 256 CUs keep the narrower tile.
                if (i32_m + fx.Int32(127)) // fx.Int32(128) * fx.Int32(N // 128) <= fx.Int32(512):
                    emit(128, 128, 4, xcd_hw_order=True)
                else:
                    emit(128, 256, 8, xcd_hw_order=True)

            dispatch()
    else:
        emit(tile_m, tile_n, xcd_swizzle)


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
                        fx.make_layout(1 << 30, 1),
                    )
                ),
                max_size=False,
                num_records_bytes=num_records_bytes,
            )
            return fx.logical_divide(buf, frag_layout)

        in_t = _tiled(tmp, Float32, fx.Int64(slab_dw_i) * fx.Int64(split_k) * 4)
        out_t = _tiled(out, out_elem, fx.Int64(n_out_dw_i) * 4)
        in_copy = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), Float32)  # 2 x f32
        out_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), out_elem)  # 2 x 16b

        if vt < n_out_dw_i:
            acc = fx.full(_REDUCE_VEC, 0.0, Float32)
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
