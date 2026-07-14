# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""8-wave a8w4 GEMM (fp8 E4M3 A x preshuffled MXFP4 B, per-1x32 e8m0 scales, bf16 out; gfx950).

Structural template = fp8_gemm_8wave (block=512, 256x256 tile, 8 waves in a 2x4 wave
grid so BM=256 needs only 32 accumulators/wave). a8w4 semantics = mxfp4_preshuffle:
  * A = fp8 codes, staged in LDS; the f8f6f4 A-fragment is "split-16@64" (two b128 halves
    64 K apart, packed to i32[8]).
  * B = preshuffled MXFP4 (E2M1), streamed gmem->register via buffer_load_dwordx4.
  * Scales = per-1x32 e8m0 (uint8) for A and B, 256-K granular (two 128-K MFMA steps share
    one word: hi/lo 16b halves via scale_shift).
  * MFMA = MFMA_Scale(16,16,128, Float8E4M3FN, Float4E2M1FN, opsel_a=, opsel_b=).
A streams gmem->LDS via double-buffered async DMA (buffer_load_lds) with an XOR LDS swizzle
to kill 4-way bank conflicts, then ds_read_b128 into the A fragment.

Hot-loop pipeline: the next-tile A gmem->LDS DMA is split into per-slice thunks and
interleaved *between* the MFMA sub-groups (source order) rather than issued as one front-loaded
burst -- so the DMA runs in the shadow of this tile's MFMAs. This interleave (vs the old
front-load-then-sched-hint pattern) lifted 8192^3 from ~2.78 to ~3.05 TF. Accumulators are
VGPR-resident on the fx.gemm scaled path (the ``v_mfma_scale_f32_16x16x128_f8f6f4`` inline-asm
form is rejected by the backend for its scale operands, so AGPR pinning is not usable here).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import (
    BFloat16,
    Float4E2M1FN,
    Float8E4M3FN,
    Float16,
    Float32,
    Int8,
    Int32,
    T,
)
from flydsl.expr.typing import Vector as Vec


def _scale_mma_atoms():
    """16 (opsel_a, opsel_b) scaled-MFMA atoms; A=fp8, B=fp4."""
    return {
        (osa, osb): fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, Float8E4M3FN, Float4E2M1FN, opsel_a=osa, opsel_b=osb)
        )
        for osa in range(4)
        for osb in range(4)
    }


def _bq_view(arg_bq_addr, row_elems, KH4, k_tiles, k_halves):
    """Preshuffled B view for one N-row tile; index [l//16, l%16, kt, half, None] -> i32[4]."""
    col_base = rocdl.readfirstlane(T.i32, row_elems * KH4)
    i32_ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=16)
    off_i64 = fx.Int64(col_base)
    base_iter = fx.inttoptr(i32_ptr_ty, arg_bq_addr + off_i64 * fx.Int64(4))
    shape = (4, 16, k_tiles, k_halves, 4)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, (64, 4, k_halves * 256, 256, 1))))
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def compile_mxfp4_8w(
    *,
    K: int,
    N: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    BLOCK_K: int = 256,
    out_dtype: str = "bf16",
    waves_per_eu=None,
    wave_grid=(2, 4),
    use_xcd=False,
):
    BM, BN, BK = BLOCK_M, BLOCK_N, BLOCK_K
    assert BM % 128 == 0 and BN % 256 == 0 and BK % 128 == 0
    assert K % 256 == 0 and K % BK == 0
    out_elem = BFloat16 if out_dtype == "bf16" else Float16

    # ---- Wave grid: WAVE_M (M) x WAVE_N (N). Each wave owns BM/WAVE_M rows x BN/WAVE_N cols.
    WAVE_M = wave_grid[0]
    WAVE_N = wave_grid[1]
    assert WAVE_M * WAVE_N == 8
    WM = BM // WAVE_M  # rows per wave (M direction) = 128
    WN = BN // WAVE_N  # cols per wave (N direction) = 64
    m_chunks = WM // 16  # 16-row M sub-tiles per wave = 8
    num_acc_n = WN // 16  # 16-col N sub-tiles per wave = 4
    n_acc = m_chunks * num_acc_n  # accumulators/wave = 32

    # ---- A LDS geometry (fp8 codes, 1 byte/code), row-major [m][k_byte].
    A_ROW_B = BK  # bytes of one A row per K-tile (fp8 = K bytes)
    A_LDS_B = BM * A_ROW_B  # LDS A buffer bytes (whole BM shared by all 8 waves)
    A_ROW_I32 = A_ROW_B // 4
    k_blk16 = A_ROW_B // 16  # XOR swizzle period (16B blocks per row)

    # fp8 A fragment: split-16@64. Each half = i32[4]; two halves 64 K apart -> i32[8].
    A_KH_I32 = 32  # i32 stride between consecutive 16x16x128 K-halves in a row
    A_GK_I32 = 4  # i32 stride between the two lane groups (lane//16) within a half
    A_HI_OFF = 16  # i32 offset from lo half to hi half (64 K apart)
    A_NDW = 8

    K_HALF = K // 2
    KH4 = K_HALF // 4
    K_TILES = K // BK
    k_halves = BK // 128  # 16x16x128 MFMA K-steps per K-tile

    # e8m0 scales: 256-K granular, MFMA 128-K -> tiles_per_chunk K-tiles share a word.
    tiles_per_chunk = 256 // BK  # 1 for BK=256, 2 for BK=128
    _scale_chunk_dw = (K // 32 // 4 // 2) * 64  # e8m0 stride (dwords), per shuffle_scale_w4
    _scale_k0_dw = 64

    # Per-wave m-/n-pairs (two 16-blocks share one e8m0 word, N_Pack=2 / M pairing).
    n_pairs = max(1, num_acc_n // 2)
    m_pairs = max(1, m_chunks // 2)

    # A gmem->LDS cooperative 16B loads per thread (block=512 -> 512 threads).
    BLOCK_THREADS = 512
    n_coop = A_LDS_B // 16 // BLOCK_THREADS

    a_row_bytes = K  # fp8, 1 byte/code

    @fx.struct
    class SharedA:
        a0: fx.Array[Int8, A_LDS_B, 16]
        a1: fx.Array[Int8, A_LDS_B, 16]

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel_gemm(
        arg_c: fx.Int64,
        arg_a: fx.Int64,
        arg_b: fx.Int64,
        arg_scale_a: fx.Int64,
        arg_scale_b: fx.Int64,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        scale_atoms = _scale_mma_atoms()

        tid = fx.Int32(fx.thread_idx.x)
        bid_x, bid_y, _bid_z = fx.block_idx
        wave = rocdl.readfirstlane(T.i32, tid // 64)
        wave_m = rocdl.readfirstlane(T.i32, wave // WAVE_N)  # 0..1
        wave_n = rocdl.readfirstlane(T.i32, wave % WAVE_N)  # 0..3
        lane = tid % 64
        lane_div_16 = lane // 16
        lane_mod_16 = lane % 16

        # XCD-friendly block remap over the linearized grid (L2 locality on large shapes).
        if const_expr(use_xcd):
            NUM_XCDS = 8
            n_pid_m = (i32_m + (BM - 1)) // BM
            n_pid_n = i32_n // BN
            num_wg = n_pid_m * n_pid_n
            wgid = bid_x * n_pid_n + bid_y
            intra_xcd = wgid // NUM_XCDS
            xcd = wgid % NUM_XCDS
            wgid_remap = xcd * (num_wg // NUM_XCDS) + intra_xcd
            rm = wgid_remap // n_pid_n
            rn = wgid_remap % n_pid_n
            use_simple = (num_wg < NUM_XCDS) | (num_wg % NUM_XCDS != 0)
            pid_m = arith.select(use_simple, bid_x, rm)
            pid_n = arith.select(use_simple, bid_y, rn)
        else:
            pid_m = bid_x
            pid_n = bid_y
        bx_m = pid_m * BM
        by_n = pid_n * BN
        wm_row = wave_m * WM  # this wave's M base within the block tile
        wn_col = wave_n * WN  # this wave's N base within the block tile

        a_rstride = fx.Int32(a_row_bytes)

        # ---- A source view (bounded to real M rows; ragged OOB reads 0).
        _i8g = fx.PointerType.get(T.i8, address_space=fx.AddressSpace.Global, alignment=16)
        a_nrec = fx.Int64(i32_m) * fx.Int64(a_row_bytes)
        a_flat = fx.rocdl.make_buffer_tensor(
            fx.Tensor(fx.make_view(fx.inttoptr(_i8g, arg_a), fx.make_layout(65536 * a_row_bytes, 1))),
            max_size=False,
            num_records_bytes=a_nrec,
        )
        a_flat_div = fx.logical_divide(a_flat, fx.make_layout(1, 1))

        lds = fx.SharedAllocator().allocate(SharedA).peek()
        sA0_i32 = fx.recast_iter(Int32, lds.a0.ptr)
        lds_db = fx.Int32(fx.ptrtoint(lds.a1.ptr)) - fx.Int32(fx.ptrtoint(lds.a0.ptr))
        lds_db_i32 = lds_db // 4
        lds_copy = fx.make_copy_atom(fx.UniversalCopy128b(), Int32)
        dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        _i8s = fx.PointerType.get(Int8.ir_type, fx.AddressSpace.Shared, 512)
        sA0_i8 = fx.recast_iter(_i8s, lds.a0.ptr)

        def _iter_of(parity):
            return fx.add_offset(sA0_i32, parity * lds_db_i32)

        def _lds_view(base_iter, off_i32):
            return fx.make_view(fx.add_offset(base_iter, off_i32), fx.make_layout(4, 1))

        # ---- Async A: gmem->LDS DMA. 512 threads cooperatively load the whole BM x BK A tile.
        def _dma_a_slice(kt, parity, i):
            base_off = rocdl.readfirstlane(T.i32, parity * lds_db)
            lds_ptr = fx.add_offset(sA0_i8, base_off)
            base_k_byte = kt * A_ROW_B
            lin = (i * BLOCK_THREADS + tid) * 16
            row = lin // A_ROW_B
            col = lin % A_ROW_B
            col_swz = col ^ ((row % k_blk16) * 16)
            gmem_byte = (bx_m + row) * a_rstride + base_k_byte + col_swz
            lds_lin = row * A_ROW_B + col  # store un-swizzled position in LDS row-major
            dst = fx.make_view(fx.add_offset(lds_ptr, fx.Int32(lds_lin)), fx.make_layout(1, 1))
            src = fx.slice(a_flat_div, (None, gmem_byte))
            fx.copy(dma_atom, src, dst)

        def dma_a_to_lds(kt, parity):
            for i in range_constexpr(n_coop):
                _dma_a_slice(kt, parity, i)

        def _read16(base_iter, off_i32):
            t = fx.make_rmem_tensor(4, Int32)
            fx.copy(lds_copy, _lds_view(base_iter, off_i32), t)
            return t

        # ---- Read A fragment for this wave's WM rows, ONE K-half: split-16@64 fp8, packed i32[8].
        def read_a_one(base_iter, mi, kh):
            row = wm_row + mi * 16 + lane_mod_16
            row_base = row * A_ROW_I32
            lo_blk = kh * (A_KH_I32 // 4) + lane_div_16 * (A_GK_I32 // 4)
            off = row_base + (lo_blk ^ (row % k_blk16)) * 4
            hi_off = row_base + ((lo_blk + A_HI_OFF // 4) ^ (row % k_blk16)) * 4
            lo = Vec(fx.memref_load_vec(_read16(base_iter, off)))
            hi = Vec(fx.memref_load_vec(_read16(base_iter, hi_off)))
            t = fx.make_rmem_tensor(A_NDW, Int32)
            t.store(lo.shuffle(hi, list(range(A_NDW))))
            return t

        def read_a_kh(base_iter, kh):
            return [read_a_one(base_iter, mi, kh) for mi in range_constexpr(m_chunks)]

        # ---- B: preshuffled MXFP4 streamed gmem->register (fp4 is tiny; DMA-interleaved).
        n_col_base = by_n + wn_col
        bq_views = [_bq_view(arg_b, n_col_base + ni * 16, KH4, K_TILES, k_halves) for ni in range_constexpr(num_acc_n)]
        b_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
        bs_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        # ---- e8m0 scale buffers.
        _i32g = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
        _sc_layout = fx.make_layout(1 << 28, 1)
        _a_sc_chunks = (i32_m + 31) // 32
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
        # This wave's M scale chunks (32-row granular) and N scale chunks (32-col granular).
        a_sc_base = [((bx_m + wm_row) // 32 + mp) * fx.Int32(_scale_chunk_dw) for mp in range_constexpr(m_pairs)]
        nsb = (by_n + wn_col) // 32
        b_sc_base = [(nsb + np) * _scale_chunk_dw for np in range_constexpr(n_pairs)]
        sc_lane = lane_div_16 * 16 + lane_mod_16

        def load_b_kh(kt, kh):
            ops = []
            for ni in range_constexpr(num_acc_n):
                bf = fx.make_rmem_tensor(4, Int32)
                fx.copy_atom_call(b_copy, bq_views[ni][lane_div_16, lane_mod_16, kt, kh, None], bf)
                ops.append(bf)
            return ops

        def load_sc(chunk_kt):
            koff = chunk_kt * _scale_k0_dw
            sa = [
                Vec(
                    fly.copy_atom_call_ssa(
                        [T.vec(1, T.i32)],
                        bs_copy,
                        sa_flat[None, rocdl.readfirstlane(T.i32, a_sc_base[mp] + koff) + sc_lane],
                    )
                )[0]
                for mp in range_constexpr(m_pairs)
            ]
            sb = [
                Vec(
                    fly.copy_atom_call_ssa(
                        [T.vec(1, T.i32)],
                        bs_copy,
                        sb_flat[None, rocdl.readfirstlane(T.i32, b_sc_base[np] + koff) + sc_lane],
                    )
                )[0]
                for np in range_constexpr(n_pairs)
            ]
            return sa, sb

        def _mfma_one(c_frags, av_mi, bv_ni, sa_v, sb_v, kh, mi, ni):
            np_i, in_b = ni // 2, ni % 2
            mp_i, im = mi // 2, mi % 2
            cf = c_frags[mi * num_acc_n + ni]
            fx.gemm(
                scale_atoms[(kh * 2 + im, kh * 2 + in_b)],
                cf,
                av_mi,
                bv_ni,
                cf,
                scale_a=sa_v[mp_i],
                scale_b=sb_v[np_i],
            )

        def compute(base_iter, kt, accs, sa_v, sb_v, scale_shift, prefetch):
            # Per-K-half fused reads (A from LDS, B from gmem) + fx.gemm scaled MFMA. The
            # next-tile A-DMA prefetch thunks are interleaved between the MFMA groups (source
            # order) so the DMA runs in the shadow of this tile's MFMAs.
            if const_expr(scale_shift is not None):
                sa_v = [v.shrui(scale_shift) for v in sa_v]
                sb_v = [v.shrui(scale_shift) for v in sb_v]
            c_frags = [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(n_acc)]
            for idx in range_constexpr(n_acc):
                c_frags[idx].store(Vec(accs[idx]))

            n_groups = k_halves * m_chunks
            pf_groups = [[] for _ in range_constexpr(n_groups)]
            for pi in range_constexpr(len(prefetch)):
                pf_groups[pi % n_groups].append(prefetch[pi])

            for kh in range_constexpr(k_halves):
                av = read_a_kh(base_iter, kh)
                bv = load_b_kh(kt, kh)
                for mi in range_constexpr(m_chunks):
                    for thunk in pf_groups[kh * m_chunks + mi]:
                        thunk()
                    for ni in range_constexpr(num_acc_n):
                        _mfma_one(c_frags, av[mi], bv[ni], sa_v, sb_v, kh, mi, ni)

            for idx in range_constexpr(n_acc):
                accs[idx] = c_frags[idx].load().ir_value()
            return accs

        accs_init = [Vec.filled(4, 0.0, Float32).ir_value() for _ in range_constexpr(n_acc)]

        # Double-buffered LDS-A: prefetch tile iv+1 into the other buffer while MFMAs compute
        # tile iv. B streams gmem->registers (fp4 is tiny), DMA-interleaved with the MFMAs.
        dma_a_to_lds(fx.Int32(0), fx.Int32(0))
        rocdl.s_waitcnt(0)
        gpu.barrier()
        for iv, state in range(fx.Index(0), fx.Index(K_TILES), fx.Index(1), init=accs_init):
            accs = list(state)
            kt = fx.Int32(iv)
            nxt = (kt + 1) % 2
            cur = kt % 2
            nkt = kt + 1
            pf_kt = nkt - nkt // K_TILES
            chunk_kt = kt if tiles_per_chunk == 1 else kt // tiles_per_chunk
            scale_shift = None if tiles_per_chunk == 1 else (kt % tiles_per_chunk) * 16
            base_iter = _iter_of(cur)
            sa_v, sb_v = load_sc(chunk_kt)
            # Next-tile A DMA issued as per-group thunks interleaved inside compute (source-order
            # overlap with this tile's MFMAs), instead of one front-loaded burst.
            prefetch = [(lambda i=i: _dma_a_slice(pf_kt, nxt, i)) for i in range_constexpr(n_coop)]
            accs = compute(base_iter, kt, accs, sa_v, sb_v, scale_shift, prefetch)
            rocdl.s_waitcnt(0)  # drain the next-tile A DMA before the barrier / next-tile reads
            gpu.barrier()
            results = yield accs
        accs = results

        # ---- Epilogue. Each lane owns 4 rows of each (mi,ni) accumulator.
        c_stride = N
        c_nrec = fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(2)
        c_ptr_ty = fx.PointerType.get(out_elem.ir_type, address_space=fx.AddressSpace.Global, alignment=2)
        c_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(c_ptr_ty, arg_c), fx.make_layout(1 << 28, 1))),
                max_size=False,
                num_records_bytes=c_nrec,
            ),
            fx.make_layout(1, 1),
        )
        c_copy = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_elem)
        c_rstride = fx.Int32(c_stride)
        col_w = by_n + wn_col + lane_mod_16
        for mi in range_constexpr(m_chunks):
            row_m = bx_m + wm_row + mi * 16 + lane_div_16 * 4
            for ni in range_constexpr(num_acc_n):
                col = col_w + ni * 16
                acc = Vec(accs[mi * num_acc_n + ni]).to(out_elem)
                for ii in range_constexpr(4):
                    cf = fx.make_rmem_tensor(1, out_elem)
                    cf.store(Vec.from_elements([acc[ii]], out_elem))
                    off = (row_m + ii) * c_rstride + col
                    fx.copy(c_copy, cf, c_flat[None, off])

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
    ):
        c_addr = fx.Int64(fx.ptrtoint(arg_c))
        a_addr = fx.Int64(fx.ptrtoint(arg_a))
        b_addr = fx.Int64(fx.ptrtoint(arg_b))
        sa_addr = fx.Int64(fx.ptrtoint(arg_scale_a))
        sb_addr = fx.Int64(fx.ptrtoint(arg_scale_b))
        wpe = waves_per_eu if (waves_per_eu is not None and waves_per_eu > 0) else None
        gx = (i32_m + (BM - 1)) // BM
        gy = i32_n // BN
        kernel_gemm(
            c_addr,
            a_addr,
            b_addr,
            sa_addr,
            sb_addr,
            i32_m,
            i32_n,
            value_attrs={"rocdl.waves_per_eu": wpe, "rocdl.flat_work_group_size": "512,512"},
        ).launch(grid=(gx, gy, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_gemm
