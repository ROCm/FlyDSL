# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP4 preshuffle GEMM (layout API): MXFP4 (E2M1) A x MXFP4 B with per-32 E8M0
block scales consumed inside a scaled 16x16x128 MFMA (no epilogue scale).

Faithful plain-GEMM port of the proven aiter layout-API MXFP4 kernels
(``mxfp4_moe_layout`` / ``mxfp4_gemm2_v2``). Data layout (CK weight preshuffle via
``shuffle_weight_w4(.,16)`` + ``shuffle_scale_w4``, A packed 2 codes/byte) matches
``tests/kernels/utils/fp4_utils``.

Core mechanism (the unlock vs a raw-intrinsic port):
  * Pre-build the 16 ``(opsel_a, opsel_b)`` ``MFMA_Scale(16,16,128, Float4E2M1FN)``
    atoms (opsel is a TYPE param). cbsz/blgp(=4 for fp4) are inferred from the elem type.
  * One scaled MFMA = ``fx.gemm(atom, c, a, b, c, scale_a=sa, scale_b=sb)`` over
    rank-1 register fragments; the per-32 E8M0 scale word rides ``scale_a=/scale_b=``
    and ``opsel`` selects which of the 4 packed E8M0 bytes (2 K-halves x 2 m|n).
"""

from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm
from flydsl._mlir.dialects import vector as _vector
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import BFloat16, Float4E2M1FN, Float16, Float32, Int8, T
from flydsl.expr.typing import Vector as Vec


def _raw(v):
    """Unwrap an fx value to a raw ir.Value."""
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _scale_mma_atoms():
    """All 16 (opselA, opselB) scaled-MFMA atoms (opsel is a type param -> one atom
    per pair, built once at trace time). cbsz/blgp(=4 fp4) inferred from Float4E2M1FN."""
    return {
        (osa, osb): fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, Float4E2M1FN, opsel_a=osa, opsel_b=osb))
        for osa in range(4)
        for osb in range(4)
    }


def _gemm_mma(atoms, a_frag, b_frag, c_frag, opsel_a, opsel_b, sa, sb):
    """One scaled MFMA via fx.gemm over rank-1 register fragments (-> one MmaAtomCall);
    C accumulates in place (d == c); per-32 e8m0 scales ride scale_a=/scale_b=."""
    fx.gemm(atoms[(opsel_a, opsel_b)], c_frag, a_frag, b_frag, c_frag, scale_a=sa, scale_b=sb)


def _bq_view(arg_bq_addr, row_elems, KH4, k_tiles, k_halves):
    """Layout view over the CK-preshuffled B weight for one N-row tile (i32 units).

    ``row_elems`` = logical N-row; uniform per-(wave) base = readfirstlane(row*KH4)
    (KH4 = K_HALF//4 i32 col stride). i32 strides: klane[0,4)->64, nlane[0,16)->4,
    K_tile->k_halves*256 (one tile = k_halves 128-K blocks), half[0,k_halves)->256,
    kpack4[0,4)->1. Index [lane//16, lane%16, kt, half, None] -> i32<4:1> (16B = 32 fp4)."""
    col_base = rocdl.readfirstlane(T.i32, _raw(row_elems) * fx.Int32(KH4))
    i32_ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=16)
    off_i64 = fx.Int64(arith.ExtUIOp(T.i64, _raw(col_base)).result)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_bq_addr) + off_i64 * fx.Int64(4))
    shape = (4, 16, k_tiles, k_halves, 4)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, (64, 4, k_halves * 256, 256, 1))))
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def _aq_view(arg_aq_addr, bx_m, KH4, k_tiles, k_halves, m_chunks):
    """Layout view over row-major (M, K/2-byte) MXFP4 A -> the 16x16x128 MFMA A
    operand as i32<4:1> (16B = 32 fp4) per (m-chunk, k-half), avoiding sub-byte
    f4 register fragments (which the rmem-SSA promotion can't handle). Index
    ``view[lane%16, lane//16, kt, kh, mi, None]``. i32 strides: row(lane%16)->KH4,
    kgroup(lane//16)->4, kt->32, kh->16, m_chunk->16*KH4, kpack4->1."""
    row_base = rocdl.readfirstlane(T.i32, _raw(bx_m) * fx.Int32(KH4))
    i32_ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=16)
    off_i64 = fx.Int64(arith.ExtUIOp(T.i64, _raw(row_base)).result)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_aq_addr) + off_i64 * fx.Int64(4))
    shape = (16, 4, k_tiles, k_halves, m_chunks, 4)
    stride = (KH4, 4, 32, 16, 16 * KH4, 1)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, stride)))
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def compile_mxfp4_gemm_v2(
    *,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a_dtype: str = "fp4",
    out_dtype: str = "bf16",
    waves_per_eu: Optional[int] = None,
    enable_scheduler: Optional[bool] = None,
    dsrd_preload: int = -1,
    dvmem_preload: int = -1,
    use_async_copy: bool = False,
):
    """Compile MXFP4 (A4W4) / W4A6 preshuffle GEMM (layout API) ->
    fn(C, A, B, scale_a, scale_b, bias, M, N, stream).

    a_dtype="fp4": MXFP4 (E2M1) A, packed 2 codes/byte (16 B = 32 codes per MFMA op).
    a_dtype="fp6": MXFP6 (E2M3) A, FP8-padded packed FP6 -- 32 B per K=32 chunk
        (24 B packed codes + 8 B zero pad); that 32-B chunk is the MFMA A operand
        (cbsz=2, low 24 B used). B is MXFP4 either way.
    B: CK-preshuffled MXFP4 (shuffle_weight_w4(.,16)); scale_a/scale_b: e8m0
    (shuffle_scale_w4); C: (M, N) out_dtype. bias unused (parity).
    """
    if a_dtype not in ("fp4", "fp6"):
        raise ValueError(f"a_dtype must be 'fp4' or 'fp6', got {a_dtype!r}")
    is_fp6 = a_dtype == "fp6"
    BM, BN, BK = tile_m, tile_n, tile_k
    if BK not in (128, 256) or K % BK != 0:
        raise ValueError(f"tile_k must be 128 or 256 dividing K; got tile_k={BK}, K={K}")
    if K % 256 != 0:
        raise ValueError(f"K must be a multiple of 256 (e8m0 scale chunk); got K={K}")
    out_elem = BFloat16 if out_dtype == "bf16" else Float16

    K_HALF = K // 2
    KH4 = K_HALF // 4
    K_TILES = K // BK
    k_halves = BK // 128  # 16x16x128 MFMA k-steps per K-tile
    # e8m0 scale chunks are 256-K granular while B is 128-K granular; a tile_k=128
    # tile is half a scale chunk, so tiles_per_chunk consecutive K-tiles share one
    # scale word (the lower/upper 16b select the 128-K half, like v1's fp4_scale_half).
    tiles_per_chunk = 256 // BK  # 1 for tile_k=256, 2 for tile_k=128
    m_chunks = BM // 16
    num_acc_n = (BN // 4) // 16  # 16-col n-subblocks per wave
    # e8m0 scale strides (i32/dword units), per fp4_utils shuffle_scale_w4.
    _scale_chunk_dw = (K // 32 // 4 // 2) * 64
    _scale_k0_dw = 64

    # LDS A tile (row-major [m][col]): loaded once per K-tile by all 256 threads
    # (cooperative), shared by the 4 N-waves -> removes the 4x redundant A gmem reads
    # that dominate the no-LDS version. fp4 packs 2 codes/byte (BK//2 bytes/K-tile-row);
    # fp6 is FP8-padded 32 B per K=32 chunk (BK bytes/K-tile-row).
    a_row_bytes = K if is_fp6 else K // 2  # A bytes per full M-row
    cbsz_a = 2 if is_fp6 else 4  # MFMA A format: cbsz=2 (E2M3) / cbsz=4 (E2M1)
    A_ROW_B = BK if is_fp6 else BK // 2  # A bytes per row in a K-tile
    A_LDS_B = BM * A_ROW_B  # bytes per LDS A buffer
    n_coop = A_LDS_B // 256 // 16  # 16B cooperative loads per thread

    n_pairs = max(1, num_acc_n // 2)
    m_pairs = max(1, m_chunks // 2)

    # ── Scheduler counts (sched_group_barrier interleave, ported from v1/v2-fp8).
    sched_mfma_total = k_halves * m_chunks * num_acc_n  # MFMAs per loop iter
    sched_num_ds_load = m_chunks * k_halves  # A LDS reads/thread (read_a)
    sched_num_gmem = n_coop + num_acc_n * k_halves + m_pairs + n_pairs  # vmem loads (A coop + B + scales)
    sched_num_a_dswr = 0 if use_async_copy else n_coop  # A LDS writes/thread (none for DMA)

    # Policy: the sched-group-barrier interleave only helps lean (num_acc_n<=2)
    # tiles, where too few MFMAs hide the load latency; on fat tiles it serializes
    # and regresses. When on, preload ALL loads up front (issue every vmem/ds_read
    # before the MFMA stream) — partial spreads measured worse on this loop shape.
    # fp6 reads 2x the A bytes/tile from LDS; the up-front-preload scheduler then
    # over-serializes the longer ds_read stream and regresses even on lean tiles, so
    # leave the compiler's own scheduler to overlap them.
    if enable_scheduler is None:
        enable_scheduler = (num_acc_n <= 2) and not is_fp6
    if dsrd_preload < 0:
        dsrd_preload = sched_num_ds_load
    if dvmem_preload < 0:
        dvmem_preload = sched_num_gmem

    @fx.struct
    class SharedA:
        a0: fx.Array[Int8, A_LDS_B, 16]
        a1: fx.Array[Int8, A_LDS_B, 16]

    @flyc.kernel
    def kernel_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Int64,
        arg_b: fx.Int64,
        arg_scale_a: fx.Int64,
        arg_scale_b: fx.Int64,
        arg_bias: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        tiled_mma: fx.TiledMma,
    ):
        tid = fx.thread_idx.x
        bid_x, bid_y, _ = fx.block_idx
        wave = rocdl.readfirstlane(T.i32, fx.Int32(tid) // fx.Int32(64))
        lane = fx.Int32(tid) % fx.Int32(64)
        lane_div_16 = lane // fx.Int32(16)
        lane_mod_16 = lane % fx.Int32(16)
        bx_m = bid_x * fx.Int32(BM)
        by_n = bid_y * fx.Int32(BN)

        # ── A: cooperative gmem->LDS (row-major [m][col]) then ds_read the MFMA
        # operands (shared across the 4 N-waves -> no 4x redundant A gmem reads).
        a_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
        _i8g = fx.PointerType.get(T.i8, address_space=fx.AddressSpace.Global, alignment=16)
        # Bound A to the ACTUAL M rows (a_row_bytes/row): a block covering rows past M
        # then reads OOB -> 0 (vs faulting on unmapped memory for M < tile_m).
        a_nrec = fx.Int64(i32_m) * fx.Int64(a_row_bytes)
        a_flat = fx.rocdl.make_buffer_tensor(
            fx.Tensor(fx.make_view(fx.inttoptr(_i8g, fx.Int64(arg_a)), fx.make_layout(65536 * a_row_bytes, 1))),
            max_size=False,
            num_records_bytes=a_nrec,
        )
        a_flat_div = fx.logical_divide(a_flat, fx.make_layout(1, 1))
        lds = fx.SharedAllocator().allocate(SharedA).peek()
        _ptr3 = ir.Type.parse("!llvm.ptr<3>")
        lds_base0 = fx.Int32(fx.ptrtoint(lds.a0.ptr))
        lds_db = fx.Int32(fx.ptrtoint(lds.a1.ptr)) - lds_base0  # ping/pong byte stride

        def _base_of(parity):  # parity in {0,1} (runtime) -> LDS base
            return lds_base0 + parity * lds_db

        def _lds_p(base, byte_off):
            return llvm.inttoptr(_ptr3, _raw(fx.Int64(base + byte_off)))

        def coop_load_a(kt, base):  # cooperative gmem A tile -> LDS (row-major)
            base_k_byte = kt * fx.Int32(A_ROW_B)
            for i in range_constexpr(n_coop):
                lin = (fx.Int32(i * 256) + fx.Int32(tid)) * fx.Int32(16)
                row = lin // fx.Int32(A_ROW_B)
                col = lin % fx.Int32(A_ROW_B)
                gmem_byte = (bx_m + row) * fx.Int32(a_row_bytes) + base_k_byte + col
                v = fly.copy_atom_call_ssa([T.vec(4, T.i32)], a_copy, a_flat_div[None, gmem_byte])
                llvm.store(_raw(v), _lds_p(base, lin))

        # ── Async A: direct gmem->LDS DMA (buffer_load_lds), no register stage. Same
        # row-major LDS layout as coop_load_a (lane L -> base + L*16 by hardware), so
        # read_a is unchanged. Issued AFTER the B/scale vmem loads so the in-order
        # vmcnt drain that the MFMAs force on B does NOT also drain this DMA -> it
        # overlaps the MFMAs (then a single s_waitcnt(0) before the barrier).
        if const_expr(use_async_copy):
            dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
            _i8s = fx.PointerType.get(Int8.ir_type, fx.AddressSpace.Shared, 512)
            sA0_i8 = fx.recast_iter(_i8s, lds.a0.ptr)

        def dma_a_to_lds(kt, parity):
            base_off = rocdl.readfirstlane(T.i32, parity * lds_db + wave * fx.Int32(64 * 16))
            lds_ptr = fx.add_offset(sA0_i8, base_off)
            base_k_byte = kt * fx.Int32(A_ROW_B)
            for i in range_constexpr(n_coop):
                if const_expr(i > 0):
                    lds_ptr = fx.add_offset(lds_ptr, fx.Int32(256 * 16))
                lin = (fx.Int32(i * 256) + fx.Int32(tid)) * fx.Int32(16)
                row = lin // fx.Int32(A_ROW_B)
                col = lin % fx.Int32(A_ROW_B)
                gmem_byte = (bx_m + row) * fx.Int32(a_row_bytes) + base_k_byte + col
                dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                src = fx.slice(a_flat_div, (None, gmem_byte))
                fx.copy(dma_atom, src, dst)

        def read_a(base):
            # fp4: each lane's K=128 MFMA A operand is 16 B (32 codes); k-group (lane//16)
            # strides 16 B, the 128-K half strides 64 B. fp6: the operand is one 32-B
            # FP8-padded chunk (32 codes); k-group strides 32 B, the 128-K half 128 B.
            av = []
            for mi in range_constexpr(m_chunks):
                for kh in range_constexpr(k_halves):
                    if const_expr(is_fp6):
                        off = (
                            (fx.Int32(mi * 16) + lane_mod_16) * fx.Int32(A_ROW_B)
                            + fx.Int32(kh * 128)
                            + lane_div_16 * fx.Int32(32)
                        )
                        # Two clean 16 B ds_read_b128 into consecutive regs (low 24 B is
                        # the FP6 operand, top 8 B is the storage zero pad). A single
                        # vec<8xi32> load instead lets the backend coalesce fragment
                        # tails via ds_read2_b64 -> v_mov spam that stalls the MFMAs.
                        lo = llvm.load(T.vec(4, T.i32), _lds_p(base, off))
                        hi = llvm.load(T.vec(4, T.i32), _lds_p(base, off + fx.Int32(16)))
                        av.append(_vector.shuffle(lo, hi, [0, 1, 2, 3, 4, 5, 6, 7]))
                    else:
                        off = (
                            (fx.Int32(mi * 16) + lane_mod_16) * fx.Int32(A_ROW_B)
                            + fx.Int32(kh * 64)
                            + lane_div_16 * fx.Int32(16)
                        )
                        av.append(llvm.load(T.vec(4, T.i32), _lds_p(base, off)))
            return av

        n_col_base = by_n + wave * fx.Int32(BN // 4)
        bq_views = [
            _bq_view(arg_b, n_col_base + fx.Int32(ni * 16), KH4, K_TILES, k_halves) for ni in range_constexpr(num_acc_n)
        ]
        b_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
        bs_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        # e8m0 scales: flat buffers based at the ALLOCATION START, bounded to the real
        # size, so a block covering rows past M reads OOB -> 0. (The earlier per-m-pair
        # base-pointer advance put the base itself past the allocation for M < tile_m,
        # which num_records can't protect -> HIP illegal access.) The m-pair/n-pair +
        # lane + chunk offset is folded into the access (i32/dword units).
        _i32g = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
        _sc_layout = fx.make_layout(1 << 28, 1)
        a_sc_nrec = fx.Int64((i32_m + fx.Int32(31)) // fx.Int32(32)) * fx.Int64(_scale_chunk_dw) * fx.Int64(4)
        b_sc_nrec = fx.Int64((N // 32) * _scale_chunk_dw * 4)
        sa_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(_i32g, fx.Int64(arg_scale_a)), _sc_layout)),
                max_size=False,
                num_records_bytes=a_sc_nrec,
            ),
            fx.make_layout(1, 1),
        )
        sb_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(_i32g, fx.Int64(arg_scale_b)), _sc_layout)),
                max_size=False,
                num_records_bytes=b_sc_nrec,
            ),
            fx.make_layout(1, 1),
        )
        a_sc_base = [
            (bx_m // fx.Int32(32) + fx.Int32(mp)) * fx.Int32(_scale_chunk_dw) for mp in range_constexpr(m_pairs)
        ]
        nsb = by_n // fx.Int32(32) + wave * fx.Int32(BN // 128)
        b_sc_base = [(nsb + fx.Int32(np)) * fx.Int32(_scale_chunk_dw) for np in range_constexpr(n_pairs)]
        sc_lane = lane_div_16 * fx.Int32(16) + lane_mod_16

        # ── Accumulators + 2-stage software pipeline: prefetch tile k+1's A/B/scale
        # operands (carried as loop state) while the MFMAs compute tile k, hiding
        # global-load latency. Operands carried UNPADDED (vec<4xi32>/<1xi32>) to keep
        # VGPR pressure down; padded to vec<8xi32> only at the MFMA.
        mfma_ty = T.vec(4, T.f32)
        n_acc = m_chunks * num_acc_n

        def _pad8(v4):
            # fp4 MFMA reads only the low 16B; widen vec<4xi32>->vec<8xi32> with a
            # single shuffle (upper half is don't-care) instead of an 8-elem rebuild,
            # so almost no VALU is interspersed with the MFMAs.
            return _vector.shuffle(v4, v4, [0, 1, 2, 3, 4, 5, 6, 7])

        def load_b(kt):
            # B weight packs for K-tile kt (gmem, not wave-redundant).
            ops = []
            for ni in range_constexpr(num_acc_n):
                for kh in range_constexpr(k_halves):
                    ops.append(
                        fly.copy_atom_call_ssa(
                            [T.vec(4, T.i32)], b_copy, bq_views[ni][lane_div_16, lane_mod_16, kt, kh, None]
                        )
                    )
            return ops

        def load_sc(chunk_kt):
            # e8m0 scales for one 256-K scale chunk (shared by tiles_per_chunk K-tiles;
            # the 128-K half is selected at the MFMA via the runtime scale_shift). Returns
            # (sa, sb) i32 scale words per m-pair / n-pair.
            koff = chunk_kt * fx.Int32(_scale_k0_dw)
            # readfirstlane the uniform (m-pair base + chunk) part so it lands in an SGPR
            # soffset; only sc_lane is the per-lane voffset. Base stays at arg_scale, so
            # num_records still protects OOB (no pointer advance).
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

        def compute(accs, av, bv, sa_v, sb_v, scale_shift=None):
            # tile_k=128: two 128-K tiles share one 256-K scale word; logical-shift the
            # word so the active 128-K half lands in the low bytes opsel (kh*2+im, kh=0)
            # reads. tile_k=256 (scale_shift=None) keeps both halves (kh in {0,1}).
            if const_expr(scale_shift is not None):
                sh = _raw(scale_shift)
                sa_v = [arith.shrui(_raw(v), sh) for v in sa_v]
                sb_v = [arith.shrui(_raw(v), sh) for v in sb_v]
            # kh OUTERMOST so consecutive MFMAs write distinct accumulators (independent
            # -> dense MFMA issue); the per-acc kh accumulation dependency is spaced by
            # all (mi,ni) MFMAs, hiding the MFMA latency. Widen operands ONCE per kh
            # (reused across the mi/ni grid) to keep VALU out of the MFMA stream.
            for kh in range_constexpr(k_halves):
                # fp6 A is already a 32-B (vec<8xi32>) operand; fp4 A is 16 B widened to
                # vec<8xi32> (upper half don't-care). B (MXFP4) is always widened.
                if const_expr(is_fp6):
                    av8 = [av[mi * k_halves + kh] for mi in range_constexpr(m_chunks)]
                else:
                    av8 = [_pad8(av[mi * k_halves + kh]) for mi in range_constexpr(m_chunks)]
                bv8 = [_pad8(bv[ni * k_halves + kh]) for ni in range_constexpr(num_acc_n)]
                for ni in range_constexpr(num_acc_n):
                    np_i, in_b = ni // 2, ni % 2
                    for mi in range_constexpr(m_chunks):
                        mp_i, im = mi // 2, mi % 2
                        idx = mi * num_acc_n + ni
                        accs[idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            mfma_ty,
                            [
                                av8[mi],
                                bv8[ni],
                                accs[idx],
                                cbsz_a,
                                4,
                                kh * 2 + im,
                                sa_v[mp_i],
                                kh * 2 + in_b,
                                sb_v[np_i],
                            ],
                        )
            return accs

        # ── Scheduler hints: constrain the post-RA scheduler to interleave the
        # MFMAs with the A-coop/B/scale vmem loads, A LDS reads (read_a), and A
        # LDS writes (coop_load_a) — spreads load latency across the MFMA stream.
        def build_scheduler(numer, denom):
            if const_expr(denom <= 0):
                return []
            if const_expr(numer <= 0):
                return [0] * denom
            out = []
            prev = 0
            for i in range_constexpr(denom):
                cur = ((i + 1) * numer + (denom - 1)) // denom
                out.append(cur - prev)
                prev = cur
            return out

        def hot_loop_scheduler():
            mfma_total = sched_mfma_total
            dswr_tail = min(sched_num_a_dswr, mfma_total)
            dsrd_preload_eff = min(int(dsrd_preload), sched_num_ds_load)
            dvmem_preload_eff = min(int(dvmem_preload), sched_num_gmem)
            vmem_remaining = sched_num_gmem - dvmem_preload_eff
            dsrd_remaining = sched_num_ds_load - dsrd_preload_eff
            if const_expr(0 < vmem_remaining < mfma_total):
                vmem_schedule = build_scheduler(vmem_remaining, vmem_remaining) + [0] * (mfma_total - vmem_remaining)
            else:
                vmem_schedule = build_scheduler(vmem_remaining, mfma_total)
            dsrd_schedule = build_scheduler(dsrd_remaining, mfma_total)
            dswr_start = max(mfma_total - dswr_tail - 2, 0)
            last_dsrd_mfma_idx = -1
            for sched_idx in range_constexpr(mfma_total):
                if const_expr(dsrd_schedule[sched_idx]):
                    last_dsrd_mfma_idx = sched_idx
            dswr_start = max(dswr_start, last_dsrd_mfma_idx + 1)
            idx_ds_read = dsrd_preload_eff
            idx_gmem_load = dvmem_preload_eff
            idx_ds_write = 0
            if const_expr(dvmem_preload_eff):
                rocdl.sched_vmem(dvmem_preload_eff)
            if const_expr(dsrd_preload_eff):
                rocdl.sched_dsrd(dsrd_preload_eff)
            for mfma_idx in range_constexpr(mfma_total):
                rocdl.sched_mfma(1)
                n_dsrd = dsrd_schedule[mfma_idx]
                if const_expr(n_dsrd and (idx_ds_read < sched_num_ds_load)):
                    if const_expr(idx_ds_read + n_dsrd > sched_num_ds_load):
                        n_dsrd = sched_num_ds_load - idx_ds_read
                    if const_expr(n_dsrd):
                        rocdl.sched_dsrd(n_dsrd)
                        idx_ds_read += n_dsrd
                n_vmem = vmem_schedule[mfma_idx]
                if const_expr(n_vmem and (idx_gmem_load < sched_num_gmem)):
                    if const_expr(idx_gmem_load + n_vmem > sched_num_gmem):
                        n_vmem = sched_num_gmem - idx_gmem_load
                    if const_expr(n_vmem):
                        rocdl.sched_vmem(n_vmem)
                        idx_gmem_load += n_vmem
                if const_expr((idx_ds_write < dswr_tail) and (mfma_idx >= dswr_start)):
                    rocdl.sched_dswr(1)
                    idx_ds_write += 1
            if const_expr(idx_ds_write < sched_num_a_dswr):
                rocdl.sched_dswr(sched_num_a_dswr - idx_ds_write)
            rocdl.sched_barrier(0)

        accs_init = [Vec.filled(4, 0.0, Float32).ir_value() for _ in range_constexpr(n_acc)]

        # Double-buffered cooperative LDS-A: prefetch tile iv+1's A into the OTHER LDS
        # buffer while the MFMAs compute tile iv (A from LDS, B/scales per-tile). At our
        # 3-wave occupancy the B/scale latency is already hidden, so register-prefetching
        # B (v1's 2-wave lever) only adds VGPR pressure and regresses the peak.
        if const_expr(use_async_copy):
            dma_a_to_lds(fx.Int32(0), fx.Int32(0))
            rocdl.s_waitcnt(0)
        else:
            coop_load_a(fx.Int32(0), _base_of(fx.Int32(0)))
        gpu.barrier()
        # 1 tile/iter ping-pong. tile_k=256: one K-tile = one 256-K scale chunk. tile_k=128:
        # two consecutive K-tiles share one 256-K chunk (chunk_kt = kt // tiles_per_chunk);
        # the active 128-K half is selected at the MFMA via the runtime scale_shift. (A
        # 2-tiles/iter variant with compile-time halves measured slower: the shared scale
        # regs lengthen VGPR live ranges and drop occupancy.)
        for iv, state in range(fx.Index(0), fx.Index(K_TILES), fx.Index(1), init=accs_init):
            accs = list(state)
            kt = fx.Int32(iv)
            cur = kt % fx.Int32(2)
            nxt = (kt + fx.Int32(1)) % fx.Int32(2)
            nkt = kt + fx.Int32(1)
            pf_kt = nkt - nkt // fx.Int32(K_TILES)  # clamp last-iter prefetch to K_TILES-1
            chunk_kt = kt if tiles_per_chunk == 1 else kt // fx.Int32(tiles_per_chunk)
            scale_shift = None if tiles_per_chunk == 1 else (kt % fx.Int32(tiles_per_chunk)) * fx.Int32(16)
            if const_expr(not use_async_copy):
                coop_load_a(pf_kt, _base_of(nxt))  # prefetch A tile iv+1 -> LDS
            av = read_a(_base_of(cur))
            bv = load_b(kt)
            sa_v, sb_v = load_sc(chunk_kt)
            if const_expr(use_async_copy):
                dma_a_to_lds(pf_kt, nxt)  # A DMA AFTER B/scale loads -> overlaps the MFMAs
            accs = compute(accs, av, bv, sa_v, sb_v, scale_shift)  # overlaps the A prefetch
            if const_expr(enable_scheduler):
                hot_loop_scheduler()
            if const_expr(use_async_copy):
                rocdl.s_waitcnt(0)  # drain the A DMA before the barrier
            gpu.barrier()
            results = yield accs
        accs = results

        # ── Epilogue: manual C store whose (row,col) matches the B/scale n-layout
        # (MFMA 16x16 C: lane l -> col = base + l%16, rows = m*16 + (l//16)*4 + ii).
        # No per-token scale (MX scale already folded into the accumulator).
        # Bound C to the ACTUAL M rows (out_elem is 2 bytes): when M < tile_m (or M not
        # a multiple of tile_m) the block covers padding rows whose stores must be dropped
        # by the descriptor, else they write past the allocation (HIP illegal access).
        c_nrec = fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(2)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=False, num_records_bytes=c_nrec)
        col_w = by_n + wave * fx.Int32(BN // 4) + lane_mod_16
        for mi in range_constexpr(m_chunks):
            row_m = bx_m + fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4)
            for ni in range_constexpr(num_acc_n):
                col = col_w + fx.Int32(ni * 16)
                acc = Vec(accs[mi * num_acc_n + ni])
                for ii in range_constexpr(4):
                    val = acc[ii].to(out_elem)
                    off = (row_m + fx.Int32(ii)) * fx.Int32(N) + col
                    buffer_ops.buffer_store(val.ir_value(), c_rsrc, off)

    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        from flydsl.compiler.kernel_function import CompilationContext

        CompilationContext.get_current()
        a_addr = fx.Int64(fx.ptrtoint(fx.get_iter(arg_a)))
        b_addr = fx.Int64(fx.ptrtoint(fx.get_iter(arg_b)))
        sa_addr = fx.Int64(fx.ptrtoint(fx.get_iter(arg_scale_a)))
        sb_addr = fx.Int64(fx.ptrtoint(fx.get_iter(arg_scale_b)))
        mma_atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, Float4E2M1FN))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((1, 4, 1), (0, 1, 0)),
            fx.make_tile(None, None, fx.make_layout((32, 4), (1, 32))),
        )
        M_max = 65536
        arg_c_2d = fx.Tensor(fx.make_view(fx.get_iter(arg_c), fx.make_layout((M_max, N), (N, 1))))
        gx = (i32_m + (BM - 1)) // BM
        gy = i32_n // BN
        kernel_gemm(
            arg_c_2d,
            a_addr,
            b_addr,
            sa_addr,
            sb_addr,
            arg_bias,
            i32_m,
            i32_n,
            tiled_mma,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu},
        ).launch(grid=(gx, gy, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm
