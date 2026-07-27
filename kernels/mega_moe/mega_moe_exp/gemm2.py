# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""GEMM2 compute for fused MegaMoE v2 stage2."""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import _to_raw as _raw
from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import (
    BFloat16,
    Float4E2M1FN,
    Float8E4M3FN,
    Float32,
    Int8,
    Int32,
    T,
)
from flydsl.expr.typing import Vector as Vec

from .mxfp4_gemm_common import _fabs_f32 as fabs_f32
from .mxfp4_gemm_common import _lds_swizzle_mask as lds_swizzle_mask
from .mxfp4_gemm_common import (
    flat_buffer_view,
    global_typed_ptr,
    kBS_stride_k0_dw,
    kStages,
    lds_dma_atom_128,
    lds_dma_dst,
    lds_swizzle_mask_f8,
    lds_typed_ptr,
    lds_vec_load,
)


def bq_view(
    arg_bq,
    row_elems,
    KH4,
    K_TILES_TOTAL,
    K_HALVES,
    num_records_bytes=None,
):
    """View one preshuffled B N-tile as i32<4:1>, optionally bounded to real K."""
    col_base = rocdl.readfirstlane(T.i32, _raw(row_elems) * fx.Int32(KH4))
    i32_ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=16)
    off_i64 = fx.Int64(col_base)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_bq) + off_i64 * fx.Int64(4))
    # i32 strides: klane 64, nlane 4, K-tile K_HALVES*256, half 256, kpack4 1.
    shape = (4, 16, K_TILES_TOTAL, K_HALVES, 4)
    view = fx.Tensor(
        fx.make_view(
            base_iter,
            fx.make_layout(shape, (64, 4, K_HALVES * 256, 256, 1)),
        )
    )
    if num_records_bytes is not None:
        return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def scale_view(arg_scale, base_dw, K_TILES_TOTAL, k0_stride_dw=64, num_records_bytes=None):
    """View one e8m0 scale word, optionally bounded to the real buffer extent."""
    base_dw = rocdl.readfirstlane(T.i32, _raw(base_dw))
    i32_ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
    off_i64 = fx.Int64(base_dw)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_scale) + off_i64 * fx.Int64(4))
    shape = (4, 16, K_TILES_TOTAL, 1)
    stride = (16, 1, k0_stride_dw, 1)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, stride)))
    if num_records_bytes is not None:
        return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def scale_mma_atoms(a_dtype):
    """16 (opselA,opselB) scaled-MFMA atoms; A elem is fp8/fp4, B is fp4."""
    elem_a = Float8E4M3FN if a_dtype == "fp8" else Float4E2M1FN
    return {
        (osa, osb): fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, elem_a, Float4E2M1FN, opsel_a=osa, opsel_b=osb)
        )
        for osa in range(4)
        for osb in range(4)
    }


def mma_one_j(
    J,
    in_b,
    sa,
    sb,
    bq_frags_kt,
    a_frags,
    c_frags,
    atoms,
    i0=0,
    single_rg=False,
    rg_off=0,
    k_halves=2,
):
    """Run one scaled-MFMA cluster over a 32-row A-scale group."""
    if const_expr(single_rg):
        steps = tuple((2 * k + rg_off, k, i0, bq_frags_kt[J][k]) for k in range(k_halves))
    else:
        steps = tuple((2 * k + im, k, i0 + im, bq_frags_kt[J][k]) for k in range(k_halves) for im in range(2))
    for osa, k, i, bJ in steps:
        osb = 2 * k + in_b
        fx.gemm(
            atoms[(osa, osb)],
            c_frags[i][J],
            a_frags[i][k],
            bJ,
            c_frags[i][J],
            scale_a=sa,
            scale_b=sb,
        )


def issue_a_load_lds_dt(
    arg_aq,
    s_aq_base,
    slot,
    kt,
    m_row,
    wave,
    lane,
    is_f8,
    KH_TILE_A,
    K_BYTES,
    BM=32,
):
    """Load one A tile through a tile-local descriptor so allocations may span the 4 GiB buffer ABI."""
    lanes_per_row = KH_TILE_A // 16  # 8 (fp4) / 16 (fp8)
    rows_per_call = 64 // lanes_per_row  # 8 (fp4) / 4 (fp8)
    a_lane_row = lane // lanes_per_row
    rows_per_wave = BM // 4  # rows each wave loads (BM32: 8, BM64: 16)
    # BM16 fp4: partial-wave round-robin (waves 2,3 re-load, harmless); BM>=32 byte-identical per-wave blocks.
    partial_wave_gather = rows_per_wave < rows_per_call
    if const_expr(partial_wave_gather):
        n_gather_calls = BM // rows_per_call
        gather_base_row = (wave % fx.Int32(n_gather_calls)) * rows_per_call
        n_row_groups = 1
    else:
        gather_base_row = wave * rows_per_wave
        n_row_groups = rows_per_wave // rows_per_call
    lane_col = (lane % lanes_per_row) * 16
    base_i32 = s_aq_base
    atom = lds_dma_atom_128()
    tile_base = fx.Int64(arg_aq) + fx.Int64(m_row) * fx.Int64(K_BYTES)
    src = flat_buffer_view(
        tile_base,
        None,
        T.i32,
        align=16,
        elem_bytes=4,
        fold=False,
        num_records_bytes=fx.Int64(BM) * fx.Int64(K_BYTES),
    )
    for g in range_constexpr(n_row_groups):
        lds_row = gather_base_row + g * rows_per_call
        mask = (
            lds_swizzle_mask_f8(lds_row + a_lane_row) if const_expr(is_f8) else lds_swizzle_mask(lds_row + a_lane_row)
        )
        tile_row = lds_row + a_lane_row
        voffset = (lane_col ^ mask) + tile_row * K_BYTES
        off = fx.Int32(slot * (BM * KH_TILE_A)) + lds_row * KH_TILE_A
        v_e = (voffset + kt * KH_TILE_A) // 4  # per-lane i32-elem index
        fx.copy(atom, src[v_e, None], lds_dma_dst(base_i32, off, elem_ty=T.i32, align=16))


@flyc.jit
def gemm2_compute_v2(
    lds_base_i32,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_aq,
    i32_max_m_blocks,
    bx_i32,
    lane,
    wave,
    i32_inter,
    i32_hidden,
    i32_kpad,
    i32_npad,
    *,
    BM,
    BN=256,
    BK=256,
    use_nt,
    INTER_MAX,
    aStages,
    a_dtype,
    has_pad=False,
    SBM=None,
    g2_bhoist=True,
    g2_ascale_pf=True,
    expert_offset=0,
):
    """Run the GEMM2 K-loop and return accumulators for the selected epilogue."""
    # K-loop knobs control two-stage B, hoisted B, and one-tile-ahead A-scale prefetch.
    # SBM (sort padding unit) >= BM (compute tile); SBM==BM default byte-identical.
    if SBM is None:
        SBM = BM
    kMChunks = BM // 16  # 16-row MFMA row-groups
    kHalves = BK // 128  # 16x16x128 MFMA K-steps per K-tile
    tilesPerScaleChunk = 256 // BK  # K-tiles sharing one 256-K E8M0 word
    numAccN = (BN // 4) // 16  # 16-column MFMA subblocks per wave
    nPairs = max(1, numAccN // 2)  # one B-scale per two 16-column subblocks
    # BM16: single 16-row block owning a 32-row scale chunk (chunk==m_block_idx, rg0-only).
    is_bm16 = BM < 32
    rg_off = 0
    kScaleSubBlocks = max(1, kMChunks // 2)
    is_f8_a = a_dtype == "fp8"  # only the A path differs
    a_pack = 1 if is_f8_a else 2
    KH_TILE_A = BK // a_pack
    slot_bytes = BM * KH_TILE_A
    # Contraction K = inter_dim runtime (i32_inter); INTER_MAX caps compile-time view/fragment bounds.
    K_rt = fx.Int32(i32_inter)
    K_BYTES = K_rt // fx.Int32(a_pack)  # A row stride bytes (runtime)
    kc_rt = K_rt // fx.Int32(256)  # (K//32)//4//2
    K_TILES_RT = K_rt // fx.Int32(BK)  # runtime K-tile trip count
    kAS_per_chunk_dw = kc_rt * fx.Int32(64)
    kBS_stride_n0_dw = kc_rt * fx.Int32(64)
    # N_OUT = model_dim/hidden is the gemm2 output N dim; runtime via i32_hidden (no K-loop dependency).
    N_OUT_rt = fx.Int32(i32_hidden)
    kbs_per_expert_dw = (N_OUT_rt // fx.Int32(32)) * kBS_stride_n0_dw  # (N_OUT//16//2)*stride
    num_n_blocks = N_OUT_rt // fx.Int32(BN)
    KH4 = K_rt // fx.Int32(8)  # i32 col stride (= K_HALF//4)
    K_TILES_MAX = INTER_MAX // BK
    K_SCALE_CHUNKS_MAX = INTER_MAX // 256

    # Padded shapes bound B to real K and zero weight tiles beyond real N.
    bq_num_records = None
    N_real = None
    if const_expr(has_pad):
        K_real = K_rt - fx.Int32(i32_kpad)
        halves_real = (K_real + fx.Int32(127)) // fx.Int32(128)
        bq_num_records = halves_real * fx.Int32(1024)
        N_real = N_OUT_rt - fx.Int32(i32_npad)

    # Map each compute block to its SBM-padded expert metadata row.
    m_block_idx = bx_i32 // num_n_blocks
    n_block_idx = bx_i32 - m_block_idx * num_n_blocks
    eids_ptr = global_typed_ptr(arg_eids, T.i32)
    if const_expr(SBM == BM):
        e = rocdl.readfirstlane(T.i32, _raw(eids_ptr[m_block_idx]))
        m_row = m_block_idx * BM
    else:
        m_row = m_block_idx * BM
        e = rocdl.readfirstlane(T.i32, _raw(eids_ptr[m_row // fx.Int32(SBM)]))
    if const_expr(expert_offset != 0):
        e = e - fx.Int32(expert_offset)

    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16

    s_aq_base = lds_base_i32
    mma_atoms = scale_mma_atoms(a_dtype)

    # A activation: global->LDS DMA (issue_a_load_lds), then LDS->reg ds-read (issue_a_ds_read).
    A_NDW = 8 if is_f8_a else 4  # fp8 packs two 128-K halves -> i32<8:1>; fp4 -> i32<4:1>
    a_frags = [[fx.make_rmem_tensor(A_NDW, Int32) for _ in range_constexpr(kHalves)] for _ in range_constexpr(kMChunks)]

    def issue_a_load_lds(slot, kt):
        issue_a_load_lds_dt(
            arg_aq,
            s_aq_base,
            slot,
            kt,
            m_row,
            wave,
            lane,
            is_f8_a,
            KH_TILE_A,
            K_BYTES,
            BM=BM,
        )

    def issue_a_ds_read(slot):
        # A ds-read for one slot into a_frags: fp8 -> i32<8:1> (two 128-K halves), fp4 -> i32<4:1>.
        for k in range_constexpr(kHalves):
            for i in range_constexpr(kMChunks):
                lds_row = lane_mod_16 + i * 16
                row_off = fx.Int32(slot * slot_bytes) + lds_row * KH_TILE_A
                if const_expr(is_f8_a):
                    mask = lds_swizzle_mask_f8(lane_mod_16)
                    col0 = lane_div_16 * 16 + k * 128
                    col_lo = col0 ^ mask
                    col_hi = (col0 + 64) ^ mask
                    lo = Vec(
                        lds_vec_load(
                            s_aq_base,
                            row_off + col_lo,
                            Vec.make_type(2, fx.Int64),
                            fx.Int64,
                            align=16,
                        )
                    )
                    hi = Vec(
                        lds_vec_load(
                            s_aq_base,
                            row_off + col_hi,
                            Vec.make_type(2, fx.Int64),
                            fx.Int64,
                            align=16,
                        )
                    )
                    a64 = Vec.from_elements([lo[0], lo[1], hi[0], hi[1]], fx.Int64)
                    a_frags[i][k].store(a64.bitcast(fx.Int32))
                else:
                    mask = lds_swizzle_mask(lane_mod_16)
                    lds_col = (lane_div_16 * 16 + k * 64) ^ mask
                    vec = lds_vec_load(
                        s_aq_base,
                        row_off + lds_col,
                        Vec.make_type(4, fx.Int32),
                        fx.Int32,
                        align=16,
                    )
                    a_frags[i][k].store(Vec(vec))

    # The shared e8m0 scale layout bounds each A-scale view to its remaining bytes.
    sc_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

    asc_per_mb = fx.Int32(kScaleSubBlocks) * kAS_per_chunk_dw * fx.Int32(4)
    asc_num = fx.Int64(i32_max_m_blocks) * fx.Int64(asc_per_mb)
    scale_chunk0 = m_block_idx if const_expr(is_bm16) else m_row // 32

    def make_ascale_view(sub):
        base_dw = (scale_chunk0 + fx.Int32(sub)) * kAS_per_chunk_dw
        nrec = asc_num - fx.Int64(base_dw) * fx.Int64(4)
        return scale_view(
            arg_ascale,
            base_dw,
            K_SCALE_CHUNKS_MAX,
            k0_stride_dw=64,
            num_records_bytes=nrec,
        )

    ascale_views = [make_ascale_view(sub) for sub in range_constexpr(kScaleSubBlocks)]
    sc_frag_tmpl = ascale_views[0][0, 0, 0, None]  # i32<1:1> (one e8m0 word)

    def load_a_scale_tile(kt):
        # One i32 A-scale register per 32-row chunk (kScaleSubBlocks).
        chunk_kt = kt if const_expr(tilesPerScaleChunk == 1) else kt // fx.Int32(tilesPerScaleChunk)
        out = []
        for sub in range_constexpr(kScaleSubBlocks):
            saf = fx.make_fragment_like(sc_frag_tmpl)
            fx.copy(
                sc_copy_atom,
                ascale_views[sub][lane_div_16, lane_mod_16, chunk_kt, None],
                saf,
            )
            out.append(_raw(Vec(saf.load())[0]))
        return out

    # B-weight + B-scale: global->register, streamed per K-tile (not LDS-staged).
    # Use the explicit buffer-load path so use_nt reaches the ISA cache-policy operand.
    bq_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_bq)

    def make_bq_view(j):
        col = n_block_idx * BN + wave * (BN // 4) + j * 16
        nrec = bq_num_records
        if const_expr(has_pad):
            # N-skip: fully-pad-N tile (col >= 16-aligned N_real) -> 0 records so weight loads OOB -> 0.
            nrec = (col < N_real).select(bq_num_records, fx.Int32(0))
        return bq_view(
            arg_bq,
            e * N_OUT_rt + col,
            KH4,
            K_TILES_MAX,
            kHalves,
            num_records_bytes=nrec,
        )

    bq_base_dw = [
        rocdl.readfirstlane(
            T.i32,
            (e * N_OUT_rt + n_block_idx * BN + wave * (BN // 4) + j * 16) * KH4,
        )
        for j in range_constexpr(numAccN)
    ]

    mni_base = n_block_idx * (BN // 16 // 2) + wave * (BN // 64 // 2)
    bscale_views = [
        scale_view(
            arg_bscale,
            e * kbs_per_expert_dw + (mni_base + mw) * kBS_stride_n0_dw,
            K_SCALE_CHUNKS_MAX,
            k0_stride_dw=kBS_stride_k0_dw,
        )
        for mw in range_constexpr(nPairs)
    ]

    frag_tmpl = make_bq_view(0)[0, 0, 0, 0, None]  # i32<4:1> (16B = 32 fp4)
    # B-scale word template shares the A-scale layout (sc_frag_tmpl).

    def issue_b_load_into(bqf, bsf, kt_rt):
        # Issue B-weight + B-scale vmem loads for K-tile kt_rt into the given (per-stage) fragments.
        for j in range_constexpr(numAccN):
            for half in range_constexpr(kHalves):
                bq_off_dw = (
                    bq_base_dw[j]
                    + lane_div_16 * fx.Int32(64)
                    + lane_mod_16 * fx.Int32(4)
                    + kt_rt * fx.Int32(kHalves * 256)
                    + fx.Int32(half * 256)
                )
                load_mask = None
                if const_expr(has_pad):
                    col = n_block_idx * BN + wave * (BN // 4) + j * 16
                    load_mask = (col < N_real) & (kt_rt * fx.Int32(kHalves) + fx.Int32(half) < halves_real)
                bq_vec = buffer_ops.buffer_load(
                    bq_rsrc,
                    bq_off_dw,
                    vec_width=4,
                    dtype=T.i32,
                    mask=load_mask,
                    cache_modifier=2 if use_nt else 0,
                )
                bqf[j][half].store(Vec(bq_vec))
        chunk_kt = kt_rt if const_expr(tilesPerScaleChunk == 1) else kt_rt // fx.Int32(tilesPerScaleChunk)
        for mw in range_constexpr(nPairs):
            fx.copy(
                sc_copy_atom,
                bscale_views[mw][lane_div_16, lane_mod_16, chunk_kt, None],
                bsf[mw],
            )

    def stream_b_tile(kt_rt):
        # Fresh per-iter fragments (B streamed, not register-resident) then issue_b_load_into.
        bqf = [[fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)] for _ in range_constexpr(numAccN)]
        bsf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(nPairs)]
        issue_b_load_into(bqf, bsf, kt_rt)
        return bqf, bsf

    # Scaled-MFMA clusters over the loaded A / B / scale fragments.
    def shift_scale_word(scale, kt_rt):
        if const_expr(tilesPerScaleChunk == 1):
            return scale
        scale_shift = (kt_rt % fx.Int32(tilesPerScaleChunk)) * fx.Int32(16)
        return _raw(fx.Int32(scale).shrui(scale_shift))

    def mfma_cluster(bqf, bsf, sa, kt_rt):
        # opsel (no gate/up split): mni=J//2, in_b=J%2; sa is a per-32-row-chunk list.
        sa = [shift_scale_word(sa[sub], kt_rt) for sub in range_constexpr(kScaleSubBlocks)]
        sb_words = [shift_scale_word(_raw(Vec(bsf[mni].load())[0]), kt_rt) for mni in range_constexpr(nPairs)]
        for J in range_constexpr(numAccN):
            mni, in_b = J // 2, J % 2
            sb = sb_words[mni]
            if const_expr(is_bm16):
                mma_one_j(
                    J,
                    in_b,
                    sa[0],
                    sb,
                    bqf,
                    a_frags,
                    c_frags,
                    mma_atoms,
                    i0=0,
                    single_rg=True,
                    rg_off=rg_off,
                    k_halves=kHalves,
                )
                continue
            for sub in range_constexpr(kScaleSubBlocks):
                mma_one_j(
                    J,
                    in_b,
                    sa[sub],
                    sb,
                    bqf,
                    a_frags,
                    c_frags,
                    mma_atoms,
                    i0=2 * sub,
                    k_halves=kHalves,
                )

    # C accumulator: register fragments, zeroed then accumulated in place; (un)packed to K-loop carry.
    zero4 = Vec.filled(4, 0.0, Float32)
    c_frags = [[fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(numAccN)] for _ in range_constexpr(kMChunks)]
    for i in range_constexpr(kMChunks):
        for J in range_constexpr(numAccN):
            c_frags[i][J].store(zero4)

    def load_c_carry():
        return [c_frags[i][J].load() for i in range(kMChunks) for J in range(numAccN)]

    def store_c_carry(state):
        n = 0
        for i in range_constexpr(kMChunks):
            for J in range_constexpr(numAccN):
                c_frags[i][J].store(state[n])
                n += 1
        return n

    if const_expr(BM == 64 and BN == 256):
        # BM64/BN256 uses the 1-stage B path unconditionally; do not depend on env knobs.
        for kt_iv, state in range(
            fx.Int32(0),
            K_TILES_RT,
            fx.Int32(1),
            init=load_c_carry(),
        ):
            store_c_carry(state)
            kt_rt = fx.Int32(kt_iv)
            gpu.barrier()
            issue_a_ds_read(kt_rt % fx.Int32(aStages))
            nxt = kt_rt + fx.Int32(kStages)
            if nxt < K_TILES_RT:
                issue_a_load_lds(nxt % fx.Int32(aStages), nxt)
            bqf, bsf = stream_b_tile(kt_rt)
            sa = load_a_scale_tile(kt_rt)
            mfma_cluster(bqf, bsf, sa, kt_rt)
            results = yield load_c_carry()
        store_c_carry(results)
    else:
        # 2-stage B pipeline: consume carried "current" B, prefetch next tile into the same fragments via scf.for state.
        cur_bqf = [
            [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)] for _ in range_constexpr(numAccN)
        ]
        cur_bsf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(nPairs)]
        nxt_bqf = [
            [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)] for _ in range_constexpr(numAccN)
        ]
        nxt_bsf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(nPairs)]
        # g2_ascale_pf: carry the A-scale through scf.for state, same rotating-buffer model as B.
        cur_saf = nxt_saf = None
        if const_expr(g2_ascale_pf):
            cur_saf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(kScaleSubBlocks)]
            nxt_saf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(kScaleSubBlocks)]

        def load_b_carry():
            # Flat CURRENT (to-consume) B-weight, B-scale, then (opt) A-scale values.
            out = []
            for j in range_constexpr(numAccN):
                for half in range_constexpr(kHalves):
                    out.append(cur_bqf[j][half].load())
            for mw in range_constexpr(nPairs):
                out.append(cur_bsf[mw].load())
            if const_expr(g2_ascale_pf):
                for sub in range_constexpr(kScaleSubBlocks):
                    out.append(cur_saf[sub].load())
            return out

        def store_b_carry(state, base):
            n = base
            for j in range_constexpr(numAccN):
                for half in range_constexpr(kHalves):
                    cur_bqf[j][half].store(state[n])
                    n += 1
            for mw in range_constexpr(nPairs):
                cur_bsf[mw].store(state[n])
                n += 1
            if const_expr(g2_ascale_pf):
                for sub in range_constexpr(kScaleSubBlocks):
                    cur_saf[sub].store(state[n])
                    n += 1
            return n

        def rotate_b_carry():
            # Yield the PREFETCHED (next-tile) values -> become "current" next iteration.
            out = []
            for j in range_constexpr(numAccN):
                for half in range_constexpr(kHalves):
                    out.append(nxt_bqf[j][half].load())
            for mw in range_constexpr(nPairs):
                out.append(nxt_bsf[mw].load())
            if const_expr(g2_ascale_pf):
                for sub in range_constexpr(kScaleSubBlocks):
                    out.append(nxt_saf[sub].load())
            return out

        def issue_a_scale_load_into(saf, kt_rt):
            # A-scale vmem load(s) for K-tile kt_rt into the given (per-stage) fragment(s).
            sa = load_a_scale_tile(kt_rt)
            for sub in range_constexpr(kScaleSubBlocks):
                saf[sub].store(sa[sub])

        def load_carry():
            return load_c_carry() + load_b_carry()

        def store_carry(state):
            base = store_c_carry(state)
            store_b_carry(state, base)

        def yield_carry():
            return load_c_carry() + rotate_b_carry()

        # Prologue: prefetch tile 0's B/B-scale into "current" (VALUES enter via init=load_carry()).
        issue_b_load_into(cur_bqf, cur_bsf, fx.Int32(0))
        if const_expr(g2_ascale_pf):
            issue_a_scale_load_into(cur_saf, fx.Int32(0))
        rocdl.sched_barrier(0)

        def prefetch_next_b(kt_rt):
            # Prefetch NEXT tile's B; if none, copy current through (rotate_b_carry state, unused after loop).
            nxt_b = kt_rt + fx.Int32(1)
            if nxt_b < K_TILES_RT:
                issue_b_load_into(nxt_bqf, nxt_bsf, nxt_b)
                if const_expr(g2_ascale_pf):
                    issue_a_scale_load_into(nxt_saf, nxt_b)
            else:
                for j in range_constexpr(numAccN):
                    for half in range_constexpr(kHalves):
                        nxt_bqf[j][half].store(cur_bqf[j][half].load())
                for mw in range_constexpr(nPairs):
                    nxt_bsf[mw].store(cur_bsf[mw].load())
                if const_expr(g2_ascale_pf):
                    for sub in range_constexpr(kScaleSubBlocks):
                        nxt_saf[sub].store(cur_saf[sub].load())

        for kt_iv, state in range(
            fx.Int32(0),
            K_TILES_RT,
            fx.Int32(1),
            init=load_carry(),
        ):
            store_carry(state)
            kt_rt = fx.Int32(kt_iv)
            if const_expr(g2_bhoist):
                prefetch_next_b(kt_rt)
            gpu.barrier()
            issue_a_ds_read(kt_rt % fx.Int32(aStages))
            nxt_a = kt_rt + fx.Int32(kStages)
            if nxt_a < K_TILES_RT:
                issue_a_load_lds(nxt_a % fx.Int32(aStages), nxt_a)
            # A-scale from the prefetch carry (g2_ascale_pf) or loaded synchronously here.
            if const_expr(g2_ascale_pf):
                sa = [_raw(Vec(cur_saf[sub].load())[0]) for sub in range_constexpr(kScaleSubBlocks)]
            else:
                sa = load_a_scale_tile(kt_rt)
            if const_expr(not g2_bhoist):
                prefetch_next_b(kt_rt)
            # Fence the MFMA chain from the B vmem loads (next-tile loads ride ahead of compute).
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            mfma_cluster(cur_bqf, cur_bsf, sa, kt_rt)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            results = yield yield_carry()
        store_carry(results)

    # Load the C fragments (fp8/fp4 unified onto the same fx.gemm path) and hand them to the epilog.
    accm_vecs = [[c_frags[i][J].load() for J in range(numAccN)] for i in range(kMChunks)]
    return accm_vecs, m_row, n_block_idx, N_OUT_rt


@flyc.jit
def gemm2_body_v2(
    lds_base_i32,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_stids,
    arg_sweights,
    i32_M,
    i32_max_m_blocks,
    arg_out,
    bx_i32,
    lane,
    wave,
    arg_aq,
    i32_inter,
    i32_hidden,
    i32_kpad,
    i32_npad,
    *,
    BM,
    BN=256,
    BK=256,
    use_nt,
    INTER_MAX,
    aStages,
    a_dtype,
    use_reduce=False,
    topk=1,
    has_pad=False,
    SBM=None,
    g2_bhoist=True,
    g2_ascale_pf=True,
    g2_bf16_lds=False,
    route_out_fp8=False,
):
    """Standalone gemm2 (compute + atomic/reduce local-write epilog); byte-identical to aiter."""
    if SBM is None:
        SBM = BM
    accm_vecs, m_row, n_block_idx, N_OUT_rt = gemm2_compute_v2(
        lds_base_i32,
        arg_ascale,
        arg_bq,
        arg_bscale,
        arg_eids,
        arg_aq,
        i32_max_m_blocks,
        bx_i32,
        lane,
        wave,
        i32_inter,
        i32_hidden,
        i32_kpad,
        i32_npad,
        BM=BM,
        BN=BN,
        BK=BK,
        use_nt=use_nt,
        INTER_MAX=INTER_MAX,
        aStages=aStages,
        a_dtype=a_dtype,
        has_pad=has_pad,
        SBM=SBM,
        g2_bhoist=g2_bhoist,
        g2_ascale_pf=g2_ascale_pf,
    )
    atomic_bf16_epilog(
        lds_base_i32,
        accm_vecs,
        arg_out,
        arg_stids,
        arg_sweights,
        m_row,
        n_block_idx,
        wave,
        lane,
        i32_M,
        BM,
        N_OUT_rt,
        BN=BN,
        use_reduce=use_reduce,
        topk=topk,
        SBM=SBM,
        g2_bf16_lds=g2_bf16_lds,
        route_out_fp8=route_out_fp8,
    )


# ---- Atomic bf16 epilogue (shared store path; gemm2 down-proj) ----
def atomic_bf16_epilog(
    lds_acc_base,
    accm,
    arg_out,
    arg_stids,
    arg_sweights,
    m_row,
    n_block_idx,
    wave,
    lane,
    i32_M,
    BM,
    N_OUT,
    *,
    BN=256,
    use_reduce=False,
    topk=1,
    SBM=None,
    g2_bf16_lds=False,
    route_out_fp8=False,
):
    if SBM is None:
        SBM = BM
    kMChunks = BM // 16
    M_REPS = BM // 8  # BM32: 4, BM16: 2
    numAccN = (BN // 4) // 16  # 16-column MFMA subblocks per wave
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    lds_base_fptr = lds_typed_ptr(lds_acc_base, T.f32)
    lds_base_bf16 = lds_typed_ptr(lds_acc_base, T.bf16, align=2) if const_expr(g2_bf16_lds) else None

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // 32
    n_lane = tx_i32 % 32
    store_vec = 2
    store_group_n = 32 * store_vec
    col_start = n_lane * store_vec
    wave_n = BN // 4

    def flat_buffer(arg, elem_ty, align):
        ptr = global_typed_ptr(arg, elem_ty, align=align)
        view = fx.Tensor(fx.make_view(ptr, fx.make_layout((1, 1), (1, 1))))
        return fx.rocdl.make_buffer_tensor(view, max_size=True)

    stids = flat_buffer(arg_stids, T.i32, 4)
    sweights = flat_buffer(arg_sweights, T.f32, 4)
    out_bf16 = flat_buffer(arg_out, T.bf16, 4)
    out_i8 = flat_buffer(arg_out, T.i8, 4)

    load_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), Int32)
    load_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), Float32)
    store_bf16x2 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), BFloat16)
    atomic_bf16x2 = fx.make_copy_atom(fx.rocdl.BufferAtomicPkAdd(BFloat16), BFloat16)
    store_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), Int32)
    store_i8 = fx.make_copy_atom(fx.rocdl.BufferCopy8b(2), Int8)

    def load_scalar(atom, src, index, elem_ty):
        frag = fx.make_rmem_tensor(1, elem_ty)
        fx.copy(atom, src[None, index], frag)
        return Vec(frag.load())[0]

    # Prefetch sorted_token_ids / sorted_weights (invariant); latency overlaps stores+barriers.
    packed = []
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + mr * 8 + m_lane
        packed.append(load_scalar(load_i32, stids, sorted_pos, Int32))
        weight.append(load_scalar(load_f32, sweights, sorted_pos, Float32))

    # pre-store fence+barrier (HIP run_one __syncthreads() before the epilog).
    gpu.barrier()

    # write accm -> lds_acc cshuffle. f32 path: scalar f32 stores (weight applied on readback).
    if const_expr(g2_bf16_lds):
        for i in range_constexpr(kMChunks):
            row_base = fx.Int32(i * 16) + lane_div_16 * 4
            w_row = [load_scalar(load_f32, sweights, m_row + row_base + v, Float32) for v in range_constexpr(4)]
            for J in range_constexpr(numAccN):
                col = wave * wave_n + J * 16 + lane_mod_16
                vec = Vec(accm[i][J])
                for v in range_constexpr(4):
                    idx = (row_base + v) * BN + col
                    lds_base_bf16[idx] = fx.BFloat16(fx.Float32(vec[v]) * fx.Float32(w_row[v]))
    else:
        for i in range_constexpr(kMChunks):
            row_base = fx.Int32(i * 16) + lane_div_16 * 4
            for J in range_constexpr(numAccN):
                col = wave * wave_n + J * 16 + lane_mod_16
                vec = Vec(accm[i][J])
                for v in range_constexpr(4):
                    idx = (row_base + v) * BN + col
                    lds_base_fptr[idx] = fx.Float32(vec[v])

    gpu.barrier()

    def store_one_mr(mr):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if const_expr(use_reduce):
            # Use i64 for large token*topk output offsets.
            out_row = fx.Int64(token_id * fx.Int32(topk) + (packed[mr] >> fx.Int32(24)))
            if const_expr(route_out_fp8):
                row_base_addr = out_row * fx.Int64(N_OUT + (N_OUT // fx.Int32(8)))
            else:
                row_base_addr = out_row * fx.Int64(N_OUT) + fx.Int64(n_block_idx * BN + col_start)
        else:
            out_row = token_id
            row_base_addr = out_row * N_OUT + n_block_idx * BN + col_start
        if const_expr(use_reduce and route_out_fp8):
            route_vec = 8
            route_group_n = 32 * route_vec
            for rg in range_constexpr((BN + route_group_n - 1) // route_group_n):
                col_lane8 = rg * route_group_n + n_lane * fx.Int32(route_vec)

                def store_route_group(col_lane8):
                    col_g0 = n_block_idx * BN + col_lane8
                    vals = []
                    for q in range_constexpr(route_vec):
                        idx_q = row_in_block * BN + col_lane8 + fx.Int32(q)
                        if const_expr(g2_bf16_lds):
                            # bf16 LDS already has routing weight baked in at write time.
                            vals.append(fx.Float32(lds_base_bf16[idx_q]))
                        else:
                            vals.append(fx.Float32(lds_base_fptr[idx_q]) * weight[mr])
                    local_max = fabs_f32(vals[0])
                    for q in range_constexpr(1, route_vec):
                        local_max = local_max.maximumf(fabs_f32(vals[q]))
                    amax_bits = local_max.bitcast(fx.Int32)
                    ax_e = (amax_bits >> fx.Int32(23)) & fx.Int32(0xFF)
                    e8m0 = ax_e - fx.Int32(7)
                    e8m0 = (e8m0 < fx.Int32(1)).select(fx.Int32(1), e8m0)
                    e8m0 = (amax_bits == fx.Int32(0)).select(fx.Int32(0), e8m0)
                    block_scale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
                    bs_raw = _raw(block_scale)
                    pk_ty = T.vec(2, T.i16)
                    packed_lo = _raw(Vec.filled([2], 0, fx.Int16))
                    packed_lo = rocdl.cvt_scalef32_pk_fp8_f32(pk_ty, packed_lo, _raw(vals[0]), _raw(vals[1]), bs_raw, 0)
                    packed_lo = rocdl.cvt_scalef32_pk_fp8_f32(pk_ty, packed_lo, _raw(vals[2]), _raw(vals[3]), bs_raw, 1)
                    packed_hi = _raw(Vec.filled([2], 0, fx.Int16))
                    packed_hi = rocdl.cvt_scalef32_pk_fp8_f32(pk_ty, packed_hi, _raw(vals[4]), _raw(vals[5]), bs_raw, 0)
                    packed_hi = rocdl.cvt_scalef32_pk_fp8_f32(pk_ty, packed_hi, _raw(vals[6]), _raw(vals[7]), bs_raw, 1)
                    row_val_off = row_base_addr + fx.Int64(col_g0)
                    packed_frag = fx.make_rmem_tensor(1, Int32)
                    packed_frag.store(Vec(packed_lo).bitcast(Int32))
                    fx.copy(store_i32, packed_frag, out_i8[None, row_val_off])
                    packed_frag.store(Vec(packed_hi).bitcast(Int32))
                    fx.copy(store_i32, packed_frag, out_i8[None, row_val_off + fx.Int64(4)])
                    scale_off = row_base_addr + fx.Int64(N_OUT) + fx.Int64(col_g0 // fx.Int32(route_vec))
                    scale_frag = fx.make_rmem_tensor(1, Int8)
                    scale_frag.store(Vec.from_elements([e8m0.to(Int8)], Int8))
                    fx.copy(store_i8, scale_frag, out_i8[None, scale_off])

                @flyc.jit
                def store_route_group_if_valid(col_lane8):
                    if col_lane8 < fx.Int32(BN):
                        store_route_group(col_lane8)

                store_route_group_if_valid(col_lane8)
        else:
            for s in range_constexpr(BN // store_group_n):
                # adjacent ee=0,1 contiguous -> one 2-wide load.
                idx0 = row_in_block * BN + col_start + s * store_group_n
                if const_expr(g2_bf16_lds):
                    pk = Vec(
                        lds_vec_load(
                            lds_acc_base,
                            idx0 * 2,
                            Vec.make_type(store_vec, BFloat16),
                            BFloat16,
                            align=4,
                        )
                    )
                else:
                    v2 = Vec(
                        lds_vec_load(
                            lds_acc_base,
                            idx0 * 4,
                            Vec.make_type(store_vec, Float32),
                            Float32,
                            align=8,
                        )
                    )
                    pk = Vec.from_elements([v2[0] * weight[mr], v2[1] * weight[mr]], Float32).to(BFloat16)
                out_frag = fx.make_rmem_tensor(store_vec, BFloat16)
                out_frag.store(pk)
                out_off = row_base_addr + fx.Int64(s * store_group_n)
                if const_expr(use_reduce):
                    fx.copy(store_bf16x2, out_frag, out_bf16[None, out_off])
                else:
                    fx.copy(atomic_bf16x2, out_frag, out_bf16[None, out_off])

    for mr in range_constexpr(M_REPS):
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)

        @flyc.jit
        def store_if_valid(token_id, mr):
            if token_id < i32_M:
                store_one_mr(mr)

        store_if_valid(token_id, mr)


def get_gemm2_autotune_configs(a_dtype="fp8"):
    """Return the pruned Stage2 autotune space."""
    cfgs = [
        # (default) 2-stage B + A-scale prefetch + spatial-partition remap, f32 cshuffle, non-persist.
        dict(BN=256, BK=256, use_nt=True, g2_bhoist=True, g2_ascale_pf=True, g2_spart=402, persist=False),
        dict(BN=128, BK=256, use_nt=True, g2_bhoist=True, g2_ascale_pf=True, g2_spart=402, persist=False),
        dict(BN=256, BK=256, use_nt=True, g2_bhoist=True, g2_ascale_pf=True, g2_spart=0, persist=False),
        dict(BN=128, BK=256, use_nt=True, g2_bhoist=True, g2_ascale_pf=True, g2_spart=0, persist=False),
        dict(BN=256, BK=256, use_nt=False, g2_bhoist=True, g2_ascale_pf=True, g2_spart=402, persist=False),
        dict(BN=256, BK=256, use_nt=True, g2_bhoist=False, g2_ascale_pf=True, g2_spart=402, persist=False),
        dict(BN=256, BK=256, use_nt=True, g2_bhoist=True, g2_ascale_pf=False, g2_spart=402, persist=False),
        dict(BN=256, BK=256, use_nt=True, g2_bhoist=True, g2_ascale_pf=True, g2_spart=202, persist=False),
    ]
    bn_values = (128, 256)
    bn_env = os.environ.get("MEGA_G2_BN_VALUES")
    if bn_env:
        bn_values = tuple(int(value) for value in bn_env.split(","))
        if not bn_values or any(value not in (128, 256) for value in bn_values):
            raise ValueError("MEGA_G2_BN_VALUES must contain comma-separated values from {128, 256}")
    persist_cu_values = (8, 16, 32, 64, 128, 240, 256)
    persist_cu_env = os.environ.get("MEGA_G2_PERSIST_CU_VALUES")
    if persist_cu_env:
        persist_cu_values = tuple(int(value) for value in persist_cu_env.split(","))
        if not persist_cu_values or any(value <= 0 or value > 256 for value in persist_cu_values):
            raise ValueError("MEGA_G2_PERSIST_CU_VALUES must contain comma-separated values in [1, 256]")
    persist_strided_values = (False,)
    persist_strided_env = os.environ.get("MEGA_G2_PERSIST_STRIDED")
    if persist_strided_env is not None:
        if persist_strided_env not in ("0", "1"):
            raise ValueError("MEGA_G2_PERSIST_STRIDED must be '0' or '1'")
        persist_strided_values = (persist_strided_env == "1",)
    bf16_lds_values = (False,)
    bf16_lds_env = os.environ.get("MEGA_G2_BF16_LDS")
    if bf16_lds_env is not None:
        if bf16_lds_env not in ("0", "1"):
            raise ValueError("MEGA_G2_BF16_LDS must be '0' or '1'")
        bf16_lds_values = (bf16_lds_env == "1",)
    cfgs.extend(
        dict(
            BN=bn,
            BK=256,
            use_nt=use_nt,
            g2_bhoist=True,
            g2_ascale_pf=True,
            g2_spart=402,
            persist=True,
            persist_cu=slots,
            **({"persist_strided": True} if persist_strided else {}),
            **({"g2_bf16_lds": True} if g2_bf16_lds else {}),
        )
        for bn in bn_values
        for use_nt in (True, False)
        for slots in persist_cu_values
        for persist_strided in persist_strided_values
        for g2_bf16_lds in bf16_lds_values
    )
    if persist_strided_env is None and bf16_lds_env is None and 240 in persist_cu_values:
        cfgs.extend(
            dict(
                BN=bn,
                BK=256,
                use_nt=use_nt,
                g2_bhoist=True,
                g2_ascale_pf=True,
                g2_spart=402,
                persist=True,
                persist_cu=240,
                persist_strided=True,
            )
            for bn in bn_values
            for use_nt in (True, False)
        )
    if bf16_lds_env is None and 128 in bn_values:
        cfgs.extend(
            dict(
                BN=128,
                BK=256,
                use_nt=use_nt,
                g2_bhoist=True,
                g2_ascale_pf=True,
                g2_spart=402,
                persist=True,
                persist_cu=slots,
                persist_strided=persist_strided_values[0],
                g2_bf16_lds=True,
            )
            for use_nt in (True, False)
            for slots in persist_cu_values
            if slots in (64, 128, 240, 256)
        )
    no_persist = os.environ.get("MEGA_G2_NO_PERSIST") == "1"
    force_persist = os.environ.get("MEGA_G2_FORCE_PERSIST") == "1"
    if no_persist and force_persist:
        raise ValueError("MEGA_G2_NO_PERSIST and MEGA_G2_FORCE_PERSIST cannot both be enabled")
    if no_persist:
        cfgs = [config for config in cfgs if not config["persist"]]
    elif force_persist:
        cfgs = [config for config in cfgs if config["persist"]]
    force_use_nt = os.environ.get("MEGA_G2_FORCE_USE_NT")
    if force_use_nt is not None:
        if force_use_nt not in ("0", "1"):
            raise ValueError("MEGA_G2_FORCE_USE_NT must be '0' or '1'")
        use_nt = force_use_nt == "1"
        cfgs = [config for config in cfgs if config["use_nt"] == use_nt]
    # BK128 is inaccurate here; BM128 at BN256 fits gfx950's 160 KiB block LDS.
    return [dict(config, BM=BM) for BM in (16, 32, 64, 128) for config in cfgs]


# ---- gemm2 (down-proj) compile + launch driver (ported from aiter mxmoe_dispatcher.py) ----
def _norm_sbm(SBM, BM):
    """Resolve SBM (sort_block_m): None -> SBM==BM."""
    return BM if SBM is None else SBM


def _spart_output_tile_index(block_1d_id, M0, N0, group_num, m01):
    """Map a 1D block ID to a spatially local output tile."""
    gn = fx.Int32(group_num)
    n0 = fx.Int32(N0)
    m01c = fx.Int32(m01)

    # group_size = ceil(M0*N0 / GroupNum); big_group_num = GroupNum - (group_size*GroupNum - M0*N0)
    mn = M0 * n0
    group_size = (mn + gn - fx.Int32(1)) // gn
    big_group_num = gn - (group_size * gn - mn)

    group_id_y = block_1d_id // gn
    group_id_x = block_1d_id - group_id_y * gn

    # remap = group_id_x <= big_group_num ? gx*gs + gy : gx*gs + big - gx + gy
    remap_a = group_id_x * group_size + group_id_y
    remap_b = group_id_x * group_size + big_group_num - group_id_x + group_id_y
    remap = (group_id_x <= big_group_num).select(remap_a, remap_b)

    idx_M0 = remap // n0
    idx_N0 = remap - idx_M0 * n0

    # M0_tmp = M0 / M01 ; M0_mod_M01 = M0 - M0_tmp*M01 ; M01_adapt = (idx_M0 < M0 - M0_mod) ? M01 : M0_mod
    M0_tmp = M0 // m01c
    M0_mod = M0 - M0_tmp * m01c
    M01_adapt = (idx_M0 < (M0 - M0_mod)).select(m01c, M0_mod)

    idx_M00 = idx_M0 // m01c
    idx_M01 = idx_M0 - idx_M00 * m01c
    idx_local = idx_N0 + idx_M01 * n0

    N_out = idx_local // M01_adapt
    loc_mod = idx_local - N_out * M01_adapt

    m_block_idx = loc_mod + idx_M00 * m01c
    n_block_idx = N_out
    return m_block_idx, n_block_idx


def _resolve_g2_knobs(g2_bhoist, g2_ascale_pf, g2_spart, g2_bf16_lds, use_reduce):
    """Env-default the gemm2 perf knobs (explicit arg wins), matching aiter compile_gemm2_a4w4_port."""
    if g2_bhoist is None:
        g2_bhoist = os.environ.get("MXFP4_G2_BHOIST", "1") == "1"
    g2_bhoist = bool(g2_bhoist)
    if g2_ascale_pf is None:
        g2_ascale_pf = os.environ.get("MXFP4_G2_ASCALE_PF", "1") == "1"
    g2_ascale_pf = bool(g2_ascale_pf)
    if g2_spart is None:
        g2_spart = int(os.environ.get("MXFP4_G2_SPART", "402"))
    g2_spart = int(g2_spart)
    g2_group_num = g2_spart // 100 if g2_spart > 0 else 0
    g2_m01 = g2_spart % 100 if g2_spart > 0 else 0
    if g2_spart > 0 and (g2_group_num < 1 or g2_m01 < 1):
        raise AssertionError(f"g2_spart={g2_spart} must encode GroupNum>=1,M01>=1 as GroupNum*100+M01 (e.g. 402)")
    if g2_bf16_lds is None:
        g2_bf16_lds = os.environ.get("MXFP4_G2_BF16_LDS", "1") == "1" and use_reduce
    g2_bf16_lds = bool(g2_bf16_lds) and use_reduce
    return g2_bhoist, g2_ascale_pf, g2_spart, g2_group_num, g2_m01, g2_bf16_lds


def compile_gemm2_a4w4_port(
    BM=32,
    BN=256,
    BK=256,
    use_nt=False,
    HIDDEN_MAX=8192,
    epilog="atomic",
    INTER_MAX=8192,
    a_dtype="fp4",
    topk=1,
    SBM=None,
    persist=False,
    cu_num=0,
    has_pad=False,
    g2_bhoist=None,
    g2_ascale_pf=None,
    g2_spart=None,
    g2_bf16_lds=None,
    out_dtype="bf16",
):
    """Compile A4W4 GEMM2 with atomic or per-route reduction output."""
    SBM = _norm_sbm(SBM, BM)
    if BM not in (16, 32, 64, 128) or epilog not in ("atomic", "reduce"):
        raise AssertionError(
            f"mxfp4_moe_gemm2 supports only (BM in {{16,32,64,128}}, epilog in {{'atomic','reduce'}}); "
            f"got (BM={BM}, epilog={epilog})"
        )
    if SBM % BM != 0:
        raise AssertionError(f"SBM ({SBM}) must be a multiple of BM ({BM})")
    use_reduce = epilog == "reduce"
    out_dtype = str(out_dtype).strip().lower()
    if out_dtype not in ("bf16", "fp8"):
        raise AssertionError(f"out_dtype must be 'bf16' or 'fp8', got {out_dtype!r}")
    route_out_fp8 = out_dtype == "fp8"
    if route_out_fp8 and not use_reduce:
        raise AssertionError("out_dtype='fp8' is supported only with epilog='reduce'")
    (
        g2_bhoist,
        g2_ascale_pf,
        g2_spart,
        g2_group_num,
        g2_m01,
        g2_bf16_lds,
    ) = _resolve_g2_knobs(g2_bhoist, g2_ascale_pf, g2_spart, g2_bf16_lds, use_reduce)
    if a_dtype not in ("fp4", "fp8"):
        raise AssertionError(f"a_dtype must be 'fp4' or 'fp8', got {a_dtype!r}")
    assert INTER_MAX % BK == 0, f"INTER_MAX must be a multiple of {BK}, got {INTER_MAX}"
    is_f8 = a_dtype == "fp8"
    KH_TILE_A = BK // (1 if is_f8 else 2)  # A LDS K-tile bytes (fp8 256, fp4 128)
    slot_bytes = BM * KH_TILE_A
    aStages = 2 if g2_bf16_lds else 3
    c_lds_bytes = BM * BN * (2 if g2_bf16_lds else 4)
    lds_bytes = max(c_lds_bytes, aStages * slot_bytes)
    # HIDDEN_MAX buckets compiled launchers while the output width remains runtime.
    assert HIDDEN_MAX % BN == 0, f"HIDDEN_MAX must be a multiple of {BN}, got {HIDDEN_MAX}"

    # Kernel-name tags empty on the default so its name/IR stays byte-identical (each variant distinct).
    atag = "_a8" if is_f8 else ""
    etag = "atomic" if not use_reduce else f"reduce_tk{topk}"
    sbm_tag = "" if SBM == BM else f"_sbm{SBM}"
    if persist and cu_num <= 0:
        raise AssertionError(f"persist=True requires cu_num>0, got {cu_num}")
    persist_tag = "" if not persist else f"_persist_cu{cu_num}"
    pad_tag = "_pad" if has_pad else ""  # has_pad adds the runtime pad kernarg + weight-OOB pad-skip
    bh_tag = "_bhoist" if g2_bhoist else ""
    apf_tag = "_apf" if g2_ascale_pf else ""
    spart_tag = f"_spart{g2_group_num}x{g2_m01}" if g2_spart > 0 else ""
    bf16lds_tag = "_bf16lds" if g2_bf16_lds else ""
    out_tag = "_fp8out" if route_out_fp8 else ""
    tag = (
        f"hmax{HIDDEN_MAX}_imax{INTER_MAX}_bm{BM}{'_nt' if use_nt else ''}_{etag}{atag}{sbm_tag}"
        f"{persist_tag}{pad_tag}{bh_tag}{apf_tag}{spart_tag}{bf16lds_tag}{out_tag}_v2"
    )
    name = f"gemm2_a4w4_port_{tag}"

    @fx.struct
    class SharedStorage:
        buf: fx.Array[Int8, lds_bytes, 16]

    @flyc.jit
    def _gemm2_kernel_body(
        arg_aq,
        arg_ascale,
        arg_bq,
        arg_bscale,
        arg_eids,
        arg_cumsum,
        arg_stids,
        arg_sweights,
        arg_out,
        bx_i32,
        lane,
        wave,
        i32_M,
        i32_max_m_blocks,
        i32_inter,
        i32_hidden,
        i32_kpad,
        i32_npad,
    ):
        # Share the JIT body while the default variant folds zero padding away.
        num_n_blocks = fx.Int32(i32_hidden) // fx.Int32(BN)  # N_OUT//BN runtime (i32_hidden = model_dim)
        k_bytes = fx.Int32(i32_inter) // fx.Int32(1 if is_f8 else 2)  # A row stride bytes (runtime)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds.buf.ptr))

        # Preload the first kStages K-tiles (the streaming prologue).
        def issue_all_a_loads(m_row0):
            for slot in range_constexpr(kStages):
                issue_a_load_lds_dt(
                    arg_aq,
                    lds_base_i32,
                    slot,
                    slot,
                    m_row0,
                    wave,
                    lane,
                    is_f8,
                    KH_TILE_A,
                    k_bytes,
                    BM=BM,
                )

        # One (m_block, n_block) unit for a synthesized unit_bx; non-persist calls once, persist per m-tile.
        def run_unit(unit_bx):
            gemm2_body_v2(
                lds_base_i32,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_stids,
                arg_sweights,
                i32_M,
                i32_max_m_blocks,
                arg_out,
                unit_bx,
                lane,
                wave,
                arg_aq,
                i32_inter,
                i32_hidden,
                i32_kpad,
                i32_npad,
                BM=BM,
                BN=BN,
                BK=BK,
                use_nt=use_nt,
                INTER_MAX=INTER_MAX,
                aStages=aStages,
                a_dtype=a_dtype,
                use_reduce=use_reduce,
                topk=topk,
                has_pad=has_pad,
                SBM=SBM,
                g2_bhoist=g2_bhoist,
                g2_ascale_pf=g2_ascale_pf,
                g2_bf16_lds=g2_bf16_lds,
                route_out_fp8=route_out_fp8,
            )

        if const_expr(not persist and g2_spart <= 0):
            # One-shot naive linear block->(m,n): issue A->LDS before the cumsum load (latency overlap).
            issue_all_a_loads((bx_i32 // num_n_blocks) * fx.Int32(BM))
            rocdl.sched_barrier(0)

            cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
            total_m_blocks = cumsum0 // BM
            bound = total_m_blocks * fx.Int32(num_n_blocks)

            if fx.Int32(bx_i32) < bound:
                run_unit(bx_i32)
        elif const_expr(not persist):
            # One-shot with spatial-partitioner remap (g2_spart>0): needs M0=total_m_blocks so cumsum is read FIRST.
            cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
            total_m_blocks = cumsum0 // BM
            bound = total_m_blocks * fx.Int32(num_n_blocks)

            if fx.Int32(bx_i32) < bound:
                m_block_idx, n_block_idx = _spart_output_tile_index(
                    bx_i32, total_m_blocks, num_n_blocks, g2_group_num, g2_m01
                )
                unit_bx = m_block_idx * fx.Int32(num_n_blocks) + n_block_idx
                issue_all_a_loads(m_block_idx * fx.Int32(BM))
                rocdl.sched_barrier(0)
                run_unit(unit_bx)
        else:
            # Persistent blocks grid-stride M tiles by cu_num.
            m_tile0 = bx_i32 // fx.Int32(num_n_blocks)
            n_block = bx_i32 - m_tile0 * fx.Int32(num_n_blocks)
            c_stride = fx.Int32(cu_num)

            cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
            total_m_blocks = cumsum0 // BM
            # ceil((total_m_blocks - m_tile0) / cu_num), clamped to 0 when m_tile0 >= total_m_blocks.
            diff = total_m_blocks - m_tile0
            rem = (diff > fx.Int32(0)).select(diff, fx.Int32(0))
            n_iters = (rem + c_stride - fx.Int32(1)) // c_stride
            for _it in range(
                fx.Int32(0),
                n_iters,
                fx.Int32(1),
            ):
                m_block = m_tile0 + fx.Int32(_it) * c_stride
                unit_bx = m_block * fx.Int32(num_n_blocks) + n_block
                gpu.barrier()  # Separate persistent iterations sharing LDS.
                issue_all_a_loads(m_block * fx.Int32(BM))
                rocdl.sched_barrier(0)
                if fx.Int32(m_block) < total_m_blocks:
                    run_unit(unit_bx)

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def gemm2_kernel(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        i32_inter: fx.Int32,
        i32_hidden: fx.Int32,
        i32_kpad: fx.Int32,
        i32_npad: fx.Int32,
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,  # unused (atomic epilog); kept for signature parity
    ):
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = fx.Int32(tx)
        bx_i32 = fx.Int32(bx)
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        _gemm2_kernel_body(
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            arg_out,
            bx_i32,
            lane,
            wave,
            i32_M,
            i32_max_m_blocks,
            i32_inter,
            i32_hidden,
            i32_kpad,
            i32_npad,
        )

    @flyc.jit
    def launch_gemm2(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        i32_grid_blocks: fx.Int32,
        i32_inter: fx.Int32,
        i32_hidden: fx.Int32,
        i32_kpad: fx.Int32,
        i32_npad: fx.Int32,
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,
        stream: fx.Stream,
    ):
        # i32_max_m_blocks sizes buffer resources; i32_grid_blocks bounds the launch to real m-blocks.
        num_n_blocks = fx.Int32(i32_hidden) // fx.Int32(BN)
        grid_x = i32_grid_blocks * num_n_blocks
        gemm2_kernel(
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            i32_M,
            i32_max_m_blocks,
            i32_inter,
            i32_hidden,
            i32_kpad,
            i32_npad,
            arg_out,
            arg_out_scale,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2
