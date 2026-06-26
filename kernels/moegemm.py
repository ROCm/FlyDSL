# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Layout-API MXFP4 MoE GEMM device bodies (BM32, gemm1 up/gate + gemm2 down).

The BM32 core is shared: same CK-preshuffled B weight + e8m0 scale layout and the
same scaled 16x16x128 fp4 MFMA opsel. Holds the layout-API B / B-scale primitives
(copy atoms, views, fragment templates, MFMA_Scale atoms, gemm_mma), the A-side
LDS loaders, both gemm bodies, and the atomic-bf16 epilogue. Basics come from
``utils``; compile + launch from ``moe_dispatcher``.

fp4 A rides fx.gemm register fragments; fp8 A (a8w4) the raw mfma_scale intrinsic
(cbsz=0). C accumulates in place (fx.gemm d == c).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Float4E2M1FN, T
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemPtr

from .utils import (
    BK,
    BN,
    KH_TILE,
    _e8m0_from_amax,
    _fabs_f32,
    _gep1,
    _gep3,
    _global_base_ptr1,
    _global_ptr1,
    _lds_base_ptr3,
    _lds_ptr3,
    _lds_swizzle_mask,
    _lds_swizzle_mask_f8,
    _raw,
    _silu_mul_batch,
    _udiv,
    k_half_for,
    k_tiles_total_for,
    kas_per_chunk_dw_for,
    kbs_per_expert_dw_for,
    kBS_stride_k0_dw,
    kbs_stride_n0_dw_for,
    kmchunks,
    kStages,
    kunroll_for,
    num_n_blocks_for,
)

# BM32: the single supported variant (both gemm1 and gemm2).
BM = 32
kAStages = 3
kSubBlocks = 1
kMChunks = 2  # BM // 16; also the gemm1 epilog row-rep count


# ===========================================================================
# Shared layout-API primitives (B / B-scale data movement + scaled MFMA)
# ===========================================================================
def b_copy_atom(nontemporal):
    """BufferCopy128b (4x i32 = one 128b weight chunk). nt rides cache_modifier."""
    return fx.make_copy_atom(fx.rocdl.BufferCopy128b(2 if nontemporal else 0), 32)


def bscale_copy_atom():
    """BufferCopy32b (1x i32 e8m0 scale word); always cached (scales reuse heavily)."""
    return fx.make_copy_atom(fx.rocdl.BufferCopy32b(0), 32)


def bq_view(arg_bq, row_elems, KH4, K_TILES_TOTAL):
    """Layout view over the preshuffled B weight for one N-row tile.

    Base = readfirstlane(row_elems * KH4) is uniform per wave (KH4 = K_HALF//4);
    per-lane (klane,nlane)/K-tile/half/kpack4 are layout axes -> VGPR voffset (not
    a divergent-pointer waterfall). view[lane//16, lane%16, kt, half, None] is an
    i32<4:1> (16B = 32 fp4) slice.
    """
    col_base = rocdl.readfirstlane(T.i32, _raw(row_elems) * fx.Int32(KH4))
    i32_ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=16)
    off_i64 = fx.Int64(arith.ExtUIOp(T.i64, _raw(col_base)).result)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_bq) + off_i64 * fx.Int64(4))
    # i32 strides: klane[0,4)->64, nlane[0,16)->4, K_tile->512, half[0,2)->256, kpack4->1
    shape = (4, 16, K_TILES_TOTAL, 2, 4)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, (64, 4, 512, 256, 1))))
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def bscale_view(arg_bscale, base_dw, K_TILES_TOTAL, k0_stride_dw=64):
    """Layout view over the e8m0 B-scale for one n-pack word. base_dw is the uniform
    per-wave dword base; per-lane (klane,nlane)/K-tile are layout axes.
    view[lane//16, lane%16, kt, None] is an i32<1:1> scale word."""
    base_dw = rocdl.readfirstlane(T.i32, _raw(base_dw))
    i32_ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
    off_i64 = fx.Int64(arith.ExtUIOp(T.i64, _raw(base_dw)).result)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_bscale) + off_i64 * fx.Int64(4))
    shape = (4, 16, K_TILES_TOTAL, 1)
    stride = (16, 1, k0_stride_dw, 1)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, stride)))
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def bq_frag_tmpl(view):
    """i32<4:1> fragment template sliced from a bq_view (16B = 32 fp4)."""
    return view[0, 0, 0, 0, None]


def bscale_frag_tmpl(view):
    """i32<1:1> fragment template sliced from a bscale_view (one e8m0 word)."""
    return view[0, 0, 0, None]


def scale_mma_atoms():
    """The 16 (opselA,opselB) scaled-MFMA atoms (opsel is a type param -> one atom
    per pair); cbsz/blgp(=4 for fp4) inferred from Float4E2M1FN."""
    return {
        (osa, osb): fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, Float4E2M1FN, opsel_a=osa, opsel_b=osb))
        for osa in range(4)
        for osb in range(4)
    }


def gemm_mma(atoms, a_frag, b_frag, c_frag, opsel_a, opsel_b, sa, sb):
    """One scaled MFMA via fx.gemm over rank-1 fragments; C accumulates in place."""
    fx.gemm(
        atoms[(opsel_a, opsel_b)],
        c_frag,
        a_frag,
        b_frag,
        c_frag,
        scale_a=sa,
        scale_b=sb,
    )


# ===========================================================================
# Shared A ds-read + per-J MMA cluster (used by both gemm bodies)
# ===========================================================================
def _issue_a_ds_read_dt(saq, slot, slot_bytes, KH_TILE_A, lane_div_16, lane_mod_16, is_f8, a_vals, a_frags):
    """A ds-read for one slot: fp4 -> Vec4 i32 into a_frags; fp8 -> Vec8 i32 (two
    halves 64B apart) into a_vals."""
    base_ptr = _lds_base_ptr3(saq.get())
    for k in range_constexpr(2):
        for i in range_constexpr(kMChunks):
            lds_row = lane_mod_16 + fx.Int32(i * 16)
            row_off = fx.Int32(slot * slot_bytes) + lds_row * fx.Int32(KH_TILE_A)
            if const_expr(is_f8):
                mask = _lds_swizzle_mask_f8(lane_mod_16)
                col0 = lane_div_16 * fx.Int32(16) + fx.Int32(k * 128)
                col_lo = col0 ^ mask
                col_hi = (col0 + fx.Int32(64)) ^ mask
                lo = Vec(llvm.load(T.vec(2, T.i64), _gep3(base_ptr, row_off + col_lo)))
                hi = Vec(llvm.load(T.vec(2, T.i64), _gep3(base_ptr, row_off + col_hi)))
                a64 = Vec.from_elements([lo[0], lo[1], hi[0], hi[1]], fx.Int64)
                a_vals[i][k] = _raw(a64.bitcast(fx.Int32))
            else:
                mask = _lds_swizzle_mask(lane_mod_16)
                lds_col = (lane_div_16 * fx.Int32(16) + fx.Int32(k * 64)) ^ mask
                vec = llvm.load(T.vec(4, T.i32), _gep3(base_ptr, row_off + lds_col))
                a_frags[i][k].store(Vec(vec))


def _mma_one_j(J, in_b, sa, sb, bq_frags_kt, is_f8, cbsz_a, a_vals, a_frags, accm, c_frags, atoms):
    """One J-cluster (4 scaled MFMAs). fp4 via gemm_mma; fp8 via the raw mfma_scale
    intrinsic. mni=J//2; in_b (gate mode) is resolved by the caller."""
    if const_expr(is_f8):
        bJ0 = Vec(bq_frags_kt[J][0].load())
        bJ1 = Vec(bq_frags_kt[J][1].load())
        for osa, k, i in ((0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1)):
            bJ = bJ0 if k == 0 else bJ1
            osb = (0 if k == 0 else 2) + in_b
            accm[i][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4, [a_vals[i][k], bJ, accm[i][J], cbsz_a, 4, osa, sa, osb, sb]
            )
    else:
        bJ0, bJ1 = bq_frags_kt[J][0], bq_frags_kt[J][1]
        gemm_mma(atoms, a_frags[0][0], bJ0, c_frags[0][J], 0, 0 + in_b, sa, sb)
        gemm_mma(atoms, a_frags[1][0], bJ0, c_frags[1][J], 1, 0 + in_b, sa, sb)
        gemm_mma(atoms, a_frags[0][1], bJ1, c_frags[0][J], 2, 2 + in_b, sa, sb)
        gemm_mma(atoms, a_frags[1][1], bJ1, c_frags[1][J], 3, 2 + in_b, sa, sb)


# ===========================================================================
# gemm1 (up/gate-proj)
# ===========================================================================
@flyc.jit
def _gemm1_body_v2(
    allocator,
    lds_off,
    arg_aq,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_sti,
    arg_aqout,
    arg_ascaleout,
    bx_i32,
    lane,
    wave,
    i32_ntok,
    i32_total_m_blocks,
    *,
    K,
    INTER,
    NE,
    interleave,
    b_nontemporal,
    a_dtype,
    out_dtype,
):
    # A dtype: only the A path differs (LDS tile, ds-read, mfma format); fp8 uses the
    # raw mfma_scale intrinsic (cbsz=0), fp4 the fx.gemm fragment path.
    is_f8_a = a_dtype == "fp8"
    # out dtype: only the epilogue requant/pack/store differs.
    is_f8_out = out_dtype == "fp8"
    out_max = 448.0 if is_f8_out else 6.0  # e4m3 / e2m1 max
    out_pack = 1 if is_f8_out else 2
    a_pack = 1 if is_f8_a else 2
    am = 2 // a_pack  # A row-group calls per 8-row sub (fp8=2, fp4=1)
    KH_TILE_A = BK // a_pack  # A bytes/K-tile row in LDS (fp8=256, fp4=128)
    cbsz_a = 0 if is_f8_a else 4  # mfma A-format (fp8=0, fp4=4)
    # K-/INTER-derived sizes (compile-time ints).
    _kc = (K // 32) // 4 // 2
    K_HALF = K // 2
    K_BYTES = K // a_pack  # a_quant row stride in bytes (= K_HALF for fp4)
    K_TILES_TOTAL = K // BK
    kUnroll = K_TILES_TOTAL - kStages
    kAS_per_chunk_dw = _kc * 64
    kBS_stride_n0_dw = _kc * 64
    N_OUT = 2 * INTER
    kBS_per_expert_dw = (N_OUT // 16 // 2) * kBS_stride_n0_dw
    NUM_N_BLOCKS = N_OUT // 256
    OUT_AS_PER_CHUNK_DW = ((INTER // 32) // 4 // 2) * 64
    K_G2_BYTES = INTER // out_pack  # output intermediate row stride (fp4 INTER/2, fp8 INTER)

    # block -> (m_block_idx, n_block_idx) ; e = sorted_expert_ids[m_block_idx]
    n_block_idx = bx_i32 % fx.Int32(NUM_N_BLOCKS)
    m_block_idx = bx_i32 // fx.Int32(NUM_N_BLOCKS)
    e = rocdl.readfirstlane(T.i32, llvm.load(T.i32, _global_ptr1(arg_eids, m_block_idx * fx.Int32(4))))
    m_row = m_block_idx * fx.Int32(BM)

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

    # buffer resources (A-gather + scales)
    aq_num_records = arith.index_cast(T.index, _raw(i32_ntok * fx.Int32(K_BYTES)))
    aq_rsrc = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_aq)), num_records_bytes=aq_num_records)
    _asc_per_mb = max(BM // 32, 1) * kAS_per_chunk_dw * 4
    ascale_num = arith.index_cast(T.index, _raw(i32_total_m_blocks)) * fx.Index(_asc_per_mb)
    ascale_rsrc = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_ascale)), num_records_bytes=ascale_num)

    # LDS views (s_aq / s_asc, union-overlapping lds_acc)
    lds_base = allocator.get_base()
    s_aq = SmemPtr(lds_base, lds_off, T.i8, shape=(kAStages * BM * KH_TILE_A,))
    s_asc = SmemPtr(
        lds_base,
        lds_off + kAStages * BM * KH_TILE_A,
        T.i8,
        shape=(kSubBlocks * K_TILES_TOTAL * 256,),
    )
    lds_acc = SmemPtr(lds_base, lds_off, T.f32, shape=(BM * BN,))

    # A-gather rows: buffer_load_lds fills 64*16B/wave (fp4 8 rows x 128B, fp8 4 rows
    # x 256B). Row = sorted_token_ids & 0xFFFFFF (drop topk high byte); pad rows carry
    # token_id==M (OOB) so the bounds-checked load returns 0.
    lanes_per_row = KH_TILE_A // 16  # 8 (fp4) / 16 (fp8)
    rows_per_call = 64 // lanes_per_row  # 8 (fp4) / 4 (fp8)
    a_lane_row = lane // fx.Int32(lanes_per_row)
    mask24_i32 = arith.constant(0xFFFFFF, type=T.i32)
    cached_actual_row = []
    for sub in range_constexpr(kSubBlocks):
        for h in range_constexpr(am):
            idx = m_row + wave * fx.Int32(BM // 4) + fx.Int32(sub * 8 + h * rows_per_call) + a_lane_row
            cached_actual_row.append(
                arith.andi(
                    llvm.load(T.i32, _global_ptr1(arg_sti, idx * fx.Int32(4))),
                    mask24_i32,
                )
            )

    # B-scale n-pack words (gate/up split differs by gate mode).
    if const_expr(interleave):
        mni_base = n_block_idx * fx.Int32(BN // 32) + wave * fx.Int32(BN // 128)
        np_list = [mni_base, mni_base + fx.Int32(1)]
    else:
        np_gate = n_block_idx * fx.Int32(BN // 64) + wave
        np_list = [np_gate, np_gate + fx.Int32(N_OUT // 64)]

    def issue_a_load_lds(slot, kt):
        # lane L -> LDS[base+L*16]; fp8 splits each 8-row sub into `am` row-groups.
        lane_col = (lane % fx.Int32(lanes_per_row)) * fx.Int32(16)
        base_i32 = fx.Int32(memref_dialect.extract_aligned_pointer_as_index(s_aq.get()))
        for sub in range_constexpr(kSubBlocks):
            for h in range_constexpr(am):
                lds_row = wave * fx.Int32(BM // 4) + fx.Int32(sub * 8 + h * rows_per_call)
                mask = (
                    _lds_swizzle_mask_f8(lds_row + a_lane_row)
                    if const_expr(is_f8_a)
                    else _lds_swizzle_mask(lds_row + a_lane_row)
                )
                voffset = (lane_col ^ mask) + cached_actual_row[sub * am + h] * fx.Int32(K_BYTES)
                off = fx.Int32(slot * (BM * KH_TILE_A)) + lds_row * fx.Int32(KH_TILE_A)
                rocdl.raw_ptr_buffer_load_lds(
                    aq_rsrc,
                    _lds_ptr3(base_i32, off),
                    fx.Int32(16),
                    voffset,
                    fx.Int32(kt * KH_TILE_A),
                    fx.Int32(0),
                    fx.Int32(0),
                )

    def issue_a_ds_read(slot):
        _issue_a_ds_read_dt(s_aq, slot, BM * KH_TILE_A, KH_TILE_A, lane_div_16, lane_mod_16, is_f8_a, _a_vals, _a_frags)

    def issue_a_scale_load():
        chunk_base = m_row // fx.Int32(32)
        v16 = (wave * fx.Int32(64) + lane) * fx.Int32(16)
        v4 = (wave * fx.Int32(64) + lane) * fx.Int32(4)
        asc_base = fx.Int32(memref_dialect.extract_aligned_pointer_as_index(s_asc.get()))
        for sub in range_constexpr(kSubBlocks):
            s_chunk = rocdl.readfirstlane(T.i32, (chunk_base + fx.Int32(sub)) * fx.Int32(kAS_per_chunk_dw * 4))
            lds_sub = fx.Int32(sub * kAS_per_chunk_dw * 4)
            rocdl.raw_ptr_buffer_load_lds(
                ascale_rsrc,
                _lds_ptr3(asc_base, lds_sub + wave * fx.Int32(1024)),
                fx.Int32(16),
                v16,
                s_chunk,
                fx.Int32(0),
                fx.Int32(0),
            )
            for d in range_constexpr(3):
                byte_off = 4096 + d * 1024
                s_off = rocdl.readfirstlane(T.i32, s_chunk + fx.Int32(byte_off))
                rocdl.raw_ptr_buffer_load_lds(
                    ascale_rsrc,
                    _lds_ptr3(asc_base, lds_sub + fx.Int32(byte_off) + wave * fx.Int32(256)),
                    fx.Int32(4),
                    v4,
                    s_off,
                    fx.Int32(0),
                    fx.Int32(0),
                )

    def issue_a_scale_ds_read(kt):
        base_ptr = _lds_base_ptr3(s_asc.get())
        out = []
        for sub in range_constexpr(kSubBlocks):
            lds_dw = fx.Int32(sub * kAS_per_chunk_dw) + fx.Int32(kt * 64) + lane_div_16 * fx.Int32(16) + lane_mod_16
            out.append(llvm.load(T.i32, _gep3(base_ptr, lds_dw * fx.Int32(4))))
        return out

    # B load: CK preshuffle as an fx.make_layout view over bq. The descriptor base
    # MUST stay uniform per wave (folding the per-lane part in makes make_buffer_tensor
    # emit a per-lane WATERFALL, ~14x slower), so the base is the uniform col offset
    # and the per-lane (klane,nlane) are layout axes -> a VGPR voffset at copy time.
    # nt/cached rides on the copy atom's cache_modifier (2=nt/0=cached).
    KH4 = K_HALF // 4  # i32 stride for the col axis
    _b_copy_atom = b_copy_atom(b_nontemporal)
    _bs_copy_atom = bscale_copy_atom()

    N0_HALF = N_OUT // 32  # separate-mode gate/up column split

    # B-load view per j-tile; gate mode only changes which N-row `col` maps to.
    def _make_bq_view_for_jtile(j):
        if const_expr(interleave):
            col = n_block_idx * fx.Int32(BN) + wave * fx.Int32(BN // 4) + fx.Int32(j * 16)
        else:
            tile_il = n_block_idx * fx.Int32(16) + wave * fx.Int32(4) + fx.Int32(j)
            col = ((tile_il & fx.Int32(1)) * fx.Int32(N0_HALF) + (tile_il >> fx.Int32(1))) * fx.Int32(16)
        return bq_view(arg_bq, e * fx.Int32(N_OUT) + col, KH4, K_TILES_TOTAL)

    _bq_views = [_make_bq_view_for_jtile(j) for j in range_constexpr(4)]

    # B-scale view per n-pack word (shared layout primitive).
    _bscale_views = [
        bscale_view(
            arg_bscale,
            e * fx.Int32(kBS_per_expert_dw) + np_list[mw] * fx.Int32(kBS_stride_n0_dw),
            K_TILES_TOTAL,
            k0_stride_dw=kBS_stride_k0_dw,
        )
        for mw in range_constexpr(2)
    ]

    # B fragments: i32<4:1> (16B = 32 fp4), per-stage (kStages) prefetch double-buffer.
    _frag_tmpl = bq_frag_tmpl(_bq_views[0])  # i32<4:1>
    _bq_frags = [
        [[fx.make_fragment_like(_frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(4)]
        for _ in range_constexpr(kStages)
    ]
    # fp4: A in fx.gemm fragments (refilled per K), C accumulates in place. fp8: A is
    # a per-iter Vec8 i32 (_a_vals), C a raw f32x4 accumulator (accm, zero-init).
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    _a_vals = _a_frags = _c_frags = accm = None
    if const_expr(is_f8_a):
        _a_vals = [[None, None] for _ in range(kMChunks)]
        accm = [[zero4 for _ in range(4)] for _ in range(kMChunks)]
    else:
        _a_frags = [[fx.make_fragment_like(_frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(kMChunks)]
        _c_frags = [
            [fx.make_fragment_like(_frag_tmpl, dtype=fx.Float32.ir_type) for _ in range_constexpr(4)]
            for _ in range_constexpr(kMChunks)
        ]
    # B-scale fragments: i32<1:1>, per-stage double-buffer like _bq_frags.
    _bs_frag_tmpl = bscale_frag_tmpl(_bscale_views[0])  # i32<1:1>
    _bs_frags = [[fx.make_fragment_like(_bs_frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(kStages)]

    def issue_b_load_j(stage, K_C, j):
        view = _bq_views[j]
        for half in range_constexpr(2):
            fx.copy(
                _b_copy_atom,
                view[lane_div_16, lane_mod_16, K_C, half, None],
                _bq_frags[stage][j][half],
            )

    def issue_b_scale_load(stage, K_C):
        for mw in range_constexpr(2):
            fx.copy(
                _bs_copy_atom,
                _bscale_views[mw][lane_div_16, lane_mod_16, K_C, None],
                _bs_frags[stage][mw],
            )

    # MMA: fp4 via fx.gemm (one per mfma); fp8 via the raw scaled-MFMA intrinsic.
    _scale_mma_atoms = scale_mma_atoms() if const_expr(not is_f8_a) else None

    def mfma_cluster(stage, a_scale, J):
        # interleave: mni=J//2 (n0), in_b=J%2 (gate/up); separate: swapped.
        if const_expr(interleave):
            mni, in_b = J // 2, J % 2
        else:
            mni, in_b = J % 2, J // 2
        sb = _raw(Vec(_bs_frags[stage][mni].load())[0])
        sa = a_scale[0]  # kSubBlocks == 1
        _mma_one_j(
            J, in_b, sa, sb, _bq_frags[stage], is_f8_a, cbsz_a, _a_vals, _a_frags, accm, _c_frags, _scale_mma_atoms
        )

    # zero C (fp4 fragments accumulate in place thereafter; fp8 accm pre-init above).
    if const_expr(not is_f8_a):
        for i in range_constexpr(kMChunks):
            for J in range_constexpr(4):
                _c_frags[i][J].store(zero4)

    # prologue: stages 0,1
    issue_a_scale_load()
    for K_C in range_constexpr(kStages):
        issue_a_load_lds(K_C, K_C)
        for j in range_constexpr(4):
            issue_b_load_j(K_C, K_C, j)
        issue_b_scale_load(K_C, K_C)

    # main loop. sched_barrier/s_setprio fence the mfma chain from the B loads so it
    # stays dense (closes the small-M gap).
    for OFFSET in range_constexpr(kUnroll):
        K_C = kStages + OFFSET
        read_slot = OFFSET % kAStages
        write_slot = K_C % kAStages
        slot_b = OFFSET % kStages
        gpu.barrier()
        issue_a_ds_read(read_slot)
        asc_cur = issue_a_scale_ds_read(K_C - kStages)
        issue_a_load_lds(write_slot, K_C)
        for J in range_constexpr(4):
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            mfma_cluster(slot_b, asc_cur, J)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            issue_b_load_j(slot_b, K_C, J)
            rocdl.sched_barrier(0)
        issue_b_scale_load(slot_b, K_C)

    # drain: last kStages
    for S in range_constexpr(kStages):
        kt = K_TILES_TOTAL - kStages + S
        gpu.barrier()
        issue_a_ds_read(kt % kAStages)
        asc_cur = issue_a_scale_ds_read(kt)
        for J in range_constexpr(4):
            mfma_cluster(kt % kStages, asc_cur, J)

    gpu.barrier()
    s_aq._view_cache = None
    s_asc._view_cache = None
    lds_acc._view_cache = None

    # epilog: cshuffle -> SwiGLU -> fp4 + e8m0 requant (raw math)
    wave_n = wave
    lds_acc_base = _lds_base_ptr3(lds_acc.get())

    # accumulators: fp4 from C fragments, fp8 from accm.
    if const_expr(is_f8_a):
        _acc_vecs = [[Vec(accm[i][J]) for J in range(4)] for i in range(kMChunks)]
    else:
        _acc_vecs = [[Vec(_c_frags[i][J].load()) for J in range(4)] for i in range(kMChunks)]

    def _acc(i, J, v):
        return _acc_vecs[i][J][v]

    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            is_up = (J % 2) == 1
            J_local = J // 2
            col_local = wave_n * fx.Int32(32) + fx.Int32(J_local * 16) + lane_mod_16
            lds_col = (fx.Int32(128) + col_local) if is_up else col_local
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + lds_col
                llvm.StoreOp(
                    _raw(fx.Float32(_acc(i, J, v))),
                    _gep3(lds_acc_base, idx * fx.Int32(4)),
                )

    gpu.barrier()

    tx_i32 = arith.index_cast(T.i32, gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(16)
    n_lane = tx_i32 % fx.Int32(16)
    wave_grp = n_lane // fx.Int32(4)
    kk = n_lane % fx.Int32(4)

    aqout_base = _global_base_ptr1(arg_aqout)
    scales_per_mr = [None] * kMChunks

    for mr in range_constexpr(kMChunks):
        row_local = fx.Int32(mr * 16) + m_lane
        gate_vs = [None] * 8
        up_vs = [None] * 8
        for ee in range_constexpr(8):
            col_in_grp = fx.Int32(8) * kk + fx.Int32(ee)
            gate_col = wave_grp * fx.Int32(32) + col_in_grp
            up_col = fx.Int32(128) + gate_col
            gate_off = (row_local * fx.Int32(BN) + gate_col) * fx.Int32(4)
            up_off = (row_local * fx.Int32(BN) + up_col) * fx.Int32(4)
            gate_vs[ee] = fx.Float32(llvm.load(T.f32, _gep3(lds_acc_base, gate_off)))
            up_vs[ee] = fx.Float32(llvm.load(T.f32, _gep3(lds_acc_base, up_off)))
        result = _silu_mul_batch(gate_vs, up_vs)

        local_max = _fabs_f32(result[0])
        for ee in range_constexpr(1, 8):
            local_max = local_max.maximumf(_fabs_f32(result[ee]))
        local_max = local_max.maximumf(local_max.shuffle_xor(fx.Int32(1), fx.Int32(64)))
        local_max = local_max.maximumf(local_max.shuffle_xor(fx.Int32(2), fx.Int32(64)))

        e8m0, qscale = _e8m0_from_amax(local_max, out_max)
        scales_per_mr[mr] = e8m0

        qscale_raw = _raw(qscale)
        # byte position of this lane's 8 elems (fp8 doubles it; row stride is INTER).
        byte_pos_fp4 = n_block_idx * fx.Int32(BN // 4) + wave_grp * fx.Int32(16) + kk * fx.Int32(4)
        out_row = m_row + row_local
        if const_expr(is_f8_out):
            # 8 f32 -> 8 fp8: lo holds elems 0..3, hi 4..7 (2 fp8 per cvt half).
            v2i16 = T.vec(2, T.i16)
            lo = _raw(Vec.filled(2, 0, fx.Int16))
            lo = rocdl.cvt_scalef32_pk_fp8_f32(v2i16, lo, _raw(result[0]), _raw(result[1]), qscale_raw, 0)
            lo = rocdl.cvt_scalef32_pk_fp8_f32(v2i16, lo, _raw(result[2]), _raw(result[3]), qscale_raw, 1)
            hi = _raw(Vec.filled(2, 0, fx.Int16))
            hi = rocdl.cvt_scalef32_pk_fp8_f32(v2i16, hi, _raw(result[4]), _raw(result[5]), qscale_raw, 0)
            hi = rocdl.cvt_scalef32_pk_fp8_f32(v2i16, hi, _raw(result[6]), _raw(result[7]), qscale_raw, 1)
            store_off = out_row * fx.Int32(K_G2_BYTES) + byte_pos_fp4 * fx.Int32(2)
            llvm.StoreOp(lo, _gep1(aqout_base, store_off), alignment=4, nontemporal=True)
            llvm.StoreOp(hi, _gep1(aqout_base, store_off + fx.Int32(4)), alignment=4, nontemporal=True)
        else:
            packed_i32 = _raw(fx.Int32(0))
            for w in range_constexpr(4):
                packed_i32 = rocdl.cvt_scalef32_pk_fp4_f32(
                    T.i32,
                    packed_i32,
                    _raw(result[2 * w]),
                    _raw(result[2 * w + 1]),
                    qscale_raw,
                    w,
                )
            store_off = out_row * fx.Int32(K_G2_BYTES) + byte_pos_fp4
            llvm.StoreOp(
                _raw(fx.Int32(packed_i32)),
                _gep1(aqout_base, store_off),
                alignment=4,
                nontemporal=True,
            )

    ascaleout_base = _global_base_ptr1(arg_ascaleout)
    if kk == fx.Int32(0):
        ku = n_block_idx >> fx.Int32(1)
        ikxdl = n_block_idx & fx.Int32(1)
        for sub in range_constexpr(kSubBlocks):
            chunk = m_block_idx * fx.Int32(kSubBlocks) + fx.Int32(sub)
            dword_off = chunk * fx.Int32(OUT_AS_PER_CHUNK_DW) + ku * fx.Int32(64) + wave_grp * fx.Int32(16) + m_lane
            pair_i32 = scales_per_mr[sub * 2 + 0] | (scales_per_mr[sub * 2 + 1] << fx.Int32(8))
            pair_i16 = arith.TruncIOp(T.i16, _raw(pair_i32)).result
            addr = dword_off * fx.Int32(4) + ikxdl * fx.Int32(2)
            llvm.StoreOp(
                pair_i16,
                _gep1(ascaleout_base, addr),
                alignment=2,
            )


def _lds_bytes_for(K_TILES_TOTAL, KH_TILE_A=KH_TILE):
    s_aq_bytes = kAStages * BM * KH_TILE_A  # fp8 A tile is 2x (256B vs 128B)
    s_asc_bytes = kSubBlocks * K_TILES_TOTAL * 256
    lds_acc_bytes = BM * BN * 4
    return max(s_aq_bytes + s_asc_bytes, lds_acc_bytes)


# ===========================================================================
# gemm2 (down-proj)
# ===========================================================================
def _issue_a_load_lds_dt(aq_rsrc, saq, slot, kt, m_row, wave, lane, is_f8, KH_TILE_A, K_BYTES):
    """A->LDS for one K-tile. gemm2 A is the already-sorted intermediate, so the row
    is the sorted row directly (no gather). Mirrors gemm1's issue_a_load_lds."""
    am = 2 if is_f8 else 1  # row-group calls per 8-row wave (fp8 4 rows/call)
    lanes_per_row = KH_TILE_A // 16  # 8 (fp4) / 16 (fp8)
    rows_per_call = 64 // lanes_per_row  # 8 (fp4) / 4 (fp8)
    a_lane_row = lane // fx.Int32(lanes_per_row)
    lane_col = (lane % fx.Int32(lanes_per_row)) * fx.Int32(16)
    base_i32 = fx.Int32(memref_dialect.extract_aligned_pointer_as_index(saq.get()))
    for h in range_constexpr(am):
        lds_row = wave * fx.Int32(BM // 4) + fx.Int32(h * rows_per_call)
        mask = (
            _lds_swizzle_mask_f8(lds_row + a_lane_row) if const_expr(is_f8) else _lds_swizzle_mask(lds_row + a_lane_row)
        )
        car = m_row + lds_row + a_lane_row  # direct sorted row
        voffset = (lane_col ^ mask) + car * fx.Int32(K_BYTES)
        off = fx.Int32(slot * (BM * KH_TILE_A)) + lds_row * fx.Int32(KH_TILE_A)
        rocdl.raw_ptr_buffer_load_lds(
            aq_rsrc,
            _lds_ptr3(base_i32, off),
            fx.Int32(16),
            voffset,
            fx.Int32(kt * KH_TILE_A),
            fx.Int32(0),
            fx.Int32(0),
        )


@flyc.jit
def _gemm2_body_v2(
    allocator,
    lds_off,
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
    aq_rsrc,
    *,
    use_nt,
    NE,
    N_OUT,
    D_INTER,
    aStages,
    a_dtype,
):
    _aStages = aStages
    # A dtype: fp4 (gemm1 fp4-out) or fp8 (mxfp8); only the A path differs.
    is_f8_a = a_dtype == "fp8"
    KH_TILE_A = BK // (1 if is_f8_a else 2)
    K_BYTES = D_INTER // (1 if is_f8_a else 2)
    slot_bytes = BM * KH_TILE_A
    cbsz_a = 0 if is_f8_a else 4
    # K-derived sizes (parametrized over contraction K = inter_dim = D_INTER).
    _K = D_INTER
    _K_HALF = k_half_for(_K)
    _K_TILES_TOTAL = k_tiles_total_for(_K)
    _kUnroll = kunroll_for(_K)
    _kAS_per_chunk_dw = kas_per_chunk_dw_for(_K)
    _kBS_stride_n0_dw = kbs_stride_n0_dw_for(_K)
    _kbs_per_expert_dw = kbs_per_expert_dw_for(N_OUT, _K)
    _num_n_blocks = num_n_blocks_for(N_OUT)
    KH4 = _K_HALF // 4

    # block -> (m_block_idx, n_block_idx) ; e = sorted_expert_ids[m_block_idx]
    m_block_idx = _udiv(bx_i32, _num_n_blocks)
    n_block_idx = bx_i32 - m_block_idx * fx.Int32(_num_n_blocks)
    e = rocdl.readfirstlane(T.i32, llvm.load(T.i32, _global_ptr1(arg_eids, m_block_idx * fx.Int32(4))))
    m_row = m_block_idx * fx.Int32(BM)

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

    # A-scale buffer resource + uniform base (A-scale load stays raw).
    _asc_per_mb = (BM // 32) * _kAS_per_chunk_dw * 4
    _asc_num = arith.index_cast(T.index, _raw(i32_max_m_blocks)) * fx.Index(_asc_per_mb)
    ascale_rsrc = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_ascale)), num_records_bytes=_asc_num)
    a_scale_s_base = rocdl.readfirstlane(T.i32, (m_row // fx.Int32(32)) * fx.Int32(_kAS_per_chunk_dw) * fx.Int32(4))
    v_voff_scale = ((lane_div_16 * fx.Int32(16)) + lane_mod_16) * fx.Int32(4)

    def load_a_scale_tile(kt):
        return buffer_ops.buffer_load(
            ascale_rsrc,
            (v_voff_scale + fx.Int32(kt * 256)) // fx.Int32(4),
            vec_width=1,
            dtype=T.i32,
            soffset_bytes=a_scale_s_base,
        )

    lds_base = allocator.get_base()
    saq = SmemPtr(lds_base, lds_off, T.i8, shape=(_aStages * slot_bytes,))
    lds_acc = SmemPtr(lds_base, lds_off, T.f32, shape=(BM * BN,))

    # -- B / B-scale layout-API views (shared primitives) ---------------------
    _b_copy_atom = b_copy_atom(use_nt)
    _bs_copy_atom = bscale_copy_atom()

    def _make_bq_view(j):
        col = n_block_idx * fx.Int32(BN) + wave * fx.Int32(BN // 4) + fx.Int32(j * 16)
        return bq_view(arg_bq, e * fx.Int32(N_OUT) + col, KH4, _K_TILES_TOTAL)

    _bq_views = [_make_bq_view(j) for j in range_constexpr(4)]

    mni_base = n_block_idx * fx.Int32(BN // 16 // 2) + wave * fx.Int32(BN // 64 // 2)
    _bscale_views = [
        bscale_view(
            arg_bscale,
            e * fx.Int32(_kbs_per_expert_dw) + (mni_base + fx.Int32(mw)) * fx.Int32(_kBS_stride_n0_dw),
            _K_TILES_TOTAL,
            k0_stride_dw=kBS_stride_k0_dw,
        )
        for mw in range_constexpr(2)
    ]

    # B / B-scale fragments are PER-TILE (all tiles loaded up front, B not LDS-bound);
    # A is refilled per K iter; C accumulates in place.
    _frag_tmpl = bq_frag_tmpl(_bq_views[0])  # i32<4:1>
    _bs_frag_tmpl = bscale_frag_tmpl(_bscale_views[0])  # i32<1:1>
    _bq_frags = [
        [[fx.make_fragment_like(_frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(4)]
        for _ in range_constexpr(_K_TILES_TOTAL)
    ]
    _bs_frags = [
        [fx.make_fragment_like(_bs_frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(_K_TILES_TOTAL)
    ]
    # fp4: A in fx.gemm fragments. fp8: A a per-iter Vec8 i32, C a raw f32x4 (accm).
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    _a_vals = _a_frags = _c_frags = accm = None
    if const_expr(is_f8_a):
        _a_vals = [[None, None] for _ in range(kMChunks)]
        accm = [[zero4 for _ in range(4)] for _ in range(kMChunks)]
    else:
        _a_frags = [[fx.make_fragment_like(_frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(kMChunks)]
        _c_frags = [
            [fx.make_fragment_like(_frag_tmpl, dtype=fx.Float32.ir_type) for _ in range_constexpr(4)]
            for _ in range_constexpr(kMChunks)
        ]

    def issue_b_load_tile(kt):
        for j in range_constexpr(4):
            for half in range_constexpr(2):
                fx.copy(
                    _b_copy_atom,
                    _bq_views[j][lane_div_16, lane_mod_16, kt, half, None],
                    _bq_frags[kt][j][half],
                )

    def issue_b_scale_tile(kt):
        for mw in range_constexpr(2):
            fx.copy(
                _bs_copy_atom,
                _bscale_views[mw][lane_div_16, lane_mod_16, kt, None],
                _bs_frags[kt][mw],
            )

    def issue_a_ds_read(slot):
        _issue_a_ds_read_dt(saq, slot, slot_bytes, KH_TILE_A, lane_div_16, lane_mod_16, is_f8_a, _a_vals, _a_frags)

    def issue_a_load_lds(slot, kt):
        _issue_a_load_lds_dt(aq_rsrc, saq, slot, kt, m_row, wave, lane, is_f8_a, KH_TILE_A, K_BYTES)

    _scale_mma_atoms = scale_mma_atoms() if const_expr(not is_f8_a) else None

    def mfma_cluster(kt, sa):
        # opsel (gemm2 has no gate/up split): mni=J//2, in_b=J%2.
        for J in range_constexpr(4):
            mni, in_b = J // 2, J % 2
            sb = _raw(Vec(_bs_frags[kt][mni].load())[0])
            _mma_one_j(
                J, in_b, sa, sb, _bq_frags[kt], is_f8_a, cbsz_a, _a_vals, _a_frags, accm, _c_frags, _scale_mma_atoms
            )

    # zero C (fp4 fragments accumulate in place; fp8 accm pre-init above).
    if const_expr(not is_f8_a):
        for i in range_constexpr(kMChunks):
            for J in range_constexpr(4):
                _c_frags[i][J].store(zero4)

    # Load ALL B-q + B-scale + A-scale tiles up front (B is not LDS-bound), as v1.
    a_scale_v = [load_a_scale_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
    for kt in range_constexpr(_K_TILES_TOTAL):
        issue_b_load_tile(kt)
        issue_b_scale_tile(kt)

    if const_expr(_K_TILES_TOTAL <= kStages):
        # Fast path: all tiles preloaded in LDS by the kernel.
        for kt in range_constexpr(_K_TILES_TOTAL):
            gpu.barrier()
            issue_a_ds_read(kt % kStages)
            mfma_cluster(kt, a_scale_v[kt])
    else:
        # Streaming double-buffered K-loop (triple-buffered LDS): process tile
        # kt=OFFSET (read slot kt%aStages) and stream the next tile into its slot.
        for OFFSET in range_constexpr(_kUnroll):
            kt = OFFSET
            gpu.barrier()
            issue_a_ds_read(kt % _aStages)
            issue_a_load_lds((kStages + OFFSET) % _aStages, kStages + OFFSET)
            mfma_cluster(kt, a_scale_v[kt])
        for S in range_constexpr(kStages):
            kt = _K_TILES_TOTAL - kStages + S
            gpu.barrier()
            issue_a_ds_read(kt % _aStages)
            mfma_cluster(kt, a_scale_v[kt])

    # epilog: atomic bf16. fp8 reads accm; fp4 loads the C fragments.
    saq._view_cache = None
    lds_acc._view_cache = None
    if const_expr(is_f8_a):
        accm_vecs = accm
    else:
        accm_vecs = [[_c_frags[i][J].load() for J in range(4)] for i in range(kMChunks)]
    _atomic_bf16_epilog(
        lds_acc,
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
        N_OUT,
    )


# ===========================================================================
# Atomic bf16 epilogue (shared store path; gemm2 down-proj)
# ===========================================================================
def _atomic_bf16_epilog(
    lds_acc,
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
):
    _kMChunks = kmchunks(BM)
    M_REPS = BM // 8  # BM32: 4, BM16: 2
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base = _lds_base_ptr3(lds_acc.get())

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)
    stids_base = _global_base_ptr1(arg_stids)
    sweights_base = _global_base_ptr1(arg_sweights)
    out_base = _global_base_ptr1(arg_out)

    # Prefetch sorted_token_ids / sorted_weights (invariant) before the stores +
    # barriers so their global latency overlaps instead of hitting the atomic loop.
    packed = []
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + fx.Int32(mr * 8) + m_lane
        packed.append(llvm.load(T.i32, _gep1(stids_base, sorted_pos * fx.Int32(4)), invariant=True))
        weight.append(llvm.load(T.f32, _gep1(sweights_base, sorted_pos * fx.Int32(4)), invariant=True))

    # pre-store fence+barrier (HIP run_one __syncthreads() before the epilog).
    gpu.barrier()

    # write accm -> lds_acc cshuffle (scalar f32 stores, as HIP does)
    for i in range_constexpr(_kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            col = wave * fx.Int32(64) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(vec[v]), _gep3(lds_base, idx * fx.Int32(4)))

    gpu.barrier()

    # read back + weighted atomic add (token_id / weight prefetched above)
    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:
            row_base_addr = token_id * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + col_start
            for s in range_constexpr(4):
                # adjacent ee=0,1 are contiguous -> one <2xf32> load (as HIP vectorizes)
                idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                v2 = Vec(llvm.load(T.vec(2, T.f32), _gep3(lds_base, idx0 * fx.Int32(4))))
                pk = Vec.from_elements([v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32).to(fx.BFloat16)
                off = (row_base_addr + fx.Int32(s * 64)) * fx.Int32(2)  # bf16 byte off
                out_ptr = _gep1(out_base, off)
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    out_ptr,
                    _raw(pk),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )
