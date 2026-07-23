# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Grouped-MoE GEMM2 (down-proj), ported from aiter mxmoe_gemm_v2.py (layout-API, BM32).

`gemm2_body_v2` is the per-tile device body (A2 sorted-row @ W2 -> f32 acc); the outer persistent
kernel `compile_group_gemm2` drives it over the (m_block, n_block) tile grid. The default epilogue is
the LOCAL weighted atomic-bf16 combine (`atomic_bf16_epilog`); the cross-rank P2P-scatter combine
lives in mega_moe_stage2.py. CDNA4-only (mfma_scale_f32_16x16x128_f8f6f4).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Float4E2M1FN, T
from flydsl.expr.typing import Vector as Vec
from kernels.common.mem_ops import buffer_atomic_add
from kernels.gemm.fp8_gemm_utils import ceildiv


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def global_base_ptr1(addr_i64):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(addr_i64)))


def gep1(base_ptr, byte_off):
    return buffer_ops.get_element_ptr(base_ptr, byte_offset=_raw(byte_off), elem_type=T.i8)


def spart_output_tile_index(block_1d_id, M0, N0, group_num, m01):
    """ck_tile GemmSpatiallyLocalTilePartitioner: 1D block id -> spatially-local (m_block, n_block).

    block_1d_id/M0 runtime; N0/group_num/m01 compile-time. Mirrors aiter mxmoe_dispatcher; improves
    L2 locality (analogue of the production xcd_swizzle). g2_spart encodes group_num*100 + m01.
    """
    gn = fx.Int32(group_num)
    n0 = fx.Int32(N0)
    m01c = fx.Int32(m01)
    mn = M0 * n0
    group_size = (mn + gn - fx.Int32(1)) // gn
    big_group_num = gn - (group_size * gn - mn)
    group_id_y = block_1d_id // gn
    group_id_x = block_1d_id - group_id_y * gn
    remap_a = group_id_x * group_size + group_id_y
    remap_b = group_id_x * group_size + big_group_num - group_id_x + group_id_y
    remap = (group_id_x <= big_group_num).select(remap_a, remap_b)
    idx_M0 = remap // n0
    idx_N0 = remap - idx_M0 * n0
    M0_tmp = M0 // m01c
    M0_mod = M0 - M0_tmp * m01c
    M01_adapt = (idx_M0 < (M0 - M0_mod)).select(m01c, M0_mod)
    idx_M00 = idx_M0 // m01c
    idx_M01 = idx_M0 - idx_M00 * m01c
    idx_local = idx_N0 + idx_M01 * n0
    n_out = idx_local // M01_adapt
    loc_mod = idx_local - n_out * M01_adapt
    m_block_idx = loc_mod + idx_M00 * m01c
    n_block_idx = n_out
    return m_block_idx, n_block_idx


def spart_group_m01(g2_spart):
    """Decode g2_spart (GroupNum*100 + M01) -> (group_num, m01); 0 -> naive (None)."""
    if not g2_spart or g2_spart <= 0:
        return None
    group_num = g2_spart // 100
    m01 = g2_spart % 100
    assert group_num >= 1 and m01 >= 1, f"g2_spart={g2_spart} must encode GroupNum>=1,M01>=1"
    return group_num, m01


def waitcnt_barrier(vmcnt=63, lgkmcnt=63):
    """s_waitcnt + s_barrier via inline asm (mega-internal; avoids kernels/moe dep).

    Bypasses LLVM SIInsertWaitcnts's conservative vmcnt(0) lgkmcnt(0) before every s_barrier so
    remote SLC P2P stores can drain async (xGMI overlap) while we only wait on LDS.
    """
    parts = []
    if vmcnt < 63 or lgkmcnt < 63:
        wc = []
        if vmcnt < 63:
            wc.append(f"vmcnt({vmcnt})")
        if lgkmcnt < 63:
            wc.append(f"lgkmcnt({lgkmcnt})")
        parts.append("s_waitcnt " + " ".join(wc))
    parts.append("s_barrier")
    llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string="\n".join(parts),
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
    )


# -- tiling constants (BM32, from aiter mxmoe_gemm_v2) --
BN = BK = 256
KH_TILE = BK // 2  # 128 packed-fp4 bytes per K-tile
kStages = 2
kBS_stride_k0_dw = 64
BM = 32
kAStages = 3
kSubBlocks = 1
kMChunks = 2  # BM // 16


# -- pointer / LDS helpers ----------------------------------------------------
def lds_dma_dst(base_i32, byte_off_i32, elem_ty=None, align=16):
    if elem_ty is None:
        elem_ty = T.i32
    lds_ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Shared, align)
    lds_ptr = fx.inttoptr(lds_ptr_ty, fx.Int32(base_i32 + byte_off_i32))
    return fx.make_view(lds_ptr, fx.make_layout(1, 1))


def flat_buffer_view(arg, base_elems, elem_ty, *, align, elem_bytes, fold=True, num_records_bytes=None):
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Global, align)
    if fold:
        base = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(base_elems)))
        off_i64 = fx.Int64(fx.Uint32(base))
        base_iter = fx.inttoptr(ptr_ty, fx.Int64(arg) + off_i64 * fx.Int64(elem_bytes))
    else:
        base_iter = fx.inttoptr(ptr_ty, fx.Int64(arg))
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout((1, 1), (1, 1))))
    if num_records_bytes is not None:
        return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)
    return fx.rocdl.make_buffer_tensor(view, max_size=True)


def lds_typed_ptr(base_i32, elem_ty, align=4):
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Shared, align)
    return fx.inttoptr(ptr_ty, fx.Int32(base_i32))


def lds_vec_load(base_i32, byte_off_i32, result_type, elem_ty, align=4):
    elem_ir_ty = elem_ty.ir_type if hasattr(elem_ty, "ir_type") else elem_ty
    ptr = lds_typed_ptr(fx.Int32(base_i32) + byte_off_i32, elem_ir_ty, align=align)
    return fx.ptr_load(ptr, result_type=result_type)


def lds_swizzle_mask(row):
    return (row & 14) << 3


def lds_swizzle_mask_f8(row):
    return (row & 15) << 4


def global_typed_ptr(arg, elem_ty, align=4):
    """Typed global fx.Pointer over a raw i64 device address; index in ELEMENTS."""
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Global, align)
    return fx.inttoptr(ptr_ty, fx.Int64(arg))


# -- B / B-scale data movement + scaled MFMA (shared layout-API primitives) ----
def lds_dma_atom_128():
    return fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)


def b_copy_atom(nontemporal):
    return fx.make_copy_atom(fx.rocdl.BufferCopy128b(2 if nontemporal else 0), 32)


def bscale_copy_atom():
    return fx.make_copy_atom(fx.rocdl.BufferCopy32b(0), 32)


def bq_view(arg_bq, row_elems, KH4, K_TILES_TOTAL, num_records_bytes=None):
    col_base = rocdl.readfirstlane(T.i32, fx.Int32(row_elems) * fx.Int32(KH4))
    i32_ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=16)
    off_i64 = fx.Int64(col_base)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_bq) + off_i64 * fx.Int64(4))
    shape = (4, 16, K_TILES_TOTAL, 2, 4)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, (64, 4, 512, 256, 1))))
    # has_pad OOB pad-skip: num_records_bytes sizes the buffer to REAL K so pad reads -> 0.
    if num_records_bytes is not None:
        return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def bscale_view(arg_bscale, base_dw, K_TILES_TOTAL, k0_stride_dw=64):
    base_dw = rocdl.readfirstlane(T.i32, fx.Int32(base_dw))
    i32_ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
    off_i64 = fx.Int64(base_dw)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_bscale) + off_i64 * fx.Int64(4))
    shape = (4, 16, K_TILES_TOTAL, 1)
    stride = (16, 1, k0_stride_dw, 1)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, stride)))
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def bq_frag_tmpl(view):
    return view[0, 0, 0, 0, None]


def bscale_frag_tmpl(view):
    return view[0, 0, 0, None]


def scale_mma_atoms(is_f8_a):
    elem_a = fx.Float8E4M3FN if is_f8_a else Float4E2M1FN
    return {
        (osa, osb): fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                16,
                16,
                128,
                elem_a,
                Float4E2M1FN,
                opsel_a=osa,
                opsel_b=osb,
            )
        )
        for osa in range(4)
        for osb in range(4)
    }


def gemm_mma(atoms, a_frag, b_frag, c_frag, opsel_a, opsel_b, sa, sb):
    fx.gemm(atoms[(opsel_a, opsel_b)], c_frag, a_frag, b_frag, c_frag, scale_a=sa, scale_b=sb)


def issue_a_ds_read_dt(s_aq_base, slot, slot_bytes, KH_TILE_A, lane_div_16, lane_mod_16, is_f8, a_frags):
    for k in range_constexpr(2):
        for i in range_constexpr(kMChunks):
            lds_row = lane_mod_16 + i * 16
            row_off = fx.Int32(slot * slot_bytes) + lds_row * KH_TILE_A
            if const_expr(is_f8):
                mask = lds_swizzle_mask_f8(lane_mod_16)
                col0 = lane_div_16 * 16 + k * 128
                col_lo = col0 ^ mask
                col_hi = (col0 + 64) ^ mask
                lo = Vec(lds_vec_load(s_aq_base, row_off + col_lo, Vec.make_type(2, fx.Int64), fx.Int64, align=16))
                hi = Vec(lds_vec_load(s_aq_base, row_off + col_hi, Vec.make_type(2, fx.Int64), fx.Int64, align=16))
                a64 = Vec.from_elements([lo[0], lo[1], hi[0], hi[1]], fx.Int64)
                a_frags[i][k].store(a64.bitcast(fx.Int32))
            else:
                mask = lds_swizzle_mask(lane_mod_16)
                lds_col = (lane_div_16 * 16 + k * 64) ^ mask
                vec = lds_vec_load(s_aq_base, row_off + lds_col, Vec.make_type(4, fx.Int32), fx.Int32, align=16)
                a_frags[i][k].store(Vec(vec))


def mma_one_j(J, in_b, sa, sb, bq_frags_kt, a_frags, c_frags, atoms):
    bJ0, bJ1 = bq_frags_kt[J][0], bq_frags_kt[J][1]
    gemm_mma(atoms, a_frags[0][0], bJ0, c_frags[0][J], 0, 0 + in_b, sa, sb)
    gemm_mma(atoms, a_frags[1][0], bJ0, c_frags[1][J], 1, 0 + in_b, sa, sb)
    gemm_mma(atoms, a_frags[0][1], bJ1, c_frags[0][J], 2, 2 + in_b, sa, sb)
    gemm_mma(atoms, a_frags[1][1], bJ1, c_frags[1][J], 3, 2 + in_b, sa, sb)


def issue_a_load_lds_dt(arg_aq, aq_num_records, s_aq_base, slot, kt, m_row, wave, lane, is_f8, KH_TILE_A, K_BYTES):
    am = 2 if is_f8 else 1
    lanes_per_row = KH_TILE_A // 16
    rows_per_call = 64 // lanes_per_row
    a_lane_row = lane // lanes_per_row
    lane_col = (lane % lanes_per_row) * 16
    base_i32 = s_aq_base
    atom = lds_dma_atom_128()
    src = flat_buffer_view(arg_aq, None, T.i32, align=16, elem_bytes=4, fold=False, num_records_bytes=aq_num_records)
    for h in range_constexpr(am):
        lds_row = wave * (BM // 4) + h * rows_per_call
        mask = (
            lds_swizzle_mask_f8(lds_row + a_lane_row) if const_expr(is_f8) else lds_swizzle_mask(lds_row + a_lane_row)
        )
        car = m_row + lds_row + a_lane_row  # direct sorted row
        voffset = (lane_col ^ mask) + car * K_BYTES
        off = fx.Int32(slot * (BM * KH_TILE_A)) + lds_row * KH_TILE_A
        v_e = (voffset + kt * KH_TILE_A) // 4
        fx.copy(atom, src[v_e, None], lds_dma_dst(base_i32, off, elem_ty=T.i32, align=16))


def gemm2_compute(
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
    *,
    N_OUT,
    D_INTER,
    aStages,
    a_dtype,
    use_nt,
    expert_offset=0,
    SBM=None,
    inter_dim_pad=0,
    model_dim_pad=0,
    g2_bhoist=True,
    g2_ascale_pf=True,
):
    """A2 @ W2 -> per-tile f32 accumulators (returns accm_vecs [kMChunks][4] of Vec4 f32).

    Splits the compute from the epilogue so both the local-atomic and the P2P-scatter combine can
    reuse it. block -> (m_block, n_block); e = eids[m_block]; A2 row = m_block*BM + ... (direct sorted).
    """
    is_f8_a = a_dtype == "fp8"
    KH_TILE_A = BK // (1 if is_f8_a else 2)
    K_BYTES = D_INTER // (1 if is_f8_a else 2)
    slot_bytes = BM * KH_TILE_A
    K = D_INTER
    kc = (K // 32) // 4 // 2
    K_HALF = K // 2
    K_TILES_TOTAL = K // BK
    kAS_per_chunk_dw = kc * 64
    kBS_stride_n0_dw = kc * 64
    kbs_per_expert_dw = (N_OUT // 16 // 2) * kBS_stride_n0_dw
    num_n_blocks = N_OUT // 256
    KH4 = K_HALF // 4

    # has_pad OOB pad-skip (compile-time): K-skip sizes the 16N B-weight buffer to REAL K;
    # N-skip zeros fully-pad-N w2 tiles (col >= N_real). B-scale is NOT shrunk.
    has_pad = (inter_dim_pad > 0) or (model_dim_pad > 0)
    bq_num_records = None
    N_real = None
    if const_expr(has_pad):
        K_real = D_INTER - inter_dim_pad
        halves_real = (K_real + 127) // 128
        bq_num_records = halves_real * 1024
        N_real = N_OUT - model_dim_pad

    # SBM (sort padding unit) >= BM (compute tile); SBM==BM default byte-identical.
    _SBM = BM if SBM is None else SBM
    m_block_idx = bx_i32 // num_n_blocks
    n_block_idx = bx_i32 - m_block_idx * num_n_blocks
    m_row = m_block_idx * BM
    # e = sorted_expert_ids[sort_block]; SBM==BM -> sort_block == m_block_idx.
    _eid_idx = m_block_idx if const_expr(_SBM == BM) else (m_row // fx.Int32(_SBM))
    eids_rsrc = buffer_ops.create_buffer_resource_from_addr(fx.Int64(arg_eids))
    e = fx.Int32(
        rocdl.readfirstlane(
            T.i32,
            buffer_ops.buffer_load(eids_rsrc, _eid_idx, vec_width=1, dtype=fx.Int32),
        )
    )
    if const_expr(expert_offset != 0):
        e = e - fx.Int32(expert_offset)

    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16

    asc_per_mb = (BM // 32) * kAS_per_chunk_dw * 4
    asc_num = fx.Int64(i32_max_m_blocks) * fx.Int64(asc_per_mb)
    ascale_rsrc = buffer_ops.create_buffer_resource_from_addr(fx.Int64(arg_ascale), num_records_bytes=asc_num)
    a_scale_s_base = rocdl.readfirstlane(T.i32, (m_row // 32) * kAS_per_chunk_dw * 4)
    v_voff_scale = ((lane_div_16 * 16) + lane_mod_16) * 4

    def load_a_scale_tile(kt):
        return buffer_ops.buffer_load(
            ascale_rsrc, (v_voff_scale + kt * 256) // 4, vec_width=1, dtype=T.i32, soffset_bytes=a_scale_s_base
        )

    s_aq_base = lds_base_i32

    b_catom = b_copy_atom(use_nt)
    bs_copy_atom = bscale_copy_atom()

    def make_bq_view(j):
        col = n_block_idx * BN + wave * (BN // 4) + j * 16
        nrec = None
        if const_expr(has_pad):
            # N-skip: fully-pad-N tile (col >= N_real) -> 0 records so weight loads OOB -> 0.
            nrec = (col < fx.Int32(N_real)).select(fx.Int32(bq_num_records), fx.Int32(0))
        return bq_view(arg_bq, e * N_OUT + col, KH4, K_TILES_TOTAL, num_records_bytes=nrec)

    bq_views = [make_bq_view(j) for j in range_constexpr(4)]

    mni_base = n_block_idx * (BN // 16 // 2) + wave * (BN // 64 // 2)
    bscale_views = [
        bscale_view(
            arg_bscale,
            e * kbs_per_expert_dw + (mni_base + mw) * kBS_stride_n0_dw,
            K_TILES_TOTAL,
            k0_stride_dw=kBS_stride_k0_dw,
        )
        for mw in range_constexpr(2)
    ]

    frag_tmpl = bq_frag_tmpl(bq_views[0])
    bs_frag_tmpl = bscale_frag_tmpl(bscale_views[0])
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    a_ndw = 8 if is_f8_a else 4
    a_frags = [[fx.make_rmem_tensor(a_ndw, fx.Int32) for _ in range_constexpr(2)] for _ in range_constexpr(kMChunks)]
    c_frags = [[fx.make_rmem_tensor(4, fx.Float32) for _ in range_constexpr(4)] for _ in range_constexpr(kMChunks)]

    # B weight + B-scale streamed per K-tile into a 2-stage register double buffer (g2_bhoist):
    # only cur/nxt fragments are register-resident (vs one set per K-tile), cutting VGPR pressure
    # at large inter_dim. g2_ascale_pf carries A-scale through the same cur/nxt rotation.
    def _new_bqf():
        return [[fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(4)]

    def _new_bsf():
        return [fx.make_fragment_like(bs_frag_tmpl) for _ in range_constexpr(2)]

    def issue_b_load_into(bqf, bsf, kt):
        for j in range_constexpr(4):
            for half in range_constexpr(2):
                fx.copy(b_catom, bq_views[j][lane_div_16, lane_mod_16, kt, half, None], bqf[j][half])
        for mw in range_constexpr(2):
            fx.copy(bs_copy_atom, bscale_views[mw][lane_div_16, lane_mod_16, kt, None], bsf[mw])

    def issue_a_ds_read(slot):
        issue_a_ds_read_dt(s_aq_base, slot, slot_bytes, KH_TILE_A, lane_div_16, lane_mod_16, is_f8_a, a_frags)

    aq_num_records = fx.Int64(i32_max_m_blocks) * fx.Int64(BM * K_BYTES)

    def issue_a_load_lds(slot, kt):
        issue_a_load_lds_dt(arg_aq, aq_num_records, s_aq_base, slot, kt, m_row, wave, lane, is_f8_a, KH_TILE_A, K_BYTES)

    mma_atoms = scale_mma_atoms(is_f8_a)

    def mfma_cluster(bqf, bsf, sa):
        for J in range_constexpr(4):
            mni, in_b = J // 2, J % 2
            sb = Vec(bsf[mni].load())[0]
            mma_one_j(J, in_b, sa, sb, bqf, a_frags, c_frags, mma_atoms)

    for i in range_constexpr(kMChunks):
        for J in range_constexpr(4):
            c_frags[i][J].store(zero4)

    cur_bqf, cur_bsf = _new_bqf(), _new_bsf()
    nxt_bqf, nxt_bsf = _new_bqf(), _new_bsf()

    # Prologue: prime first min(kStages, K_TILES) A-tiles into LDS + tile-0 B/scale into cur.
    n_a_pre = min(kStages, K_TILES_TOTAL)
    for kt in range_constexpr(n_a_pre):
        issue_a_load_lds(kt, kt)
    issue_b_load_into(cur_bqf, cur_bsf, 0)
    cur_sa = load_a_scale_tile(0) if const_expr(g2_ascale_pf) else None

    for kt in range_constexpr(K_TILES_TOTAL):
        _has_next = kt + 1 < K_TILES_TOTAL
        # g2_bhoist: prefetch next-tile B (+A-scale) above the LDS barrier so vmem rides ahead.
        if const_expr(g2_bhoist and _has_next):
            issue_b_load_into(nxt_bqf, nxt_bsf, kt + 1)
            if const_expr(g2_ascale_pf):
                nxt_sa = load_a_scale_tile(kt + 1)
        gpu.barrier()
        issue_a_ds_read(kt % aStages)
        if const_expr(kt + kStages < K_TILES_TOTAL):
            issue_a_load_lds((kt + kStages) % aStages, kt + kStages)
        sa = cur_sa if const_expr(g2_ascale_pf) else load_a_scale_tile(kt)
        if const_expr((not g2_bhoist) and _has_next):
            issue_b_load_into(nxt_bqf, nxt_bsf, kt + 1)
            if const_expr(g2_ascale_pf):
                nxt_sa = load_a_scale_tile(kt + 1)
        mfma_cluster(cur_bqf, cur_bsf, sa)
        # rotate cur <- nxt (prefetched values become current next iteration).
        cur_bqf, nxt_bqf = nxt_bqf, cur_bqf
        cur_bsf, nxt_bsf = nxt_bsf, cur_bsf
        if const_expr(g2_ascale_pf and _has_next):
            cur_sa = nxt_sa

    accm_vecs = [[Vec(c_frags[i][J].load()) for J in range(4)] for i in range(kMChunks)]
    return accm_vecs, m_row, n_block_idx


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
    N_OUT,
    *,
    use_reduce=False,
    topk=1,
    g2_bf16_lds=False,
    guard_padding=True,
):
    """LOCAL weighted bf16 combine (single-rank / intra-node).

    Default: weighted atomic-fadd into out[token_id]. use_reduce: nontemporal store into
    out[token_id*topk + slot] (slot = stids>>24). g2_bf16_lds: cshuffle in bf16 (weight baked at
    write) instead of f32. guard_padding gates padding rows (token_id >= i32_M) via a device if.
    """
    kMChunks_l = BM // 16
    M_REPS = BM // 8
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    lds_base_fptr = lds_typed_ptr(lds_acc_base, T.f32)
    lds_base_bf16 = lds_typed_ptr(lds_acc_base, T.bf16, align=2) if const_expr(g2_bf16_lds) else None

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // 32
    n_lane = tx_i32 % 32
    col_start = n_lane * 2
    stids_rsrc = buffer_ops.create_buffer_resource_from_addr(fx.Int64(arg_stids))
    sweights_rsrc = buffer_ops.create_buffer_resource_from_addr(fx.Int64(arg_sweights))
    out_rsrc = buffer_ops.create_buffer_resource_from_addr(fx.Int64(arg_out))

    packed = []
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + mr * 8 + m_lane
        packed.append(buffer_ops.buffer_load(stids_rsrc, sorted_pos, vec_width=1, dtype=fx.Int32))
        weight.append(buffer_ops.buffer_load(sweights_rsrc, sorted_pos, vec_width=1, dtype=fx.Float32))

    gpu.barrier()

    # cshuffle accm -> LDS. bf16 path bakes the routing weight in at write time; f32 defers to readback.
    if const_expr(g2_bf16_lds):
        for i in range_constexpr(kMChunks_l):
            row_base = fx.Int32(i * 16) + lane_div_16 * 4
            w_row = [
                buffer_ops.buffer_load(sweights_rsrc, m_row + row_base + v, vec_width=1, dtype=fx.Float32)
                for v in range_constexpr(4)
            ]
            for J in range_constexpr(4):
                col = wave * 64 + J * 16 + lane_mod_16
                vec = Vec(accm[i][J])
                for v in range_constexpr(4):
                    idx = (row_base + v) * BN + col
                    lds_base_bf16[idx] = fx.BFloat16(fx.Float32(vec[v]) * fx.Float32(w_row[v]))
    else:
        for i in range_constexpr(kMChunks_l):
            row_base = fx.Int32(i * 16) + lane_div_16 * 4
            for J in range_constexpr(4):
                col = wave * 64 + J * 16 + lane_mod_16
                vec = Vec(accm[i][J])
                for v in range_constexpr(4):
                    idx = (row_base + v) * BN + col
                    lds_base_fptr[idx] = fx.Float32(vec[v])

    gpu.barrier()

    def _store_one(mr):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if const_expr(use_reduce):
            out_row = fx.Int64(token_id * fx.Int32(topk) + (packed[mr] >> fx.Int32(24)))
            row_base_addr = out_row * fx.Int64(N_OUT) + fx.Int64(n_block_idx * BN + col_start)
        else:
            row_base_addr = token_id * N_OUT + n_block_idx * BN + col_start
        for s in range_constexpr(4):
            idx0 = row_in_block * BN + col_start + s * 64
            if const_expr(g2_bf16_lds):
                pk = Vec(lds_vec_load(lds_acc_base, idx0 * 2, Vec.make_type(2, fx.BFloat16), fx.BFloat16, align=4))
            else:
                v2 = Vec(lds_vec_load(lds_acc_base, idx0 * 4, Vec.make_type(2, fx.Float32), fx.Float32, align=8))
                pk = Vec.from_elements([v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32).to(fx.BFloat16)
            if const_expr(use_reduce):
                off = (row_base_addr + fx.Int64(s * 64)) * fx.Int64(2)
                out_ptr = gep1(global_base_ptr1(arg_out), off)
                llvm.StoreOp(_raw(pk), out_ptr, alignment=4, nontemporal=True)
            else:
                off = (row_base_addr + s * 64) * 2
                buffer_atomic_add(pk, out_rsrc, off, fx.Int32(0), fx.Int32(0))

    for mr in range_constexpr(M_REPS):
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if const_expr(guard_padding):
            if token_id < i32_M:
                _store_one(mr)
        else:
            _store_one(mr)


def lds_bytes_for_gemm2(D_INTER, a_dtype, aStages=kAStages, g2_bf16_lds=False):
    KH_TILE_A = BK // (1 if a_dtype == "fp8" else 2)
    s_aq_bytes = aStages * BM * KH_TILE_A
    lds_acc_bytes = BM * BN * (2 if g2_bf16_lds else 4)
    return max(s_aq_bytes, lds_acc_bytes)


def compile_group_gemm2(
    *,
    model_dim: int,  # N_OUT (down-proj output)
    inter_dim: int,  # D_INTER (contraction)
    experts: int,
    a_dtype: str = "fp8",
    aStages: int = kAStages,
    use_nt: bool = True,
    num_cu: int = 256,
    grid_mult: int = 1,
    SBM: int = None,
    inter_dim_pad: int = 0,
    model_dim_pad: int = 0,
    g2_bhoist: bool = True,
    g2_ascale_pf: bool = True,
    g2_bf16_lds: bool = False,
    g2_spart: int = 0,
    epilog: str = "atomic",
    topk: int = 1,
    expert_offset: int = 0,
):
    N_OUT = model_dim
    D_INTER = inter_dim
    num_n_blocks = N_OUT // 256
    assert N_OUT % 256 == 0
    _spart = spart_group_m01(g2_spart)
    use_reduce = epilog == "reduce"
    g2_bf16_lds = bool(g2_bf16_lds) and use_reduce
    grid_x = num_cu * grid_mult
    lds_bytes = lds_bytes_for_gemm2(D_INTER, a_dtype, aStages, g2_bf16_lds)
    TOTAL_THREADS = 256  # 4 waves

    @flyc.kernel(known_block_size=[TOTAL_THREADS, 1, 1])
    def kernel(
        addr_out: fx.Int64,
        addr_aq: fx.Int64,
        addr_ascale: fx.Int64,
        addr_bq: fx.Int64,
        addr_bscale: fx.Int64,
        addr_eids: fx.Int64,
        addr_stids: fx.Int64,
        addr_sweights: fx.Int64,
        addr_num_valid: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
    ):
        lds_base_i32 = fx.Int32(0)  # dynamic LDS starts at offset 0 (no static shared in this kernel)
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid % fx.Int32(64)
        wave = tid // fx.Int32(64)

        nv_rsrc = buffer_ops.create_buffer_resource_from_addr(addr_num_valid)
        num_valid = buffer_ops.buffer_load(nv_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Int32)
        num_m_tiles = ceildiv(num_valid, fx.Int32(BM))
        total_work = num_m_tiles * fx.Int32(num_n_blocks)

        itv = fx.Int32(fx.block_idx.x)
        while itv < total_work:
            # g2_spart: remap the linear tile id to a spatially-local (m,n) for L2 reuse.
            if const_expr(_spart is not None):
                _mb, _nb = spart_output_tile_index(itv, num_m_tiles, num_n_blocks, _spart[0], _spart[1])
                unit_bx = _mb * fx.Int32(num_n_blocks) + _nb
            else:
                unit_bx = itv
            accm_vecs, m_row, n_block_idx = gemm2_compute(
                lds_base_i32,
                addr_ascale,
                addr_bq,
                addr_bscale,
                addr_eids,
                addr_aq,
                i32_max_m_blocks,
                unit_bx,
                lane,
                wave,
                N_OUT=N_OUT,
                D_INTER=D_INTER,
                aStages=aStages,
                a_dtype=a_dtype,
                use_nt=use_nt,
                expert_offset=expert_offset,
                SBM=SBM,
                inter_dim_pad=inter_dim_pad,
                model_dim_pad=model_dim_pad,
                g2_bhoist=g2_bhoist,
                g2_ascale_pf=g2_ascale_pf,
            )
            atomic_bf16_epilog(
                lds_base_i32,
                accm_vecs,
                addr_out,
                addr_stids,
                addr_sweights,
                m_row,
                n_block_idx,
                wave,
                lane,
                i32_M,
                N_OUT,
                use_reduce=use_reduce,
                topk=topk,
                g2_bf16_lds=g2_bf16_lds,
            )
            itv = itv + fx.Int32(grid_x)

    @flyc.jit
    def launch(
        addr_out: fx.Int64,
        addr_aq: fx.Int64,
        addr_ascale: fx.Int64,
        addr_bq: fx.Int64,
        addr_bscale: fx.Int64,
        addr_eids: fx.Int64,
        addr_stids: fx.Int64,
        addr_sweights: fx.Int64,
        addr_num_valid: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            addr_out,
            addr_aq,
            addr_ascale,
            addr_bq,
            addr_bscale,
            addr_eids,
            addr_stids,
            addr_sweights,
            addr_num_valid,
            i32_M,
            i32_max_m_blocks,
        ).launch(grid=(grid_x, 1, 1), block=(TOTAL_THREADS, 1, 1), smem=lds_bytes, stream=stream)

    return launch
