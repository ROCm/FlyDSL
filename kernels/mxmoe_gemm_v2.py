# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Layout-API MXFP4 MoE GEMM device bodies (BM32): gemm1 up/gate + gemm2 down."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import _to_raw as _raw
from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Float4E2M1FN, T
from flydsl.expr.typing import Vector as Vec

from .fp8_gemm_utils import flat_buffer_view, lds_dma_atom_128

# -- shape constants (KIMI defaults; per-shape values come from the compile args) --
NE = 385  # #experts
TOPK_DEFAULT = 9
H_DEFAULT = 7168  # model_dim: gemm1 D_HIDDEN (contraction) / gemm2 N_OUT (output)
INTER_DEFAULT = 512  # inter_dim: gemm1 D_INTER (output) / gemm2 D_INTER (contraction)
INTER_MAX_DEFAULT = 8192  # compile-time cap for runtime inter_dim (gemm2 B-view / LDS bounds)
MAX_M = 655360
# tiling (BM-independent).
BN = BK = 256
KH_TILE = BK // 2  # 128 packed-fp4 bytes per K-tile
kStages = 2
kBS_stride_k0_dw = 64  # e8m0 scale-layout K-independent stride
LOG2E = 1.4426950408889634


# -- pointer / LDS helpers ----------------------------------------------------
def lds_dma_dst(base_i32, byte_off_i32, elem_ty=None, align=16):
    """LDS dst view for buffer_load_lds DMA; gotcha: FlyDSL AddressSpace.Shared = LDS (enum 2, not addrspace 3)."""
    if elem_ty is None:
        elem_ty = T.i32
    lds_ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Shared, align)
    lds_ptr = fx.inttoptr(lds_ptr_ty, fx.Int32(base_i32 + byte_off_i32))
    return fx.make_view(lds_ptr, fx.make_layout(1, 1))


def global_base_ptr1(addr_i64):
    """One ptr<1> base from a raw i64 device address (bare data_ptr() kernarg)."""
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(addr_i64)))


def gep1(base_ptr, byte_off_i32):
    """getelementptr i8, base_ptr, byte_off_i32  (ptr<1>)."""
    return buffer_ops.get_element_ptr(base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8)


def global_typed_ptr(arg, elem_ty, align=4):
    """Typed global fx.Pointer over a raw i64 device address; index in ELEMENTS (ptr[i]), not bytes."""
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Global, align)
    return fx.inttoptr(ptr_ty, _raw(fx.Int64(arg)))


def lds_typed_ptr(base_i32, elem_ty, align=4):
    """Typed LDS (Shared) fx.Pointer over an i32 LDS base; index in ELEMENTS (ptr[i]), not bytes."""
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Shared, align)
    return fx.inttoptr(ptr_ty, fx.Int32(base_i32))


def lds_vec_load(base_i32, byte_off_i32, result_type, elem_ty, align=4):
    """Typed LDS ds-read at a BYTE offset from the i32 LDS base; mirrors raw llvm.load (vector or scalar)."""
    elem_ir_ty = elem_ty.ir_type if hasattr(elem_ty, "ir_type") else elem_ty
    ptr = lds_typed_ptr(fx.Int32(base_i32) + byte_off_i32, elem_ir_ty, align=align)
    return fx.ptr_load(ptr, result_type=result_type)


def lds_swizzle_mask(row):
    """lds_swizzle_mask<ROW_BYTES=BK/2=128>(row) = (row & 14) << 3 (fp4 A tile)."""
    return (row & 14) << 3


def lds_swizzle_mask_f8(row):
    """lds_swizzle_mask<ROW_BYTES=256>(row) = (row & 15) << 4 (fp8 A tile)."""
    return (row & 15) << 4


# -- e8m0 / SwiGLU quant math -------------------------------------------------
SWIGLU_ALPHA = 1.702


def silu_mul_batch(gs, us):
    """silu(g)*u via exp2/rcp (matches HIP silu_mul_fast)."""
    e = [fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(-LOG2E)))) for g in gs]
    sig = [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + ei))) for ei in e]
    return [gs[i] * sig[i] * us[i] for i in range(len(gs))]


def swiglu_mul_batch(gs, us, swiglu_limit=0.0):
    """swiglu(g,u) = g*sigmoid(alpha*g)*(u+1) via exp2/rcp; mirrors main mixed_moe swiglu.

    Clamp g<=limit and -limit<=u<=limit before the activation (limit defaults to 7.0 when
    swiglu_limit==0, matching main's _swiglu_mul_vec4).
    """
    limit = float(swiglu_limit) if swiglu_limit != 0 else 7.0
    lim = fx.Float32(limit)
    neg_lim = fx.Float32(-limit)
    gs = [g.minimumf(lim) for g in gs]
    us = [u.minimumf(lim).maximumf(neg_lim) for u in us]
    e = [fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(SWIGLU_ALPHA * -LOG2E)))) for g in gs]
    sig = [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + ei))) for ei in e]
    return [gs[i] * sig[i] * (us[i] + fx.Float32(1.0)) for i in range(len(gs))]


def fabs_f32(x):
    """fabsf via sign-bit clear (FlyDSL has no arith.absf)."""
    abs_bits = _raw(x).bitcast(T.i32) & _raw(fx.Int32(0x7FFFFFFF))
    return fx.Float32(abs_bits.bitcast(T.f32))


def e8m0_from_amax(amax_f32, dtype_max=6.0):
    """(e8m0_i32, quant_scale_f32) = ceil_pow2(amax/dtype_max) clamped to 254 (dtype_max: fp4=6, fp8=448)."""
    wi = fx.Int32(_raw(amax_f32 * fx.Float32(1.0 / dtype_max)).bitcast(T.i32))
    bexp = (wi + 0x7FFFFF).shrui(fx.Int32(23)) & 0xFF
    e8m0 = (bexp < 254).select(bexp, fx.Int32(254))
    qscale = fx.Float32(_raw(e8m0 << 23).bitcast(T.f32))
    return e8m0, qscale


# BM is a per-launch parameter (32 default, 64 supported); the bodies derive
# kMChunks = BM//16 (16-row MFMA row-groups) and kSubBlocks = BM//32 (32-row
# A-scale chunks / scale-register groups) from it. BM=32 stays byte-identical.
BM = 32
kAStages = 3


# ---- Shared layout-API primitives (B / B-scale data movement + scaled MFMA) ----
def b_copy_atom(nontemporal):
    """BufferCopy128b (4x i32 = one 128b weight chunk). nt rides cache_modifier."""
    return fx.make_copy_atom(fx.rocdl.BufferCopy128b(2 if nontemporal else 0), 32)


def bscale_copy_atom():
    """BufferCopy32b (1x i32 e8m0 scale word); always cached (scales reuse heavily)."""
    return fx.make_copy_atom(fx.rocdl.BufferCopy32b(0), 32)


def bq_view(arg_bq, row_elems, KH4, K_TILES_TOTAL):
    """Layout view over preshuffled B for one N-row tile; slice -> i32<4:1> (16B=32 fp4)."""
    col_base = rocdl.readfirstlane(T.i32, _raw(row_elems) * fx.Int32(KH4))
    i32_ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=16)
    off_i64 = fx.Int64(col_base)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_bq) + off_i64 * fx.Int64(4))
    # i32 strides: klane[0,4)->64, nlane[0,16)->4, K_tile->512, half[0,2)->256, kpack4->1
    shape = (4, 16, K_TILES_TOTAL, 2, 4)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, (64, 4, 512, 256, 1))))
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def bscale_view(arg_bscale, base_dw, K_TILES_TOTAL, k0_stride_dw=64):
    """Layout view over e8m0 B-scale for one n-pack word; slice -> i32<1:1> scale word."""
    base_dw = rocdl.readfirstlane(T.i32, _raw(base_dw))
    i32_ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
    off_i64 = fx.Int64(base_dw)
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
    """16 (opselA,opselB) scaled-MFMA atoms; cbsz/blgp=4 for fp4 from Float4E2M1FN."""
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


# ---- Shared A ds-read + per-J MMA cluster (used by both gemm bodies) ----
def issue_a_ds_read_dt(
    s_aq_base, slot, slot_bytes, KH_TILE_A, lane_div_16, lane_mod_16, is_f8, a_vals, a_frags, kMChunks
):
    """A ds-read for one slot: fp4 -> Vec4 i32 into a_frags; fp8 -> Vec8 i32 into a_vals."""
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
                a_vals[i][k] = _raw(a64.bitcast(fx.Int32))
            else:
                mask = lds_swizzle_mask(lane_mod_16)
                lds_col = (lane_div_16 * 16 + k * 64) ^ mask
                vec = lds_vec_load(s_aq_base, row_off + lds_col, Vec.make_type(4, fx.Int32), fx.Int32, align=16)
                a_frags[i][k].store(Vec(vec))


def mma_one_j(J, in_b, sa, sb, bq_frags_kt, is_f8, cbsz_a, a_vals, a_frags, accm, c_frags, atoms, i0=0):
    """One J-cluster (4 scaled MFMAs) for one 32-row A-scale group: row-groups i0, i0+1.

    ``sa`` is the A-scale register for the 32-row group (its 4 bytes = 2 k-halves x 2
    row-groups, picked by opsel_a 0..3). ``i0`` is the first of this group's two 16-row
    row-groups (BM32: i0=0 only; BM64: i0 in {0,2}). fp4 via gemm_mma, fp8 via raw mfma_scale.
    """
    if const_expr(is_f8):
        bJ0 = Vec(bq_frags_kt[J][0].load())
        bJ1 = Vec(bq_frags_kt[J][1].load())
        for osa, k, di in ((0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1)):
            i = i0 + di
            bJ = bJ0 if k == 0 else bJ1
            osb = (0 if k == 0 else 2) + in_b
            accm[i][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4, [a_vals[i][k], bJ, accm[i][J], cbsz_a, 4, osa, sa, osb, sb]
            )
    else:
        bJ0, bJ1 = bq_frags_kt[J][0], bq_frags_kt[J][1]
        gemm_mma(atoms, a_frags[i0 + 0][0], bJ0, c_frags[i0 + 0][J], 0, 0 + in_b, sa, sb)
        gemm_mma(atoms, a_frags[i0 + 1][0], bJ0, c_frags[i0 + 1][J], 1, 0 + in_b, sa, sb)
        gemm_mma(atoms, a_frags[i0 + 0][1], bJ1, c_frags[i0 + 0][J], 2, 2 + in_b, sa, sb)
        gemm_mma(atoms, a_frags[i0 + 1][1], bJ1, c_frags[i0 + 1][J], 3, 2 + in_b, sa, sb)


# ---- gemm1 (up/gate-proj) ----
@flyc.jit
def gemm1_body_v2(
    lds_base_i32,
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
    i32_inter,
    *,
    BM,
    K,
    interleave,
    b_nontemporal,
    a_dtype,
    out_dtype,
    act="silu",
    swiglu_limit=0.0,
    SBM=None,
):
    # SBM (sort_block_m) is the moe_sorting padding unit; BM (=tile_m) is the compute tile.
    # SBM==BM (default) -> byte-identical single-block-per-sort-block behavior. SBM>BM (a
    # multiple) packs SBM//BM compute blocks into one SBM sort block that all share one expert:
    # the expert id is looked up at sei[(m_block_idx*BM)//SBM] instead of sei[m_block_idx].
    if SBM is None:
        SBM = BM
    # BM-derived constants (module BM=32 default; BM=64 doubles rows/block).
    kMChunks = BM // 16  # 16-row MFMA row-groups (BM32: 2, BM64: 4)
    kSubBlocks = BM // 32  # 32-row A-scale chunks / scale-register groups (BM32: 1, BM64: 2)
    # A dtype: only the A path differs; fp8 uses raw mfma_scale (cbsz=0), fp4 the fx.gemm path.
    is_f8_a = a_dtype == "fp8"
    # out dtype: only the epilogue requant/pack/store differs.
    is_f8_out = out_dtype == "fp8"
    out_max = 448.0 if is_f8_out else 6.0  # e4m3 / e2m1 max
    out_pack = 1 if is_f8_out else 2
    a_pack = 1 if is_f8_a else 2
    KH_TILE_A = BK // a_pack  # A bytes/K-tile row in LDS (fp8=256, fp4=128)
    cbsz_a = 0 if is_f8_a else 4  # mfma A-format (fp8=0, fp4=4)
    # K-/INTER-derived sizes (compile-time ints).
    kc = (K // 32) // 4 // 2
    K_HALF = K // 2
    K_BYTES = K // a_pack  # a_quant row stride in bytes (= K_HALF for fp4)
    K_TILES_TOTAL = K // BK
    kUnroll = K_TILES_TOTAL - kStages
    kAS_per_chunk_dw = kc * 64
    kBS_stride_n0_dw = kc * 64  # hidden-K-derived (compile-time)
    # INTER (inter_dim) is the gemm1 N-output dim; runtime via i32_inter (no K-loop dependency).
    INTER_rt = fx.Int32(i32_inter)
    N_OUT = INTER_rt * fx.Int32(2)
    kBS_per_expert_dw = (N_OUT // fx.Int32(32)) * fx.Int32(kBS_stride_n0_dw)  # (N_OUT//16//2)*stride
    NUM_N_BLOCKS = N_OUT // fx.Int32(256)
    OUT_AS_PER_CHUNK_DW = (INTER_rt // fx.Int32(256)) * fx.Int32(64)  # ((INTER//32)//4//2)*64
    K_G2_BYTES = INTER_rt // fx.Int32(out_pack)  # output row stride (fp4 INTER/2, fp8 INTER)

    # block -> (m_block_idx, n_block_idx) ; e = sorted_expert_ids[sort_block] where the sort block
    # is the SBM-padded block this compute block falls into (SBM==BM: sort_block == m_block_idx,
    # emitted identically to the pre-sbm path).
    n_block_idx = bx_i32 % NUM_N_BLOCKS
    m_block_idx = bx_i32 // NUM_N_BLOCKS
    eids_ptr = global_typed_ptr(arg_eids, T.i32)
    if const_expr(SBM == BM):
        e = rocdl.readfirstlane(T.i32, _raw(eids_ptr[m_block_idx]))
        m_row = m_block_idx * BM
    else:
        m_row = m_block_idx * BM
        e = rocdl.readfirstlane(T.i32, _raw(eids_ptr[m_row // fx.Int32(SBM)]))

    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16

    # LDS base offsets (i8): s_aq | s_asc contiguous; lds_acc (f32) unions the region.
    s_aq_base = lds_base_i32
    s_asc_base = lds_base_i32 + fx.Int32(kAStages * BM * KH_TILE_A)
    lds_acc_base = lds_base_i32

    # A-gather rows: row = sorted_token_ids & 0xFFFFFF; pad rows are OOB so buffer_load_lds returns 0.
    lanes_per_row = KH_TILE_A // 16  # 8 (fp4) / 16 (fp8)
    rows_per_call = 64 // lanes_per_row  # 8 (fp4) / 4 (fp8)
    a_lane_row = lane // lanes_per_row
    rows_per_wave = BM // 4  # rows each wave gathers (BM32: 8, BM64: 16)
    n_row_groups = rows_per_wave // rows_per_call  # DMA calls/wave (fp4: BM/32; fp8: BM/16)
    sti_ptr = global_typed_ptr(arg_sti, T.i32)
    cached_actual_row = []
    for g in range_constexpr(n_row_groups):
        idx = m_row + wave * rows_per_wave + g * rows_per_call + a_lane_row
        cached_actual_row.append(sti_ptr[idx] & 0xFFFFFF)

    # B-scale n-pack words (gate/up split differs by gate mode).
    if const_expr(interleave):
        mni_base = n_block_idx * (BN // 32) + wave * (BN // 128)
        np_list = [mni_base, mni_base + 1]
    else:
        np_gate = n_block_idx * (BN // 64) + wave
        np_list = [np_gate, np_gate + N_OUT // fx.Int32(64)]

    # A-gather global->LDS DMA: per-lane src (no fold); aq_rsrc bounds make OOB padded rows load 0.
    a_gather_atom = lds_dma_atom_128()
    a_gather_src = flat_buffer_view(
        arg_aq,
        None,
        T.i32,
        align=16,
        elem_bytes=4,
        fold=False,
        num_records_bytes=i32_ntok * K_BYTES,
    )

    def issue_a_load_lds(slot, kt):
        # lane L -> LDS[base+L*16]; each wave gathers rows_per_wave rows in rows_per_call chunks.
        lane_col = (lane % lanes_per_row) * 16
        base_i32 = s_aq_base
        for g in range_constexpr(n_row_groups):
            lds_row = wave * rows_per_wave + g * rows_per_call
            mask = (
                lds_swizzle_mask_f8(lds_row + a_lane_row)
                if const_expr(is_f8_a)
                else lds_swizzle_mask(lds_row + a_lane_row)
            )
            voffset = (lane_col ^ mask) + cached_actual_row[g] * K_BYTES
            off = fx.Int32(slot * (BM * KH_TILE_A)) + lds_row * KH_TILE_A
            v_e = (voffset + kt * KH_TILE_A) // 4  # per-lane i32-elem index
            fx.copy(
                a_gather_atom,
                a_gather_src[v_e, None],
                lds_dma_dst(base_i32, off, elem_ty=T.i32, align=16),
            )

    def issue_a_ds_read(slot):
        issue_a_ds_read_dt(
            s_aq_base, slot, BM * KH_TILE_A, KH_TILE_A, lane_div_16, lane_mod_16, is_f8_a, a_vals, a_frags, kMChunks
        )

    asc_dma128 = lds_dma_atom_128()
    asc_dma32 = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), 32)  # 4B A-scale chunk

    def issue_a_scale_load():
        # global->LDS DMA: 16B + 3x4B chunks; per-chunk dword base folded to a VGPR voffset.
        chunk_base = m_row // 32
        v16_e = (wave * 64 + lane) * 4  # 16B chunk: per-lane i32-elem
        v4_e = wave * 64 + lane  # 4B chunk: per-lane i32-elem
        asc_base = s_asc_base
        for sub in range_constexpr(kSubBlocks):
            base_dw = (chunk_base + sub) * kAS_per_chunk_dw  # s_chunk/4
            lds_sub = sub * kAS_per_chunk_dw * 4
            src16 = flat_buffer_view(arg_ascale, base_dw, T.i32, align=16, elem_bytes=4)
            fx.copy(
                asc_dma128,
                src16[v16_e, None],
                lds_dma_dst(asc_base, lds_sub + wave * 1024, elem_ty=T.i32, align=16),
            )
            for d in range_constexpr(3):
                byte_off = 4096 + d * 1024
                src4 = flat_buffer_view(arg_ascale, base_dw + byte_off // 4, T.i32, align=16, elem_bytes=4)
                fx.copy(
                    asc_dma32,
                    src4[v4_e, None],
                    lds_dma_dst(asc_base, lds_sub + byte_off + wave * 256, elem_ty=T.i32, align=4),
                )

    def issue_a_scale_ds_read(kt):
        asc_ptr = lds_typed_ptr(s_asc_base, T.i32)
        out = []
        for sub in range_constexpr(kSubBlocks):
            lds_dw = fx.Int32(sub * kAS_per_chunk_dw) + kt * 64 + lane_div_16 * 16 + lane_mod_16
            out.append(asc_ptr[lds_dw])
        return out

    # B load: CK-preshuffle view over bq; base MUST stay wave-uniform (per-lane fold -> WATERFALL ~14x).
    KH4 = K_HALF // 4  # i32 stride for the col axis
    b_catom = b_copy_atom(b_nontemporal)
    bs_copy_atom = bscale_copy_atom()

    N0_HALF = N_OUT // fx.Int32(32)  # separate-mode gate/up column split

    # B-load view per j-tile; gate mode only changes which N-row `col` maps to.
    def make_bq_view_for_jtile(j):
        if const_expr(interleave):
            col = n_block_idx * BN + wave * (BN // 4) + j * 16
        else:
            tile_il = n_block_idx * 16 + wave * 4 + j
            col = ((tile_il & 1) * N0_HALF + (tile_il >> 1)) * 16
        return bq_view(arg_bq, e * N_OUT + col, KH4, K_TILES_TOTAL)

    bq_views = [make_bq_view_for_jtile(j) for j in range_constexpr(4)]

    # B-scale view per n-pack word (shared layout primitive).
    bscale_views = [
        bscale_view(
            arg_bscale,
            e * kBS_per_expert_dw + np_list[mw] * kBS_stride_n0_dw,
            K_TILES_TOTAL,
            k0_stride_dw=kBS_stride_k0_dw,
        )
        for mw in range_constexpr(2)
    ]

    # B fragments: i32<4:1> (16B = 32 fp4), per-stage (kStages) prefetch double-buffer.
    frag_tmpl = bq_frag_tmpl(bq_views[0])  # i32<4:1>
    bq_frags = [
        [[fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(4)]
        for _ in range_constexpr(kStages)
    ]
    # fp4: A in fx.gemm fragments, C accumulates in place. fp8: A a per-iter Vec8 i32, C a raw f32x4.
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    a_vals = a_frags = c_frags = accm = None
    if const_expr(is_f8_a):
        a_vals = [[None, None] for _ in range(kMChunks)]
        accm = [[zero4 for _ in range(4)] for _ in range(kMChunks)]
    else:
        a_frags = [[fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(kMChunks)]
        c_frags = [
            [fx.make_fragment_like(frag_tmpl, dtype=fx.Float32.ir_type) for _ in range_constexpr(4)]
            for _ in range_constexpr(kMChunks)
        ]
    # B-scale fragments: i32<1:1>, per-stage double-buffer like _bq_frags.
    bs_frag_tmpl = bscale_frag_tmpl(bscale_views[0])  # i32<1:1>
    bs_frags = [[fx.make_fragment_like(bs_frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(kStages)]

    def issue_b_load_j(stage, K_C, j):
        view = bq_views[j]
        for half in range_constexpr(2):
            fx.copy(
                b_catom,
                view[lane_div_16, lane_mod_16, K_C, half, None],
                bq_frags[stage][j][half],
            )

    def issue_b_scale_load(stage, K_C):
        for mw in range_constexpr(2):
            fx.copy(
                bs_copy_atom,
                bscale_views[mw][lane_div_16, lane_mod_16, K_C, None],
                bs_frags[stage][mw],
            )

    # MMA: fp4 via fx.gemm (one per mfma); fp8 via the raw scaled-MFMA intrinsic.
    mma_atoms = scale_mma_atoms() if const_expr(not is_f8_a) else None

    def mfma_cluster(stage, a_scale, J):
        # interleave: mni=J//2 (n0), in_b=J%2 (gate/up); separate: swapped.
        if const_expr(interleave):
            mni, in_b = J // 2, J % 2
        else:
            mni, in_b = J % 2, J // 2
        sb = _raw(Vec(bs_frags[stage][mni].load())[0])
        # One 32-row A-scale group per kSubBlock (its register holds row-groups 2*sub, 2*sub+1).
        for sub in range_constexpr(kSubBlocks):
            mma_one_j(
                J,
                in_b,
                a_scale[sub],
                sb,
                bq_frags[stage],
                is_f8_a,
                cbsz_a,
                a_vals,
                a_frags,
                accm,
                c_frags,
                mma_atoms,
                i0=2 * sub,
            )

    # zero C (fp4 fragments accumulate in place thereafter; fp8 accm pre-init above).
    if const_expr(not is_f8_a):
        for i in range_constexpr(kMChunks):
            for J in range_constexpr(4):
                c_frags[i][J].store(zero4)

    # prologue: stages 0,1
    issue_a_scale_load()
    for K_C in range_constexpr(kStages):
        issue_a_load_lds(K_C, K_C)
        for j in range_constexpr(4):
            issue_b_load_j(K_C, K_C, j)
        issue_b_scale_load(K_C, K_C)

    # main loop. sched_barrier/s_setprio fence the mfma chain from the B loads (closes the small-M gap).
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

    # epilog: cshuffle -> SwiGLU -> fp4 + e8m0 requant (raw math)
    wave_n = wave
    lds_acc_fptr = lds_typed_ptr(lds_acc_base, T.f32)

    # accumulators: fp4 from C fragments, fp8 from accm.
    if const_expr(is_f8_a):
        acc_vecs = [[Vec(accm[i][J]) for J in range(4)] for i in range(kMChunks)]
    else:
        acc_vecs = [[Vec(c_frags[i][J].load()) for J in range(4)] for i in range(kMChunks)]

    def acc(i, J, v):
        return acc_vecs[i][J][v]

    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * 4
        for J in range_constexpr(4):
            is_up = (J % 2) == 1
            J_local = J // 2
            col_local = wave_n * 32 + J_local * 16 + lane_mod_16
            lds_col = (128 + col_local) if is_up else col_local
            for v in range_constexpr(4):
                idx = (row_base + v) * BN + lds_col
                lds_acc_fptr[idx] = fx.Float32(acc(i, J, v))

    gpu.barrier()

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // 16
    n_lane = tx_i32 % 16
    wave_grp = n_lane // 4
    kk = n_lane % 4

    # Output store via fx.copy (BufferCopy32b nt) over an i32 view; wave-uniform row base in view base.
    out_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(2), fx.Int32)  # nt i32 store
    out_reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    aqout_view = flat_buffer_view(arg_aqout, m_row * (K_G2_BYTES // 4), T.i32, align=4, elem_bytes=4)
    scales_per_mr = [None] * kMChunks

    for mr in range_constexpr(kMChunks):
        row_local = fx.Int32(mr * 16) + m_lane
        gate_vs = [None] * 8
        up_vs = [None] * 8
        for ee in range_constexpr(8):
            col_in_grp = 8 * kk + ee
            gate_col = wave_grp * 32 + col_in_grp
            up_col = 128 + gate_col
            gate_idx = row_local * BN + gate_col
            up_idx = row_local * BN + up_col
            gate_vs[ee] = fx.Float32(lds_acc_fptr[gate_idx])
            up_vs[ee] = fx.Float32(lds_acc_fptr[up_idx])
        if const_expr(act == "swiglu"):
            result = swiglu_mul_batch(gate_vs, up_vs, swiglu_limit)
        else:
            result = silu_mul_batch(gate_vs, up_vs)

        local_max = fabs_f32(result[0])
        for ee in range_constexpr(1, 8):
            local_max = local_max.maximumf(fabs_f32(result[ee]))
        local_max = local_max.maximumf(local_max.shuffle_xor(fx.Int32(1), fx.Int32(64)))
        local_max = local_max.maximumf(local_max.shuffle_xor(fx.Int32(2), fx.Int32(64)))

        e8m0, qscale = e8m0_from_amax(local_max, out_max)
        scales_per_mr[mr] = e8m0

        qscale_raw = _raw(qscale)
        # byte position of this lane's 8 elems (fp8 doubles it; row stride is INTER).
        byte_pos_fp4 = n_block_idx * (BN // 4) + wave_grp * 16 + kk * 4
        if const_expr(is_f8_out):
            # 8 f32 -> 8 fp8: lo holds elems 0..3, hi 4..7 (2 fp8 per cvt half).
            v2i16 = T.vec(2, T.i16)
            lo = _raw(Vec.filled(2, 0, fx.Int16))
            lo = rocdl.cvt_scalef32_pk_fp8_f32(v2i16, lo, _raw(result[0]), _raw(result[1]), qscale_raw, 0)
            lo = rocdl.cvt_scalef32_pk_fp8_f32(v2i16, lo, _raw(result[2]), _raw(result[3]), qscale_raw, 1)
            hi = _raw(Vec.filled(2, 0, fx.Int16))
            hi = rocdl.cvt_scalef32_pk_fp8_f32(v2i16, hi, _raw(result[4]), _raw(result[5]), qscale_raw, 0)
            hi = rocdl.cvt_scalef32_pk_fp8_f32(v2i16, hi, _raw(result[6]), _raw(result[7]), qscale_raw, 1)
            # i32-elem off (uniform m_row in view base); lo at off, hi at off+1 (each vec2xi16 = one i32).
            elem_off = row_local * (K_G2_BYTES // 4) + (byte_pos_fp4 // 2)
            lo_i32 = Vec(lo).bitcast(fx.Int32)
            hi_i32 = Vec(hi).bitcast(fx.Int32)
            fx.memref_store_vec(Vec.filled(1, lo_i32[0], fx.Int32), out_reg)
            fx.copy(out_copy_atom, out_reg, aqout_view[elem_off, None])
            fx.memref_store_vec(Vec.filled(1, hi_i32[0], fx.Int32), out_reg)
            fx.copy(out_copy_atom, out_reg, aqout_view[elem_off + 1, None])
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
            elem_off = row_local * (K_G2_BYTES // 4) + (byte_pos_fp4 // 4)
            fx.memref_store_vec(Vec.filled(1, fx.Int32(packed_i32), fx.Int32), out_reg)
            fx.copy(out_copy_atom, out_reg, aqout_view[elem_off, None])

    # ascaleout store via fx.copy (BufferCopy16b) over an i16 view; wave-uniform byte base in view base.
    asc_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.Int16)
    asc_reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int16)
    if kk == 0:
        ku = n_block_idx >> 1
        ikxdl = n_block_idx & 1
        for sub in range_constexpr(kSubBlocks):
            chunk = m_block_idx * kSubBlocks + sub
            # uniform i16 base = (chunk*OUT_AS_PER_CHUNK_DW + ku*64)*2 + ikxdl
            base_i16 = (chunk * OUT_AS_PER_CHUNK_DW + ku * 64) * 2 + ikxdl
            asc_view = flat_buffer_view(arg_ascaleout, base_i16, T.i16, align=2, elem_bytes=2)
            pair_i32 = scales_per_mr[sub * 2 + 0] | (scales_per_mr[sub * 2 + 1] << 8)
            pair_i16 = fx.Int16(pair_i32)
            # per-lane i16 offset = (wave_grp*16 + m_lane)*2
            asc_off = (wave_grp * 16 + m_lane) * 2
            fx.memref_store_vec(Vec.filled(1, pair_i16, fx.Int16), asc_reg)
            fx.copy(asc_copy_atom, asc_reg, asc_view[asc_off, None])


def lds_bytes_for(K_TILES_TOTAL, KH_TILE_A=KH_TILE, BM=BM):
    kSubBlocks = BM // 32
    s_aq_bytes = kAStages * BM * KH_TILE_A  # fp8 A tile is 2x (256B vs 128B)
    s_asc_bytes = kSubBlocks * K_TILES_TOTAL * 256
    lds_acc_bytes = BM * BN * 4
    return max(s_aq_bytes + s_asc_bytes, lds_acc_bytes)


# ---- gemm2 (down-proj) ----
def issue_a_load_lds_dt(
    arg_aq, aq_num_records, s_aq_base, slot, kt, m_row, wave, lane, is_f8, KH_TILE_A, K_BYTES, BM=BM
):
    """A->LDS DMA for one K-tile; gemm2 A is the already-sorted row, OOB-zero via aq_rsrc bounds."""
    lanes_per_row = KH_TILE_A // 16  # 8 (fp4) / 16 (fp8)
    rows_per_call = 64 // lanes_per_row  # 8 (fp4) / 4 (fp8)
    a_lane_row = lane // lanes_per_row
    rows_per_wave = BM // 4  # rows each wave loads (BM32: 8, BM64: 16)
    n_row_groups = rows_per_wave // rows_per_call
    lane_col = (lane % lanes_per_row) * 16
    base_i32 = s_aq_base
    atom = lds_dma_atom_128()
    src = flat_buffer_view(arg_aq, None, T.i32, align=16, elem_bytes=4, fold=False, num_records_bytes=aq_num_records)
    for g in range_constexpr(n_row_groups):
        lds_row = wave * rows_per_wave + g * rows_per_call
        mask = (
            lds_swizzle_mask_f8(lds_row + a_lane_row) if const_expr(is_f8) else lds_swizzle_mask(lds_row + a_lane_row)
        )
        car = m_row + lds_row + a_lane_row  # direct sorted row
        voffset = (lane_col ^ mask) + car * K_BYTES
        off = fx.Int32(slot * (BM * KH_TILE_A)) + lds_row * KH_TILE_A
        v_e = (voffset + kt * KH_TILE_A) // 4  # per-lane i32-elem index
        fx.copy(atom, src[v_e, None], lds_dma_dst(base_i32, off, elem_ty=T.i32, align=16))


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
    aq_rsrc,
    arg_aq,
    i32_inter,
    *,
    BM,
    use_nt,
    N_OUT,
    INTER_MAX,
    aStages,
    a_dtype,
    use_reduce=False,
    topk=1,
    SBM=None,
):
    # SBM (sort_block_m) is the moe_sorting padding unit; BM (=tile_m) is the compute tile.
    # SBM==BM (default) is byte-identical. SBM>BM packs SBM//BM compute blocks per SBM sort block;
    # the expert id is looked up at sei[(m_block_idx*BM)//SBM] instead of sei[m_block_idx].
    if SBM is None:
        SBM = BM
    aStages = aStages
    kMChunks = BM // 16  # 16-row MFMA row-groups (BM32: 2, BM64: 4)
    kSubBlocks = BM // 32  # 32-row A-scale chunks / scale-register groups (BM32: 1, BM64: 2)
    # A dtype: fp4 (gemm1 fp4-out) or fp8 (mxfp8); only the A path differs.
    is_f8_a = a_dtype == "fp8"
    a_pack = 1 if is_f8_a else 2
    KH_TILE_A = BK // a_pack
    slot_bytes = BM * KH_TILE_A
    cbsz_a = 0 if is_f8_a else 4
    # Contraction K = inter_dim is runtime (i32_inter); INTER_MAX caps compile-time view/fragment bounds.
    K_rt = fx.Int32(i32_inter)
    K_BYTES = K_rt // fx.Int32(a_pack)  # A row stride bytes (runtime)
    kc_rt = K_rt // fx.Int32(256)  # (K//32)//4//2
    K_TILES_RT = K_rt // fx.Int32(BK)  # runtime K-tile trip count
    kAS_per_chunk_dw = kc_rt * fx.Int32(64)
    kBS_stride_n0_dw = kc_rt * fx.Int32(64)
    kbs_per_expert_dw = fx.Int32(N_OUT // 16 // 2) * kBS_stride_n0_dw
    num_n_blocks = N_OUT // 256
    KH4 = K_rt // fx.Int32(8)  # i32 col stride (= K_HALF//4)
    K_TILES_MAX = INTER_MAX // BK

    # block -> (m_block_idx, n_block_idx) ; e = sorted_expert_ids[sort_block] where the sort block
    # is the SBM-padded block this compute block falls into (SBM==BM: sort_block == m_block_idx,
    # emitted identically to the pre-sbm path).
    m_block_idx = bx_i32 // num_n_blocks
    n_block_idx = bx_i32 - m_block_idx * num_n_blocks
    eids_ptr = global_typed_ptr(arg_eids, T.i32)
    if const_expr(SBM == BM):
        e = rocdl.readfirstlane(T.i32, _raw(eids_ptr[m_block_idx]))
        m_row = m_block_idx * BM
    else:
        m_row = m_block_idx * BM
        e = rocdl.readfirstlane(T.i32, _raw(eids_ptr[m_row // fx.Int32(SBM)]))

    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16

    # A-scale buffer resource + uniform base (A-scale load stays raw).
    asc_per_mb = fx.Int32(BM // 32) * kAS_per_chunk_dw * fx.Int32(4)
    asc_num = fx.Index(i32_max_m_blocks) * fx.Index(asc_per_mb)
    ascale_rsrc = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_ascale)), num_records_bytes=asc_num)
    a_scale_s_base = rocdl.readfirstlane(T.i32, (m_row // 32) * kAS_per_chunk_dw * fx.Int32(4))
    v_voff_scale = ((lane_div_16 * 16) + lane_mod_16) * 4

    def load_a_scale_tile(kt):
        # One i32 A-scale register per 32-row chunk (kSubBlocks); chunk sub at sub*kAS_per_chunk_dw dwords.
        out = []
        for sub in range_constexpr(kSubBlocks):
            out.append(
                buffer_ops.buffer_load(
                    ascale_rsrc,
                    (v_voff_scale + kt * 256) // 4 + sub * kAS_per_chunk_dw,
                    vec_width=1,
                    dtype=T.i32,
                    soffset_bytes=a_scale_s_base,
                )
            )
        return out

    s_aq_base = lds_base_i32
    lds_acc_base = lds_base_i32  # f32 acc unions the A-tile region

    # -- B / B-scale layout-API views (shared primitives) ---------------------
    b_catom = b_copy_atom(use_nt)
    bs_copy_atom = bscale_copy_atom()

    def make_bq_view(j):
        col = n_block_idx * BN + wave * (BN // 4) + j * 16
        return bq_view(arg_bq, e * fx.Int32(N_OUT) + col, KH4, K_TILES_MAX)

    bq_views = [make_bq_view(j) for j in range_constexpr(4)]

    mni_base = n_block_idx * (BN // 16 // 2) + wave * (BN // 64 // 2)
    bscale_views = [
        bscale_view(
            arg_bscale,
            e * kbs_per_expert_dw + (mni_base + mw) * kBS_stride_n0_dw,
            K_TILES_MAX,
            k0_stride_dw=kBS_stride_k0_dw,
        )
        for mw in range_constexpr(2)
    ]

    # B / B-scale fragments are streamed PER-ITER (one K-tile worth); A refilled per K via LDS.
    frag_tmpl = bq_frag_tmpl(bq_views[0])  # i32<4:1>
    bs_frag_tmpl = bscale_frag_tmpl(bscale_views[0])  # i32<1:1>
    # fp4: A in fx.gemm fragments. fp8: A a per-iter Vec8 i32, C a raw f32x4 (accm).
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    a_vals = a_frags = c_frags = accm = None
    if const_expr(is_f8_a):
        a_vals = [[None, None] for _ in range(kMChunks)]
        accm = [[zero4 for _ in range(4)] for _ in range(kMChunks)]
    else:
        a_frags = [[fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(kMChunks)]
        c_frags = [
            [fx.make_fragment_like(frag_tmpl, dtype=fx.Float32.ir_type) for _ in range_constexpr(4)]
            for _ in range_constexpr(kMChunks)
        ]

    def stream_b_tile(kt_rt):
        # One K-tile of B / B-scale into fresh per-iter fragments (B streamed, not register-resident).
        bqf = [[fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(4)]
        bsf = [fx.make_fragment_like(bs_frag_tmpl) for _ in range_constexpr(2)]
        for j in range_constexpr(4):
            for half in range_constexpr(2):
                fx.copy(b_catom, bq_views[j][lane_div_16, lane_mod_16, kt_rt, half, None], bqf[j][half])
        for mw in range_constexpr(2):
            fx.copy(bs_copy_atom, bscale_views[mw][lane_div_16, lane_mod_16, kt_rt, None], bsf[mw])
        return bqf, bsf

    def issue_a_ds_read(slot):
        issue_a_ds_read_dt(
            s_aq_base, slot, slot_bytes, KH_TILE_A, lane_div_16, lane_mod_16, is_f8_a, a_vals, a_frags, kMChunks
        )

    aq_num_records = fx.Index(i32_max_m_blocks) * fx.Index(fx.Int32(BM) * K_BYTES)

    def issue_a_load_lds(slot, kt):
        issue_a_load_lds_dt(
            arg_aq, aq_num_records, s_aq_base, slot, kt, m_row, wave, lane, is_f8_a, KH_TILE_A, K_BYTES, BM=BM
        )

    mma_atoms = scale_mma_atoms() if const_expr(not is_f8_a) else None

    def mfma_cluster(bqf, bsf, sa):
        # opsel (gemm2 has no gate/up split): mni=J//2, in_b=J%2. sa is a per-32-row-chunk list.
        for J in range_constexpr(4):
            mni, in_b = J // 2, J % 2
            sb = _raw(Vec(bsf[mni].load())[0])
            for sub in range_constexpr(kSubBlocks):
                mma_one_j(
                    J,
                    in_b,
                    sa[sub],
                    sb,
                    bqf,
                    is_f8_a,
                    cbsz_a,
                    a_vals,
                    a_frags,
                    accm,
                    c_frags,
                    mma_atoms,
                    i0=2 * sub,
                )

    # zero C (fp4 fragments accumulate in place; fp8 accm pre-init above).
    if const_expr(not is_f8_a):
        for i in range_constexpr(kMChunks):
            for J in range_constexpr(4):
                c_frags[i][J].store(zero4)

    # Runtime-trip scf.for K-loop: stream A->LDS (triple-buffered) + B per tile; carry C / accm.
    aStagesC = aStages

    def load_carry():
        if const_expr(is_f8_a):
            return [accm[i][J] for i in range(kMChunks) for J in range(4)]
        return [c_frags[i][J].load() for i in range(kMChunks) for J in range(4)]

    def store_carry(state):
        n = 0
        for i in range_constexpr(kMChunks):
            for J in range_constexpr(4):
                if const_expr(is_f8_a):
                    accm[i][J] = state[n]
                else:
                    c_frags[i][J].store(state[n])
                n += 1

    for kt_iv, state in range(fx.Index(0), fx.Index(K_TILES_RT), fx.Index(1), init=load_carry()):
        store_carry(state)
        kt_rt = fx.Int32(kt_iv)
        gpu.barrier()
        issue_a_ds_read(kt_rt % fx.Int32(aStagesC))
        nxt = kt_rt + fx.Int32(kStages)
        if nxt < K_TILES_RT:
            issue_a_load_lds(nxt % fx.Int32(aStagesC), nxt)
        bqf, bsf = stream_b_tile(kt_rt)
        sa = load_a_scale_tile(kt_rt)
        mfma_cluster(bqf, bsf, sa)
        results = yield load_carry()
    store_carry(results)

    # epilog: atomic bf16. fp8 reads accm; fp4 loads the C fragments.
    if const_expr(is_f8_a):
        accm_vecs = accm
    else:
        accm_vecs = [[c_frags[i][J].load() for J in range(4)] for i in range(kMChunks)]
    atomic_bf16_epilog(
        lds_acc_base,
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
        use_reduce=use_reduce,
        topk=topk,
        SBM=SBM,
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
    use_reduce=False,
    topk=1,
    SBM=None,
):
    if SBM is None:
        SBM = BM
    kMChunks = BM // 16
    M_REPS = BM // 8  # BM32: 4, BM16: 2
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    lds_base_fptr = lds_typed_ptr(lds_acc_base, T.f32)

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // 32
    n_lane = tx_i32 % 32
    col_start = n_lane * 2
    stids_base = global_base_ptr1(arg_stids)
    sweights_base = global_base_ptr1(arg_sweights)
    out_base = global_base_ptr1(arg_out)

    # Prefetch sorted_token_ids / sorted_weights (invariant) so their latency overlaps the stores+barriers.
    packed = []
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + mr * 8 + m_lane
        packed.append(llvm.load(T.i32, gep1(stids_base, sorted_pos * 4), invariant=True))
        weight.append(llvm.load(T.f32, gep1(sweights_base, sorted_pos * 4), invariant=True))

    # pre-store fence+barrier (HIP run_one __syncthreads() before the epilog).
    gpu.barrier()

    # write accm -> lds_acc cshuffle (scalar f32 stores, as HIP does)
    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * 4
        for J in range_constexpr(4):
            col = wave * 64 + J * 16 + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + v) * BN + col
                lds_base_fptr[idx] = fx.Float32(vec[v])

    gpu.barrier()

    # read back + weighted store. atomic: fadd into out[token_id] (per-token accumulate).
    # reduce: plain non-atomic store into out[token_id*topk + s] (unique per (token,topk) slot;
    # host reduces over topk). Mirrors main mixed_moe_gemm_2stage accumulate=True/False.
    # ``if token_id < i32_M`` gates out padding rows (sentinel token_id == M). The default
    # (SBM==BM and BM<=32, atomic) keeps the legacy plain-Python ``if`` (byte-identical: the store
    # body always traces; the OOB padding-row write lands in allocator slack and never faulted).
    # BM>=64 or SBM>BM or reduce promotes the guard to a real device ``scf.if`` so the padding-row
    # store is genuinely skipped: at those the OOB write hit an illegal address --
    #   - BM>=64 / sbm-decoupled: larger padding stride (a8w4/small-token; sbm padding).
    #   - reduce: the padding-row out_row = M*topk + slot overshoots the [tokens*topk, H] buffer by
    #     up to ``topk`` rows (topk x the atomic overshoot), which faults at large-M (e.g. DSV3
    #     32768*9): the exactly-sized 4GB reduce buffer has no allocator slack to absorb it.
    guard_padding = BM >= 64 or SBM != BM or use_reduce

    def store_one_mr(mr):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if const_expr(use_reduce):
            # reduce out_row = token_id*topk + slot can reach tokens*topk (large-M, e.g. DSV3
            # 32768*9); out_row*N_OUT*2 (bf16 byte off) then overflows i32. Compute the reduce
            # element base in i64 so the store byte offset never wraps. (atomic out_row=token_id
            # <= M keeps the i32 path byte-identical.)
            out_row = fx.Int64(token_id * fx.Int32(topk) + (packed[mr] >> fx.Int32(24)))
            row_base_addr = out_row * fx.Int64(N_OUT) + fx.Int64(n_block_idx * BN + col_start)
        else:
            out_row = token_id
            row_base_addr = out_row * N_OUT + n_block_idx * BN + col_start
        for s in range_constexpr(4):
            # adjacent ee=0,1 are contiguous -> one <2xf32> load (as HIP vectorizes)
            idx0 = row_in_block * BN + col_start + s * 64
            v2 = Vec(lds_vec_load(lds_acc_base, idx0 * 4, Vec.make_type(2, fx.Float32), fx.Float32, align=8))
            pk = Vec.from_elements([v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32).to(fx.BFloat16)
            if const_expr(use_reduce):
                off = (row_base_addr + fx.Int64(s * 64)) * fx.Int64(2)  # bf16 byte off (i64)
            else:
                off = (row_base_addr + s * 64) * 2  # bf16 byte off
            out_ptr = gep1(out_base, off)
            if const_expr(use_reduce):
                llvm.StoreOp(_raw(pk), out_ptr, alignment=4, nontemporal=True)
            else:
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    out_ptr,
                    _raw(pk),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )

    for mr in range_constexpr(M_REPS):
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if const_expr(guard_padding):
            _if_valid = scf.IfOp(_raw(token_id < i32_M))
            with ir.InsertionPoint(_if_valid.then_block):
                store_one_mr(mr)
                scf.YieldOp([])
        else:
            if token_id < i32_M:
                store_one_mr(mr)
