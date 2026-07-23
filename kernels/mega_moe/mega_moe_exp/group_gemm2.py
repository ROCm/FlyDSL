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
from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Float4E2M1FN, T
from flydsl.expr.typing import Vector as Vec
from kernels.common.mem_ops import buffer_atomic_add
from kernels.gemm.fp8_gemm_utils import ceildiv

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


# -- B / B-scale data movement + scaled MFMA (shared layout-API primitives) ----
def lds_dma_atom_128():
    return fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)


def b_copy_atom(nontemporal):
    return fx.make_copy_atom(fx.rocdl.BufferCopy128b(2 if nontemporal else 0), 32)


def bscale_copy_atom():
    return fx.make_copy_atom(fx.rocdl.BufferCopy32b(0), 32)


def bq_view(arg_bq, row_elems, KH4, K_TILES_TOTAL):
    col_base = rocdl.readfirstlane(T.i32, fx.Int32(row_elems) * fx.Int32(KH4))
    i32_ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=16)
    off_i64 = fx.Int64(col_base)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_bq) + off_i64 * fx.Int64(4))
    shape = (4, 16, K_TILES_TOTAL, 2, 4)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, (64, 4, 512, 256, 1))))
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
    kUnroll = K_TILES_TOTAL - kStages
    kAS_per_chunk_dw = kc * 64
    kBS_stride_n0_dw = kc * 64
    kbs_per_expert_dw = (N_OUT // 16 // 2) * kBS_stride_n0_dw
    num_n_blocks = N_OUT // 256
    KH4 = K_HALF // 4

    m_block_idx = bx_i32 // num_n_blocks
    n_block_idx = bx_i32 - m_block_idx * num_n_blocks
    eids_rsrc = buffer_ops.create_buffer_resource_from_addr(fx.Int64(arg_eids))
    e = fx.Int32(
        rocdl.readfirstlane(
            T.i32,
            buffer_ops.buffer_load(eids_rsrc, m_block_idx, vec_width=1, dtype=fx.Int32),
        )
    )
    if const_expr(expert_offset != 0):
        e = e - fx.Int32(expert_offset)
    m_row = m_block_idx * BM

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
        return bq_view(arg_bq, e * N_OUT + col, KH4, K_TILES_TOTAL)

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
    bq_frags = [
        [[fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(4)]
        for _ in range_constexpr(K_TILES_TOTAL)
    ]
    bs_frags = [
        [fx.make_fragment_like(bs_frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(K_TILES_TOTAL)
    ]
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    a_ndw = 8 if is_f8_a else 4
    a_frags = [[fx.make_rmem_tensor(a_ndw, fx.Int32) for _ in range_constexpr(2)] for _ in range_constexpr(kMChunks)]
    c_frags = [[fx.make_rmem_tensor(4, fx.Float32) for _ in range_constexpr(4)] for _ in range_constexpr(kMChunks)]

    def issue_b_load_tile(kt):
        for j in range_constexpr(4):
            for half in range_constexpr(2):
                fx.copy(b_catom, bq_views[j][lane_div_16, lane_mod_16, kt, half, None], bq_frags[kt][j][half])

    def issue_b_scale_tile(kt):
        for mw in range_constexpr(2):
            fx.copy(bs_copy_atom, bscale_views[mw][lane_div_16, lane_mod_16, kt, None], bs_frags[kt][mw])

    def issue_a_ds_read(slot):
        issue_a_ds_read_dt(s_aq_base, slot, slot_bytes, KH_TILE_A, lane_div_16, lane_mod_16, is_f8_a, a_frags)

    aq_num_records = fx.Int64(i32_max_m_blocks) * fx.Int64(BM * K_BYTES)

    def issue_a_load_lds(slot, kt):
        issue_a_load_lds_dt(arg_aq, aq_num_records, s_aq_base, slot, kt, m_row, wave, lane, is_f8_a, KH_TILE_A, K_BYTES)

    mma_atoms = scale_mma_atoms(is_f8_a)

    def mfma_cluster(kt, sa):
        for J in range_constexpr(4):
            mni, in_b = J // 2, J % 2
            sb = Vec(bs_frags[kt][mni].load())[0]
            mma_one_j(J, in_b, sa, sb, bq_frags[kt], a_frags, c_frags, mma_atoms)

    for i in range_constexpr(kMChunks):
        for J in range_constexpr(4):
            c_frags[i][J].store(zero4)

    a_scale_v = [load_a_scale_tile(kt) for kt in range_constexpr(K_TILES_TOTAL)]
    for kt in range_constexpr(K_TILES_TOTAL):
        issue_b_load_tile(kt)
        issue_b_scale_tile(kt)

    if const_expr(K_TILES_TOTAL <= kStages):
        # Fast path: preload all A-tiles into their LDS slots.
        for kt in range_constexpr(K_TILES_TOTAL):
            issue_a_load_lds(kt, kt)
        for kt in range_constexpr(K_TILES_TOTAL):
            gpu.barrier()
            issue_a_ds_read(kt % kStages)
            mfma_cluster(kt, a_scale_v[kt])
    else:
        # Prologue: prime the first kStages A-tiles before the streaming loop.
        for kt in range_constexpr(kStages):
            issue_a_load_lds(kt, kt)
        for OFFSET in range_constexpr(kUnroll):
            kt = OFFSET
            gpu.barrier()
            issue_a_ds_read(kt % aStages)
            issue_a_load_lds((kStages + OFFSET) % aStages, kStages + OFFSET)
            mfma_cluster(kt, a_scale_v[kt])
        for S in range_constexpr(kStages):
            kt = K_TILES_TOTAL - kStages + S
            gpu.barrier()
            issue_a_ds_read(kt % aStages)
            mfma_cluster(kt, a_scale_v[kt])

    accm_vecs = [[Vec(c_frags[i][J].load()) for J in range(4)] for i in range(kMChunks)]
    return accm_vecs, m_row, n_block_idx


def atomic_bf16_epilog(
    lds_acc_base, accm, arg_out, arg_stids, arg_sweights, m_row, n_block_idx, wave, lane, i32_M, N_OUT
):
    """LOCAL weighted atomic-bf16 combine (single-rank / intra-node)."""
    kMChunks_l = BM // 16
    M_REPS = BM // 8
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    lds_base_fptr = lds_typed_ptr(lds_acc_base, T.f32)

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

    for i in range_constexpr(kMChunks_l):
        row_base = fx.Int32(i * 16) + lane_div_16 * 4
        for J in range_constexpr(4):
            col = wave * 64 + J * 16 + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + v) * BN + col
                lds_base_fptr[idx] = fx.Float32(vec[v])

    gpu.barrier()

    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:
            row_base_addr = token_id * N_OUT + n_block_idx * BN + col_start
            for s in range_constexpr(4):
                idx0 = row_in_block * BN + col_start + s * 64
                v2 = Vec(lds_vec_load(lds_acc_base, idx0 * 4, Vec.make_type(2, fx.Float32), fx.Float32, align=8))
                pk = Vec.from_elements([v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32).to(fx.BFloat16)
                off = (row_base_addr + s * 64) * 2
                buffer_atomic_add(pk, out_rsrc, off, fx.Int32(0), fx.Int32(0))


def lds_bytes_for_gemm2(D_INTER, a_dtype, aStages=kAStages):
    KH_TILE_A = BK // (1 if a_dtype == "fp8" else 2)
    s_aq_bytes = aStages * BM * KH_TILE_A
    lds_acc_bytes = BM * BN * 4
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
):
    N_OUT = model_dim
    D_INTER = inter_dim
    num_n_blocks = N_OUT // 256
    assert N_OUT % 256 == 0
    grid_x = num_cu * grid_mult
    lds_bytes = lds_bytes_for_gemm2(D_INTER, a_dtype, aStages)
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
            accm_vecs, m_row, n_block_idx = gemm2_compute(
                lds_base_i32,
                addr_ascale,
                addr_bq,
                addr_bscale,
                addr_eids,
                addr_aq,
                i32_max_m_blocks,
                itv,
                lane,
                wave,
                N_OUT=N_OUT,
                D_INTER=D_INTER,
                aStages=aStages,
                a_dtype=a_dtype,
                use_nt=use_nt,
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
