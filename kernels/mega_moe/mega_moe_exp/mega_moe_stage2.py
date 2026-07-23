# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""MegaMoE fused stage-2 = aiter-v2 group_gemm2 (down-proj) + cross-rank P2P-scatter combine.

Reuses `group_gemm2.gemm2_compute` for the A2@W2 math, then scatters each output token row P2P to
its SOURCE rank's `shmem_comb_inp_tok[dest_lid*k + s]` (weighted in the cshuffle write, SLC store,
VMEM-relaxed barrier for xGMI overlap) — exactly the current MegaMoE fused_p2p_scatter combine. The
host then runs the trimmed `combine_no_stage1` (weight P2P + reduction). CDNA4-only.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from kernels.gemm.fp8_gemm_utils import ceildiv

from .group_gemm2 import (
    BM,
    BN,
    gemm2_compute,
    lds_bytes_for_gemm2,
    lds_typed_ptr,
    lds_vec_load,
    spart_group_m01,
    spart_output_tile_index,
    waitcnt_barrier,
)


def p2p_scatter_epilog(
    lds_acc_base,
    accm,
    addr_stids,
    addr_sweights,
    addr_tis,
    addr_p2p_comb_inp,
    srcmap_row_base,
    n_block_idx,
    wave,
    lane,
    *,
    N_OUT,
    npes,
    topk,
    log2_max_tok,
    mask_max_tok,
    doweight,
):
    """cshuffle accm -> LDS -> weighted bf16 -> P2P store to dest rank's shmem_comb_inp_tok slot.

    dest slot = dest_lid*topk + s (s = stids>>24); dest_enc = shmem_tok_id_to_src[t]; token row is
    N_OUT bf16 wide, this tile writes its n_block*BN..+BN column slice. Padding is dropped by the
    global-source-token and top-k-slot bounds.
    """
    M_REPS = BM // 8
    kMChunks = BM // 16
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    lds_base_fptr = lds_typed_ptr(lds_acc_base, T.f32)

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // 32
    n_lane = tx_i32 % 32
    col_start = n_lane * 2
    out_elem_bytes = 2  # bf16
    token_nbytes = N_OUT * out_elem_bytes

    r_stids = buffer_ops.create_buffer_resource_from_addr(addr_stids)
    r_sweights = buffer_ops.create_buffer_resource_from_addr(addr_sweights)
    r_tis = buffer_ops.create_buffer_resource_from_addr(addr_tis)
    r_p2p = buffer_ops.create_buffer_resource_from_addr(addr_p2p_comb_inp)

    # prefetch per-row sorted id + weight (invariant across the store loop).
    packed = []
    weight = []
    # srcmap/weight live in the FIXED-SLOT arrays (srcmap_em/wts_em, written by dispatch at le*cap+off);
    # this tile's compact rows map to fixed-slot rows via srcmap_row_base = trb[m_block] (+ intra-tile row).
    for mr in range_constexpr(M_REPS):
        sorted_pos = srcmap_row_base + fx.Int32(mr * 8) + m_lane
        packed.append(buffer_ops.buffer_load(r_stids, sorted_pos, vec_width=1, dtype=fx.Int32))
        weight.append(buffer_ops.buffer_load(r_sweights, sorted_pos, vec_width=1, dtype=fx.Float32))

    gpu.barrier()

    # cshuffle accm -> lds_acc (f32), matching aiter atomic_bf16_epilog write layout.
    for i in range_constexpr(kMChunks):
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
        p = packed[mr]
        t = p & fx.Int32(0x00FFFFFF)
        s = p >> fx.Int32(24)
        valid = (t < fx.Int32(npes * (mask_max_tok + 1))) & (s < fx.Int32(topk))
        t_safe = valid.select(t, fx.Int32(0))
        dest_enc = buffer_ops.buffer_load(r_tis, t_safe, vec_width=1, dtype=fx.Int32)
        dest_pe = dest_enc >> fx.Int32(log2_max_tok)
        dest_lid = dest_enc & fx.Int32(mask_max_tok)
        valid = valid & (dest_pe < fx.Int32(npes))
        dest_pe_safe = valid.select(dest_pe, fx.Int32(0))
        peer_base = buffer_ops.buffer_load(r_p2p, dest_pe_safe, vec_width=1, dtype=fx.Int64)
        rsrc_dst = buffer_ops.create_buffer_resource_from_addr(peer_base)
        slot = dest_lid * fx.Int32(topk) + s
        dest_row_byte = slot * fx.Int32(token_nbytes) + n_block_idx * fx.Int32(BN * out_elem_bytes)
        wmul = weight[mr] if const_expr(doweight) else fx.Float32(1.0)
        if valid:
            for sc in range_constexpr(4):
                idx0 = row_in_block * BN + col_start + fx.Int32(sc * 64)
                v2 = Vec(
                    lds_vec_load(lds_acc_base, idx0 * fx.Int32(4), Vec.make_type(2, fx.Float32), fx.Float32, align=8)
                )
                pk = Vec.from_elements([v2[0] * wmul, v2[1] * wmul], fx.Float32).to(fx.BFloat16)
                col_byte = (col_start + fx.Int32(sc * 64)) * fx.Int32(out_elem_bytes)
                off_bytes = dest_row_byte + col_byte
                buffer_ops.buffer_store(pk, rsrc_dst, off_bytes, offset_is_bytes=True, cache_modifier=2)


def compile_mega_moe_stage2(
    *,
    model_dim: int,  # N_OUT (down-proj output = hidden)
    inter_dim: int,  # D_INTER (contraction)
    experts: int,
    topk: int,
    rank: int,
    npes: int,
    max_tok: int,
    a_dtype: str = "fp8",
    aStages: int = 3,
    use_nt: bool = True,
    doweight: bool = True,
    num_cu: int = 256,
    grid_mult: int = 1,
    SBM: int = None,
    inter_dim_pad: int = 0,
    model_dim_pad: int = 0,
    g2_bhoist: bool = True,
    g2_ascale_pf: bool = True,
    g2_spart: int = 0,
):
    assert max_tok > 0 and (max_tok & (max_tok - 1)) == 0, "max_tok must be power of two"
    log2_max_tok = max_tok.bit_length() - 1
    mask_max_tok = max_tok - 1
    N_OUT = model_dim
    D_INTER = inter_dim
    num_n_blocks = N_OUT // 256
    assert N_OUT % 256 == 0
    _spart = spart_group_m01(g2_spart)
    grid_x = num_cu * grid_mult
    lds_bytes = lds_bytes_for_gemm2(D_INTER, a_dtype, aStages)
    TOTAL_THREADS = 256

    @flyc.kernel(known_block_size=[TOTAL_THREADS, 1, 1])
    def kernel(
        addr_aq: fx.Int64,
        addr_ascale: fx.Int64,
        addr_bq: fx.Int64,
        addr_bscale: fx.Int64,
        addr_eids: fx.Int64,
        addr_stids: fx.Int64,
        addr_sweights: fx.Int64,
        addr_trb: fx.Int64,  # tile_row_base (fixed-slot base per compact tile) for srcmap/weight decode
        addr_num_valid: fx.Int64,
        addr_tis: fx.Int64,
        addr_p2p_comb_inp: fx.Int64,
        i32_max_m_blocks: fx.Int32,
    ):
        lds_base_i32 = fx.Int32(0)
        trb_rsrc = buffer_ops.create_buffer_resource_from_addr(addr_trb)
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid % fx.Int32(64)
        wave = tid // fx.Int32(64)

        nv_rsrc = buffer_ops.create_buffer_resource_from_addr(addr_num_valid)
        num_valid = buffer_ops.buffer_load(nv_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Int32)
        num_m_tiles = ceildiv(num_valid, fx.Int32(BM))
        total_work = num_m_tiles * fx.Int32(num_n_blocks)

        itv = fx.Int32(fx.block_idx.x)
        while itv < total_work:
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
                expert_offset=rank * experts,
                SBM=SBM,
                inter_dim_pad=inter_dim_pad,
                model_dim_pad=model_dim_pad,
                g2_bhoist=g2_bhoist,
                g2_ascale_pf=g2_ascale_pf,
            )
            # compact tile -> fixed-slot base for srcmap/weight (a2 DATA/SCALE are compact; srcmap_em/
            # wts_em are fixed-slot). m_block_idx = m_row // BM (gemm2 returns m_row = m_block*BM).
            trb_base = buffer_ops.buffer_load(trb_rsrc, m_row // fx.Int32(BM), vec_width=1, dtype=fx.Int32)
            p2p_scatter_epilog(
                lds_base_i32,
                accm_vecs,
                addr_stids,
                addr_sweights,
                addr_tis,
                addr_p2p_comb_inp,
                trb_base,
                n_block_idx,
                wave,
                lane,
                N_OUT=N_OUT,
                npes=npes,
                topk=topk,
                log2_max_tok=log2_max_tok,
                mask_max_tok=mask_max_tok,
                doweight=doweight,
            )
            # overlap: wait only on LDS (cshuffle) so remote SLC P2P stores drain async (xGMI overlap).
            waitcnt_barrier(vmcnt=63, lgkmcnt=0)
            itv = itv + fx.Int32(grid_x)

    @flyc.jit
    def launch(
        addr_aq: fx.Int64,
        addr_ascale: fx.Int64,
        addr_bq: fx.Int64,
        addr_bscale: fx.Int64,
        addr_eids: fx.Int64,
        addr_stids: fx.Int64,
        addr_sweights: fx.Int64,
        addr_trb: fx.Int64,
        addr_num_valid: fx.Int64,
        addr_tis: fx.Int64,
        addr_p2p_comb_inp: fx.Int64,
        i32_max_m_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            addr_aq,
            addr_ascale,
            addr_bq,
            addr_bscale,
            addr_eids,
            addr_stids,
            addr_sweights,
            addr_trb,
            addr_num_valid,
            addr_tis,
            addr_p2p_comb_inp,
            i32_max_m_blocks,
        ).launch(grid=(grid_x, 1, 1), block=(TOTAL_THREADS, 1, 1), smem=lds_bytes, stream=stream)

    return launch
