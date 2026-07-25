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


def p2p_scatter_prefetch(r_stids, r_sweights, r_tis, srcmap_row_base, m_lane, recv_cap, BM):
    """Issue this tile's scatter-metadata global loads (srcmap, weight, and the dependent
    tok_id_to_src -> dest_enc) BEFORE the gemm2 MFMA, so the chained VMEM latency is hidden behind the
    matmul (V1 does the same via K-loop prefetch). Returns per-row register lists consumed by the epilog.
    """
    M_REPS = BM // 8
    packed, weight, dest_enc = [], [], []
    for mr in range_constexpr(M_REPS):
        sorted_pos = srcmap_row_base + fx.Int32(mr * 8) + m_lane
        p = buffer_ops.buffer_load(r_stids, sorted_pos, vec_width=1, dtype=fx.Int32)
        packed.append(p)
        weight.append(buffer_ops.buffer_load(r_sweights, sorted_pos, vec_width=1, dtype=fx.Float32))
        t = p & fx.Int32(0x00FFFFFF)
        t_safe = (t < fx.Int32(recv_cap)).select(t, fx.Int32(0))
        dest_enc.append(buffer_ops.buffer_load(r_tis, t_safe, vec_width=1, dtype=fx.Int32))
    return packed, weight, dest_enc


def p2p_scatter_epilog(
    lds_acc_base,
    accm,
    packed,
    weight,
    dest_enc,
    peer_bases,
    n_block_idx,
    wave,
    lane,
    *,
    N_OUT,
    npes,
    topk,
    log2_max_tok,
    mask_max_tok,
    recv_cap,
    comb_inp_nbytes,
    doweight,
    BM=BM,
    BN=BN,
):
    """cshuffle accm -> LDS -> weighted bf16 -> P2P store to dest rank's shmem_comb_inp_tok slot.

    dest slot = dest_lid*topk + s (s = stids>>24); dest_enc = shmem_tok_id_to_src[t]; token row is
    N_OUT bf16 wide, this tile writes its n_block*BN..+BN column slice.

    Perf (V1-parity): scatter metadata (``packed`` srcmap / ``weight`` / ``dest_enc``) is PREFETCHED by
    the caller before the MFMA (:func:`p2p_scatter_prefetch`), so the chained srcmap->tok_id_to_src load
    latency is hidden behind the matmul; ``peer_bases`` are the npes P2P base ptrs pre-loaded into
    registers (no per-row P2P-table VMEM load). The store loop is branchless -- invalid rows are
    redirected past the (bounded) buffer end and dropped by hardware OOB.
    """
    M_REPS = BM // 8
    kMChunks = BM // 16
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    lds_base_fptr = lds_typed_ptr(lds_acc_base, T.f32)

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // 32
    n_lane = tx_i32 % 32
    out_elem_bytes = 2  # bf16
    token_nbytes = N_OUT * out_elem_bytes
    col8 = n_lane * fx.Int32(8)
    n_off = n_block_idx * fx.Int32(BN * out_elem_bytes) + col8 * fx.Int32(out_elem_bytes)

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
        p = packed[mr]
        t = p & fx.Int32(0x00FFFFFF)
        s = p >> fx.Int32(24)
        de = dest_enc[mr]
        dest_pe = de >> fx.Int32(log2_max_tok)
        dest_lid = de & fx.Int32(mask_max_tok)
        valid = (t < fx.Int32(recv_cap)) & (s < fx.Int32(topk)) & (dest_pe < fx.Int32(npes))
        # peer base from caller-cached registers (no per-row P2P-table VMEM load): select by dest_pe.
        dest_pe_safe = valid.select(dest_pe, fx.Int32(0))
        peer_base = peer_bases[0]
        for pe in range(1, npes):
            peer_base = (dest_pe_safe == fx.Int32(pe)).select(peer_bases[pe], peer_base)
        rsrc_dst = buffer_ops.create_buffer_resource_from_addr(peer_base, num_records_bytes=comb_inp_nbytes)
        slot = dest_lid * fx.Int32(topk) + s
        # invalid rows -> offset past the buffer end so the bounded rsrc drops the store (branchless),
        # instead of aliasing a real slot (notably (pe0,lid0,s0) == output token 0 slot 0).
        off_bytes = valid.select(slot * fx.Int32(token_nbytes) + n_off, fx.Int32(comb_inp_nbytes))
        wmul = weight[mr] if const_expr(doweight) else fx.Float32(1.0)
        row_in_block = fx.Int32(mr * 8) + m_lane
        idx0 = row_in_block * BN + col8
        v8 = Vec(
            lds_vec_load(lds_acc_base, idx0 * fx.Int32(4), Vec.make_type(8, fx.Float32), fx.Float32, align=16)
        )
        pk = Vec.from_elements([v8[i] * wmul for i in range(8)], fx.Float32).to(fx.BFloat16)
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
    recv_cap: int = None,
    comb_inp_nbytes: int = None,
    bench_no_scatter: bool = False,
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
    BM: int = BM,
    BN: int = BN,
):
    assert max_tok > 0 and (max_tok & (max_tok - 1)) == 0, "max_tok must be power of two"
    assert BM in (16, 32, 64, 128), f"BM must be in {{16,32,64,128}}, got {BM}"
    log2_max_tok = max_tok.bit_length() - 1
    mask_max_tok = max_tok - 1
    N_OUT = model_dim
    D_INTER = inter_dim
    # SBM = sort padding unit (fixed-slot tile = sort_block_m rows). The gemm2 compute tile BM may be
    # smaller; srcmap/weight/trb are laid out per SBM, so decode indexes by SBM (not BM).
    _SBM = BM if SBM is None else SBM
    assert _SBM % BM == 0, f"SBM ({_SBM}) must be a multiple of BM ({BM})"
    # t (source-token id) indexes tok_id_to_src, sized effective_max_recv. Bound the scatter guard by
    # it; default to the (looser, pre-fix) npes*max_tok when the caller doesn't pass the real capacity.
    _recv_cap = npes * max_tok if recv_cap is None else int(recv_cap)
    # Per-peer combine-input buffer size in bytes (dest_lid*topk+s rows, each N_OUT*2 bf16 bytes). Bounds
    # the scatter dest so invalid/padding rows redirected to this offset are dropped (hardware OOB).
    _comb_inp_nbytes = max_tok * topk * N_OUT * 2 if comb_inp_nbytes is None else int(comb_inp_nbytes)
    num_n_blocks = N_OUT // 256
    assert N_OUT % 256 == 0
    _spart = spart_group_m01(g2_spart)
    grid_x = num_cu * grid_mult
    lds_bytes = lds_bytes_for_gemm2(D_INTER, a_dtype, aStages, False, BM, BN)
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
        # P2P peer-base table is kernel-invariant: load the npes shmem_comb_inp_tok base ptrs into
        # registers ONCE (not per scatter row), so the epilog selects the dest peer base from registers.
        if const_expr(not bench_no_scatter):
            _r_p2p_tbl = buffer_ops.create_buffer_resource_from_addr(addr_p2p_comb_inp)
            _peer_bases = [
                buffer_ops.buffer_load(_r_p2p_tbl, fx.Int32(_pe), vec_width=1, dtype=fx.Int64)
                for _pe in range(npes)
            ]
            # scatter-metadata buffer resources + this thread's m-lane, hoisted (used by the per-tile
            # prefetch issued before each tile's MFMA).
            _r_stids = buffer_ops.create_buffer_resource_from_addr(addr_stids)
            _r_sweights = buffer_ops.create_buffer_resource_from_addr(addr_sweights)
            _r_tis = buffer_ops.create_buffer_resource_from_addr(addr_tis)
            _m_lane = fx.Int32(gpu.thread_id("x")) // fx.Int32(32)

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
            # Prefetch this tile's scatter metadata BEFORE the MFMA (coords match gemm2_compute's own
            # m_row/n_block derivation), so the srcmap->tok_id_to_src load chain overlaps the matmul.
            if const_expr(not bench_no_scatter):
                _m_block_idx = unit_bx // fx.Int32(num_n_blocks)
                _n_block_pre = unit_bx - _m_block_idx * fx.Int32(num_n_blocks)
                _m_row_pre = _m_block_idx * fx.Int32(BM)
                _trb_base = buffer_ops.buffer_load(
                    trb_rsrc, _m_row_pre // fx.Int32(_SBM), vec_width=1, dtype=fx.Int32
                )
                _srcmap_row_base = _trb_base + (_m_row_pre - (_m_row_pre // fx.Int32(_SBM)) * fx.Int32(_SBM))
                _pf_packed, _pf_weight, _pf_dest_enc = p2p_scatter_prefetch(
                    _r_stids, _r_sweights, _r_tis, _srcmap_row_base, _m_lane, _recv_cap, BM
                )
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
                BM=BM,
                BN=BN,
                expert_offset=rank * experts,
                SBM=SBM,
                inter_dim_pad=inter_dim_pad,
                model_dim_pad=model_dim_pad,
                g2_bhoist=g2_bhoist,
                g2_ascale_pf=g2_ascale_pf,
            )
            # compact tile -> fixed-slot base for srcmap/weight (a2 DATA/SCALE are compact; srcmap_em/
            # wts_em are fixed-slot, laid out per SBM). trb is per-SBM sort tile, so index by m_row//SBM;
            # a BM<SBM compute tile then offsets (m_row % SBM) rows into that sort tile's fixed slots.
            if const_expr(bench_no_scatter):
                # Perf-attribution only: keep the gemm2 compute live (sink one accm element to the peer
                # buffer) but skip the whole p2p_scatter_epilog, so this run times gemm2 compute alone.
                _rp = buffer_ops.create_buffer_resource_from_addr(addr_p2p_comb_inp)
                _pb = buffer_ops.buffer_load(_rp, fx.Int32(0), vec_width=1, dtype=fx.Int64)
                _rs = buffer_ops.create_buffer_resource_from_addr(_pb, num_records_bytes=_comb_inp_nbytes)
                _sv = Vec(accm_vecs[0][0])  # 4xf32 accumulator fragment -> one 16B b128 store keeps gemm2 live
                buffer_ops.buffer_store(
                    _sv, _rs, (unit_bx & fx.Int32(255)) * fx.Int32(16), offset_is_bytes=True
                )
            else:
                p2p_scatter_epilog(
                    lds_base_i32,
                    accm_vecs,
                    _pf_packed,
                    _pf_weight,
                    _pf_dest_enc,
                    _peer_bases,
                    _n_block_pre,
                    wave,
                    lane,
                    N_OUT=N_OUT,
                    npes=npes,
                    topk=topk,
                    log2_max_tok=log2_max_tok,
                    mask_max_tok=mask_max_tok,
                    recv_cap=_recv_cap,
                    comb_inp_nbytes=_comb_inp_nbytes,
                    doweight=doweight,
                    BM=BM,
                    BN=BN,
                )
            # LDS-reuse sync across persistent iters (cshuffle acc unions the next tile's A-LDS region):
            # vmcnt=63 keeps P2P stores draining async (already the aiter "no release fence" win); the
            # lgkmcnt=0 + s_barrier guards the acc LDS from the next iter's A-load DMA. Not the vmem fence.
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
