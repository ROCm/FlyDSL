# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: I001
"""Fused stage2 with FP8xFP4 grouped-GEMM2 and weighted cross-rank P2P scatter."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from kernels.gemm.fp8_gemm_utils import ceildiv
from kernels.moe.mixed_moe_gemm_2stage.common import _barrier

from .group_gemm2 import BM, BN, gemm2_compute, lds_bytes_for_gemm2, lds_typed_ptr, lds_vec_load

# fmt: off
def p2p_scatter_epilog(lds_acc_base, accm, addr_stids, addr_sweights, addr_tis, addr_p2p_comb_inp, srcmap_row_base,
    n_block_idx, wave, lane, *, N_OUT, npes, topk, log2_max_tok, mask_max_tok):
# fmt: on
    """CShuffle, weight, and P2P-store one GEMM2 tile."""
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

    packed = []
    weight = []
    # Compact A2 rows map to fixed-slot srcmap/weight rows through tile_row_base.
    for mr in range_constexpr(M_REPS):
        sorted_pos = srcmap_row_base + fx.Int32(mr * 8) + m_lane
        packed.append(buffer_ops.buffer_load(r_stids, sorted_pos, vec_width=1, dtype=fx.Int32))
        weight.append(buffer_ops.buffer_load(r_sweights, sorted_pos, vec_width=1, dtype=fx.Float32))

    gpu.barrier()

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
        if valid:
            for sc in range_constexpr(4):
                idx0 = row_in_block * BN + col_start + fx.Int32(sc * 64)
                v2 = Vec(
                    lds_vec_load(lds_acc_base, idx0 * fx.Int32(4), Vec.make_type(2, fx.Float32), fx.Float32, align=8)
                )
                pk = Vec.from_elements([v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32).to(fx.BFloat16)
                col_byte = (col_start + fx.Int32(sc * 64)) * fx.Int32(out_elem_bytes)
                off_bytes = dest_row_byte + col_byte
                buffer_ops.buffer_store(pk, rsrc_dst, off_bytes, offset_is_bytes=True, cache_modifier=2)


# fmt: off
def compile_mega_moe_stage2(*, model_dim: int, inter_dim: int, experts: int, topk: int, rank: int, npes: int,
    max_tok: int, aStages: int=3, use_nt: bool=True, num_cu: int=256, grid_mult: int=1):
# fmt: on
    assert max_tok > 0 and (max_tok & (max_tok - 1)) == 0, "max_tok must be power of two"
    log2_max_tok = max_tok.bit_length() - 1
    mask_max_tok = max_tok - 1
    N_OUT = model_dim
    D_INTER = inter_dim
    num_n_blocks = N_OUT // 256
    assert N_OUT % 256 == 0
    grid_x = num_cu * grid_mult
    lds_bytes = lds_bytes_for_gemm2(D_INTER, aStages)
    TOTAL_THREADS = 256

    @flyc.kernel(known_block_size=[TOTAL_THREADS, 1, 1])
    # fmt: off
    def kernel(addr_aq: fx.Int64, addr_ascale: fx.Int64, addr_bq: fx.Int64, addr_bscale: fx.Int64,
        addr_eids: fx.Int64, addr_stids: fx.Int64, addr_sweights: fx.Int64, addr_trb: fx.Int64,
        addr_num_valid: fx.Int64, addr_tis: fx.Int64, addr_p2p_comb_inp: fx.Int64, i32_max_m_blocks: fx.Int32):
    # fmt: on
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
            # fmt: off
            accm_vecs, m_row, n_block_idx = gemm2_compute(lds_base_i32, addr_ascale, addr_bq, addr_bscale,
                addr_eids, addr_aq, i32_max_m_blocks, itv, lane, wave, N_OUT=N_OUT, D_INTER=D_INTER,
                aStages=aStages, use_nt=use_nt, expert_offset=rank * experts)
            # fmt: on
            # A2 data/scales are compact; srcmap/weights remain fixed-slot.
            trb_base = buffer_ops.buffer_load(trb_rsrc, m_row // fx.Int32(BM), vec_width=1, dtype=fx.Int32)
            # fmt: off
            p2p_scatter_epilog(lds_base_i32, accm_vecs, addr_stids, addr_sweights, addr_tis, addr_p2p_comb_inp,
                trb_base, n_block_idx, wave, lane, N_OUT=N_OUT, npes=npes, topk=topk, log2_max_tok=log2_max_tok,
                mask_max_tok=mask_max_tok)
            # fmt: on
            # overlap: wait only on LDS (cshuffle) so remote SLC P2P stores drain async (xGMI overlap).
            _barrier(vmcnt=63, lgkmcnt=0)
            itv = itv + fx.Int32(grid_x)

    @flyc.jit
    # fmt: off
    def launch(addr_aq: fx.Int64, addr_ascale: fx.Int64, addr_bq: fx.Int64, addr_bscale: fx.Int64,
        addr_eids: fx.Int64, addr_stids: fx.Int64, addr_sweights: fx.Int64, addr_trb: fx.Int64,
        addr_num_valid: fx.Int64, addr_tis: fx.Int64, addr_p2p_comb_inp: fx.Int64, i32_max_m_blocks: fx.Int32,
        stream: fx.Stream):
    # fmt: on
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
