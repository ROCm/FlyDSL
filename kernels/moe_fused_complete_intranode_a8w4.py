# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Complete intranode A8W4 stage1 **pipeline**: metadata dispatch + fused GEMM1 / activation.

Orchestrates two FlyDSL JIT launches on one stream (ordered execution):

1. :func:`kernels.moe_metadata_dispatch_recv_meta.make_metadata_dispatch_recv_meta_jit`
2. :func:`kernels.moe_fused_dispatch_gather_gemm1_a8w4.compile_fused_dispatch_gather_gemm1_a8w4`
   with ``intranode_peer_gather`` defaulting to ``(npes > 1)`` (pass ``intranode_peer_gather=False`` to
   :func:`compile_complete_intranode_fused_stage1_a8w4` for gather-free diagnostics). Multi-PE runs use
   inlined ``recv_meta`` + P2P row gather + grid barrier before MFMA; single-PE runs compile the
   gather-free path for lower latency.

**Host contract:** zero the i32 at ``addr_peer_gather_grid_bar`` immediately before each fused GEMM1 launch
(multi-PE path; harmless on single-PE). ``addr_peer_x_staging`` must match ``arg_x``.
"""

from __future__ import annotations

import functools
from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import Stream

from kernels.moe_fused_dispatch_gather_gemm1_a8w4 import (
    compile_fused_dispatch_gather_gemm1_a8w4,
)
from kernels.moe_metadata_dispatch_recv_meta import make_metadata_dispatch_recv_meta_jit


@functools.lru_cache(maxsize=32)
def _make_complete_pipeline(
    *,
    rank: int,
    npes: int,
    experts_per_rank: int,
    experts_per_token: int,
    max_tok_per_rank: int,
    dispatch_block_num: int,
    dispatch_warp_num_per_block: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    intranode_peer_gather: bool,
):
    meta = make_metadata_dispatch_recv_meta_jit(
        rank=rank,
        npes=npes,
        experts_per_rank=experts_per_rank,
        experts_per_token=experts_per_token,
        max_tok_per_rank=max_tok_per_rank,
        block_num=dispatch_block_num,
        warp_num_per_block=dispatch_warp_num_per_block,
    )
    gemm1 = compile_fused_dispatch_gather_gemm1_a8w4(
        model_dim=int(model_dim),
        inter_dim=int(inter_dim),
        experts=int(experts),
        topk=int(topk),
        tile_m=int(tile_m),
        tile_n=int(tile_n),
        tile_k=int(tile_k),
        doweight_stage1=bool(doweight_stage1),
        a_dtype="fp8",
        b_dtype="fp4",
        intranode_peer_gather=bool(intranode_peer_gather),
    )

    @flyc.jit
    def launch_complete_intranode_stage1_a8w4(
        # --- metadata dispatch ---
        addr_idx: fx.Int64,
        addr_wts: fx.Int64,
        addr_tok_off: fx.Int64,
        addr_recv_num: fx.Int64,
        addr_dest_ctr: fx.Int64,
        addr_disp_bar: fx.Int64,
        addr_tok_map: fx.Int64,
        addr_total_rv: fx.Int64,
        addr_p2p_tok_off: fx.Int64,
        addr_p2p_tis: fx.Int64,
        addr_p2p_recv_num: fx.Int64,
        addr_p2p_recv_meta: fx.Int64,
        i32_cur_tok: fx.Int32,
        # --- fused gather+GEMM1 (staging base must match arg_x) ---
        addr_recv_meta_flat: fx.Int64,
        addr_p2p_base_x: fx.Int64,
        addr_peer_x_staging: fx.Int64,
        i32_total_recv: fx.Int32,
        addr_peer_gather_grid_bar: fx.Int64,
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_max_token_ids: fx.Tensor,
        arg_bias: fx.Tensor,
        arg_out_scale_sorted: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: Stream = Stream(None),
    ):
        meta(
            addr_idx,
            addr_wts,
            addr_tok_off,
            addr_recv_num,
            addr_dest_ctr,
            addr_disp_bar,
            addr_tok_map,
            addr_total_rv,
            addr_p2p_tok_off,
            addr_p2p_tis,
            addr_p2p_recv_num,
            addr_p2p_recv_meta,
            i32_cur_tok,
            stream=stream,
        )
        gemm1(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_max_token_ids,
            arg_bias,
            arg_out_scale_sorted,
            i32_tokens_in,
            i32_inter_in,
            i32_k_in,
            i32_size_expert_ids_in,
            addr_recv_meta_flat,
            addr_p2p_base_x,
            addr_peer_x_staging,
            i32_total_recv,
            fx.Int32(npes),
            addr_peer_gather_grid_bar,
            stream=stream,
        )

    return launch_complete_intranode_stage1_a8w4


def compile_complete_intranode_fused_stage1_a8w4(
    *,
    rank: int,
    npes: int,
    experts_per_rank: int,
    experts_per_token: int,
    max_tok_per_rank: int,
    dispatch_block_num: int,
    dispatch_warp_num_per_block: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    intranode_peer_gather: Optional[bool] = None,
):
    use_peer_gather = bool(npes > 1) if intranode_peer_gather is None else bool(intranode_peer_gather)
    return _make_complete_pipeline(
        rank=int(rank),
        npes=int(npes),
        experts_per_rank=int(experts_per_rank),
        experts_per_token=int(experts_per_token),
        max_tok_per_rank=int(max_tok_per_rank),
        dispatch_block_num=int(dispatch_block_num),
        dispatch_warp_num_per_block=int(dispatch_warp_num_per_block),
        model_dim=int(model_dim),
        inter_dim=int(inter_dim),
        experts=int(experts),
        topk=int(topk),
        tile_m=int(tile_m),
        tile_n=int(tile_n),
        tile_k=int(tile_k),
        doweight_stage1=bool(doweight_stage1),
        intranode_peer_gather=use_peer_gather,
    )


__all__ = ["compile_complete_intranode_fused_stage1_a8w4"]
