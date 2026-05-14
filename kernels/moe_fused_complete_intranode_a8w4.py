# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Intranode A8W4 stage1 — single host entry, multi-kernel MegaMoE path.

One call to ``launch_intranode_stage1_fused_a8w4`` (no host sort/scatter between steps). Each iter
launches on the same stream:

1. ``metadata_dispatch+pull_to_pool+reset`` — routing/handshake, local pool build, metadata reset
2. ``gemm_local`` — A8W4 MoE GEMM reading A from the local pool

Stage1 only (no GEMM2 / combine). Set ``MOE_STAGE1_LOG=1`` for per-rank phase traces.
"""

from __future__ import annotations

import os
import sys
import time

import flydsl.expr as fx
from flydsl.expr import Stream

from kernels.moe_fused_dispatch_gather_gemm1_a8w4 import compile_dispatch_gemm1_local_a8w4
from kernels.moe_pull_to_pool_a8w4 import make_stage1_metadata_pull_to_pool_a8w4_jit
from kernels.moe_sorting_flydsl import get_moe_sorting_workspace, moe_sorting_const_sizes


def _stage1_trace(rank: int, msg: str) -> None:
    if os.environ.get("MOE_STAGE1_LOG", "0").strip().lower() in ("0", "false", "no"):
        return
    line = f"[stage1_fused r{rank}] {msg} t={time.monotonic():.3f}"
    print(line, flush=True)
    print(line, file=sys.stderr, flush=True)


def _select_stage1_grid(
    *,
    npes: int,
    max_tok_per_rank: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    requested_blocks: int,
    requested_wpb: int,
) -> tuple[int, int]:
    """Select K1/K2/K4 launch geometry.

    The shape-aware tables come from ``sweep_deepseek_stage1_grid_a8w4.py``
    and use a routing-agnostic robust choice across all-to-all and
    same-pe-groups, not the per-routing absolute best. This keeps routing mode
    out of the public compile API while avoiding known bad grids.
    """
    default_blocks = min(64, max(8, int(npes) * 8))
    default_wpb = 4

    is_flash_stage1 = (
        int(model_dim) == 7168
        and int(inter_dim) == 2048
        and int(experts) == 256
        and int(topk) == 8
    )
    if is_flash_stage1 and int(npes) == 8:
        mtpr = int(max_tok_per_rank)
        if mtpr <= 512:
            default_blocks, default_wpb = 64, 8
        elif mtpr <= 2048:
            default_blocks, default_wpb = 80, 8
        elif mtpr <= 8192:
            default_blocks, default_wpb = 96, 8
        else:
            default_blocks, default_wpb = 128, 8

    is_v4_pro_stage1 = (
        int(model_dim) == 7168
        and int(inter_dim) == 3072
        and int(experts) == 384
        and int(topk) == 6
    )
    if is_v4_pro_stage1 and int(npes) == 8:
        mtpr = int(max_tok_per_rank)
        if mtpr <= 32:
            default_blocks, default_wpb = 64, 8
        elif mtpr <= 64:
            default_blocks, default_wpb = 64, 4
        elif mtpr <= 512:
            default_blocks, default_wpb = 64, 8
        elif mtpr <= 1024:
            default_blocks, default_wpb = 96, 8
        elif mtpr <= 2048:
            default_blocks, default_wpb = 64, 8
        elif mtpr <= 8192:
            default_blocks, default_wpb = 96, 8
        else:
            default_blocks, default_wpb = 128, 8

    return max(int(requested_blocks), default_blocks), max(int(requested_wpb), default_wpb)


def _make_intranode_stage1_fused_pipeline(
    *,
    rank: int,
    npes: int,
    experts_per_rank: int,
    experts_per_token: int,
    max_tok_per_rank: int,
    dispatch_block_num: int,
    dispatch_warp_num_per_block: int,
    token_row_bytes: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
):
    scale_mx_blocks = int(model_dim) // 32
    stage1_dispatch_block_num, stage1_dispatch_warp_num_per_block = _select_stage1_grid(
        npes=npes,
        max_tok_per_rank=max_tok_per_rank,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        requested_blocks=dispatch_block_num,
        requested_wpb=dispatch_warp_num_per_block,
    )
    pull_block_num = stage1_dispatch_block_num
    _stage1_trace(rank, "JIT compile stage1_metadata_pull_to_pool")
    metadata_pull = make_stage1_metadata_pull_to_pool_a8w4_jit(
        rank=rank,
        npes=npes,
        experts_per_rank=experts_per_rank,
        experts_per_token=experts_per_token,
        max_tok_per_rank=max_tok_per_rank,
        num_experts=int(experts),
        tile_m=int(tile_m),
        block_num=pull_block_num,
        warp_num_per_block=stage1_dispatch_warp_num_per_block,
        token_row_bytes=token_row_bytes,
        scale_mx_blocks=scale_mx_blocks,
    )
    _stage1_trace(rank, "JIT compile gemm_local GEMM1")
    gemm1 = compile_dispatch_gemm1_local_a8w4(
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
        compile_rank=int(rank),
        max_tok_per_rank=int(max_tok_per_rank),
    )
    _stage1_trace(rank, "JIT compile done")
    _max_padded, _max_m_blocks, _ = moe_sorting_const_sizes(
        max_tok_per_rank=max_tok_per_rank,
        experts_per_token=experts_per_token,
        num_experts=int(experts),
        block_size=int(tile_m),
    )
    _sort_ws_min_ints = (
        3 * int(experts)
        + 3 * int(_max_padded)
        + 3 * int(_max_m_blocks)
        + 3 * int(npes)
        + 1
    )
    _sort_ws = get_moe_sorting_workspace(
        num_experts=int(experts),
        device=f"cuda:{int(rank)}",
        min_ints=_sort_ws_min_ints,
    )
    _k2_barrier_phase_idx = 3 * int(experts) + 3 * int(_max_padded) + 3 * int(_max_m_blocks) + 1
    _sort_ws[_k2_barrier_phase_idx].fill_(4)
    _sort_ws_addr = fx.Int64(_sort_ws.data_ptr())

    def launch_intranode_stage1_fused_a8w4(
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
        addr_recv_meta: fx.Int64,
        addr_tis: fx.Int64,
        addr_inp_tok: fx.Int64,
        addr_x_staging: fx.Int64,
        i32_cur_tok: fx.Int32,
        addr_p2p_base_x: fx.Int64,
        addr_p2p_scale: fx.Int64,
        i32_pool_base: fx.Int32,
        addr_pull_row_i64: fx.Int64,
        addr_p2p_pull_row: fx.Int64,
        arg_scale_mx_u8,
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
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: Stream = Stream(None),
    ):
        # Keep the workspace tensor alive for the lifetime of this launch closure;
        # K12R receives only its raw data pointer.
        _ = _sort_ws
        _stage1_trace(rank, "launch stage1_metadata_pull_to_pool")
        metadata_pull(
            addr_idx,
            addr_wts,
            addr_tok_off,
            addr_recv_num,
            addr_dest_ctr,
            addr_disp_bar,
            addr_tok_map,
            addr_total_rv,
            addr_tis,
            addr_p2p_tok_off,
            addr_p2p_tis,
            addr_p2p_recv_num,
            addr_p2p_recv_meta,
            addr_inp_tok,
            addr_x_staging,
            addr_pull_row_i64,
            addr_p2p_pull_row,
            i32_cur_tok,
            addr_recv_meta,
            addr_p2p_base_x,
            addr_p2p_scale,
            fx.Int64(arg_scale_x.data_ptr()),
            fx.Int64(arg_sorted_token_ids.data_ptr()),
            fx.Int64(arg_expert_ids.data_ptr()),
            fx.Int64(arg_max_token_ids.data_ptr()),
            fx.Int64(arg_sorted_weights.data_ptr()),
            _sort_ws_addr,
            i32_pool_base,
            stream=stream,
        )
        _stage1_trace(rank, "launch stage1_metadata_pull_to_pool returned")
        _stage1_trace(rank, "launch gemm_local GEMM1")
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
            addr_x_staging,
            i32_pool_base,
            stream=stream,
        )
        _stage1_trace(rank, "launch gemm_local GEMM1 returned")
        _stage1_trace(rank, "launch done")

    return launch_intranode_stage1_fused_a8w4


def compile_intranode_stage1_fused_a8w4(
    *,
    rank: int,
    npes: int,
    experts_per_rank: int,
    experts_per_token: int,
    max_tok_per_rank: int,
    dispatch_block_num: int,
    dispatch_warp_num_per_block: int,
    token_row_bytes: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool = False,
):
    """Build the stage1 fused launch closure (metadata + pull_to_pool + gemm_local)."""
    return _make_intranode_stage1_fused_pipeline(
        rank=rank,
        npes=npes,
        experts_per_rank=experts_per_rank,
        experts_per_token=experts_per_token,
        max_tok_per_rank=max_tok_per_rank,
        dispatch_block_num=dispatch_block_num,
        dispatch_warp_num_per_block=dispatch_warp_num_per_block,
        token_row_bytes=token_row_bytes,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=doweight_stage1,
    )


def compile_complete_intranode_fused_stage1_a8w4(**kwargs):
    """Alias for :func:`compile_intranode_stage1_fused_a8w4`."""
    return compile_intranode_stage1_fused_a8w4(**kwargs)


__all__ = [
    "compile_intranode_stage1_fused_a8w4",
    "compile_complete_intranode_fused_stage1_a8w4",
]
