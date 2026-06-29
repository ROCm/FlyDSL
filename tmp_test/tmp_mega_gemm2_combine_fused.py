# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused MoE Stage2 GEMM + EP combine kernel.

Rewrites the GEMM2 epilogue's local store_pair into a remote P2P
buffer_store, inlining combine Stage 1's P2P scatter. The host then
launches a trimmed combine (``combine_no_stage1``) to run Stage 2/3.
"""
from __future__ import annotations

from .tmp_mega_gemm_2stage import compile_mixed_moe_gemm2  # colocated: both GEMMs in one file

__all__ = ["compile_fused_moe_gemm2_combine"]


def compile_fused_moe_gemm2_combine(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    persist_m: int = 4,
    sort_block_m: int = 0,
    b_nt: int = 2,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    rank: int,
    npes: int,
    max_tok_per_rank: int,
    experts_per_token: int = 1,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    xcd_swizzle: int = 0,
    use_token_flag_sync: bool = False,
    doweight_fused: bool = True,
):
    """Compile the fused GEMM2+combine kernel and return its host launcher."""
    # Plan B slot = dest_lid * k + s; reusing baseline shmem_comb_inp_tok
    # (size mr) requires mt*k <= mr, i.e. topk <= npes.
    if topk > npes:
        raise ValueError(
            f"fused GEMM2+combine (Plan B) requires topk <= npes; "
            f"got topk={topk}, npes={npes}."
        )
    if topk != experts_per_token:
        raise ValueError(
            f"topk ({topk}) must equal experts_per_token ({experts_per_token}) "
            "in Plan B; sorted_token_ids' s field is used directly as j_global."
        )
    if model_dim % tile_n != 0:
        raise ValueError(f"model_dim={model_dim} must be divisible by tile_n={tile_n}")
    if inter_dim % tile_k != 0:
        raise ValueError(f"inter_dim={inter_dim} must be divisible by tile_k={tile_k}")
    if (max_tok_per_rank & (max_tok_per_rank - 1)) != 0:
        # Epilogue decodes dest_enc -> (dest_pe, dest_lid) via shift+mask.
        raise ValueError(
            f"max_tok_per_rank={max_tok_per_rank} must be a power of two."
        )

    out_s = str(out_dtype).strip().lower()
    fp8_cast = out_s in (
        "fp8", "fp8e4m3", "fp8e4m3fn", "f8e4m3fn", "float8_e4m3fn"
    )
    if not fp8_cast and out_s not in ("bf16", "bfloat16", "f16", "fp16"):
        raise ValueError(
            f"fused mode output must be bf16/f16 or fp8e4m3fn, got {out_dtype!r}"
        )

    # fused_cfg[3]=enable_weights is pinned False: in-kernel weight P2P
    # scatter (~16B writes) is unreliable under fabric saturation, so weights
    # are handled by the subsequent combine_no_stage1 Stage 1 instead.
    # Under fp8_cast GEMM2 stays bf16 internally; store_pair does the
    # bf16->fp32->cvt_pk_fp8 conversion and writes 1B/elem.
    fused_cfg = (
        int(npes),
        int(rank),
        int(max_tok_per_rank),
        False,
        int(experts_per_token),
        bool(fp8_cast),
    )
    gemm2_out_dtype = "bf16" if fp8_cast else out_dtype

    return compile_mixed_moe_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        persist_m=persist_m,
        sort_block_m=sort_block_m,
        b_nt=b_nt,
        a_dtype=a_dtype, b_dtype=b_dtype, out_dtype=gemm2_out_dtype,
        accumulate=False,
        doweight_stage2=bool(doweight_fused),
        enable_bias=False,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        use_cshuffle_epilog=True,
        xcd_swizzle=xcd_swizzle,
        fused_p2p_scatter=fused_cfg,
        use_token_flag_sync=use_token_flag_sync,
    )
