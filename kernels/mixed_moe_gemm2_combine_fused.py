# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused MoE Stage2 GEMM + EP combine kernel.

Rewrites the GEMM2 epilogue's local store_pair into a remote P2P
buffer_store, inlining combine Stage 1's P2P scatter.

Modes:
  - ``stage1_only``: inline Stage 1 only; the host then launches a
    trimmed combine to run Stage 2/3. Grid persistence not required.
  - ``full``: inline Stage 2 (xdev barrier) + Stage 3 (weighted accum)
    as well; requires the GEMM2 grid to be fully resident (PR2).
"""
from __future__ import annotations

from .mixed_moe_gemm_2stage import compile_mixed_moe_gemm2

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
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    rank: int,
    npes: int,
    max_tok_per_rank: int,
    experts_per_token: int = 1,
    mode: str = "stage1_only",
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    xcd_swizzle: int = 0,
    use_token_flag_sync: bool = False,
):
    """Compile the fused GEMM2+combine kernel and return its host launcher."""
    if mode == "full":
        raise NotImplementedError(
            "fused mode='full' not yet implemented (PR2). "
            "Use mode='stage1_only' for the PR1 MVP."
        )
    if mode != "stage1_only":
        raise ValueError(f"mode must be 'stage1_only' or 'full', got {mode!r}")

    # Plan B (slot = dest_lid * k + s) requires mt*k <= mr to reuse the
    # baseline shmem_comb_inp_tok (size = mr), which is equivalent to k <= npes.
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

    # In-kernel weight P2P scatter is unreliable (~16B writes drop under
    # fabric saturation, and s_waitcnt+gpu.barrier cannot grid-wide wait
    # for other blocks' token writes to drain), so weights are always
    # handled by the subsequent lightweight combine_no_stage1 Stage 1.
    # fused_cfg[3] = enable_weights is therefore pinned to False.
    #
    # With fp8_cast=True the GEMM2 still emits bf16 internally (preserves
    # the c_shuffle LDS/frag dtype); the store_pair site does the bf16 ->
    # fp32 -> cvt_pk_fp8 conversion and writes 1B/elem out.
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
        a_dtype=a_dtype, b_dtype=b_dtype, out_dtype=gemm2_out_dtype,
        accumulate=False,
        doweight_stage2=False,
        enable_bias=False,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        use_cshuffle_epilog=True,
        xcd_swizzle=xcd_swizzle,
        fused_p2p_scatter=fused_cfg,
        use_token_flag_sync=use_token_flag_sync,
    )


def estimate_max_resident_blocks(
    *,
    chip: str = "gfx950",
    block_dim: int = 256,
    lds_bytes_per_block: int = 0,
    vgpr_per_thread: int = 96,
):
    """Estimate grid-wide max resident block count for ``mode='full'``
    occupancy checks.

    PR1 placeholder: hard-coded CU count + 1024 thread/CU budget; will be
    wired to live ROCm device-property queries later.
    """
    cu_count = {
        "gfx950": 256,
        "gfx942": 304,
        "gfx12":  64,
    }.get(chip.lower(), 192)
    blocks_per_cu = max(1, 1024 // max(1, block_dim))
    return cu_count * blocks_per_cu
