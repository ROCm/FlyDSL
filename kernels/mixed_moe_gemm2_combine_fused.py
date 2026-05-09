# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""融合 MoE Stage2 GEMM + EP Combine kernel。

将 GEMM2 epilogue 中的本地 store_pair 改写为远端 P2P buffer_store，
内联 combine Stage 1 的 P2P scatter。

模式：
  - ``stage1_only``: 仅内联 Stage 1，host 端再 launch 裁剪后的 combine
    跑 Stage 2/3。不需要 grid persistent。
  - ``full``: Stage 2 (xdev barrier) + Stage 3 (weighted-accum) 一并内联，
    要求 GEMM2 grid 完整 resident（PR2）。
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
):
    """编译融合 GEMM2+combine kernel，返回 host launcher。"""
    if mode == "full":
        raise NotImplementedError(
            "fused mode='full' not yet implemented (PR2). "
            "Use mode='stage1_only' for the PR1 MVP."
        )
    if mode != "stage1_only":
        raise ValueError(f"mode must be 'stage1_only' or 'full', got {mode!r}")

    if topk != 1:
        raise ValueError(f"fused GEMM2+combine 仅支持 topk=1，got topk={topk}.")
    if model_dim % tile_n != 0:
        raise ValueError(f"model_dim={model_dim} must be divisible by tile_n={tile_n}")
    if inter_dim % tile_k != 0:
        raise ValueError(f"inter_dim={inter_dim} must be divisible by tile_k={tile_k}")
    if (max_tok_per_rank & (max_tok_per_rank - 1)) != 0:
        # epilogue 用 shift+mask 解码 dest_enc -> (dest_pe, dest_lid)
        raise ValueError(
            f"max_tok_per_rank={max_tok_per_rank} must be a power of two."
        )

    out_s = str(out_dtype).strip().lower()
    fp8_cast = out_s in (
        "fp8", "fp8e4m3", "fp8e4m3fn", "f8e4m3fn", "float8_e4m3fn"
    )
    if not fp8_cast and out_s not in ("bf16", "bfloat16", "f16", "fp16"):
        raise ValueError(
            f"fused 模式输出必须是 bf16/f16 或 fp8e4m3fn, got {out_dtype!r}"
        )

    # in-kernel 权重 P2P scatter 不可靠（fabric 饱和时 ~16B 小写会丢，
    # s_waitcnt+gpu.barrier 无法 grid-wide 等待其它块的 token 写排空），
    # 一律由后续的 combine_no_stage1 轻量 Stage 1 处理。fused_cfg[3] 即
    # enable_weights，固定 False。
    #
    # fp8_cast=True 时 GEMM2 内部仍输出 bf16（保持 c_shuffle LDS/frag dtype），
    # store_pair 处再做 bf16->fp32->cvt_pk_fp8 转换并按 1B/elem 写出。
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
    )


def estimate_max_resident_blocks(
    *,
    chip: str = "gfx950",
    block_dim: int = 256,
    lds_bytes_per_block: int = 0,
    vgpr_per_thread: int = 96,
):
    """估算 grid-wide 最大常驻 block 数，供 ``mode='full'`` 做 occupancy check。

    PR1 占位实现：硬编码 CU 数 + 1024 thread/CU 配额；后续接 ROCm 属性查询。
    """
    cu_count = {
        "gfx950": 256,
        "gfx942": 304,
        "gfx12":  64,
    }.get(chip.lower(), 192)
    blocks_per_cu = max(1, 1024 // max(1, block_dim))
    return cu_count * blocks_per_cu
