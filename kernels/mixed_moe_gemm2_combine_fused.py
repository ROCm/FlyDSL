# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""融合 MoE Stage2 GEMM + EP Combine kernel。

设计目标
========

本文件提供 ``compile_fused_moe_gemm2_combine``，将 ``mixed_moe_gemm_2stage.py``
中的 GEMM2 epilogue 与 ``dispatch_combine_intranode_kernel.py`` 中的 EP combine
进行 epilogue 级别的源融合：

  * 把 GEMM2 的 LDS CShuffle epilogue 中的 **本地 store_pair** 改写为
    **远端 P2P buffer_store**，实现 Stage 1 (P2P scatter) 的 in-kernel 内联；
  * 配合 :func:`dispatch_combine_intranode_kernel.make_combine_jit`
    新增的 ``skip_stage1=True`` flag，外部 combine kernel 只跑 Stage 2 / 3。

两种实现变体
============

``mode='stage1_only'`` （PR1 MVP）
    只把 Stage 1 inline 进 GEMM2 epilogue。GEMM2 grid 尺寸不变（没有 grid-wide
    barrier 的复杂性），随后 host 端再 launch 一次裁掉 Stage 1 的 combine kernel
    （走 ``op.combine_no_stage1``）完成 Stage 2/3。

    优点：
      - 实现简单（只动 epilogue 的 store_pair）
      - 不需要 grid persistent；GEMM2 块数完全不受 ``max_resident_blocks`` 约束

    缺点：
      - 仍是两次 launch，省下的是 Stage 1 的全局读 + LDS scatter 开销

``mode='full'`` （PR1 后期或 PR2）
    把 Stage 2 (CrossDeviceBarrier) + Stage 3 (本地 weighted-accum) 一并 inline
    到 GEMM2 kernel 末尾，单 kernel 端到端覆盖 GEMM2 + combine 全流程。

    要求：
      - GEMM2 grid 必须能完整 resident（grid-wide barrier 必须所有块同时 alive）
      - Wrapper 用 ``estimate_max_resident_blocks`` 在 host 端做 occupancy check，
        若不满足则强制走 ``stage1_only`` fallback。

当前文件状态
============

PR1 MVP 阶段：
  - **接口、签名、文件结构已就绪**
  - **kernel body 尚未实现**：``compile_fused_moe_gemm2_combine`` 暂时
    抛 :class:`NotImplementedError` 并附 TODO 列表
  - 配套的 ``mixed_moe_gemm2_combine_fused_op.py`` 包含 ``READY = False`` 标志，
    测试脚本可据此 graceful skip fused 路径（仅跑 baseline）

后续工作（按依赖顺序）：

1. epilogue 改写
   ----------------
   把 :func:`mixed_moe_gemm_2stage.compile_mixed_moe_gemm2` 中
   ``c_shuffle_epilog`` 的 ``store_pair`` 替换为：

       tis_val   = buffer_load(rsrc_tis, t)              # i32, dispatch 写入
       dest_pe   = tis_val >> log2_max_tok               # 高位
       dest_lid  = tis_val & mask_max_tok                # 低位
       p2p_base  = sgpr_p2p_table[dest_pe]               # SGPR scalar select
       dst_off   = (rank * max_tok + dest_lid) * model_dim_bytes + col_offset
       buffer_store(frag, create_buffer_resource_from_addr(p2p_base + dst_off), 0)

   关键是 ``s=0`` (topk 折叠为 1)，且 P2P 写入用 plain ``buffer_store``
   （非 atomic），依赖 dispatch 端去重保证不冲突。

2. SGPR P2P 表
   ----------------
   prologue 把 ``addr_p2p_comb_inp`` 的 npes 个 i64 base 一次性 buffer_load
   并立即 ``readfirstlane`` 进 SGPR；epilogue 用 ``v_cndmask`` 链按 ``dest_pe``
   选 base，避免 LDS p2p_bases 表的 wave-uniform broadcast 开销。

3. tile 调度对齐
   ----------------
   ``persist_m`` 沿 M 持久化要保证：同一持久化 chain 上 tile_m 起点对应
   连续 token，否则 epilogue 内的 ``tis_val`` 反查需要 LDS prefetch table。
   推荐 ``persist_m=4`` 与已有 GEMM2 测试一致。

4. full 变体
   ----------------
   GEMM2 主体 + barrier_acquire（atomic_add comb_bar，spin-wait equals
   block_num）+ Stage 3 weighted-accum，复用 SmemAllocator 的 LDS 区域
   （Stage 3 的 reduce buffer 与 GEMM2 的 LDS A/B buffer 时间上不重叠，
   可以共享）。需要分配一个 grid-wide barrier scratch（int32[1] 全 0 init）。
"""
from __future__ import annotations

import torch

from .mixed_moe_gemm_2stage import compile_mixed_moe_gemm2

__all__ = ["compile_fused_moe_gemm2_combine"]


def compile_fused_moe_gemm2_combine(
    *,
    # ── GEMM2 形状 / 调度 ──
    model_dim: int,
    inter_dim: int,
    experts: int,                 # = num_experts_per_rank（EP 场景）
    topk: int,                    # = 1（EP 场景下 reinterpret）
    tile_m: int,
    tile_n: int,
    tile_k: int,
    persist_m: int = 4,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",      # 必须等于 combine 输入 dtype
    # ── EP / combine 拓扑 ──
    rank: int,
    npes: int,
    max_tok_per_rank: int,
    experts_per_token: int = 1,   # = dispatch 路由 k；决定 weight scatter 槽宽
    # ── 融合模式 ──
    mode: str = "stage1_only",    # 'stage1_only' | 'full'
    enable_weights: bool = True,
    enable_std_moe: bool = False,
    use_p2p_read: bool = False,   # 仅 'full' mode 用得上
    # ── 兼容选项（与 compile_mixed_moe_gemm2 对齐）──
    enable_bias: bool = False,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
):
    """编译融合 GEMM2+combine kernel。

    返回值与 :func:`compile_mixed_moe_gemm2` 同型：一个 ``@flyc.jit`` 装饰的
    host launcher（``launch_fused_moe_gemm2_combine``），签名在
    ``mode='stage1_only'`` 与 ``mode='full'`` 下略有差异——

    stage1_only::

        launch(
            arg_x, arg_w, arg_scale_x, arg_scale_w,
            arg_sorted_token_ids, arg_expert_ids,
            arg_sorted_weights, arg_num_valid_ids, arg_bias,
            # P2P / shmem
            addr_tis,                  # tok_id_to_src (i32[max_recv])
            addr_p2p_comb_inp,         # i64[npes]
            # （可选）权重 P2P
            addr_wts_buf,              # f32[max_recv * k]，combine 输入权重
            addr_p2p_comb_inp_wts,     # i64[npes]
            i32_tokens, i32_n, i32_k, i32_size_expert_ids,
            stream,
        )

    full::

        与 stage1_only 一致，再额外要：
            addr_xdb_mem, addr_xdb_flag, addr_p2p_xdb_mem,  # CrossDeviceBarrier
            addr_comb_bar,                                  # grid-wide barrier
            addr_comb_inp_local,                            # 本地 shmem_comb_inp
            addr_comb_out, addr_tok_map, addr_trecv,        # Stage 3 输入
            addr_comb_inp_wts_local, addr_comb_out_wts,     # 权重 Stage 3
            cur_rank_num_token,                             # = max_tok_per_rank

    .. note::

       PR1 MVP 阶段，本函数 **未实现**。调用时直接抛
       :class:`NotImplementedError`，并打印实现 checklist；
       配套的 :class:`FlyDSLMoeGemm2CombineOp` 通过 ``READY = False`` flag
       告知验收脚本 graceful skip。
    """
    if mode not in ("stage1_only", "full"):
        raise ValueError(f"mode must be 'stage1_only' or 'full', got {mode!r}")

    # ── 入参合法性预检（kernel body 实现前先把所有错误暴露在编译期）──
    if topk != 1:
        raise ValueError(
            f"fused GEMM2+combine 仅支持 topk=1（EP 场景把 received tokens 视作 "
            f"per-token 单 expert），got topk={topk}. enable_std_moe 多 expert "
            f"折叠分支留待 PR3。"
        )
    if model_dim % tile_n != 0:
        raise ValueError(f"model_dim={model_dim} must be divisible by tile_n={tile_n}")
    if inter_dim % tile_k != 0:
        raise ValueError(f"inter_dim={inter_dim} must be divisible by tile_k={tile_k}")
    if (max_tok_per_rank & (max_tok_per_rank - 1)) != 0:
        # epilogue 用位运算分解 dest_enc -> (dest_pe, dest_lid)，要求是 2 的幂
        raise ValueError(
            f"max_tok_per_rank={max_tok_per_rank} must be a power of two "
            f"(epilogue uses shift+mask to decode tis encoding)."
        )

    out_s = str(out_dtype).strip().lower()
    if out_s not in ("bf16", "bfloat16", "f16", "fp16"):
        raise ValueError(
            f"fused 模式输出必须是 combine 可读的浮点类型 (bf16/f16), got {out_dtype!r}"
        )

    if mode == "full":
        # PR1 MVP 仅实现 stage1_only：把 Stage 1 P2P scatter 内联进 GEMM2
        # epilogue。Stage 2/3 留给 wrapper 串一次 op.combine_no_stage1。
        # full 路径需要 grid-persistent + cross-device barrier inline，留待 PR2。
        raise NotImplementedError(
            "fused mode='full' not yet implemented: requires grid-persistent "
            "GEMM2 + inline xdev barrier + Stage 3 weighted-accum. "
            "Use mode='stage1_only' (default) for the PR1 MVP."
        )

    # ── stage1_only：复用 compile_mixed_moe_gemm2 + fused_p2p_scatter ──
    # 必须是 hashable（compile_mixed_moe_gemm2 走 lru_cache）。
    # 5-tuple = (npes, rank, max_tok_per_rank, enable_weights, experts_per_token)
    # experts_per_token 是 dispatch 路由 k；fused epilogue 用它决定每个
    # (pe, lid) 槽位写多少个 f32 weight（与 baseline combine stage1 完全一致）。
    #
    # NOTE: 强制把 fused 路径的 enable_weights 设为 False。当 fused_gemm2 的
    # token P2P 写入很重时，把 ~16-byte 的小权重写挂在同一个 kernel 末尾会
    # 因为 fabric 饱和而出现写丢失（即便加了 s_waitcnt(0) + gpu.barrier，
    # 这两者只能同步本 workgroup，无法 grid-wide 等待其它块的 token 写排空）。
    # 将权重 P2P scatter 延后到 combine_no_stage1 的轻量 Stage 1 里去做
    # （make_combine_jit(skip_stage1=True, skip_stage1_keep_wts=True)），
    # 跑在干净 fabric 上更可靠。
    fused_cfg = (
        int(npes),
        int(rank),
        int(max_tok_per_rank),
        False,
        int(experts_per_token),
    )

    return compile_mixed_moe_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        persist_m=persist_m,
        a_dtype=a_dtype, b_dtype=b_dtype, out_dtype=out_dtype,
        # fused 路径强制：accumulate=False, doweight_stage2=False, bias=False
        accumulate=False,
        doweight_stage2=False,
        enable_bias=False,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        use_cshuffle_epilog=True,
        fused_p2p_scatter=fused_cfg,
    )


def estimate_max_resident_blocks(
    *,
    chip: str = "gfx950",
    block_dim: int = 256,            # = 4 warps * 64 lanes
    lds_bytes_per_block: int = 0,
    vgpr_per_thread: int = 96,
):
    """粗略估算 chip 上 ``block_dim`` × LDS/VGPR 配置下 per-CU 最大常驻 block 数，
    再乘以 CU 数得到 grid-wide 上限。

    用法：fused op wrapper 在 ``mode='full'`` 路径前先做 occupancy check，
    若 GEMM2 自然 grid > 估算上限则 fallback 到 ``stage1_only``。

    .. note::

       PR1 阶段是占位实现：MI355X 大约 256 CU、每 CU 4 active blocks（256t/blk
       占满 1024-thread CU 配额）→ ~1024 blocks，对绝大多数 GEMM2 形状都够用。
       下一阶段会接通 ROCm device 属性查询替换硬编码常量。
    """
    cu_count = {
        "gfx950": 256,
        "gfx942": 304,
        "gfx12":  64,
    }.get(chip.lower(), 192)
    blocks_per_cu = max(1, 1024 // max(1, block_dim))
    return cu_count * blocks_per_cu


# 把 baseline 路径暴露出来，供 op wrapper 在 fused kernel 未就绪时透传。
def compile_baseline_gemm2(*args, **kwargs):
    """直接转发 :func:`mixed_moe_gemm_2stage.compile_mixed_moe_gemm2`。

    op wrapper 在 ``READY = False`` 阶段会通过本函数走 baseline GEMM2 路径，
    并搭配未裁剪的 combine 使用，等价于 ``--bench-op baseline`` 链路。
    """
    return compile_mixed_moe_gemm2(*args, **kwargs)
