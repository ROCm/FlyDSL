# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""融合 MoE Stage2 GEMM + EP Combine 算子包装。

封装 :func:`mixed_moe_gemm2_combine_fused.compile_fused_moe_gemm2_combine`
的 host 端调度，并借用 :class:`FlyDSLDispatchCombineIntraNodeOp` 已分配好的
shmem buffer / P2P 地址表 / xdev barrier 等基础设施。

PR1 MVP 阶段
============

* 文件、类、接口、自适应 mode 选择均已就绪
* :data:`READY` = ``False`` 表明 fused kernel 实现尚未接通；测试脚本据此
  graceful skip ``--bench-op fused``，只跑 baseline
* 接通 fused kernel 后将 ``READY`` 改为 ``True`` 即可启用 fused 路径
"""
from __future__ import annotations

import os
from typing import Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

from .dispatch_combine_intranode_op import (
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
)
from .mixed_moe_gemm2_combine_fused import (
    compile_fused_moe_gemm2_combine,
    estimate_max_resident_blocks,
)


__all__ = ["FlyDSLMoeGemm2CombineOp"]


_FORCE_MODE_ENV = "FLYDSL_GEMM2_COMBINE_FORCE_MODE"


class FlyDSLMoeGemm2CombineOp:
    """Fused MoE-GEMM2 + EP-Combine 算子。

    Parameters
    ----------
    cfg
        与已构造的 dispatch op 共用的 :class:`FlyDSLDispatchCombineConfig`。
    disp_op
        已构造好的 :class:`FlyDSLDispatchCombineIntraNodeOp`。本算子不重新
        分配任何 shmem / P2P 表，全部从 ``disp_op`` 借用。
    inter_dim
        GEMM2 输入维度（GEMM1 输出维度）。GEMM2 输出维度 = ``cfg.hidden_dim``。
    tile_m, tile_n, tile_k, persist_m
        GEMM2 tile 调度参数。
    a_dtype, b_dtype
        GEMM2 输入/权重 dtype，与 :func:`compile_mixed_moe_gemm2` 同义。
    force_mode
        ``"auto"``  - 按 occupancy 自适应：能装下就 ``full``，否则 ``stage1_only``。
        ``"full"``     - 强制单 kernel 融合（grid 不够时会失败）。
        ``"stage1_only"``/``"fallback"`` - 强制两 kernel 路径。
        亦可由环境变量 ``FLYDSL_GEMM2_COMBINE_FORCE_MODE`` 覆盖。
    """

    # 切换为 True 后 acceptance 脚本与上层调用方会启用 fused 路径。
    READY: bool = True

    def __init__(
        self,
        *,
        cfg: FlyDSLDispatchCombineConfig,
        disp_op: FlyDSLDispatchCombineIntraNodeOp,
        inter_dim: int,
        tile_m: int = 32, tile_n: int = 128, tile_k: int = 128,
        persist_m: int = 4,
        a_dtype: str = "fp8",
        b_dtype: str = "fp4",
        force_mode: str = "auto",
    ):
        self.cfg       = cfg
        self.disp_op   = disp_op
        self.inter_dim = inter_dim
        self.tile_m    = tile_m
        self.tile_n    = tile_n
        self.tile_k    = tile_k
        self.persist_m = persist_m
        self.a_dtype   = a_dtype
        self.b_dtype   = b_dtype

        env_force = os.environ.get(_FORCE_MODE_ENV, "").strip().lower()
        if env_force:
            force_mode = env_force
        if force_mode in ("fallback", "stage1_only"):
            force_mode = "stage1_only"
        if force_mode not in ("auto", "full", "stage1_only"):
            raise ValueError(
                f"force_mode must be one of auto/full/stage1_only/fallback, "
                f"got {force_mode!r}"
            )
        self.force_mode = force_mode

        out_dtype = "bf16" if cfg.data_type == torch.bfloat16 else "f16"
        self._out_dtype_str = out_dtype

        max_resident = estimate_max_resident_blocks(
            chip=cfg.chip, block_dim=256,
        )
        # GEMM2 自然 grid 估算：(model_dim/tile_n) × ceil(blocks/persist_m)
        # blocks = ceil(max_recv / tile_m) * num_experts_per_rank（aiter sort）
        max_recv = cfg.world_size * cfg.max_num_inp_token_per_rank
        per_e_blocks = (max_recv + tile_m - 1) // tile_m
        nat_blocks = (cfg.hidden_dim // tile_n) * (
            (per_e_blocks * cfg.num_experts_per_rank + persist_m - 1) // persist_m
        )

        if self.force_mode == "auto":
            chosen = "full" if nat_blocks <= max_resident else "stage1_only"
        else:
            chosen = self.force_mode
        self.chosen_mode = chosen

        self._launch_fn = None
        self._compiled = None
        # 预分配所有 dummy tensor 在 ctor 阶段而不是 run 第 1 次调用时 alloc.
        # 这是 cudagraph capture 兼容性要求：capture 内部不能调用 torch.zeros/
        # torch.empty (会触发 hipMalloc → hipErrorInvalidArgument).
        # 上层若在 setup 阶段构造 fused_op 后立即进 cudagraph capture,
        # 第 1 次 chain_fn 就会跑 fused_op.run, 必须 alloc 已经完成.
        _dev = torch.device("cuda", cfg.rank)
        self._dummy_out = torch.zeros(1, dtype=cfg.data_type, device=_dev)
        self._dummy_bias = torch.empty(0, dtype=cfg.data_type, device=_dev)
        # _dummy_inp 用于 _run_stage1_only 的 combine_no_stage1 调用 (skip_stage1
        # 路径下 kernel 不读它, 但需要 valid tensor 占位).
        _mr = cfg.world_size * cfg.max_num_inp_token_per_rank
        self._dummy_inp = torch.zeros(_mr, cfg.hidden_dim,
                                      dtype=cfg.data_type, device=_dev)

    # ── 公开接口 ──────────────────────────────────────────────────────────
    def info(self) -> dict:
        return dict(
            ready=self.READY,
            chosen_mode=self.chosen_mode,
            force_mode=self.force_mode,
            tile=(self.tile_m, self.tile_n, self.tile_k),
            persist_m=self.persist_m,
            a_dtype=self.a_dtype, b_dtype=self.b_dtype,
            out_dtype=self._out_dtype_str,
            inter_dim=self.inter_dim,
        )

    def run(
        self,
        *,
        a2: torch.Tensor,
        w2: torch.Tensor,
        a2_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        sorted_weights: torch.Tensor,
        num_valid_ids: torch.Tensor,
        wts_buf: Optional[torch.Tensor] = None,
        dispatch_total_recv: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        """执行 fused GEMM2+combine。

        参数语义与 :func:`run_moe_stage2` 对齐；额外接收：
          - ``wts_buf``: dispatch 写入到 combine 的 per-token 输入权重 buffer
            （f32[max_recv*k]）。若为 ``None`` 则使用 ``sorted_weights`` 作为
            占位（smoke test 场景）。
          - ``dispatch_total_recv``: dispatch 写入的 ``total_recv`` 标量
            （仅 ``full`` 模式需要；``stage1_only`` 不使用）。

        返回：(out_tok, out_wts) — 与 :py:meth:`FlyDSLDispatchCombineIntraNodeOp.combine`
        相同的 shmem_comb_out_tok / shmem_comb_out_wts 视图。
        """
        if not self.READY:
            raise NotImplementedError(
                "FlyDSLMoeGemm2CombineOp.run(): fused kernel not yet wired. "
                "set FlyDSLMoeGemm2CombineOp.READY = True after kernels/"
                "mixed_moe_gemm2_combine_fused.py:compile_fused_moe_gemm2_combine "
                "is implemented."
            )

        if self.chosen_mode == "full":
            return self._run_full(
                a2, w2, a2_scale, w2_scale,
                sorted_token_ids, sorted_expert_ids, sorted_weights,
                num_valid_ids, bias,
            )
        return self._run_stage1_only(
            a2=a2, w2=w2, a2_scale=a2_scale, w2_scale=w2_scale,
            sorted_token_ids=sorted_token_ids,
            sorted_expert_ids=sorted_expert_ids,
            sorted_weights=sorted_weights,
            num_valid_ids=num_valid_ids,
            wts_buf=wts_buf,
            bias=bias,
            stream=stream,
        )

    # ── 内部：两条路径骨架（READY=True 后才会被调用）──────────────────────
    def _ensure_launch_fn(self):
        if self._launch_fn is not None:
            return
        cfg = self.cfg
        # Debug switch: disable in-kernel weight scatter (P2P write to remote
        # shmem_comb_inp_wts).  Helps isolate token-vs-weight P2P race when
        # the fused chain hangs.  Set FLYDSL_FUSED_DISABLE_WTS=1 to run the
        # fused kernel without touching the weight buffer (combine's stage 3b
        # then reads stale / uninitialized data, only valid for hang triage).
        _en_wts = os.environ.get("FLYDSL_FUSED_DISABLE_WTS", "0") != "1"
        self._launch_fn = compile_fused_moe_gemm2_combine(
            model_dim=cfg.hidden_dim,
            inter_dim=self.inter_dim,
            experts=cfg.num_experts_per_rank,
            topk=1,
            tile_m=self.tile_m, tile_n=self.tile_n, tile_k=self.tile_k,
            persist_m=self.persist_m,
            a_dtype=self.a_dtype, b_dtype=self.b_dtype,
            out_dtype=self._out_dtype_str,
            rank=cfg.rank, npes=cfg.world_size,
            max_tok_per_rank=cfg.max_num_inp_token_per_rank,
            experts_per_token=cfg.num_experts_per_token,
            mode=self.chosen_mode,
            enable_weights=_en_wts,
            enable_std_moe=cfg.enable_std_moe,
            use_p2p_read=not cfg.use_external_inp_buf,
        )

    def _run_stage1_only(self, *, a2, w2, a2_scale, w2_scale,
                         sorted_token_ids, sorted_expert_ids, sorted_weights,
                         num_valid_ids, wts_buf=None, bias=None, stream=None):
        """fused stage1 kernel + 标准 combine(skip_stage1=True)。

        Step A: launch fused GEMM2 kernel (epilogue 直接 P2P scatter 到 remote
                shmem_comb_inp_{tok,wts})
        Step B: launch combine_no_stage1 (Stage 2 xdev barrier + Stage 3 local
                weighted-accum)
        """
        self._ensure_launch_fn()

        cfg     = self.cfg
        disp_op = self.disp_op
        if stream is None:
            stream = torch.cuda.current_stream()
        s_fx = fx.Stream(stream.cuda_stream)

        # GEMM2 形状（与 baseline run_moe_stage2 对齐）
        # NOTE: a2 在测试脚本中以 1D view 传入（a2_storage = a2_view.view(-1)），
        # 所以 a2.shape[0] = max_recv * inter_dim 是 *元素总数*，不是 token 数。
        # baseline 路径明确把 gemm2_in["max_recv"] 作为 tokens 传入，这里也用
        # cfg.world_size * cfg.max_num_inp_token_per_rank 计算 max_recv，与
        # _build_gemm2_static_inputs 完全一致。否则会导致：
        #   - kernel 把 tokens_in 当成 max_recv*inter_dim（巨大），
        #   - row_valid 早退断言全部失效（t_ok 永远 true），
        #   - sorted_token_ids 中的 padding 哨兵 (= max_recv) 被当成有效 row，
        #   - epilogue 用 t=max_recv 越界 buffer_load addr_tis → garbage dest_pe,
        #   - P2P scatter 飞到任意远程地址，污染 combine 的 control state，
        #   - combine 在 CrossDeviceBarrier 阶段 wait 永久挂死。
        n_in            = cfg.hidden_dim
        k_in            = self.inter_dim
        size_expert_ids = sorted_expert_ids.numel()
        tokens_in       = cfg.world_size * cfg.max_num_inp_token_per_rank

        # P2P 地址表 / shmem 入口（来自 disp_op）
        addr_tis              = disp_op._fx_tis
        addr_p2p_comb_inp     = disp_op._fx_p2p_comb_inp
        addr_p2p_comb_inp_wts = disp_op._fx_p2p_comb_inp_wts
        # weight 源：fused epilogue 按 (tok_i*k + s) f32 索引读，与 baseline
        # combine stage1 完全一致（baseline 用的是 `addr_disp_out_wts`）。
        # 默认走 `disp_op.shmem_disp_out_wts`（dispatch 写入的 per-token×k
        # 权重，shape = f32[max_recv*k]）；若上层显式传 ``wts_buf`` 则用之，
        # 但其布局必须满足 `wts[t*k + s]` 的语义。
        if wts_buf is not None:
            addr_wts_buf = fx.Int64(wts_buf.data_ptr())
        else:
            addr_wts_buf = disp_op._fx_disp_out_wts

        # arg_out 在 fused 路径下不被读写，但仍需是有效 tensor。
        if not hasattr(self, "_dummy_out") or self._dummy_out is None:
            self._dummy_out = torch.zeros(1, dtype=cfg.data_type, device=a2.device)
        arg_out = self._dummy_out

        # 维持 bias dummy tensor 的生命周期（避免每次 run 都 alloc/free 与 cudagraph 冲突）
        if bias is None:
            if not hasattr(self, "_dummy_bias") or self._dummy_bias is None:
                self._dummy_bias = torch.empty(0, dtype=cfg.data_type, device=a2.device)
            bias_t = self._dummy_bias
        else:
            bias_t = bias

        if self._compiled is None:
            args = (
                arg_out, a2, w2, a2_scale, w2_scale,
                sorted_token_ids, sorted_expert_ids, sorted_weights,
                num_valid_ids, bias_t,
                addr_tis, addr_p2p_comb_inp, addr_wts_buf, addr_p2p_comb_inp_wts,
                fx.Int32(tokens_in), fx.Int32(n_in), fx.Int32(k_in),
                fx.Int32(size_expert_ids),
                s_fx,
            )
            self._compiled = flyc.compile(self._launch_fn, *args)
        else:
            self._compiled(
                arg_out, a2, w2, a2_scale, w2_scale,
                sorted_token_ids, sorted_expert_ids, sorted_weights,
                num_valid_ids, bias_t,
                addr_tis, addr_p2p_comb_inp, addr_wts_buf, addr_p2p_comb_inp_wts,
                tokens_in, n_in, k_in, size_expert_ids,
                s_fx,
            )

        # Debug switch: 跳过 stage 2/3，便于隔离 GEMM2-P2P-scatter 自身的问题。
        if os.environ.get("FLYDSL_FUSED_SKIP_COMBINE", "0") == "1":
            return None, None

        # Stage 2/3：调用 disp_op.combine_no_stage1
        # `input`/`weights`/`indices` 在 skip_stage1=True 路径下不会被 kernel
        # 读取（Stage 1 整段在 Python 层裁掉），传入 dummy tensor 即可。
        if not hasattr(self, "_dummy_inp") or self._dummy_inp is None:
            mr = cfg.world_size * cfg.max_num_inp_token_per_rank
            hd = cfg.hidden_dim
            self._dummy_inp = torch.zeros(mr, hd, dtype=cfg.data_type, device=a2.device)

        # Debug switch: 用 baseline combine（带 stage1）替代 combine_no_stage1，
        # 验证 skip_stage1 路径是否是 hang 元凶。
        if os.environ.get("FLYDSL_FUSED_USE_FULL_COMBINE", "0") == "1":
            out_tok, out_wts = disp_op.combine(self._dummy_inp, None, None)
        else:
            out_tok, out_wts = disp_op.combine_no_stage1(
                self._dummy_inp, None, None
            )
        return out_tok, out_wts

    def _run_full(self, a2, w2, a2_scale, w2_scale,
                  sti, eids, sw, num_valid_ids, bias):
        """单 kernel 端到端 GEMM2 + Stage1/2/3。"""
        raise NotImplementedError("_run_full: pending kernel implementation")
