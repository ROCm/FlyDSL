# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""端到端节点内 EP-MoE 单算子：megav1 stage-1（单发 dispatch⊕GEMM1，atom 契约）
+ stage-2（GEMM2 + EP combine），对外一个干净 ``forward(x_bf16, wts, topk_ids)``。

封装的就是 ``tests/kernels/bench_moe_intranode_stage1_groupgemm.py::_run_stage2_e2e``
里已验证（精度通过）的 **MEGA-ATOM NATIVE combine** 接线，不引入任何新机制：

stage-1（``FusedMoEMegaStage1``，一次 megakernel launch，恒输出 atom 契约 a2）产出
atom 契约：``_out``（a2，value@logical ``t*topk+s``）、``_osd``（a2-scale@sorted）、
``_sti``（sorted_token_ids，编码 src_global=dest）、``_se_atom``（sorted_expert_ids）、
``_sw_atom``（combine 权重，logical 布局）。

stage-2 复用一个 ``FlyDSLDispatchCombineIntraNodeOp``（combine 基础设施：P2P comb
表 / xdev barrier / shmem_comb buffers / total_recv / tok_id_to_src），两种生产模式
**对外 I/O 完全一致**（同一份 a2/scale/sorted 表输入，同一份 combine 输出）：

  * ``stage2_mode="fused"``     : ``compile_fused_moe_gemm2_combine``
    （GEMM2 epilogue 内联 Stage-1 P2P scatter + combine Stage-2/3）。**已验证。**
  * ``stage2_mode="nonfused"``  : ``compile_mixed_moe_gemm2``（accumulate）+
    ``comb_op.combine``（Stage-1/2/3）。底层是同一个 GEMM2 kernel（融合版只多
    ``fused_p2p_scatter``）。

输出
----
combine 输出 dtype = ``comb_op`` 的 ``data_type``（本模块固定 **bf16**，给下一层
直接消费）。``forward`` 返回 ``[run_tokens, model_dim]`` bf16。

zero-bridge 说明
----------------
``forward`` 内只有两段 kernel（stage-1 megakernel + stage-2 GEMM2+combine），无冗余
dispatch、无 host 数据搬运。两个 combine 入参的零桥接来源：

  * ``total_recv``（combine scatter-back 的 distinct-recv 计数）由 stage-1 megakernel
    **核内直接写进 combine op 的 buffer**（Plan A：构造期把 ``comb_op.total_recv``
    作为 ``total_recv_buf`` 传给 stage-1，megakernel 每次 launch 原地写并自清零）。
  * ``tok_id_to_src``（scatter-back 映射）= identity（``arange``），因 megav1 的
    ``_sti`` 已编码 ``src_global = dest_token``；构造期设一次，``forward`` 内不碰。
"""
from __future__ import annotations

import os
from typing import Optional

import torch

import flydsl.expr as fx
import mori.shmem as ms

from .tmp_mega_megakernel import FusedMoEMegaStage1
from .dispatch_combine_intranode_op import (
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
)

try:
    from aiter import dtypes as _adt
    from aiter.ops.quant import per_1x32_mx_quant_hip
    _HAS_AITER_QUANT = True
except Exception:  # noqa: BLE001
    _adt = None
    per_1x32_mx_quant_hip = None
    _HAS_AITER_QUANT = False


__all__ = ["MegaMoEExp"]


class MegaMoEExp:
    """端到端 EP-MoE 单算子（megav1 stage-1 ⊕ stage-2 GEMM2+combine）。

    Parameters
    ----------
    rank, world_size, model_dim, inter_dim, experts, topk
        MoE / EP 维度。``experts`` 须被 ``world_size`` 整除。
    quant
        ``"a8w4"``（激活 fp8 / 权重 fp4）或 ``"a4w4"``（均 fp4）。
    w1, w1_scale
        gate/up 权重（fp4 packed）+ e8m0 scale，megav1 GEMM1 直接消费。**仅本 rank 的
        ``experts // world_size`` 个本地专家**（ATOM/EP 约定，GEMM1 按 local 专家号索引）；
        不接受全局 ``experts`` 张量——既为省显存，也因 >4GB 权重会触发 32-bit buffer
        ``num_records`` 截断。
    w2, w2_scale
        down 权重（fp4 packed, ``shuffle_weight`` 后 1-D uint8）+ e8m0 scale，stage-2 GEMM2
        直接消费。同样**仅本 rank 的本地专家**。
    max_tok_per_rank
        每 rank 最大输入 token（dispatch/combine 缓冲上界）；须为 2 的幂（融合
        epilogue 用 shift/mask 解码 dest_enc）。
    stage2_mode
        ``"fused"``（默认，已验证）或 ``"nonfused"``。
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        model_dim: int,
        inter_dim: int,
        experts: int,
        topk: int,
        quant: str,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        max_tok_per_rank: int,
        tune_tokens: Optional[int] = None,
        network: Optional[str] = None,
        mega_scheme: str = "fixedslot",
        tile_m: int = -1,
        tile_n: int = -1,
        tile_k: int = 256,
        gemm2_tile_m: int = -1,
        gemm2_tile_n: int = -1,
        gemm2_tile_k: int = -1,
        gemm2_persist_m: int = -1,
        warp_num_per_block: int = 4,
        waves_per_eu: int = 4,
        use_async_copy: bool = True,
        block_num: Optional[int] = None,
        stage2_mode: str = "fused",
        xcd_swizzle: int = 0,
    ):
        assert quant in ("a8w4", "a4w4"), quant
        assert stage2_mode in ("fused", "nonfused"), stage2_mode
        if not _HAS_AITER_QUANT:
            raise RuntimeError("MegaMoE needs aiter (per_1x32_mx_quant_hip)")
        if (max_tok_per_rank & (max_tok_per_rank - 1)) != 0:
            raise ValueError(f"max_tok_per_rank={max_tok_per_rank} must be a power of two")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.model_dim = int(model_dim)
        self.inter_dim = int(inter_dim)
        self.experts = int(experts)
        self.epr = experts // world_size
        self.topk = int(topk)
        self.quant = quant
        self.mtpr = int(max_tok_per_rank)
        self.stage2_mode = stage2_mode
        self.dev = torch.device("cuda", rank)
        self._is_fp4 = (quant == "a4w4")
        self._qd = _adt.fp4x2 if self._is_fp4 else _adt.fp8
        self._stp = None if self._is_fp4 else _adt.fp8_e8m0

        # ---- stage-2 combine 基础设施 op（bf16 combine 输出）；先建,以便 stage-1 megakernel
        # 直接把 total_recv 写进它(零拷贝桥接) ----
        try:
            _cu = int(torch.cuda.get_device_properties(self.dev).multi_processor_count)
        except Exception:  # noqa: BLE001
            _cu = 256
        bn = block_num if block_num is not None else (
            min(_cu, 64) if self.mtpr <= 32 else (min(_cu, 128) if self.mtpr <= 128 else _cu))
        # NOTE: the legacy shared ``warp_num_per_block``/``block_num`` config
        # fields were removed upstream (PR #712); launch geometry now comes
        # from the per-phase ``dispatch_*``/``combine_*`` defaults plus the
        # auto-loaded tuning table. ``bn``/``warp_num_per_block`` were already
        # inert here (combine used its per-phase defaults), so they are no
        # longer forwarded; pin ``combine_block_num=bn`` below if an explicit
        # CU-aware combine geometry is desired.
        self.comb_cfg = FlyDSLDispatchCombineConfig(
            rank=rank, world_size=world_size, hidden_dim=model_dim,
            max_num_inp_token_per_rank=self.mtpr, num_experts_per_rank=self.epr,
            num_experts_per_token=topk, data_type=torch.bfloat16,
            scale_dim=0, scale_type_size=0, enable_std_moe=False,
        )
        self.comb_op = FlyDSLDispatchCombineIntraNodeOp(self.comb_cfg)
        torch.cuda.synchronize(); ms.shmem_barrier_all()

        if stage2_mode != "fused":
            # 非融合（compile_mixed_moe_gemm2 + comb_op.combine）I/O 与融合等价（已确认：
            # 同一个 GEMM2 kernel，融合版只多 fused_p2p_scatter），但其精确接线尚未在 megav1
            # 上跑过验证。为避免把未验证代码塞进生产路径，暂不实现。
            raise NotImplementedError(
                "stage2_mode='nonfused' pending wiring+validation on megav1; "
                "use stage2_mode='fused' (validated MEGA-ATOM NATIVE combine recipe)."
            )

        self.w2 = w2 if w2.is_contiguous() else w2.contiguous()
        self.w2_scale = w2_scale if w2_scale.is_contiguous() else w2_scale.contiguous()
        self.max_recv = world_size * self.mtpr

        # megav1 _sti 编码 src_global=dest -> combine tok_id_to_src 恒为 identity(常量)；
        # 只在构造时设一次(combine 只读不写、megav1 不碰它)-> forward 里无需每步重设。
        self.comb_op.shmem_tok_id_to_src.copy_(
            torch.arange(self.max_recv, device=self.dev, dtype=torch.int32))
        torch.cuda.synchronize(); ms.shmem_barrier_all()

        # ---- fused GEMM2 ghost-gate 边界(mega 自闭环) ----
        # GEMM2 epilogue 的 ghost gate 用 `t < total_recv` 丢 padding 行；megav1 Plan-A 的 _sti
        # 编码 t = src_global ∈ [0, world*mtpr)，padding sentinel == max_recv。把 gate 句柄
        # _fx_out_total_recv 重指到常量 max_recv：gate 变 `t < max_recv`(真实行全留、padding 丢)；
        # combine_no_stage1 读真实 total_recv 走另一句柄(_fx_trecv，同 buffer)。MUST 在 stage1 build
        # 之前设好，因为 stage1.forward 把 comb_op._fx_out_total_recv 作为 GEMM2 ghost-gate 地址传入。
        self._gemm2_gate_bound = torch.tensor(
            [self.max_recv], dtype=torch.int32, device=self.dev)
        self.comb_op._fx_out_total_recv = fx.Int64(self._gemm2_gate_bound.data_ptr())
        torch.cuda.synchronize(); ms.shmem_barrier_all()

        # combine_no_stage1 的占位输入 (megav1 不读它；GEMM2 epilogue P2P 直写 shmem_comb_inp)。
        # ctor 期分配一次 (cudagraph-safe)。
        self._dummy_inp = torch.zeros(
            self.max_recv, model_dim, dtype=self.comb_cfg.data_type, device=self.dev)

        # ---- stage-1 + merged GEMM2: 单发 megakernel (GEMM1+silu ⊕ GEMM2+combine-Stage1) ----
        # GEMM2 在 stage1 的同一次 launch 内作为第二个 persistent phase 运行 (l2_ready 流水重叠)。
        # stage1's sort_block_m == max(32, unit_size); gemm2 tile_m 须整除它 (每个 gemm2 tile 落在
        # 单个专家的 sort_block_m-padded 行内)。tune miss (gemm2_tile_m<=0) 时退回 32。
        _eff_g2_tm = int(gemm2_tile_m) if int(gemm2_tile_m) > 0 else 32
        _eff_g2_tn = int(gemm2_tile_n) if int(gemm2_tile_n) > 0 else 128
        _eff_g2_tk = int(gemm2_tile_k) if int(gemm2_tile_k) > 0 else 256
        _eff_g2_pm = int(gemm2_persist_m) if int(gemm2_persist_m) > 0 else 4
        self.stage1 = FusedMoEMegaStage1(
            rank=rank, world_size=world_size, model_dim=model_dim, inter_dim=inter_dim,
            experts=experts, topk=topk, quant=quant, w1=w1, w1_scale=w1_scale,
            max_tok_per_rank=max_tok_per_rank, tune_tokens=tune_tokens,
            network=network, scheme=mega_scheme,
            unit_size=int(tile_m), tile_n=int(tile_n), tile_k=int(tile_k),
            warp_num_per_block=int(warp_num_per_block), waves_per_eu=int(waves_per_eu),
            use_async_copy=bool(use_async_copy), out_dtype="auto",
            total_recv_buf=self.comb_op.total_recv,
            fuse_gemm2=True, w2=self.w2, w2_scale=self.w2_scale, comb_op=self.comb_op,
            gemm2_tile_m=_eff_g2_tm, gemm2_tile_n=_eff_g2_tn, gemm2_tile_k=_eff_g2_tk,
            gemm2_persist_m=_eff_g2_pm, gemm2_xcd_swizzle=int(xcd_swizzle),
            gemm2_doweight=True,
        )
        self.a2_dtype = self.stage1.a_dtype   # "fp8" | "fp4"  (== stage-1 out_dtype 'auto')
        self._g2 = None   # merged into stage1; no separate GEMM2 op

    def quantize(self, x_bf16):
        """便捷量化：bf16 激活 -> (fp8/fp4 payload, e8m0 scale uint8)。

        生产里激活量化是 stage-1 的外部 kernel；``forward`` 直接收量化结果。
        这里提供与生产一致的量化以便上层/测试调用。
        """
        if self._is_fp4:
            mq, msq = per_1x32_mx_quant_hip(x_bf16.contiguous(), quant_dtype=self._qd)
        else:
            mq, msq = per_1x32_mx_quant_hip(x_bf16.contiguous(), quant_dtype=self._qd,
                                            scale_type=self._stp)
        return mq, msq.view(torch.uint8)

    def forward_bf16(self, x_bf16, wts, topk_ids, *, stream=None, slice_output: bool = True):
        """带量化的端到端入口：**bf16 激活进，bf16 combine 出**。

        内部先用 ``per_1x32_mx_quant_hip``(经 ``self.quantize``)把 bf16 激活量化成
        fp8/fp4 + e8m0 scale，再走与 ``forward`` 完全相同的两段 kernel。
        """
        x_q, scales = self.quantize(x_bf16)
        return self.forward(x_q, scales, wts, topk_ids,
                            stream=stream, slice_output=slice_output)

    def forward(self, x_q, scales, wts, topk_ids, *, stream=None, slice_output: bool = True):
        """端到端 MoE：fp8/fp4 量化激活 -> combine 后 bf16 输出。

        Parameters
        ----------
        x_q
            本 rank 量化激活（a8w4: fp8 ``[run_tokens, model_dim]`` / a4w4: fp4
            ``[run_tokens, model_dim//2]``）。bf16 源可先调 ``quantize``。
        scales
            激活 e8m0 行 scale（``uint8`` view）。
        wts
            routing weights ``[run_tokens, topk]`` f32。
        topk_ids
            ``[run_tokens, topk]`` 全局 expert id。
        slice_output
            ``True`` 返回 ``[run_tokens, model_dim]`` bf16，否则返回完整
            ``[max_tok_per_rank, model_dim]`` combine 输出视图。

        流程：fp8/fp4 → stage1(megav1 单发核) → fp8/fp4 a2 → stage2(GEMM2+combine) → bf16。
        **纯两段 kernel,中间零桥接**：stage-1 megakernel 把 ``total_recv``(distinct recv,Plan A)
        **直接写进 combine op 的 buffer**;``tok_id_to_src``(identity 常量)在构造时已设。
        forward 不含任何额外 host/device 拷贝或 dispatch。
        """
        run_tokens = int(x_q.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        wc = wts.contiguous()
        ic = topk_ids.to(torch.int32).contiguous()

        # ---- stage-1 + merged GEMM2: 单发 megakernel ----
        # 一次 launch 内完成 dispatch ⊕ GEMM1+silu ⊕ GEMM2+combine-Stage1 P2P scatter；
        # 原地写 combine op 的 total_recv + atom 契约 a2/scale/sorted 表，并把 GEMM2 输出 P2P
        # scatter 进 comb_op 的 shmem_comb_inp。  (旧的独立 self._g2.run 已并入此 launch。)
        self.stage1.forward(x_q, wc, scales, ic, stream=stream)

        if self.stage2_mode == "fused":
            # combine Stage 2/3 (scatter-back + 本地 reduce) -- GEMM2 epilogue 已做完 Stage 1
            # P2P scatter。doweight 在 GEMM2 epilogue 内完成 -> enable_weights=False。
            ret = self.comb_op.combine_no_stage1(
                self._dummy_inp,
                None,
                None,
                cur_tok=run_tokens,
                enable_weights=False,
            )
            out_tok = ret[0] if isinstance(ret, (tuple, list)) else ret
        else:  # pragma: no cover - guarded in __init__
            raise NotImplementedError("stage2_mode='nonfused' not yet available")

        if out_tok is None:
            cfg = self.comb_cfg
            out_tok = (self.comb_op.shmem_comb_out_tok.view(torch.int8)[: self.mtpr * cfg.token_bytes]
                       .view(cfg.data_type).view(self.mtpr, cfg.token_view_dim))
        return out_tok[:run_tokens] if slice_output else out_tok

    __call__ = forward
