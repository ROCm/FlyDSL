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
from .tmp_mega_gemm2_combine_op import FlyDSLMoeGemm2CombineOp

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

        # ---- stage-1: megav1 单发核（恒输出 atom 契约 a2）；total_recv_buf 指向 combine op 的
        # total_recv -> megakernel 原地写,forward 里无需任何 total_recv 拷贝 ----
        # 大 bs(buffer≥3GB)下 facade 自动切 compact dispatch（compact dense rx 绕开 4GB 输入墙 +
        # GEMM1 写 atom-logical a2,stage2 接线完全不变）；小 bs 走 fixed-slot。两路对外 I/O 一致。
        self.stage1 = FusedMoEMegaStage1(
            rank=rank, world_size=world_size, model_dim=model_dim, inter_dim=inter_dim,
            experts=experts, topk=topk, quant=quant, w1=w1, w1_scale=w1_scale,
            max_tok_per_rank=max_tok_per_rank, tune_tokens=tune_tokens,
            network=network, scheme=mega_scheme,
            unit_size=int(tile_m), tile_n=int(tile_n), tile_k=int(tile_k),
            warp_num_per_block=int(warp_num_per_block), waves_per_eu=int(waves_per_eu),
            use_async_copy=bool(use_async_copy), out_dtype="auto",
            total_recv_buf=self.comb_op.total_recv,
        )
        self.a2_dtype = self.stage1.a_dtype   # "fp8" | "fp4"  (== stage-1 out_dtype 'auto')

        self.w2 = w2 if w2.is_contiguous() else w2.contiguous()
        self.w2_scale = w2_scale if w2_scale.is_contiguous() else w2_scale.contiguous()
        self.max_recv = world_size * self.mtpr
        # megav1 _sti 编码 src_global=dest -> combine tok_id_to_src 恒为 identity(常量)；
        # 只在构造时设一次(combine 只读不写、megav1 不碰它)-> forward 里无需每步重设。
        self.comb_op.shmem_tok_id_to_src.copy_(
            torch.arange(self.max_recv, device=self.dev, dtype=torch.int32))
        torch.cuda.synchronize(); ms.shmem_barrier_all()

        # ---- fused GEMM2 ghost-gate 边界(mega 自闭环,不碰共用 gemm2 kernel) ----
        # 共用 gemm2 epilogue 的 ghost gate 用 `t < total_recv` 丢 padding 行。ATOM 契约里
        # t 是紧凑 recv 槽位 [0,total_recv);但 megav1 Plan-A 的 _sti 编码 t = src_global ∈
        # [0, world*mtpr),padding sentinel == world*mtpr(== max_recv)。若 gate 用 distinct
        # recv 计数当边界,会误丢 src_global>=total_recv 的真实行(高位 dest rank → 输出为 0)。
        # gate 读的是 comb_op._fx_out_total_recv 句柄;而 combine_no_stage1 读真实计数走的是
        # 另一句柄 _fx_trecv(同指 total_recv buffer),且 megav1 从不调用 comb_op.dispatch
        # (_fx_out_total_recv 的唯一写者),故 _fx_out_total_recv 在 mega 里唯一消费者就是
        # gemm2 ghost gate。这里把它重指到常量 max_recv:gate 变 `t < max_recv`,真实行全留、
        # padding(==max_recv)丢;combine 仍读真实 total_recv。gemm2 kernel / a2 布局 / 零桥接均不动。
        self._gemm2_gate_bound = torch.tensor(
            [self.max_recv], dtype=torch.int32, device=self.dev)
        self.comb_op._fx_out_total_recv = fx.Int64(self._gemm2_gate_bound.data_ptr())
        torch.cuda.synchronize(); ms.shmem_barrier_all()

        # ---- stage-2 backend ----
        self._g2 = None
        if stage2_mode == "fused":
            # Thread stage1's row-padding / sorted_expert_ids granularity into gemm2.  Stage1 emits
            # _se_atom (sorted_expert_ids) ONE entry per sort_block_m rows, and gemm2 reads it as
            # expert_ids[bx_m // sort_block_m]; the two MUST agree or gemm2 selects the wrong expert's
            # W2 (silent garbage).  Previously gemm2 used its own default sort_block_m (== its tile_m,
            # 32) which only matched stage1 when stage1's GEMM tile_m was 32 -> a tuned gemm2 tile_m=64
            # produced corrupt output.  gemm2 also requires sort_block_m % tile_m == 0 (each tile_m tile
            # stays within one expert), so the effective gemm2 tile_m must divide stage1's sort_block_m.
            _s1_sbm = int(self.stage1.sort_block_m)
            _eff_g2_tm = int(gemm2_tile_m) if int(gemm2_tile_m) > 0 else 32  # op default tile_m=32
            if _s1_sbm % _eff_g2_tm != 0:
                raise ValueError(
                    f"gemm2 tile_m={_eff_g2_tm} does not divide stage1 sort_block_m={_s1_sbm} "
                    f"(stage1 unit_size={self.stage1.unit_size}); each gemm2 tile_m tile must stay "
                    f"within one expert's sort_block_m-padded rows.  Set gemm2_tile_m to a divisor of "
                    f"{_s1_sbm} (e.g. 32), or raise stage1's tile_m so sort_block_m is a multiple of "
                    f"{_eff_g2_tm}.")
            g2_kwargs = dict(
                comb_cfg=self.comb_cfg, comb_op=self.comb_op, inter_dim=int(inter_dim),
                a_dtype=self.a2_dtype, b_dtype="fp4", sort_block_m=_s1_sbm,
            )
            # Only override FlyDSLMoeGemm2CombineOp's built-in tile/persist/xcd when a
            # tuned gemm2 tile is supplied (gemm2_tile_m > 0). On a tune miss the
            # caller (mega router) leaves these at -1, so the op keeps its own default
            # behavior -- the mega path never overrides the gemm2+combine defaults.
            if int(gemm2_tile_m) > 0:
                g2_kwargs.update(
                    tile_m=int(gemm2_tile_m), tile_n=int(gemm2_tile_n),
                    tile_k=int(gemm2_tile_k), persist_m=int(gemm2_persist_m),
                    xcd_swizzle=int(xcd_swizzle),
                )
            self._g2 = FlyDSLMoeGemm2CombineOp(**g2_kwargs)
        else:
            # 非融合（compile_mixed_moe_gemm2 + comb_op.combine）I/O 与融合等价（已确认：
            # 同一个 GEMM2 kernel，融合版只多 fused_p2p_scatter），但其精确接线（gemm2
            # 走 flyc.compile 模板、combine 的 zero_copy 注册 buffer / cfg.zero_copy /
            # accumulate 布局 / megav1 atom 契约下的 combine indices）尚未在 megav1 上跑过
            # 验证。为避免把未验证代码塞进生产路径，暂不实现；GPU 空闲后接线 + 验证再开放。
            raise NotImplementedError(
                "stage2_mode='nonfused' pending wiring+validation on megav1; "
                "use stage2_mode='fused' (validated MEGA-ATOM NATIVE combine recipe)."
            )

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

        # ---- stage-1: megav1 单发核；原地写 combine op 的 total_recv + atom 契约 a2/scale/sorted 表 ----
        self.stage1.forward(x_q, wc, scales, ic, stream=stream)

        if self.stage2_mode == "fused":
            ret = self._g2.run(
                a2=self.stage1._out.view(-1), w2=self.w2,
                a2_scale=self.stage1._osd, w2_scale=self.w2_scale,
                sorted_token_ids=self.stage1._sti,
                sorted_expert_ids=self.stage1._se_atom,
                sorted_weights=self.stage1._wts_sorted,
                num_valid_ids=self.stage1._nv,
                wts_buf=self.stage1._sw_atom,
                cur_tok=run_tokens, stream=stream,
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
