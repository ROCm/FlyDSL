# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""融合 MoE Stage2 GEMM + EP Combine 算子包装。

封装 :func:`compile_fused_moe_gemm2_combine` 的 host 调度，并复用
:class:`FlyDSLDispatchCombineIntraNodeOp` 的 shmem buffer / P2P 表 / xdev barrier。
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
_VALID_MODES = ("auto", "full", "stage1_only")


def _resolve_force_mode(force_mode: str) -> str:
    """归一化 force_mode：env 覆盖 + ``fallback`` 同义于 ``stage1_only``。"""
    env = os.environ.get(_FORCE_MODE_ENV, "").strip().lower()
    mode = (env or force_mode).strip().lower()
    if mode == "fallback":
        mode = "stage1_only"
    if mode not in _VALID_MODES:
        raise ValueError(
            f"force_mode must be one of {_VALID_MODES} (or 'fallback'), got {mode!r}"
        )
    return mode


class FlyDSLMoeGemm2CombineOp:
    """Fused MoE-GEMM2 + EP-Combine 算子。

    Parameters
    ----------
    comb_cfg, comb_op
        共用 dispatch op 的 config 与已分配的 shmem / P2P 基础设施。
    inter_dim
        GEMM2 输入维度（GEMM1 输出）；输出维度 = ``comb_cfg.hidden_dim``。
    tile_m, tile_n, tile_k, persist_m
        GEMM2 tile 调度参数。
    a_dtype, b_dtype
        GEMM2 输入/权重 dtype。
    force_mode
        ``auto`` / ``full`` / ``stage1_only``（``fallback`` 同义）。
        亦可由环境变量 ``FLYDSL_GEMM2_COMBINE_FORCE_MODE`` 覆盖。
    """

    # 历史兼容：上层测试脚本通过 ``getattr(cls, 'READY', False)`` 判断 fused
    # kernel 是否已就绪，从而决定是否 graceful skip。fused 路径全量上线后恒为 True。
    READY: bool = True

    def __init__(
        self,
        *,
        comb_cfg: FlyDSLDispatchCombineConfig,
        comb_op: FlyDSLDispatchCombineIntraNodeOp,
        inter_dim: int,
        tile_m: int = 32, tile_n: int = 128, tile_k: int = 128,
        persist_m: int = 4,
        a_dtype: str = "fp8",
        b_dtype: str = "fp4",
        force_mode: str = "auto",
        xcd_swizzle: int = 0,
    ):
        self.comb_cfg    = comb_cfg
        self.comb_op     = comb_op
        self.inter_dim   = inter_dim
        self.tile_m      = tile_m
        self.tile_n      = tile_n
        self.tile_k      = tile_k
        self.persist_m   = persist_m
        self.a_dtype     = a_dtype
        self.b_dtype     = b_dtype
        self.xcd_swizzle = xcd_swizzle
        self.force_mode  = _resolve_force_mode(force_mode)

        # fp8_direct_cast: GEMM2 epilogue 直走 cvt_pk_fp8_f32 + 1B/elem P2P scatter，
        # 避免 baseline 的 input.to(fp8) 拷贝，保证 fused/baseline 在同一字节流上对比。
        self._fp8_cast = (
            getattr(comb_cfg, "quant_type", "none") == "fp8_direct_cast"
            and comb_cfg.data_type == torch.bfloat16
        )
        if self._fp8_cast:
            self._out_dtype_str = "fp8e4m3fn"
        else:
            self._out_dtype_str = "bf16" if comb_cfg.data_type == torch.bfloat16 else "f16"

        self.chosen_mode = self._select_mode()

        self._launch_fn = None
        self._compiled = None
        self._alloc_dummy_tensors()

    def _select_mode(self) -> str:
        """auto: 若 GEMM2 自然 grid 能整 resident 选 ``full``，否则 ``stage1_only``。"""
        if self.force_mode != "auto":
            return self.force_mode
        comb_cfg = self.comb_cfg
        max_resident = estimate_max_resident_blocks(chip=comb_cfg.chip, block_dim=256)
        max_recv = comb_cfg.world_size * comb_cfg.max_num_inp_token_per_rank
        per_e_blocks = (max_recv + self.tile_m - 1) // self.tile_m
        nat_blocks = (comb_cfg.hidden_dim // self.tile_n) * (
            (per_e_blocks * comb_cfg.num_experts_per_rank + self.persist_m - 1)
            // self.persist_m
        )
        return "full" if nat_blocks <= max_resident else "stage1_only"

    def _alloc_dummy_tensors(self):
        """cudagraph 兼容：所有占位 tensor 必须在 ctor 阶段一次性 alloc。

        capture 内部不能调用 torch.zeros / torch.empty（会触发 hipMalloc 报错）。
        """
        comb_cfg = self.comb_cfg
        dev = torch.device("cuda", comb_cfg.rank)
        max_recv = comb_cfg.world_size * comb_cfg.max_num_inp_token_per_rank

        # arg_out 在 fused 路径下不读写，但 launcher 签名要求有效 tensor。
        self._dummy_out = torch.zeros(1, dtype=comb_cfg.data_type, device=dev)
        self._dummy_bias = torch.empty(0, dtype=comb_cfg.data_type, device=dev)

        # combine_no_stage1 的 input 占位：fp8_cast 模式直接以 fp8 分配，
        # 避开 combine 内部 bf16->fp8 .to() + .contiguous() 拷贝（~12us / chain）。
        inp_dtype = torch.float8_e4m3fn if self._fp8_cast else comb_cfg.data_type
        self._dummy_inp = torch.zeros(
            max_recv, comb_cfg.hidden_dim, dtype=inp_dtype, device=dev
        )

    def info(self) -> dict:
        return dict(
            chosen_mode=self.chosen_mode,
            force_mode=self.force_mode,
            tile=(self.tile_m, self.tile_n, self.tile_k),
            persist_m=self.persist_m,
            xcd_swizzle=self.xcd_swizzle,
            a_dtype=self.a_dtype, b_dtype=self.b_dtype,
            out_dtype=self._out_dtype_str,
            fp8_cast=self._fp8_cast,
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

        ``wts_buf`` 是 dispatch 写入 combine 的 f32[max_recv*k] 权重 buffer，
        None 时回退到 ``comb_op.shmem_disp_out_wts``。
        ``dispatch_total_recv`` 仅 ``full`` 模式使用，``stage1_only`` 忽略。

        返回 ``(out_tok, out_wts)``，等价于 ``comb_op.combine`` 的输出视图。
        """
        del dispatch_total_recv  # reserved for full mode
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

    def _ensure_launch_fn(self):
        if self._launch_fn is not None:
            return
        comb_cfg = self.comb_cfg
        self._launch_fn = compile_fused_moe_gemm2_combine(
            model_dim=comb_cfg.hidden_dim,
            inter_dim=self.inter_dim,
            experts=comb_cfg.num_experts_per_rank,
            topk=1,
            tile_m=self.tile_m, tile_n=self.tile_n, tile_k=self.tile_k,
            persist_m=self.persist_m,
            a_dtype=self.a_dtype, b_dtype=self.b_dtype,
            out_dtype=self._out_dtype_str,
            rank=comb_cfg.rank, npes=comb_cfg.world_size,
            max_tok_per_rank=comb_cfg.max_num_inp_token_per_rank,
            experts_per_token=comb_cfg.num_experts_per_token,
            mode=self.chosen_mode,
            xcd_swizzle=self.xcd_swizzle,
        )

    def _run_stage1_only(self, *, a2, w2, a2_scale, w2_scale,
                         sorted_token_ids, sorted_expert_ids, sorted_weights,
                         num_valid_ids, wts_buf=None, bias=None, stream=None):
        """fused GEMM2 (epilogue P2P scatter) + combine_no_stage1 (Stage 2/3)。"""
        self._ensure_launch_fn()

        comb_cfg = self.comb_cfg
        comb_op  = self.comb_op
        if stream is None:
            stream = torch.cuda.current_stream()
        s_fx = fx.Stream(stream.cuda_stream)

        # tokens_in 必须用 max_recv = world_size * max_num_inp_token_per_rank，
        # 与 baseline run_moe_stage2 / _build_gemm2_static_inputs 完全一致。
        # 误传 a2.shape[0]（1D view 下 = max_recv*inter_dim）会导致 row_valid
        # 早退失效 → padding 哨兵被当有效 row → epilogue 越界读 addr_tis →
        # P2P 飞到任意远端地址污染 combine 控制状态 → xdev barrier 永久挂死。
        tokens_in       = comb_cfg.world_size * comb_cfg.max_num_inp_token_per_rank
        n_in            = comb_cfg.hidden_dim
        k_in            = self.inter_dim
        size_expert_ids = sorted_expert_ids.numel()

        # 权重源默认走 shmem_disp_out_wts（f32[max_recv*k]，索引语义
        # wts[t*k + s]）；上层若传 wts_buf 必须保持同一布局。
        addr_wts_buf = (
            fx.Int64(wts_buf.data_ptr())
            if wts_buf is not None
            else comb_op._fx_disp_out_wts
        )

        bias_t = bias if bias is not None else self._dummy_bias

        # 首调用用 fx 包装的 typed scalars 让 flyc 推断 launcher 签名；
        # 之后命中缓存的 self._compiled 直接传 raw int 即可。
        common_args = (
            self._dummy_out, a2, w2, a2_scale, w2_scale,
            sorted_token_ids, sorted_expert_ids, sorted_weights,
            num_valid_ids, bias_t,
            comb_op._fx_tis,
            comb_op._fx_p2p_comb_inp,
            addr_wts_buf,
            comb_op._fx_p2p_comb_inp_wts,
        )
        if self._compiled is None:
            self._compiled = flyc.compile(
                self._launch_fn,
                *common_args,
                fx.Int32(tokens_in), fx.Int32(n_in), fx.Int32(k_in),
                fx.Int32(size_expert_ids),
                s_fx,
            )
        else:
            self._compiled(
                *common_args,
                tokens_in, n_in, k_in, size_expert_ids,
                s_fx,
            )

        return comb_op.combine_no_stage1(self._dummy_inp, None, None)

    def _run_full(self, a2, w2, a2_scale, w2_scale,
                  sti, eids, sw, num_valid_ids, bias):
        """单 kernel 端到端 GEMM2 + Stage1/2/3。"""
        raise NotImplementedError("_run_full: pending kernel implementation (PR2)")
