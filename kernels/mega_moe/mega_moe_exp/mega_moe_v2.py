# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Experimental MegaMoE v2, isolated from the production MegaMoE implementation."""

import os

import mori.shmem as ms
import torch

import flydsl.expr as fx
from kernels.comm.flydsl_dispatch_combine_intranode_op import FlyDSLDispatchCombineIntraNodeOp
from kernels.common.tensor_shim import _run_compiled

from ..mega_moe import MegaMoE, Stage1Output

__all__ = ["MegaMoEV2"]


class MegaMoEV2(MegaMoE):
    """Experimental fused dispatch/GEMM1 + GEMM2/combine implementation."""

    # fmt: off
    def __init__(self, *args, stage1_dispatch_cu: int | None = None, stage1_grid_mult: int | None = None, **kwargs):
    # fmt: on
        if not kwargs.get("enable_fused_stage1", True) or not kwargs.get("enable_fused_stage2", True):
            raise ValueError("MegaMoEV2 requires enable_fused_stage1=True and enable_fused_stage2=True")
        if kwargs.get("quant") != "a8w4":
            raise ValueError("MegaMoEV2 currently supports quant='a8w4' only")
        self._v2_dispatch_cu = None if stage1_dispatch_cu is None else int(stage1_dispatch_cu)
        self._v2_grid_mult = None if stage1_grid_mult is None else int(stage1_grid_mult)
        kwargs["tile_m"] = 32
        super().__init__(*args, **kwargs)

    def _build_fused_stage1(self, w1, w1_scale):
        from .mega_moe_stage1 import make_stage1_autotuner

        cfg = self._s1cfg
        self._s1_scale_dim = cfg["scale_dim"]
        self.sort_block_m = 32
        self._s1_w1 = w1.contiguous()
        self._s1_w1_scale = w1_scale.contiguous()
        op = self.comb_op._gm
        assert op is not None, "combine op was built without enable_group_major"
        self._s1_op = op
        self._s1_nvm = op.num_valid_max
        self._s1_max_blocks = op.max_blocks
        self._s1_cap = op.ll_cap
        self._s1_epoch_parity = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self._s1_epoch_expected = torch.zeros(2, dtype=torch.int32, device=self.dev)
        self._s1_epoch_ready = torch.zeros(2, dtype=torch.int32, device=self.dev)
        self._s1_num_cu = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self._s1_use_xcd = os.environ.get("MEGA_V2_NOXCD", "0") != "1"
        self._s1_mega = make_stage1_autotuner(self._v2_dispatch_cu, self._v2_grid_mult)

        v = op._ll_views()
        self._s1_rx = v["rx_em"]
        self._s1_scale_i32 = v["scale_em_i32"]

        inter_dim = self.inter_dim
        a2rows = self._s1_max_blocks * self.sort_block_m
        self._s1_a2rows = a2rows
        self._s1_out = torch.zeros((a2rows, inter_dim), dtype=torch.float8_e4m3fn, device=self.dev)
        prows = ((a2rows + 255) // 256) * 256
        pcols = (((inter_dim // 32) + 7) // 8) * 8
        self._s1_osd = torch.zeros(prows * pcols + inter_dim, dtype=torch.uint8, device=self.dev)
        self._build_v2_disp_table()

        # fmt: off
        self._s1_output = Stage1Output(a2=self._s1_out, a2_scale=self._s1_osd, sorted_token_ids=op.srcmap_em,
            sorted_expert_ids=op.sorted_expert_ids, sorted_weights=op.wts_em.view(torch.float32),
            num_valid_ids=op.num_valid, wts_buf=None)
        # fmt: on

    def _build_v2_disp_table(self):
        op = self._s1_op
        table = [
            0,
            0,
            0,
            0,
            op.gb1.data_ptr(),
            op.running.data_ptr(),
            op.done2.data_ptr(),
            op.ll_count.data_ptr(),
            op.p2p_rx_em.data_ptr(),
            op.p2p_scale_em.data_ptr(),
            op.p2p_idx_em.data_ptr(),
            op.p2p_wts_em.data_ptr(),
            op.p2p_srcmap_em.data_ptr(),
            op.p2p_running.data_ptr(),
            op.p2p_done2.data_ptr(),
            op.sorted_expert_ids.data_ptr(),
            op.tile_row_base.data_ptr(),
            op.num_valid.data_ptr(),
            0,
            op.srcmap_em.data_ptr(),
            0,
            op.total_recv.data_ptr(),
            op.dest_ctr.data_ptr(),
            op.recv_num.data_ptr(),
            op.p2p_recv_num.data_ptr(),
        ]
        self._s1_disp = torch.tensor(table, dtype=torch.int64, device=self.dev)

    def _run_fused_stage1(self, x, wts, scales, topk_ids, stream=None) -> "Stage1Output":
        if stream is None:
            stream = fx.Stream(torch.cuda.current_stream())
        cur_tok = int(x.shape[0])
        if x.dtype != torch.float8_e4m3fn or not x.is_contiguous():
            raise ValueError("x must be contiguous float8_e4m3fn")
        if wts.dtype != torch.float32 or not wts.is_contiguous():
            raise ValueError("wts must be contiguous float32")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        if not scales.is_contiguous():
            raise ValueError("scales must be contiguous")
        op = self._s1_op
        # fmt: off
        self._s1_mega(self._s1_out, self._s1_rx, self._s1_w1, self._s1_scale_i32, self._s1_w1_scale,
            op.tile_row_base, op.sorted_expert_ids, op.num_valid, self._s1_osd, fx.Int32(self._s1_nvm),
            fx.Int32(self.inter_dim * 2), fx.Int32(self.model_dim), fx.Int32(self._s1_max_blocks),
            fx.Int64(self._s1_disp.data_ptr()), fx.Int32(cur_tok), fx.Int64(x.data_ptr()),
            fx.Int64(topk_ids.data_ptr()), fx.Int64(wts.data_ptr()), fx.Int64(scales.data_ptr()),
            fx.Int64(self._s1_epoch_parity.data_ptr()), fx.Int64(self._s1_epoch_expected.data_ptr()),
            fx.Int64(self._s1_epoch_ready.data_ptr()), stream, model_dim=self.model_dim, inter_dim=self.inter_dim,
            rank=self.rank, experts_per_rank=self.epr, fuse_npes=self.world_size, fuse_topk=self.topk,
            fuse_cap=self._s1_cap, fuse_mtpr=self.mtpr, fuse_scale_dim=self._s1_scale_dim,
            sort_block_m=self.sort_block_m, num_cu=self._s1_num_cu, use_xcd=self._s1_use_xcd,
            tune_tokens=cur_tok, dispatch_constraint=-1 if self._v2_dispatch_cu is None else self._v2_dispatch_cu,
            grid_constraint=-1 if self._v2_grid_mult is None else self._v2_grid_mult,
            autotune_schema=self._s1_mega.schema)
        # fmt: on
        if os.environ.get("MEGA_V2_DIAG", "0") == "1" or getattr(self, "_validate_stage1", False):
            torch.cuda.synchronize()
            ms.shmem_barrier_all()
            nvv = int(op.num_valid.view(-1)[0].item())
            nrows = min(max(nvv, 1), self._s1_a2rows)
            a2f = self._s1_out.view(self._s1_a2rows, -1)[:nrows].float()
            tm = self.sort_block_m
            nt = nvv // tm
            trb = op.tile_row_base[:nt].to(torch.int64)
            fixed_rows = (trb[:, None] + torch.arange(tm, device=self.dev)[None, :]).reshape(-1)
            src = op.srcmap_em[fixed_rows]
            valid = ((src & 0x00FFFFFF) < self.max_recv) & ((src >> 24) < self.topk)
            compact_rows = torch.arange(nvv, device=self.dev, dtype=torch.int64)[valid]
            src_valid = src[valid]
            src_tok = (src_valid & 0x00FFFFFF).to(torch.int64) % self.mtpr
            src_slot = (src_valid >> 24).to(torch.int64)
            rx = self._s1_rx[fixed_rows[valid]].float()
            rx_e8 = (
                self._s1_scale_i32.view(torch.uint8)
                .view(self._s1_nvm, -1)[fixed_rows[valid], : self.model_dim // 32]
                .float()
            )
            rx_deq = rx.view(-1, self.model_dim // 32, 32).mul(torch.pow(2.0, rx_e8 - 127.0)[:, :, None])
            x_e8 = scales.view(torch.uint8).view(-1, self.model_dim // 32)[src_tok].float()
            x_deq = (
                x.view(-1, self.model_dim)[src_tok]
                .float()
                .view(-1, self.model_dim // 32, 32)
                .mul(torch.pow(2.0, x_e8 - 127.0)[:, :, None])
            )
            dispatch_rel = (torch.norm(rx_deq - x_deq) / torch.norm(x_deq)).item()
            dispatch_werr = (
                (op.wts_em[fixed_rows[valid]] - wts.view(-1, self.topk)[src_tok, src_slot]).abs().max()
            )
            scale_cols = (self.inter_dim // 32 + 7) // 8 * 8
            col_s = torch.arange(self.inter_dim // 32, device=self.dev, dtype=torch.int64)
            d0, d1, d2 = compact_rows >> 5, (compact_rows >> 4) & 1, compact_rows & 15
            d3, d4, d5 = col_s >> 3, (col_s >> 2) & 1, col_s & 3
            scale_off = (
                d0[:, None] * (scale_cols * 32)
                + d3[None, :] * 256
                + d5[None, :] * 64
                + d2[:, None] * 4
                + d4[None, :] * 2
                + d1[:, None]
            )
            e8 = self._s1_osd[scale_off]
            deq = (
                a2f[compact_rows]
                .view(-1, self.inter_dim // 32, 32)
                .mul(torch.pow(2.0, e8.float() - 127.0)[:, :, None])
                .reshape(-1, self.inter_dim)
            )
            self._diag_s1_deq = deq
            self._diag_s1_input = x_deq.reshape(-1, self.model_dim)
            self._diag_s1_eids = op.sorted_expert_ids[:nt].repeat_interleave(tm)[valid]
            print(
                f"[v2-diag rank={self.rank}] stage1: num_valid={nvv} real_rows={int(valid.sum().item())} "
                f"finite={bool(torch.isfinite(deq).all().item())} absmax={deq.abs().max().item():.3e} "
                f"dispatch_relL2={dispatch_rel:.3e} weight_maxerr={dispatch_werr.item():.3e}",
                flush=True,
            )
        return self._s1_output

    def forward(self, x_bf16, wts, topk_ids, *, stream=None, slice_output=True):
        run_tokens = int(x_bf16.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        if x_bf16.dtype != torch.bfloat16 or not x_bf16.is_contiguous():
            raise ValueError("x_bf16 must be contiguous bfloat16")
        if wts.dtype != torch.float32 or not wts.is_contiguous():
            raise ValueError("wts must be contiguous float32")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        x_q, scales = self.quantize(x_bf16)
        s1 = self._run_fused_stage1(x_q, wts, scales, topk_ids, stream=stream)
        return self._run_stage2(s1, run_tokens, stream, slice_output)

    def forward_prequant(self, x_q, scales, wts, topk_ids, *, stream=None, slice_output=True):
        run_tokens = int(x_q.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        s1 = self._run_fused_stage1(x_q, wts, scales, topk_ids, stream=stream)
        return self._run_stage2(s1, run_tokens, stream, slice_output)

    forward_bf16 = forward
    __call__ = forward

    def _build_fused_stage2(self, **kw):
        from .mega_moe_stage2 import compile_mega_moe_stage2

        FlyDSLDispatchCombineIntraNodeOp._ENABLE_COMBINE_NO_STAGE1 = True
        comb_cfg = self.comb_cfg
        dev = torch.device("cuda", comb_cfg.rank)
        max_recv = comb_cfg.world_size * comb_cfg.max_num_inp_token_per_rank
        k = comb_cfg.num_experts_per_token
        cu_num = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        # fmt: off
        self._g2v2_launch = compile_mega_moe_stage2(model_dim=comb_cfg.hidden_dim, inter_dim=self.inter_dim,
            experts=comb_cfg.num_experts_per_rank, topk=k, rank=comb_cfg.rank, npes=comb_cfg.world_size,
            max_tok=comb_cfg.max_num_inp_token_per_rank, num_cu=cu_num, grid_mult=1)
        # fmt: on
        self._g2_dummy_inp = torch.zeros(max_recv, comb_cfg.hidden_dim, dtype=comb_cfg.combine_dtype, device=dev)

    def _run_fused_stage2(self, s1, run_tokens, stream=None):
        comb_op = self.comb_op
        comb_cfg = self.comb_cfg
        if stream is None:
            stream = torch.cuda.current_stream()
        s_fx = fx.Stream(stream.cuda_stream)
        size_expert_ids = s1.sorted_expert_ids.numel()
        args = (
            fx.Int64(s1.a2.view(-1).data_ptr()),
            fx.Int64(s1.a2_scale.data_ptr()),
            fx.Int64(self.w2.data_ptr()),
            fx.Int64(self.w2_scale.data_ptr()),
            fx.Int64(s1.sorted_expert_ids.data_ptr()),
            fx.Int64(s1.sorted_token_ids.data_ptr()),
            fx.Int64(s1.sorted_weights.data_ptr()),
            fx.Int64(self._s1_op.tile_row_base.data_ptr()),  # trb: compact tile -> fixed-slot srcmap/weight row
            fx.Int64(s1.num_valid_ids.data_ptr()),
            comb_op._fx_tis,
            comb_op._fx_p2p_comb_inp,
        )
        _run_compiled(self._g2v2_launch, *args, fx.Int32(size_expert_ids), s_fx)
        diag = os.environ.get("MEGA_V2_DIAG", "0") == "1"
        if diag:
            torch.cuda.synchronize()
            ms.shmem_barrier_all()
            partial = (
                comb_op.shmem_comb_inp_tok.view(torch.bfloat16)
                .view(-1, comb_cfg.hidden_dim)[: run_tokens * comb_cfg.num_experts_per_token]
                .float()
            )
            inf_pos = torch.nonzero(torch.isinf(partial), as_tuple=False)
            inf_rows = torch.unique(inf_pos[:, 0]).tolist() if inf_pos.numel() else []
            finite_vals = partial[torch.isfinite(partial)]
            finite_absmax = finite_vals.abs().max().item() if finite_vals.numel() else float("nan")
            print(
                f"[v2-diag rank={self.rank}] stage2 partial: finite={bool(torch.isfinite(partial).all().item())} "
                f"nan={int(torch.isnan(partial).sum().item())} inf={int(torch.isinf(partial).sum().item())} "
                f"inf_rows={inf_rows} finite_absmax={finite_absmax:.3e}",
                flush=True,
            )
        ret = comb_op.combine_no_stage1(self._g2_dummy_inp, None, None, cur_tok=run_tokens, enable_weights=True)
        if diag:
            torch.cuda.synchronize()
            ms.shmem_barrier_all()
            if self.rank == 0:
                out = ret[0][:run_tokens].float()
                print(
                    f"[v2-diag] combine out: finite={bool(torch.isfinite(out).all().item())} "
                    f"nan={int(torch.isnan(out).sum().item())} inf={int(torch.isinf(out).sum().item())} "
                    f"absmax={out.abs().max().item():.3e}",
                    flush=True,
                )
        return ret
