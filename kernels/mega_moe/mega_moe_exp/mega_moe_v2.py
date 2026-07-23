# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Experimental MegaMoE v2, isolated from the production MegaMoE implementation."""

import os

import mori.shmem as ms
import torch
import torch.distributed as dist

import flydsl.expr as fx
from kernels.comm.flydsl_dispatch_combine_intranode_op import FlyDSLDispatchCombineIntraNodeOp
from kernels.common.tensor_shim import _run_compiled

from ..mega_moe import MegaMoE, Stage1Output

__all__ = ["MegaMoEV2"]


class MegaMoEV2(MegaMoE):
    """Experimental fused dispatch/GEMM1 + GEMM2/combine implementation."""

    def __init__(self, *args, **kwargs):
        if not kwargs.get("enable_fused_stage1", True) or not kwargs.get("enable_fused_stage2", True):
            raise ValueError("MegaMoEV2 requires enable_fused_stage1=True and enable_fused_stage2=True")
        super().__init__(*args, **kwargs)

    def _build_fused_stage1(self, w1, w1_scale):
        from .group_gemm.tune_config import get_config
        from .mega_moe_stage1 import compile_mega_moe_stage1

        cfg = self._s1cfg
        out_dtype = self.a2_dtype
        self._s1_scale_dim = cfg["scale_dim"]
        self._s1_compact = False  # v2 = non-compact fixed-slot tile list (se/trb)
        tc = get_config(16)  # decode tune (bs16); TODO per-bs recompile
        self.sort_block_m = int(tc["tile_m"])
        self._s1_tile_n = int(tc["tile_n"])
        self._s1_w1 = w1.contiguous()
        self._s1_w1_scale = w1_scale.contiguous()
        op = self.comb_op._gm
        assert op is not None, "combine op was built without enable_group_major"
        self._s1_op = op
        self._s1_nvm = op.num_valid_max
        self._s1_max_blocks = op.max_blocks
        self._s1_cap = op.ll_cap
        self._s1_meta = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self._s1_ready = torch.zeros(1, dtype=torch.int32, device=self.dev)
        cu_num = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        # Split roles by block ID: the first dispatch CUs are non-spinning producers, followed by
        # the tuned oversubscribed GEMM grid.
        _gm = int(tc["grid_mult"])
        _dispatch_cu = min(32, cu_num)

        self._s1_mega = None
        for pe in range(self.world_size):
            if self.rank == pe:
                self._s1_mega = compile_mega_moe_stage1(
                    model_dim=self.model_dim,
                    inter_dim=self.inter_dim,
                    experts=self.experts,
                    topk=1,
                    rank=self.rank,
                    experts_per_rank=self.epr,
                    fuse_npes=self.world_size,
                    fuse_topk=self.topk,
                    fuse_cap=self._s1_cap,
                    fuse_mtpr=self.mtpr,
                    fuse_scale_dim=self._s1_scale_dim,
                    fuse_scale_type_size=1,
                    tile_m=int(tc["tile_m"]),
                    tile_n=int(tc["tile_n"]),
                    tile_k=int(tc["tile_k"]),
                    sort_block_m=int(tc["tile_m"]),
                    num_waves=int(tc["num_waves"]),
                    grid_mult=_gm,
                    wgm=int(tc.get("wgm", 1)),
                    sched_nmajor=bool(tc.get("sched_nmajor", False)),
                    pipe_weights=bool(tc.get("pipe_weights", True)),
                    mfma_amajor=bool(tc.get("mfma_amajor", False)),
                    swizzle_a=bool(tc.get("swizzle_a", False)),
                    a_dtype=self.a2_dtype,
                    out_dtype=out_dtype,
                    num_cu=cu_num,
                    num_dispatch_cu=_dispatch_cu,
                    use_xcd=os.environ.get("MEGA_V2_NOXCD", "0") != "1",
                    skip_gemm=os.environ.get("MEGA_V2_SKIPGEMM", "0") == "1",
                )
            if dist.is_initialized():
                dist.barrier()

        v = op._ll_views()
        self._s1_rx = v["rx_em"]
        self._s1_scale_i32 = v["scale_em_i32"]

        inter_dim = self.inter_dim
        nf4 = out_dtype == "fp4"
        nf8 = out_dtype == "fp8"
        self._s1_out_is_quant = nf4 or nf8
        a2rows = self._s1_max_blocks * self.sort_block_m  # fixed-slot rows
        self._s1_a2rows = a2rows
        if nf4:
            self._s1_out = torch.zeros((a2rows, inter_dim // 2), dtype=torch.uint8, device=self.dev)
        elif nf8:
            self._s1_out = torch.zeros((a2rows, inter_dim), dtype=torch.float8_e4m3fn, device=self.dev)
        else:
            self._s1_out = torch.zeros((a2rows, 1, inter_dim), dtype=torch.float16, device=self.dev)
        self._s1_bias = torch.empty((0,), dtype=torch.float32, device=self.dev)
        if self._s1_out_is_quant:
            prows = ((a2rows + 255) // 256) * 256
            pcols = (((inter_dim // 32) + 7) // 8) * 8
            self._s1_osd = torch.zeros(prows * pcols + inter_dim, dtype=torch.uint8, device=self.dev)
        else:
            self._s1_osd = torch.empty((0,), dtype=torch.uint8, device=self.dev)

        # buffers referenced by _build_disp_table (non-compact table idx 19/20/25); the split-role
        # prologue does NOT read 19/20/25, so these are unused placeholders, just must exist.
        self._s1_sti = torch.zeros(self._s1_nvm, dtype=torch.int32, device=self.dev)
        self._s1_se_atom = torch.zeros(self._s1_nvm // 32 + 8, dtype=torch.int32, device=self.dev)
        self._s1_sw_atom = torch.zeros(self.world_size * self.mtpr * self.topk, dtype=torch.float32, device=self.dev)
        self._s1_wts_sorted = torch.zeros(self._s1_nvm, dtype=torch.float32, device=self.dev)

        self._build_disp_table()

        # stage2 handoff: a2 fixed-slot; metadata from op (srcmap=(s<<24)|t, se, trb, nv, per-slot wts).
        self._s1_output = Stage1Output(
            a2=self._s1_out,
            a2_scale=self._s1_osd,
            sorted_token_ids=op.srcmap_em,  # fused (s<<24)|t per fixed slot -> stage2 decode
            sorted_expert_ids=op.sorted_expert_ids,
            sorted_weights=op.wts_em.view(torch.float32),  # per-fixed-slot routing weight
            num_valid_ids=op.num_valid,
            wts_buf=None,
        )

    def _run_fused_stage1(self, x, wts, scales, topk_ids, stream=None) -> "Stage1Output":
        if stream is None:
            stream = fx.Stream(torch.cuda.current_stream())
        cur_tok = int(x.shape[0])
        xc = x.contiguous()
        wc = wts.contiguous()
        ic = topk_ids.to(torch.int32).contiguous()
        sc = scales.contiguous()
        op = self._s1_op
        op.total_recv.zero_()
        self._s1_ready.zero_()
        a_mat = self._s1_agv(self._s1_rx)
        self._s1_mega(
            self._s1_out,
            a_mat,
            self._s1_w1,
            self._s1_scale_i32,
            self._s1_w1_scale,
            op.tile_row_base,  # sorted_token_ids arg = fixed-slot row bases
            op.sorted_expert_ids,
            op.num_valid,
            self._s1_osd,
            fx.Int32(self._s1_nvm),
            fx.Int32(self.inter_dim * 2),
            fx.Int32(self.model_dim),
            fx.Int32(self._s1_max_blocks),
            fx.Int64(self._s1_disp.data_ptr()),
            fx.Int32(cur_tok),
            fx.Int64(xc.data_ptr()),
            fx.Int64(ic.data_ptr()),
            fx.Int64(wc.data_ptr()),
            fx.Int64(sc.data_ptr()),
            fx.Int64(self._s1_ready.data_ptr()),
            stream=stream,
        )
        if os.environ.get("MEGA_V2_DIAG", "0") == "1":
            torch.cuda.synchronize()
            nvv = int(self._s1_op.num_valid.view(-1)[0].item())
            nrows = min(max(nvv, 1), self._s1_a2rows)
            a2f = self._s1_out.view(self._s1_a2rows, -1)[:nrows].float()
            tm = self.sort_block_m
            nt = nvv // tm
            trb = self._s1_op.tile_row_base[:nt].to(torch.int64)
            fixed_rows = (trb[:, None] + torch.arange(tm, device=self.dev)[None, :]).reshape(-1)
            src = self._s1_op.srcmap_em[fixed_rows]
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
            x_e8 = sc.view(torch.uint8).view(-1, self.model_dim // 32)[src_tok].float()
            x_deq = (
                xc.view(-1, self.model_dim)[src_tok]
                .float()
                .view(-1, self.model_dim // 32, 32)
                .mul(torch.pow(2.0, x_e8 - 127.0)[:, :, None])
            )
            dispatch_rel = (torch.norm(rx_deq - x_deq) / torch.norm(x_deq)).item()
            dispatch_werr = (
                (self._s1_op.wts_em[fixed_rows[valid]] - wc.view(-1, self.topk)[src_tok, src_slot]).abs().max()
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
            self._diag_s1_src = src[valid]
            self._diag_s1_eids = self._s1_op.sorted_expert_ids[:nt].repeat_interleave(tm)[valid]
            print(
                f"[v2-diag rank={self.rank}] stage1: num_valid={nvv} real_rows={int(valid.sum().item())} "
                f"a2.finite={bool(torch.isfinite(a2f).all().item())} raw_absmax={a2f.abs().max().item():.3e} "
                f"deq_finite={bool(torch.isfinite(deq).all().item())} deq_absmax={deq.abs().max().item():.3e} "
                f"e8=[{int(e8.min().item())},{int(e8.max().item())}] "
                f"dispatch_relL2={dispatch_rel:.3e} weight_maxerr={dispatch_werr.item():.3e}",
                flush=True,
            )
        return self._s1_output

    def _build_fused_stage2(self, **kw):
        from .mega_moe_stage2 import compile_mega_moe_stage2

        FlyDSLDispatchCombineIntraNodeOp._ENABLE_COMBINE_NO_STAGE1 = True
        comb_cfg = self.comb_cfg
        dev = torch.device("cuda", comb_cfg.rank)
        max_recv = comb_cfg.world_size * comb_cfg.max_num_inp_token_per_rank
        k = comb_cfg.num_experts_per_token
        cu_num = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self._g2v2_launch = compile_mega_moe_stage2(
            model_dim=comb_cfg.hidden_dim,
            inter_dim=self.inter_dim,
            experts=comb_cfg.num_experts_per_rank,
            topk=k,
            rank=comb_cfg.rank,
            npes=comb_cfg.world_size,
            max_tok=comb_cfg.max_num_inp_token_per_rank,
            a_dtype="fp8",
            doweight=True,
            num_cu=cu_num,
            grid_mult=1,
        )
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
