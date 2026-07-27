# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""MegaMoE v2 fused dispatch, GEMM1, GEMM2, and combine implementation."""

import mori.shmem as ms
import torch

import flydsl.expr as fx
from kernels.comm.flydsl_dispatch_combine_intranode_op import (
    FlyDSLDispatchCombineIntraNodeOp,
)

from ..mega_moe import MegaMoE, Stage1Output
from .dispatch import DISPATCH_TABLE_SIZE, DispatchSlot

__all__ = ["MegaMoEV2"]


class MegaMoEV2(MegaMoE):
    """Fused dispatch/GEMM1 + GEMM2/combine implementation.

    All EP ranks must invoke an instance with the same token count and routing
    mode. One instance supports one in-flight stage1 launch at a time because
    its epoch counters, workspaces, and outputs are reused.
    """

    # fmt: off
    def __init__(self, *args, stage1_dispatch_cu: int | None = None, stage1_grid_mult: int | None = None,
        stage1_tile_m_values: tuple[int, ...] | None = None, **kwargs):
    # fmt: on
        if not kwargs.get("enable_fused_stage1", True) or not kwargs.get("enable_fused_stage2", True):
            raise ValueError("MegaMoEV2 requires enable_fused_stage1=True and enable_fused_stage2=True")
        if kwargs.get("quant") != "a8w4":
            raise ValueError("MegaMoEV2 currently supports quant='a8w4' only")
        self._v2_dispatch_cu = None if stage1_dispatch_cu is None else int(stage1_dispatch_cu)
        self._v2_grid_mult = None if stage1_grid_mult is None else int(stage1_grid_mult)
        if stage1_tile_m_values is None:
            # Compact metadata is owned by one stage1 sort tile size. Prefer
            # the graph's actual tuning bucket when the caller supplies it;
            # max_tok_per_rank is only the allocation capacity and may be
            # shared by several decode batch sizes.
            token_capacity = int(
                kwargs["max_tok_per_rank"]
                if kwargs.get("tune_tokens") is None
                else kwargs["tune_tokens"]
            )
            stage1_tile_m_values = (
                (32,) if token_capacity <= 256 else
                (64,) if token_capacity <= 1024 else
                (128,)
            )
        self._v2_tile_m_values = tuple(
            sorted(
                {
                    int(tile_m)
                    for tile_m in stage1_tile_m_values
                }
            )
        )
        if not self._v2_tile_m_values or any(tile_m not in (32, 64, 128) for tile_m in self._v2_tile_m_values):
            raise ValueError("stage1_tile_m_values must contain only 32, 64, and/or 128")
        if len(self._v2_tile_m_values) != 1:
            raise ValueError(
                "stage1_tile_m_values must contain exactly one tile size so metadata capacity is unambiguous"
            )
        compact_tile_m = self._v2_tile_m_values[0]
        kwargs["tile_m"] = compact_tile_m
        super().__init__(*args, **kwargs)

    def _build_fused_stage1(self, w1, w1_scale):
        from .mega_moe_stage1 import make_stage1_autotuner

        cfg = self._s1cfg
        self._s1_scale_dim = cfg["scale_dim"]
        self.sort_block_m = self._v2_tile_m_values[0]
        self._s1_w1 = w1.contiguous()
        self._s1_w1_scale = w1_scale.contiguous()
        op = self.comb_op._gm
        assert op is not None, "combine op was built without enable_group_major"
        self._s1_op = op
        self._s1_nvm = op.num_valid_max
        self._s1_cap = op.ll_cap
        self._s1_epoch_parity = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self._s1_epoch_expected = torch.zeros(2, dtype=torch.int32, device=self.dev)
        self._s1_num_cu = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        self._allocate_dispatch_workspace(op)
        self._s1_mega = make_stage1_autotuner(
            self._v2_dispatch_cu, self._v2_grid_mult, self._v2_tile_m_values
        )

        v = op._ll_views()
        self._s1_rx = v["rx_em"]
        self._s1_scale_i32 = v["scale_em_i32"]

        inter_dim = self.inter_dim
        a2rows = self._s1_nvm
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

    def _allocate_dispatch_workspace(self, op):
        total_experts = self.world_size * self.epr
        workspace = {
            "local_hist": torch.zeros(total_experts, dtype=torch.int32, device=self.dev),
            "local_cursor": torch.zeros(total_experts, dtype=torch.int32, device=self.dev),
            "pair_order": torch.empty(self.mtpr * self.topk, dtype=torch.int32, device=self.dev),
            "pair_base": torch.empty(total_experts, dtype=torch.int32, device=self.dev),
            "pair_ready": torch.zeros(2, dtype=torch.int32, device=self.dev),
            "entry_count": torch.zeros(10, dtype=torch.int64, device=self.dev),
            "epoch_gate": torch.zeros(10, dtype=torch.int32, device=self.dev),
            "pair_order_ready": torch.zeros(2, dtype=torch.int32, device=self.dev),
            "work_head": torch.zeros(8 * 16, dtype=torch.int32, device=self.dev),
            "work_tail": torch.zeros(1, dtype=torch.int32, device=self.dev),
            "expert_tile_end": torch.empty(
                self.epr,
                dtype=torch.int32,
                device=self.dev,
            ),
            "active_experts": torch.empty(
                total_experts,
                dtype=torch.int32,
                device=self.dev,
            ),
            "active_count": torch.zeros(1, dtype=torch.int32, device=self.dev),
        }
        workspace["bigcnt"] = op._sym((self.world_size * self.epr,), torch.int32)
        workspace["count_done"] = op._sym((2 * self.world_size,), torch.int32)
        workspace["my_base"] = op._sym((total_experts,), torch.int32)
        workspace["plan_ready"] = op._sym((2 * self.world_size,), torch.int32)
        workspace["payload_ready"] = op._sym((2 * self.epr,), torch.int32)
        ms.shmem_barrier_all()
        workspace["p2p_bigcnt"] = op._p2p_table(workspace["bigcnt"])
        workspace["p2p_count_done"] = op._p2p_table(workspace["count_done"])
        workspace["p2p_my_base"] = op._p2p_table(workspace["my_base"])
        workspace["p2p_plan_ready"] = op._p2p_table(workspace["plan_ready"])
        workspace["p2p_payload_ready"] = op._p2p_table(workspace["payload_ready"])
        self._s1_dispatch_workspace = workspace

    def _build_v2_disp_table(self):
        op = self._s1_op
        workspace = self._s1_dispatch_workspace
        table = [0] * DISPATCH_TABLE_SIZE
        table[DispatchSlot.PAIR_BASE] = workspace["pair_base"].data_ptr()
        table[DispatchSlot.P2P_TOKEN] = op.p2p_rx_em.data_ptr()
        table[DispatchSlot.P2P_SCALE] = op.p2p_scale_em.data_ptr()
        table[DispatchSlot.P2P_WEIGHT] = op.p2p_wts_em.data_ptr()
        table[DispatchSlot.P2P_SRCMAP] = op.p2p_srcmap_em.data_ptr()
        table[DispatchSlot.SORTED_EXPERT] = op.sorted_expert_ids.data_ptr()
        table[DispatchSlot.TILE_ROW_BASE] = op.tile_row_base.data_ptr()
        table[DispatchSlot.NUM_VALID] = op.num_valid.data_ptr()
        table[DispatchSlot.SRCMAP] = op.srcmap_em.data_ptr()
        table[DispatchSlot.LOCAL_HIST] = workspace["local_hist"].data_ptr()
        table[DispatchSlot.COUNT_MATRIX] = workspace["bigcnt"].data_ptr()
        table[DispatchSlot.P2P_COUNT_MATRIX] = workspace["p2p_bigcnt"].data_ptr()
        table[DispatchSlot.COUNT_DONE] = workspace["count_done"].data_ptr()
        table[DispatchSlot.P2P_COUNT_DONE] = workspace["p2p_count_done"].data_ptr()
        table[DispatchSlot.TASK_ROW_BASE] = workspace["my_base"].data_ptr()
        table[DispatchSlot.LOCAL_CURSOR] = workspace["local_cursor"].data_ptr()
        table[DispatchSlot.P2P_PAYLOAD_READY] = workspace["p2p_payload_ready"].data_ptr()
        table[DispatchSlot.PAIR_ORDER] = workspace["pair_order"].data_ptr()
        table[DispatchSlot.P2P_TASK_ROW_BASE] = workspace["p2p_my_base"].data_ptr()
        table[DispatchSlot.P2P_PLAN_READY] = workspace["p2p_plan_ready"].data_ptr()
        table[DispatchSlot.PLAN_READY] = workspace["plan_ready"].data_ptr()
        table[DispatchSlot.PAIR_READY] = workspace["pair_ready"].data_ptr()
        table[DispatchSlot.ENTRY_COUNT] = workspace["entry_count"].data_ptr()
        table[DispatchSlot.EPOCH_GATE] = workspace["epoch_gate"].data_ptr()
        table[DispatchSlot.PAIR_ORDER_READY] = workspace[
            "pair_order_ready"
        ].data_ptr()
        table[DispatchSlot.WORK_HEAD] = workspace["work_head"].data_ptr()
        table[DispatchSlot.WORK_TAIL] = workspace["work_tail"].data_ptr()
        table[DispatchSlot.EXPERT_TILE_END] = workspace[
            "expert_tile_end"
        ].data_ptr()
        table[DispatchSlot.ACTIVE_EXPERTS] = workspace["active_experts"].data_ptr()
        table[DispatchSlot.ACTIVE_COUNT] = workspace["active_count"].data_ptr()
        table[DispatchSlot.RUNNING] = op.running.data_ptr()
        table[DispatchSlot.P2P_RUNNING] = op.p2p_running.data_ptr()
        self._s1_disp = torch.tensor(table, dtype=torch.int64, device=self.dev)

    def _run_fused_stage1(self, x, wts, scales, topk_ids, stream=None) -> "Stage1Output":
        if stream is None:
            stream = fx.Stream(torch.cuda.current_stream())
        cur_tok = int(x.shape[0])
        if cur_tok > self.mtpr:
            raise ValueError(f"run_tokens={cur_tok} > max_tok_per_rank={self.mtpr}")
        if x.dtype != torch.float8_e4m3fn or not x.is_contiguous():
            raise ValueError("x must be contiguous float8_e4m3fn")
        if tuple(x.shape) != (cur_tok, self.model_dim):
            raise ValueError(f"x must have shape ({cur_tok}, {self.model_dim})")
        if wts.dtype != torch.float32 or not wts.is_contiguous():
            raise ValueError("wts must be contiguous float32")
        if tuple(wts.shape) != (cur_tok, self.topk):
            raise ValueError(f"wts must have shape ({cur_tok}, {self.topk})")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        if tuple(topk_ids.shape) != (cur_tok, self.topk):
            raise ValueError(f"topk_ids must have shape ({cur_tok}, {self.topk})")
        if not scales.is_contiguous():
            raise ValueError("scales must be contiguous")
        op = self._s1_op
        # fmt: off
        self._s1_mega(self._s1_out, self._s1_rx, self._s1_w1, self._s1_scale_i32, self._s1_w1_scale,
            op.tile_row_base, op.sorted_expert_ids, op.num_valid, self._s1_osd, fx.Int32(self._s1_nvm),
            fx.Int64(self._s1_disp.data_ptr()), fx.Int32(cur_tok), fx.Int64(x.data_ptr()),
            fx.Int64(topk_ids.data_ptr()), fx.Int64(wts.data_ptr()), fx.Int64(scales.data_ptr()),
            fx.Int64(self._s1_epoch_parity.data_ptr()), fx.Int64(self._s1_epoch_expected.data_ptr()),
            stream, model_dim=self.model_dim, inter_dim=self.inter_dim,
            rank=self.rank, experts_per_rank=self.epr, fuse_npes=self.world_size, fuse_topk=self.topk,
            fuse_cap=self._s1_cap, fuse_mtpr=self.mtpr, fuse_scale_dim=self._s1_scale_dim,
            sort_block_m=self.sort_block_m, num_cu=self._s1_num_cu,
            tune_tokens=cur_tok, dispatch_constraint=-1 if self._v2_dispatch_cu is None else self._v2_dispatch_cu,
            grid_constraint=-1 if self._v2_grid_mult is None else self._v2_grid_mult,
            tile_m_constraint=",".join(str(v) for v in self._v2_tile_m_values),
            autotune_schema=self._s1_mega.schema)
        # fmt: on
        self._s1_active_tile_m = int(self._s1_mega.last_config.kwargs["sort_block_m"])
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
        from .mega_moe_stage2 import make_gemm2_autotuner

        FlyDSLDispatchCombineIntraNodeOp._ENABLE_COMBINE_NO_STAGE1 = True
        comb_cfg = self.comb_cfg
        dev = torch.device("cuda", comb_cfg.rank)
        max_recv = comb_cfg.world_size * comb_cfg.max_num_inp_token_per_rank
        k = comb_cfg.num_experts_per_token
        cu_num = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self._g2v2_inter = int(self.inter_dim)
        self._g2v2_hidden = int(comb_cfg.hidden_dim)
        # Full flydsl Autotuner (disk-cached best config, per-M key, collective bench). Invariants form
        # the tuning key + compile params; the tuner varies BK/use_nt/g2_*/persist.
        self._g2_tuner = make_gemm2_autotuner(a_dtype="fp8")
        self._g2_invariants = dict(
            model_dim=int(comb_cfg.hidden_dim), inter_dim=int(self.inter_dim),
            experts=int(comb_cfg.num_experts_per_rank), topk=int(k), rank=int(comb_cfg.rank),
            npes=int(comb_cfg.world_size), max_tok=int(comb_cfg.max_num_inp_token_per_rank),
            recv_cap=int(comb_cfg.effective_max_recv),
            comb_inp_nbytes=int(comb_cfg.max_num_inp_token_per_rank) * int(k) * int(comb_cfg.hidden_dim) * 2,
            HIDDEN_MAX=int(comb_cfg.hidden_dim), INTER_MAX=int(self.inter_dim), cu_num=int(cu_num),
        )
        self._g2_dummy_inp = torch.zeros(max_recv, comb_cfg.hidden_dim, dtype=comb_cfg.combine_dtype, device=dev)

    def _run_fused_stage2(self, s1, run_tokens, stream=None):
        comb_op = self.comb_op
        tile_row_base = self._s1_op.tile_row_base
        active_tile_m = self._s1_active_tile_m
        if stream is None:
            stream = torch.cuda.current_stream()
        s_fx = fx.Stream(stream.cuda_stream)
        size_expert_ids = s1.sorted_expert_ids.numel()
        # flydsl gemm2 Autotuner: positional runner args then per-M tuning key (tune_tokens + invariants).
        self._g2_tuner(
            fx.Int64(s1.a2.view(-1).data_ptr()), fx.Int64(s1.a2_scale.data_ptr()),
            fx.Int64(self.w2.data_ptr()), fx.Int64(self.w2_scale.data_ptr()),
            fx.Int64(s1.sorted_expert_ids.data_ptr()), fx.Int64(s1.num_valid_ids.data_ptr()),
            fx.Int64(s1.sorted_token_ids.data_ptr()), fx.Int64(s1.sorted_weights.data_ptr()),
            fx.Int64(tile_row_base.data_ptr()), comb_op._fx_tis, comb_op._fx_p2p_comb_inp,
            int(size_expert_ids), self._g2v2_inter, self._g2v2_hidden, s_fx,
            tune_tokens=int(run_tokens), SBM=active_tile_m,
            **self._g2_invariants,
        )
        return comb_op.combine_no_stage1(self._g2_dummy_inp, None, None, cur_tok=run_tokens, enable_weights=False)
