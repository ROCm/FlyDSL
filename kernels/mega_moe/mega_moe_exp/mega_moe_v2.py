# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Experimental MegaMoE v2, isolated from the production MegaMoE implementation."""

import mori.shmem as ms
import torch

import flydsl.expr as fx
from kernels.comm.flydsl_dispatch_combine_intranode_op import (
    FlyDSLDispatchCombineIntraNodeOp,
    FlyDSLDispatchGroupMajorOp,
)
from kernels.common.tensor_shim import _run_compiled

from ..mega_moe import MegaMoE, Stage1Output
from .planner import (
    DISPATCH_TABLE_SIZE,
    SMALL_FIXED_TABLE_SIZE,
    DispatchSlot,
    SmallFixedSlot,
    make_stage1_dispatch_plan,
)

__all__ = ["MegaMoEV2"]

_SMALL_STAGE1_TUNED_BY_LOCAL_EXPERTS = {
    48: {
        "max_tokens": 256,
        "buckets": {
            8: {
                "num_waves": 8,
                "num_dispatch_cu": 8,
                "grid_mult": 2,
                "tile_n": 256,
            },
            224: {
                "num_waves": 8,
                "num_dispatch_cu": 48,
                "grid_mult": 4,
                "tile_n": 256,
            },
            256: {
                "num_waves": 8,
                "num_dispatch_cu": 192,
                "grid_mult": 2,
                "tile_n": 512,
            },
        },
    },
    96: {
        "max_tokens": 64,
        "buckets": {
            64: {
                "num_waves": 4,
                "num_dispatch_cu": 128,
                "grid_mult": 3,
                "tile_n": 128,
            },
        },
    },
}


class MegaMoEV2(MegaMoE):
    """Experimental fused dispatch/GEMM1 + GEMM2/combine implementation."""

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
        self._v2_tile_m_values = (
            (32,) if stage1_tile_m_values is None else tuple(int(tile_m) for tile_m in stage1_tile_m_values)
        )
        if not self._v2_tile_m_values or any(tile_m not in (32, 64, 128) for tile_m in self._v2_tile_m_values):
            raise ValueError("stage1_tile_m_values must contain only 32, 64, and/or 128")
        compact_tile_m = max(self._v2_tile_m_values)
        kwargs["tile_m"] = compact_tile_m
        make_stage1_dispatch_plan(
            batch_size=kwargs["max_tok_per_rank"],
            npes=kwargs["world_size"],
            experts_per_rank=kwargs["experts"] // kwargs["world_size"],
            topk=kwargs["topk"],
            tile_m=compact_tile_m,
            row_bytes=kwargs["model_dim"],
            use_per_tile_payload_resource=True,
        )
        super().__init__(*args, **kwargs)

    def _build_fused_stage1(self, w1, w1_scale):
        from .mega_moe_stage1 import make_stage1_autotuner

        cfg = self._s1cfg
        self._s1_scale_dim = cfg["scale_dim"]
        self.sort_block_m = max(self._v2_tile_m_values)
        self._s1_w1 = w1.contiguous()
        self._s1_w1_scale = w1_scale.contiguous()
        op = self.comb_op._gm
        assert op is not None, "combine op was built without enable_group_major"
        self._s1_op = op
        self._s1_nvm = op.num_valid_max
        self._s1_cap = op.ll_cap
        self._s1_epoch_parity = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self._s1_epoch_expected = torch.zeros(2, dtype=torch.int32, device=self.dev)
        self._allocate_dispatch_workspace(op)
        self._s1_num_cu = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self._s1_use_xcd = True
        self._s1_mega = make_stage1_autotuner(
            self._v2_dispatch_cu, self._v2_grid_mult, self._v2_tile_m_values
        )

        v = op._ll_views()
        self._s1_rx = v["rx_em"]
        self._s1_scale_i32 = v["scale_em_i32"]

        inter_dim = self.inter_dim
        a2rows = self._s1_nvm
        self._s1_a2rows = a2rows
        self._s1_out = torch.zeros((a2rows, inter_dim), dtype=torch.float8_e4m3fn, device=self.dev)
        prows = ((a2rows + 255) // 256) * 256
        pcols = (((inter_dim // 32) + 7) // 8) * 8
        self._s1_osd = torch.zeros(prows * pcols + inter_dim, dtype=torch.uint8, device=self.dev)
        self._build_v2_disp_table()
        self._build_small_fixed_stage1()

        # fmt: off
        self._s1_output = Stage1Output(a2=self._s1_out, a2_scale=self._s1_osd, sorted_token_ids=op.srcmap_em,
            sorted_expert_ids=op.sorted_expert_ids, sorted_weights=op.wts_em.view(torch.float32),
            num_valid_ids=op.num_valid, wts_buf=None)
        # fmt: on

    def _build_small_fixed_stage1(self):
        topology_tuned = _SMALL_STAGE1_TUNED_BY_LOCAL_EXPERTS.get(self.epr, {})
        default_max_tokens = topology_tuned.get("max_tokens", 64)
        self._s1_small_max_tokens = int(default_max_tokens)
        if self._s1_small_max_tokens <= 0:
            raise ValueError("small fixed-slot token threshold must be positive")
        if self.world_size * self.mtpr > 0x00FFFFFF:
            raise ValueError("small fixed-slot srcmap encoding exceeds 24-bit token field")
        op = FlyDSLDispatchGroupMajorOp(
            rank=self.rank,
            world_size=self.world_size,
            hidden_dim=self.model_dim,
            max_tok_per_rank=self._s1_small_max_tokens,
            experts_per_rank=self.epr,
            topk=self.topk,
            data_type=self._s1cfg["data_type"],
            unit_size=32,
            scale_dim=self._s1_scale_dim,
            scale_type_size=1,
            compact=False,
        )
        self._s1_small_op = op
        self._s1_small_nvm = op.num_valid_max
        self._s1_small_cap = op.ll_cap
        views = op._ll_views()
        self._s1_small_rx = views["rx_em"]
        self._s1_small_scale_i32 = views["scale_em_i32"]

        a2rows = op.max_blocks * 32
        self._s1_small_a2rows = a2rows
        self._s1_small_out = torch.zeros(
            (a2rows, self.inter_dim),
            dtype=torch.float8_e4m3fn,
            device=self.dev,
        )
        prows = ((a2rows + 255) // 256) * 256
        pcols = (((self.inter_dim // 32) + 7) // 8) * 8
        self._s1_small_osd = torch.zeros(
            prows * pcols + self.inter_dim,
            dtype=torch.uint8,
            device=self.dev,
        )

        ws = {
            "route_done": torch.zeros(1, dtype=torch.int32, device=self.dev),
            "leader_claim": torch.zeros(2, dtype=torch.int32, device=self.dev),
            "meta_ready": torch.zeros(2, dtype=torch.int32, device=self.dev),
            "epoch_parity": torch.zeros(1, dtype=torch.int32, device=self.dev),
            "epoch_expected": torch.zeros(2, dtype=torch.int32, device=self.dev),
        }
        ws["entry_done"] = op._sym((2 * self.world_size,), torch.int32)
        ws["source_done"] = op._sym((2 * self.world_size,), torch.int32)
        ms.shmem_barrier_all()
        ws["p2p_entry_done"] = op._p2p_table(ws["entry_done"])
        ws["p2p_source_done"] = op._p2p_table(ws["source_done"])
        self._s1_small_workspace = ws

        table = [0] * SMALL_FIXED_TABLE_SIZE
        table[SmallFixedSlot.RUNNING] = op.running.data_ptr()
        table[SmallFixedSlot.P2P_RUNNING] = op.p2p_running.data_ptr()
        table[SmallFixedSlot.P2P_TOKEN] = op.p2p_rx_em.data_ptr()
        table[SmallFixedSlot.P2P_SCALE] = op.p2p_scale_em.data_ptr()
        table[SmallFixedSlot.P2P_WEIGHT] = op.p2p_wts_em.data_ptr()
        table[SmallFixedSlot.P2P_SRCMAP] = op.p2p_srcmap_em.data_ptr()
        table[SmallFixedSlot.EXPERT_COUNT] = op.ll_count.data_ptr()
        table[SmallFixedSlot.SORTED_EXPERT] = op.sorted_expert_ids.data_ptr()
        table[SmallFixedSlot.TILE_ROW_BASE] = op.tile_row_base.data_ptr()
        table[SmallFixedSlot.NUM_VALID] = op.num_valid.data_ptr()
        table[SmallFixedSlot.ROUTE_DONE] = ws["route_done"].data_ptr()
        table[SmallFixedSlot.LEADER_CLAIM] = ws["leader_claim"].data_ptr()
        table[SmallFixedSlot.META_READY] = ws["meta_ready"].data_ptr()
        table[SmallFixedSlot.SOURCE_DONE] = ws["source_done"].data_ptr()
        table[SmallFixedSlot.P2P_SOURCE_DONE] = ws["p2p_source_done"].data_ptr()
        table[SmallFixedSlot.ENTRY_DONE] = ws["entry_done"].data_ptr()
        table[SmallFixedSlot.P2P_ENTRY_DONE] = ws["p2p_entry_done"].data_ptr()
        self._s1_small_disp = torch.tensor(table, dtype=torch.int64, device=self.dev)

        # fmt: off
        self._s1_small_output = Stage1Output(a2=self._s1_small_out, a2_scale=self._s1_small_osd,
            sorted_token_ids=op.srcmap_em, sorted_expert_ids=op.sorted_expert_ids,
            sorted_weights=op.wts_em.view(torch.float32), num_valid_ids=op.num_valid, wts_buf=None)
        # fmt: on

    def _run_small_fixed_stage1(self, x, wts, scales, topk_ids, stream):
        from .mega_moe_stage1 import compile_mega_moe_stage1

        topology_tuned = _SMALL_STAGE1_TUNED_BY_LOCAL_EXPERTS.get(self.epr, {})
        route_tokens = int(x.shape[0])
        buckets = topology_tuned.get("buckets", {})
        matching_buckets = [tokens for tokens in buckets if tokens >= route_tokens]
        tuned = buckets[min(matching_buckets)] if matching_buckets else {}
        num_waves = int(tuned.get("num_waves", 4))
        min_dispatch_cu = (route_tokens * self.topk + num_waves - 1) // num_waves
        min_dispatch_cu = ((min_dispatch_cu + 7) // 8) * 8
        default_dispatch_cu = max(
            min_dispatch_cu,
            tuned.get("num_dispatch_cu", min_dispatch_cu),
        )
        small_dispatch_cu = int(default_dispatch_cu)
        small_grid_mult = int(tuned.get("grid_mult", 4))
        small_tile_n = int(tuned.get("tile_n", 128))
        launch = compile_mega_moe_stage1(
            model_dim=self.model_dim,
            inter_dim=self.inter_dim,
            rank=self.rank,
            experts_per_rank=self.epr,
            fuse_npes=self.world_size,
            fuse_topk=self.topk,
            fuse_cap=self._s1_small_cap,
            fuse_mtpr=self.mtpr,
            fuse_scale_dim=self._s1_scale_dim,
            sort_block_m=32,
            tile_n=small_tile_n,
            tile_k=256,
            num_waves=num_waves,
            grid_mult=small_grid_mult,
            wgm=2,
            sched_nmajor=False,
            pipe_weights=True,
            mfma_amajor=True,
            swizzle_a=True,
            use_xcd=self._s1_use_xcd,
            use_tile_resource=True,
            waves_per_eu_hint=2,
            num_cu=self._s1_num_cu,
            num_dispatch_cu=small_dispatch_cu,
            small_fixed=True,
            small_fixed_route_tokens=route_tokens,
        )
        op = self._s1_small_op
        ws = self._s1_small_workspace
        _run_compiled(
            launch,
            self._s1_small_out,
            self._s1_small_rx,
            self._s1_w1,
            self._s1_small_scale_i32,
            self._s1_w1_scale,
            op.tile_row_base,
            op.sorted_expert_ids,
            op.num_valid,
            self._s1_small_osd,
            fx.Int32(self._s1_small_nvm),
            fx.Int64(self._s1_small_disp.data_ptr()),
            fx.Int32(int(x.shape[0])),
            fx.Int64(x.data_ptr()),
            fx.Int64(topk_ids.data_ptr()),
            fx.Int64(wts.data_ptr()),
            fx.Int64(scales.data_ptr()),
            fx.Int64(ws["epoch_parity"].data_ptr()),
            fx.Int64(ws["epoch_expected"].data_ptr()),
            stream,
        )
        self._s1_active_tile_m = 32
        return self._s1_small_output

    def _allocate_dispatch_workspace(self, op):
        total_experts = self.world_size * self.epr
        workspace = {
            "local_hist": torch.zeros(total_experts, dtype=torch.int32, device=self.dev),
            "local_cursor": torch.zeros(total_experts, dtype=torch.int32, device=self.dev),
            "pair_order": torch.empty(self.mtpr * self.topk, dtype=torch.int32, device=self.dev),
            "pair_base": torch.empty(total_experts, dtype=torch.int32, device=self.dev),
            "pair_ready": torch.zeros(2, dtype=torch.int32, device=self.dev),
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
        if cur_tok <= self._s1_small_max_tokens:
            return self._run_small_fixed_stage1(
                x,
                wts,
                scales,
                topk_ids,
                stream,
            )
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
            sort_block_m=self.sort_block_m, num_cu=self._s1_num_cu, use_xcd=self._s1_use_xcd,
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
        if s1 is self._s1_small_output:
            tile_row_base = self._s1_small_op.tile_row_base
            active_tile_m = 32
        elif s1 is self._s1_output:
            tile_row_base = self._s1_op.tile_row_base
            active_tile_m = self._s1_active_tile_m
        else:
            raise ValueError("stage1 output does not belong to this MegaMoEV2 instance")
        if active_tile_m != 32:
            raise ValueError(
                f"stage2 requires M32 metadata, got active stage1 tile M{active_tile_m}"
            )
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
            fx.Int64(tile_row_base.data_ptr()),
            fx.Int64(s1.num_valid_ids.data_ptr()),
            comb_op._fx_tis,
            comb_op._fx_p2p_comb_inp,
        )
        _run_compiled(self._g2v2_launch, *args, fx.Int32(size_expert_ids), s_fx)
        ret = comb_op.combine_no_stage1(self._g2_dummy_inp, None, None, cur_tok=run_tokens, enable_weights=False)
        return ret
