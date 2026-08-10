# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""MegaMoE v2 fused dispatch, GEMM1, GEMM2, and combine implementation."""

from dataclasses import dataclass

import mori.shmem as ms
import torch

import flydsl.expr as fx
from kernels.comm.flydsl_dispatch_combine_intranode_op import (
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
)

from .dispatch import DISPATCH_TABLE_SIZE, DispatchSlot
from .mega_moe_config import (
    FIXED_SLOT_MAX_MTPR,
    MegaMoEConfig,
    Stage1Config,
    select_mega_moe_config,
)
from .quant import (
    convert_aiter_lqq_to_megamoe,
    moe_smooth_per_token_scaled_quant,
    per_1x32_mx_quant,
    repack_megamoe_lqq_for_int8_loader,
    smooth_per_token_scaled_quant,
)

__all__ = ["MegaMoEV2"]

_SUPPORTED_QUANT_MODES = ("a8w4", "a8w4smooth", "w8a8smooth")
_SUPPORTED_WEIGHT_FORMATS = ("megamoe", "aiter_lqq")


def _dispatch_quant_config(quant: str, model_dim: int):
    """Return the dispatch dtype and scale ABI for a MegaMoEV2 quant mode."""
    if quant not in _SUPPORTED_QUANT_MODES:
        raise ValueError(
            f"unsupported quant={quant!r}; expected one of {_SUPPORTED_QUANT_MODES}"
        )
    if quant == "a8w4":
        return torch.float8_e4m3fn, int(model_dim) // 32, 1
    return torch.int8, 1, 4


def _as_packed_i32(tensor):
    tensor = tensor.contiguous()
    return tensor if tensor.dtype == torch.int32 else tensor.view(torch.int32)


@dataclass(slots=True)
class Int8Stage1Output:
    """Stable Stage1 buffers consumed by host requant and the future Stage2 ABI."""

    a2: torch.Tensor
    sorted_token_ids: torch.Tensor
    sorted_expert_ids: torch.Tensor
    sorted_weights: torch.Tensor
    num_valid_ids: torch.Tensor
    sort_block_m: int


class MegaMoEV2:
    """Fused dispatch, GEMM1, GEMM2, and combine with one in-flight launch per instance."""

    # fmt: off
    def __init__(self, *, rank: int, world_size: int, model_dim: int, inter_dim: int, experts: int, topk: int,
        quant: str, w1: torch.Tensor, w1_scale: torch.Tensor, w2: torch.Tensor, w2_scale: torch.Tensor,
        max_tok_per_rank: int, mega_scheme: str = "fixedslot", swiglu_limit: float = 0.0,
        w1_lqq_scale: torch.Tensor | None = None, w1_lqq_zero: torch.Tensor | None = None,
        w2_lqq_scale: torch.Tensor | None = None, w2_lqq_zero: torch.Tensor | None = None,
        fc1_smooth_scale: torch.Tensor | None = None, fc2_smooth_scale: torch.Tensor | None = None,
        weight_format: str = "megamoe"):
    # fmt: on
        dispatch_dtype, scale_dim, scale_type_size = _dispatch_quant_config(quant, model_dim)
        if weight_format not in _SUPPORTED_WEIGHT_FORMATS:
            raise ValueError(
                f"unsupported weight_format={weight_format!r}; "
                f"expected one of {_SUPPORTED_WEIGHT_FORMATS}"
            )
        self.quant = quant
        is_a8w4smooth = quant == "a8w4smooth"
        self._is_int8_smooth = quant in ("a8w4smooth", "w8a8smooth")
        if weight_format == "aiter_lqq":
            if not is_a8w4smooth:
                raise ValueError(
                    "weight_format='aiter_lqq' is only valid for quant='a8w4smooth'"
                )
            if any(
                tensor is None
                for tensor in (w1_lqq_scale, w1_lqq_zero, w2_lqq_scale, w2_lqq_zero)
            ):
                raise ValueError(
                    "weight_format='aiter_lqq' requires w1/w2 LQQ scale and zero tensors"
                )
            w1, w1_lqq_scale, w1_lqq_zero = convert_aiter_lqq_to_megamoe(
                w1, w1_lqq_scale, w1_lqq_zero
            )
            w2, w2_lqq_scale, w2_lqq_zero = convert_aiter_lqq_to_megamoe(
                w2, w2_lqq_scale, w2_lqq_zero
            )
        if is_a8w4smooth:
            # Public/checkpoint MegaMoE weights retain the legacy packed
            # layout; both INT8 GEMMs consume direct K64-pair loader order.
            w1 = repack_megamoe_lqq_for_int8_loader(
                w1, 2 * int(inter_dim), int(model_dim)
            )
            w2 = repack_megamoe_lqq_for_int8_loader(
                w2, int(model_dim), int(inter_dim)
            )
        if self._is_int8_smooth:
            if fc1_smooth_scale is None or fc2_smooth_scale is None:
                raise ValueError(
                    f"quant={quant!r} requires fc1_smooth_scale and fc2_smooth_scale"
                )
            if is_a8w4smooth and any(
                tensor is None
                for tensor in (w1_lqq_scale, w1_lqq_zero, w2_lqq_scale, w2_lqq_zero)
            ):
                raise ValueError(
                    "quant='a8w4smooth' requires w1/w2_lqq_scale and w1/w2_lqq_zero"
                )
        if experts % world_size != 0:
            raise ValueError(f"experts={experts} must be divisible by world_size={world_size}")
        if max_tok_per_rank <= 0 or max_tok_per_rank & (max_tok_per_rank - 1):
            raise ValueError(f"max_tok_per_rank={max_tok_per_rank} must be a power of two")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.model_dim = int(model_dim)
        self.inter_dim = int(inter_dim)
        self.experts = int(experts)
        self.epr = int(experts // world_size)
        self.topk = int(topk)
        self.mtpr = int(max_tok_per_rank)
        self.swiglu_limit = float(swiglu_limit)
        if self.swiglu_limit < 0:
            raise ValueError("swiglu_limit must be non-negative")
        self.dev = torch.device("cuda", rank)
        self.max_recv = self.world_size * self.mtpr
        compact = self.mtpr > FIXED_SLOT_MAX_MTPR
        capacity_tile_m = 128 if compact else 32
        self._s1_fixed_slot = not compact
        self._s1_scale_dim = scale_dim
        # fmt: off
        self.comb_cfg = FlyDSLDispatchCombineConfig(rank=self.rank, world_size=self.world_size,
            hidden_dim=self.model_dim, max_num_inp_token_per_rank=self.mtpr, num_experts_per_rank=self.epr,
            num_experts_per_token=self.topk, combine_dtype=torch.bfloat16,
            dispatch_dtype=dispatch_dtype, scale_dim=self._s1_scale_dim, scale_type_size=scale_type_size,
            enable_std_moe=False, enable_group_major=True, gm_unit_size=capacity_tile_m,
            gm_scheme=mega_scheme, gm_compact=compact, max_total_recv_tokens=self.world_size)
        # fmt: on
        self.comb_op = FlyDSLDispatchCombineIntraNodeOp(self.comb_cfg)
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        self.w2 = w2 if w2.is_contiguous() else w2.contiguous()
        self.w2_scale = w2_scale if w2_scale.is_contiguous() else w2_scale.contiguous()
        if self._is_int8_smooth:
            self._build_int8_host(
                w1,
                w1_scale,
                w1_lqq_scale=w1_lqq_scale,
                w1_lqq_zero=w1_lqq_zero,
                w2_lqq_scale=w2_lqq_scale,
                w2_lqq_zero=w2_lqq_zero,
                fc1_smooth_scale=fc1_smooth_scale,
                fc2_smooth_scale=fc2_smooth_scale,
            )
        else:
            # Keep the established A8W4 build path unchanged.
            self._build_fused_stage1(w1, w1_scale)
            self._build_fused_stage2()

    def _build_int8_host(
        self,
        w1,
        w1_scale,
        *,
        w1_lqq_scale,
        w1_lqq_zero,
        w2_lqq_scale,
        w2_lqq_zero,
        fc1_smooth_scale,
        fc2_smooth_scale,
    ):
        """Prepare the Stage-1/2 INT8 host contract without fusing quant kernels."""
        op = self.comb_op._gm
        assert op is not None, "combine op was built without enable_group_major"
        self._s1_op = op
        self._int8_w1 = w1.contiguous()
        self._int8_w1_scale = w1_scale.contiguous()
        self._int8_w1_lqq_scale = (
            _as_packed_i32(w1_lqq_scale) if w1_lqq_scale is not None else None
        )
        self._int8_w1_lqq_zero = (
            _as_packed_i32(w1_lqq_zero) if w1_lqq_zero is not None else None
        )
        self._int8_w2_lqq_scale = (
            _as_packed_i32(w2_lqq_scale) if w2_lqq_scale is not None else None
        )
        self._int8_w2_lqq_zero = (
            _as_packed_i32(w2_lqq_zero) if w2_lqq_zero is not None else None
        )
        self._int8_fc1_smooth = fc1_smooth_scale.to(torch.float32).contiguous()
        local_begin = self.rank * self.epr
        self._int8_fc2_smooth = (
            fc2_smooth_scale.to(torch.float32)[local_begin : local_begin + self.epr]
            .contiguous()
        )

        # Front quant is token/slot-major because each route selects expert-specific
        # FC1 smooth scales. Requant uses the ATOM row contract over all received rows.
        front_rows = self.mtpr * self.topk
        requant_rows = self.max_recv * self.topk
        self._int8_front_q = torch.zeros(
            (front_rows, self.model_dim), dtype=torch.int8, device=self.dev
        )
        self._int8_front_scale = torch.zeros(
            (front_rows,), dtype=torch.float32, device=self.dev
        )
        self._int8_requant_q = torch.zeros(
            (requant_rows, self.inter_dim), dtype=torch.int8, device=self.dev
        )
        self._int8_requant_scale = torch.zeros(
            (requant_rows,), dtype=torch.float32, device=self.dev
        )
        from .mega_moe_stage1 import run_mega_moe_stage1

        self.sort_block_m = 32
        metadata_blocks = (op.num_valid_max + self.sort_block_m - 1) // self.sort_block_m
        if metadata_blocks > op.max_blocks:
            op.max_blocks = metadata_blocks
            op.sorted_expert_ids = torch.zeros(
                metadata_blocks, dtype=torch.int32, device=self.dev
            )
            op.tile_row_base = torch.zeros(
                metadata_blocks, dtype=torch.int32, device=self.dev
            )
        self._s1_nvm = op.num_valid_max
        self._s1_cap = op.ll_cap
        self._s1_epoch_parity = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self._s1_epoch_expected = torch.zeros(2, dtype=torch.int32, device=self.dev)
        self._s1_num_cu = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        self._allocate_dispatch_workspace(op, metadata_blocks)
        self._build_v2_disp_table()
        views = op._ll_views()
        self._s1_rx = views["rx_em"]
        self._s1_scale_i32 = views["scale_em_i32"]
        # Fixed-slot padding is consumed by the compact INT8 requant/Stage2
        # metadata path. Seed every row with the invalid source token so the
        # first graph capture cannot observe recycled symmetric-heap contents.
        op.srcmap_em.fill_(self.world_size * self.mtpr)
        ms.shmem_barrier_all()
        atom_rows = self.max_recv * self.topk
        self._int8_a2 = torch.zeros(
            (atom_rows, self.inter_dim), dtype=torch.float16, device=self.dev
        )
        self._int8_sorted_tokens = torch.zeros(
            self._s1_nvm, dtype=torch.int32, device=self.dev
        )
        self._int8_sorted_experts = torch.zeros(
            metadata_blocks, dtype=torch.int32, device=self.dev
        )
        self._int8_sorted_weights = torch.zeros(
            self._s1_nvm, dtype=torch.float32, device=self.dev
        )
        self._int8_dummy_out_scale = torch.empty(
            1, dtype=torch.uint8, device=self.dev
        )
        self._int8_stage1_output = Int8Stage1Output(
            a2=self._int8_a2,
            sorted_token_ids=self._int8_sorted_tokens,
            sorted_expert_ids=self._int8_sorted_experts,
            sorted_weights=self._int8_sorted_weights,
            num_valid_ids=op.num_valid,
            sort_block_m=self.sort_block_m,
        )
        self._int8_stage1 = run_mega_moe_stage1
        self._build_fused_stage2()

    def _build_fused_stage1(self, w1, w1_scale):
        from .mega_moe_stage1 import run_mega_moe_stage1

        self.sort_block_m = 32
        self._s1_w1 = w1.contiguous()
        self._s1_w1_scale = w1_scale.contiguous()
        op = self.comb_op._gm
        assert op is not None, "combine op was built without enable_group_major"
        self._s1_op = op
        # Payload capacity follows the largest SBM; metadata covers the smallest candidate.
        metadata_blocks = (op.num_valid_max + self.sort_block_m - 1) // self.sort_block_m
        if metadata_blocks > op.max_blocks:
            op.max_blocks = metadata_blocks
            op.sorted_expert_ids = torch.zeros(metadata_blocks, dtype=torch.int32, device=self.dev)
            op.tile_row_base = torch.zeros(metadata_blocks, dtype=torch.int32, device=self.dev)
        self._s1_nvm = op.num_valid_max
        self._s1_cap = op.ll_cap
        self._s1_epoch_parity = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self._s1_epoch_expected = torch.zeros(2, dtype=torch.int32, device=self.dev)
        self._s1_num_cu = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self._allocate_dispatch_workspace(op, metadata_blocks)
        self._s1_mega = run_mega_moe_stage1

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

    def _allocate_dispatch_workspace(self, op, metadata_blocks):
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
            "expert_tile_end": torch.empty(self.epr, dtype=torch.int32, device=self.dev),
            "max_expert_tiles": torch.zeros(1, dtype=torch.int32, device=self.dev),
            "payload_chunk_done": torch.zeros(total_experts, dtype=torch.int32, device=self.dev),
            "tile_expected": torch.zeros(metadata_blocks, dtype=torch.int32, device=self.dev),
            "active_payload_blocks": torch.zeros(1, dtype=torch.int32, device=self.dev),
            "payload_blocks_per_destination": torch.zeros(self.world_size, dtype=torch.int32, device=self.dev),
            "payload_chunks_per_destination": torch.zeros(self.world_size, dtype=torch.int32, device=self.dev),
            "group_done": torch.zeros(1, dtype=torch.int32, device=self.dev),
            "dest_counter": torch.zeros(self.world_size, dtype=torch.int32, device=self.dev),
        }
        workspace["bigcnt"] = op._sym((self.world_size * self.epr,), torch.int32)
        workspace["count_done"] = op._sym((2 * self.world_size,), torch.int32)
        workspace["my_base"] = op._sym((total_experts,), torch.int32)
        workspace["plan_ready"] = op._sym((2 * self.world_size,), torch.int32)
        workspace["payload_ready"] = op._sym((2 * self.epr,), torch.int32)
        workspace["launch_ready"] = op._sym((self.world_size,), torch.int32)
        workspace["tile_ready"] = op._sym((metadata_blocks,), torch.int32)
        workspace["payload_ready_rows"] = op._sym((1,), torch.int32)
        workspace["recv_num"] = op._sym((self.world_size,), torch.int32)
        workspace["recv_num"].fill_(-1)
        ms.shmem_barrier_all()
        workspace["p2p_bigcnt"] = op._p2p_table(workspace["bigcnt"])
        workspace["p2p_count_done"] = op._p2p_table(workspace["count_done"])
        workspace["p2p_my_base"] = op._p2p_table(workspace["my_base"])
        workspace["p2p_plan_ready"] = op._p2p_table(workspace["plan_ready"])
        workspace["p2p_payload_ready"] = op._p2p_table(workspace["payload_ready"])
        workspace["p2p_launch_ready"] = op._p2p_table(workspace["launch_ready"])
        workspace["p2p_tile_ready"] = op._p2p_table(workspace["tile_ready"])
        workspace["p2p_payload_ready_rows"] = op._p2p_table(workspace["payload_ready_rows"])
        workspace["p2p_recv_num"] = op._p2p_table(workspace["recv_num"])
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
        table[DispatchSlot.PAIR_ORDER_READY] = workspace["pair_order_ready"].data_ptr()
        table[DispatchSlot.WORK_HEAD] = workspace["work_head"].data_ptr()
        table[DispatchSlot.WORK_TAIL] = workspace["work_tail"].data_ptr()
        table[DispatchSlot.EXPERT_TILE_END] = workspace["expert_tile_end"].data_ptr()
        table[DispatchSlot.GROUP_DONE] = workspace["group_done"].data_ptr()
        table[DispatchSlot.RUNNING] = op.running.data_ptr()
        table[DispatchSlot.P2P_RUNNING] = op.p2p_running.data_ptr()
        table[DispatchSlot.LAUNCH_READY] = workspace["launch_ready"].data_ptr()
        table[DispatchSlot.P2P_LAUNCH_READY] = workspace["p2p_launch_ready"].data_ptr()
        table[DispatchSlot.MAX_EXPERT_TILES] = workspace["max_expert_tiles"].data_ptr()
        table[DispatchSlot.PAYLOAD_CHUNK_DONE] = workspace["payload_chunk_done"].data_ptr()
        table[DispatchSlot.TILE_READY] = workspace["tile_ready"].data_ptr()
        table[DispatchSlot.P2P_TILE_READY] = workspace["p2p_tile_ready"].data_ptr()
        table[DispatchSlot.TILE_EXPECTED] = workspace["tile_expected"].data_ptr()
        table[DispatchSlot.ACTIVE_PAYLOAD_BLOCKS] = workspace["active_payload_blocks"].data_ptr()
        table[DispatchSlot.PAYLOAD_READY_ROWS] = workspace["payload_ready_rows"].data_ptr()
        table[DispatchSlot.P2P_PAYLOAD_READY_ROWS] = workspace["p2p_payload_ready_rows"].data_ptr()
        table[DispatchSlot.PAYLOAD_BLOCKS_PER_DESTINATION] = workspace[
            "payload_blocks_per_destination"
        ].data_ptr()
        table[DispatchSlot.PAYLOAD_CHUNKS_PER_DESTINATION] = workspace[
            "payload_chunks_per_destination"
        ].data_ptr()
        table[DispatchSlot.TOTAL_RECV] = op.total_recv.data_ptr()
        table[DispatchSlot.DEST_COUNTER] = workspace["dest_counter"].data_ptr()
        table[DispatchSlot.RECV_NUM] = workspace["recv_num"].data_ptr()
        table[DispatchSlot.P2P_RECV_NUM] = workspace["p2p_recv_num"].data_ptr()
        self._s1_disp = torch.tensor(table, dtype=torch.int64, device=self.dev)

    def _select_config(self, tokens: int) -> MegaMoEConfig:
        config = select_mega_moe_config(
            tokens,
            self.mtpr,
            experts_per_rank=self.epr,
            model_dim=self.model_dim,
            inter_dim=self.inter_dim,
            quant_mode=self.quant,
        )
        self._active_config = config
        return config

    def _run_fused_stage1(self, x, wts, scales, topk_ids, stream=None, config: Stage1Config | None = None):
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
        if config is None:
            config = self._select_config(cur_tok).stage1
        op = self._s1_op
        # fmt: off
        self._s1_mega(
            self._s1_out, self._s1_rx, self._s1_w1, self._s1_scale_i32, self._s1_w1_scale,
            op.tile_row_base, op.sorted_expert_ids, op.num_valid, self._s1_osd, fx.Int32(self._s1_nvm),
            fx.Int64(self._s1_disp.data_ptr()), fx.Int32(cur_tok), fx.Int64(x.data_ptr()),
            fx.Int64(topk_ids.data_ptr()), fx.Int64(wts.data_ptr()), fx.Int64(scales.data_ptr()),
            fx.Int64(self._s1_epoch_parity.data_ptr()), fx.Int64(self._s1_epoch_expected.data_ptr()),
            stream, model_dim=self.model_dim, inter_dim=self.inter_dim, rank=self.rank,
            experts_per_rank=self.epr, fuse_npes=self.world_size, fuse_topk=self.topk,
            fuse_cap=self._s1_cap, fuse_mtpr=self.mtpr, fuse_scale_dim=self._s1_scale_dim,
            fixed_slot_dispatch=self._s1_fixed_slot, num_cu=self._s1_num_cu,
            sort_block_m=config.sort_block_m, tile_n=config.tile_n, tile_k=config.tile_k,
            num_waves=config.num_waves, grid_mult=config.grid_mult, pipe_weights=config.pipe_weights,
            mfma_amajor=config.mfma_amajor, swizzle_a=config.swizzle_a,
            async_a_copy=config.async_a_copy, num_dispatch_cu=config.num_dispatch_cu,
            use_tile_resource=config.use_tile_resource,
            waves_per_eu_hint=config.waves_per_eu_hint, b_nt=config.b_nt,
            work_shards=config.work_shards, external_grouping=config.external_grouping,
            external_counting=config.external_counting, payload_chunk_rows=config.payload_chunk_rows,
            payload_tile_ready=config.payload_tile_ready,
            swiglu_limit=self.swiglu_limit)
        # fmt: on
        self._s1_active_tile_m = config.sort_block_m
        return self._s1_active_tile_m

    def quantize(self, x_bf16):
        if self._is_int8_smooth:
            raise NotImplementedError(
                "INT8 smooth modes use route-aware front quantization inside forward(); "
                "quantize() remains the A8W4 MXFP8 helper"
            )
        return per_1x32_mx_quant(x_bf16, quant_mode="fp8")

    def _run_joint(self, x, scales, wts, topk_ids, run_tokens, stream, slice_output):
        config = self._select_config(run_tokens)
        self._run_fused_stage1(x, wts, scales, topk_ids, stream=stream, config=config.stage1)
        return self._run_stage2(run_tokens, stream, slice_output, config)

    def _run_stage2(self, run_tokens, stream, slice_output, config: MegaMoEConfig):
        ret = self._run_fused_stage2(run_tokens, config, stream)
        out_tok = ret[0] if isinstance(ret, (tuple, list)) else ret
        if out_tok is None:
            cfg = self.comb_cfg
            out_tok = (
                self.comb_op.shmem_comb_out_tok.view(torch.int8)[: self.mtpr * cfg.combine_token_bytes]
                .view(cfg.combine_dtype)
                .view(self.mtpr, cfg.combine_token_view_dim)
            )
        return out_tok[:run_tokens] if slice_output else out_tok

    def _run_int8_front_quant(self, x_bf16, topk_ids):
        """Run the standalone AITER front quant into preallocated route-major buffers."""
        run_tokens = int(x_bf16.shape[0])
        out = self._int8_front_q[: run_tokens * self.topk].view(
            run_tokens, self.topk, self.model_dim
        )
        scales = self._int8_front_scale[: run_tokens * self.topk].view(
            run_tokens, self.topk, 1
        )
        smooth_per_token_scaled_quant(
            out,
            x_bf16.view(run_tokens, 1, self.model_dim).expand(
                run_tokens, self.topk, self.model_dim
            ),
            scales,
            self._int8_fc1_smooth,
            topk_ids,
            smooth_scale_map_hash=None,
            enable_ps=True,
        )
        return out, scales

    def _run_int8_stage1(self, front_q, front_scale, wts, topk_ids, *, stream=None):
        """Dispatch per-route INT8 rows and run gfx950 INT8 GEMM1."""
        if stream is None:
            stream = fx.Stream(torch.cuda.current_stream())
        cur_tok = int(front_q.shape[0])
        if front_q.dtype != torch.int8 or not front_q.is_contiguous():
            raise ValueError("front_q must be contiguous int8")
        if tuple(front_q.shape) != (cur_tok, self.topk, self.model_dim):
            raise ValueError(
                f"front_q must have shape ({cur_tok}, {self.topk}, {self.model_dim})"
            )
        if front_scale.dtype != torch.float32 or not front_scale.is_contiguous():
            raise ValueError("front_scale must be contiguous float32")
        if tuple(front_scale.shape) != (cur_tok, self.topk, 1):
            raise ValueError(
                f"front_scale must have shape ({cur_tok}, {self.topk}, 1)"
            )
        config = self._select_config(cur_tok).stage1
        qscale = (
            self._int8_w1_lqq_scale
            if self._int8_w1_lqq_scale is not None
            else self._int8_w1_scale
        )
        qzero = (
            self._int8_w1_lqq_zero
            if self._int8_w1_lqq_zero is not None
            else self._int8_w1_scale
        )
        self._int8_stage1(
            self._int8_a2,
            self._s1_rx,
            self._int8_w1,
            self._s1_scale_i32,
            self._int8_w1_scale,
            self._s1_op.tile_row_base,
            self._s1_op.sorted_expert_ids,
            self._s1_op.num_valid,
            self._int8_dummy_out_scale,
            fx.Int32(self._s1_nvm),
            fx.Int64(self._s1_disp.data_ptr()),
            fx.Int32(cur_tok),
            fx.Int64(front_q.data_ptr()),
            fx.Int64(topk_ids.data_ptr()),
            fx.Int64(wts.data_ptr()),
            fx.Int64(front_scale.data_ptr()),
            fx.Int64(self._s1_epoch_parity.data_ptr()),
            fx.Int64(self._s1_epoch_expected.data_ptr()),
            stream,
            model_dim=self.model_dim,
            inter_dim=self.inter_dim,
            rank=self.rank,
            experts_per_rank=self.epr,
            fuse_npes=self.world_size,
            fuse_topk=self.topk,
            fuse_cap=self._s1_cap,
            fuse_mtpr=self.mtpr,
            fuse_scale_dim=4,
            fixed_slot_dispatch=self._s1_fixed_slot,
            num_cu=self._s1_num_cu,
            sort_block_m=config.sort_block_m,
            tile_n=config.tile_n,
            tile_k=256,
            num_waves=config.num_waves,
            grid_mult=config.grid_mult,
            pipe_weights=False,
            mfma_amajor=False,
            swizzle_a=config.swizzle_a,
            async_a_copy=False,
            num_dispatch_cu=config.num_dispatch_cu,
            use_tile_resource=False,
            waves_per_eu_hint=config.waves_per_eu_hint,
            b_nt=config.b_nt,
            work_shards=config.work_shards,
            external_grouping=config.external_grouping,
            external_counting=config.external_counting,
            payload_chunk_rows=config.payload_chunk_rows,
            payload_tile_ready=config.payload_tile_ready,
            swiglu_limit=self.swiglu_limit,
            quant_mode=self.quant,
            compact_src=self._int8_sorted_tokens,
            compact_experts=self._int8_sorted_experts,
            compact_weights=self._int8_sorted_weights,
            qscale_w=qscale,
            qzero_w=qzero,
        )
        self._int8_stage1_output.sort_block_m = config.sort_block_m
        return self._int8_stage1_output

    def _run_int8_requant(self, stage1_output, topk_ids):
        """Requantize Stage-1 output into the preallocated INT8 GEMM2 input."""
        required = (
            "a2",
            "sorted_token_ids",
            "sorted_expert_ids",
            "num_valid_ids",
            "sort_block_m",
        )
        missing = [name for name in required if not hasattr(stage1_output, name)]
        if missing:
            raise NotImplementedError(
                "MegaMoEV2 INT8 requant requires Stage-1 metadata fields: "
                + ", ".join(missing)
            )
        rows = self.max_recv * self.topk
        requant_q = self._int8_requant_q.view(
            self.max_recv, self.topk, self.inter_dim
        )
        requant_scale = self._int8_requant_scale
        moe_smooth_per_token_scaled_quant(
            requant_q,
            stage1_output.a2.view(self.max_recv, self.topk, self.inter_dim),
            requant_scale,
            self._int8_fc2_smooth,
            topk_ids,
            stage1_output.sorted_token_ids,
            stage1_output.sorted_expert_ids,
            stage1_output.num_valid_ids,
            int(stage1_output.sort_block_m),
            None,
            False,
            False,
        )
        return requant_q.view(rows, self.inter_dim), requant_scale

    def _run_int8_stage2(self, requant_q, requant_scale, stage1_output, run_tokens, stream, slice_output):
        """Run gfx950 K64 INT8 GEMM2 and the existing weighted P2P combine."""
        if stream is None:
            stream = torch.cuda.current_stream()
        s_fx = fx.Stream(stream.cuda_stream)
        config = self._active_config
        stage2 = config.stage2
        invariants = self._g2_invariants_by_quant["none"]
        self._g2_run(
            fx.Int64(requant_q.data_ptr()),
            fx.Int64(requant_scale.data_ptr()),
            fx.Int64(self.w2.data_ptr()),
            fx.Int64(self.w2_scale.data_ptr()),
            fx.Int64(stage1_output.sorted_expert_ids.data_ptr()),
            fx.Int64(stage1_output.num_valid_ids.data_ptr()),
            fx.Int64(
                self._s1_dispatch_workspace["max_expert_tiles"].data_ptr()
            ),
            fx.Int64(stage1_output.sorted_token_ids.data_ptr()),
            fx.Int64(stage1_output.sorted_weights.data_ptr()),
            fx.Int64(self._s1_op.tile_row_base.data_ptr()),
            self.comb_op._fx_p2p_comb_inp,
            self._s1_nvm,
            self.inter_dim,
            self.model_dim,
            s_fx,
            BM=32,
            SBM=int(stage1_output.sort_block_m),
            BN=128,
            BK=256,
            use_nt=stage2.use_nt,
            g2_bhoist=False,
            g2_ascale_pf=False,
            g2_spart=stage2.spatial_partition,
            persist=stage2.persist,
            persist_cu=stage2.persist_cu,
            persist_strided=stage2.persist_strided,
            skew_cu=stage2.skew_cu,
            g2_bf16_lds=False,
            quant_mode=self.quant,
            qscale_w=self._int8_w2_lqq_scale,
            qzero_w=self._int8_w2_lqq_zero,
            **invariants,
        )
        self._g2_active_block_m = 32
        ret = self.comb_op.combine_no_stage1(
            self._g2_combine_placeholder,
            None,
            None,
            cur_tok=run_tokens,
            enable_weights=False,
            stage2_p2p_quant="none",
        )
        out_tok = ret[0] if isinstance(ret, (tuple, list)) else ret
        if out_tok is None:
            cfg = self.comb_cfg
            out_tok = (
                self.comb_op.shmem_comb_out_tok.view(torch.int8)[
                    : self.mtpr * cfg.combine_token_bytes
                ]
                .view(cfg.combine_dtype)
                .view(self.mtpr, cfg.combine_token_view_dim)
            )
        return out_tok[:run_tokens] if slice_output else out_tok

    def _forward_int8(self, x_bf16, wts, topk_ids, *, stream=None, slice_output=True):
        """Host-visible INT8 pipeline: front quant -> Stage1 -> requant -> Stage2."""
        front_q, front_scale = self._run_int8_front_quant(x_bf16, topk_ids)
        stage1_output = self._run_int8_stage1(
            front_q, front_scale, wts, topk_ids, stream=stream
        )
        requant_q, requant_scale = self._run_int8_requant(stage1_output, topk_ids)
        return self._run_int8_stage2(
            requant_q,
            requant_scale,
            stage1_output,
            int(x_bf16.shape[0]),
            stream,
            slice_output,
        )

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
        if self._is_int8_smooth:
            return self._forward_int8(
                x_bf16[:run_tokens],
                wts,
                topk_ids,
                stream=stream,
                slice_output=slice_output,
            )
        x_q, scales = self.quantize(x_bf16)
        return self._run_joint(x_q, scales, wts, topk_ids, run_tokens, stream, slice_output)

    def forward_prequant(self, x_q, scales, wts, topk_ids, *, stream=None, slice_output=True):
        if self._is_int8_smooth:
            raise NotImplementedError(
                "forward_prequant is not defined for route-aware INT8 smooth modes; "
                "use forward(x_bf16, wts, topk_ids)"
            )
        run_tokens = int(x_q.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        return self._run_joint(x_q, scales, wts, topk_ids, run_tokens, stream, slice_output)

    forward_bf16 = forward
    __call__ = forward

    def _build_fused_stage2(self):
        from .mega_moe_stage2 import run_mega_moe_stage2

        FlyDSLDispatchCombineIntraNodeOp._ENABLE_COMBINE_NO_STAGE1 = True
        comb_cfg = self.comb_cfg
        dev = torch.device("cuda", comb_cfg.rank)
        k = comb_cfg.num_experts_per_token
        cu_num = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self._g2v2_inter = int(self.inter_dim)
        self._g2v2_hidden = int(comb_cfg.hidden_dim)
        self._g2_run = run_mega_moe_stage2
        self._g2_invariants_by_quant = {}
        for p2p_quant in ("none", "fp8_blockwise_1x32"):
            p2p_row_nbytes = (
                int(comb_cfg.hidden_dim) + int(comb_cfg.hidden_dim) // 32
                if p2p_quant == "fp8_blockwise_1x32"
                else int(comb_cfg.hidden_dim) * 2
            )
            self._g2_invariants_by_quant[p2p_quant] = {
                "model_dim": int(comb_cfg.hidden_dim), "inter_dim": int(self.inter_dim),
                "experts": int(comb_cfg.num_experts_per_rank), "topk": int(k), "rank": int(comb_cfg.rank),
                "npes": int(comb_cfg.world_size), "max_tok": int(comb_cfg.max_num_inp_token_per_rank),
                "recv_cap": int(self.max_recv),
                "comb_inp_nbytes": int(comb_cfg.max_num_inp_token_per_rank) * int(k) * p2p_row_nbytes,
                "HIDDEN_MAX": int(comb_cfg.hidden_dim), "INTER_MAX": int(self.inter_dim), "cu_num": int(cu_num),
                "p2p_quant_type": p2p_quant, "fixed_slot_dispatch": bool(self._s1_fixed_slot),
            }
        self._g2_combine_placeholder = torch.empty(
            1, comb_cfg.hidden_dim, dtype=comb_cfg.combine_dtype, device=dev
        )

    def _run_fused_stage2(self, run_tokens, config: MegaMoEConfig, stream=None):
        comb_op = self.comb_op
        op = self._s1_op
        if stream is None:
            stream = torch.cuda.current_stream()
        s_fx = fx.Stream(stream.cuda_stream)
        stage2 = config.stage2
        p2p_quant = config.p2p_quant
        invariants = self._g2_invariants_by_quant[p2p_quant]
        # fmt: off
        self._g2_run(
            fx.Int64(self._s1_out.view(-1).data_ptr()), fx.Int64(self._s1_osd.data_ptr()),
            fx.Int64(self.w2.data_ptr()), fx.Int64(self.w2_scale.data_ptr()),
            fx.Int64(op.sorted_expert_ids.data_ptr()), fx.Int64(op.num_valid.data_ptr()),
            fx.Int64(self._s1_dispatch_workspace["max_expert_tiles"].data_ptr()),
            fx.Int64(op.srcmap_em.data_ptr()), fx.Int64(op.wts_em.data_ptr()),
            fx.Int64(op.tile_row_base.data_ptr()), comb_op._fx_p2p_comb_inp, self._s1_nvm,
            self._g2v2_inter, self._g2v2_hidden, s_fx, BM=stage2.block_m,
            SBM=config.stage1.sort_block_m, BN=stage2.block_n, BK=stage2.block_k,
            use_nt=stage2.use_nt, g2_bhoist=stage2.b_hoist,
            g2_ascale_pf=stage2.ascale_prefetch, g2_spart=stage2.spatial_partition,
            persist=stage2.persist, persist_cu=stage2.persist_cu,
            persist_strided=stage2.persist_strided, skew_cu=stage2.skew_cu,
            g2_bf16_lds=stage2.bf16_lds, **invariants)
        # fmt: on
        self._g2_active_block_m = stage2.block_m
        return comb_op.combine_no_stage1(
            self._g2_combine_placeholder, None, None, cur_tok=run_tokens, enable_weights=False,
            stage2_p2p_quant=p2p_quant,
        )
