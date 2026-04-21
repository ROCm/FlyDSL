"""FlyDSL-backed InterNodeV1LL dispatch path (phase1 + phase2)."""
from __future__ import annotations

from dataclasses import dataclass

import flydsl.expr as fx
import torch
import mori.shmem as ms
from mori.shmem import mori_shmem_create_tensor

from mori import cpp as mori_cpp
from mori.ops.dispatch_combine import (
    EpDispatchCombineConfig,
    EpDispatchCombineQuantType,
    _normalize_quant_type,
)

from .dispatch_combine_internode_v1ll_copy_staging import (
    make_ep_dispatch_copy_to_staging_jit,
)
from .dispatch_combine_intranode_kernel import make_dispatch_jit


@dataclass
class FlyDSLInterNodeV1LLDispatchContext:
    """Runtime pointers required by the dispatch engine."""

    weight_ptr: int
    scale_ptr: int
    staging_ptr: int
    total_ptr: int


class FlyDSLInterNodeV1LLDispatchEngine:
    """
    Dispatch engine for InterNodeV1LL.

    - Phase1: optional FlyDSL copy-to-staging.
    - Phase2: FlyDSL dispatch routing (no mori HIP phase2 kernel launch).
    """

    def __init__(
        self,
        *,
        config: EpDispatchCombineConfig,
        handle_info: dict,
        data_type: torch.dtype,
        use_flydsl_copy_staging: bool,
        handle,
        dispatch_out_ptrs: tuple[int, int, int, int, int],
        get_dispatch_src_token_pos_fn,
    ):
        self._config = config
        self._handle = handle
        self._use_copy = use_flydsl_copy_staging
        self._copy_jit = None
        self._scale_bytes = config.scale_dim * config.scale_type_size
        self._dispatch_out_ptrs = dispatch_out_ptrs
        self._get_dispatch_src_token_pos_fn = get_dispatch_src_token_pos_fn
        self._get_staging_ptr_fn = getattr(
            mori_cpp, "get_dispatch_combine_staging_ptr", None
        )
        self._dev = torch.device("cuda", config.rank)
        self._max_recv = config.world_size * config.max_num_inp_token_per_rank

        # Auxiliary routing/state buffers for FlyDSL phase2.
        self._tok_off = mori_shmem_create_tensor((1,), torch.int32)
        self._recv_num = mori_shmem_create_tensor((config.world_size,), torch.int32)
        self._dest_ctr = torch.zeros(config.world_size, dtype=torch.int32, device=self._dev)
        self._disp_bar = torch.zeros(1, dtype=torch.int32, device=self._dev)
        self._tok_map = torch.full(
            (config.max_num_inp_token_per_rank * config.num_experts_per_token,),
            config.world_size * self._max_recv,
            dtype=torch.int32,
            device=self._dev,
        )
        self._std_disp_tok_map = torch.full(
            (self._max_recv * config.num_experts_per_token,),
            -1,
            dtype=torch.int64,
            device=self._dev,
        )
        self._std_disp_grid_bar = torch.zeros(1, dtype=torch.int32, device=self._dev)
        self._src_tok_pos_ptr, _ = self._get_dispatch_src_token_pos_fn(self._handle)

        out_ptr, outW_ptr, outS_ptr, outI_ptr, _ = self._dispatch_out_ptrs
        r = config.rank
        npes = config.world_size
        ms.shmem_barrier_all()
        self._p2p_tok_off = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_tis = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_out_wts = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_out_idx = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_out_tok = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_recv_num = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_out_scales = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        for pe in range(npes):
            self._p2p_tok_off[pe] = ms.shmem_ptr_p2p(self._tok_off.data_ptr(), r, pe)
            self._p2p_tis[pe] = ms.shmem_ptr_p2p(self._src_tok_pos_ptr, r, pe)
            self._p2p_out_wts[pe] = ms.shmem_ptr_p2p(outW_ptr, r, pe)
            self._p2p_out_idx[pe] = ms.shmem_ptr_p2p(outI_ptr, r, pe)
            self._p2p_out_tok[pe] = ms.shmem_ptr_p2p(out_ptr, r, pe)
            self._p2p_recv_num[pe] = ms.shmem_ptr_p2p(self._recv_num.data_ptr(), r, pe)
            self._p2p_out_scales[pe] = (
                ms.shmem_ptr_p2p(outS_ptr, r, pe) if outS_ptr else 0
            )

        mp = int(handle_info["multi_processor_count"])
        wpb = config.warp_num_per_block
        self._phase2_jit = make_dispatch_jit(
            rank=config.rank,
            npes=config.world_size,
            experts_per_rank=config.num_experts_per_rank,
            experts_per_token=config.num_experts_per_token,
            hidden_dim=config.hidden_dim,
            max_tok_per_rank=config.max_num_inp_token_per_rank,
            block_num=config.block_num,
            warp_num_per_block=wpb,
            data_type=data_type,
            scale_dim=config.scale_dim,
            scale_type_size=config.scale_type_size,
            enable_std_moe=False,
        )
        self._phase2_stdmoe_jit = make_dispatch_jit(
            rank=config.rank,
            npes=config.world_size,
            experts_per_rank=config.num_experts_per_rank,
            experts_per_token=config.num_experts_per_token,
            hidden_dim=config.hidden_dim,
            max_tok_per_rank=config.max_num_inp_token_per_rank,
            block_num=config.block_num,
            warp_num_per_block=wpb,
            data_type=data_type,
            scale_dim=config.scale_dim,
            scale_type_size=config.scale_type_size,
            enable_std_moe=True,
        )

        if not use_flydsl_copy_staging:
            return

        self._copy_jit = make_ep_dispatch_copy_to_staging_jit(
            rank=config.rank,
            world_size=config.world_size,
            max_tok_per_rank=config.max_num_inp_token_per_rank,
            hidden_dim=config.hidden_dim,
            experts_per_token=config.num_experts_per_token,
            scale_dim=config.scale_dim,
            scale_type_size=config.scale_type_size,
            multiprocessor_count=mp,
            warp_num_per_block=wpb,
            data_type=data_type,
        )

    @property
    def copy_enabled(self) -> bool:
        return self._phase2_jit is not None

    def can_use_fly_copy(
        self,
        *,
        quant_type: object,
        weight_ptr: int,
        scale_ptr: int,
        staging_ptr: int,
    ) -> bool:
        """Gate FlyDSL copy path to match mori behavior."""
        if self._phase2_jit is None:
            return False
        if _normalize_quant_type(quant_type) == EpDispatchCombineQuantType.Fp8DirectCast:
            return False
        if weight_ptr == 0:
            return False
        if self._scale_bytes > 0 and scale_ptr == 0:
            return False
        return True

    @property
    def tok_map_ptr(self) -> int:
        return self._tok_map.data_ptr()

    @property
    def src_tok_pos_ptr(self) -> int:
        return self._src_tok_pos_ptr

    def prepare_context(
        self,
        *,
        handle: int,
        hidden_dim: int,  # kept for call-site compatibility
        input: torch.Tensor,
        indices: torch.Tensor,
        weight_ptr: int,
        scale_ptr: int,
        rdma_block_num: int,  # kept for call-site compatibility
        dtype_to_int_fn,
    ) -> FlyDSLInterNodeV1LLDispatchContext:
        """Prepare handle inference state for downstream combine APIs."""
        mori_cpp.prepare_inference_args(
            handle,
            inp_ptr=input.data_ptr(),
            dtype=dtype_to_int_fn(input.dtype),
            num_tokens=input.size(0),
            weight_ptr=weight_ptr,
            scale_ptr=scale_ptr,
            indices_ptr=indices.data_ptr(),
        )
        _, _, _, _, total_ptr = self._dispatch_out_ptrs
        staging_ptr = self._get_staging_ptr_fn(handle) if self._get_staging_ptr_fn else 0
        return FlyDSLInterNodeV1LLDispatchContext(
            weight_ptr=weight_ptr,
            scale_ptr=scale_ptr,
            staging_ptr=staging_ptr,
            total_ptr=total_ptr,
        )

    def run_copy_and_launch(  # kept name to minimize call-site churn
        self,
        *,
        launch_kernel_fn,
        kernel_name: str,
        block_num: int,
        warp_per_block: int,
        shared_mem: int,
        stream,
        input_ptr: int,
        indices_ptr: int,
        ctx: FlyDSLInterNodeV1LLDispatchContext,
        cur_tok: int,
        packed_recv_x_ptr: int = 0,
        packed_recv_count_ptr: int = 0,
        packed_recv_src_info_ptr: int = 0,
        disp_tok_map_ptr: int = 0,
        disp_grid_bar_ptr: int = 0,
    ) -> None:
        del launch_kernel_fn, block_num, warp_per_block, shared_mem

        if (
            self._use_copy
            and self._copy_jit is not None
            and self._get_staging_ptr_fn is not None
            and ctx.staging_ptr != 0
        ):
            staging_ptr = ctx.staging_ptr
            self._copy_jit(
                fx.Int64(input_ptr),
                fx.Int64(indices_ptr),
                fx.Int64(ctx.weight_ptr),
                fx.Int64(ctx.scale_ptr),
                fx.Int64(staging_ptr),
                fx.Int32(cur_tok),
            )

        # Reset per-launch auxiliary state.
        self._tok_off.zero_()
        self._recv_num.zero_()
        self._dest_ctr.zero_()
        self._disp_bar.zero_()
        self._tok_map.fill_(self._config.world_size * self._max_recv)
        self._std_disp_tok_map.fill_(-1)
        self._std_disp_grid_bar.zero_()
        is_stdmoe = kernel_name.endswith("_stdmoe")
        jit = self._phase2_stdmoe_jit if is_stdmoe else self._phase2_jit
        if is_stdmoe and disp_tok_map_ptr == 0:
            disp_tok_map_ptr = self._std_disp_tok_map.data_ptr()
        if is_stdmoe and disp_grid_bar_ptr == 0:
            disp_grid_bar_ptr = self._std_disp_grid_bar.data_ptr()
        jit(
            fx.Int64(input_ptr),
            fx.Int64(indices_ptr),
            fx.Int64(ctx.weight_ptr),
            fx.Int64(self._dispatch_out_ptrs[0]),
            fx.Int64(self._dispatch_out_ptrs[1]),
            fx.Int64(self._dispatch_out_ptrs[3]),
            fx.Int64(self._tok_off.data_ptr()),
            fx.Int64(self._recv_num.data_ptr()),
            fx.Int64(self._dest_ctr.data_ptr()),
            fx.Int64(self._disp_bar.data_ptr()),
            fx.Int64(self._tok_map.data_ptr()),
            fx.Int64(self._src_tok_pos_ptr),
            fx.Int64(ctx.total_ptr),
            fx.Int64(self._p2p_tok_off.data_ptr()),
            fx.Int64(self._p2p_tis.data_ptr()),
            fx.Int64(self._p2p_out_wts.data_ptr()),
            fx.Int64(self._p2p_out_idx.data_ptr()),
            fx.Int64(self._p2p_out_tok.data_ptr()),
            fx.Int64(self._p2p_recv_num.data_ptr()),
            fx.Int64(ctx.scale_ptr),
            fx.Int64(self._p2p_out_scales.data_ptr()),
            fx.Int64(packed_recv_x_ptr),
            fx.Int64(packed_recv_count_ptr),
            fx.Int64(packed_recv_src_info_ptr),
            fx.Int64(disp_tok_map_ptr),
            fx.Int64(disp_grid_bar_ptr),
            fx.Int32(cur_tok),
            torch.cuda.current_stream(),
        )

        # Keep src-token map in raw local encoding for combine:
        # src = rank * max_tok_per_rank + tok_id.
