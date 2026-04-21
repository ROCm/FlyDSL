"""
InterNode V1 LL dispatch/combine with FlyDSL copy-to-staging + FlyDSL combine tails.

- **Dispatch phase 1** (``EpDispatchCopyToStaging``): FlyDSL when prerequisites hold;
  otherwise full HIP (same as :class:`mori.ops.dispatch_combine.EpDispatchCombineOp`),
  including ``quant_type=fp8_direct_cast`` (HIP path casts / packs like mori).
- **Dispatch phase 2** (``EpDispatchInterNodeV1KernelLowLatency`` or ``…_stdmoe``): HIP.
- **Combine**: FlyDSL ``EpCombineSync`` + ``EpCombineAll`` only for bf16 + ``quant none``;
  **fp8** uses full HIP combine (``EpCombineSync`` includes bf16→internal fp8 cast).
- **Standard MoE** (``ENABLE_STANDARD_MOE_ADAPT``): ``dispatch_standard_moe`` can use
  Fly copy + ``EpDispatchInterNodeV1KernelLowLatency_*_stdmoe`` when the same Fly
  prerequisites hold; ``combine_standard_moe`` stays **all HIP** because mori's
  ``EpCombineSync`` skips the hidden copy under std-moe, which FlyDSL sync does not
  mirror.

Set ``use_flydsl_copy_staging=False`` for pure HIP (mori default).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mori import cpp as mori_cpp
from mori.ops.dispatch_combine import (
    TOPK_IDX_DTYPE,
    WARP_SIZE,
    EpDispatchCombineConfig,
    EpDispatchCombineKernelType,
    EpDispatchCombineOp,
    EpDispatchCombineQuantType,
    _DTYPE_SUFFIX,
    _cpp_dispatch_combine_factory,
    _current_stream,
    _normalize_quant_type,
    dtype_to_int,
    from_gpu_ptr,
)

from .dispatch_combine_internode_v1ll_copy_staging import (
    make_ep_dispatch_copy_to_staging_jit,
)
from .dispatch_combine_internode_v1ll_combine import (
    make_ep_combine_all_jit,
    make_ep_combine_sync_jit,
)


@dataclass
class FlyDSLDispatchCombineInterNodeV1LLConfig:
    """Python-side config mirroring :class:`mori.ops.dispatch_combine.EpDispatchCombineConfig`."""

    data_type: torch.dtype
    rank: int
    world_size: int
    hidden_dim: int
    scale_dim: int
    scale_type_size: int
    max_token_type_size: int
    max_num_inp_token_per_rank: int
    num_experts_per_rank: int
    num_experts_per_token: int
    warp_num_per_block: int = 8
    block_num: int = 80
    max_total_recv_tokens: int = 0
    use_external_inp_buf: bool = True
    gpu_per_node: int = 8
    rdma_block_num: int = 0
    num_qp_per_pe: int = 1
    quant_type: str = "none"

    def to_mori_config(self) -> EpDispatchCombineConfig:
        from mori.ops.dispatch_combine import _normalize_quant_type

        return EpDispatchCombineConfig(
            data_type=self.data_type,
            rank=self.rank,
            world_size=self.world_size,
            hidden_dim=self.hidden_dim,
            scale_dim=self.scale_dim,
            scale_type_size=self.scale_type_size,
            max_token_type_size=self.max_token_type_size,
            max_num_inp_token_per_rank=self.max_num_inp_token_per_rank,
            num_experts_per_rank=self.num_experts_per_rank,
            num_experts_per_token=self.num_experts_per_token,
            warp_num_per_block=self.warp_num_per_block,
            block_num=self.block_num,
            max_total_recv_tokens=self.max_total_recv_tokens,
            use_external_inp_buf=self.use_external_inp_buf,
            kernel_type=EpDispatchCombineKernelType.InterNodeV1LL,
            gpu_per_node=self.gpu_per_node,
            rdma_block_num=self.rdma_block_num,
            num_qp_per_pe=self.num_qp_per_pe,
            quant_type=_normalize_quant_type(self.quant_type),
        )


class FlyDSLDispatchCombineInterNodeV1LLOp(EpDispatchCombineOp):
    """InterNodeV1LL with optional FlyDSL first dispatch stage (copy to staging)."""

    def __init__(self, config: FlyDSLDispatchCombineInterNodeV1LLConfig, *, use_flydsl_copy_staging: bool = True):
        mori_cfg = config.to_mori_config()
        super().__init__(mori_cfg)
        self._fly_use_copy = use_flydsl_copy_staging
        self._fly_copy_jit = None
        self._fly_combine_sync_jit = {}
        self._fly_combine_all_jit = {}
        self._vll_scale_bytes = mori_cfg.scale_dim * mori_cfg.scale_type_size
        if use_flydsl_copy_staging:
            mp = int(self._handle_info["multi_processor_count"])
            wpb = mori_cfg.warp_num_per_block
            self._fly_copy_jit = make_ep_dispatch_copy_to_staging_jit(
                rank=mori_cfg.rank,
                world_size=mori_cfg.world_size,
                max_tok_per_rank=mori_cfg.max_num_inp_token_per_rank,
                hidden_dim=mori_cfg.hidden_dim,
                experts_per_token=mori_cfg.num_experts_per_token,
                scale_dim=mori_cfg.scale_dim,
                scale_type_size=mori_cfg.scale_type_size,
                multiprocessor_count=mp,
                warp_num_per_block=wpb,
                data_type=config.data_type,
            )
            for _hw in (False, True):
                self._fly_combine_sync_jit[_hw] = make_ep_combine_sync_jit(
                    hidden_dim=mori_cfg.hidden_dim,
                    experts_per_token=mori_cfg.num_experts_per_token,
                    multiprocessor_count=mp,
                    warp_num_per_block=wpb,
                    has_weights=_hw,
                    data_type=config.data_type,
                )
                self._fly_combine_all_jit[_hw] = make_ep_combine_all_jit(
                    rank=mori_cfg.rank,
                    world_size=mori_cfg.world_size,
                    gpu_per_node=mori_cfg.gpu_per_node,
                    max_tok_per_rank=mori_cfg.max_num_inp_token_per_rank,
                    hidden_dim=mori_cfg.hidden_dim,
                    experts_per_token=mori_cfg.num_experts_per_token,
                    num_experts_per_rank=mori_cfg.num_experts_per_rank,
                    multiprocessor_count=mp,
                    warp_num_per_block=wpb,
                    has_weights=_hw,
                    data_type=config.data_type,
                )

    def _fly_dispatch_copy_prerequisites(
        self, *, weight_ptr: int, scale_ptr: int, staging_ptr: int
    ) -> bool:
        """Same gating as mori ``EpDispatchCopyToStaging`` + Fly pack; fp8 → HIP only."""
        if not self._fly_use_copy or self._fly_copy_jit is None:
            return False
        if (
            _normalize_quant_type(self.config.quant_type)
            == EpDispatchCombineQuantType.Fp8DirectCast
        ):
            return False
        if staging_ptr == 0:
            return False
        if self._vll_scale_bytes > 0 and scale_ptr == 0:
            return False
        if weight_ptr == 0:
            return False
        return True

    def dispatch(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
    ):
        if not self._fly_use_copy or self._fly_copy_jit is None:
            return super().dispatch(
                input,
                weights,
                scales,
                indices,
                block_num=block_num,
                rdma_block_num=rdma_block_num,
                warp_per_block=warp_per_block,
            )

        import flydsl.expr as fx

        hidden_dim = input.size(1)
        weight_ptr = weights.data_ptr() if weights is not None else 0
        has_scales = scales is not None and self.config.scale_dim > 0
        scale_ptr = scales.data_ptr() if has_scales else 0
        actual_bn, actual_rbn, actual_wpb = self._resolve_launch_params(
            block_num, rdma_block_num, warp_per_block
        )
        stream = _current_stream()
        self._dispatch_dtype = input.dtype
        sfx = _DTYPE_SUFFIX[input.dtype]

        mori_cpp.prepare_inference_args(
            self._handle,
            inp_ptr=input.data_ptr(),
            dtype=dtype_to_int(input.dtype),
            num_tokens=input.size(0),
            weight_ptr=weight_ptr,
            scale_ptr=scale_ptr,
            indices_ptr=indices.data_ptr(),
        )
        args_ptr = mori_cpp.build_args(
            self._handle,
            rdma_block_num=actual_rbn,
            hidden_dim=hidden_dim,
        )

        shared_mem = self._dispatch_shared_mem(actual_wpb)
        staging_ptr = mori_cpp.get_dispatch_combine_staging_ptr(self._handle)
        if not self._fly_dispatch_copy_prerequisites(
            weight_ptr=weight_ptr, scale_ptr=scale_ptr, staging_ptr=staging_ptr
        ):
            return super().dispatch(
                input,
                weights,
                scales,
                indices,
                block_num=block_num,
                rdma_block_num=rdma_block_num,
                warp_per_block=warp_per_block,
            )

        cur_tok = int(input.size(0))
        self._fly_copy_jit(
            fx.Int64(input.data_ptr()),
            fx.Int64(indices.data_ptr()),
            fx.Int64(weight_ptr),
            fx.Int64(scale_ptr),
            fx.Int64(staging_ptr),
            fx.Int32(cur_tok),
        )

        self._launch(
            f"EpDispatchInterNodeV1KernelLowLatency_{sfx}",
            (actual_bn,),
            (WARP_SIZE * actual_wpb,),
            shared_mem,
            stream,
            args_ptr,
        )

        out_ptr, outW_ptr, outS_ptr, outI_ptr, total_ptr = self._dispatch_out_ptrs
        max_recv = self._cpp_config.max_num_tokens_to_recv()
        out = from_gpu_ptr(out_ptr, (max_recv, hidden_dim), input.dtype)
        out_weights = from_gpu_ptr(
            outW_ptr, (max_recv, self.config.num_experts_per_token), torch.float32
        )
        out_scales = None
        if has_scales and outS_ptr:
            out_scales = from_gpu_ptr(outS_ptr, (max_recv, self.config.scale_dim), scales.dtype)
        out_indices = from_gpu_ptr(
            outI_ptr, (max_recv, self.config.num_experts_per_token), TOPK_IDX_DTYPE
        )
        total_recv = from_gpu_ptr(total_ptr, (1,), TOPK_IDX_DTYPE)

        return (out, out_weights, out_scales, out_indices, total_recv)

    def combine(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
        use_external_inp_buf: int = -1,
        call_reset: bool = False,
    ):
        hidden_dim = input.size(1)
        weight_ptr = (
            weights.data_ptr() if weights is not None and weights.size(0) != 0 else 0
        )
        actual_bn, actual_rbn, actual_wpb = self._resolve_launch_params(
            block_num, rdma_block_num, warp_per_block
        )
        stream = _current_stream()
        self._combine_dtype = input.dtype
        sfx = _DTYPE_SUFFIX[input.dtype]
        quant_type = _normalize_quant_type(self.config.quant_type)
        kt = self.config.kernel_type.value

        mori_cpp.prepare_inference_args(
            self._handle,
            inp_ptr=input.data_ptr(),
            dtype=dtype_to_int(input.dtype),
            num_tokens=self._get_cur_rank_num_token(self._handle),
            weight_ptr=weight_ptr,
            scale_ptr=0,
            indices_ptr=indices.data_ptr(),
        )
        args_ptr = mori_cpp.build_args(
            self._handle,
            rdma_block_num=actual_rbn,
            hidden_dim=hidden_dim,
            use_external_inp_buf=use_external_inp_buf,
        )

        shared_mem = self._combine_shared_mem(actual_wpb)
        bsz = WARP_SIZE * actual_wpb

        use_fly_combine = (
            self._fly_use_copy
            and kt == EpDispatchCombineKernelType.InterNodeV1LL.value
            and sfx == "bf16"
            and quant_type == EpDispatchCombineQuantType.None_
            and self._fly_combine_sync_jit
        )

        if use_fly_combine:
            staging_ptr = mori_cpp.get_dispatch_combine_staging_ptr(self._handle)
            send_map_ptr = mori_cpp.get_dispatch_combine_disp_send_map_ptr(self._handle)
            block_flag_ptr = mori_cpp.get_dispatch_combine_block_flag_counter_ptr(self._handle)
            total_recv_ptr = mori_cpp.get_dispatch_combine_total_recv_num_ptr(self._handle)
            wts_shmem_ptr = mori_cpp.get_dispatch_combine_inp_weights_shmem_ptr(self._handle)
            combine_inp = self.get_registered_combine_input_buffer(input.dtype, hidden_dim)
            out_ptr, outW_ptr = mori_cpp.get_combine_output_ptrs(
                self._handle, bool(weight_ptr)
            )
            if (
                staging_ptr == 0
                or send_map_ptr == 0
                or block_flag_ptr == 0
                or total_recv_ptr == 0
                or combine_inp.data_ptr() == 0
                or out_ptr == 0
                or (weight_ptr != 0 and (wts_shmem_ptr == 0 or outW_ptr == 0))
            ):
                use_fly_combine = False

        if not use_fly_combine:
            return super().combine(
                input,
                weights,
                indices,
                block_num=block_num,
                rdma_block_num=rdma_block_num,
                warp_per_block=warp_per_block,
                use_external_inp_buf=use_external_inp_buf,
                call_reset=call_reset,
            )

        import flydsl.expr as fx

        hw = bool(weight_ptr)
        cur_tok = int(self._get_cur_rank_num_token(self._handle))
        self._fly_combine_sync_jit[hw](
            fx.Int64(input.data_ptr()),
            fx.Int64(combine_inp.data_ptr()),
            fx.Int64(weight_ptr if hw else 0),
            fx.Int64(wts_shmem_ptr if hw else 0),
            fx.Int64(total_recv_ptr),
        )
        self._launch_multi(
            [
                f"EpCombineSyncBarrier_{sfx}",
                f"EpCombineInterNodeV1KernelLowLatency_{sfx}",
            ],
            [1, actual_bn],
            [WARP_SIZE, bsz],
            [0, shared_mem],
            stream,
            args_ptr,
        )
        self._fly_combine_all_jit[hw](
            fx.Int64(total_recv_ptr),
            fx.Int64(block_flag_ptr),
            fx.Int64(send_map_ptr),
            fx.Int64(staging_ptr),
            fx.Int64(out_ptr),
            fx.Int64(outW_ptr if hw else 0),
            fx.Int64(indices.data_ptr()),
            fx.Int32(cur_tok),
        )

        out = from_gpu_ptr(
            out_ptr,
            (self.config.max_num_inp_token_per_rank, hidden_dim),
            input.dtype,
        )
        out_weights = None
        if weight_ptr and outW_ptr:
            out_weights = from_gpu_ptr(
                outW_ptr,
                (
                    self.config.max_num_inp_token_per_rank,
                    self.config.num_experts_per_token,
                ),
                weights.dtype,
            )

        if call_reset:
            self._reset_func(self._handle, _current_stream())
        return (out, out_weights)

    def dispatch_standard_moe(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
    ):
        """Same as :meth:`EpDispatchCombineOp.dispatch_standard_moe` but Fly copy for V1LL when allowed."""
        set_fn = _cpp_dispatch_combine_factory(
            "set_standard_moe_output_buffers", allow_missing=True
        )
        if set_fn is None:
            raise RuntimeError(
                "dispatch_standard_moe is not available. "
                "Rebuild with ENABLE_STANDARD_MOE_ADAPT=ON."
            )
        block_num, rdma_block_num, warp_per_block = self.get_launch_config(
            is_dispatch=True,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            warp_per_block=warp_per_block,
        )

        hidden_dim = input.size(1)
        num_local_experts = self.config.num_experts_per_rank
        max_tokens_per_expert = (
            self.config.world_size * self.config.max_num_inp_token_per_rank
        )
        actual_bn, actual_rbn, actual_wpb = self._resolve_launch_params(
            block_num, rdma_block_num, warp_per_block
        )
        stream = _current_stream()
        sfx = _DTYPE_SUFFIX[input.dtype]

        packed_recv_x = torch.empty(
            (num_local_experts, max_tokens_per_expert, hidden_dim),
            dtype=input.dtype,
            device=input.device,
        )
        packed_recv_src_info = torch.empty(
            (num_local_experts, max_tokens_per_expert),
            dtype=torch.int32,
            device=input.device,
        )
        packed_recv_layout_range = torch.empty(
            0, dtype=torch.int64, device=input.device
        )

        set_fn(self._handle, packed_recv_x.data_ptr(), packed_recv_src_info.data_ptr())

        weight_ptr = weights.data_ptr() if weights is not None else 0
        has_scales = scales is not None and self.config.scale_dim > 0
        scale_ptr = scales.data_ptr() if has_scales else 0
        mori_cpp.prepare_inference_args(
            self._handle,
            inp_ptr=input.data_ptr(),
            dtype=dtype_to_int(input.dtype),
            num_tokens=input.size(0),
            weight_ptr=weight_ptr,
            scale_ptr=scale_ptr,
            indices_ptr=indices.data_ptr(),
        )
        args_ptr = mori_cpp.build_args(
            self._handle,
            rdma_block_num=actual_rbn,
            hidden_dim=hidden_dim,
        )

        grid = (actual_bn,)
        block = (WARP_SIZE * actual_wpb,)
        shared_mem = self._dispatch_shared_mem(actual_wpb)
        kt = self.config.kernel_type.value

        if kt == EpDispatchCombineKernelType.InterNodeV1LL.value:
            mp = self._handle_info["multi_processor_count"]
            staging_ptr = mori_cpp.get_dispatch_combine_staging_ptr(self._handle)
            if self._fly_dispatch_copy_prerequisites(
                weight_ptr=weight_ptr, scale_ptr=scale_ptr, staging_ptr=staging_ptr
            ):
                import flydsl.expr as fx

                cur_tok = int(input.size(0))
                self._fly_copy_jit(
                    fx.Int64(input.data_ptr()),
                    fx.Int64(indices.data_ptr()),
                    fx.Int64(weight_ptr),
                    fx.Int64(scale_ptr),
                    fx.Int64(staging_ptr),
                    fx.Int32(cur_tok),
                )
                self._launch(
                    f"EpDispatchInterNodeV1KernelLowLatency_{sfx}_stdmoe",
                    grid,
                    block,
                    shared_mem,
                    stream,
                    args_ptr,
                )
            else:
                self._launch(
                    f"EpDispatchCopyToStaging_{sfx}", (mp,), block, 0, stream, args_ptr
                )
                self._launch(
                    f"EpDispatchInterNodeV1KernelLowLatency_{sfx}_stdmoe",
                    grid,
                    block,
                    shared_mem,
                    stream,
                    args_ptr,
                )
        elif kt == EpDispatchCombineKernelType.IntraNode.value:
            self._launch(
                f"EpDispatchIntraNodeKernel_{sfx}_stdmoe",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        else:
            raise ValueError(
                "dispatch_standard_moe only supports IntraNode/InterNodeV1LL"
            )

        packed_recv_count_ptr = mori_cpp.get_standard_moe_packed_recv_count_ptr(
            self._handle
        )
        packed_recv_count = from_gpu_ptr(
            packed_recv_count_ptr, (num_local_experts,), torch.int32
        )

        return (
            packed_recv_x,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
        )
