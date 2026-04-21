"""
InterNode V1 LL dispatch/combine with pure FlyDSL kernels.

- **Dispatch phase 1** (``EpDispatchCopyToStaging``): FlyDSL when enabled.
- **Dispatch phase 2** (low-latency routing): FlyDSL.
- Unsupported/legacy paths still fall back to HIP via ``super().dispatch``.
- **Combine**: pure FlyDSL combine kernel for bf16 + ``quant none``.
- **Standard MoE** (``ENABLE_STANDARD_MOE_ADAPT``): ``dispatch_standard_moe`` can use
  Fly dispatch path (including stdmoe mapping); ``combine_standard_moe`` stays **all HIP** because mori's
  ``EpCombineSync`` skips the hidden copy under std-moe, which FlyDSL sync does not
  mirror.

Set ``use_flydsl_copy_staging=False`` to disable FlyDSL phase-1 copy while keeping
FlyDSL phase-2 dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import mori.shmem as ms
from mori.shmem import mori_shmem_create_tensor

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

from .dispatch_combine_internode_v1ll_dispatch import (
    FlyDSLInterNodeV1LLDispatchEngine,
)
from .dispatch_combine_intranode_kernel import make_combine_jit


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
            quant_type=self.quant_type,
        )


class FlyDSLDispatchCombineInterNodeV1LLOp(EpDispatchCombineOp):
    """InterNodeV1LL with optional FlyDSL first dispatch stage (copy to staging)."""

    def __init__(self, config: FlyDSLDispatchCombineInterNodeV1LLConfig, *, use_flydsl_copy_staging: bool = True):
        mori_cfg = config.to_mori_config()
        super().__init__(mori_cfg)
        self._fly_use_copy = use_flydsl_copy_staging
        self._fly_dispatch = FlyDSLInterNodeV1LLDispatchEngine(
            config=mori_cfg,
            handle_info=self._handle_info,
            data_type=config.data_type,
            use_flydsl_copy_staging=use_flydsl_copy_staging,
            handle=self._handle,
            dispatch_out_ptrs=self._dispatch_out_ptrs,
            get_dispatch_src_token_pos_fn=self._get_dispatch_src_token_pos_func,
        )
        self._fly_combine_jit = {}
        self._dev = torch.device("cuda", mori_cfg.rank)
        npes = mori_cfg.world_size
        mt = mori_cfg.max_num_inp_token_per_rank
        mr = npes * mt
        k = mori_cfg.num_experts_per_token
        token_bytes = mori_cfg.hidden_dim * torch.tensor([], dtype=config.data_type).element_size()
        tok_i16_mr = (mr * token_bytes + 1) // 2
        self._fly_comb_inp_tok = mori_shmem_create_tensor((tok_i16_mr,), torch.int16)
        self._fly_comb_inp_wts = mori_shmem_create_tensor((mr * k,), torch.float32)
        self._fly_xdev_bar_mem = mori_shmem_create_tensor((npes,), torch.int64)
        ms.shmem_barrier_all()
        self._fly_xdev_flag = torch.ones(1, dtype=torch.int64, device=self._dev)
        self._fly_comb_bar = torch.zeros(1, dtype=torch.int32, device=self._dev)
        self._fly_p2p_comb_inp = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._fly_p2p_comb_inp_wts = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._fly_p2p_xdb_mem = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        for pe in range(npes):
            self._fly_p2p_comb_inp[pe] = ms.shmem_ptr_p2p(
                self._fly_comb_inp_tok.data_ptr(), mori_cfg.rank, pe
            )
            self._fly_p2p_comb_inp_wts[pe] = ms.shmem_ptr_p2p(
                self._fly_comb_inp_wts.data_ptr(), mori_cfg.rank, pe
            )
            self._fly_p2p_xdb_mem[pe] = ms.shmem_ptr_p2p(
                self._fly_xdev_bar_mem.data_ptr(), mori_cfg.rank, pe
            )
        for _hw in (False, True):
            self._fly_combine_jit[_hw] = make_combine_jit(
                rank=mori_cfg.rank,
                npes=mori_cfg.world_size,
                experts_per_rank=mori_cfg.num_experts_per_rank,
                experts_per_token=mori_cfg.num_experts_per_token,
                hidden_dim=mori_cfg.hidden_dim,
                max_tok_per_rank=mori_cfg.max_num_inp_token_per_rank,
                block_num=mori_cfg.block_num,
                warp_num_per_block=mori_cfg.warp_num_per_block,
                data_type=config.data_type,
                enable_weights=_hw,
                enable_std_moe=False,
                use_p2p_read=False,
            )

    def _fly_dispatch_copy_prerequisites(
        self, *, weight_ptr: int, scale_ptr: int, staging_ptr: int
    ) -> bool:
        """Same gating as mori ``EpDispatchCopyToStaging`` + Fly pack; fp8 → HIP only."""
        return self._fly_dispatch.can_use_fly_copy(
            quant_type=self.config.quant_type,
            weight_ptr=weight_ptr,
            scale_ptr=scale_ptr,
            staging_ptr=staging_ptr,
        )

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
        if not self._fly_dispatch.copy_enabled:
            return super().dispatch(
                input,
                weights,
                scales,
                indices,
                block_num=block_num,
                rdma_block_num=rdma_block_num,
                warp_per_block=warp_per_block,
            )

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

        ctx = self._fly_dispatch.prepare_context(
            handle=self._handle,
            hidden_dim=hidden_dim,
            input=input,
            indices=indices,
            weight_ptr=weight_ptr,
            scale_ptr=scale_ptr,
            rdma_block_num=actual_rbn,
            dtype_to_int_fn=dtype_to_int,
        )

        shared_mem = self._dispatch_shared_mem(actual_wpb)
        if not self._fly_dispatch_copy_prerequisites(
            weight_ptr=weight_ptr, scale_ptr=scale_ptr, staging_ptr=ctx.staging_ptr
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
        self._fly_dispatch.run_copy_and_launch(
            launch_kernel_fn=self._launch,
            kernel_name=f"EpDispatchInterNodeV1KernelLowLatency_{sfx}",
            block_num=actual_bn,
            warp_per_block=actual_wpb,
            shared_mem=shared_mem,
            stream=stream,
            input_ptr=input.data_ptr(),
            indices_ptr=indices.data_ptr(),
            ctx=ctx,
            cur_tok=cur_tok,
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
        _, actual_rbn, _ = self._resolve_launch_params(
            block_num, rdma_block_num, warp_per_block
        )
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
        _ = mori_cpp.build_args(
            self._handle,
            rdma_block_num=actual_rbn,
            hidden_dim=hidden_dim,
            use_external_inp_buf=use_external_inp_buf,
        )

        use_fly_combine = (
            kt == EpDispatchCombineKernelType.InterNodeV1LL.value
            and sfx == "bf16"
            and quant_type == EpDispatchCombineQuantType.None_
            and self._fly_combine_jit
        )
        if not use_fly_combine:
            raise RuntimeError(
                "InterNodeV1LL combine is configured for pure FlyDSL only; "
                f"unsupported combine config: dtype={input.dtype}, quant={self.config.quant_type}, kernel_type={self.config.kernel_type}"
            )

        import flydsl.expr as fx

        hw = bool(weight_ptr)
        cur_tok = int(self._get_cur_rank_num_token(self._handle))
        out_ptr, outW_ptr = mori_cpp.get_combine_output_ptrs(self._handle, hw)
        total_recv_ptr = self._dispatch_out_ptrs[4]
        if (
            out_ptr == 0
            or total_recv_ptr == 0
            or self._fly_dispatch.tok_map_ptr == 0
            or self._fly_dispatch.src_tok_pos_ptr == 0
            or self._fly_comb_inp_tok.data_ptr() == 0
            or self._fly_xdev_bar_mem.data_ptr() == 0
            or (hw and outW_ptr == 0)
        ):
            raise RuntimeError("Pure FlyDSL combine setup failed due to invalid buffer pointers")

        self._fly_combine_jit[hw](
            fx.Int64(input.data_ptr()),
            fx.Int64(self._fly_comb_inp_tok.data_ptr()),
            fx.Int64(out_ptr),
            fx.Int64(self._fly_xdev_bar_mem.data_ptr()),
            fx.Int64(self._fly_xdev_flag.data_ptr()),
            fx.Int64(self._fly_dispatch.tok_map_ptr),
            fx.Int64(self._fly_comb_bar.data_ptr()),
            fx.Int64(total_recv_ptr),
            fx.Int64(self._fly_dispatch.src_tok_pos_ptr),
            fx.Int64(self._fly_p2p_comb_inp.data_ptr()),
            fx.Int64(self._fly_p2p_xdb_mem.data_ptr()),
            fx.Int64(weight_ptr if hw else 0),
            fx.Int64(self._fly_comb_inp_wts.data_ptr() if hw else 0),
            fx.Int64(outW_ptr if hw else 0),
            fx.Int64(self._fly_p2p_comb_inp_wts.data_ptr() if hw else 0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int32(cur_tok),
            torch.cuda.current_stream(),
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

    def get_dispatch_src_token_pos(self):
        """Expose src-token map in mori FlatTokenIndex format for parity checks."""
        src = from_gpu_ptr(
            self._fly_dispatch.src_tok_pos_ptr,
            (self.config.world_size * self.config.max_num_inp_token_per_rank,),
            torch.int32,
        ).clone()
        max_tok = self.config.max_num_inp_token_per_rank
        world_stride = self.config.world_size * max_tok
        rank_id = torch.div(src, max_tok, rounding_mode="floor")
        tok_id = torch.remainder(src, max_tok)
        src.copy_(rank_id * world_stride + tok_id)
        return src

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
            get_staging_ptr = getattr(mori_cpp, "get_dispatch_combine_staging_ptr", None)
            staging_ptr = get_staging_ptr(self._handle) if get_staging_ptr else 0
            if self._fly_dispatch_copy_prerequisites(
                weight_ptr=weight_ptr, scale_ptr=scale_ptr, staging_ptr=staging_ptr
            ):
                ctx = self._fly_dispatch.prepare_context(
                    handle=self._handle,
                    hidden_dim=hidden_dim,
                    input=input,
                    indices=indices,
                    weight_ptr=weight_ptr,
                    scale_ptr=scale_ptr,
                    rdma_block_num=actual_rbn,
                    dtype_to_int_fn=dtype_to_int,
                )
                self._fly_dispatch.run_copy_and_launch(
                    launch_kernel_fn=self._launch,
                    kernel_name=f"EpDispatchInterNodeV1KernelLowLatency_{sfx}_stdmoe",
                    block_num=actual_bn,
                    warp_per_block=actual_wpb,
                    shared_mem=shared_mem,
                    stream=stream,
                    input_ptr=input.data_ptr(),
                    indices_ptr=indices.data_ptr(),
                    ctx=ctx,
                    cur_tok=int(input.size(0)),
                    packed_recv_x_ptr=packed_recv_x.data_ptr(),
                    packed_recv_count_ptr=mori_cpp.get_standard_moe_packed_recv_count_ptr(
                        self._handle
                    ),
                    packed_recv_src_info_ptr=packed_recv_src_info.data_ptr(),
                    disp_tok_map_ptr=0,
                    disp_grid_bar_ptr=0,
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
