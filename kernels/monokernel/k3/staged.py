# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Staged MLA/KDA baselines retained beside the Kimi-K3 MonoKernel."""

from __future__ import annotations

from contextlib import contextmanager

import torch

from kernels.moe.moe_sorting_kernel import moe_sorting_flydsl
from kernels.monokernel.config import (
    EPS,
    KIMI_K3_CONFIG,
    ConvStateLayout,
    KvCacheLayout,
)
from kernels.monokernel.formats import quantize_mxfp8
from kernels.monokernel.k3.attn_res import KimiK3AttnRes
from kernels.monokernel.k3.kda import KimiK3KdaAttention
from kernels.monokernel.k3.mla import KimiK3MlaAttention
from kernels.monokernel.k3.moe import kimi_k3_mxfp4_gemm1, kimi_k3_mxfp4_gemm2
from kernels.monokernel.k3.router import SigmoidTopkRouter
from kernels.monokernel.k3.router_projection import FusedRouterProjection
from kernels.monokernel.k3.tail import FusedKimiK3Tail
from kernels.monokernel.k3.torch_fusions import (
    CudaStageProfiler,
    compiled_attn_res_no_delta,
    compiled_attn_res_with_delta,
    compiled_rmsnorm,
    compiled_rmsnorm_out,
    rmsnorm,
    situ,
)
from kernels.monokernel.mxfp8_linear import Mxfp8Linear
from kernels.monokernel.packing import (
    pack_bf16,
    pack_mxfp8_scale,
    pack_mxfp8_weight,
)
from kernels.monokernel.symmetric_allreduce import SymmetricBf16Allreduce
from kernels.monokernel.weights import LayerWeights, prepare_mxfp4_expert_storage

_TP_SIZE = 8
_ROUTING_TILE_M = 16


class _KimiK3MlaPath:
    """Internal staged Kimi-K3 MLA reference/performance path.

    The MLA core remains the persistent shared/reuse kernel.  The K3-specific
    tail composes the model's attention-residual mixer, 896-way top-16 router,
    latent A16W4 experts, BF16 shared experts, latent output transform, and two
    graph-safe TP reductions.
    """

    def __init__(
        self,
        weights: LayerWeights,
        samples: int,
        *,
        layer_idx: int,
        rank: int,
        npes: int = _TP_SIZE,
        group=None,
        reduce_group=None,
        topk: int = 2048,
        fuse_attn_res: bool = True,
        fuse_router: bool = True,
        fuse_shared_experts: bool = True,
        reduce_backend: str = "symmetric",
        kv_cache_layout: KvCacheLayout | str = KvCacheLayout.SPLIT,
        mtp: bool = False,
        agentic_batch_size: int = 0,
        conv_state_layout: ConvStateLayout = ConvStateLayout.CHANNEL_MAJOR,
        attention_symmetric_allreduce=None,
        moe_symmetric_allreduce: SymmetricBf16Allreduce | None = None,
        state_dtype: torch.dtype = torch.float32,
        defer_collectives: bool = False,
        packed_artifacts: dict[str, object] | None = None,
        monokernel_only: bool = False,
    ) -> None:
        config = weights.config
        if config != KIMI_K3_CONFIG:
            raise ValueError("the Kimi-K3 staged path requires Kimi-K3 weights")
        if npes != _TP_SIZE:
            raise ValueError(f"the Kimi-K3 staged path requires TP8, got TP{npes}")
        if weights.rank != rank or weights.npes != npes:
            raise ValueError(f"weight shard is rank {weights.rank}/TP{weights.npes}, requested rank {rank}/TP{npes}")
        if not 0 <= rank < npes:
            raise ValueError(f"rank must be in [0, {npes}), got {rank}")
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")
        if config.routed_hidden is None or config.shared_inter is None or config.attn_res_block_size is None:
            raise ValueError("Kimi-K3 latent-MoE/AttnRes dimensions are missing")

        self.W = weights
        self.t = weights.t
        self.config = config
        self.S = samples
        self.rank = rank
        self.npes = npes
        self.group = group
        self.reduce_group = reduce_group
        if reduce_backend not in {"symmetric", "nccl"}:
            raise ValueError(f"unsupported reduce backend {reduce_backend!r}; expected 'symmetric' or 'nccl'")
        self.reduce_backend = reduce_backend
        self.layer_idx = layer_idx
        self.topk = topk
        self.fuse_attn_res = fuse_attn_res
        self.inline_pre_attn = fuse_attn_res and layer_idx == 0
        self.fuse_router = fuse_router
        self.fuse_shared_experts = fuse_shared_experts
        self.monokernel_only = monokernel_only
        self.routed_hidden = config.routed_hidden
        self.shared_inter = config.shared_inter
        self.hidden_shard = config.hidden // npes
        packed_artifacts = packed_artifacts or {}

        expected = {
            "w_r",
            "bias",
            "w_latent_down",
            "g_latent",
            "w_latent_up",
            "w_shared_ug",
            "w_shared_dn",
            "w_ug",
            "s_ug",
            "w_dn",
            "s_dn",
            "g_self_res",
            "w_self_res",
            "g_mlp_res",
            "w_mlp_res",
        }
        missing = sorted(expected.difference(self.t))
        if missing:
            raise ValueError(f"missing Kimi-K3 MonoKernel weights: {', '.join(missing)}")
        if self.t["w_latent_up"].shape != (self.hidden_shard, self.routed_hidden):
            raise ValueError(
                f"w_latent_up must be the rank-local output-row shard [{self.hidden_shard}, {self.routed_hidden}]"
            )

        self.attention = self._build_attention(
            weights,
            samples,
            rank=rank,
            npes=npes,
            group=group,
            reduce_group=reduce_group,
            topk=topk,
            reduce_backend=reduce_backend,
            kv_cache_layout=kv_cache_layout,
            mtp=mtp,
            agentic_batch_size=agentic_batch_size,
            conv_state_layout=conv_state_layout,
            attention_symmetric_allreduce=attention_symmetric_allreduce,
            state_dtype=state_dtype,
            defer_collectives=defer_collectives,
            packed_artifacts=packed_artifacts.get("attention"),
            monokernel_only=monokernel_only,
        )
        retain_staged_workspaces = not monokernel_only
        device = torch.device("cuda", torch.cuda.current_device())
        self.pre_attn = torch.empty(samples, config.hidden, dtype=torch.bfloat16, device=device)
        self.pre_updated = torch.empty_like(self.pre_attn)
        self.moe_input = torch.empty_like(self.pre_attn)
        self.updated_prefix = torch.empty_like(self.pre_attn)
        self.pre_attn_res = self.post_attn_res = None
        if not monokernel_only:
            self.pre_attn_res = KimiK3AttnRes(
                samples,
                config.hidden,
                self.previous_valid_blocks,
                False,
                self.block_write_idx if self.is_block_write_layer else -1,
            )
            self.post_attn_res = KimiK3AttnRes(
                samples,
                config.hidden,
                self.previous_valid_blocks + int(self.is_block_write_layer),
                not self.is_block_write_layer,
                -1,
                quantize_output=True,
                source_override_idx=0 if self.inline_pre_attn else -1,
            )

        injected_moe = packed_artifacts.get("moe_packed")
        if injected_moe is None:
            (
                self.w_ug,
                self.s_ug,
                self.w_dn,
                self.s_dn,
            ) = prepare_mxfp4_expert_storage(weights)
        else:
            self.w_ug = injected_moe["w_ug"]
            self.s_ug = injected_moe["s_ug"]
            self.w_dn = injected_moe["w_dn"]
            self.s_dn = injected_moe["s_dn"]
        if packed_artifacts:
            required = {
                "w_router",
                "w_latent_down",
                "s_latent_down",
                "w_shared_ug",
                "s_shared_ug",
                "w_shared_dn",
                "s_shared_dn",
                "w_latent_up",
                "s_latent_up",
            }
            missing = sorted(required.difference(packed_artifacts))
            if missing:
                raise ValueError(
                    f"missing shared Kimi packed artifacts: {', '.join(missing)}"
                )
            self.w_router = packed_artifacts["w_router"]
            self.latent_projection = Mxfp8Linear.from_packed(
                packed_artifacts["w_latent_down"],
                packed_artifacts["s_latent_down"],
                n=self.routed_hidden,
                k=config.hidden,
                rows=samples,
                workspace_only=monokernel_only,
            )
            self.shared_projection = Mxfp8Linear.from_packed(
                packed_artifacts["w_shared_ug"],
                packed_artifacts["s_shared_ug"],
                n=2 * self.shared_inter,
                k=config.hidden,
                rows=samples,
                workspace_only=monokernel_only,
            )
            self.w_shared_dn = packed_artifacts["w_shared_dn"]
            self.s_shared_dn = packed_artifacts["s_shared_dn"]
            self.w_latent_up = packed_artifacts["w_latent_up"]
            self.s_latent_up = packed_artifacts["s_latent_up"]
        else:
            self.w_router = pack_bf16(self.t["w_r"])
            latent_weight, latent_scale = quantize_mxfp8(
                self.t["w_latent_down"]
            )
            shared_weight, shared_scale = quantize_mxfp8(
                self.t["w_shared_ug"]
            )
            shared_down_weight, shared_down_scale = quantize_mxfp8(
                self.t["w_shared_dn"]
            )
            latent_up_weight, latent_up_scale = quantize_mxfp8(
                self.t["w_latent_up"]
            )
            self.latent_projection = Mxfp8Linear(
                latent_weight,
                latent_scale,
                samples,
                workspace_only=monokernel_only,
            )
            self.shared_projection = Mxfp8Linear(
                shared_weight,
                shared_scale,
                samples,
                workspace_only=monokernel_only,
            )
            self.w_shared_dn = pack_mxfp8_weight(shared_down_weight)
            self.s_shared_dn = pack_mxfp8_scale(shared_down_scale)
            self.w_latent_up = pack_mxfp8_weight(latent_up_weight)
            self.s_latent_up = pack_mxfp8_scale(latent_up_scale)
        self.w_latent_down = self.latent_projection.weight
        self.s_latent_down = self.latent_projection.scale
        self.w_shared_ug = self.shared_projection.weight
        self.s_shared_ug = self.shared_projection.scale
        local_moe = {
            "w_r": self.w_router,
            "w_latent_down": self.w_latent_down,
            "s_latent_down": self.s_latent_down,
            "w_shared_ug": self.w_shared_ug,
            "s_shared_ug": self.s_shared_ug,
            "w_ug": self.w_ug,
            "s_ug": self.s_ug,
            "w_dn": self.w_dn,
            "s_dn": self.s_dn,
            "w_shared_dn": self.w_shared_dn,
            "s_shared_dn": self.s_shared_dn,
            "w_latent_up": self.w_latent_up,
            "s_latent_up": self.s_latent_up,
        }
        canonical_moe = packed_artifacts.get("moe_packed")
        attention_moe = getattr(self.attention, "moe_packed", None)
        if canonical_moe is None and attention_moe:
            canonical_moe = attention_moe
        if canonical_moe is None:
            canonical_moe = local_moe
        elif any(
            canonical_moe.get(name) is not tensor
            for name, tensor in local_moe.items()
        ):
            raise ValueError("Kimi MoE packed artifact alias mismatch")
        self.moe_packed = canonical_moe
        self.shared_activation_owner = None
        if (
            monokernel_only
            and not retain_staged_workspaces
            and isinstance(self.attention, KimiK3KdaAttention)
        ):
            # KDA's fused full-layer launch requires this graph-stable backing
            # allocation to remain live. Keep only the exact activation tile;
            # the unused scale workspace and projection object are discarded.
            self.shared_activation_owner = self.shared_projection.activation
            self.shared_projection = None
        elif monokernel_only and not retain_staged_workspaces:
            # The fused application kernel consumes only the shared packed
            # weights; staged projection activation workspaces are dead.
            self.shared_projection = None

        self.max_sorted = 0
        self.sorted_token_ids = self.sorted_weights = None
        self.sorted_expert_ids = self.num_valid_ids = None
        self.inter_sorted = None
        self.router_logits = self.router_scores = None
        self.topk_keys = self.topk_ids_i64 = None
        self.topk_ids = self.topk_weights = None
        if retain_staged_workspaces:
            # At most one padded BM tile is needed per selected route: there
            # can be no more active experts than routes.
            max_sorted = samples * config.top_k * _ROUTING_TILE_M
            max_blocks = (
                max_sorted + _ROUTING_TILE_M - 1
            ) // _ROUTING_TILE_M
            self.max_sorted = max_sorted
            self.sorted_token_ids = torch.empty(
                max_sorted, dtype=torch.int32, device=device
            )
            self.sorted_weights = torch.empty(
                max_sorted, dtype=torch.float32, device=device
            )
            self.sorted_expert_ids = torch.empty(
                max_blocks, dtype=torch.int32, device=device
            )
            self.num_valid_ids = torch.empty(
                2, dtype=torch.int32, device=device
            )
            self.inter_sorted = torch.empty(
                max_sorted,
                config.inter,
                dtype=torch.bfloat16,
                device=device,
            )
            self.router_logits = torch.empty(
                samples,
                config.n_experts,
                dtype=torch.bfloat16,
                device=device,
            )
            self.router_scores = torch.empty(
                samples,
                config.n_experts,
                dtype=torch.float32,
                device=device,
            )
            self.topk_keys = torch.empty(
                samples,
                config.top_k,
                dtype=torch.float32,
                device=device,
            )
            self.topk_ids_i64 = torch.empty(
                samples,
                config.top_k,
                dtype=torch.int64,
                device=device,
            )
            self.topk_ids = torch.empty(
                samples,
                config.top_k,
                dtype=torch.int32,
                device=device,
            )
            self.topk_weights = torch.empty(
                samples,
                config.top_k,
                dtype=torch.float32,
                device=device,
            )
        self.router_select = self.router_projection = None
        if not monokernel_only:
            self.router_select = SigmoidTopkRouter(
                config.n_experts, config.top_k, samples
            )
            self.router_projection = FusedRouterProjection(
                config.hidden,
                config.n_experts,
                config.top_k,
                samples,
                samples * self.routed_hidden,
                self.routed_hidden,
                2 * self.shared_inter,
                config.situ_beta,
                config.situ_linear_beta,
            )
        self.router_score_mailbox = None
        self.latent = self.routed_partial = self.routed_reduced = None
        self.latent_norm = self.shared_gu = self.shared_mid = None
        self.shared_partial = self.tail = self.final_partial = None
        self.moe_delta = None
        if retain_staged_workspaces:
            self.router_score_mailbox = torch.zeros(
                samples * config.n_experts * 2,
                dtype=torch.int32,
                device=device,
            )
            self.latent = torch.empty(
                samples,
                self.routed_hidden,
                dtype=torch.bfloat16,
                device=device,
            )
            self.routed_partial = torch.empty_like(self.latent)
            self.routed_reduced = torch.empty_like(self.latent)
            self.latent_norm = torch.empty_like(self.latent)
            self.shared_gu = torch.empty(
                samples,
                2 * self.shared_inter,
                dtype=torch.bfloat16,
                device=device,
            )
            self.shared_mid = torch.empty(
                samples,
                self.shared_inter,
                dtype=torch.bfloat16,
                device=device,
            )
            self.shared_partial = torch.empty(
                samples,
                config.hidden,
                dtype=torch.bfloat16,
                device=device,
            )
            self.tail = torch.empty(
                samples,
                self.hidden_shard,
                dtype=torch.bfloat16,
                device=device,
            )
            self.final_partial = torch.empty_like(self.shared_partial)
            self.moe_delta = torch.empty_like(self.shared_partial)
        self.output = torch.empty(
            samples, config.hidden, dtype=torch.bfloat16, device=device
        )
        self.attention_delta = torch.empty_like(self.output)
        self._profiler = CudaStageProfiler()
        if reduce_backend == "symmetric":
            self.symmetric_allreduce = moe_symmetric_allreduce
            if self.symmetric_allreduce is None and not defer_collectives:
                self.symmetric_allreduce = SymmetricBf16Allreduce(
                    (
                        samples * self.routed_hidden,
                        samples * config.hidden,
                    ),
                    rank=rank,
                    npes=npes,
                    group=group,
                    final_hidden=config.hidden,
                    final_shard_width=self.hidden_shard,
                    rmsnorm_width=self.routed_hidden,
                )
        else:
            if moe_symmetric_allreduce is not None:
                raise ValueError("an injected MoE all-reduce requires reduce_backend='symmetric'")
            self.symmetric_allreduce = None
        self.fused_tail = self._build_fused_tail()

        # The sorter also clears this output buffer before atomic stage2.
        self.moe_buf = self.routed_partial
        if reduce_group is None:
            raise ValueError("the Kimi-K3 staged path requires a GPU-capable TP reduce_group")

    def packed_artifacts(self) -> dict[str, object]:
        return {
            "attention": self.attention.packed_artifacts(),
            "moe_packed": self.moe_packed,
            "w_router": self.w_router,
            "w_latent_down": self.w_latent_down,
            "s_latent_down": self.s_latent_down,
            "w_shared_ug": self.w_shared_ug,
            "s_shared_ug": self.s_shared_ug,
            "w_shared_dn": self.w_shared_dn,
            "s_shared_dn": self.s_shared_dn,
            "w_latent_up": self.w_latent_up,
            "s_latent_up": self.s_latent_up,
        }

    def full_plan_workspace_tensors(self) -> tuple[torch.Tensor, ...]:
        """Enumerate exact bucket-local tensors retained by fused layers."""

        if not self.monokernel_only:
            raise ValueError("workspace enumeration requires monokernel_only")
        tensors = (
            self.pre_attn,
            self.pre_updated,
            self.moe_input,
            self.updated_prefix,
            self.latent_projection.activation,
            self.latent_projection.activation_scale,
            self.output,
            self.attention_delta,
            *self.attention.full_plan_workspace_tensors(),
        )
        if self.shared_activation_owner is not None:
            tensors = (*tensors, self.shared_activation_owner)
        if self.routed_partial is None:
            return tensors
        return (
            *tensors,
            self.shared_projection.activation,
            self.shared_projection.activation_scale,
            self.sorted_token_ids,
            self.sorted_weights,
            self.sorted_expert_ids,
            self.num_valid_ids,
            self.inter_sorted,
            self.router_logits,
            self.router_scores,
            self.topk_keys,
            self.topk_ids_i64,
            self.topk_ids,
            self.topk_weights,
            self.router_score_mailbox,
            self.latent,
            self.routed_partial,
            self.routed_reduced,
            self.latent_norm,
            self.shared_gu,
            self.shared_mid,
            self.shared_partial,
            self.tail,
            self.final_partial,
            self.moe_delta,
        )

    def _build_fused_tail(self):
        if (
            self.monokernel_only
            or self.symmetric_allreduce is None
            or not self.fuse_shared_experts
        ):
            return None
        return FusedKimiK3Tail(
            self.S,
            self.config.hidden,
            self.routed_hidden,
            self.shared_inter,
            self.rank,
            self.npes,
            self.symmetric_allreduce.max_pairs,
        )

    def initialize_collectives(
        self,
        attention: SymmetricBf16Allreduce | None = None,
        moe: SymmetricBf16Allreduce | None = None,
    ) -> tuple[SymmetricBf16Allreduce, SymmetricBf16Allreduce]:
        """Publish shared reductions after all ranks built local artifacts."""

        initialize_attention = getattr(
            self.attention,
            "initialize_symmetric_allreduce",
            None,
        )
        if initialize_attention is None:
            raise ValueError("deferred collectives require KDA attention")
        attention = initialize_attention(attention)
        if self.symmetric_allreduce is None:
            self.symmetric_allreduce = (
                moe
                if moe is not None
                else SymmetricBf16Allreduce(
                    (
                        self.S * self.routed_hidden,
                        self.S * self.config.hidden,
                    ),
                    rank=self.rank,
                    npes=self.npes,
                    group=self.group,
                    final_hidden=self.config.hidden,
                    final_shard_width=self.hidden_shard,
                    rmsnorm_width=self.routed_hidden,
                )
            )
        elif moe is not None and self.symmetric_allreduce is not moe:
            raise ValueError("MoE all-reduce was initialized with a different resource")
        self.fused_tail = self._build_fused_tail()
        return attention, self.symmetric_allreduce

    def _build_attention(
        self,
        weights: LayerWeights,
        samples: int,
        *,
        rank: int,
        npes: int,
        group,
        reduce_group,
        topk: int,
        reduce_backend: str,
        kv_cache_layout: KvCacheLayout | str,
        mtp: bool,
        agentic_batch_size: int,
        conv_state_layout: ConvStateLayout,
        attention_symmetric_allreduce,
        state_dtype: torch.dtype,
        defer_collectives: bool,
        packed_artifacts: dict[str, torch.Tensor] | None,
        monokernel_only: bool,
    ):
        if attention_symmetric_allreduce is not None:
            raise ValueError("Kimi-K3 MLA does not accept a KDA all-reduce")
        del (
            reduce_group,
            reduce_backend,
            mtp,
            agentic_batch_size,
            conv_state_layout,
            state_dtype,
            defer_collectives,
            packed_artifacts,
            monokernel_only,
        )
        return KimiK3MlaAttention(
            weights,
            samples,
            rank=rank,
            npes=npes,
            group=group,
            topk=topk,
            attention_input_norm=self.inline_pre_attn,
            kv_cache_layout=kv_cache_layout,
        )

    @contextmanager
    def _profile_stage(self, name: str):
        with self._profiler.stage(name):
            yield

    def start_stage_profile(self) -> None:
        """Collect one eager forward's per-stage GPU event timings."""

        self._profiler.start()

    def finish_stage_profile(self) -> dict[str, float]:
        """Synchronize and return the active stage profile in microseconds."""

        return self._profiler.finish()

    @property
    def is_block_write_layer(self) -> bool:
        return self.layer_idx % self.config.attn_res_block_size == 0

    @property
    def block_write_idx(self) -> int:
        return self.layer_idx // self.config.attn_res_block_size

    @property
    def previous_valid_blocks(self) -> int:
        block = self.config.attn_res_block_size
        return (self.layer_idx + block - 1) // block

    def _attn_res(
        self,
        prefix: torch.Tensor,
        delta: torch.Tensor | None,
        blocks: torch.Tensor,
        norm_weight: torch.Tensor,
        qk_weight: torch.Tensor,
        output_norm_weight: torch.Tensor | None,
        num_blocks: int,
        block_write_idx: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fuse_attn_res and output_norm_weight is not None:
            source_blocks = blocks[:, :num_blocks]
            if num_blocks == 0:
                updated = prefix if delta is None else (prefix.float() + delta.float()).to(torch.bfloat16)
                mixed = compiled_rmsnorm(updated, output_norm_weight)
            elif delta is None:
                updated = prefix
                mixed = compiled_attn_res_no_delta(
                    prefix,
                    source_blocks,
                    norm_weight,
                    qk_weight,
                    output_norm_weight,
                )
            else:
                mixed, updated = compiled_attn_res_with_delta(
                    prefix,
                    delta,
                    source_blocks,
                    norm_weight,
                    qk_weight,
                    output_norm_weight,
                )
            if block_write_idx >= 0:
                blocks[:, block_write_idx].copy_(updated)
            return mixed, updated

        updated = prefix if delta is None else (prefix.float() + delta.float()).to(torch.bfloat16)
        if block_write_idx >= 0:
            blocks[:, block_write_idx].copy_(updated)
        if num_blocks == 0:
            mixed = updated
        else:
            sources = torch.cat((blocks[:, :num_blocks], updated[:, None]), dim=1)
            sf = sources.float()
            normalized = sf * torch.rsqrt(sf.square().mean(-1, keepdim=True) + EPS)
            logits = (normalized * norm_weight.float() * qk_weight.float()).sum(-1)
            mixed = (torch.softmax(logits, dim=-1)[..., None] * sf).sum(1).to(torch.bfloat16)
        if output_norm_weight is not None:
            mixed = rmsnorm(mixed, output_norm_weight)
        return mixed, updated

    def _launch_projection(
        self,
        projection: FusedRouterProjection,
        hidden_states: torch.Tensor,
        epoch_layer: int,
    ) -> None:
        projection(
            hidden_states,
            self.latent_projection.activation,
            self.latent_projection.activation_scale,
            self.w_router,
            self.w_latent_down,
            self.s_latent_down,
            self.w_shared_ug,
            self.s_shared_ug,
            self.t["bias"],
            self.router_score_mailbox,
            self.router_scores,
            self.topk_ids,
            self.topk_weights,
            self.sorted_token_ids,
            self.sorted_weights,
            self.sorted_expert_ids,
            self.num_valid_ids,
            self.moe_buf,
            self.latent,
            self.shared_gu,
            self.shared_mid,
            self.attention.step,
            epoch_layer,
        )

    def _route_and_sort(self, hidden_states: torch.Tensor, epoch_layer: int) -> None:
        if self.fuse_router:
            self._launch_projection(self.router_projection, hidden_states, epoch_layer)
        else:
            torch.mm(hidden_states, self.t["w_r"].t(), out=self.router_logits)
            torch.sigmoid(self.router_logits.float(), out=self.router_scores)
            corrected = self.router_scores + self.t["bias"]
            torch.topk(
                corrected,
                self.config.top_k,
                dim=-1,
                sorted=True,
                out=(self.topk_keys, self.topk_ids_i64),
            )
            self.topk_ids.copy_(self.topk_ids_i64)
            torch.gather(self.router_scores, 1, self.topk_ids_i64, out=self.topk_weights)
            self.topk_weights.div_(self.topk_weights.sum(-1, keepdim=True))
        if not self.fuse_router:
            moe_sorting_flydsl(
                self.topk_ids,
                self.topk_weights,
                self.sorted_token_ids,
                self.sorted_weights,
                self.sorted_expert_ids,
                self.num_valid_ids,
                self.moe_buf,
                self.config.n_experts,
                unit_size=_ROUTING_TILE_M,
                num_local_tokens=self.S,
            )

    def _reduce(
        self,
        source: torch.Tensor,
        output: torch.Tensor,
        *,
        region: int,
        epoch_layer: int,
    ) -> torch.Tensor:
        if self.symmetric_allreduce is not None:
            return self.symmetric_allreduce.reduce(
                region,
                source,
                output,
                self.attention.step,
                epoch_layer,
            )

        import torch.distributed as dist

        output.copy_(source)
        dist.all_reduce(output, group=self.reduce_group)
        return output

    def _moe(
        self,
        hidden_states: torch.Tensor,
        epoch_layer: int,
        residual: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        with self._profile_stage("router_sort"):
            self._route_and_sort(hidden_states, epoch_layer)

        with self._profile_stage("routed_gemm1"):
            kimi_k3_mxfp4_gemm1(
                self.latent,
                self.w_ug,
                self.s_ug,
                sorted_expert_ids=self.sorted_expert_ids,
                num_valid_ids=self.num_valid_ids,
                sorted_token_ids=self.sorted_token_ids,
                output=self.inter_sorted,
                samples=self.S,
                situ_beta=self.config.situ_beta,
                situ_linear_beta=self.config.situ_linear_beta,
            )
        with self._profile_stage("routed_gemm2"):
            kimi_k3_mxfp4_gemm2(
                self.inter_sorted,
                self.w_dn,
                self.s_dn,
                sorted_expert_ids=self.sorted_expert_ids,
                num_valid_ids=self.num_valid_ids,
                sorted_token_ids=self.sorted_token_ids,
                sorted_weights=self.sorted_weights,
                output=self.routed_partial,
                samples=self.S,
                max_sorted=self.max_sorted,
            )

        if self.fused_tail is None:
            # Shared experts are tensor-parallel over their combined 6144-wide
            # intermediate; each rank computes its own 768-wide shard.
            self._shared_experts(hidden_states)

        with self._profile_stage("routed_reduce"):
            if self.fused_tail is not None:
                pass
            elif self.symmetric_allreduce is not None:
                self.symmetric_allreduce.reduce_rmsnorm(
                    self.routed_partial,
                    self.routed_reduced,
                    self.t["g_latent"],
                    self.latent_norm,
                    self.attention.step,
                    epoch_layer,
                )
            else:
                self._reduce(
                    self.routed_partial,
                    self.routed_reduced,
                    region=0,
                    epoch_layer=epoch_layer,
                )
        with self._profile_stage("latent_tail"):
            if self.symmetric_allreduce is None:
                compiled_rmsnorm_out(self.routed_reduced, self.t["g_latent"], self.latent_norm)
            if self.fused_tail is None:
                torch.mm(self.latent_norm, self.t["w_latent_up"].t(), out=self.tail)

        with self._profile_stage("final_reduce"):
            if self.fused_tail is not None:
                return self.fused_tail(
                    self.routed_partial,
                    self.routed_reduced,
                    self.t["g_latent"],
                    self.symmetric_allreduce.rmsnorm_scratch,
                    self.shared_mid,
                    self.latent_norm,
                    self.w_shared_dn,
                    self.s_shared_dn,
                    self.w_latent_up,
                    self.s_latent_up,
                    residual,
                    self.shared_partial,
                    self.tail,
                    self.final_partial,
                    self.moe_delta,
                    output,
                    self.symmetric_allreduce.peer_buffer.local_address,
                    self.symmetric_allreduce.peer_buffer.addresses,
                    self.attention.step,
                    epoch_layer,
                )
            if self.symmetric_allreduce is not None:
                return self.symmetric_allreduce.reduce_final(
                    self.shared_partial,
                    self.tail,
                    residual,
                    self.final_partial,
                    self.moe_delta,
                    output,
                    self.attention.step,
                    epoch_layer,
                )
            self.final_partial.copy_(self.shared_partial)
            lo = self.rank * self.hidden_shard
            hi = lo + self.hidden_shard
            self.final_partial[:, lo:hi].add_(self.tail)
            self._reduce(
                self.final_partial,
                self.moe_delta,
                region=1,
                epoch_layer=epoch_layer,
            )
            with self._profile_stage("output_add"):
                torch.add(residual, self.moe_delta, out=output)
            return output

    def _shared_experts(self, hidden_states: torch.Tensor) -> None:
        with self._profile_stage("shared_experts"):
            if self.fuse_shared_experts:
                torch.mm(self.shared_mid, self.t["w_shared_dn"].t(), out=self.shared_partial)
            else:
                torch.mm(hidden_states, self.t["w_shared_ug"].t(), out=self.shared_gu)
                self.shared_mid.copy_(situ(self.shared_gu, self.config.situ_beta, self.config.situ_linear_beta))
                torch.mm(self.shared_mid, self.t["w_shared_dn"].t(), out=self.shared_partial)

    def forward(
        self,
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
        cur_pos: torch.Tensor,
        kv_cache: torch.Tensor,
        pe_cache: torch.Tensor,
        indices: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        x_out: torch.Tensor | None = None,
        epoch_layer: int = 0,
        advance: bool = True,
    ) -> torch.Tensor:
        if (
            block_residual.ndim != 3
            or block_residual.shape[0] != self.S
            or block_residual.shape[2] != self.config.hidden
        ):
            raise ValueError(
                "block_residual must have shape "
                f"[{self.S}, blocks, {self.config.hidden}], got {tuple(block_residual.shape)}"
            )
        if block_residual.shape[1] <= self.block_write_idx:
            raise ValueError(f"block_residual needs index {self.block_write_idx}, got {block_residual.shape[1]} blocks")

        with self._profile_stage("pre_attn_res"):
            if self.inline_pre_attn:
                attention_input = prefix_sum
            elif self.fuse_attn_res:
                self.pre_attn_res(
                    prefix_sum,
                    prefix_sum,
                    block_residual,
                    self.t["g_self_res"],
                    self.t["w_self_res"],
                    self.t["g_in"],
                    self.pre_updated,
                    self.pre_attn,
                )
                attention_input = self.pre_attn
            else:
                self.pre_attn, _ = self._attn_res(
                    prefix_sum,
                    None,
                    block_residual,
                    self.t["g_self_res"],
                    self.t["w_self_res"],
                    self.t["g_in"],
                    self.previous_valid_blocks,
                    self.block_write_idx if self.is_block_write_layer else -1,
                )
                attention_input = self.pre_attn
        with self._profile_stage("attention"):
            self.attention.forward(
                attention_input,
                cur_pos,
                kv_cache,
                pe_cache,
                indices,
                cos,
                sin,
                x_out=self.attention_delta,
                layer=epoch_layer,
                advance=False,
            )

        post_prefix = self.attention_delta if self.is_block_write_layer else prefix_sum
        post_delta = None if self.is_block_write_layer else self.attention_delta
        with self._profile_stage("post_attn_res"):
            if self.fuse_attn_res:
                self.post_attn_res(
                    post_prefix,
                    (prefix_sum if self.inline_pre_attn else (post_prefix if post_delta is None else post_delta)),
                    block_residual,
                    self.t["g_mlp_res"],
                    self.t["w_mlp_res"],
                    self.t["g_post"],
                    self.updated_prefix,
                    self.moe_input,
                    self.latent_projection.activation,
                    self.latent_projection.activation_scale,
                )
            else:
                self.moe_input, self.updated_prefix = self._attn_res(
                    post_prefix,
                    post_delta,
                    block_residual,
                    self.t["g_mlp_res"],
                    self.t["w_mlp_res"],
                    self.t["g_post"],
                    self.previous_valid_blocks + int(self.is_block_write_layer),
                )
                self.latent_projection.quantize_input(self.moe_input)
        target = self.output if x_out is None else x_out
        self._moe(self.moe_input, epoch_layer, self.updated_prefix, target)
        if advance:
            self.advance_step()
        return target

    def advance_step(self) -> None:
        self.attention.advance_step()

    @contextmanager
    def capture(self):
        """Context used by callers recording the graph-stable decode path."""

        yield

    def close(self) -> None:
        if self.symmetric_allreduce is not None:
            self.symmetric_allreduce.close()
        self.attention.close()

    def release_packed_sources(self) -> None:
        """Keep only tensors read by the selected fused runtime path."""

        keep = {
            "bias",
            "g_in",
            "g_latent",
            "g_mlp_res",
            "g_post",
            "g_self_res",
            "w_mlp_res",
            "w_self_res",
        }
        self.t = {name: value for name, value in self.t.items() if name in keep}
        self.W = None
        release = getattr(self.attention, "release_packed_sources", None)
        if callable(release):
            release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class _KimiK3KdaStagedPath(_KimiK3MlaPath):
    """Internal staged Kimi-K3 KDA + latent-MoE reference path."""

    def _build_attention(
        self,
        weights: LayerWeights,
        samples: int,
        *,
        rank: int,
        npes: int,
        group,
        reduce_group,
        topk: int,
        reduce_backend: str,
        kv_cache_layout: KvCacheLayout | str,
        mtp: bool,
        agentic_batch_size: int,
        conv_state_layout: ConvStateLayout,
        attention_symmetric_allreduce,
        state_dtype: torch.dtype,
        defer_collectives: bool,
        packed_artifacts: dict[str, torch.Tensor] | None,
        monokernel_only: bool,
    ):
        del topk, kv_cache_layout
        return KimiK3KdaAttention(
            weights,
            samples,
            rank=rank,
            npes=npes,
            group=group,
            reduce_group=reduce_group,
            reduce_backend=reduce_backend,
            # The persistent attention kernel wins at S<=4.  At S=8 the
            # staged GEMMs retain better occupancy and remain the faster path.
            single_launch_attention=samples <= 4 or mtp,
            mtp=mtp,
            agentic_batch_size=agentic_batch_size,
            conv_state_layout=conv_state_layout,
            symmetric_allreduce=attention_symmetric_allreduce,
            state_dtype=state_dtype,
            defer_collectives=defer_collectives,
            packed_artifacts=packed_artifacts,
            monokernel_only=monokernel_only,
        )

    def forward(
        self,
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
        state_indices: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        *,
        x_out: torch.Tensor | None = None,
        epoch_layer: int = 0,
        advance: bool = True,
    ) -> torch.Tensor:
        """Run KDA decode with explicit slot-indexed state pools."""

        if (
            block_residual.ndim != 3
            or block_residual.shape[0] != self.S
            or block_residual.shape[2] != self.config.hidden
        ):
            raise ValueError(
                "block_residual must have shape "
                f"[{self.S}, blocks, {self.config.hidden}], got {tuple(block_residual.shape)}"
            )
        if block_residual.shape[1] <= self.block_write_idx:
            raise ValueError(
                f"block_residual needs index {self.block_write_idx}, " f"got {block_residual.shape[1]} blocks"
            )

        with self._profile_stage("pre_attn_res"):
            if self.inline_pre_attn:
                attention_input = prefix_sum
            elif self.fuse_attn_res:
                self.pre_attn_res(
                    prefix_sum,
                    prefix_sum,
                    block_residual,
                    self.t["g_self_res"],
                    self.t["w_self_res"],
                    self.t["g_in"],
                    self.pre_updated,
                    self.pre_attn,
                )
                attention_input = self.pre_attn
            else:
                self.pre_attn, _ = self._attn_res(
                    prefix_sum,
                    None,
                    block_residual,
                    self.t["g_self_res"],
                    self.t["w_self_res"],
                    self.t["g_in"],
                    self.previous_valid_blocks,
                    self.block_write_idx if self.is_block_write_layer else -1,
                )
                attention_input = self.pre_attn
        with self._profile_stage("attention"):
            self.attention.forward(
                attention_input,
                state_indices,
                conv_state,
                recurrent_state,
                x_out=self.attention_delta,
                layer=epoch_layer,
                advance=False,
            )

        post_prefix = self.attention_delta if self.is_block_write_layer else prefix_sum
        post_delta = None if self.is_block_write_layer else self.attention_delta
        with self._profile_stage("post_attn_res"):
            if self.fuse_attn_res:
                self.post_attn_res(
                    post_prefix,
                    (prefix_sum if self.inline_pre_attn else (post_prefix if post_delta is None else post_delta)),
                    block_residual,
                    self.t["g_mlp_res"],
                    self.t["w_mlp_res"],
                    self.t["g_post"],
                    self.updated_prefix,
                    self.moe_input,
                    self.latent_projection.activation,
                    self.latent_projection.activation_scale,
                )
            else:
                self.moe_input, self.updated_prefix = self._attn_res(
                    post_prefix,
                    post_delta,
                    block_residual,
                    self.t["g_mlp_res"],
                    self.t["w_mlp_res"],
                    self.t["g_post"],
                    self.previous_valid_blocks + int(self.is_block_write_layer),
                )
                self.latent_projection.quantize_input(self.moe_input)
        target = self.output if x_out is None else x_out
        self._moe(self.moe_input, epoch_layer, self.updated_prefix, target)
        if advance:
            self.advance_step()
        return target
