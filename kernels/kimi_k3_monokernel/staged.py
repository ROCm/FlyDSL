# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Staged MLA/KDA baselines retained beside the Kimi-K3 MonoKernel."""

from __future__ import annotations

from contextlib import contextmanager

import torch

from kernels.kimi_k3_monokernel.attn_res import KimiK3AttnRes
from kernels.kimi_k3_monokernel.kda import KimiK3KdaAttention
from kernels.kimi_k3_monokernel.mla import KimiK3MlaAttention
from kernels.kimi_k3_monokernel.moe import kimi_k3_mxfp4_gemm1, kimi_k3_mxfp4_gemm2
from kernels.kimi_k3_monokernel.mxfp8_linear import Mxfp8Linear
from kernels.kimi_k3_monokernel.router import SigmoidTopkRouter
from kernels.kimi_k3_monokernel.router_projection import FusedRouterProjection
from kernels.kimi_k3_monokernel.routed_pipeline import RoutedTilePipeline
from kernels.kimi_k3_monokernel.symmetric_allreduce import SymmetricBf16Allreduce
from kernels.kimi_k3_monokernel.tail import FusedKimiK3Tail
from kernels.kimi_k3_monokernel.torch_fusions import (
    CudaStageProfiler,
    compiled_attn_res_no_delta,
    compiled_attn_res_with_delta,
    compiled_rmsnorm,
    compiled_rmsnorm_out,
    rmsnorm,
    situ,
)
from kernels.moe.moe_sorting_kernel import moe_sorting_flydsl
from kernels.monokernel.config import EPS, KIMI_K3_CONFIG, KvCacheLayout
from kernels.monokernel.formats import quantize_mxfp8
from kernels.monokernel.packing import (
    pack_a16w4_scale,
    pack_a16w4_weight,
    pack_bf16,
    pack_mxfp8_scale,
    pack_mxfp8_weight,
)
from kernels.monokernel.weights import LayerWeights

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
        routed_pipeline: bool = False,
        routed_producers: int | None = None,
        routed_prefetch: int | None = None,
        routed_grid: int = 256,
        routed_sample_group: int = 0,
        routed_down_tile: int = 0,
        routed_tp_mode: str = "complete",
        routed_up_order: str = "sample",
        routed_soft_profile: int = 0,
        routed_route_prefetch: bool = False,
        routed_trace: bool = False,
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
        self.reduce_group = reduce_group
        if reduce_backend not in {"symmetric", "nccl"}:
            raise ValueError(f"unsupported reduce backend {reduce_backend!r}; expected 'symmetric' or 'nccl'")
        if routed_pipeline and (reduce_backend != "symmetric" or not fuse_shared_experts):
            raise ValueError("routed_pipeline requires the symmetric fused tail")
        self.reduce_backend = reduce_backend
        self.layer_idx = layer_idx
        self.topk = topk
        self.fuse_attn_res = fuse_attn_res
        self.inline_pre_attn = fuse_attn_res and layer_idx == 0
        self.fuse_router = fuse_router
        self.fuse_shared_experts = fuse_shared_experts
        self.routed_hidden = config.routed_hidden
        self.shared_inter = config.shared_inter
        self.hidden_shard = config.hidden // npes

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
        )
        device = torch.device("cuda", torch.cuda.current_device())
        self.pre_attn = torch.empty(samples, config.hidden, dtype=torch.bfloat16, device=device)
        self.pre_updated = torch.empty_like(self.pre_attn)
        self.moe_input = torch.empty_like(self.pre_attn)
        self.updated_prefix = torch.empty_like(self.pre_attn)
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

        # A16W4 production layouts.  Raw checkpoint-format tensors remain in W
        # for reference checks; these packed copies are launch-ready.
        self.w_ug = pack_a16w4_weight(self.t["w_ug"])
        self.s_ug = pack_a16w4_scale(self.t["s_ug"])
        self.w_dn = pack_a16w4_weight(self.t["w_dn"])
        self.s_dn = pack_a16w4_scale(self.t["s_dn"])
        self.w_router = pack_bf16(self.t["w_r"])
        latent_weight, self.s_latent_down = quantize_mxfp8(self.t["w_latent_down"])
        shared_weight, self.s_shared_ug = quantize_mxfp8(self.t["w_shared_ug"])
        shared_down_weight, self.s_shared_dn = quantize_mxfp8(self.t["w_shared_dn"])
        latent_up_weight, self.s_latent_up = quantize_mxfp8(self.t["w_latent_up"])
        self.latent_projection = Mxfp8Linear(latent_weight, self.s_latent_down, samples)
        self.shared_projection = Mxfp8Linear(shared_weight, self.s_shared_ug, samples)
        self.w_latent_down = self.latent_projection.weight
        self.s_latent_down = self.latent_projection.scale
        self.w_shared_ug = self.shared_projection.weight
        self.s_shared_ug = self.shared_projection.scale
        self.w_shared_dn = pack_mxfp8_weight(shared_down_weight)
        self.s_shared_dn = pack_mxfp8_scale(self.s_shared_dn)
        self.w_latent_up = pack_mxfp8_weight(latent_up_weight)
        self.s_latent_up = pack_mxfp8_scale(self.s_latent_up)

        # At most one padded BM tile is needed per selected route: there can be
        # no more active experts than routes.  The old ``routes + E*(BM-1)``
        # bound made the expert GEMMs launch tens of thousands of empty CTAs at
        # low token counts.
        max_sorted = samples * config.top_k * _ROUTING_TILE_M
        max_blocks = (max_sorted + _ROUTING_TILE_M - 1) // _ROUTING_TILE_M
        self.max_sorted = max_sorted
        self.sorted_token_ids = torch.empty(max_sorted, dtype=torch.int32, device=device)
        self.sorted_weights = torch.empty(max_sorted, dtype=torch.float32, device=device)
        self.sorted_expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)
        self.num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
        self.inter_sorted = torch.empty(max_sorted, config.inter, dtype=torch.bfloat16, device=device)

        self.router_logits = torch.empty(samples, config.n_experts, dtype=torch.bfloat16, device=device)
        self.router_scores = torch.empty(samples, config.n_experts, dtype=torch.float32, device=device)
        self.topk_keys = torch.empty(samples, config.top_k, dtype=torch.float32, device=device)
        self.topk_ids_i64 = torch.empty(samples, config.top_k, dtype=torch.int64, device=device)
        self.topk_ids = torch.empty(samples, config.top_k, dtype=torch.int32, device=device)
        self.topk_weights = torch.empty(samples, config.top_k, dtype=torch.float32, device=device)
        self.router_select = SigmoidTopkRouter(config.n_experts, config.top_k, samples)
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
        self.router_score_mailbox = torch.zeros(
            samples * config.n_experts * 2,
            dtype=torch.int32,
            device=device,
        )
        self.latent = torch.empty(samples, self.routed_hidden, dtype=torch.bfloat16, device=device)
        self.routed_partial = torch.empty_like(self.latent)
        self.routed_reduced = torch.empty_like(self.latent)
        self.latent_norm = torch.empty_like(self.latent)
        self.shared_gu = torch.empty(samples, 2 * self.shared_inter, dtype=torch.bfloat16, device=device)
        self.shared_mid = torch.empty(samples, self.shared_inter, dtype=torch.bfloat16, device=device)
        self.shared_partial = torch.empty(samples, config.hidden, dtype=torch.bfloat16, device=device)
        self.tail = torch.empty(samples, self.hidden_shard, dtype=torch.bfloat16, device=device)
        self.final_partial = torch.empty_like(self.shared_partial)
        self.moe_delta = torch.empty_like(self.shared_partial)
        self.output = torch.empty_like(self.shared_partial)
        self.attention_delta = torch.empty_like(self.shared_partial)
        self._profiler = CudaStageProfiler()
        self.symmetric_allreduce = (
            SymmetricBf16Allreduce(
                (self.routed_partial.numel(), self.final_partial.numel()),
                rank=rank,
                npes=npes,
                group=group,
                final_hidden=config.hidden,
                final_shard_width=self.hidden_shard,
                rmsnorm_width=self.routed_hidden,
            )
            if reduce_backend == "symmetric"
            else None
        )
        self.routed_pipeline = (
            RoutedTilePipeline(
                samples,
                self.t["w_ug"],
                self.t["s_ug"],
                self.t["w_dn"],
                self.t["s_dn"],
                npes=npes,
                rank=rank,
                max_pairs=self.symmetric_allreduce.max_pairs,
                producers=routed_producers,
                prefetch=routed_prefetch,
                grid=routed_grid,
                sample_group=routed_sample_group,
                down_tile=routed_down_tile,
                tp_mode=routed_tp_mode,
                up_order=routed_up_order,
                soft_profile=routed_soft_profile,
                route_prefetch=routed_route_prefetch,
                trace=routed_trace,
            )
            if routed_pipeline
            else None
        )
        self.fused_tail = (
            FusedKimiK3Tail(
                samples,
                config.hidden,
                self.routed_hidden,
                self.shared_inter,
                rank,
                npes,
                self.symmetric_allreduce.max_pairs,
                routed_pre_reduced=routed_pipeline,
            )
            if self.symmetric_allreduce is not None and self.fuse_shared_experts
            else None
        )

        # The sorter also clears this output buffer before atomic stage2.
        self.moe_buf = self.routed_partial
        if reduce_group is None:
            raise ValueError("the Kimi-K3 staged path requires a GPU-capable TP reduce_group")

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
    ):
        del reduce_group, reduce_backend, mtp
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

        if self.routed_pipeline is not None:
            with self._profile_stage("routed_pipeline"):
                self.routed_pipeline(
                    self.latent,
                    self.topk_ids,
                    self.topk_weights,
                    self.routed_partial,
                    self.routed_reduced,
                    self.attention.step,
                    epoch_layer,
                    symmetric=self.symmetric_allreduce.peer_buffer.local_address,
                    peers=self.symmetric_allreduce.peer_buffer.addresses,
                )
        else:
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
