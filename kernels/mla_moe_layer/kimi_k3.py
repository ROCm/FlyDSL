# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Kimi-K3 TP8 full-attention MLA + AttnRes + latent-MoE layer."""

from __future__ import annotations

from contextlib import contextmanager

import torch

from kernels.mla_moe_layer.config import EPS, KIMI_K3_CONFIG
from kernels.mla_moe_layer.indexed_layer import KimiK3MlaLayer
from kernels.mla_moe_layer.packing import pack_a16w4_scale, pack_a16w4_weight, pack_bf16
from kernels.mla_moe_layer.reference import LayerWeights
from kernels.mla_moe_layer.router import SigmoidTopkRouter
from kernels.mla_moe_layer.router_projection import FusedRouterProjection
from kernels.mla_moe_layer.symmetric_allreduce import SymmetricBf16Allreduce
from kernels.mla_moe_layer.torch_fusions import (
    CudaStageProfiler,
    compiled_attn_res_no_delta,
    compiled_attn_res_with_delta,
    compiled_rmsnorm,
    compiled_rmsnorm_out,
    compiled_shared_experts,
    rmsnorm,
    situ,
)
from kernels.moe.moe_2stage_a16wmix import flydsl_a16w4_gemm1, flydsl_a16w4_gemm2
from kernels.moe.moe_sorting_kernel import moe_sorting_flydsl

_TP_SIZE = 8
_ROUTING_TILE_M = 16


class KimiK3MlaMoeLayer:
    """One production TP8 Kimi-K3 decode layer.

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
        timeline: bool = False,
        fuse_attn_res: bool = True,
        fuse_router: bool = True,
        fuse_shared_experts: bool = True,
        reduce_backend: str = "symmetric",
    ) -> None:
        config = weights.config
        if config != KIMI_K3_CONFIG:
            raise ValueError("KimiK3MlaMoeLayer requires Kimi-K3 weights")
        if npes != _TP_SIZE:
            raise ValueError(f"Kimi-K3 full MLA+MoE currently requires TP8, got TP{npes}")
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
        self.reduce_backend = reduce_backend
        self.layer_idx = layer_idx
        self.topk = topk
        self.fuse_attn_res = fuse_attn_res
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
            raise ValueError(f"missing Kimi-K3 full-layer weights: {', '.join(missing)}")
        if self.t["w_latent_up"].shape != (self.hidden_shard, self.routed_hidden):
            raise ValueError(
                f"w_latent_up must be the rank-local output-row shard [{self.hidden_shard}, {self.routed_hidden}]"
            )

        self.attention = KimiK3MlaLayer(
            weights,
            samples,
            rank=rank,
            npes=npes,
            group=group,
            topk=topk,
            timeline=timeline,
            moe_mode="a16w4",
        )
        device = torch.device("cuda", torch.cuda.current_device())

        # A16W4 production layouts.  Raw checkpoint-format tensors remain in W
        # for reference checks; these packed copies are launch-ready.
        self.w_ug = pack_a16w4_weight(self.t["w_ug"])
        self.s_ug = pack_a16w4_scale(self.t["s_ug"])
        self.w_dn = pack_a16w4_weight(self.t["w_dn"])
        self.s_dn = pack_a16w4_scale(self.t["s_dn"])
        self.w_router = pack_bf16(self.t["w_r"])

        max_sorted = samples * config.top_k + config.n_experts * (_ROUTING_TILE_M - 1)
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
        self.router_projection = FusedRouterProjection(config.hidden, config.n_experts, config.top_k, samples)
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
            )
            if reduce_backend == "symmetric"
            else None
        )

        # The sorter also clears this output buffer before atomic stage2.
        self.moe_buf = self.routed_partial
        if reduce_group is None:
            raise ValueError("Kimi-K3 full MLA+MoE requires a GPU-capable TP reduce_group")

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

    def _route_and_sort(self, hidden_states: torch.Tensor, epoch_layer: int) -> None:
        if self.fuse_router:
            self.router_projection(
                hidden_states,
                self.w_router,
                self.t["bias"],
                self.router_score_mailbox,
                self.router_scores,
                self.topk_ids,
                self.topk_weights,
                self.attention.step,
                epoch_layer,
            )
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

    def _moe(self, hidden_states: torch.Tensor, epoch_layer: int) -> torch.Tensor:
        with self._profile_stage("router_sort"):
            self._route_and_sort(hidden_states, epoch_layer)
        with self._profile_stage("latent_down"):
            torch.mm(hidden_states, self.t["w_latent_down"].t(), out=self.latent)

        # Shared experts are tensor-parallel over their combined 6144-wide
        # intermediate; each rank computes its own 768-wide shard.
        self._shared_experts(hidden_states)

        with self._profile_stage("routed_gemm1"):
            flydsl_a16w4_gemm1(
                a_bf16=self.latent,
                w1_u8=self.w_ug,
                w1_scale_u8=self.s_ug,
                sorted_expert_ids=self.sorted_expert_ids,
                cumsum_tensor=self.num_valid_ids,
                m_indices=self.sorted_token_ids,
                inter_sorted_bf16=self.inter_sorted,
                n_tokens=self.S,
                NE=self.config.n_experts,
                D_HIDDEN=self.routed_hidden,
                D_INTER=self.config.inter,
                topk=self.config.top_k,
                tile_m=_ROUTING_TILE_M,
                act="situv2",
                situ_beta=self.config.situ_beta,
                situ_linear_beta=self.config.situ_linear_beta,
                w_dtype="mxfp4",
                use_csv_config=True,
            )
        with self._profile_stage("routed_gemm2"):
            flydsl_a16w4_gemm2(
                inter_sorted_bf16=self.inter_sorted,
                w2_u8=self.w_dn,
                w2_scale_u8=self.s_dn,
                sorted_expert_ids=self.sorted_expert_ids,
                cumsum_tensor=self.num_valid_ids,
                sorted_token_ids=self.sorted_token_ids,
                sorted_weights=self.sorted_weights,
                flat_out=self.routed_partial,
                M_logical=self.S,
                max_sorted=self.max_sorted,
                NE=self.config.n_experts,
                D_HIDDEN=self.routed_hidden,
                D_INTER=self.config.inter,
                topk=self.config.top_k,
                tile_m=_ROUTING_TILE_M,
                tile_k=128,
                w_dtype="mxfp4",
                use_csv_config=True,
            )

        with self._profile_stage("routed_reduce"):
            self._reduce(
                self.routed_partial,
                self.routed_reduced,
                region=0,
                epoch_layer=epoch_layer,
            )
        with self._profile_stage("latent_tail"):
            compiled_rmsnorm_out(self.routed_reduced, self.t["g_latent"], self.latent_norm)
            torch.mm(self.latent_norm, self.t["w_latent_up"].t(), out=self.tail)

        with self._profile_stage("final_reduce"):
            self.final_partial.copy_(self.shared_partial)
            lo = self.rank * self.hidden_shard
            hi = lo + self.hidden_shard
            self.final_partial[:, lo:hi].add_(self.tail)
            return self._reduce(
                self.final_partial,
                self.moe_delta,
                region=1,
                epoch_layer=epoch_layer,
            )

    def _shared_experts(self, hidden_states: torch.Tensor) -> None:
        with self._profile_stage("shared_experts"):
            if self.fuse_shared_experts:
                compiled_shared_experts(
                    hidden_states,
                    self.t["w_shared_ug"],
                    self.t["w_shared_dn"],
                    self.shared_gu,
                    self.shared_mid,
                    self.shared_partial,
                    self.config.situ_beta,
                    self.config.situ_linear_beta,
                )
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
        with self._profile_stage("attention"):
            self.attention.forward(
                self.pre_attn,
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
            self.moe_input, self.updated_prefix = self._attn_res(
                post_prefix,
                post_delta,
                block_residual,
                self.t["g_mlp_res"],
                self.t["w_mlp_res"],
                self.t["g_post"],
                self.previous_valid_blocks + int(self.is_block_write_layer),
            )
        moe_delta = self._moe(self.moe_input, epoch_layer)
        target = self.output if x_out is None else x_out
        with self._profile_stage("output_add"):
            torch.add(self.updated_prefix, moe_delta, out=target)
        if advance:
            self.advance_step()
        return target

    def advance_step(self) -> None:
        self.attention.advance_step()

    @contextmanager
    def capture(self):
        """Context used by callers recording the graph-stable full layer."""

        yield

    def close(self) -> None:
        if self.symmetric_allreduce is not None:
            self.symmetric_allreduce.close()
        self.attention.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
