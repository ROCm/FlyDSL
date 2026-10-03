# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Host wrapper for the single-launch Kimi-K3 decode MonoKernel."""

from __future__ import annotations

import torch

from kernels.monokernel.config import ConvStateLayout
from kernels.monokernel.k3.staged import _KimiK3KdaStagedPath
from kernels.monokernel.symmetric_allreduce import SymmetricBf16Allreduce
from kernels.monokernel.weights import LayerWeights


class KimiK3MonoKernel(_KimiK3KdaStagedPath):
    """Run KDA, AttnRes, latent-MoE, TP reductions, and residual update in one launch."""

    def __init__(
        self,
        weights: LayerWeights,
        samples: int,
        *,
        layer_idx: int,
        rank: int,
        npes: int = 8,
        group=None,
        reduce_group=None,
        mtp: bool = False,
        agentic_batch_size: int = 0,
        conv_state_layout: ConvStateLayout = ConvStateLayout.CHANNEL_MAJOR,
        attention_symmetric_allreduce: SymmetricBf16Allreduce | None = None,
        moe_symmetric_allreduce: SymmetricBf16Allreduce | None = None,
        state_dtype: torch.dtype = torch.float32,
        defer_collectives: bool = False,
        packed_artifacts: dict[str, object] | None = None,
    ) -> None:
        if agentic_batch_size:
            if (
                not mtp
                or samples != agentic_batch_size * 8
                or state_dtype is not torch.float16
            ):
                raise ValueError(
                    "Kimi Agentic MonoKernel requires q=8 MTP rows "
                    "and FP16 recurrent state"
                )
        elif state_dtype is not torch.float32:
            raise ValueError("Kimi-K3 single-launch MonoKernel requires FP32 state")
        super().__init__(
            weights,
            samples,
            layer_idx=layer_idx,
            rank=rank,
            npes=npes,
            group=group,
            reduce_group=reduce_group,
            fuse_attn_res=True,
            fuse_router=True,
            fuse_shared_experts=True,
            reduce_backend="symmetric",
            mtp=mtp,
            agentic_batch_size=agentic_batch_size,
            conv_state_layout=conv_state_layout,
            attention_symmetric_allreduce=attention_symmetric_allreduce,
            moe_symmetric_allreduce=moe_symmetric_allreduce,
            state_dtype=state_dtype,
            defer_collectives=defer_collectives,
            packed_artifacts=packed_artifacts,
            monokernel_only=True,
        )
        self.attention.configure_monokernel(
            layer_idx,
            fuse_moe=True,
        )
        canonical_moe = getattr(self.attention, "moe_packed", None)
        if canonical_moe is not None:
            self.moe_packed = canonical_moe
            self.w_router = canonical_moe["w_r"]
            self.w_latent_down = canonical_moe["w_latent_down"]
            self.s_latent_down = canonical_moe["s_latent_down"]
            self.w_shared_ug = canonical_moe["w_shared_ug"]
            self.s_shared_ug = canonical_moe["s_shared_ug"]
            self.w_ug = canonical_moe["w_ug"]
            self.s_ug = canonical_moe["s_ug"]
            self.w_dn = canonical_moe["w_dn"]
            self.s_dn = canonical_moe["s_dn"]
            self.w_shared_dn = canonical_moe["w_shared_dn"]
            self.s_shared_dn = canonical_moe["s_shared_dn"]
            self.w_latent_up = canonical_moe["w_latent_up"]
            self.s_latent_up = canonical_moe["s_latent_up"]
            self.latent_projection.weight = self.w_latent_down
            self.latent_projection.scale = self.s_latent_down

    def forward(
        self,
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
        state_indices: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        *,
        x_out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        epoch_layer: int = 0,
        advance: bool = True,
    ) -> torch.Tensor:
        """Run one complete Kimi-K3 decode layer."""

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

        target = self.output if x_out is None else x_out
        self.attention.forward(
            prefix_sum,
            state_indices,
            conv_state,
            recurrent_state,
            x_out=self.attention_delta,
            num_accepted_tokens=num_accepted_tokens,
            block_residual=block_residual,
            pre_updated=self.pre_updated,
            pre_output=self.pre_attn,
            updated_prefix=self.updated_prefix,
            moe_input=self.moe_input,
            quantized_moe_input=self.latent_projection.activation,
            quantized_moe_scale=self.latent_projection.activation_scale,
            monokernel_output=target,
            moe_symmetric=self.symmetric_allreduce.peer_buffer.local_address,
            moe_peers=self.symmetric_allreduce.peer_buffer.addresses,
            layer=epoch_layer,
            advance=advance,
        )
        return target


class KimiK3StagedAgenticOp:
    """Explicit TP8/DCP1 q8 production route using a fused KDA front and staged MoE tails."""

    def __init__(
        self,
        weights: LayerWeights,
        *,
        batch_size: int,
        layer_idx: int,
        rank: int,
        npes: int = 8,
        group=None,
        reduce_group=None,
        packed_artifacts: dict[str, object] | None = None,
    ) -> None:
        if batch_size not in (1, 2, 4):
            raise ValueError(f"staged Agentic KDA supports B1/B2/B4, got B{batch_size}")
        self.batch_size = batch_size
        self.S = batch_size * 8
        common = dict(
            layer_idx=layer_idx,
            rank=rank,
            npes=npes,
            group=group,
            reduce_group=reduce_group,
            mtp=True,
            conv_state_layout=ConvStateLayout.TIME_MAJOR,
            state_dtype=torch.float16,
        )
        self.front = _KimiK3KdaStagedPath(
            weights,
            self.S,
            agentic_batch_size=batch_size,
            packed_artifacts=packed_artifacts,
            monokernel_only=True,
            **common,
        )
        self.front.attention.configure_monokernel(layer_idx, fuse_moe=False)
        packed = self.front.packed_artifacts()
        self.tails = tuple(
            _KimiK3KdaStagedPath(
                weights,
                8,
                agentic_batch_size=1,
                packed_artifacts=packed,
                **common,
            )
            for _ in range(batch_size)
        )
        for tail in self.tails:
            # The staged router consumes BF16 correction bias; the fused
            # application kernel performs the same narrowing internally.
            tail.t = dict(tail.t)
            tail.t["bias"] = tail.t["bias"].to(torch.bfloat16)
            tail.attention.step = self.front.attention.step
        self.output = torch.empty_like(self.front.output)

    def forward(
        self,
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
        snapshot_slots: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        *,
        num_accepted_tokens: torch.Tensor,
        x_out: torch.Tensor | None = None,
        epoch_layer: int = 0,
        advance: bool = True,
    ) -> torch.Tensor:
        target = self.output if x_out is None else x_out
        self.front.attention.forward(
            prefix_sum,
            snapshot_slots,
            conv_state,
            recurrent_state,
            x_out=self.front.attention_delta,
            num_accepted_tokens=num_accepted_tokens,
            block_residual=block_residual,
            pre_updated=self.front.pre_updated,
            pre_output=self.front.pre_attn,
            updated_prefix=self.front.updated_prefix,
            moe_input=self.front.moe_input,
            quantized_moe_input=self.front.latent_projection.activation,
            quantized_moe_scale=self.front.latent_projection.activation_scale,
            layer=epoch_layer,
            advance=False,
        )
        for request, tail in enumerate(self.tails):
            rows = slice(request * 8, (request + 1) * 8)
            tail._moe(
                self.front.moe_input[rows],
                epoch_layer,
                self.front.updated_prefix[rows],
                target[rows],
            )
        if advance:
            self.front.advance_step()
        return target

    def packed_artifacts(self) -> dict[str, object]:
        return self.front.packed_artifacts()

    def close(self) -> None:
        for tail in self.tails:
            tail.close()
        self.front.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


__all__ = ["KimiK3MonoKernel", "KimiK3StagedAgenticOp"]
