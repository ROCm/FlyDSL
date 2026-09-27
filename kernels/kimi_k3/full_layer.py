# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Host wrapper for the single-launch Kimi-K3 KDA decode layer."""

from __future__ import annotations

import torch

from kernels.mla_moe_layer.kimi_k3 import KimiK3KdaMoeLayer


class KimiK3KdaFullLayer(KimiK3KdaMoeLayer):
    """KDA, AttnRes, latent-MoE, TP reductions, and residual update in one launch."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.fuse_attn_res and self.symmetric_allreduce is not None:
            self.attention.configure_full_layer(self.layer_idx, full_moe=True)

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
        """Run one complete Kimi-K3 KDA decode layer."""

        if not self.attention.fuse_attn_res:
            return super().forward(
                prefix_sum,
                block_residual,
                state_indices,
                conv_state,
                recurrent_state,
                x_out=x_out,
                epoch_layer=epoch_layer,
                advance=advance,
            )
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
            block_residual=block_residual,
            pre_updated=self.pre_updated,
            pre_output=self.pre_attn,
            updated_prefix=self.updated_prefix,
            moe_input=self.moe_input,
            quantized_moe_input=self.latent_projection.activation,
            quantized_moe_scale=self.latent_projection.activation_scale,
            full_output=target,
            moe_symmetric=self.symmetric_allreduce.peer_buffer.local_address,
            moe_peers=self.symmetric_allreduce.peer_buffer.addresses,
            layer=epoch_layer,
            advance=False,
        )
        if advance:
            self.advance_step()
        return target


__all__ = ["KimiK3KdaFullLayer"]
