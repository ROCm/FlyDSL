# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Kimi-K3 TP8 KDA decode attention."""

from __future__ import annotations

import torch

from kernels.gemm.gemm_a16w16_gfx950 import gemm_a16w16
from kernels.mla_moe_layer.config import KIMI_K3_CONFIG, MAX_LAYERS_PER_STEP
from kernels.mla_moe_layer.kda_recurrence import KimiK3KdaConvRecurrence
from kernels.mla_moe_layer.reference import LayerWeights
from kernels.mla_moe_layer.symmetric_allreduce import SymmetricBf16Allreduce

_TP_SIZE = 8
_HEAD_DIM = 128
_CONV_WIDTH = 4
_INPUT_GEMM_ALIGNMENT = 32
_INPUT_GEMM_CONFIG = {
    "block_m": 16,
    "block_n": 32,
    "block_k": 128,
    "stages": 6,
    "split_k": 1,
    "m_waves": 1,
    "n_waves": 2,
    "k_waves": 1,
    "group_m": 0,
    "use_half_tile_interleaved": False,
}
_OUTPUT_GEMM_CONFIG = {
    "block_m": 16,
    "block_n": 64,
    "block_k": 128,
    "stages": 4,
    "split_k": 1,
    "m_waves": 1,
    "n_waves": 4,
    "k_waves": 1,
    "group_m": 0,
    "use_half_tile_interleaved": False,
}
_OUTPUT_GEMM_CONFIG_S8 = {
    "block_m": 32,
    "block_n": 64,
    "block_k": 128,
    "stages": 4,
    "split_k": 1,
    "m_waves": 2,
    "n_waves": 4,
    "k_waves": 1,
    "group_m": 0,
    "use_half_tile_interleaved": False,
}


class KimiK3KdaAttention:
    """Production-shape KDA decode shard with slot-indexed recurrent state."""

    def __init__(
        self,
        weights: LayerWeights,
        samples: int,
        *,
        rank: int,
        npes: int = _TP_SIZE,
        group=None,
        reduce_group=None,
        reduce_backend: str = "symmetric",
        launches_per_step: int = MAX_LAYERS_PER_STEP,
    ) -> None:
        config = weights.config
        if config != KIMI_K3_CONFIG:
            raise ValueError("KimiK3KdaAttention requires Kimi-K3 weights")
        if weights.heads != config.local_heads:
            raise ValueError(f"KDA requires {config.local_heads} local heads, got {weights.heads}")
        if npes != _TP_SIZE:
            raise ValueError(f"Kimi-K3 KDA currently requires TP8, got TP{npes}")
        if weights.rank != rank or weights.npes != npes:
            raise ValueError(
                f"weight shard is rank {weights.rank}/TP{weights.npes}, " f"requested rank {rank}/TP{npes}"
            )
        if reduce_backend not in {"symmetric", "nccl"}:
            raise ValueError(f"unsupported reduce backend {reduce_backend!r}; expected 'symmetric' or 'nccl'")
        if not 1 <= launches_per_step <= MAX_LAYERS_PER_STEP:
            raise ValueError(f"launches_per_step must be in [1, {MAX_LAYERS_PER_STEP}], " f"got {launches_per_step}")

        self.W = weights
        self.t = weights.t
        self.config = config
        self.S = samples
        self.rank = rank
        self.npes = npes
        self.reduce_group = reduce_group
        self.reduce_backend = reduce_backend
        self.launches_per_step = launches_per_step
        self.local_projection = config.local_heads * _HEAD_DIM

        expected = {
            "w_kda_in",
            "w_kda_fb",
            "w_kda_conv",
            "kda_a_log",
            "kda_dt_bias",
            "g_kda_out",
            "w_kda_o",
        }
        missing = sorted(expected.difference(self.t))
        if missing:
            raise ValueError(f"missing Kimi-K3 KDA weights: {', '.join(missing)}")

        fused_width = 4 * self.local_projection + config.local_heads + _HEAD_DIM
        shapes = {
            "w_kda_in": (fused_width, config.hidden),
            "w_kda_fb": (self.local_projection, _HEAD_DIM),
            "w_kda_conv": (3 * self.local_projection, _CONV_WIDTH),
            "kda_a_log": (config.local_heads,),
            "kda_dt_bias": (config.local_heads, _HEAD_DIM),
            "g_kda_out": (_HEAD_DIM,),
            "w_kda_o": (config.hidden, self.local_projection),
        }
        for name, shape in shapes.items():
            if self.t[name].shape != shape:
                raise ValueError(f"{name} must have shape {list(shape)}")
        bf16_weights = expected.difference({"kda_a_log"})
        if any(self.t[name].dtype != torch.bfloat16 for name in bf16_weights):
            raise ValueError("KDA projection, convolution, and norm weights must be BF16")
        if self.t["kda_a_log"].dtype != torch.float32:
            raise ValueError("kda_a_log must be FP32")
        if any(not self.t[name].is_contiguous() for name in expected):
            raise ValueError("KDA weights must be contiguous")

        device = self.t["w_kda_in"].device
        padded_fused_width = (fused_width + _INPUT_GEMM_ALIGNMENT - 1) // _INPUT_GEMM_ALIGNMENT
        padded_fused_width *= _INPUT_GEMM_ALIGNMENT
        self.fused_input_storage = torch.empty(
            samples,
            padded_fused_width,
            dtype=torch.bfloat16,
            device=device,
        )
        self.fused_input = self.fused_input_storage[:, :fused_width]
        self.w_kda_in_padded = torch.zeros(
            padded_fused_width,
            config.hidden,
            dtype=torch.bfloat16,
            device=device,
        )
        self.w_kda_in_padded[:fused_width].copy_(self.t["w_kda_in"])
        self.partial = torch.empty(samples, config.hidden, dtype=torch.bfloat16, device=device)
        self.output = torch.empty_like(self.partial)
        self.step = torch.zeros(1, dtype=torch.int32, device=device)
        self.normed = torch.empty(samples, config.local_heads, _HEAD_DIM, dtype=torch.bfloat16, device=device)
        self.core = KimiK3KdaConvRecurrence(samples, fuse_gate_projection=True)
        self.symmetric_allreduce = (
            SymmetricBf16Allreduce(
                (self.partial.numel(),),
                rank=rank,
                npes=npes,
                group=group,
            )
            if reduce_backend == "symmetric"
            else None
        )
        if reduce_group is None:
            raise ValueError("Kimi-K3 KDA attention requires a GPU-capable TP reduce_group")

    def forward(
        self,
        hidden_states: torch.Tensor,
        state_indices: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        *,
        x_out: torch.Tensor | None = None,
        layer: int = 0,
        advance: bool = True,
    ) -> torch.Tensor:
        """Run one decode token per sample and mutate both KDA state pools."""

        if not 0 <= layer < self.launches_per_step:
            raise ValueError(f"layer must be in [0, {self.launches_per_step}), got {layer}")
        expected_hidden = (self.S, self.config.hidden)
        if (
            hidden_states.shape != expected_hidden
            or hidden_states.dtype != torch.bfloat16
            or not hidden_states.is_contiguous()
        ):
            raise ValueError(f"hidden_states must be contiguous BF16 {list(expected_hidden)}")

        gemm_a16w16(
            hidden_states,
            self.w_kda_in_padded.T,
            out=self.fused_input_storage,
            user_kwargs=_INPUT_GEMM_CONFIG,
            layout="nt",
        )
        projection = self.local_projection
        heads = self.config.local_heads
        mixed_qkv = self.fused_input[:, : 3 * projection]
        output_gate = self.fused_input[:, 3 * projection : 4 * projection]
        beta = self.fused_input[:, 4 * projection : 4 * projection + heads].view(self.S, 1, heads)
        f_a = self.fused_input[:, 4 * projection + heads :]
        self.core(
            mixed_qkv,
            None,
            beta,
            self.t["w_kda_conv"],
            conv_state,
            self.t["kda_dt_bias"],
            self.t["kda_a_log"],
            state_indices,
            recurrent_state,
            output_gate.view(self.S, heads, _HEAD_DIM),
            self.t["g_kda_out"],
            self.normed.view(self.S, 1, heads, _HEAD_DIM),
            f_a=f_a,
            f_b_weight=self.t["w_kda_fb"],
        )
        target = self.output if x_out is None else x_out
        if target.shape != expected_hidden or target.dtype != torch.bfloat16 or not target.is_contiguous():
            raise ValueError(f"x_out must be contiguous BF16 {list(expected_hidden)}")
        if self.symmetric_allreduce is not None:
            gemm_a16w16(
                self.normed.view(self.S, projection),
                self.t["w_kda_o"].T,
                out=target,
                user_kwargs=_OUTPUT_GEMM_CONFIG_S8 if self.S == 8 else _OUTPUT_GEMM_CONFIG,
                layout="nt",
                symmetric_allreduce={
                    "symmetric": self.symmetric_allreduce.peer_buffer.local_address,
                    "peers": self.symmetric_allreduce.peer_buffer.addresses.data_ptr(),
                    "step": self.step.data_ptr(),
                    "rank": self.rank,
                    "layer": layer,
                    "npes": self.npes,
                    "max_pairs": self.symmetric_allreduce.max_pairs,
                    "layer_slots": MAX_LAYERS_PER_STEP,
                },
            )
        else:
            import torch.distributed as dist

            gemm_a16w16(
                self.normed.view(self.S, projection),
                self.t["w_kda_o"].T,
                out=self.partial,
                user_kwargs=_OUTPUT_GEMM_CONFIG,
                layout="nt",
            )
            target.copy_(self.partial)
            dist.all_reduce(target, group=self.reduce_group)
        if advance:
            self.advance_step()
        return target

    def advance_step(self) -> None:
        self.step.add_(1)

    def close(self) -> None:
        if self.symmetric_allreduce is not None:
            self.symmetric_allreduce.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
