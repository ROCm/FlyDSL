# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host wrapper for the Kimi-K3 MLA attention path."""

from __future__ import annotations

import torch

from kernels.kimi_k3_monokernel.mla_kernel import build_kimi_k3_mla_attention
from kernels.monokernel.config import (
    KIMI_K3_CONFIG,
    MAX_LAYERS_PER_STEP,
    KvCacheLayout,
    MoeMode,
    as_kv_cache_layout,
    validate_shard,
)
from kernels.monokernel.layout import layout
from kernels.monokernel.packing import pack_layer_weights
from kernels.monokernel.runtime import SymmetricPeerBuffer
from kernels.monokernel.weights import LayerWeights

__all__ = ["KimiK3MlaAttention"]


class KimiK3MlaAttention:
    """One TP rank of Kimi-K3 MLA attention, excluding AttnRes and MoE."""

    def __init__(
        self,
        weights: LayerWeights,
        samples: int,
        *,
        rank: int = 0,
        npes: int = 1,
        group=None,
        topk: int = 2048,
        launches_per_step: int = MAX_LAYERS_PER_STEP,
        attention_input_norm: bool = False,
        kv_cache_layout: KvCacheLayout | str = KvCacheLayout.SPLIT,
    ) -> None:
        if weights.config != KIMI_K3_CONFIG:
            raise ValueError(f"KimiK3MlaAttention requires Kimi-K3 weights, got {weights.config.name!r}")
        validate_shard(samples, weights.heads, rank, npes, topk, KIMI_K3_CONFIG)
        if not 1 <= launches_per_step <= MAX_LAYERS_PER_STEP:
            raise ValueError(f"launches_per_step must be in [1, {MAX_LAYERS_PER_STEP}], got {launches_per_step}")

        self.W = weights
        self.S = samples
        self.rank = rank
        self.npes = npes
        self.topk = topk
        self.launches_per_step = launches_per_step
        self.kv_cache_layout = as_kv_cache_layout(kv_cache_layout)
        self.packed = pack_layer_weights(
            weights.t,
            MoeMode.A16W4,
            KIMI_K3_CONFIG,
            attention_only=True,
        )
        dedicated_input_norm = attention_input_norm and samples > 4
        self.scratch_layout, symmetric_layout = layout(
            samples,
            weights.heads,
            npes,
            topk,
            MoeMode.A16W4,
            KIMI_K3_CONFIG,
            attention_only=True,
            dedicated_input_norm=dedicated_input_norm,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        self.scratch = torch.zeros(self.scratch_layout["_bytes"], dtype=torch.uint8, device=device)
        self.peer_buffer = SymmetricPeerBuffer(symmetric_layout["_bytes"], rank=rank, npes=npes, group=group)
        self.launch = build_kimi_k3_mla_attention(
            samples,
            weights.heads,
            npes,
            topk,
            launches_per_step=launches_per_step,
            attention_input_norm=attention_input_norm,
            kv_cache_layout=self.kv_cache_layout,
        )
        self.step = torch.zeros(1, dtype=torch.int32, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cur_pos: torch.Tensor,
        kv_cache: torch.Tensor,
        pe_cache: torch.Tensor,
        sparse_indices: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        x_out: torch.Tensor | None = None,
        layer: int = 0,
        advance: bool = True,
    ) -> torch.Tensor:
        """Launch one Kimi-K3 MLA attention invocation."""

        if not 0 <= layer < self.launches_per_step:
            raise ValueError(f"layer must be in [0, {self.launches_per_step}), got {layer}")
        if self.kv_cache_layout is KvCacheLayout.ATOM:
            cache_width = KIMI_K3_CONFIG.kv_lora + KIMI_K3_CONFIG.pe_dim
            if kv_cache.ndim != 2 or kv_cache.shape[1] != cache_width:
                raise ValueError(f"ATOM KV cache must have shape [tokens, {cache_width}], got {tuple(kv_cache.shape)}")
            if kv_cache.data_ptr() != pe_cache.data_ptr():
                raise ValueError("ATOM KV cache layout requires the same fused tensor for kv_cache and pe_cache")

        tensors = dict(self.W.t, **self.packed)
        if x_out is None:
            x_out = torch.empty(self.S, KIMI_K3_CONFIG.hidden, dtype=torch.bfloat16, device=hidden_states.device)
        pointer = lambda value: value.data_ptr()  # noqa: E731
        self.launch(
            pointer(hidden_states),
            pointer(x_out),
            pointer(cur_pos),
            pointer(kv_cache),
            pointer(pe_cache),
            pointer(sparse_indices),
            pointer(cos),
            pointer(sin),
            pointer(tensors["g_in"]),
            pointer(tensors["g_q"]),
            pointer(tensors["g_kv"]),
            pointer(tensors["g_post"]),
            pointer(tensors["w_qkv_a"]),
            pointer(tensors["s_qkv_a"]),
            pointer(tensors["w_q_b"]),
            pointer(tensors["s_q_b"]),
            pointer(tensors["w_uk"]),
            pointer(tensors["s_uk"]),
            pointer(tensors["w_uv"]),
            pointer(tensors["s_uv"]),
            pointer(tensors["w_o"]),
            pointer(tensors["s_o"]),
            0,
            0,
            0,
            0,
            0,
            0,
            pointer(self.scratch),
            self.peer_buffer.local_address,
            pointer(self.peer_buffer.addresses),
            0,
            pointer(self.step),
            self.rank,
            layer,
            stream=torch.cuda.current_stream(),
        )
        if advance:
            self.advance_step()
        return x_out

    def advance_step(self) -> None:
        self.step.add_(1)

    def close(self) -> None:
        """Release this rank's remote HIP IPC mappings."""

        self.peer_buffer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
