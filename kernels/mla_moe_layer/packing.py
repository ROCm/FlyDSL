# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host-side weight packing for GLM-5 and Kimi-K3 MLA launch wrappers."""

from __future__ import annotations

import torch

from kernels.mla_moe_layer.config import (
    GLM5_CONFIG,
    AttentionWeight,
    ExpertWeight,
    LayerConfig,
    MoeMode,
    as_layer_config,
    moe_format,
)


def pack_fp8(q: torch.Tensor) -> torch.Tensor:
    """Pack FP8 ``[..., N, K]`` for the kernel's 16-row, 64-K MFMA tiles."""

    *lead, rows, k = q.shape
    if rows % 16 or k % 64:
        raise ValueError(f"FP8 matrix dimensions must be divisible by (16, 64), got {(rows, k)}")
    w8 = q.view(torch.uint8).reshape(*lead, rows // 16, 16, k // 64, 2, 4, 8)
    nlead = len(lead)
    order = list(range(nlead)) + [nlead + position for position in (0, 2, 4, 1, 3, 5)]
    return w8.permute(*order).contiguous().view(-1)


def pack_bf16(w: torch.Tensor) -> torch.Tensor:
    """Pack BF16 ``[N, K]`` for the kernel's 16-row, 64-K MFMA tiles."""

    if w.ndim != 2:
        raise ValueError(f"BF16 packing expects a matrix, got shape {tuple(w.shape)}")
    rows, k = w.shape
    if rows % 16 or k % 64:
        raise ValueError(f"BF16 matrix dimensions must be divisible by (16, 64), got {(rows, k)}")
    w16 = w.view(torch.int16).reshape(rows // 16, 16, k // 64, 2, 4, 8)
    return w16.permute(0, 2, 3, 4, 1, 5).contiguous().view(-1)


def pack_mxfp4(q: torch.Tensor) -> torch.Tensor:
    """Pack MXFP4 for four BF16 MFMA K32 steps in each 128-K tile.

    A lane loads four dwords. Dword ``s`` holds its eight FP4 values for K32
    step ``s``, matching :func:`pack_bf16` after in-kernel FP4 conversion.
    """

    q = q.view(torch.uint8)
    *lead, rows, packed_k = q.shape
    k = packed_k * 2
    if rows % 16 or k % 128:
        raise ValueError(f"MXFP4 matrix dimensions must be divisible by (16, 128), got {(rows, k)}")
    w4 = q.reshape(*lead, rows // 16, 16, k // 128, 4, 4, 4).view(torch.int32).squeeze(-1)
    nlead = len(lead)
    order = list(range(nlead)) + [nlead + position for position in (0, 2, 4, 1, 3)]
    return w4.permute(*order).contiguous().view(torch.uint8).view(-1)


def pack_layer_weights(
    tensors: dict[str, torch.Tensor],
    moe_mode: MoeMode | str = MoeMode.W8A8,
    model_config: LayerConfig | str = GLM5_CONFIG,
    attention_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Pack every matrix consumed by the fused layer kernel."""

    config = as_layer_config(model_config)
    attention_names = ("w_qkv_a", "w_q_b", "w_uk", "w_uv", "w_o")
    expert_names = ("w_ug", "w_dn")
    required = attention_names if attention_only else (*attention_names, *expert_names, "w_r")
    missing = [name for name in required if name not in tensors]
    if missing:
        raise ValueError(f"missing layer weights: {', '.join(missing)}")
    pack_attention = pack_bf16 if config.attention_weight is AttentionWeight.BF16 else pack_fp8
    packed = {name: pack_attention(tensors[name]) for name in attention_names}
    if attention_only:
        return packed
    weight = moe_format(moe_mode).weight
    pack_expert = pack_mxfp4 if weight is ExpertWeight.MXFP4_BLOCK32 else pack_fp8
    packed.update({name: pack_expert(tensors[name]) for name in expert_names})
    packed["w_r"] = pack_bf16(tensors["w_r"])
    return packed
