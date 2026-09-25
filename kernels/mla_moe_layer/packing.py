# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host-side weight packing shared by GLM-5 MLA and MoE launch wrappers."""

from __future__ import annotations

import torch


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


def pack_layer_weights(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Pack every matrix consumed by the fused layer kernel."""

    fp8_names = ("w_qkv_a", "w_q_b", "w_uk", "w_uv", "w_o", "w_ug", "w_dn")
    missing = [name for name in (*fp8_names, "w_r") if name not in tensors]
    if missing:
        raise ValueError(f"missing layer weights: {', '.join(missing)}")
    packed = {name: pack_fp8(tensors[name]) for name in fp8_names}
    packed["w_r"] = pack_bf16(tensors["w_r"])
    return packed
