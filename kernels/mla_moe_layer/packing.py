# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host-side weight packing shared by GLM-5 MLA and MoE launch wrappers."""

from __future__ import annotations

import torch

from kernels.mla_moe_layer.config import (
    ExpertWeight,
    MoeMode,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
    RouterWeightLayout,
    moe_format,
    resolve_storage_layouts,
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


def pack_bf16_atom(w: torch.Tensor) -> torch.Tensor:
    """Keep the row-major BF16 layout used by ATOM's unquantized router."""

    if w.ndim != 2:
        raise ValueError(f"BF16 packing expects a matrix, got shape {tuple(w.shape)}")
    return w.contiguous().view(-1)


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


def pack_mxfp4_atom(q: torch.Tensor) -> torch.Tensor:
    """Pack MXFP4 values exactly like AITER ``shuffle_weight(..., (16, 16))``.

    ATOM keeps each logical row's 16 packed bytes for a K32 step together,
    whereas the native mono-kernel layout keeps one lane's four K32 dwords
    together.  Leading dimensions (normally the expert id) are preserved by
    flattening them into a batch dimension during the shuffle.
    """

    q = q.view(torch.uint8)
    *lead, rows, packed_k = q.shape
    if rows % 16 or packed_k % 32:
        raise ValueError(f"ATOM MXFP4 matrix dimensions must be divisible by (16, 64), got {(rows, packed_k * 2)}")
    w4 = q.reshape(-1, rows // 16, 16, packed_k // 32, 2, 16)
    return w4.permute(0, 1, 3, 4, 2, 5).contiguous().view(*lead, rows, packed_k).view(-1)


def pack_mxfp4_scale_atom(scale: torch.Tensor) -> torch.Tensor:
    """Pack E8M0 scales exactly like ATOM/AITER's non-interleaved ``shuffle_scale``.

    Production GLM-5 expert shapes need no AITER padding: the flattened row
    count is divisible by 256 and ``K/32`` is divisible by eight.  Reject other
    shapes instead of returning a larger buffer whose expert strides would no
    longer match the mono-kernel contract.
    """

    scale = scale.view(torch.uint8)
    if scale.ndim < 2:
        raise ValueError(f"MXFP4 scale packing expects at least a matrix, got shape {tuple(scale.shape)}")
    rows, cols = scale.numel() // scale.shape[-1], scale.shape[-1]
    if rows % 256 or cols % 8:
        raise ValueError(f"ATOM MXFP4 scale dimensions must be divisible by (256, 8), got {(rows, cols)}")
    packed = scale.reshape(rows // 32, 2, 16, cols // 8, 2, 4)
    return packed.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1)


def pack_layer_weights(
    tensors: dict[str, torch.Tensor],
    moe_mode: MoeMode | str = MoeMode.W8A8,
    mxfp4_weight_layout: Mxfp4WeightLayout | str | None = None,
    mxfp4_scale_layout: Mxfp4ScaleLayout | str | None = None,
    router_weight_layout: RouterWeightLayout | str | None = None,
) -> dict[str, torch.Tensor]:
    """Pack every matrix consumed by the fused layer kernel."""

    attention_names = ("w_qkv_a", "w_q_b", "w_uk", "w_uv", "w_o")
    expert_names = ("w_ug", "w_dn")
    missing = [name for name in (*attention_names, *expert_names, "w_r") if name not in tensors]
    if missing:
        raise ValueError(f"missing layer weights: {', '.join(missing)}")
    packed = {name: pack_fp8(tensors[name]) for name in attention_names}
    weight = moe_format(moe_mode).weight
    weight_layout, scale_layout, router_layout, _ = resolve_storage_layouts(
        moe_mode,
        mxfp4_weight_layout,
        mxfp4_scale_layout,
        router_weight_layout,
    )
    if weight is ExpertWeight.MXFP4_BLOCK32:
        pack_expert = pack_mxfp4_atom if weight_layout is Mxfp4WeightLayout.ATOM else pack_mxfp4
    else:
        if weight_layout is not Mxfp4WeightLayout.NATIVE or scale_layout is not Mxfp4ScaleLayout.NATIVE:
            raise ValueError("ATOM MXFP4 layouts require an MXFP4 expert mode")
        pack_expert = pack_fp8
    packed.update({name: pack_expert(tensors[name]) for name in expert_names})
    if scale_layout is Mxfp4ScaleLayout.ATOM:
        packed.update({name: pack_mxfp4_scale_atom(tensors[name]) for name in ("s_ug", "s_dn")})
    packed["w_r"] = (
        pack_bf16_atom(tensors["w_r"]) if router_layout is RouterWeightLayout.ATOM else pack_bf16(tensors["w_r"])
    )
    return packed
