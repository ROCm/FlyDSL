# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared host-side weight container for model-specific MonoKernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kernels.monokernel.config import (
    GLM5_CONFIG,
    LayerConfig,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
)


@dataclass
class LayerWeights:
    """One tensor-parallel rank's weights and model geometry."""

    heads: int
    t: dict[str, torch.Tensor]
    config: LayerConfig = GLM5_CONFIG
    rank: int = 0
    npes: int = 1
    mxfp4_weight_layout: Mxfp4WeightLayout = Mxfp4WeightLayout.NATIVE
    mxfp4_scale_layout: Mxfp4ScaleLayout = Mxfp4ScaleLayout.NATIVE
    physical_experts: int | None = None


def _atom_mxfp4_storage_view(
    tensor: torch.Tensor,
    *,
    name: str,
    logical_rows: int,
    logical_k: int,
    scale: bool,
) -> torch.Tensor:
    if not tensor.is_contiguous() or tensor.element_size() != 1:
        raise ValueError(f"{name} ATOM storage must be contiguous one-byte values")
    if scale:
        groups = logical_k // 32
        expected_bytes = ((logical_rows + 255) // 256 * 256) * ((groups + 7) // 8 * 8)
    else:
        if not getattr(tensor, "is_shuffled", False):
            raise ValueError(f"{name} ATOM weight must carry the preshuffled marker")
        expected_bytes = logical_rows * logical_k // 2
    if tensor.numel() != expected_bytes:
        raise ValueError(f"{name} has {tensor.numel()} bytes, expected {expected_bytes}")
    return tensor.view(torch.uint8).view(-1)


def prepare_mxfp4_expert_storage(weights: LayerWeights) -> tuple[torch.Tensor, ...]:
    """Return launch-ready expert tensors without changing checkpoint storage."""

    from kernels.monokernel.packing import pack_a16w4_scale, pack_a16w4_weight

    config = weights.config
    tensors = weights.t
    experts = config.n_experts if weights.physical_experts is None else weights.physical_experts
    expert_hidden = config.hidden if config.routed_hidden is None else config.routed_hidden
    ug_rows = experts * 2 * config.inter
    dn_rows = experts * expert_hidden
    if weights.mxfp4_weight_layout is Mxfp4WeightLayout.ATOM:
        w_ug = _atom_mxfp4_storage_view(
            tensors["w_ug"], name="w_ug", logical_rows=ug_rows, logical_k=expert_hidden, scale=False
        )
        w_dn = _atom_mxfp4_storage_view(
            tensors["w_dn"], name="w_dn", logical_rows=dn_rows, logical_k=config.inter, scale=False
        )
    else:
        w_ug = pack_a16w4_weight(tensors["w_ug"])
        w_dn = pack_a16w4_weight(tensors["w_dn"])
    if weights.mxfp4_scale_layout is Mxfp4ScaleLayout.ATOM:
        s_ug = _atom_mxfp4_storage_view(
            tensors["s_ug"], name="s_ug", logical_rows=ug_rows, logical_k=expert_hidden, scale=True
        )
        s_dn = _atom_mxfp4_storage_view(
            tensors["s_dn"], name="s_dn", logical_rows=dn_rows, logical_k=config.inter, scale=True
        )
    else:
        s_ug = pack_a16w4_scale(tensors["s_ug"])
        s_dn = pack_a16w4_scale(tensors["s_dn"])
    return w_ug, s_ug, w_dn, s_dn


__all__ = ["LayerWeights", "prepare_mxfp4_expert_storage"]
